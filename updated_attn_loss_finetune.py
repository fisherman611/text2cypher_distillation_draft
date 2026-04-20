import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
import deepspeed
from huggingface_hub import login

hf_token = os.getenv("HF_READ_TOKEN")
# from kaggle_secrets import UserSecretsClient
# secret_label = "huggingface"
# hf_token = UserSecretsClient().get_secret(secret_label)

if hf_token:
    login(token=hf_token, add_to_git_credential=False)

import random
import json
from tqdm import tqdm
import math
import datetime
from types import SimpleNamespace

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    GenerationConfig)

from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR

from arguments import get_args

from data_utils.lm_datasets import LMTrainDataset
from data_utils.data_utils import LLMDataset
from utils import get_optimizer_params, get_optimizer_params_peft, print_args, initialize
from utils import print_rank, get_rank
from utils import save_rank
from utils import all_gather
from utils import load_parallel, save_parallel
from utils import get_tokenizer, get_model

from distillm import forward_kl, reverse_kl, js_distance, tv_distance
from distillm import skewed_forward_kl, skewed_reverse_kl, csd
from distillm import SampleGenerator, ReplayBuffer

from rouge_metric import compute_metrics

from peft import PeftModel
from ed_eval import ed_evaluate

torch.set_num_threads(4)


def get_teacher_model(args, device):
    config = AutoConfig.from_pretrained(args.teacher_model_path)
    if args.model_parallel:
        raise NotImplementedError
    else:
        config.is_model_parallel = False
        try: model = AutoModelForCausalLM.from_pretrained(args.teacher_model_path, config=config, device_map={"": device}, torch_dtype=torch.bfloat16)
        except:
            model = AutoModelForCausalLM.from_pretrained(args.teacher_model_path, config=config, device_map={"": device}, torch_dtype=torch.float32)
            model = model.half()
        
        if args.teacher_peft_path is not None:
            model = PeftModel.from_pretrained(model, args.teacher_peft_path)
            model = model.merge_and_unload()
            print("merge_and_unload")

        if dist.get_rank() == 0:
            print(' > number of parameters: {}'.format(
                sum([p.nelement() for p in model.parameters()])), flush=True)

    model.eval()
    
    return model


def get_optimizer(args, model):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, DDP):
        model = model.module

    if args.peft is not None:
        param_groups = get_optimizer_params_peft(args, model)
    else:
        param_groups = get_optimizer_params(args, model)

    # Use AdamW.
    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    print_rank(f'Optimizer = {optimizer.__class__.__name__}')
    return optimizer


def get_learning_rate_scheduler(args, optimizer):
    if args.total_iters is None:
        args.total_iters = args.train_iters_per_epoch * args.epochs
    if args.lr_decay_style == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters)
    elif args.lr_decay_style == "cosine":
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.total_iters,
            eta_min=args.lr_min)
    elif args.lr_decay_style == "noam":
        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters,
            num_training_steps=args.total_iters,
            power=0.5)
    elif args.lr_decay_style == "wrmup_cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_ratio * args.total_iters,
            num_training_steps=args.total_iters)
    else:
        raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")

    return lr_scheduler


def setup_model_and_optimizer(args, ds_config, device, set_optim=True):
    # get the model
    model = get_model(args, device)
    # get the optimizer and lr_scheduler
    if set_optim:
        optimizer = get_optimizer(args, model)
        lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    else:
        optimizer, lr_scheduler = None, None
        
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        mpu=None,
        config_params=ds_config
    )
    
    # get the memory usage
    print_rank("Model mem\n", torch.cuda.memory_summary())
    return model, optimizer, lr_scheduler


def prepare_dataset(args, tokenizer):
    data = {}
    rng_sample = random.Random(args.seed)
    
    from torch.utils.data import Subset
    
    if args.do_train:
        if not args.slice_data:
            # Full data
            data["train"] = LMTrainDataset(args, tokenizer, args.data_dir, "train", args.train_num, args.train_ratio, rng_sample)
            data["dev"] = LMTrainDataset(args, tokenizer, args.data_dir, "valid", args.dev_num, args.dev_ratio, rng_sample)
        else:
            # Sliced data for testing
            full_train = LMTrainDataset(args, tokenizer, args.data_dir, "train", args.train_num, args.train_ratio, rng_sample)
            data["train"] = Subset(full_train, range(min(100, len(full_train))))
            data["train"].collate = full_train.collate
            data["train"].move_to_device = full_train.move_to_device
            print_rank("train num", len(data["train"]))
            
            full_dev = LMTrainDataset(args, tokenizer, args.data_dir, "valid", args.dev_num, args.dev_ratio, rng_sample)
            data["dev"] = Subset(full_dev, range(min(20, len(full_dev))))
            data["dev"].collate = full_dev.collate
            data["dev"].move_to_device = full_dev.move_to_device
            if hasattr(full_dev, 'answers'):
                data["dev"].answers = [full_dev.answers[i] for i in data["dev"].indices]

    # if not args.slice_data:
    #     # Full data
    #     data["test"] = LMTrainDataset(args, tokenizer, args.data_dir, "test", args.dev_num, args.dev_ratio, rng_sample)
    # else:
    #     # Sliced data
    #     full_test = LMTrainDataset(args, tokenizer, args.data_dir, "test", args.dev_num, args.dev_ratio, rng_sample)
    #     data["test"] = Subset(full_test, range(min(20, len(full_test))))
    #     data["test"].collate = full_test.collate
    #     data["test"].move_to_device = full_test.move_to_device
    #     if hasattr(full_test, 'answers'):
    #         data["test"].answers = [full_test.answers[i] for i in data["test"].indices]
        
    # pre-trained dataset
    if args.do_train and args.lm_data_dir is not None:
        data["pt_train"] = LMTrainDataset(args, tokenizer, args.lm_data_dir, "train", args.train_num, args.train_ratio, rng_sample)
        print_rank("train num", len(data["pt_train"]))
    return data


def pt_loss(args, model, model_batch, no_model_batch):
    loss_mask = (no_model_batch["label"] != -100).int()
    outputs = model(**model_batch, return_dict=True, use_cache=False)
    logits = outputs.logits
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    lm_loss = loss_fn(logits.view(-1, logits.size(-1)), no_model_batch["label"].view(-1))
    return lm_loss


def get_distil_loss(args, tokenizer, model, teacher_model, model_batch, no_model_batch, logits):
    with torch.no_grad():
        teacher_model.eval()
        teacher_outputs = teacher_model(**model_batch, use_cache=False)
        teacher_logits = teacher_outputs.logits
    if args.model_parallel:
        raise NotImplementedError
    else:
        if "sfkl" in args.type:
            distil_loss = skewed_forward_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
        elif "srkl" in args.type:
            distil_loss = skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
        elif "jsd" in args.type:
            distil_loss = js_distance(logits, teacher_logits, no_model_batch)
        elif "tvd" in args.type:
            distil_loss = tv_distance(logits, teacher_logits, no_model_batch)
        elif "fkl" in args.type or args.type == "kd":
            distil_loss = forward_kl(logits, teacher_logits, no_model_batch)
        elif "rkl" in args.type:
            distil_loss = reverse_kl(logits, teacher_logits, no_model_batch)
        elif "csd" in args.type:
            distil_loss = csd(logits, teacher_logits, no_model_batch)
        else:
            raise NotImplementedError
    return distil_loss


def get_teacher_lm_loss(args, tokenizer, model, teacher_model, model_batch):
    with torch.no_grad():
        t_gen_out = teacher_model.generate(
            **model_batch,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_length=args.max_length,
            top_k=0,
            top_p=1,
            temperature=1.0,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=False)
    
    full_ids = t_gen_out.sequences
    
    input_ids = full_ids[:, :-1]
    mask = (input_ids != tokenizer.pad_token_id).long()
    labels = full_ids[:, 1:]    
    labels = torch.masked_fill(labels, mask==0, -100)
    labels[:, :model_batch["input_ids"].size(1)-1] = -100
    loss_mask = (labels != -100).float()
    
    new_batch = {
        "input_ids": input_ids,
        "attention_mask": mask,
    }
    
    if args.model_type in ["gpt2"]:
        position_ids = torch.cumsum(mask, dim=-1) - 1
        position_ids = torch.masked_fill(position_ids, mask==0, 0)    
        new_batch["position_ids"] = position_ids    
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    outputs = model(**new_batch, return_dict=True, use_cache=False)
    logits = outputs.logits
    lm_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

    return lm_loss


def get_logit_distil_loss(args, teacher_logits, no_model_batch, logits):
    """Compute the original logit-level KD loss.

    This is the same branch selection as get_distil_loss(), but it receives
    teacher_logits directly. The attention-loss finetune path forwards teacher
    once per step, then reuses teacher_logits and captured attention.
    """
    if args.model_parallel:
        raise NotImplementedError
    else:
        if "sfkl" in args.type:
            distil_loss = skewed_forward_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
        elif "srkl" in args.type:
            distil_loss = skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
        elif "jsd" in args.type:
            distil_loss = js_distance(logits, teacher_logits, no_model_batch)
        elif "tvd" in args.type:
            distil_loss = tv_distance(logits, teacher_logits, no_model_batch)
        elif "fkl" in args.type or args.type == "kd":
            distil_loss = forward_kl(logits, teacher_logits, no_model_batch)
        elif "rkl" in args.type:
            distil_loss = reverse_kl(logits, teacher_logits, no_model_batch)
        elif "csd" in args.type:
            distil_loss = csd(logits, teacher_logits, no_model_batch)
        else:
            raise NotImplementedError
    return distil_loss


def resolve_layer_index(idx, num_layers):
    """Allow negative layer indices, e.g. -1 means the last layer."""
    return idx if idx >= 0 else num_layers + idx


def get_module_by_path(root, path):
    obj = root
    for part in path.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def get_transformer_layers(model):
    """Find transformer layers on raw HF, PEFT, or DeepSpeed-wrapped models."""
    candidates = [
        "module.base_model.model.model.layers",
        "base_model.model.model.layers",
        "module.model.layers",
        "model.layers",
        "module.model.model.layers",
        "model.model.layers",
    ]
    for path in candidates:
        layers = get_module_by_path(model, path)
        if layers is not None:
            return layers, path
    raise AttributeError("Could not find transformer layers for attention hooks.")


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """Apply Qwen/LLaMA-style rotary embeddings to [B, H, L, D] Q/K."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


def repeat_kv(hidden_states, num_key_value_groups):
    """Repeat grouped-query attention KV heads to match query heads."""
    if num_key_value_groups == 1:
        return hidden_states
    bsz, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        bsz, num_kv_heads, num_key_value_groups, slen, head_dim
    )
    return hidden_states.reshape(bsz, num_kv_heads * num_key_value_groups, slen, head_dim)


def reconstruct_attention_from_self_attn_inputs(self_attn, hidden_states, position_embeddings, attention_mask):
    """Reconstruct the exact eager attention probabilities from a Qwen3 attention block.

    We intentionally do not use output_attentions=True in the training forward.
    Instead, a forward pre-hook receives the self-attention inputs and recomputes:
        q_proj/k_proj -> q_norm/k_norm -> RoPE -> repeat_kv -> QK^T -> mask -> softmax

    This keeps the training code compatible with PEFT/DeepSpeed wrappers and lets
    us capture only selected layers.
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self_attn.head_dim)

    query_states = self_attn.q_proj(hidden_states).view(hidden_shape)
    key_states = self_attn.k_proj(hidden_states).view(hidden_shape)

    if hasattr(self_attn, "q_norm"):
        query_states = self_attn.q_norm(query_states)
    if hasattr(self_attn, "k_norm"):
        key_states = self_attn.k_norm(key_states)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rope(query_states, key_states, cos, sin)

    num_key_value_groups = getattr(
        self_attn,
        "num_key_value_groups",
        query_states.size(1) // key_states.size(1),
    )
    key_states = repeat_kv(key_states, num_key_value_groups)

    scaling = getattr(self_attn, "scaling", 1.0 / math.sqrt(self_attn.head_dim))
    scores = torch.matmul(query_states.float(), key_states.float().transpose(-1, -2)) * scaling

    if attention_mask is not None:
        # Qwen3 passes the internal additive 4D causal/padding mask to self_attn.
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        scores = scores + causal_mask.float()

    return F.softmax(scores, dim=-1)


def get_attention_layer_mapping(args, student_model, teacher_model):
    """Return selected student/teacher layer indices for attention distillation."""
    student_layers, _ = get_transformer_layers(student_model)
    teacher_layers, _ = get_transformer_layers(teacher_model)

    s_mapping = args.attention_student_layer_mapping
    t_mapping = args.attention_teacher_layer_mapping
    if s_mapping is None:
        s_mapping = args.student_layer_mapping if args.student_layer_mapping != [-1] else [-1]
    if t_mapping is None:
        t_mapping = args.teacher_layer_mapping if args.teacher_layer_mapping != [-1] else [-1]

    s_mapping = [resolve_layer_index(i, len(student_layers)) for i in s_mapping]
    t_mapping = [resolve_layer_index(i, len(teacher_layers)) for i in t_mapping]
    if len(s_mapping) != len(t_mapping):
        raise ValueError("Attention student/teacher layer mappings must have the same length.")
    return s_mapping, t_mapping


def register_attention_hooks(model, selected_layers, captured, detach=False, require_training=False):
    """Register forward pre-hooks that capture reconstructed attention.

    detach=False for the student so attention loss can backprop through LoRA.
    detach=True for the teacher because it is a frozen no_grad target.
    """
    layers, _ = get_transformer_layers(model)
    handles = []

    def make_hook(layer_idx):
        def hook(module, args, kwargs):
            if require_training and not module.training:
                return

            hidden_states = args[0] if len(args) > 0 else kwargs["hidden_states"]
            position_embeddings = kwargs.get("position_embeddings")
            attention_mask = kwargs.get("attention_mask")
            if position_embeddings is None and len(args) > 1:
                position_embeddings = args[1]
            if attention_mask is None and len(args) > 2:
                attention_mask = args[2]

            attn = reconstruct_attention_from_self_attn_inputs(
                module,
                hidden_states,
                position_embeddings,
                attention_mask,
            )
            captured[layer_idx] = attn.detach() if detach else attn
        return hook

    for layer_idx in selected_layers:
        handles.append(
            layers[layer_idx].self_attn.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True)
        )
    return handles


def find_between(text, start_marker, end_marker=None):
    start = text.index(start_marker) + len(start_marker)
    end = text.index(end_marker, start) if end_marker is not None else len(text)
    return text[start:end].strip()


def build_token_span_mask(offsets, char_start, char_end, max_length, device):
    mask = torch.zeros(max_length, dtype=torch.bool, device=device)
    for idx, (tok_start, tok_end) in enumerate(offsets[:max_length]):
        if tok_start == tok_end:
            continue
        if tok_end <= char_start or tok_start >= char_end:
            continue
        mask[idx] = True
    return mask


def build_query_schema_masks_from_text(args, tokenizer, model_batch, no_model_batch):
    """Fallback mask builder when LMTrainDataset has not yet emitted masks.

    Production-friendly option: create query_mask/schema_mask directly in
    data_utils/lm_datasets.py using offset_mapping. This fallback decodes each
    prompt and retokenizes it, so it is useful for experimentation but not ideal
    for very long training runs.
    """
    device = model_batch["input_ids"].device
    input_ids = model_batch["input_ids"]
    labels = no_model_batch["label"]
    bs, max_length = input_ids.shape
    query_mask = torch.zeros(bs, max_length, dtype=torch.bool, device=device)
    schema_mask = torch.zeros(bs, max_length, dtype=torch.bool, device=device)

    for i in range(bs):
        target_positions = torch.where(labels[i] != -100)[0]
        if target_positions.numel() == 0:
            continue
        prompt_end = int(target_positions[0].item()) + 1
        prompt_ids = input_ids[i, :prompt_end].detach().cpu().tolist()
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)

        try:
            query_text = find_between(prompt_text, "QUESTION:\n", "\n\nSCHEMA:")
            schema_text = find_between(prompt_text, "SCHEMA:\n", "\n\nGenerate a Cypher query")
            query_start = prompt_text.find(query_text)
            schema_start = prompt_text.find(schema_text)
            if query_start < 0 or schema_start < 0:
                continue
        except ValueError:
            continue

        encoded = tokenizer(
            prompt_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=True,
            max_length=prompt_end,
        )
        offsets = encoded["offset_mapping"]
        query_mask[i] = build_token_span_mask(
            offsets,
            query_start,
            query_start + len(query_text),
            max_length,
            device,
        )
        schema_mask[i] = build_token_span_mask(
            offsets,
            schema_start,
            schema_start + len(schema_text),
            max_length,
            device,
        )

    return query_mask, schema_mask


def get_attention_region_masks(args, tokenizer, model_batch, no_model_batch):
    cypher_mask = (no_model_batch["label"] != -100) & model_batch["attention_mask"].bool()

    query_mask = no_model_batch.get("query_mask")
    schema_mask = no_model_batch.get("schema_mask")
    if query_mask is not None:
        query_mask = query_mask.bool()
    if schema_mask is not None:
        schema_mask = schema_mask.bool()

    if query_mask is None or schema_mask is None:
        fallback_query, fallback_schema = build_query_schema_masks_from_text(
            args,
            tokenizer,
            model_batch,
            no_model_batch,
        )
        if query_mask is None:
            query_mask = fallback_query
        if schema_mask is None:
            schema_mask = fallback_schema

    return query_mask, schema_mask, cypher_mask


def align_attention_heads(student_attn, teacher_attn, args):
    """Make student/teacher attention head dimensions compatible.

    Student and teacher models can have different head counts, for example
    Qwen3-0.6B has 16 heads while Qwen3-4B has 32. In that case, compare their
    head-averaged attention maps.
    """
    reduction = getattr(args, "attention_head_reduction", "auto")
    if reduction == "mean" or (
        reduction == "auto" and student_attn.size(1) != teacher_attn.size(1)
    ):
        student_attn = student_attn.mean(dim=1, keepdim=True)
        teacher_attn = teacher_attn.mean(dim=1, keepdim=True)
    elif student_attn.size(1) != teacher_attn.size(1):
        raise ValueError(
            "Student/teacher attention heads differ "
            f"({student_attn.size(1)} vs {teacher_attn.size(1)}). "
            "Use --attention-head-reduction auto or mean."
        )
    return student_attn, teacher_attn


def masked_attention_distribution_loss(student_attn, teacher_attn, pair_mask, args):
    """Compare student/teacher attention distributions on a masked region.

    pair_mask has shape [B, 1, L, L]. Rows are target/Cypher positions and
    columns are query/schema/prefix positions depending on the branch.
    """
    eps = args.attention_eps
    student_attn, teacher_attn = align_attention_heads(student_attn, teacher_attn, args)
    pair_mask = pair_mask.to(student_attn.device)

    if args.attention_loss_type == "raw_mse":
        # Plain masked MSE on the attention map. This keeps the original
        # attention mass instead of renormalizing each selected row.
        diff = ((student_attn.float() - teacher_attn.float()) ** 2) * pair_mask
        return diff.sum() / pair_mask.sum().clamp(min=1.0)

    if args.attention_loss_type == "mass_mse":
        # Sum-squared error normalized by teacher attention mass in the region.
        # This matches the "reduce sum then divide by sum of attention map" idea.
        diff = ((student_attn.float() - teacher_attn.float()) ** 2) * pair_mask
        denom = (teacher_attn.float() * pair_mask).sum().clamp(min=eps)
        return diff.sum() / denom

    if args.attention_loss_type == "cka":
        return masked_attention_cka_loss(student_attn, teacher_attn, pair_mask, eps)

    s = student_attn.float() * pair_mask
    t = teacher_attn.float() * pair_mask
    s_sum = s.sum(dim=-1, keepdim=True)
    t_sum = t.sum(dim=-1, keepdim=True)
    valid_rows = (pair_mask.any(dim=-1, keepdim=True) & (s_sum > eps) & (t_sum > eps)).float()

    if valid_rows.sum() == 0:
        return student_attn.new_tensor(0.0)

    s_dist = s / s_sum.clamp(min=eps)
    t_dist = t / t_sum.clamp(min=eps)

    if args.attention_loss_type == "mse":
        per_row = ((s_dist - t_dist) ** 2 * pair_mask).sum(dim=-1, keepdim=True)
    elif args.attention_loss_type == "js":
        mixed = 0.5 * (s_dist + t_dist)
        s_kl = (s_dist * ((s_dist + eps).log() - (mixed + eps).log()) * pair_mask).sum(dim=-1, keepdim=True)
        t_kl = (t_dist * ((t_dist + eps).log() - (mixed + eps).log()) * pair_mask).sum(dim=-1, keepdim=True)
        per_row = 0.5 * (s_kl + t_kl)
    else:
        per_row = (t_dist * ((t_dist + eps).log() - (s_dist + eps).log()) * pair_mask).sum(dim=-1, keepdim=True)

    return (per_row * valid_rows).sum() / valid_rows.sum().clamp(min=1.0)


def masked_attention_loss_with_type(student_attn, teacher_attn, pair_mask, args, loss_type):
    """Run masked attention loss with a branch-specific loss type.

    This avoids mutating args when a branch needs a different objective. In
    particular, CKA is suitable for query/schema regions whose source columns
    are shared across rows, but cypher-prefix has row-dependent causal support,
    so compute_overall_attention_loss falls back to mass_mse for that branch.
    """
    if loss_type == args.attention_loss_type:
        return masked_attention_distribution_loss(student_attn, teacher_attn, pair_mask, args)

    local_args = SimpleNamespace(
        attention_eps=args.attention_eps,
        attention_loss_type=loss_type,
        attention_head_reduction=getattr(args, "attention_head_reduction", "auto"),
    )
    return masked_attention_distribution_loss(student_attn, teacher_attn, pair_mask, local_args)


def get_branch_attention_loss_type(args, branch_name):
    """Resolve loss type for a branch, falling back to the global type.

    If global loss is CKA and cypher loss is not explicitly specified, use
    mass_mse for cypher-prefix because its causal support changes per row.
    """
    branch_loss_type = getattr(args, f"{branch_name}_attention_loss_type", None)
    if branch_loss_type is not None:
        return branch_loss_type
    return args.attention_loss_type


def masked_attention_cka_loss(student_attn, teacher_attn, pair_mask, eps=1e-8):
    """CKA loss between masked attention maps.

    We compare head-averaged attention rows as feature vectors using linear CKA:

        CKA(X, Y) = ||X_c^T Y_c||_F^2 / (||X_c^T X_c||_F ||Y_c^T Y_c||_F)

    where rows are selected target tokens and columns are selected source tokens.
    Loss is 1 - CKA. This is scale-invariant, unlike raw MSE.
    """
    pair_mask = pair_mask.to(student_attn.device)
    s = (student_attn.float() * pair_mask).mean(dim=1)  # [B, L, L]
    t = (teacher_attn.float() * pair_mask).mean(dim=1)  # [B, L, L]

    row_mask = pair_mask.squeeze(1).bool()
    s_mass = s.abs().sum(dim=-1)
    t_mass = t.abs().sum(dim=-1)
    valid_rows = row_mask.any(dim=-1) & (s_mass > eps) & (t_mass > eps)
    if valid_rows.sum() < 3:
        return student_attn.new_tensor(0.0)

    s_rows = s[valid_rows]
    t_rows = t[valid_rows]
    col_mask = row_mask[valid_rows]
    valid_cols = col_mask.any(dim=0)
    if valid_cols.sum() < 2:
        return student_attn.new_tensor(0.0)

    x = s_rows[:, valid_cols]
    y = t_rows[:, valid_cols]
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    hsic = (x.T @ y).pow(2).sum()
    x_norm = (x.T @ x).pow(2).sum().sqrt()
    y_norm = (y.T @ y).pow(2).sum().sqrt()
    denom = x_norm * y_norm
    if denom <= eps:
        # CKA is undefined for zero-variance matrices. If both are effectively
        # identical constants, make the auxiliary loss neutral.
        return x.new_tensor(0.0 if torch.allclose(x, y, atol=eps, rtol=0.0) else 1.0)

    cka = hsic / denom
    return 1.0 - cka.clamp(min=0.0, max=1.0)


def build_pair_mask(row_mask, col_mask):
    return row_mask[:, None, :, None] & col_mask[:, None, None, :]


def build_cypher_prefix_pair_mask(cypher_mask):
    bs, seq_len = cypher_mask.shape
    idx = torch.arange(seq_len, device=cypher_mask.device)
    prev_mask = idx[None, :] < idx[:, None]
    return (cypher_mask[:, None, :, None] & cypher_mask[:, None, None, :] & prev_mask[None, None, :, :])


def compute_overall_attention_loss(args, tokenizer, model_batch, no_model_batch,
                                   student_attn_cache, teacher_attn_cache,
                                   student_layer_mapping, teacher_layer_mapping,
                                   return_details=False):
    query_mask, schema_mask, cypher_mask = get_attention_region_masks(
        args,
        tokenizer,
        model_batch,
        no_model_batch,
    )

    branch_losses = []
    query_losses, schema_losses, cypher_losses = [], [], []
    for s_idx, t_idx in zip(student_layer_mapping, teacher_layer_mapping):
        if s_idx not in student_attn_cache or t_idx not in teacher_attn_cache:
            continue

        s_attn = student_attn_cache[s_idx]
        t_attn = teacher_attn_cache[t_idx].to(s_attn.device)

        if getattr(args, "use_query_attention_loss", False):
            pair_mask = build_pair_mask(cypher_mask, query_mask)
            query_loss_type = get_branch_attention_loss_type(args, "query")
            q_loss = masked_attention_loss_with_type(s_attn, t_attn, pair_mask, args, query_loss_type)
            branch_losses.append(args.w_query_attention_loss * q_loss)
            query_losses.append(q_loss)

        if getattr(args, "use_schema_attention_loss", False):
            pair_mask = build_pair_mask(cypher_mask, schema_mask)
            schema_loss_type = get_branch_attention_loss_type(args, "schema")
            s_loss = masked_attention_loss_with_type(s_attn, t_attn, pair_mask, args, schema_loss_type)
            branch_losses.append(args.w_schema_attention_loss * s_loss)
            schema_losses.append(s_loss)

        if getattr(args, "use_cypher_attention_loss", False):
            pair_mask = build_cypher_prefix_pair_mask(cypher_mask)
            # CKA assumes rows live in a reasonably shared feature space. If
            # global loss is CKA and no cypher-specific loss is set, fallback
            # to mass_mse for causal prefix attention.
            cypher_loss_type = get_branch_attention_loss_type(args, "cypher")
            c_loss = masked_attention_loss_with_type(s_attn, t_attn, pair_mask, args, cypher_loss_type)
            branch_losses.append(args.w_cypher_attention_loss * c_loss)
            cypher_losses.append(c_loss)

    if not branch_losses:
        zero = model_batch["input_ids"].new_tensor(0.0, dtype=torch.float32)
        if return_details:
            return zero, {
                "query_attention_loss": zero,
                "schema_attention_loss": zero,
                "cypher_attention_loss": zero,
            }
        return zero

    attention_loss = args.w_attention_loss * sum(branch_losses) / len(branch_losses)
    if not return_details:
        return attention_loss

    zero = attention_loss.new_tensor(0.0)

    def mean_or_zero(values):
        return sum(values) / len(values) if values else zero

    return attention_loss, {
        "query_attention_loss": mean_or_zero(query_losses),
        "schema_attention_loss": mean_or_zero(schema_losses),
        "cypher_attention_loss": mean_or_zero(cypher_losses),
    }


def reduce_metric_value(value, dp_world_size, dp_group=None):
    if value is None:
        return 0.0
    metric = value.detach().float().clone()
    dist.all_reduce(metric, dist.ReduceOp.SUM, group=dp_group)
    return metric.item() / dp_world_size


def finetune(args, tokenizer: AutoTokenizer, model: deepspeed.DeepSpeedEngine, optimizer: AdamW, lr_scheduler, dataset, device, teacher_model=None):
    print_rank("Start Fine-tuning")

    # print_inspect(model, '*')
    if args.model_parallel:
        raise NotImplementedError
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
        loss_func = nn.CrossEntropyLoss()

    sampler = DistributedSampler(dataset["train"], shuffle=True, drop_last=True, rank=dp_rank, num_replicas=dp_world_size)
    train_dataloader = DataLoader(
        dataset['train'], sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dataset["train"].collate)
    
    if "pt_train" in dataset:
        pt_sampler = DistributedSampler(dataset["pt_train"], shuffle=True, drop_last=True, rank=dp_rank, num_replicas=dp_world_size)
        pt_train_dataloader = DataLoader(
        dataset['pt_train'], sampler=pt_sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dataset["pt_train"].collate)
        pt_train_iter = iter(pt_train_dataloader)
        
    student_generator = SampleGenerator(args, tokenizer)

    step, global_step = 1, 1
    total_loss, total_distil_loss, total_time = 0.0, 0.0, 0.0
    total_kd_logit_loss = 0.0
    total_attention_loss = 0.0
    total_query_attention_loss = 0.0
    total_schema_attention_loss = 0.0
    total_cypher_attention_loss = 0.0
    
    adaptive_threshold = args.init_threshold if "adaptive" in args.type else -1.0
    prev_avg_loss = evaluate(args, tokenizer, model, dataset["dev"], "dev", 0, device, adaptive_threshold)
    replay_buffer = ReplayBuffer(args)

    attention_enabled = (
        teacher_model is not None
        and getattr(args, "use_attention_loss", False)
        and getattr(args, "w_attention_loss", 0.0) > 0
    )
    student_attn_cache, teacher_attn_cache = {}, {}
    attention_hook_handles = []
    attn_student_layers, attn_teacher_layers = [], []

    if attention_enabled:
        if not (
            getattr(args, "use_query_attention_loss", False)
            or getattr(args, "use_schema_attention_loss", False)
            or getattr(args, "use_cypher_attention_loss", False)
        ):
            # Sensible default for text2cypher: ground Cypher in query/schema first.
            args.use_query_attention_loss = True
            args.use_schema_attention_loss = True

        attn_student_layers, attn_teacher_layers = get_attention_layer_mapping(args, model, teacher_model)
        print_rank("Attention loss enabled")
        print_rank("  student attention layers", attn_student_layers)
        print_rank("  teacher attention layers", attn_teacher_layers)
        print_rank("  branches",
                   "query=", args.use_query_attention_loss,
                   "schema=", args.use_schema_attention_loss,
                   "cypher=", args.use_cypher_attention_loss)

        attention_hook_handles.extend(
            register_attention_hooks(
                model,
                attn_student_layers,
                student_attn_cache,
                detach=False,
                require_training=True,
            )
        )
        attention_hook_handles.extend(
            register_attention_hooks(
                teacher_model,
                attn_teacher_layers,
                teacher_attn_cache,
                detach=True,
                require_training=False,
            )
        )
    
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)

        model.train()
        for it, (model_batch, no_model_batch, gen_data, _, _) in enumerate(train_dataloader):
            dataset["train"].move_to_device(model_batch, no_model_batch, gen_data, device)
            student_attn_cache.clear()
            teacher_attn_cache.clear()
            
            if args.lm_data_dir is not None:
                try:
                    pt_model_batch, pt_no_model_batch, pt_gen_data = next(pt_train_iter)
                    # pt_model_batch, pt_no_model_batch, pt_gen_data = pt_train_iter.next()
                except:
                    pt_train_iter = iter(pt_train_dataloader)
                    # pt_model_batch, pt_no_model_batch, pt_gen_data = pt_train_iter.next()
                    pt_model_batch, pt_no_model_batch, pt_gen_data = next(pt_train_iter)
                    
                dataset["pt_train"].move_to_device(pt_model_batch, pt_no_model_batch, pt_gen_data, device)
            
            torch.cuda.synchronize()
            st_time = time.time()
            
            # # sampling ratio:
            samp_threshold = adaptive_threshold * (1 - global_step / args.total_iters)
            if "adaptive" in args.type:
                if args.replay_ratio == "constant":
                    samp_threshold = adaptive_threshold * 0.5
                elif args.replay_ratio == "increasing":
                    samp_threshold = adaptive_threshold * global_step / args.total_iters
                else:
                    samp_threshold = adaptive_threshold * (1 - global_step / args.total_iters)
            
            # data generation
            if args.student_gen:
                r = np.random.uniform(0, 1)
                if "mixed" in args.type and r < args.mixed_alpha:
                    model_batch = student_generator.run_sample(model, gen_data)
                    no_model_batch["label"] = model_batch.pop("no_model_batch")
                    
                    replay_buffer.move_to_memory(model_batch, no_model_batch)
                    model_batch, no_model_batch, gen_data = replay_buffer.sample()
                    model_batch, no_model_batch = replay_buffer.move_to_device(model_batch, no_model_batch, gen_data, device)
                    
                elif "adaptive" in args.type and (r < samp_threshold or (r < adaptive_threshold and len(replay_buffer) < args.capacity)):

                    model_batch = student_generator.run_sample(model, gen_data)
                    no_model_batch["label"] = model_batch.pop("no_model_batch")
                    
                    if args.model_type in ["opt"]:
                        model_batch.pop('position_ids')
                        
                    replay_buffer.move_to_memory(model_batch, no_model_batch, gen_data)
                    
                elif "adaptive" in args.type and r < adaptive_threshold:
                    model_batch, no_model_batch, gen_data = replay_buffer.sample()
                    model_batch, no_model_batch, gen_data = replay_buffer.move_to_device(model_batch, no_model_batch, gen_data, device)
                    
                model.train()

            outputs = model(**model_batch, use_cache=False)
            
            logits = outputs.logits
            if args.model_parallel:
                raise NotImplementedError
            else:
                lm_loss = loss_func(logits.float().view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))

            zero_loss = lm_loss.new_tensor(0.0)
            distil_loss = zero_loss
            logit_distil_loss = zero_loss
            attention_loss = zero_loss
            attention_details = {
                "query_attention_loss": zero_loss,
                "schema_attention_loss": zero_loss,
                "cypher_attention_loss": zero_loss,
            }
            
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_model.eval()
                    teacher_outputs = teacher_model(**model_batch, use_cache=False)
                    teacher_logits = teacher_outputs.logits

                logit_distil_loss = get_logit_distil_loss(args, teacher_logits, no_model_batch, logits)
                distil_loss = args.w_logit_kd_loss * logit_distil_loss
                if attention_enabled:
                    attention_loss, attention_details = compute_overall_attention_loss(
                        args,
                        tokenizer,
                        model_batch,
                        no_model_batch,
                        student_attn_cache,
                        teacher_attn_cache,
                        attn_student_layers,
                        attn_teacher_layers,
                        return_details=True,
                    )
                    distil_loss = distil_loss + attention_loss
                loss = (1 - args.kd_ratio) * lm_loss + args.kd_ratio * distil_loss
            else:
                loss = lm_loss
                
            if args.lm_data_dir is not None:
                assert args.lm_coef is not None
                loss += args.lm_coef * pt_loss(args, model, pt_model_batch, pt_no_model_batch)
                
            model.backward(loss)
            model.step()

            global_loss = reduce_metric_value(loss, dp_world_size, dp_group)
            global_distil_loss = reduce_metric_value(distil_loss, dp_world_size, dp_group)
            global_kd_logit_loss = reduce_metric_value(logit_distil_loss, dp_world_size, dp_group)
            global_attention_loss = reduce_metric_value(attention_loss, dp_world_size, dp_group)
            global_query_attention_loss = reduce_metric_value(attention_details["query_attention_loss"], dp_world_size, dp_group)
            global_schema_attention_loss = reduce_metric_value(attention_details["schema_attention_loss"], dp_world_size, dp_group)
            global_cypher_attention_loss = reduce_metric_value(attention_details["cypher_attention_loss"], dp_world_size, dp_group)
            total_distil_loss += global_distil_loss
            total_kd_logit_loss += global_kd_logit_loss
            total_attention_loss += global_attention_loss
            total_query_attention_loss += global_query_attention_loss
            total_schema_attention_loss += global_schema_attention_loss
            total_cypher_attention_loss += global_cypher_attention_loss
    
            torch.cuda.synchronize()
            elapsed_time = time.time() - st_time

            total_loss += global_loss
            total_time += elapsed_time

            # Logging
            def get_log(log_loss, log_distil_loss, log_kd_logit_loss,
                        log_attention_loss, log_query_attention_loss,
                        log_schema_attention_loss, log_cypher_attention_loss,
                        log_time):
                return "train | epoch {:3d} | Iter: {:6d}/{:6d} | global iter: {:6d}/{:6d} | loss: {:.4f} | ds_loss: {:.4f} | kd_logit_loss: {:.4f} | attn_loss: {:.4e} | query_attn_loss: {:.4e} | schema_attn_loss: {:.4e} | cypher_attn_loss: {:.4e} | lr: {:.4e} | scale: {:10.4f} | micro time: {:.3f} | step time: {:.3f}".format(
                    epoch,
                    step,
                    args.total_iters * args.gradient_accumulation_steps,
                    global_step,
                    args.total_iters,
                    log_loss,
                    log_distil_loss,
                    log_kd_logit_loss,
                    log_attention_loss,
                    log_query_attention_loss,
                    log_schema_attention_loss,
                    log_cypher_attention_loss,
                    lr_scheduler.get_last_lr()[0],
                    optimizer.cur_scale if hasattr(optimizer, "cur_scale") else 0,
                    elapsed_time,
                    log_time,
                )

            if args.mid_log_num > 0:
                mid_log_step = args.gradient_accumulation_steps // args.mid_log_num
                mid_log_step = 1 if mid_log_step == 0 else mid_log_step
                if step % mid_log_step == 0:
                    print_rank(get_log(
                        global_loss,
                        global_distil_loss,
                        global_kd_logit_loss,
                        global_attention_loss,
                        global_query_attention_loss,
                        global_schema_attention_loss,
                        global_cypher_attention_loss,
                        0,
                    ))

            if global_step % args.log_interval == 0 and step % args.gradient_accumulation_steps == 0:
                denom = args.log_interval * args.gradient_accumulation_steps
                log_str = get_log(
                    total_loss / denom,
                    total_distil_loss / denom,
                    total_kd_logit_loss / denom,
                    total_attention_loss / denom,
                    total_query_attention_loss / denom,
                    total_schema_attention_loss / denom,
                    total_cypher_attention_loss / denom,
                    total_time / (args.log_interval))
                print_rank("*" * 100)
                print_rank(log_str)
                print_rank(args.save)
                print_rank("*" * 100)
                save_rank(log_str, os.path.join(args.save, "log.txt"))
                total_loss, total_distil_loss, total_time = 0.0, 0.0, 0.0
                total_kd_logit_loss = 0.0
                total_attention_loss = 0.0
                total_query_attention_loss = 0.0
                total_schema_attention_loss = 0.0
                total_cypher_attention_loss = 0.0
            
            # Checkpointing
            if args.save and args.save_interval and global_step % args.save_interval == 0 and step % args.gradient_accumulation_steps == 0:
                save_dir_path = os.path.join(args.save, str(global_step))
                if args.model_parallel:
                    raise NotImplementedError
                else:
                    if dist.get_rank() == 0:
                        os.makedirs(save_dir_path, exist_ok=True)
                        print_rank(f"Model save to {save_dir_path}")
                        tokenizer.save_pretrained(save_dir_path)
                        model.module.save_pretrained(save_dir_path, safe_serialization=False)
                dist.barrier()

            # Evaluation
            if args.eval_interval and global_step % args.eval_interval == 0 and step % args.gradient_accumulation_steps == 0:
                curr_avg_loss = evaluate(args, tokenizer, model, dataset["dev"], "dev", epoch, device, adaptive_threshold)
                if "adaptive" in args.type:
                    if curr_avg_loss >= prev_avg_loss + args.loss_eps:
                        adaptive_threshold += 0.1
                        adaptive_threshold = min(adaptive_threshold, 1.0)
                        prev_avg_loss = curr_avg_loss

                # evaluate(args, tokenizer, model, dataset["test"], "test", epoch, device)
                    
                model.train()
                
            step += 1
            if step % args.gradient_accumulation_steps == 0:
                global_step += 1
            
            if global_step > args.total_iters:
                break

    for handle in attention_hook_handles:
        handle.remove()

    return model


def evaluate(args, tokenizer, model, dataset: LMTrainDataset, split, epoch, device, adaptive_threshold=None):
    
    collate_fn = dataset.collate

    if args.model_parallel:
        raise NotImplementedError
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
        loss_func = nn.CrossEntropyLoss()

    print_rank("dp size", dp_world_size)

    generation_config = GenerationConfig(
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_length=args.max_length,
        min_length=None,
        eos_token_id=[tokenizer.eos_token_id, 151643],
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=False
    )

    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    model.eval()
    all_loss = 0.0
    step = 0
    
    all_response_ids = []
    
    with torch.no_grad():
        for it, (model_batch, no_model_batch, gen_data, _, _) in enumerate(tqdm(dataloader, desc="Evaluating", disable=(dist.get_rank() != 0))):
            print_rank(f"{it}/{len(dataloader)}")
            dataset.move_to_device(model_batch, no_model_batch, gen_data, device)
            logits = model(**model_batch).logits
            if args.model_parallel:
                raise NotImplementedError
            else:
                loss = loss_func(logits.view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
            
            max_new_tokens = args.max_length - gen_data["input_ids"].size(1)
            
            if args.eval_gen:            
                gen_out = model.generate(
                    **gen_data,
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens)
                
                full_ids = gen_out.sequences
                
                full_ids = F.pad(
                    full_ids,
                    (0, args.max_length - full_ids.shape[1]),
                    value=tokenizer.pad_token_id,
                )
                
                response_ids = full_ids[:, gen_data["input_ids"].size(1):]
                all_response_ids.append(response_ids)
                    
            dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
            loss = loss / dp_world_size
            all_loss += loss.item()
            step += 1
    
    if args.eval_gen:
        all_response_ids = torch.cat(all_response_ids, dim=0)
        all_response_ids = all_gather(all_response_ids, dim=1, world_size=dp_world_size, group=dp_group, op="stack")
        all_response_ids = all_response_ids.view(-1, all_response_ids.size(-1))
        
        responses = tokenizer.batch_decode(all_response_ids, skip_special_tokens=True)
    
    if get_rank() == 0:
        if args.eval_gen:
            references = dataset.answers
            responses = responses[:len(references)]
            
            res = compute_metrics(responses, references)

            # ed_metrics = ed_evaluate(responses, references)
            # res.update(ed_metrics)
        
            eval_dir = os.path.join(args.save, "eval", str(epoch))
            print_rank(eval_dir)
            os.makedirs(eval_dir, exist_ok=True)
            with open(os.path.join(eval_dir, "answers.jsonl"), "w") as f:
                for resp in responses:
                    f.write(json.dumps({"text": resp}) + "\n")
        else:
            res = {}
    
        avg_loss = all_loss / step
        
        if "adaptive" in args.type:
            log_str = f"{split} | avg_loss: {avg_loss} | {res} | threshold: {adaptive_threshold}"
        else:
            log_str = f"{split} | avg_loss: {avg_loss} | {res}"
        print_rank(log_str)
        save_rank(log_str, os.path.join(args.save, "log.txt"))
        
    return all_loss / step


def main():
    torch.backends.cudnn.enabled = False
    
    args = get_args()
    initialize(args)
    
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30, os.path.join(args.save, "log.txt"))
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = 10000000
    
    if not args.do_train:
        ds_config["zero_optimization"]["stage"] = 0
    
    args.fp32 = not ds_config["fp16"]["enabled"]  
    args.bf16 = "bf16" in ds_config and ds_config["bf16"]["enabled"]  
    args.deepspeed_config = None
    
    # get the tokenizer
    tokenizer = get_tokenizer(args)
    print(type(tokenizer))

    dataset = prepare_dataset(
        args,
        tokenizer,
    )
    
    dp_world_size = dist.get_world_size()
    
    if args.do_train:
        args.train_iters_per_epoch = int(len(dataset["train"]) / (args.batch_size * dp_world_size * args.gradient_accumulation_steps))
        print_rank("Train iters per epoch", args.train_iters_per_epoch)
        if args.total_iters is None:
            args.total_iters = args.train_iters_per_epoch * args.epochs
        if args.epochs is None:
            args.epochs = math.ceil(args.total_iters / args.train_iters_per_epoch)
        print_rank("total_iters", args.total_iters)
        
        if args.save_interval == -1:
            args.save_interval = args.train_iters_per_epoch
        
        if args.eval_interval == -1:
            args.eval_interval = args.train_iters_per_epoch
    
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, ds_config, device, set_optim=args.do_train)
    
    if args.teacher_model_type is None:
        args.teacher_model_type = args.model_type
    
    if args.teacher_model_path is not None:
        teacher_model = get_teacher_model(args, device)
    else:
        teacher_model = None
    
    if args.do_train:
        model = finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset, device, teacher_model=teacher_model)
   
    if args.do_eval:
        # evaluate(args, tokenizer, model, dataset["test"], "test", 0, device)
        pass
        
    
if __name__ == "__main__":
    main()
