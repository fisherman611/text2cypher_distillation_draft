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

import random
import json
from tqdm import tqdm
import math
import datetime

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
from distillm import skewed_forward_kl, skewed_reverse_kl
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
    if args.do_train:
        data["train"] = LMTrainDataset(args, tokenizer, args.data_dir, "train", args.train_num, args.train_ratio, rng_sample)
        print_rank("train num", len(data["train"]))
        data["dev"] = LMTrainDataset(args, tokenizer, args.data_dir, "valid", args.dev_num, args.dev_ratio, rng_sample)

    # data["test"] = LMTrainDataset(args, tokenizer, args.data_dir, "test", args.dev_num, args.dev_ratio, rng_sample)

        
    # pre-trained dataset
    if args.do_train and args.lm_data_dir is not None:
        data["pt_train"] = LMTrainDataset(args, tokenizer, args.lm_data_dir, "train", args.train_num, args.train_ratio, rng_sample)
        print_rank("train num", len(data["pt_train"]))
    return data


def get_distil_loss(args, teacher_logits, no_model_batch, logits):
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
        else:
            raise NotImplementedError
    return distil_loss


def get_grounding_loss_config(args):
    w_rel = float(getattr(args, "w_rel_loss", 0.0))
    if w_rel == 0.0:
        w_rel = float(getattr(args, "w_span_loss", 1.0))
    if (not math.isfinite(w_rel)) or w_rel < 0.0:
        w_rel = 1.0
    return {"w_rel": min(w_rel, 1e4)}


def build_source_token_mask(attention_mask, labels):
    valid_token_mask = attention_mask.bool()
    ignored_mask = (labels == -100) & valid_token_mask
    target_mask = (labels != -100) & valid_token_mask

    source_mask = ignored_mask.clone()
    has_target = target_mask.any(dim=-1)
    if has_target.any():
        first_target_idx = target_mask.long().argmax(dim=-1)
        source_mask[has_target, first_target_idx[has_target]] = True

    no_target = (~has_target) & valid_token_mask.any(dim=-1)
    if no_target.any():
        source_mask[no_target] = valid_token_mask[no_target]

    return source_mask


def prepare_span_token_map(attention_mask, offsets_mapping, spans_offsets):
    device = attention_mask.device
    batch_size, seq_len = attention_mask.shape

    max_spans = max((len(sample_spans) for sample_spans in spans_offsets), default=0)
    if max_spans == 0:
        return None, None

    span_starts = torch.zeros(batch_size, max_spans, dtype=torch.long, device=device)
    span_ends = torch.zeros(batch_size, max_spans, dtype=torch.long, device=device)
    span_mask = torch.zeros(batch_size, max_spans, dtype=torch.bool, device=device)

    for batch_idx, sample_spans in enumerate(spans_offsets):
        if not sample_spans:
            continue
        spans_tensor = torch.tensor(sample_spans, dtype=torch.long, device=device)
        span_starts[batch_idx, : len(sample_spans)] = spans_tensor[:, 0]
        span_ends[batch_idx, : len(sample_spans)] = spans_tensor[:, 1]
        span_mask[batch_idx, : len(sample_spans)] = True

    current_offsets = offsets_mapping[:, :seq_len, :] if offsets_mapping.shape[1] != seq_len else offsets_mapping
    token_start = current_offsets[..., 0].unsqueeze(-1).to(device)
    token_end = current_offsets[..., 1].unsqueeze(-1).to(device)

    token_in_span = (token_start + 1 >= span_starts.unsqueeze(1)) & (token_end <= span_ends.unsqueeze(1))
    token_in_span = token_in_span & attention_mask.unsqueeze(-1).bool() & span_mask.unsqueeze(1)

    if not token_in_span.any():
        return None, None

    return token_in_span, span_mask


def compute_span_mean_representations(hidden_state, token_to_span_map, span_mask):
    token_weights = token_to_span_map.float()
    span_sums = torch.einsum("bld,bls->bsd", hidden_state.float(), token_weights)
    span_counts = token_weights.sum(dim=1).unsqueeze(-1).clamp(min=1e-5)
    span_repr = span_sums / span_counts
    span_repr = span_repr * span_mask.unsqueeze(-1).float()
    return span_repr


def compute_query_conditioned_representations(hidden_state, span_repr, span_mask, source_mask):
    hidden_state = hidden_state.float()
    span_repr = span_repr.float()

    scores = torch.matmul(span_repr, hidden_state.transpose(1, 2))
    scores = scores / math.sqrt(hidden_state.size(-1))
    scores = scores.masked_fill(~source_mask.unsqueeze(1), -1e4)

    attn_weights = torch.softmax(scores, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
    attn_weights = attn_weights * source_mask.unsqueeze(1).float()
    attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-5)
    attn_weights = attn_weights * span_mask.unsqueeze(-1).float()

    query_repr = torch.matmul(attn_weights, hidden_state)
    query_repr = query_repr * span_mask.unsqueeze(-1).float()

    return query_repr


def _safe_cosine_similarity(x, y, dim=-1, eps=1e-6):
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    y = torch.nan_to_num(y.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    x = F.normalize(x, p=2, dim=dim, eps=eps)
    y = F.normalize(y, p=2, dim=dim, eps=eps)
    return (x * y).sum(dim=dim).clamp(min=-1.0, max=1.0)


def compute_span_query_relation_loss(student_span, student_query, teacher_span, teacher_query, span_mask, span_lengths):
    student_rel = _safe_cosine_similarity(student_span, student_query, dim=-1)
    teacher_rel = _safe_cosine_similarity(teacher_span, teacher_query, dim=-1)
    per_span = (student_rel - teacher_rel).pow(2)
    per_span = torch.nan_to_num(per_span, nan=0.0, posinf=4.0, neginf=0.0).clamp(min=0.0, max=4.0)
    weights = span_lengths * span_mask.float()
    loss = (per_span * weights).sum() / weights.sum().clamp(min=1e-5)
    return torch.nan_to_num(loss, nan=0.0, posinf=4.0, neginf=0.0).clamp(min=0.0, max=4.0)


def compute_grounding_losses_for_layer(
    student_hidden_state,
    teacher_hidden_state,
    token_to_span_map,
    span_mask,
    source_mask,
):
    valid_sample_mask = span_mask.any(dim=-1) & source_mask.any(dim=-1)
    if not valid_sample_mask.any():
        zero = student_hidden_state.new_tensor(0.0)
        return zero

    student_hidden_state = student_hidden_state[valid_sample_mask]
    teacher_hidden_state = teacher_hidden_state[valid_sample_mask]
    token_to_span_map = token_to_span_map[valid_sample_mask]
    span_mask = span_mask[valid_sample_mask]
    source_mask = source_mask[valid_sample_mask]

    span_lengths = token_to_span_map.float().sum(dim=1)
    student_span = compute_span_mean_representations(student_hidden_state, token_to_span_map, span_mask)
    teacher_span = compute_span_mean_representations(teacher_hidden_state, token_to_span_map, span_mask)

    student_query = compute_query_conditioned_representations(student_hidden_state, student_span, span_mask, source_mask)
    teacher_query = compute_query_conditioned_representations(teacher_hidden_state, teacher_span, span_mask, source_mask)
    rel_loss = compute_span_query_relation_loss(
        student_span, student_query, teacher_span, teacher_query, span_mask, span_lengths
    )
    rel_loss = torch.nan_to_num(rel_loss, nan=0.0, posinf=4.0, neginf=0.0).clamp(min=0.0, max=4.0)
    return rel_loss


def compute_overall_relation_loss(
    attention_mask,
    labels,
    student_hidden_states,
    teacher_hidden_states,
    offsets_mapping,
    spans_offsets,
    args,
):
    token_to_span_map, span_mask = prepare_span_token_map(attention_mask, offsets_mapping, spans_offsets)
    if token_to_span_map is None:
        return attention_mask.new_tensor(0.0)

    source_mask = build_source_token_mask(attention_mask, labels)
    if not source_mask.any():
        return attention_mask.new_tensor(0.0)

    rel_total = attention_mask.new_tensor(0.0)
    valid_layers = 0

    for student_idx, teacher_idx in zip(args.student_layer_mapping, args.teacher_layer_mapping):
        student_hidden = student_hidden_states[student_idx]
        teacher_hidden = teacher_hidden_states[teacher_idx]
        if student_hidden is None:
            continue

        rel_loss = compute_grounding_losses_for_layer(
            student_hidden, teacher_hidden, token_to_span_map, span_mask, source_mask
        )
        rel_total += rel_loss
        valid_layers += 1

    if valid_layers == 0:
        return attention_mask.new_tensor(0.0)

    return rel_total / valid_layers


def soft_label_distill_loss(student_logits, teacher_logits, mask, distill_temperature = 2.0):
    student_probs = F.log_softmax(student_logits / distill_temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / distill_temperature, dim=-1)

    loss = F.kl_div(student_probs, teacher_probs, reduction='none').sum(dim=-1)
    loss = (loss * mask).sum() / mask.sum()

    return loss

def get_fdd_loss(args, t_hiddens, s_hiddens, mask, student, teacher):
    i = 0
    traj_loss, der_loss = 0.0, 0.0
    pre_s_hidden_logs, pre_t_hidden_logs = None, None
    # mask = (no_model_batch["label"] != -100).int()

    for s_idx, t_idx in zip(args.student_layer_mapping, args.teacher_layer_mapping):
        s_hidden = s_hiddens[s_idx]
        t_hidden = t_hiddens[t_idx]

        s_hidden_logits = student.module.lm_head(s_hidden)
        t_hidden_logits = teacher.lm_head(t_hidden)
        # traj_loss += forward_kl(s_hidden_logits, t_hidden_logits, no_model_batch)
        traj_loss += soft_label_distill_loss(s_hidden_logits, t_hidden_logits, mask)

        s_hidden_logs = F.log_softmax(s_hidden_logits, dim=-1)
        t_hidden_logs = F.log_softmax(t_hidden_logits, dim=-1)

        if i > 0:
            delta_hidden_student = s_hidden_logs - pre_s_hidden_logs
            delta_hidden_teacher = t_hidden_logs - pre_t_hidden_logs
            cos_sim = F.cosine_similarity(delta_hidden_student, delta_hidden_teacher, dim=-1, eps=1e-5)
            cos_sim_loss = 1 - cos_sim
            cos_sim_loss = (cos_sim_loss * mask).sum() / mask.sum()

            der_loss +=  cos_sim_loss

        pre_s_hidden_logs, pre_t_hidden_logs = s_hidden_logs, t_hidden_logs

        i += 1

    if i == 0:
        return mask.new_tensor(0.0)
    if i == 1:
        return traj_loss

    return traj_loss / i + der_loss / (i - 1)


def finetune(args, tokenizer: AutoTokenizer, model: deepspeed.DeepSpeedEngine, optimizer: AdamW, lr_scheduler, dataset, device, teacher_model=None):
    print_rank("Start Fine-tuning with updated relational grounding + FDD loss")

    # print_inspect(model, '*')
    if args.model_parallel:
        raise NotImplementedError
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
        loss_func = nn.CrossEntropyLoss()
    grounding_cfg = get_grounding_loss_config(args)

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
    total_loss, total_distil_loss, total_grounding_loss, total_rel_loss, total_fdd_loss, total_time = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    adaptive_threshold = args.init_threshold if "adaptive" in args.type else -1.0
    # prev_avg_loss = evaluate(args, tokenizer, model, dataset["dev"], "dev", 0, device, adaptive_threshold)
    prev_avg_loss = 0.0
    replay_buffer = ReplayBuffer(args)

    student_captured_hidden = []
    hook_handles = []
    def capture_hook_fn(module, inputs, output):
        if module.training: 
            if isinstance(output, tuple):
                student_captured_hidden.append(output[0])
            else:
                student_captured_hidden.append(output)

    for layer in model.base_model.model.model.layers:
        h_layer = layer.register_forward_hook(capture_hook_fn)
        hook_handles.append(h_layer)
    
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)

        model.train()
        for it, (model_batch, no_model_batch, gen_data, _, _) in enumerate(train_dataloader):
            dataset["train"].move_to_device(model_batch, no_model_batch, gen_data, device)
            student_captured_hidden.clear()
            student_captured_hidden.append(None)
            
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
                lm_loss = loss_func(logits.float().reshape(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
            
            weighted_grounding_loss = logits.new_tensor(0.0)
            weighted_rel_loss = logits.new_tensor(0.0)
            fdd_loss = logits.new_tensor(0.0)
            distil_for_log = logits.new_tensor(0.0)

            if teacher_model is not None:
                with torch.no_grad():
                    teacher_model.eval()
                    teacher_outputs = teacher_model(**model_batch, output_hidden_states=True, use_cache=False)
                    teacher_logits = teacher_outputs.logits

                distil_loss = get_distil_loss(args, teacher_logits, no_model_batch, logits)
                distil_loss = torch.nan_to_num(distil_loss, nan=0.0, posinf=100.0, neginf=0.0)

                if "offset_mapping" in no_model_batch and "span_offsets" in no_model_batch:
                    rel_loss = compute_overall_relation_loss(
                        model_batch["attention_mask"],
                        no_model_batch["label"],
                        student_captured_hidden,
                        teacher_outputs.hidden_states,
                        no_model_batch["offset_mapping"],
                        no_model_batch["span_offsets"],
                        args,
                    )
                    weighted_rel_loss = grounding_cfg["w_rel"] * rel_loss
                    grounding_cap = float(getattr(args, "grounding_loss_cap", 1e4))
                    if (not math.isfinite(grounding_cap)) or grounding_cap <= 0.0:
                        grounding_cap = 1e4
                    weighted_grounding_loss = torch.nan_to_num(
                        weighted_rel_loss,
                        nan=0.0,
                        posinf=grounding_cap,
                        neginf=0.0,
                    ).clamp(min=0.0, max=grounding_cap)

                distil_with_ground = distil_loss + weighted_grounding_loss
                distil_for_log = distil_with_ground

                fdd_loss = get_fdd_loss(
                    args,
                    teacher_outputs.hidden_states,
                    student_captured_hidden,
                    model_batch["attention_mask"],
                    model,
                    teacher_model,
                )
                fdd_loss = torch.nan_to_num(fdd_loss, nan=0.0, posinf=100.0, neginf=0.0)
                if args.fdd_weight is None:
                    loss = (1 - args.kd_ratio) * lm_loss + args.kd_ratio * (distil_with_ground + fdd_loss)
                else:
                    loss = (1 - args.kd_ratio) * lm_loss + args.kd_ratio * 2 * (
                        (1 - args.fdd_weight) * distil_with_ground + args.fdd_weight * fdd_loss
                    )
            else:
                distil_loss = logits.new_tensor(0.0)
                loss = lm_loss
                
            loss = torch.nan_to_num(loss, nan=0.0, posinf=100.0, neginf=0.0)
            model.backward(loss)
            model.step()
             
            reduced_loss = loss.detach().clone()
            dist.all_reduce(reduced_loss, dist.ReduceOp.SUM, group=dp_group)
            global_loss = reduced_loss.item() / dp_world_size

            global_distil_loss = 0.0
            global_grounding_loss = 0.0
            global_rel_loss = 0.0
            global_fdd_loss = 0.0
            if teacher_model is not None:
                reduced_distil = distil_for_log.detach().clone()
                dist.all_reduce(reduced_distil, dist.ReduceOp.SUM, group=dp_group)
                global_distil_loss = reduced_distil.item() / dp_world_size

                reduced_ground = weighted_grounding_loss.detach().clone()
                dist.all_reduce(reduced_ground, dist.ReduceOp.SUM, group=dp_group)
                global_grounding_loss = reduced_ground.item() / dp_world_size

                reduced_rel = weighted_rel_loss.detach().clone()
                dist.all_reduce(reduced_rel, dist.ReduceOp.SUM, group=dp_group)
                global_rel_loss = reduced_rel.item() / dp_world_size

                reduced_fdd = fdd_loss.detach().clone()
                dist.all_reduce(reduced_fdd, dist.ReduceOp.SUM, group=dp_group)
                global_fdd_loss = reduced_fdd.item() / dp_world_size

                total_distil_loss += global_distil_loss
                total_grounding_loss += global_grounding_loss
                total_rel_loss += global_rel_loss
                total_fdd_loss += global_fdd_loss
    
            torch.cuda.synchronize()
            elapsed_time = time.time() - st_time

            total_loss += global_loss
            total_time += elapsed_time

            # Logging
            def get_log(log_loss, log_distil_loss, log_ground_loss, log_rel_loss, log_fdd_loss, log_time):
                return "train | epoch {:3d} | Iter: {:6d}/{:6d} | global iter: {:6d}/{:6d} | loss: {:.4f} | ds_loss: {:.4f} | ground: {:.4f} | rel: {:.4f} | fdd: {:.4f} | lr: {:.4e} | scale: {:10.4f} | micro time: {:.3f} | step time: {:.3f}".format(
                    epoch,
                    step,
                    args.total_iters * args.gradient_accumulation_steps,
                    global_step,
                    args.total_iters,
                    log_loss,
                    log_distil_loss,
                    log_ground_loss,
                    log_rel_loss,
                    log_fdd_loss,
                    lr_scheduler.get_last_lr()[0],
                    optimizer.cur_scale if hasattr(optimizer, "cur_scale") else 0,
                    elapsed_time,
                    log_time,
                )

            if args.mid_log_num > 0:
                mid_log_step = args.gradient_accumulation_steps // args.mid_log_num
                mid_log_step = 1 if mid_log_step == 0 else mid_log_step
                if step % mid_log_step == 0:
                    print_rank(get_log(global_loss, global_distil_loss, global_grounding_loss, global_rel_loss, global_fdd_loss, 0))

            if global_step % args.log_interval == 0 and step % args.gradient_accumulation_steps == 0:
                log_str = get_log(
                    total_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_distil_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_grounding_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_rel_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_fdd_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_time / (args.log_interval))
                print_rank("*" * 100)
                print_rank(log_str)
                print_rank(args.save)
                print_rank("*" * 100)
                save_rank(log_str, os.path.join(args.save, "log.txt"))
                total_loss, total_distil_loss, total_grounding_loss, total_rel_loss, total_fdd_loss, total_time = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
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

    for h in hook_handles:
        h.remove()
            
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
        eos_token_id=tokenizer.eos_token_id,
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

            ed_metrics = ed_evaluate(responses, references)
            res.update(ed_metrics)
        
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
