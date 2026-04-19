import argparse
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_attention_alignment import build_one_sample, get_layers, top_attention_to_region


def repeat_kv(hidden_states, num_key_value_groups):
    if num_key_value_groups == 1:
        return hidden_states
    bsz, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        bsz, num_kv_heads, num_key_value_groups, slen, head_dim
    )
    return hidden_states.reshape(bsz, num_kv_heads * num_key_value_groups, slen, head_dim)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


def reconstruct_attention_from_self_attn_inputs(self_attn, hidden_states, position_embeddings, attention_mask):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self_attn.head_dim)

    q = self_attn.q_proj(hidden_states).view(hidden_shape)
    k = self_attn.k_proj(hidden_states).view(hidden_shape)

    if hasattr(self_attn, "q_norm"):
        q = self_attn.q_norm(q)
    if hasattr(self_attn, "k_norm"):
        k = self_attn.k_norm(k)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    cos, sin = position_embeddings
    q, k = apply_rope(q, k, cos, sin)

    num_key_value_groups = self_attn.num_key_value_groups
    k = repeat_kv(k, num_key_value_groups)

    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self_attn.scaling

    if attention_mask is not None:
        # Qwen3 passes a 4D additive mask, usually [B, 1, L, L].
        causal_mask = attention_mask[:, :, :, : k.shape[-2]]
        scores = scores + causal_mask.float()

    return F.softmax(scores, dim=-1)


def capture_reconstructed_attention(model, layer_idx, batch):
    layers, layer_path = get_layers(model)
    layer = layers[layer_idx]
    captured = {}

    def pre_hook(module, args, kwargs):
        hidden_states = args[0] if len(args) > 0 else kwargs["hidden_states"]
        position_embeddings = kwargs.get("position_embeddings")
        attention_mask = kwargs.get("attention_mask")

        if position_embeddings is None and len(args) > 1:
            position_embeddings = args[1]
        if attention_mask is None and len(args) > 2:
            attention_mask = args[2]

        captured["hook_attn"] = reconstruct_attention_from_self_attn_inputs(
            module,
            hidden_states.detach(),
            tuple(x.detach() for x in position_embeddings),
            attention_mask.detach() if attention_mask is not None else None,
        ).detach()

    handle = layer.self_attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )
    handle.remove()

    if "hook_attn" not in captured:
        raise RuntimeError("Hook did not capture reconstructed attention.")

    output_attn = outputs.attentions[layer_idx].detach()
    return captured["hook_attn"], output_attn, layer_path


def masked_stats(a, b, valid_key_mask):
    diff = (a - b).float().abs()
    mask = valid_key_mask[:, None, None, :].expand_as(diff)
    selected = diff[mask]
    return {
        "max_abs": selected.max().item(),
        "mean_abs": selected.mean().item(),
        "rmse": torch.sqrt((selected * selected).mean()).item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", default="benchmarks/Cypherbench/dev.jsonl")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--model-path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-length", type=int, default=892)
    parser.add_argument("--max-prompt-length", type=int, default=797)
    parser.add_argument("--layer-idx", type=int, default=-1)
    parser.add_argument("--top-rows", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="right")
    tokenizer.eos_token_id = 151645
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    with open(args.data_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx == args.sample_index:
                raw_item = json.loads(line)
                break
        else:
            raise IndexError(f"sample-index {args.sample_index} is out of range")

    sample = build_one_sample(raw_item, tokenizer, args.max_length, args.max_prompt_length, device)

    print("Loading model with attn_implementation='eager' so output_attentions is materialized.")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    layers, layer_path = get_layers(model)
    layer_idx = args.layer_idx if args.layer_idx >= 0 else len(layers) + args.layer_idx
    print(f"Comparing layer {layer_idx} from {layer_path}")

    hook_attn, output_attn, _ = capture_reconstructed_attention(model, layer_idx, sample)
    print(f"hook reconstructed attention: {tuple(hook_attn.shape)}")
    print(f"output_attentions attention:  {tuple(output_attn.shape)}")

    stats = masked_stats(hook_attn, output_attn, sample["attention_mask"].bool())
    print("\nDifference on valid key positions")
    for key, value in stats.items():
        print(f"{key}: {value:.8f}")

    input_ids_1d = sample["input_ids"][0]
    print("\nTop attention from hook reconstruction, Cypher -> Query")
    top_attention_to_region(
        hook_attn,
        tokenizer,
        input_ids_1d,
        sample["cypher_mask"],
        sample["query_mask"],
        row_limit=args.top_rows,
    )

    print("\nTop attention from output_attentions=True, Cypher -> Query")
    top_attention_to_region(
        output_attn,
        tokenizer,
        input_ids_1d,
        sample["cypher_mask"],
        sample["query_mask"],
        row_limit=args.top_rows,
    )

    print("\nTop attention from hook reconstruction, Cypher -> Schema")
    top_attention_to_region(
        hook_attn,
        tokenizer,
        input_ids_1d,
        sample["cypher_mask"],
        sample["schema_mask"],
        row_limit=args.top_rows,
    )

    print("\nTop attention from output_attentions=True, Cypher -> Schema")
    top_attention_to_region(
        output_attn,
        tokenizer,
        input_ids_1d,
        sample["cypher_mask"],
        sample["schema_mask"],
        row_limit=args.top_rows,
    )


if __name__ == "__main__":
    main()
