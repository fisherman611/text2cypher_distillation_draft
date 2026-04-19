import argparse
import json
import math
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_between(text, start_marker, end_marker=None):
    start = text.index(start_marker) + len(start_marker)
    end = text.index(end_marker, start) if end_marker is not None else len(text)
    return text[start:end].strip()


def overlap_mask(offsets, char_start, char_end, max_length, device=None):
    mask = torch.zeros(max_length, dtype=torch.bool, device=device)
    for idx, (tok_start, tok_end) in enumerate(offsets[:max_length]):
        if tok_end <= char_start:
            continue
        if tok_start >= char_end:
            continue
        if tok_start == tok_end:
            continue
        mask[idx] = True
    return mask


def build_one_sample(raw_item, tokenizer, max_length, max_prompt_length, device):
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": raw_item["system_prompt"]},
            {"role": "user", "content": raw_item["user_prompt"]},
        ],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_tokens = prompt_tokens[:max_prompt_length]

    full_tokens = tokenizer.encode(prompt + raw_item["response"], add_special_tokens=False)
    response_tokens = full_tokens[len(prompt_tokens):] + [tokenizer.eos_token_id]
    full_ids = (prompt_tokens + response_tokens)[:max_length]

    input_ids = torch.full((1, max_length), tokenizer.pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((1, max_length), dtype=torch.long, device=device)
    labels = torch.full((1, max_length), -100, dtype=torch.long, device=device)

    input_len = len(full_ids)
    input_ids[0, : input_len - 1] = torch.tensor(full_ids[:-1], dtype=torch.long, device=device)
    attention_mask[0, : input_len - 1] = 1
    labels[0, : input_len - 1] = torch.tensor(full_ids[1:], dtype=torch.long, device=device)

    source_len = len(prompt_tokens)
    labels[0, : source_len - 1] = -100
    cypher_mask = labels[0] != -100

    user_prompt = raw_item["user_prompt"]
    query_text = find_between(user_prompt, "QUESTION:\n", "\n\nSCHEMA:")
    schema_text = find_between(user_prompt, "SCHEMA:\n", "\n\nGenerate a Cypher query")

    query_char_start = prompt.find(query_text)
    schema_char_start = prompt.find(schema_text)
    if query_char_start < 0:
        raise ValueError("Could not find query_text in chat prompt.")
    if schema_char_start < 0:
        raise ValueError("Could not find schema_text in chat prompt.")

    query_char_end = query_char_start + len(query_text)
    schema_char_end = schema_char_start + len(schema_text)

    offset_enc = tokenizer(
        prompt,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=max_prompt_length,
    )
    offsets = offset_enc["offset_mapping"]

    query_mask = overlap_mask(offsets, query_char_start, query_char_end, max_length, device=device)
    schema_mask = overlap_mask(offsets, schema_char_start, schema_char_end, max_length, device=device)

    return {
        "prompt": prompt,
        "query_text": query_text,
        "schema_text": schema_text,
        "response": raw_item["response"],
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "query_mask": query_mask,
        "schema_mask": schema_mask,
        "cypher_mask": cypher_mask,
        "source_len": source_len,
        "input_len": input_len,
    }


def decode_positions(tokenizer, input_ids, positions, limit=40):
    pieces = []
    for pos in positions[:limit]:
        token_id = int(input_ids[pos].item())
        pieces.append((int(pos), token_id, tokenizer.decode([token_id], skip_special_tokens=False)))
    return pieces


def print_mask_summary(name, tokenizer, input_ids, mask, limit=40):
    positions = torch.where(mask)[0].tolist()
    print(f"\n{name}")
    print(f"count: {len(positions)}")
    if positions:
        print(f"span token idx: {positions[0]} -> {positions[-1]}")
    for pos, token_id, text in decode_positions(tokenizer, input_ids, positions, limit=limit):
        safe_text = repr(text)
        print(f"{pos:>4} | {token_id:>8} | {safe_text}")
    if len(positions) > limit:
        print(f"... ({len(positions) - limit} more tokens)")


def get_layers(model):
    candidates = [
        "model.layers",
        "base_model.model.model.layers",
        "module.model.layers",
        "module.base_model.model.model.layers",
    ]
    for path in candidates:
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok:
            return obj, path
    raise AttributeError("Could not find transformer layers on this model.")


def get_num_heads(model, layer):
    config = model.config
    num_heads = getattr(config, "num_attention_heads")
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        q_out = layer.self_attn.q_proj.out_features
        head_dim = q_out // num_heads
    return num_heads, num_kv_heads, head_dim


def repeat_kv(hidden_states, num_key_value_groups):
    if num_key_value_groups == 1:
        return hidden_states
    bsz, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        bsz, num_kv_heads, num_key_value_groups, slen, head_dim
    )
    return hidden_states.reshape(bsz, num_kv_heads * num_key_value_groups, slen, head_dim)


def collect_qk_outputs(model, layer_idx, batch):
    layers, path = get_layers(model)
    layer = layers[layer_idx]
    captured = {}
    handles = []

    def save_output(name):
        def hook(_module, _inputs, output):
            captured[name] = output.detach()
        return hook

    handles.append(layer.self_attn.q_proj.register_forward_hook(save_output("q")))
    handles.append(layer.self_attn.k_proj.register_forward_hook(save_output("k")))

    with torch.no_grad():
        _ = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )

    for handle in handles:
        handle.remove()

    if "q" not in captured or "k" not in captured:
        raise RuntimeError("Could not capture q_proj/k_proj outputs.")

    return captured["q"], captured["k"], layer, path


def reconstruct_prerope_attention(model, layer, q, k, attention_mask):
    num_heads, num_kv_heads, head_dim = get_num_heads(model, layer)
    bsz, seq_len, _ = q.shape
    kv_groups = num_heads // num_kv_heads

    q = q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    k = repeat_kv(k, kv_groups)

    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(head_dim)

    key_mask = attention_mask[:, None, None, :].bool()
    row_idx = torch.arange(seq_len, device=scores.device)[:, None]
    col_idx = torch.arange(seq_len, device=scores.device)[None, :]
    causal_mask = col_idx <= row_idx
    scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
    scores = scores.masked_fill(~causal_mask[None, None, :, :], torch.finfo(scores.dtype).min)
    return F.softmax(scores, dim=-1)


def top_attention_to_region(attn, tokenizer, input_ids, row_mask, col_mask, row_limit=8, top_k=5):
    attn_mean = attn.mean(dim=1)[0]  # [L, L]
    row_positions = torch.where(row_mask)[0].tolist()[:row_limit]
    col_positions = torch.where(col_mask)[0]

    if len(row_positions) == 0 or col_positions.numel() == 0:
        print("No valid rows or columns for this region.")
        return

    for row in row_positions:
        scores = attn_mean[row, col_positions]
        k = min(top_k, scores.numel())
        vals, local_idxs = torch.topk(scores, k=k)
        label_text = tokenizer.decode([int(input_ids[row].item())], skip_special_tokens=False)
        print(f"\nrow {row} input={repr(label_text)}")
        for val, local_idx in zip(vals.tolist(), local_idxs.tolist()):
            col = int(col_positions[local_idx].item())
            text = tokenizer.decode([int(input_ids[col].item())], skip_special_tokens=False)
            print(f"  -> col {col:>4} score={val:.6f} token={repr(text)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", default="benchmarks/Cypherbench/dev.jsonl")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--model-path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-length", type=int, default=892)
    parser.add_argument("--max-prompt-length", type=int, default=797)
    parser.add_argument("--print-limit", type=int, default=32)
    parser.add_argument("--with-model", action="store_true")
    parser.add_argument("--layer-idx", type=int, default=-1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.with_model else "cpu")
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
    input_ids_1d = sample["input_ids"][0]

    print("Raw example")
    print(f"QUESTION: {sample['query_text']}")
    print(f"RESPONSE: {sample['response']}")
    print("\nLengths")
    print(f"source_len(prompt): {sample['source_len']}")
    print(f"input_len(full):    {sample['input_len']}")
    print(f"attention sum:      {int(sample['attention_mask'].sum().item())}")
    print(f"query tokens:       {int(sample['query_mask'].sum().item())}")
    print(f"schema tokens:      {int(sample['schema_mask'].sum().item())}")
    print(f"cypher rows:        {int(sample['cypher_mask'].sum().item())}")

    print_mask_summary("QUERY MASK TOKENS", tokenizer, input_ids_1d, sample["query_mask"], args.print_limit)
    print_mask_summary("SCHEMA MASK TOKENS", tokenizer, input_ids_1d, sample["schema_mask"], args.print_limit)
    print_mask_summary("CYPHER MASK ROWS", tokenizer, input_ids_1d, sample["cypher_mask"], args.print_limit)

    label_ids = sample["labels"][0][sample["labels"][0] != -100]
    print("\nDecoded labels from cypher_mask rows")
    print(tokenizer.decode(label_ids.tolist(), skip_special_tokens=False))

    if not args.with_model:
        print("\nModel hook test skipped. Add --with-model to capture q_proj/k_proj and reconstruct pre-RoPE attention.")
        return

    print("\nLoading model for q/k hook test...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    layers, layer_path = get_layers(model)
    layer_idx = args.layer_idx if args.layer_idx >= 0 else len(layers) + args.layer_idx
    print(f"Hooking layer {layer_idx} from {layer_path}")

    q, k, layer, _ = collect_qk_outputs(model, layer_idx, sample)
    print(f"captured q shape: {tuple(q.shape)}")
    print(f"captured k shape: {tuple(k.shape)}")
    attn = reconstruct_prerope_attention(model, layer, q, k, sample["attention_mask"])
    print(f"reconstructed attention shape: {tuple(attn.shape)}")
    print("NOTE: This q/k hook attention is pre-RoPE, so use it as a wiring/alignment test, not final exact Qwen attention.")

    print("\nTop attention from early cypher rows -> QUERY mask")
    top_attention_to_region(attn, tokenizer, input_ids_1d, sample["cypher_mask"], sample["query_mask"])

    print("\nTop attention from early cypher rows -> SCHEMA mask")
    top_attention_to_region(attn, tokenizer, input_ids_1d, sample["cypher_mask"], sample["schema_mask"])


if __name__ == "__main__":
    main()
