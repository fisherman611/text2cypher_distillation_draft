import argparse
import json
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoTokenizer


QWEN_SEPARATOR_AFTER_UINT32 = 4294967295
GENERIC_SEPARATOR_AFTER_UINT16 = 65535


def build_prompt_and_tokens(raw_item, tokenizer, max_prompt_length):
    """Same idea as process_data.py: build chat prompt, then tokenize prompt/response."""
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
    if len(prompt_tokens) > max_prompt_length:
        prompt_tokens = prompt_tokens[:max_prompt_length]

    full_tokens = tokenizer.encode(prompt + raw_item["response"], add_special_tokens=False)
    response_tokens = full_tokens[len(prompt_tokens):] + [tokenizer.eos_token_id]

    return prompt, prompt_tokens, response_tokens


def simulate_processed_bin_item(prompt_tokens, response_tokens, model_type):
    """process_data.py writes prompt + [-1] + response into an unsigned mmap file.

    For qwen, dtype is uint32, so -1 is read back as 4294967295.
    For non-qwen uint16 data, -1 is read back as 65535.
    """
    separator = QWEN_SEPARATOR_AFTER_UINT32 if model_type == "qwen" else GENERIC_SEPARATOR_AFTER_UINT16
    return np.array(prompt_tokens + [separator] + response_tokens, dtype=np.int64)


def process_like_lm_dataset(input_ids, tokenizer, args):
    """Mirror LMTrainDataset._process_lm for a single sample."""
    pad_id = tokenizer.eos_token_id
    source_len = 1
    prompt = None

    if args.model_type == "qwen" and QWEN_SEPARATOR_AFTER_UINT32 in input_ids:
        source_len = np.where(input_ids == QWEN_SEPARATOR_AFTER_UINT32)[0][0]
        prompt = input_ids[:source_len]
        input_ids = np.concatenate([input_ids[:source_len], input_ids[source_len + 1 :]], axis=0)
    elif GENERIC_SEPARATOR_AFTER_UINT16 in input_ids:
        source_len = np.where(input_ids == GENERIC_SEPARATOR_AFTER_UINT16)[0][0]
        prompt = input_ids[:source_len]
        input_ids = np.concatenate([input_ids[:source_len], input_ids[source_len + 1 :]], axis=0)

    input_ids = input_ids[: args.max_length]
    input_len = len(input_ids)

    model_data = {
        "input_ids": torch.ones(1, args.max_length, dtype=torch.long) * pad_id,
        "attention_mask": torch.zeros(1, args.max_length),
    }
    no_model_data = {
        "label": torch.ones(1, args.max_length, dtype=torch.long) * -100,
        "loss_mask": torch.zeros(1, args.max_length),
    }
    gen_data = {
        "input_ids": torch.ones(1, args.max_prompt_length, dtype=torch.long) * pad_id,
        "attention_mask": torch.zeros(1, args.max_prompt_length, dtype=torch.long),
    }

    model_data["input_ids"][0][: input_len - 1] = torch.tensor(input_ids[:-1], dtype=torch.long)
    model_data["attention_mask"][0][: input_len - 1] = 1.0

    no_model_data["label"][0][: input_len - 1] = torch.tensor(input_ids[1:], dtype=torch.long)
    no_model_data["label"][0][: source_len - 1] = -100
    no_model_data["loss_mask"][0][: input_len - 1] = 1.0
    no_model_data["loss_mask"][0][: source_len - 1] = 0

    if prompt is not None:
        prompt = prompt[-args.max_prompt_length :]
        gen_data["input_ids"][0][-len(prompt) :] = torch.tensor(prompt, dtype=torch.long)
        gen_data["attention_mask"][0][-len(prompt) :] = 1

    return model_data, no_model_data, gen_data, source_len, input_len


def decode_token(tokenizer, token_id):
    if token_id == -100:
        return "<ignored>"
    return repr(tokenizer.decode([int(token_id)], skip_special_tokens=False))


def print_window(tokenizer, input_ids, labels, attention_mask, loss_mask, start, end):
    print("\nToken window")
    print("idx | attn | loss | input_id -> input_text | label_id -> label_text")
    print("-" * 88)
    for idx in range(start, end):
        inp = int(input_ids[idx])
        lab = int(labels[idx])
        print(
            f"{idx:>3} | {int(attention_mask[idx].item())}    | "
            f"{int(loss_mask[idx].item())}    | "
            f"{inp:>8} -> {decode_token(tokenizer, inp):<18} | "
            f"{lab:>8} -> {decode_token(tokenizer, lab)}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", default="benchmarks/Cypherbench/dev.jsonl")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--model-path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-type", default="qwen")
    parser.add_argument("--max-length", type=int, default=892)
    parser.add_argument("--max-prompt-length", type=int, default=797)
    parser.add_argument("--window", type=int, default=8)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="right")
    if args.model_type == "qwen":
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

    prompt, prompt_tokens, response_tokens = build_prompt_and_tokens(
        raw_item,
        tokenizer,
        args.max_prompt_length,
    )
    bin_item = simulate_processed_bin_item(prompt_tokens, response_tokens, args.model_type)
    model_data, no_model_data, gen_data, source_len, input_len = process_like_lm_dataset(
        bin_item,
        tokenizer,
        SimpleNamespace(
            model_type=args.model_type,
            max_length=args.max_length,
            max_prompt_length=args.max_prompt_length,
        ),
    )

    labels = no_model_data["label"][0]
    attention_mask = model_data["attention_mask"][0]
    loss_mask = no_model_data["loss_mask"][0]
    train_positions = torch.where(labels != -100)[0]
    first_train_idx = int(train_positions[0].item())
    last_train_idx = int(train_positions[-1].item())

    print("Raw sample")
    print(f"QUESTION starts with: {raw_item['user_prompt'].splitlines()[1]}")
    print(f"RESPONSE: {raw_item['response']}")

    print("\nLengths")
    print(f"prompt_tokens after truncation: {len(prompt_tokens)}")
    print(f"response_tokens incl eos:       {len(response_tokens)}")
    print(f"source_len from separator:      {source_len}")
    print(f"input_len after removing sep:   {input_len}")
    print(f"model input shape:              {tuple(model_data['input_ids'].shape)}")

    print("\nMasks")
    print(f"attention_mask sum: {int(attention_mask.sum().item())} real input tokens")
    print(f"loss_mask sum:      {int(loss_mask.sum().item())} supervised response positions")
    print(f"first loss idx:     {first_train_idx}")
    print(f"last loss idx:      {last_train_idx}")

    print("\nDecoded response reconstructed from labels")
    response_label_ids = labels[labels != -100].tolist()
    print(tokenizer.decode(response_label_ids, skip_special_tokens=False))

    start = max(0, first_train_idx - args.window)
    end = min(args.max_length, first_train_idx + args.window)
    print_window(
        tokenizer,
        model_data["input_ids"][0],
        labels,
        attention_mask,
        loss_mask,
        start,
        end,
    )

    print("\ngen_data used for generation/eval")
    print(f"gen_data input shape:         {tuple(gen_data['input_ids'].shape)}")
    print(f"gen_data attention_mask sum:  {int(gen_data['attention_mask'].sum().item())}")
    print("gen_data keeps the prompt right-aligned, so generate() starts from the prompt only.")


if __name__ == "__main__":
    main()
