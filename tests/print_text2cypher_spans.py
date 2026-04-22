import argparse
import json
import os
import sys
import types
from collections import defaultdict

# Ensure repo root is importable when script is run directly.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Lightweight stubs to import lm_datasets without full training dependencies.
if "deepspeed" not in sys.modules:
    sys.modules["deepspeed"] = types.ModuleType("deepspeed")
if "accelerate" not in sys.modules:
    accelerate_mod = types.ModuleType("accelerate")
    accelerate_mod.load_checkpoint_and_dispatch = lambda *args, **kwargs: None
    accelerate_mod.init_empty_weights = lambda *args, **kwargs: None
    sys.modules["accelerate"] = accelerate_mod
if "peft" not in sys.modules:
    peft_mod = types.ModuleType("peft")
    peft_mod.get_peft_model = lambda *args, **kwargs: None
    peft_mod.LoraConfig = object
    peft_mod.TaskType = object
    peft_mod.PeftModel = object
    sys.modules["peft"] = peft_mod
if "transformers" not in sys.modules:
    tr_mod = types.ModuleType("transformers")
    tr_mod.AutoModelForCausalLM = object
    tr_mod.AutoTokenizer = object
    tr_mod.AutoConfig = object
    sys.modules["transformers"] = tr_mod

from data_utils.lm_datasets import (
    extract_text2cypher_span_items_from_response,
    extract_text2cypher_span_offsets,
)


def build_prompt(item):
    if "prompt" in item:
        return item["prompt"]
    system_prompt = item.get("system_prompt", "")
    user_prompt = item.get("user_prompt", "")
    return f"{system_prompt}\n\n{user_prompt}"


def main():
    parser = argparse.ArgumentParser(description="Print Text2Cypher spans for one JSONL sample.")
    parser.add_argument(
        "--jsonl",
        default="benchmarks/Cypherbench/dev.jsonl",
        help="Path to input JSONL file.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Sample index in the JSONL file.",
    )
    args = parser.parse_args()

    with open(args.jsonl, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if args.index < 0 or args.index >= len(lines):
        raise IndexError(f"index={args.index} is out of range. total={len(lines)}")

    item = json.loads(lines[args.index])
    response = item["response"]
    prompt = build_prompt(item)
    full_text = prompt + response

    span_items = extract_text2cypher_span_items_from_response(response)
    offsets = extract_text2cypher_span_offsets(full_text, response)

    response_json = json.loads(response)
    cypher = response_json.get("cypher", "")

    print("=" * 80)
    print(f"File: {args.jsonl}")
    print(f"Index: {args.index}")
    print("=" * 80)
    print("Cypher:")
    print(cypher)
    print("=" * 80)

    by_type = defaultdict(list)
    for item_span in span_items:
        by_type[item_span["type"]].append(item_span)

    for key in ["clause", "pattern", "expression", "variable_alias"]:
        print(f"\n[{key}]")
        for i, sp in enumerate(by_type.get(key, []), start=1):
            print(f"{i:02d}. ({sp['start']}, {sp['end']}) -> {sp['text']}")

    print("\n[Offsets mapped on full_text]")
    for i, (s, e) in enumerate(offsets, start=1):
        print(f"{i:02d}. ({s}, {e}) -> {full_text[s:e]}")


if __name__ == "__main__":
    main()
