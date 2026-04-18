import argparse
import json
import re

from transformers import AutoTokenizer


QUESTION_PATTERN = re.compile(r"QUESTION:\s*\n(.*?)\n\s*\nSCHEMA:", re.DOTALL)


def build_query_token_ids(tokenizer, text, query_start, query_end, max_length):
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=False,
    )

    query_token_ids = []
    for token_id, (token_start, token_end) in zip(encoded["input_ids"], encoded["offset_mapping"]):
        overlaps_query = token_start < query_end and token_end > query_start
        is_real_token = token_end > token_start
        if overlaps_query and is_real_token:
            query_token_ids.append(token_id)

    return query_token_ids


def main():
    parser = argparse.ArgumentParser(
        description="Check that query attention mask only covers the natural language QUESTION span."
    )
    parser.add_argument(
        "--data-path",
        default="processed_data/benchmarks/Cypherbench/qwen/train.jsonl",
        help="Processed jsonl file containing prompt/response fields.",
    )
    parser.add_argument(
        "--model-path",
        default="Qwen/Qwen3-0.6B",
        help="Tokenizer path/name used to build the processed data.",
    )
    parser.add_argument("--max-length", type=int, default=892)
    parser.add_argument("--preview", type=int, default=3, help="Number of decoded examples to print.")
    parser.add_argument("--limit", type=int, default=-1, help="Number of rows to scan; -1 scans all rows.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="right")

    total = 0
    no_match = 0
    zero_query_tokens = 0
    suspicious = 0
    previewed = 0

    with open(args.data_path, encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            if args.limit != -1 and row_idx >= args.limit:
                break

            item = json.loads(line)
            prompt = item["prompt"]
            response = item.get("response", "")
            text = prompt + response
            total += 1

            match = QUESTION_PATTERN.search(prompt)
            if match is None:
                no_match += 1
                print(f"[NO_MATCH] row={row_idx}")
                continue

            query_start, query_end = match.span(1)
            query_text = text[query_start:query_end]
            query_token_ids = build_query_token_ids(
                tokenizer, text, query_start, query_end, args.max_length
            )
            decoded_query_tokens = tokenizer.decode(query_token_ids, skip_special_tokens=False)

            if len(query_token_ids) == 0:
                zero_query_tokens += 1
                print(f"[ZERO_QUERY_TOKENS] row={row_idx} question={query_text!r}")

            contains_non_query_text = any(
                marker in decoded_query_tokens
                for marker in ("SCHEMA:", "Generate a Cypher", "<|im_start|>", "<|im_end|>")
            )
            if contains_non_query_text:
                suspicious += 1
                print(f"[SUSPICIOUS] row={row_idx} decoded={decoded_query_tokens!r}")

            if previewed < args.preview:
                print("=" * 100)
                print(f"row: {row_idx}")
                print("CHAR QUESTION:")
                print(query_text)
                print("-" * 100)
                print("DECODED QUERY MASK TOKENS:")
                print(decoded_query_tokens)
                print("-" * 100)
                print(f"query char length: {len(query_text)}")
                print(f"query token count: {len(query_token_ids)}")
                previewed += 1

    print("=" * 100)
    print("SUMMARY")
    print(f"total rows checked: {total}")
    print(f"regex no-match rows: {no_match}")
    print(f"zero query-token rows: {zero_query_tokens}")
    print(f"suspicious decoded rows: {suspicious}")

    if no_match == 0 and zero_query_tokens == 0 and suspicious == 0:
        print("OK: query mask extraction looks correct.")
    else:
        print("CHECK: inspect the rows reported above.")


if __name__ == "__main__":
    main()
