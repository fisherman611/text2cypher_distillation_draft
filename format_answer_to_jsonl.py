import json
import argparse
import os


def process_jsonl_to_jsonl(input_file: str, output_file: str) -> None:
    """
    Process JSONL file:
    - Always output 1 line per input line
    - If valid → {"parsed_text": parsed_json}
    - If invalid → {"parsed_text": ""}
    """
    total = 0
    success = 0

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line_idx, line in enumerate(fin, start=1):
            total += 1
            line = line.strip()

            parsed_output = line  # default if anything fails

            if line:
                try:
                    record = json.loads(line)

                    text_value = record.get("text")
                    if isinstance(text_value, str):
                        try:
                            parsed_output = json.loads(text_value)
                            success += 1
                        except json.JSONDecodeError:
                            pass

                except json.JSONDecodeError:
                    pass

            fout.write(json.dumps({
                "parsed_text": parsed_output
            }, ensure_ascii=False) + "\n")

    print(f"Processed: {total} lines")
    print(f"Valid parsed: {success}")
    print(f"Invalid / fallback: {total - success}")
    print(f"Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and parse nested JSON from 'text' field in a JSONL file."
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input JSONL file"
    )

    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to output JSONL file"
    )

    args = parser.parse_args()

    # create parent directory if needed
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    process_jsonl_to_jsonl(args.input, args.output)


if __name__ == "__main__":
    main()