import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_llm import (
    MODEL_ALIASES,
    MODEL_CONFIGS,
    build_openai_client,
    generator,
)


PRED_KEYS = ("pred", "pred_cypher")


def is_empty_pred(value) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def get_pred_key(item: dict) -> str:
    for key in PRED_KEYS:
        if key in item:
            return key
    return "pred"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_schema(schema_dir: Path, graph: str, cache: dict):
    if graph in cache:
        return cache[graph]

    schema_path = schema_dir / f"{graph}_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")

    cache[graph] = load_json(schema_path)
    return cache[graph]


def resolve_model_config(model_name: str) -> tuple[str, dict]:
    model_key = MODEL_ALIASES.get(model_name, model_name)
    if model_key not in MODEL_CONFIGS:
        available = ", ".join(sorted(set(MODEL_CONFIGS) | set(MODEL_ALIASES)))
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    return model_key, MODEL_CONFIGS[model_key]


def regenerate_empty_predictions(
    data: list[dict],
    schema_dir: Path,
    client,
    model_cfg: dict,
    limit: int | None = None,
) -> tuple[list[dict], int]:
    schema_cache = {}
    regenerated = 0

    for idx, item in enumerate(data):
        pred_key = get_pred_key(item)
        if not is_empty_pred(item.get(pred_key)):
            continue

        question = item.get("question")
        graph = item.get("graph")
        if not question:
            raise ValueError(f"Item #{idx} has empty pred but no 'question' field.")
        if not graph:
            raise ValueError(f"Item #{idx} has empty pred but no 'graph' field.")

        schema = load_schema(schema_dir, graph, schema_cache)
        print(f"[{idx}] Regenerating empty {pred_key} for graph={graph}: {question}")
        item[pred_key] = generator(question, schema, client, model_cfg)
        regenerated += 1

        if limit is not None and regenerated >= limit:
            break

    return data, regenerated


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate Cypher for records whose pred/pred_cypher is empty."
    )
    parser.add_argument("--input", "-i", required=True, help="Result JSON from baseline_llm.py")
    parser.add_argument("--output", "-o", help="Output JSON path. Defaults to overwrite input with --inplace.")
    parser.add_argument("--inplace", action="store_true", help="Overwrite the input file.")
    parser.add_argument("--model", "-m", required=True, help="Model key or alias, e.g. qwen7b, deepseek")
    parser.add_argument(
        "--schema_dir",
        default=str(PROJECT_ROOT / "benchmarks" / "Cypherbench" / "graphs" / "schemas"),
        help="Directory containing files named <graph>_schema.json.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of empty predictions to regenerate.",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    if not args.inplace and not args.output:
        raise ValueError("Use --output to write a new file, or --inplace to overwrite input.")

    _, model_cfg = resolve_model_config(args.model)
    data = load_json(input_path)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of result objects.")

    client = build_openai_client()
    updated_data, regenerated = regenerate_empty_predictions(
        data=data,
        schema_dir=Path(args.schema_dir),
        client=client,
        model_cfg=model_cfg,
        limit=args.limit,
    )
    save_json(output_path, updated_data)
    print(f"Done. Regenerated {regenerated} empty predictions. Saved to {output_path}")


if __name__ == "__main__":
    main()
