"""
main_vllm.py – vLLM-accelerated text2cypher inference
======================================================

Usage
-----
python main_vllm.py \
    --benchmark Cypherbench \
    --model Qwen/Qwen3-0.6B \
    --batch-size 32 \
    --max-new-tokens 1024

Key differences from main.py
-----------------------------
* Uses vLLM's batched `generate()` instead of ``ThreadPoolExecutor`` – vLLM
  handles concurrency internally and is far more GPU-efficient.
* No model_lock is required.
* Adds ``--batch-size`` and ``--tensor-parallel-size`` arguments.
* ``--max-length`` renamed to ``--max-new-tokens`` to match vLLM semantics
  (vLLM's ``max_tokens`` counts *generated* tokens, not input + output).
"""

import os
import json
import argparse
from pathlib import Path
from time import time
from typing import List, Dict, Any, Optional

from tqdm.auto import tqdm

from src.utils import read_json_file, build_messages
from src.baseline.vllm_model import init_model, generate_batch
from src.llm_services import parse_json_from_string, parse_llm_response
from src.schema import Nl2CypherSample
from src.logger_config import setup_logger

# ---------------------------------------------------------------------------
RESULTS_DIR = "results"
LOG_DIR = "logging_data/vllm"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, f"log_{int(time())}.txt")
logger = setup_logger(__name__, log_file=log_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="vLLM-accelerated text2cypher inference"
    )
    parser.add_argument(
        "--benchmark",
        default="Cypherbench",
        choices=["Cypherbench", "Mind_the_query", "Neo4j_Text2Cypher"],
    )
    parser.add_argument(
        "--db",
        default=None,
        help='Database name. If omitted or "full", uses all data.',
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HF model name or local path")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max *generated* tokens per sample")
    parser.add_argument("--batch-size", type=int, default=32, help="Number of samples per vLLM batch")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="Fraction of GPU memory for vLLM KV cache")
    parser.add_argument("--max-model-len", type=int, default=None, help="Override model context length")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--enable-thinking", action="store_true", help="Enable Qwen3 <think> tokens")
    parser.add_argument("--limit", type=int, default=None, help="Cap on number of test samples")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data helpers (identical to main.py)
# ---------------------------------------------------------------------------

def is_full_db(db: Optional[str]) -> bool:
    return db is None or str(db).strip().lower() in {"full", "all", ""}


def load_schema_for_graph(benchmark: str, graph_name: str) -> Optional[str]:
    if benchmark == "Cypherbench":
        schema_path = Path("benchmarks") / benchmark / "graphs" / "schemas" / f"{graph_name}_schema.json"
    elif benchmark == "Mind_the_query":
        schema_path = Path("benchmarks") / benchmark / "graphs" / "schemas" / f"{graph_name}.json"
    else:
        return None

    if not schema_path.exists():
        logger.warning(f"Schema not found for graph={graph_name}: {schema_path}")
        return None

    schema = read_json_file(schema_path)
    return json.dumps(schema, indent=4, ensure_ascii=False)


def load_schema_and_subset_test_data(benchmark, db=None, limit=None):
    raw_test_data = read_json_file(Path("benchmarks") / benchmark / "test.json")

    use_all = is_full_db(db)
    subset = []

    for item in raw_test_data:
        try:
            sample = Nl2CypherSample(**item)
            if use_all or sample.graph == db:
                subset.append(sample)
        except Exception as e:
            logger.warning(f"Skipping invalid item: {e}")

    if limit is not None:
        subset = subset[:limit]

    shared_schema_str = None
    schema_map: Dict[str, Optional[str]] = {}

    if benchmark in ["Cypherbench", "Mind_the_query"]:
        if use_all:
            for graph_name in sorted({s.graph for s in subset if s.graph}):
                schema_map[graph_name] = load_schema_for_graph(benchmark, graph_name)
        else:
            shared_schema_str = load_schema_for_graph(benchmark, db)

    return subset, shared_schema_str, schema_map


def get_question_and_schema(
    sample: Nl2CypherSample,
    benchmark: str,
    shared_schema_str: Optional[str] = None,
    schema_map: Optional[Dict[str, Optional[str]]] = None,
):
    question = sample.nl_question
    if benchmark in ["Cypherbench", "Mind_the_query"]:
        schema_str = shared_schema_str if shared_schema_str is not None else (schema_map or {}).get(sample.graph)
    else:
        schema_str = sample.schema
    return question, schema_str


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

def run_batch_inference(
    test_data: List[Nl2CypherSample],
    benchmark: str,
    db: Optional[str],
    tokenizer,
    llm,
    max_new_tokens: int,
    batch_size: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool,
    shared_schema_str: Optional[str] = None,
    schema_map: Optional[Dict[str, Optional[str]]] = None,
):
    results = []
    errors = []
    run_name = db if not is_full_db(db) else "full"

    # Build (qid, question, schema, messages) for every sample upfront
    meta = []
    for sample in test_data:
        question, schema_str = get_question_and_schema(sample, benchmark, shared_schema_str, schema_map)
        qid = sample.qid if sample.qid is not None else sample.instance_id
        if qid is None:
            qid = "unknown"
        messages = build_messages(question, schema_str)
        meta.append((qid, question, sample, messages))

    # Process in batches
    total_batches = (len(meta) + batch_size - 1) // batch_size
    pbar_samples = tqdm(total=len(meta), desc="  Samples", unit="sample", position=1, leave=True)
    pbar_batches = tqdm(total=total_batches, desc=f"vLLM {benchmark}/{run_name}", unit="batch", position=0, leave=True)

    for batch_idx in range(total_batches):
        batch_meta = meta[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        batch_messages = [m[3] for m in batch_meta]

        # ------------------------------------------------------------------
        # Single vLLM call for the whole mini-batch
        # ------------------------------------------------------------------
        try:
            raw_responses = generate_batch(
                tokenizer,
                llm,
                batch_messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                enable_thinking=enable_thinking,
            )
        except Exception as e:
            # If the entire batch fails, record each sample as failed
            for qid, question, sample, _ in batch_meta:
                errors.append({"qid": qid, "error": str(e)})
                results.append({
                    "qid": qid, "graph": sample.graph, "question": question,
                    "raw_response": None, "think": "", "final_answer": "",
                    "cypher": None, "success": False, "error": str(e),
                    "sample": sample.model_dump(mode="json"),
                })
            logger.error(f"Batch {batch_idx} failed: {e}", exc_info=True)
            pbar_samples.update(len(batch_meta))
            pbar_batches.update(1)
            continue

        # ------------------------------------------------------------------
        # Parse each response
        # ------------------------------------------------------------------
        for (qid, question, sample, _), raw_response in zip(batch_meta, raw_responses):
            logger.info("-" * 80)
            logger.info(f"Processing item: {qid}")
            logger.info(f"Graph: {sample.graph} | Question: {question}")

            try:
                parsed = parse_llm_response(raw_response)
                parsed_json = parse_json_from_string(parsed["final_answer"])

                if not parsed_json or "cypher" not in parsed_json:
                    raise ValueError("Failed to parse JSON or missing 'cypher' key")

                cypher = parsed_json["cypher"]
                sample.pred_cypher = cypher
                logger.info(f"Generated cypher for {qid}: {cypher}")

                results.append({
                    "qid": qid,
                    "graph": sample.graph,
                    "question": question,
                    "raw_response": raw_response,
                    "think": parsed.get("think", ""),
                    "final_answer": parsed.get("final_answer", ""),
                    "cypher": cypher,
                    "success": True,
                    "error": None,
                    "sample": sample.model_dump(mode="json"),
                })

            except Exception as e:
                logger.error(f"Error parsing item {qid}: {e}", exc_info=True)
                sample.pred_cypher = None
                errors.append({"qid": qid, "error": str(e)})
                results.append({
                    "qid": qid,
                    "graph": sample.graph,
                    "question": question,
                    "raw_response": raw_response,
                    "think": "",
                    "final_answer": "",
                    "cypher": None,
                    "success": False,
                    "error": str(e),
                    "sample": sample.model_dump(mode="json"),
                })

            pbar_samples.update(1)

        pbar_batches.update(1)

    pbar_samples.close()
    pbar_batches.close()
    logger.info(f"Done. Success: {len(results) - len(errors)}, Failed: {len(errors)}")
    return results, errors


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    subset_test_data, schema_str, schema_map = load_schema_and_subset_test_data(
        args.benchmark, args.db, args.limit
    )

    db_name = args.db if not is_full_db(args.db) else "full"

    print(f"Loading model via vLLM: {args.model}")
    logger.info(f"Loading model: {args.model}")

    tokenizer, llm = init_model(
        model_name=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    print(
        f"Benchmark={args.benchmark} | db={db_name} | "
        f"samples={len(subset_test_data)} | batch_size={args.batch_size}"
    )

    results, errors = run_batch_inference(
        test_data=subset_test_data,
        benchmark=args.benchmark,
        db=args.db,
        tokenizer=tokenizer,
        llm=llm,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=args.enable_thinking,
        shared_schema_str=schema_str,
        schema_map=schema_map,
    )

    output = {
        "benchmark": args.benchmark,
        "db": db_name,
        "model": args.model,
        "num_samples": len(subset_test_data),
        "num_success": sum(1 for r in results if r["success"]),
        "num_failed": len(errors),
        "results": results,
        "updated_samples": [r["sample"] for r in results],
        "errors": errors,
    }

    os.makedirs(Path(RESULTS_DIR) / args.benchmark, exist_ok=True)
    output_path = Path(RESULTS_DIR) / args.benchmark / f"{db_name}_cyphers_result_vllm.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
