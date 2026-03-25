import os
import json
import argparse
import threading
from pathlib import Path
from time import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import List, Dict, Any, Optional
from tqdm.auto import tqdm

from src.utils import read_json_file, build_messages
from src.baseline.qwen3 import init_model, generate_response
from src.llm_services import parse_json_from_string, parse_llm_response
from src.schema import Nl2CypherSample
from src.logger_config import setup_logger

RESULTS_DIR = "results"
LOG_DIR = "logging_data/qwen3"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, f"log_{int(time())}.txt")
logger = setup_logger(__name__, log_file=log_path)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load Qwen model and benchmark on text2cypher problems"
    )
    parser.add_argument(
        "--benchmark",
        default="Cypherbench",
        choices=["Cypherbench", "Mind_the_query", "Neo4j_Text2Cypher"],
        help="Benchmark name",
    )
    parser.add_argument(
        "--db",
        default=None,
        help='Database name of each benchmark. If omitted or set to "full", use all data.',
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-0.6B", help="Model name or path"
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Device to run the model on (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--max-length", type=int, default=1024, help="Max generation length"
    )
    parser.add_argument(
        "--max-workers", type=int, default=4, help="Number of worker threads"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of test samples"
    )
    return parser.parse_args()


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
        logger.warning(f"Schema file not found for graph={graph_name}: {schema_path}")
        return None

    schema = read_json_file(schema_path)
    return json.dumps(schema, indent=4, ensure_ascii=False)


def load_schema_and_subset_test_data(benchmark, db=None, limit=None):
    raw_test_data = read_json_file(Path("benchmarks") / benchmark / "test.json")

    use_all = is_full_db(db)
    subset_test_data = []

    for item in raw_test_data:
        try:
            sample = Nl2CypherSample(**item)
            if use_all or sample.graph == db:
                subset_test_data.append(sample)
        except Exception as e:
            logger.warning(f"Skipping invalid item: {e}")

    if limit is not None:
        subset_test_data = subset_test_data[:limit]

    shared_schema_str = None
    schema_map = {}

    if benchmark in ["Cypherbench", "Mind_the_query"]:
        if use_all:
            unique_graphs = sorted(
                {sample.graph for sample in subset_test_data if sample.graph}
            )
            for graph_name in unique_graphs:
                schema_map[graph_name] = load_schema_for_graph(benchmark, graph_name)
        else:
            shared_schema_str = load_schema_for_graph(benchmark, db)

    return subset_test_data, shared_schema_str, schema_map


def get_question_and_schema(
    sample: Nl2CypherSample,
    benchmark: str,
    shared_schema_str: Optional[str] = None,
    schema_map: Optional[Dict[str, Optional[str]]] = None,
):
    question = sample.nl_question

    if benchmark in ["Cypherbench", "Mind_the_query"]:
        if shared_schema_str is not None:
            schema_str = shared_schema_str
        else:
            schema_str = (schema_map or {}).get(sample.graph)
    else:
        schema_str = sample.schema

    return question, schema_str


def process_item(
    sample: Nl2CypherSample,
    benchmark,
    tokenizer,
    model,
    max_length,
    model_lock,
    shared_schema_str=None,
    schema_map=None,
):
    question, schema_str = get_question_and_schema(
        sample,
        benchmark,
        shared_schema_str,
        schema_map,
    )

    qid = sample.qid if sample.qid is not None else sample.instance_id
    if qid is None:
        qid = "unknown"

    logger.info("-" * 80)
    logger.info(f"Processing item: {qid}")
    logger.info(f"Graph: {sample.graph}")
    logger.info(f"Question: {question}")

    try:
        messages = build_messages(question, schema_str)

        with model_lock:
            raw_response = generate_response(
                tokenizer,
                model,
                messages,
                max_length=max_length,
            )

        parsed = parse_llm_response(raw_response)
        parsed_json = parse_json_from_string(parsed["final_answer"])

        if not parsed_json or "cypher" not in parsed_json:
            raise ValueError("Failed to parse JSON or missing 'cypher' key")

        cypher = parsed_json["cypher"]
        sample.pred_cypher = cypher

        logger.info(f"Generated cypher for item {qid}: {cypher}")

        return {
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
        }

    except Exception as e:
        logger.error(f"Error processing item {qid}: {e}", exc_info=True)
        sample.pred_cypher = None
        return {
            "qid": qid,
            "graph": sample.graph,
            "question": question,
            "raw_response": None,
            "think": "",
            "final_answer": "",
            "cypher": None,
            "success": False,
            "error": str(e),
            "sample": sample.model_dump(mode="json"),
        }


def run_parallel_inference(
    test_data,
    benchmark,
    db,
    tokenizer,
    model,
    max_length,
    max_workers=4,
    shared_schema_str=None,
    schema_map=None,
):
    results = []
    errors = []
    model_lock = threading.Lock()

    run_name = db if not is_full_db(db) else "full"

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {}

        for sample in test_data:
            future = executor.submit(
                process_item,
                sample,
                benchmark,
                tokenizer,
                model,
                max_length,
                model_lock,
                shared_schema_str,
                schema_map,
            )
            future_to_item[future] = (
                sample.qid if sample.qid is not None else sample.instance_id
            )

        progress_bar = tqdm(
            as_completed(future_to_item),
            total=len(future_to_item),
            desc=f"Running {benchmark}/{run_name}",
        )

        for future in progress_bar:
            qid = future_to_item[future]
            try:
                result = future.result()
                results.append(result)

                if result["success"]:
                    progress_bar.set_postfix(success=qid)
                else:
                    errors.append({"qid": qid, "error": result["error"]})
                    progress_bar.set_postfix(failed=qid)

            except Exception as e:
                errors.append({"qid": qid, "error": str(e)})
                logger.error(f"Unhandled future error for item {qid}: {e}", exc_info=True)
                progress_bar.set_postfix(failed=qid)

    logger.info(f"Finished. Success: {len(results) - len(errors)}, Failed: {len(errors)}")
    return results, errors


def main():
    args = parse_args()

    subset_test_data, schema_str, schema_map = load_schema_and_subset_test_data(
        args.benchmark,
        args.db,
        args.limit,
    )

    db_name = args.db if not is_full_db(args.db) else "full"

    print(f"Loading model: {args.model}")
    logger.info(f"Loading model: {args.model}")
    tokenizer, model = init_model(args.model)

    print(
        f"Running benchmark={args.benchmark}, db={db_name}, samples={len(subset_test_data)}"
    )
    print(f"Using max_workers={args.max_workers}")

    results, errors = run_parallel_inference(
        test_data=subset_test_data,
        benchmark=args.benchmark,
        db=args.db,
        tokenizer=tokenizer,
        model=model,
        max_length=args.max_length,
        max_workers=args.max_workers,
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
    output_path = Path(RESULTS_DIR) / args.benchmark / f"{db_name}_cyphers_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()