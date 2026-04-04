import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import argparse
from metrics.executable import executable
from metrics.execution_accuracy import execution_accuracy
from metrics.provenance_subgraph_jaccard_similarity import provenance_subgraph_jaccard_similarity
from neo4j_connector import Neo4jConnector
from tqdm.auto import tqdm


def safe_compute(metric_fn, pred_cypher, target_cypher, conn, metric_name):
    try:
        return metric_fn(pred_cypher, target_cypher, conn)
    except Exception as e:
        return {
            "error": f"{metric_name} failed: {str(e)}"
        }


def main():
    parser = argparse.ArgumentParser(description="Calculate CypherBench metrics and save results to JSON")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input final_results.json"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSON file"
    )
    parser.add_argument(
        "--subset",
        default="nba",
        choices=[
            "nba",
            "flight_accident",
            "fictional_character",
            "company",
            "geography",
            "movie",
            "politics",
        ],
        help="Subset / graph name to evaluate"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit number of samples to evaluate"
    )
    parser.add_argument(
        "--host",
        default="neo4j://127.0.0.1",
        help="Neo4j host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7687,
        help="Neo4j port"
    )
    parser.add_argument(
        "--username",
        default="neo4j",
        help="Neo4j username"
    )
    parser.add_argument(
        "--password",
        default="lht11062004",
        help="Neo4j password"
    )
    parser.add_argument(
        "--name",
        default="cypherbench-db",
        help="Connector name"
    )

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        results = json.load(f)

    subset_results = [item for item in results if item.get("graph") == args.subset]

    if args.limit is not None:
        subset_results = subset_results[:args.limit]

    conn = Neo4jConnector(
        name=args.name,
        host=args.host,
        port=args.port,
        username=args.username,
        password=args.password,
        database=args.subset.replace("_", "."),
    )

    output_results = []

    for idx, item in enumerate(tqdm(subset_results, desc=f"Evaluating {args.subset}")):
        pred_cypher = item.get("pred_cypher", "")
        target_cypher = item.get("gold_cypher", "")

        exec_score = safe_compute(
            executable,
            pred_cypher,
            target_cypher,
            conn,
            "executable"
        )
        ex_score = safe_compute(
            execution_accuracy,
            pred_cypher,
            target_cypher,
            conn,
            "execution_accuracy"
        )
        psjs_score = safe_compute(
            provenance_subgraph_jaccard_similarity,
            pred_cypher,
            target_cypher,
            conn,
            "provenance_subgraph_jaccard_similarity"
        )

        output_results.append({
            "graph": item.get("graph", ""),
            "gold_cypher": target_cypher,
            "pred_cypher": pred_cypher,
            "executable": exec_score,
            "execution_accuracy": ex_score,
            "provenance_subgraph_jaccard_similarity": psjs_score,
        })

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(output_results)} results to {args.output}")


if __name__ == "__main__":
    main()