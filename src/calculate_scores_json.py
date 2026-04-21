import argparse
import copy
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import math
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.metrics import *
from src.neo4j_connector import Neo4jConnector
from src.schema import Nl2CypherSample

RETURN_PATTERN_MAPPING = {
    "n_name": "n_name",
    "n_prop": "n_prop_combined",
    "n_name_prop": "n_prop_combined",
    "n_prop_distinct": "n_prop_combined",
    "n_prop_array_distinct": "n_prop_combined",
    "n_order_by": "n_order_by",
    "n_argmax": "n_argmax",
    "n_where": "n_where",
    "n_agg": "n_agg",
    "n_group_by": "n_group_by"
}

METRIC_FUNC_MAPPING = {
    'execution_accuracy': execution_accuracy,
    'psjs': provenance_subgraph_jaccard_similarity,
    'executable': executable,
}


def compute_metrics(item: Nl2CypherSample, metrics, neo4j_conn):
    item = copy.deepcopy(item)
    for m in metrics:
        pred_cypher = item.pred_cypher
        if pred_cypher.endswith('<end_of_turn>'):
            pred_cypher = pred_cypher[:-len('<end_of_turn>')].strip()
        item.metrics[m] = METRIC_FUNC_MAPPING[m](
            pred_cypher=pred_cypher,
            target_cypher=item.gold_cypher,
            neo4j_connector=neo4j_conn
        )
    return item


def avg_and_round(nums: list[float], n: int = 4):
    return round(sum(nums) / len(nums), n) if nums else math.nan


def aggregate(results: list[tuple[str, float]]):
    res = {}
    for key, value in results:
        if key not in res:
            res[key] = []
        res[key].append(value)
    for key, values in res.items():
        res[key] = avg_and_round(values)
    return res

def calculate_result(result, metrics=['execution_accuracy', 'psjs', 'executable']):
    aggregated = {
        'overall': {
            m: avg_and_round(
                [(item["metrics"][m] if not isinstance(item["metrics"][m], dict) else 0.0)
                 for item in result if "metrics" in item and m in item["metrics"]]
            )
            for m in metrics
        }
    }
    return aggregated


# with open(r"format_results/Cypherbench/qwen3/sft_4B/calculated_scores/test_result.json", "r", encoding="utf-8") as f:
# with open(r"results\Mind_the_query\calculated_scores_Qwen3_0.6B_4B_csd\test_result.json", "r", encoding="utf-8") as f:
with open(r"results\Cypherbench\calculated_scores_Qwen3_0.6B_distill_fkl_only_query_attn_loss_log_raw_mse\test_result.json", "r", encoding="utf-8") as f:
    result = json.load(f)

metric_scores = calculate_result(result)
print(metric_scores)