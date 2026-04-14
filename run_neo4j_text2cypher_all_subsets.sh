#!/usr/bin/env bash
set -euo pipefail

INPUT="${1:-results/Neo4j_Text2Cypher/full_cyphers_result_Qwen3-0.6B.json}"
OUTPUT_DIR="${2:-results/Mind_the_query/calculated_scores_Qwen3_0.6B}"

SUBSETS=(
  bluesky
  buzzoverflow
  companies
  neoflix
  fincen
  gameofthrones
  grandstack
  movies
  network
  northwind
  offshoreleaks
  recommendations
  stackoverflow2
  twitch
  twitter
)

for subset in "${SUBSETS[@]}"; do
  echo "========================================"
  echo "Evaluating subset: ${subset}"
  echo "========================================"

  python src/calculate_scores_neo4j_text2cypher.py \
    --input "${INPUT}" \
    --output_dir "${OUTPUT_DIR}" \
    --subset "${subset}"
done

echo "Done. Results saved to: ${OUTPUT_DIR}"
