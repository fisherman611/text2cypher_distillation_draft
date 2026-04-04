#!/usr/bin/env bash

mkdir -p logs

bash scripts/qwen/sft/sft_qwen3_0.6B.sh >> logs/qwen3_0.6B_log.txt 2>&1 &
bash scripts/qwen/sft/sft_qwen3_4B.sh >> logs/qwen3_4B_log.txt 2>&1 &

wait