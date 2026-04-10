#!/usr/bin/env bash

mkdir -p logs

bash scripts/qwen/fdd/train_0.6B_4B_srkl_0.05.sh >> log_qwen3_0.6B_4B_fdd_srkl_0.05.txt 2>&1