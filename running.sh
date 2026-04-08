#!/usr/bin/env bash

mkdir -p logs

bash scripts/qwen/csd/train_0.6B_4B.sh >> log_qwen3_0.6B_4B_csd.txt 2>&1