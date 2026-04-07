#!/usr/bin/env bash

mkdir -p logs

bash scripts/qwen/fdd/train_0.6B_4B.sh >> log_qwen3_0.6B_4B_fdd.txt 2>&1