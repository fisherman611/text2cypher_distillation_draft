#!/usr/bin/env bash

mkdir -p logs

bash scripts/qwen/fdd/train_0.6B_4B_sfkl.sh >> log_qwen3_0.6B_4B_fdd_sfkl.txt 2>&1