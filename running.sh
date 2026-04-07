#!/usr/bin/env bash

mkdir -p logs

bash scripts/qwen/kd/train_0.6B_4B_rkl.sh >> logs/kd_rkl_Cypherbench.txt 2>&1