#! /bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_NAME=qwen3-4B \
CKPT=Qwen/Qwen3-4B-Instruct-2507 \
BATCH_SIZE="${BATCH_SIZE:-2}" \
LR="${LR:-0.00001}" \
GRAD_ACC="${GRAD_ACC:-8}" \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}" \
MAX_LENGTH="${MAX_LENGTH:-892}" \
SAVE_PATH="${SAVE_PATH:-./results/qwen3/updated_sft_4B}" \
UPDATED_METHOD=sft \
bash "${SCRIPT_DIR}/../run_0.6B_4B_updated.sh" "$@"
