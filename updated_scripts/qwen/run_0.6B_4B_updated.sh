#! /bin/bash

set -e

if [ -n "${UPDATED_METHOD}" ]; then
  METHOD="${UPDATED_METHOD}"
  EXTRA_ARGS=("$@")
else
  METHOD="${1:-kd}"
  if [ "$#" -gt 0 ]; then
    shift
  fi
  EXTRA_ARGS=("$@")
fi

GPUS=(${UPDATED_GPUS:-0 1})
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-66$(($RANDOM%90+10))}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH="${BASE_PATH:-.}"
DATA_DIR="${DATA_DIR:-${BASE_PATH}/processed_data/benchmarks/Cypherbench/qwen/}"

CKPT_NAME="${CKPT_NAME:-qwen3-0.6B}"
CKPT="${CKPT:-Qwen/Qwen3-0.6B}"
TEACHER_CKPT_NAME="${TEACHER_CKPT_NAME:-qwen3-4B}"
TEACHER_CKPT="${TEACHER_CKPT:-Qwen/Qwen3-4B-Instruct-2507}"
TEACHER_PEFT_PATH="${TEACHER_PEFT_PATH:-${BASE_PATH}/results/qwen3/sft_4B/e5-bs2-lr1e-05-G8-N2-NN1-lora-32-64-0.1/1065}"

BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-0.0001}"
GRAD_ACC="${GRAD_ACC:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-5}"
MAX_LENGTH="${MAX_LENGTH:-892}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-797}"
SEED="${SEED:-42}"

TYPE="fkl"
KD_RATIO="${KD_RATIO:-0.7}"
SAVE_SUFFIX="${METHOD}"
TEACHER_OPTS=" --teacher-model-path ${TEACHER_CKPT} --teacher-ckpt-name ${TEACHER_CKPT_NAME} --teacher-model-fp16 --teacher-peft-path ${TEACHER_PEFT_PATH}"
LOSS_OPTS=" --use-logit-kd"
METHOD_OPTS=" --student-gen --gen-num-beams 1 --gen-top-p 1.0 --init-threshold 0.0 --loss-eps 0.1 --capacity 1000"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${BASE_PATH}/configs/deepspeed/ds_config_fp16.json}"

case "${METHOD}" in
  csd)
    TYPE="csd"
    SAVE_SUFFIX="csd"
    ;;
  distillm)
    TYPE="adaptive-srkl"
    SAVE_SUFFIX="distillm"
    ;;
  fdd)
    TYPE="srkl"
    KD_RATIO="${KD_RATIO:-0.5}"
    SAVE_SUFFIX="fdd_updated_losses"
    LOSS_OPTS+=" --teacher_layer_mapping 11 23"
    LOSS_OPTS+=" --student_layer_mapping 9 18"
    LOSS_OPTS+=" --use-span-rep-loss --w-span-rep-loss 1.0"
    LOSS_OPTS+=" --use-span-rel-loss --w-span-rel-loss 1.0"
    LOSS_OPTS+=" --use-gen-query-rel-loss --w-gen-query-rel-loss 0.5"
    ;;
  kd)
    TYPE="${KD_TYPE:-fkl}"
    SAVE_SUFFIX="${TYPE}"
    ;;
  sfkl)
    TYPE="sfkl"
    SAVE_SUFFIX="sfkl"
    ;;
  sft)
    TYPE="lm"
    KD_RATIO=""
    SAVE_SUFFIX="sft_0.6B"
    TEACHER_OPTS=""
    LOSS_OPTS=""
    METHOD_OPTS=""
    BATCH_SIZE="${BATCH_SIZE:-4}"
    LR="${LR:-0.00005}"
    GRAD_ACC="${GRAD_ACC:-4}"
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
    MAX_LENGTH="${MAX_LENGTH:-1024}"
    DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${BASE_PATH}/configs/deepspeed/ds_config_bf16.json}"
    ;;
  *)
    echo "Unknown UPDATED_METHOD='${METHOD}'. Use one of: csd, distillm, fdd, kd, sfkl, sft." >&2
    exit 1
    ;;
esac

SAVE_PATH="${SAVE_PATH:-${BASE_PATH}/results/qwen3/updated_${CKPT_NAME}_4B_Cypherbench_${SAVE_SUFFIX}}"

OPTS=""
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+="${TEACHER_OPTS}"
OPTS+=" --model-type qwen"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"

OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 1"
OPTS+=" --dev-num -1"

OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs ${EPOCHS}"
if [ -n "${KD_RATIO}" ]; then
  OPTS+=" --kd-ratio ${KD_RATIO}"
fi

OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length ${MAX_PROMPT_LENGTH}"

OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 20"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed ${SEED}"

OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${DEEPSPEED_CONFIG}"
OPTS+=" --type ${TYPE}"

OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 0.95"
OPTS+=" --temperature 0.5"

OPTS+="${METHOD_OPTS}"
OPTS+="${LOSS_OPTS}"

OPTS+=" --peft lora"
OPTS+=" --peft-lora-r 32"
OPTS+=" --peft-lora-alpha 64"
OPTS+=" --peft-lora-dropout 0.1"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/updated_finetune.py ${OPTS} ${EXTRA_ARGS[*]}"

echo "${CMD}"
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p "${SAVE_PATH}"
CODE_BASE=HF ${CMD}
