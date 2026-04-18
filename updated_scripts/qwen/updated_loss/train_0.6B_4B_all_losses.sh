#! /bin/bash

GPUS=(0 1)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

MASTER_ADDR=localhost
MASTER_PORT=66$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH=.
CKPT_NAME="qwen3-0.6B"
CKPT="Qwen/Qwen3-0.6B"
TEACHER_CKPT_NAME="qwen3-4B"
TEACHER_CKPT="Qwen/Qwen3-4B-Instruct-2507"
TEACHER_PEFT_PATH="${BASE_PATH}/results/qwen3/sft_4B/e5-bs2-lr1e-05-G8-N2-NN1-lora-32-64-0.1/1065"
DATA_DIR="${BASE_PATH}/processed_data/benchmarks/Cypherbench/qwen/"

BATCH_SIZE=2
LR=0.0001
GRAD_ACC=8
EVAL_BATCH_SIZE=8
EPOCHS=5
MAX_LENGTH=892
MAX_PROMPT_LENGTH=797
SEED=42

SAVE_PATH="${BASE_PATH}/results/qwen3/updated_0.6B_4B_Cypherbench_all_losses"

OPTS=""
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --teacher-peft-path ${TEACHER_PEFT_PATH}"
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
OPTS+=" --kd-ratio 0.5"

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
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_fp16.json"
OPTS+=" --type srkl"

OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 0.95"
OPTS+=" --temperature 0.5"

OPTS+=" --student-gen"
OPTS+=" --gen-num-beams 1"
OPTS+=" --gen-top-p 1.0"
OPTS+=" --init-threshold 0.0"
OPTS+=" --loss-eps 0.1"
OPTS+=" --capacity 1000"

OPTS+=" --peft lora"
OPTS+=" --peft-lora-r 32"
OPTS+=" --peft-lora-alpha 64"
OPTS+=" --peft-lora-dropout 0.1"

OPTS+=" --teacher_layer_mapping 11 23"
OPTS+=" --student_layer_mapping 9 18"

OPTS+=" --use-logit-kd"
OPTS+=" --w-logit-kd 1.0"
OPTS+=" --use-query-attention-loss"
OPTS+=" --use-cypher-attention-loss"
OPTS+=" --attention-loss-type kl"
OPTS+=" --attention-head-reduction mean"
OPTS+=" --w-attention-loss 0.5"
OPTS+=" --w-query-attention-loss 1.0"
OPTS+=" --w-cypher-attention-loss 1.0"
OPTS+=" --use-span-rep-loss"
OPTS+=" --w-span-rep-loss 0.5"
OPTS+=" --use-span-rel-loss"
OPTS+=" --w-span-rel-loss 1.0"
OPTS+=" --use-gen-query-rel-loss"
OPTS+=" --w-gen-query-rel-loss 0.5"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/updated_finetune.py ${OPTS} $@"

echo "${CMD}"
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p "${SAVE_PATH}"
CODE_BASE=HF ${CMD}
