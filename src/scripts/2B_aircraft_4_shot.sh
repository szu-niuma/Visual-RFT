#!/bin/bash
set -x

wandb login

export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=8
export NNODES=1
export MASTER_PORT=29578
export CPUS_PER_TASK=32
export QUOTA=reserved

export DATA_PATH=./share_data/aircraft100_4shot
export CKPT_PATH=./share_models/Qwen2-VL-2B-Instruct
export SAVE_PATH=./share_models/Qwen2-VL-2B-Instruct_GRPO_aircraft100_4shot_8gpu

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b_GRPO_aircraft100_4shot.txt"


SRUN_ARGS=${SRUN_ARGS:-""}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

torchrun --nnodes $NNODES --nproc_per_node $GPUS_PER_NODE --node_rank $SLURM_NODEID --master_addr $(scontrol show hostname $SLURM_NODELIST | head -n1) --master_port ${MASTER_PORT} /mnt/petrelfs/liuziyu/R1-Grounding/R1-V/src/open-r1-multimodal/src/open_r1/grpo_classification.py \
    --output_dir ${SAVE_PATH}  \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --deepspeed /local_scripts/zero3.json \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 40 \
    --run_name Qwen2-VL-2B_GRPO_aircraft100_4shot \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 8