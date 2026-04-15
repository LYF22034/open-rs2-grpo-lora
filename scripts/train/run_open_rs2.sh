#!/bin/bash
echo "START TIME: $(date)"
source "/workspace/Tina/scripts/set/set_vars_runpod.sh"

export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES=0,1

BASE_MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
PY_SCRIPT="./tina/post_train_hf/grpo.py"
PY_CONFIG="./recipes/${BASE_MODEL_NAME}/grpo/train_model_open_rs2.yaml"
ACCELERATE_DS_CONFIG="./recipes/accelerate_ds_cfgs/single_gpu.yaml"

echo "Running: open_rs2 on ${BASE_MODEL_NAME} (GPU0=train, GPU1=vLLM)"

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file "${ACCELERATE_DS_CONFIG}" \
    --main_process_port=0 \
    --num_processes=1 ${PY_SCRIPT} --config ${PY_CONFIG} --vllm_device cuda:1

echo "END TIME: $(date)"
echo "DONE"
