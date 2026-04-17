#!/bin/bash
echo "START TIME: $(date)"
source "/workspace/Tina/scripts/set/set_vars_prime.sh"

BASE_MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
PY_SCRIPT="./tina/post_train_hf/grpo.py"
PY_CONFIG="./recipes/${BASE_MODEL_NAME}/grpo/train_model_open_rs2_prime.yaml"
ACCELERATE_CONFIG="./recipes/accelerate_ds_cfgs/single_process.yaml"

echo "=== PRIME + GRPO + LoRA Training ==="
echo "Base model: ${BASE_MODEL_NAME}"
echo "GPU 0: Policy (LoRA) + PRM (full) + Reference (frozen)"
echo "GPU 1: vLLM rollout"
echo "per_device_batch=12 × grad_accum=4 / num_gen=6 = 8 prompts/step (matches paper)"

cd /workspace/Tina

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file "${ACCELERATE_CONFIG}" \
    --main_process_port=0 \
    --num_processes=1 \
    ${PY_SCRIPT} --config ${PY_CONFIG} \
    --vllm_device cuda:1

echo "END TIME: $(date)"
echo "DONE"
