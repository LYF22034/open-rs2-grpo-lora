#!/bin/bash
echo "START TIME: $(date)"
source "/workspace/Tina/scripts/set/set_vars_runpod.sh"

# 关键 NCCL 设置
export CUDA_LAUNCH_BLOCKING=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export CUDA_VISIBLE_DEVICES=0,1

echo "Testing NCCL 2-GPU communication..."
python -c "
import os, torch, torch.distributed as dist
os.environ['MASTER_ADDR']='127.0.0.1'
os.environ['MASTER_PORT']='29600'
os.environ['RANK']='0'
os.environ['WORLD_SIZE']='1'
dist.init_process_group('nccl', init_method='env://')
print('NCCL 2-GPU test passed!')
dist.destroy_process_group()
"

if [ $? -ne 0 ]; then
    echo "NCCL test failed, aborting."
    exit 1
fi

BASE_MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
PY_SCRIPT="./tina/post_train_hf/grpo.py"
PY_CONFIG="./recipes/${BASE_MODEL_NAME}/grpo/train_model_open_rs2.yaml"

echo "Launching 2-GPU training via torchrun..."

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo \
torchrun --nproc_per_node=2 --master_port=29600 \
    ${PY_SCRIPT} --config ${PY_CONFIG}

echo "END TIME: $(date)"
echo "DONE"
