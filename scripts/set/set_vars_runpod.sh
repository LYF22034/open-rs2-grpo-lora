#!/bin/bash

export CUDA_LAUNCH_BLOCKING=0
export DS_LOG_LEVEL=error
export TOKENIZERS_PARALLELISM=false
export MKL_THREADING_LAYER=GNU
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

export HOME_DIR="/workspace/Tina"
export PYTHONPATH="/workspace/Tina":$PYTHONPATH
export PYTHONPATH="/workspace/Tina/tina":$PYTHONPATH

export CKPT_DIR="/workspace/ckpts"
export DATA_DIR="/workspace/datasets"
export OUTPUT_DIR="/workspace/outputs"
export LOGGING_DIR="/workspace/logs"
mkdir -p "${CKPT_DIR}" "${DATA_DIR}" "${OUTPUT_DIR}" "${LOGGING_DIR}"

export WANDB_PROJECT="Tina"
export WANDB_DIR="${OUTPUT_DIR}"

export CACHE_DIR="/workspace/.cache"
export HF_HOME="${CACHE_DIR}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRITON_CACHE_DIR="${CACHE_DIR}/triton_cache"
mkdir -p "${CACHE_DIR}" "${HF_HOME}" "${TRITON_CACHE_DIR}"

# NCCL 配置（RunPod 容器需要）
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN
