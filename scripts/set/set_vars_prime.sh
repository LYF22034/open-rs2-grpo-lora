#!/bin/bash

# Only training process sees GPU 0; GPU 1 reserved for vLLM
export CUDA_VISIBLE_DEVICES=0,1

export TOKENIZERS_PARALLELISM=false
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1

# Paths
export HOME_DIR="/workspace/Tina"
export PYTHONPATH="/workspace/Tina:/workspace/Tina/tina:$PYTHONPATH"

export CKPT_DIR="/workspace/ckpts"
export DATA_DIR="/workspace/datasets"
export OUTPUT_DIR="/workspace/outputs"
export LOGGING_DIR="/workspace/logs"

# WandB
export WANDB_PROJECT="Tina_PRIME"
export WANDB_DIR="${OUTPUT_DIR}"

# HF cache
export CACHE_DIR="/workspace/.cache"
export HF_HOME="${CACHE_DIR}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRITON_CACHE_DIR="${CACHE_DIR}/triton_cache"
