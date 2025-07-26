#!/bin/bash

# Enhanced H100 FSDP Training Script with NCCL Optimizations
# This script is optimized for H100 GPUs with enhanced NCCL settings

# Exit on any error
set -e

# Configuration
NODES=1
GPUS_PER_NODE=8
WORLD_SIZE=$((NODES * GPUS_PER_NODE))

echo "=== H100 FSDP Training Setup ==="
echo "Nodes: $NODES"
echo "GPUs per node: $GPUS_PER_NODE" 
echo "World size: $WORLD_SIZE"
echo "================================"

# Enhanced NCCL Environment Variables for H100
export NCCL_SOCKET_IFNAME=lo,eth0,ib0
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_TREE_THRESHOLD=0
export NCCL_ALGO=Tree
export NCCL_PROTO=Simple
export NCCL_MIN_NCHANNELS=4
export NCCL_MAX_NCHANNELS=16
export NCCL_BUFFSIZE=8388608
export NCCL_NTHREADS=64
export NCCL_RINGS=8
export NCCL_CHECKS_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600
export NCCL_DEBUG=INFO  # For debugging, set to WARN for production

# CUDA and PyTorch optimizations
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TOKENIZERS_PARALLELISM=false

# Memory optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,roundup_power2_divisions:16

# OMP settings for better CPU performance
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Print environment for debugging
echo "=== Environment Variables ==="
echo "NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
echo "NCCL_TIMEOUT: $NCCL_TIMEOUT"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "============================="

# Check GPU availability
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
echo "=================="

# Training parameters
BATCH_SIZE=16  # Total batch size across all GPUs
LEARNING_RATE=1e-4
MAX_EPOCHS=3
MAX_SEQ_LEN=512
GRADIENT_ACCUMULATION_STEPS=4

echo "=== Training Parameters ==="
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"  
echo "Max epochs: $MAX_EPOCHS"
echo "Max sequence length: $MAX_SEQ_LEN"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "=========================="

# Create logs directory
mkdir -p fsdp_logs
LOG_FILE="fsdp_logs/training_$(date +%Y%m%d_%H%M%S).log"

echo "=== Starting FSDP Training ==="
echo "Log file: $LOG_FILE"
echo "=============================="

# Run training with torchrun
torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NODES \
    --node_rank=0 \
    --master_addr="localhost" \
    --master_port=12355 \
    main.py \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --max_epochs $MAX_EPOCHS \
        --max_seq_len $MAX_SEQ_LEN \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --save_every 1000 \
        --eval_every 500 \
        --use_amp \
        --train_file data/train_instruction.jsonl \
        --val_file data/val_instruction.jsonl \
        --output_dir ./fsdp_outputs \
        --tokenizer_name microsoft/DialoGPT-medium \
        --warmup_steps 100 \
        --weight_decay 0.01 \
        --max_grad_norm 1.0 \
        --seed 42 2>&1 | tee $LOG_FILE

echo "=== Training Complete ==="
echo "Check log: $LOG_FILE"
echo "========================"
