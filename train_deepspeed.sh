#!/bin/bash

# DeepSpeed training script for RGTNet with auto-merge and HuggingFace config generation
set -e



# Configuration - 원래 설정으로 복원
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7



NUM_GPUS=8
BATCH_SIZE=1
GRADIENT_ACCUMULATION=1

# Model configuration
MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
EPOCHS=10
LEARNING_RATE=5e-5
MAX_SEQ_LEN=8192

# Data paths
TRAIN_FILE="data/train_instruction.jsonl"
VAL_FILE="data/val_instruction.jsonl"

# Get current date for directory naming
CURRENT_DATE=$(date +"%Y%m%d")

# Read batch size from ds_config.json
if command -v jq &> /dev/null; then
    ACTUAL_BATCH_SIZE=$(jq -r '.train_batch_size' ds_config.json)
    ACTUAL_GRAD_ACCUM=$(jq -r '.gradient_accumulation_steps' ds_config.json)
else
    ACTUAL_BATCH_SIZE="800"
    ACTUAL_GRAD_ACCUM="1"
fi

# Output paths - Use date-based naming
SAVE_PATH="models/rgtnet_deepspeed_${CURRENT_DATE}"

echo " Starting DeepSpeed training for RGTNet with auto-merge and HuggingFace config generation"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "GPUs: $NUM_GPUS"
echo "Batch size: $ACTUAL_BATCH_SIZE (from ds_config.json)"
echo "Gradient accumulation: $ACTUAL_GRAD_ACCUM (from ds_config.json)"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Max sequence length: $MAX_SEQ_LEN"
echo "Training date: $CURRENT_DATE"
echo "Checkpoint directory: $SAVE_PATH"
echo "Auto-merged model will be created in: ${SAVE_PATH}_unified/"
echo ""

# Create directories
mkdir -p "$SAVE_PATH"
mkdir -p logs

echo " Starting DeepSpeed training with auto-merge and HuggingFace config generation..."

# Run DeepSpeed training with hybrid model
deepspeed --num_gpus=$NUM_GPUS main.py \
    --pretrained_model_name $MODEL_NAME \
    --enable_role_adapters \
    --use_lora \
    --download_datasets \
    --max_seq_len $MAX_SEQ_LEN \
    --save_path $SAVE_PATH \
    --deepspeed \
    --deepspeed_config ds_config.json

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "📁 Auto-merged model with HuggingFace configs will be available in: ${SAVE_PATH}_unified/"
    echo "🎉 Training completed! Check the unified directory for the final model."
else
    echo "❌ Training failed!"
    exit 1
fi 