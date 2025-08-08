#!/bin/bash
# DeepSpeed training script for RGTNet
# 
# LoRA Usage:
#   To enable LoRA: set USE_LORA=true
#   To save only LoRA adapters: set USE_LORA=true and LORA_ONLY=true
#   Default: Full model training without LoRA
set -e
# Configuration - 원래 설정으로 복원
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8
BATCH_SIZE=1
GRADIENT_ACCUMULATION=1

# Model configuration
#MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
EPOCHS=1
LEARNING_RATE=5e-5
MAX_SEQ_LEN=2048

# LoRA configuration (set to false by default)
USE_LORA=false
LORA_ONLY=false

# Data paths
TRAIN_FILE="data/train_instruction.jsonl"
VAL_FILE="data/val_instruction.jsonl"

# Get current date for directory naming
CURRENT_DATE=$(date +"%Y%m%d")

# Extract model name for folder naming (e.g., "llama-3.2-3b-instruct" from "meta-llama/Llama-3.2-3B-Instruct")
MODEL_SHORT_NAME=$(echo "$MODEL_NAME" | cut -d'/' -f2 | tr '[:upper:]' '[:lower:]')

# Output paths - Let create_timestamped_save_path handle the full naming
SAVE_PATH="models"

echo " Starting DeepSpeed training for RGTNet"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "GPUs: $NUM_GPUS"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Max sequence length: $MAX_SEQ_LEN"
echo "Use LoRA: $USE_LORA"
echo "LoRA only saving: $LORA_ONLY"
echo "Training date: $CURRENT_DATE"
echo "Checkpoint directory: $SAVE_PATH"
echo ""

# Create directories
mkdir -p "$SAVE_PATH"
mkdir -p logs

echo " Starting DeepSpeed training..."

# Build LoRA options based on configuration
LORA_OPTIONS=""
if [ "$USE_LORA" = "true" ]; then
    LORA_OPTIONS="$LORA_OPTIONS --use_lora --enable_role_adapters"
    if [ "$LORA_ONLY" = "true" ]; then
        LORA_OPTIONS="$LORA_OPTIONS --lora_only"
    fi
fi

# Run DeepSpeed training
deepspeed --num_gpus=$NUM_GPUS main.py \
    --pretrained_model_name $MODEL_NAME \
    $LORA_OPTIONS \
    --download_datasets \
    --max_seq_len $MAX_SEQ_LEN \
    --save_path $SAVE_PATH \
    --deepspeed \
    --deepspeed_config ds_config.json \
    --epochs $EPOCHS

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
    if [ "$USE_LORA" = "true" ] && [ "$LORA_ONLY" = "true" ]; then
        echo "📁 LoRA adapters saved in: ${SAVE_PATH}/lora_adapters_epoch_*/"
        echo "🎯 LoRA-only training completed! Ready for inference."
        echo "💡 Use the LoRA adapters with base model for inference."
    else
        echo "📁 Full model saved in: ${SAVE_PATH}/"
        echo "🎯 Full model training completed! Ready for inference."
    fi
else
    echo "❌ Training failed!"
    exit 1
fi 
