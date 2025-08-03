#!/bin/bash
# Evaluation script for RGTNet
# 
# LoRA Usage:
#   To enable LoRA: set USE_LORA=true
#   Default: Evaluation without LoRA

echo "🔍 Running evaluation only with Hybrid Llama-RGTNet checkpoint..."

# LoRA configuration (set to false by default)
USE_LORA=false

# 환경 변수 설정 (single GPU for evaluation)
export CUDA_VISIBLE_DEVICES=0

# 최신 체크포인트 경로 찾기
CURRENT_DATE=$(date +"%Y%m%d")
FINAL_MODEL_DIR="models/rgtnet_final_model_${CURRENT_DATE}"

# 최신 final model 찾기
if [ ! -d "$FINAL_MODEL_DIR" ]; then
    FINAL_MODEL_DIR=$(find models/ -name "rgtnet_final_model_*" -type d | sort -r | head -1)
fi

if [ -z "$FINAL_MODEL_DIR" ] || [ ! -d "$FINAL_MODEL_DIR" ]; then
    echo "❌ No final model directory found!"
    echo "Available models:"
    ls -la models/ | grep rgtnet
    exit 1
fi

echo "📋 Evaluation Configuration:"
echo "   - Model Directory: $FINAL_MODEL_DIR"
echo "   - Base Model: meta-llama/Llama-3.2-3B-Instruct"
echo "   - Architecture: Hybrid Llama-RGTNet"
echo "   - LoRA: $USE_LORA"
echo "   - Role Adapters: Enabled"
echo "   - GPUs: 1 (single GPU for stable evaluation)"

# Create results directory
mkdir -p results

# Build LoRA options based on configuration
LORA_OPTIONS=""
if [ "$USE_LORA" = "true" ]; then
    LORA_OPTIONS="--use_lora"
fi

# Evaluation with single GPU (more stable for eval only)
python main.py \
    --pretrained_model_name meta-llama/Llama-3.2-3B-Instruct \
    --enable_role_adapters \
    $LORA_OPTIONS \
    --download_datasets \
    --max_seq_len 128 \
    --eval_only

echo "✅ Evaluation completed!" 