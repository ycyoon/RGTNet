#!/bin/bash

echo "🔍 Running evaluation only with Hybrid Llama-RGTNet checkpoint..."

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
echo "   - LoRA: Enabled"
echo "   - Role Adapters: Enabled"
echo "   - GPUs: 1 (single GPU for stable evaluation)"

# Create results directory
mkdir -p results

# Evaluation with single GPU (more stable for eval only)
python main.py \
    --pretrained_model_name meta-llama/Llama-3.2-3B-Instruct \
    --enable_role_adapters \
    --use_lora \
    --download_datasets \
    --max_seq_len 128 \
    --eval_only

echo "✅ Evaluation completed!" 