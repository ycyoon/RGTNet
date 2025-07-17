#!/bin/bash
# Test DataParallel setup script

echo "🚀 Testing DataParallel Setup"
echo "============================="

# Check GPU availability
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits

echo -e "\nTesting DataParallel with small batch..."

# Test with minimal settings
python3 main.py \
    --download_datasets \
    --epochs 1 \
    --batch_size 512 \
    --lr 1e-4 \
    --d_model 128 \
    --nhead 4 \
    --num_layers 2 \
    --max_seq_len 128 \
    --train_only \
    --save_path test_dataparallel_model.pth

echo "✅ DataParallel test completed!"
