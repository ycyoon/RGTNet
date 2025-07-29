#!/bin/bash

# RGTNet DDP 학습 스크립트 (자동 샤딩 및 병합 포함)

echo "🚀 Starting RGTNet DDP training with automatic sharding and merging..."

# 시작 시간 기록
START_TIME=$(date)
echo "📅 Training started at: $START_TIME"

# GPU 메모리 체크
echo "📊 Checking GPU memory..."
nvidia-smi

# 디렉토리 생성
echo "📁 Creating directories..."
mkdir -p models results logs

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=29600
export WORLD_SIZE=8

echo "🔥 Starting distributed training with 8 GPUs..."
echo "📋 Training Configuration:"
echo "   - Model: meta-llama/Llama-3.2-3B-Instruct"
echo "   - Epochs: 3"
echo "   - Batch size per GPU: 1"
echo "   - Gradient accumulation steps: 8"
echo "   - Effective batch size: 1 * 8 * 8 = 64"
echo "   - Learning rate: 5e-5"
echo "   - Mixed precision: Enabled"
echo "   - Auto sharding and merging: Enabled"

# DDP로 main.py 실행
torchrun --nproc_per_node=8 --master_port=$MASTER_PORT main.py \
    --pretrained_model_name "meta-llama/Llama-3.2-3B-Instruct" \
    --epochs 3 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --use_amp \
    --lr 5e-5 \
    --save_path "models/llama3.2_3b_rgtnet.pth" \
    --results_file "results/llama3.2_3b_rgtnet_results.json" \
    --enable_benchmark \
    --benchmark_freq 1 \
    --dropout 0.1 \
    --bias_delta 1.0 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --download_datasets \
    --max_seq_len 8000 \
    --max_iters 8 \
    --train_only

# 종료 시간 기록
END_TIME=$(date)
echo ""
echo "✅ DDP Training completed!"
echo "📅 Training finished at: $END_TIME"
echo ""
echo "📁 Check the following directories for outputs:"
echo "   - Models: models/ (timestamped folders)"
echo "   - Results: results/ (timestamped folders)"
echo "   - Logs: logs/"

# 최신 모델 폴더 찾기
LATEST_MODEL_DIR=$(ls -td models/llama3.2_3b_rgtnet_* 2>/dev/null | head -1)
if [ -n "$LATEST_MODEL_DIR" ]; then
    echo ""
    echo "🎯 Latest model saved in: $LATEST_MODEL_DIR"
    echo "📄 Model file: $LATEST_MODEL_DIR/llama3.2_3b_rgtnet.pth"
fi