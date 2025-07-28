#!/bin/bash

# RGTNet DDP 학습 및 벤치마크 평가 스크립트

echo "🚀 Starting RGTNet DDP training with benchmark evaluation..."

# GPU 메모리 체크
echo "📊 Checking GPU memory..."
nvidia-smi

# 디렉토리 생성
mkdir -p models results

# 7개의 GPU를 사용하여 DDP로 main.py 실행 (GPU 0은 vLLM 서버용으로 제외)
# batch_size 1, gradient_accumulation_steps 8 -> 실질적인 배치 크기 1*7*8=56
# --use_amp : 메모리 절약을 위해 Automatic Mixed Precision 사용
echo "🔥 Starting training with 8 GPUs..."
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=7 --master_port=29600 main.py \
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
    --max_seq_len 2048 \
#    --resume_from_checkpoint "models/llama3.2_3b_rgtnet.pth" \
#    --max_iters 100 \
#    --benchmark_dir "StructTransformBench/benchmark" \

echo "✅ DDP Training completed!"
