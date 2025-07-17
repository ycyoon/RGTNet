#!/bin/bash

# RGTNet DDP 학습 및 벤치마크 평가 스크립트

echo "🚀 Starting RGTNet DDP training with benchmark evaluation..."

# 2개의 GPU를 사용하여 DDP로 main.py 실행
# batch_size 1, gradient_accumulation_steps 8 -> 실질적인 배치 크기 1*2*8=16
# --use_amp : 메모리 절약을 위해 Automatic Mixed Precision 사용
torchrun --nproc_per_node=8 main.py \
    --pretrained_model_name meta-llama/Meta-Llama-3-8B \
    --epochs 1 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --use_amp \
    --lr 5e-5 \
    --save_path "models/llama3_8b_rgtnet.pth" \
    --results_file "results/llama3_8b_rgtnet_results.json" \
    --enable_benchmark \
    --benchmark_freq 1 \
    --benchmark_dir "StructTransformBench/benchmark" \
    --dropout 0.1 \
    --bias_delta 1.0 \
    --tokenizer_name "bert-base-uncased" \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --download_datasets \
    --gradient_checkpointing \
    --max_seq_len 4096

echo "✅ DDP Training completed!"


# # 논문 방식 StructTransform 벤치마크 평가 (HarmBench judge + Refusal judge)
# echo "\n🚦 논문 방식 StructTransform 벤치마크 평가 (HarmBench judge + Refusal judge) 실행..."
# python structtransform_benchmark.py \
#     --model_path "models/${MODEL_NAME}.pth" \
#     --output_path "results/${MODEL_NAME}_structtransform_benchmark.json" \
#     --benchmark_dir "StructTransformBench/benchmark" \
#     --tokenizer_name "bert-base-uncased" \
#     --d_model $D_MODEL \
#     --nhead $NHEAD \
#     --num_layers $NUM_LAYERS \
#     --max_seq_len $MAX_SEQ_LEN \
#     --dropout 0.1 \
#     --bias_delta 1.0 \
#     --batch_size 8 \
#     --device cuda

# echo "🎉 All done! Check the results directory for analysis."