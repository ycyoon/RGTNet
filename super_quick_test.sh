#!/bin/bash

# Super Quick Test - Minimal configuration for immediate testing
# This runs the fastest possible test to check if the system works

set -e

echo "🚀 RGTNet Super Quick Test"
echo "========================="

# Minimal configuration for single GPU
export CUDA_VISIBLE_DEVICES=0  # Use only 1 GPU
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1  # For debugging device issues
export OMP_NUM_THREADS=1  # Prevent threading conflicts
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Add current directory to Python path

# Create results directory
mkdir -p super_quick_results

echo "⏱️  Starting 30-second test..."

# Start GPU monitoring in background
nvidia-smi -l 1 > super_quick_results/gpu_monitor.log 2>&1 &
GPU_MONITOR_PID=$!

# Run RGTNet training (30 second test)
echo "🚀 Starting RGTNet training..."
timeout 30 /home/ycyoon/anaconda3/envs/rgtnet/bin/python -u main.py \
    --download_datasets \
    --epochs 1 \
    --batch_size 2 \
    --d_model 128 \
    --nhead 2 \
    --num_layers 1 \
    --save_path super_quick_results/model.pth \
    --results_file super_quick_results/results.json \
    --tokenizer_name bert-base-uncased \
    --train_only \
    2>&1 | tee super_quick_results/test.log

# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null || true

if [ $? -eq 0 ]; then
    echo "✅ Test completed successfully!"
    echo "📁 Check super_quick_results/test.log for details"
    
    # Show basic stats
    echo "📊 Quick Stats:"
    echo "   - GPU Usage: $(grep -c "Using.*GPU" super_quick_results/test.log || echo "0")"
    echo "   - Training batches: $(grep -c "Epoch" super_quick_results/test.log || echo "0")"
    echo "   - Model saved: $([ -f super_quick_results/model.pth ] && echo "Yes" || echo "No")"
    
    # Show file sizes
    if [ -f super_quick_results/model.pth ]; then
        echo "   - Model size: $(ls -lh super_quick_results/model.pth | awk '{print $5}')"
    fi
    
    # Show results files
    echo "   - Results files:"
    ls -la super_quick_results/*.json 2>/dev/null || echo "     No JSON results found"
    
    # Analyze GPU usage
    if [ -f super_quick_results/gpu_monitor.log ]; then
        echo "   - GPU 0 max usage: $(grep -oE "GPU 0.*?%.*?MiB" super_quick_results/gpu_monitor.log | tail -10 | grep -oE "[0-9]+%" | sort -n | tail -1 || echo "N/A")"
    fi
else
    echo "❌ Test failed!"
    echo "📁 Check super_quick_results/test.log for error details"
    exit 1
fi

echo ""
echo "🎉 Super Quick Test completed!"
echo "📁 All results saved in super_quick_results/"