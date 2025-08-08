#!/bin/bash
# Safe LLM Performance Evaluation Script
# Prevents NCCL errors and runs safely on single GPU

set -e

# Configuration
export CUDA_VISIBLE_DEVICES=0
DEVICE="cuda"

# Set environment variables to prevent NCCL issues
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
export CUDA_LAUNCH_BLOCKING=1
export MASTER_ADDR=localhost
export MASTER_PORT=12355

# Model configuration (update these paths as needed)
MODEL_PATH="models"  # Path to your trained model (RGTNet or foundation model)

# Evaluation configuration
DATA_FILE="data/val_instruction.jsonl"
MAX_SAMPLES=50  # Start with fewer samples for safety
MAX_NEW_TOKENS=128
TEMPERATURE=0.7

# Output configuration
OUTPUT_DIR="evaluation_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="${OUTPUT_DIR}/safe_llm_evaluation_${TIMESTAMP}.json"

echo "=========================================="
echo "Safe LLM Performance Evaluation"
echo "=========================================="
echo "Model Path: $MODEL_PATH"
echo "Data File: $DATA_FILE"
echo "Max Samples: $MAX_SAMPLES"
echo "Output File: $OUTPUT_FILE"
echo "Device: $DEVICE"
echo ""
echo "Environment Variables Set:"
echo "- NCCL_DEBUG: $NCCL_DEBUG"
echo "- NCCL_TIMEOUT: $NCCL_TIMEOUT"
echo "- CUDA_LAUNCH_BLOCKING: $CUDA_LAUNCH_BLOCKING"
echo "- MASTER_ADDR: $MASTER_ADDR"
echo "- MASTER_PORT: $MASTER_PORT"
echo ""

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Error: Model directory not found: $MODEL_PATH"
    echo "Please update MODEL_PATH in this script to point to your trained model."
    exit 1
fi

# Check if data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ Error: Data file not found: $DATA_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "🚀 Starting safe evaluation..."

# Run safe evaluation
python evaluate_llm_safe.py \
    --model_path "$MODEL_PATH" \
    --data_file "$DATA_FILE" \
    --output_file "$OUTPUT_FILE" \
    --max_samples "$MAX_SAMPLES" \
    --device "$DEVICE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE"

# Check if evaluation completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Safe evaluation completed successfully!"
    echo "📊 Results saved to: $OUTPUT_FILE"
    echo "📈 Summary saved to: ${OUTPUT_FILE%.json}_summary.csv"
    echo ""
    echo "📋 Quick Results Preview:"
    echo "------------------------"
    if [ -f "${OUTPUT_FILE%.json}_summary.csv" ]; then
        cat "${OUTPUT_FILE%.json}_summary.csv"
    fi
    echo ""
    echo "💡 If this worked well, you can increase MAX_SAMPLES for more comprehensive evaluation."
else
    echo "❌ Safe evaluation failed!"
    echo "💡 Try reducing MAX_SAMPLES or check if the model is compatible."
    exit 1
fi

