#!/bin/bash
# LLM Performance Evaluation Script
# Evaluates trained models using multiple metrics on golden standard dataset

set -e

# Configuration
export CUDA_VISIBLE_DEVICES=0
DEVICE="cuda"

# Model configuration (update these paths as needed)
MODEL_PATH="models"  # Path to your trained model (RGTNet or foundation model)

# Evaluation configuration
DATA_FILE="data/val_instruction.jsonl"
MAX_SAMPLES=100  # Limit samples for quick evaluation, set to null for full evaluation
MAX_NEW_TOKENS=128
TEMPERATURE=0.7

# Output configuration
OUTPUT_DIR="evaluation_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="${OUTPUT_DIR}/llm_evaluation_${TIMESTAMP}.json"

echo "=========================================="
echo "LLM Performance Evaluation"
echo "=========================================="
echo "Model Path: $MODEL_PATH"
echo "Data File: $DATA_FILE"
echo "Max Samples: $MAX_SAMPLES"
echo "Output File: $OUTPUT_FILE"
echo "Device: $DEVICE"
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

echo "🚀 Starting evaluation..."

# Run evaluation
python evaluate_llm_performance.py \
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
    echo "✅ Evaluation completed successfully!"
    echo "📊 Results saved to: $OUTPUT_FILE"
    echo "📈 Summary saved to: ${OUTPUT_FILE%.json}_summary.csv"
    echo ""
    echo "📋 Quick Results Preview:"
    echo "------------------------"
    if [ -f "${OUTPUT_FILE%.json}_summary.csv" ]; then
        cat "${OUTPUT_FILE%.json}_summary.csv"
    fi
else
    echo "❌ Evaluation failed!"
    exit 1
fi
