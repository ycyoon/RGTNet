#!/bin/bash
# Advanced LLM Performance Evaluation Script
# Evaluates trained models with comprehensive metrics and visualizations

set -e

# Configuration
export CUDA_VISIBLE_DEVICES=0
DEVICE="cuda"

# Model configuration (update these paths as needed)
MODEL_PATH="models"  # Path to your trained model (RGTNet or foundation model)

# Evaluation configuration
DATA_FILE="data/val_instruction.jsonl"
MAX_SAMPLES=200  # Limit samples for comprehensive evaluation
MAX_NEW_TOKENS=128
TEMPERATURE=0.7

# Output configuration
OUTPUT_DIR="evaluation_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="${OUTPUT_DIR}/advanced_llm_evaluation_${TIMESTAMP}.json"

echo "=========================================="
echo "Advanced LLM Performance Evaluation"
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

echo "🚀 Starting advanced evaluation..."

# Run advanced evaluation
python evaluate_llm_advanced.py \
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
    echo "✅ Advanced evaluation completed successfully!"
    echo "📊 Results saved to: $OUTPUT_FILE"
    echo "📈 Summary saved to: ${OUTPUT_FILE%.json}_summary.csv"
    echo "📊 Visualizations saved to: ${OUTPUT_FILE%.json}_visualizations/"
    echo ""
    echo "📋 Comprehensive Results Preview:"
    echo "--------------------------------"
    if [ -f "${OUTPUT_FILE%.json}_summary.csv" ]; then
        cat "${OUTPUT_FILE%.json}_summary.csv"
    fi
    echo ""
    echo "📁 Generated Files:"
    echo "------------------"
    ls -la "${OUTPUT_FILE%.json}"*
    if [ -d "${OUTPUT_FILE%.json}_visualizations" ]; then
        echo ""
        echo "📊 Visualization Files:"
        echo "----------------------"
        ls -la "${OUTPUT_FILE%.json}_visualizations/"
    fi
else
    echo "❌ Advanced evaluation failed!"
    exit 1
fi
