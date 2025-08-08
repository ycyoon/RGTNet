#!/bin/bash
# Multiple Model Evaluation Script
# Evaluates multiple models and creates comparison reports

set -e

# Configuration
export CUDA_VISIBLE_DEVICES=0
DEVICE="cuda"

# Evaluation configuration
DATA_FILE="data/val_instruction.jsonl"
MAX_SAMPLES=100  # Samples per model
CONFIG_FILE="model_config.json"

echo "=========================================="
echo "Multiple Model LLM Performance Evaluation"
echo "=========================================="
echo "Data File: $DATA_FILE"
echo "Max Samples per Model: $MAX_SAMPLES"
echo "Config File: $CONFIG_FILE"
echo "Device: $DEVICE"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Configuration file not found: $CONFIG_FILE"
    echo "Creating sample configuration file..."
    python evaluate_multiple_models.py --create_sample_config
    echo ""
    echo "📝 Please edit $CONFIG_FILE to specify your models, then run this script again."
    exit 1
fi

# Check if data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ Error: Data file not found: $DATA_FILE"
    exit 1
fi

echo "🚀 Starting multi-model evaluation..."

# Run multi-model evaluation
python evaluate_multiple_models.py \
    --config_file "$CONFIG_FILE" \
    --data_file "$DATA_FILE" \
    --max_samples "$MAX_SAMPLES" \
    --device "$DEVICE"

# Check if evaluation completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Multi-model evaluation completed successfully!"
    echo ""
    echo "📁 Generated Files:"
    echo "------------------"
    ls -la evaluation_results/
    echo ""
    echo "📊 To view comparison results, check the CSV files in evaluation_results/"
else
    echo "❌ Multi-model evaluation failed!"
    exit 1
fi
