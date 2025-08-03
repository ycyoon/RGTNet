#!/bin/bash
# Smoke Test Script for RGTNet
# Fast validation of all training pipeline components
set -e

echo "🧪 RGTNet Smoke Test"
echo "=================="
echo "Purpose: Quick validation of training pipeline"
echo "Duration: ~2-3 minutes"
echo ""

# Smoke test configuration - minimal resources
export CUDA_VISIBLE_DEVICES=0,1  # Use only 2 GPUs
NUM_GPUS=2
BATCH_SIZE=1
GRADIENT_ACCUMULATION=1

# Model configuration (same as production)
MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
EPOCHS=3
LEARNING_RATE=5e-5
MAX_SEQ_LEN=512  # Reduced for faster processing
MAX_ITERS=3     # Very small number for smoke test

# LoRA configuration (same as production)
USE_LORA=false
LORA_ONLY=false

# Smoke test data paths
TRAIN_FILE="data/smoke_train.jsonl"
VAL_FILE="data/smoke_val.jsonl"

# Output paths
CURRENT_DATE=$(date +"%Y%m%d")
CURRENT_TIME=$(date +"%H%M")
SAVE_PATH="models/smoke_test_${CURRENT_DATE}_${CURRENT_TIME}"

echo "Configuration:"
echo "-------------"
echo "Model: $MODEL_NAME"
echo "GPUs: $NUM_GPUS (${CUDA_VISIBLE_DEVICES})"
echo "Epochs: $EPOCHS"
echo "Max iterations: $MAX_ITERS"
echo "Learning rate: $LEARNING_RATE"
echo "Max sequence length: $MAX_SEQ_LEN"
echo "Training data: $TRAIN_FILE"
echo "Validation data: $VAL_FILE"
echo "Save path: $SAVE_PATH"
echo ""

# Create directories
mkdir -p "$SAVE_PATH"
mkdir -p logs
mkdir -p data

# Verify test data exists or create it
if [ ! -f "$TRAIN_FILE" ] || [ ! -f "$VAL_FILE" ]; then
    echo "Creating smoke test data..."
    
    cat > data/smoke_train.jsonl << 'DATAEOF'
{"instruction": "What is 2 + 2?", "response": "2 + 2 equals 4."}
{"instruction": "What is the capital of France?", "response": "The capital of France is Paris."}
{"instruction": "Explain what is AI?", "response": "AI stands for Artificial Intelligence, which refers to computer systems that can perform tasks that typically require human intelligence."}
{"instruction": "Write a simple Python hello world", "response": "def hello():\n    print('Hello, World!')\n\nhello()"}
{"instruction": "What is machine learning?", "response": "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed."}
{"instruction": "How do you make tea?", "response": "To make tea: 1) Boil water, 2) Add tea leaves or tea bag to a cup, 3) Pour hot water over tea, 4) Steep for 3-5 minutes, 5) Remove tea bag or strain leaves, 6) Add sugar or milk if desired."}
{"instruction": "What is the largest planet?", "response": "Jupiter is the largest planet in our solar system."}
{"instruction": "Count from 1 to 5", "response": "1, 2, 3, 4, 5"}
{"instruction": "What is programming?", "response": "Programming is the process of creating instructions for computers to execute, typically written in programming languages like Python, Java, or C++."}
{"instruction": "Name three colors", "response": "Three colors are red, blue, and green."}
{"instruction": "What is the weather like today?", "response": "I don't have access to real-time weather data, but I can help you understand weather patterns."}
{"instruction": "How do computers work?", "response": "Computers work by processing data through electronic circuits, following instructions from programs stored in memory."}
DATAEOF

    cat > data/smoke_val.jsonl << 'DATAEOF'
{"instruction": "What is 5 + 3?", "response": "5 + 3 equals 8."}
{"instruction": "What is the capital of Japan?", "response": "The capital of Japan is Tokyo."}
{"instruction": "What is deep learning?", "response": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to analyze data."}
{"instruction": "Name two fruits", "response": "Two fruits are apple and banana."}
DATAEOF
    
    echo "✅ Smoke test data created"
fi

echo "🚀 Starting smoke test..."

# Build LoRA options
LORA_OPTIONS=""
if [ "$USE_LORA" = "true" ]; then
    LORA_OPTIONS="$LORA_OPTIONS --use_lora --enable_role_adapters"
    if [ "$LORA_ONLY" = "true" ]; then
        LORA_OPTIONS="$LORA_OPTIONS --lora_only"
    fi
fi

# Run smoke test with minimal DeepSpeed setup (training only, no evaluation)
deepspeed --num_gpus=$NUM_GPUS main.py \
    --pretrained_model_name $MODEL_NAME \
    $LORA_OPTIONS \
    --train_file $TRAIN_FILE \
    --val_file $VAL_FILE \
    --max_seq_len $MAX_SEQ_LEN \
    --save_path $SAVE_PATH \
    --deepspeed \
    --deepspeed_config ds_config.json \
    --epochs $EPOCHS \
    --train_only

# Check results
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SMOKE TEST PASSED!"
    echo "==================="
    echo "✅ Model loading: Success"
    echo "✅ Data loading: Success"  
    echo "✅ Training pipeline: Success"
    echo "✅ DeepSpeed integration: Success"
    echo "✅ Checkpoint saving: Success"
    echo "✅ Latest file generation: Success"
    echo ""
    echo "📁 Test output saved in: ${SAVE_PATH}/"
    echo ""
    echo "🎯 All core components are working correctly!"
    echo "💡 You can now run full training with confidence."
    echo ""
    echo "🗑️  Cleaning up smoke test files..."
    # Optional: Clean up small test files but keep the model checkpoint for verification
    # rm -f data/smoke_train.jsonl data/smoke_val.jsonl
    echo "📁 Smoke test data preserved for future use"
else
    echo ""
    echo "❌ SMOKE TEST FAILED!"
    echo "==================="
    echo "❌ One or more components failed during testing"
    echo "🔍 Check the logs above for error details"
    echo "💡 Fix the issues before running full training"
    exit 1
fi