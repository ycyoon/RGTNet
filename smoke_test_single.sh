#!/bin/bash
# Single GPU Smoke Test - No DeepSpeed, No Distributed Training
set -e

echo "🧪 RGTNet Single GPU Smoke Test"
echo "==============================="
echo "Purpose: Quick validation without distributed complexity"
echo "Duration: ~1-2 minutes"
echo ""

# Single GPU configuration
export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1
BATCH_SIZE=1

# Model configuration
MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
EPOCHS=1
LEARNING_RATE=5e-5
MAX_SEQ_LEN=512
MAX_ITERS=3

# LoRA configuration
USE_LORA=false
LORA_ONLY=false

# Smoke test data paths
TRAIN_FILE="data/smoke_train.jsonl"
VAL_FILE="data/smoke_val.jsonl"

# Output paths
CURRENT_DATE=$(date +"%Y%m%d")
CURRENT_TIME=$(date +"%H%M")
SAVE_PATH="models/smoke_single_${CURRENT_DATE}_${CURRENT_TIME}"

echo "Configuration:"
echo "-------------"
echo "Model: $MODEL_NAME"
echo "GPUs: $NUM_GPUS"
echo "Max iterations: $MAX_ITERS"
echo "Learning rate: $LEARNING_RATE"
echo "Max sequence length: $MAX_SEQ_LEN"
echo "Save path: $SAVE_PATH"
echo ""

# Create directories
mkdir -p "$SAVE_PATH"
mkdir -p logs
mkdir -p data

# Create test data if needed
if [ ! -f "$TRAIN_FILE" ] || [ ! -f "$VAL_FILE" ]; then
    echo "Creating smoke test data..."
    
    cat > data/smoke_train.jsonl << 'DATAEOF'
{"instruction": "What is 2 + 2?", "response": "2 + 2 equals 4."}
{"instruction": "What is the capital of France?", "response": "The capital of France is Paris."}
{"instruction": "Explain what is AI?", "response": "AI stands for Artificial Intelligence."}
{"instruction": "Name three colors", "response": "Three colors are red, blue, and green."}
DATAEOF

    cat > data/smoke_val.jsonl << 'DATAEOF'
{"instruction": "What is 5 + 3?", "response": "5 + 3 equals 8."}
{"instruction": "What is the capital of Japan?", "response": "The capital of Japan is Tokyo."}
DATAEOF
    
    echo "✅ Smoke test data created"
fi

echo "🚀 Starting single GPU smoke test..."

# Build LoRA options
LORA_OPTIONS=""
if [ "$USE_LORA" = "true" ]; then
    LORA_OPTIONS="$LORA_OPTIONS --use_lora"
    if [ "$LORA_ONLY" = "true" ]; then
        LORA_OPTIONS="$LORA_OPTIONS --lora_only"
    fi
fi

# Run without DeepSpeed - single GPU only
python main.py \
    --pretrained_model_name $MODEL_NAME \
    --enable_role_adapters \
    $LORA_OPTIONS \
    --train_file $TRAIN_FILE \
    --val_file $VAL_FILE \
    --max_seq_len $MAX_SEQ_LEN \
    --save_path $SAVE_PATH \
    --epochs $EPOCHS \
    --max_iters $MAX_ITERS \
    --train_only

# Check results
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SINGLE GPU SMOKE TEST PASSED!"
    echo "================================"
    echo "✅ Model loading: Success"
    echo "✅ Data loading: Success"  
    echo "✅ Training pipeline: Success"
    echo "✅ Single GPU training: Success"
    echo ""
    echo "📁 Test output saved in: ${SAVE_PATH}/"
    echo ""
    echo "🎯 Core components work without distributed complexity!"
    echo "💡 You can now debug distributed issues separately."
else
    echo ""
    echo "❌ SINGLE GPU SMOKE TEST FAILED!"
    echo "================================"
    echo "❌ Basic pipeline has issues"
    echo "🔍 Check the logs above for error details"
    exit 1
fi