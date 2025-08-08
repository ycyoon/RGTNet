#!/bin/bash

# Enable strict error handling
set -e

# Set environment variables to avoid external API calls
export OPENAI_API_KEY=""
export LLAMA_API_KEY=""
export OPENROUTER_API_KEY=""
export API_KEY=""
export OPENAI_MAX_CONCURRENT_REQUESTS=4  # 동시 요청 수 제한

# Unset offline mode
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE

echo "🚀 HuggingFace Models Benchmark"
echo "==============================="
echo "🔧 Environment configured for local-only execution"

# Function to handle errors
handle_error() {
    echo "❌ Error occurred in script at line $1"
    echo "⚠️  Leaving vLLM servers running for debugging"
    echo "   To stop servers manually, run: pkill -f vllm"
    exit 1
}

# Set up error handler
trap 'handle_error ${LINENO}' ERR

# Find the latest RGTNet final model
#TARGET_MODEL="/home/ycyoon/work/RGTNet/models/rgtnet_llama-3.2-3b-instruct_20250803_1735/merged_epoch_0"
#TARGET_MODEL="/home/ycyoon/work/RGTNet/models/rgtnet_llama-3.2-1b-instruct_20250807_1044/merged_epoch_0"
TARGET_MODEL="/home/ycyoon/work/RGTNet/models/rgtnet_llama-3.1-8b-instruct_20250807_2116/merged_epoch_0"
BASE_MODEL="llama-3.1-8b"

echo "🎯 Using RGTNet model: $TARGET_MODEL"

# Get script directory (where this script is located)
SCRIPT_DIR="/home/ycyoon/work/RGTNet/StructTransformBench/examples"

# Start servers
echo "📡 Starting vLLM servers for HuggingFace models..."
bash ./start_multi_model_servers.sh

echo ""
echo "⏳ Checking server status and waiting for readiness..."

# Function to check if server is responding
check_server() {
    local port=$1
    local name=$2
    
    # Check if port is listening
    if netstat -tuln 2>/dev/null | grep ":$port " > /dev/null 2>&1; then
        # Try to make a simple API call
        if curl -s -f "http://localhost:$port/v1/models" > /dev/null 2>&1; then
            echo "✅ $name (port $port) is ready"
            return 0
        else
            echo "⚠️  $name (port $port) is listening but not ready"
            return 1
        fi
    else
        echo "❌ $name (port $port) is not listening"
        return 1
    fi
}

# Define servers to check - use separate arrays for bash compatibility
declare -a server_ports=("8003" "8004")
declare -a server_names=("HarmBench-Llama-2-13b" "WildGuard")

# Maximum wait time (20 minutes = 1200 seconds)
max_wait_time=1200
wait_interval=30
total_waited=0

echo "🔍 Monitoring server startup progress..."

while [ $total_waited -lt $max_wait_time ]; do
    all_ready=true
    ready_count=0
    
    echo ""
    echo "⏱️  Time elapsed: ${total_waited}s / ${max_wait_time}s"
    
    # Check each server - disable error exit temporarily
    set +e
    for i in "${!server_ports[@]}"; do
        port="${server_ports[$i]}"
        name="${server_names[$i]}"
        
        if check_server "$port" "$name"; then
            ready_count=$((ready_count + 1))
        else
            all_ready=false
        fi
    done
    set -e
    
    echo "📊 Ready servers: $ready_count/${#server_ports[@]}"
    
    if [ "$all_ready" = true ]; then
        echo ""
        echo "🎉 All servers are ready! Proceeding with benchmark..."
        break
    fi
    
    # Check if timeout reached
    if [ $total_waited -ge $max_wait_time ]; then
        echo ""
        echo "⏰ Timeout reached! Some servers failed to start within $max_wait_time seconds"
        echo "❌ Checking which servers are still failing..."
        
        set +e
        for i in "${!server_ports[@]}"; do
            port="${server_ports[$i]}"
            name="${server_names[$i]}"
            
            if ! check_server "$port" "$name"; then
                echo "💥 Failed server: $name (port $port)"
                if [ -f "server_${port}.log" ]; then
                    echo "📋 Last 10 lines of server_${port}.log:"
                    tail -10 "server_${port}.log" || echo "   Could not read log file"
                fi
            fi
        done
        set -e
        
        echo ""
        echo "🚨 Benchmark cannot proceed without all servers. Exiting..."
        exit 1
    fi
    
    echo "⏳ Waiting ${wait_interval}s before next check..."
    sleep $wait_interval
    total_waited=$((total_waited + wait_interval))
done



echo ""
echo "🎯 Starting benchmark with local HuggingFace models..."
echo "📁 Results will be saved to corresponding result directories"
echo ""

# Track result files
RESULT_FILES=()



echo ""
# Generate jailbreak dataset if it doesn't exist
JAILBREAK_DATASET="jailbreak_dataset_full.json"


echo ""
# Run benchmark with trained RGTNet models (if available)
echo "🔥 Running benchmark with trained RGTNet models..."
if [ -d "$TARGET_MODEL" ]; then
    echo "🎯 Found trained RGTNet model, running full benchmark with all templates..."
    # Use --num_prompts -1 to test all templates instead of default 5
    if python "${SCRIPT_DIR}/multi_model_benchmark.py" --trained-model rgtnet "$TARGET_MODEL" --use-local --num_prompts -1; then
        echo "✅ Trained model benchmark completed successfully"
    else
        echo "⚠️  Trained model benchmark encountered issues"
    fi

    if python "${SCRIPT_DIR}/multi_model_benchmark.py" --models $BASE_MODEL --use-local --num_prompts -1; then
        echo "✅ Trained model benchmark completed successfully"
    else
        echo "⚠️  Trained model benchmark encountered issues"
    fi
    
    # If jailbreak dataset exists, also run dataset-based evaluation
    if [ -f "$JAILBREAK_DATASET" ]; then
        echo ""
        echo "🎯 Running dataset-based jailbreak evaluation..."
        
        # Test different jailbreak methods
        for METHOD in PAIR WildteamAttack Jailbroken; do
            echo "🔄 Testing with $METHOD jailbreak method..."
            if python "${SCRIPT_DIR}/run_PAIR.py" --use-jailbreak-dataset "$JAILBREAK_DATASET" --jailbreak-method "$METHOD" --use-trained-model --trained-model-path "$TARGET_MODEL"; then
                echo "✅ $METHOD dataset-based evaluation completed"

            else
                echo "⚠️  $METHOD dataset-based evaluation encountered issues"
            fi
        done
    fi

    if [ -f "$JAILBREAK_DATASET" ]; then
        echo ""
        echo "🎯 Running dataset-based jailbreak evaluation..."
        
        # Test different jailbreak methods
        for METHOD in PAIR WildteamAttack Jailbroken; do
            echo "🔄 Testing with $METHOD jailbreak method..."
            if python "${SCRIPT_DIR}/run_PAIR.py" --use-jailbreak-dataset "$JAILBREAK_DATASET" --jailbreak-method "$METHOD"  --target-model $BASE_MODEL; then
                echo "✅ $METHOD dataset-based evaluation completed"

            else
                echo "⚠️  $METHOD dataset-based evaluation encountered issues"
            fi
        done
    fi
else
    echo "⚠️  No trained RGTNet models found, skipping trained model benchmark"
fi


echo ""
echo "🎉 Benchmark execution completed!"
echo "📊 Check the results in the generated result directories"
