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
CURRENT_DATE=$(date +"%Y%m%d")
TARGET_MODEL="/home/ycyoon/work/RGTNet/models/rgtnet_final_model_${CURRENT_DATE}"

# If today's model doesn't exist, find the latest one
if [ ! -d "$TARGET_MODEL" ]; then
    TARGET_MODEL=$(find /home/ycyoon/work/RGTNet/models/ -name "rgtnet_final_model_*" -type d | sort -r | head -1)
fi

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
declare -a server_ports=("8002" "8003" "8004" "8006")
declare -a server_names=("Meta-Llama-3.1-70B" "HarmBench-Llama-2-13b" "WildGuard" "Meta-Llama-3-70B")

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

# # First run the multi-model benchmark
# echo "🔥 Running multi-model benchmark..."
# if python "${SCRIPT_DIR}/multi_model_benchmark.py" --models llama-3.2-3b --use-local; then
#     echo "✅ Multi-model benchmark completed successfully"
#     # Find the most recent result file
#     MULTI_RESULT=$(ls -t multi_model_results_*.json 2>/dev/null | head -1)
#     [ -n "$MULTI_RESULT" ] && RESULT_FILES+=("$MULTI_RESULT")
# else
#     echo "⚠️  Multi-model benchmark encountered issues, continuing..."
# fi

echo ""
# Generate jailbreak dataset if it doesn't exist
JAILBREAK_DATASET="jailbreak_dataset_full.json"
if [ ! -f "$JAILBREAK_DATASET" ]; then
    echo "🔥 Generating jailbreak attack dataset..."
    if python "${SCRIPT_DIR}/generate_jailbreak_dataset.py" --output "$JAILBREAK_DATASET" --dataset-size 50 --methods PAIR WildteamAttack Jailbroken; then
        echo "✅ Jailbreak dataset generation completed"
    else
        echo "⚠️  Jailbreak dataset generation failed, continuing without pre-generated dataset..."
    fi
else
    echo "✅ Using existing jailbreak dataset: $JAILBREAK_DATASET"
fi

echo ""
# Run benchmark with trained RGTNet models (if available)
echo "🔥 Running benchmark with trained RGTNet models..."
if [ -d "$TARGET_MODEL" ]; then
    echo "🎯 Found trained RGTNet model, running full benchmark with all templates..."
    # Use --num_prompts -1 to test all templates instead of default 5
    if python "${SCRIPT_DIR}/multi_model_benchmark.py" --trained-model rgtnet "$TARGET_MODEL" --use-local --num_prompts -1; then
        echo "✅ Trained model benchmark completed successfully"
        TRAINED_RESULT=$(ls -t multi_model_results_*.json 2>/dev/null | head -1)
        [ -n "$TRAINED_RESULT" ] && RESULT_FILES+=("$TRAINED_RESULT")
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
            if python "${SCRIPT_DIR}/run_PAIR.py" --use-jailbreak-dataset "$JAILBREAK_DATASET" --jailbreak-method "$METHOD" --use-trained-model --trained-model-path "$TARGET_MODEL" --dataset-size 20; then
                echo "✅ $METHOD dataset-based evaluation completed"
                DATASET_RESULT=$(ls -t *${METHOD}_dataset_result.jsonl 2>/dev/null | head -1)
                [ -n "$DATASET_RESULT" ] && RESULT_FILES+=("$DATASET_RESULT")
            else
                echo "⚠️  $METHOD dataset-based evaluation encountered issues"
            fi
        done
    fi
else
    echo "⚠️  No trained RGTNet models found, skipping trained model benchmark"
fi

# echo ""
# # Run the benchmark with PAIR attack (if available)
# echo "🔥 Running PAIR attack benchmark..."
# if python -c "import easyjailbreak.attacker.PAIR_chao_2023" 2>/dev/null; then
#     # Test with foundation model
#     if python "${SCRIPT_DIR}/run_PAIR.py" --target-model llama-3.2-3b --dataset-size 10; then
#         echo "✅ PAIR attack on foundation model completed successfully"
#         PAIR_RESULT=$(ls -t PAIR_results_*.json 2>/dev/null | head -1)
#         [ -n "$PAIR_RESULT" ] && RESULT_FILES+=("$PAIR_RESULT")
#     else
#         echo "⚠️  PAIR attack on foundation model encountered issues"
#     fi
    
#     # Test with trained model (if available)
#     if [ -f "$TARGET_MODEL" ]; then
#         echo "🎯 Running PAIR attack on trained RGTNet model..."
#         if python "${SCRIPT_DIR}/run_PAIR.py" --use-trained-model --trained-model-path "$TARGET_MODEL" --dataset-size 10; then
#             echo "✅ PAIR attack on trained model completed successfully"
#             TRAINED_PAIR_RESULT=$(ls -t *trained*result*.jsonl 2>/dev/null | head -1)
#             [ -n "$TRAINED_PAIR_RESULT" ] && RESULT_FILES+=("$TRAINED_PAIR_RESULT")
#         else
#             echo "⚠️  PAIR attack on trained model encountered issues"
#         fi
#     fi
# else
#     echo "⚠️  PAIR attack module not available, skipping..."
# fi

# echo ""
# echo "🔥 Running WildteamAttack benchmark..."
# if python -c "import easyjailbreak.attacker" 2>/dev/null; then
#     if python "${SCRIPT_DIR}/run_WildteamAttack.py" --target-model llama-3.2-3b --dataset-size 10 --action generate; then
#         echo "✅ WildteamAttack generate completed successfully"
#     else
#         echo "⚠️  WildteamAttack generate encountered issues"
#     fi
    
#     if python "${SCRIPT_DIR}/run_WildteamAttack.py" --target-model llama-3.2-3b --dataset-size 10 --action attack; then
#         echo "✅ WildteamAttack attack completed successfully"
        
#         # Find the most recent WildteamAttack result files
#         # Look for summary files first
#         WILD_SUMMARY=$(ls -t logs/WildteamAttack_*/WildteamAttack_summary_*.json 2>/dev/null | head -1)
#         if [ -n "$WILD_SUMMARY" ]; then
#             RESULT_FILES+=("$WILD_SUMMARY")
#             echo "📊 Found WildteamAttack summary: $WILD_SUMMARY"
#         fi
        
#         # Look for any result files in the WildteamAttack directory
#         WILD_RESULTS=$(find logs/WildteamAttack_*/ -name "*.json" -type f 2>/dev/null | head -5)
#         for result_file in $WILD_RESULTS; do
#             if [[ "$result_file" != "$WILD_SUMMARY" ]]; then
#                 RESULT_FILES+=("$result_file")
#             fi
#         done
        
#         # Alternative fallback
#         if [ ${#RESULT_FILES[@]} -eq 0 ]; then
#             WILD_RESULT=$(ls -t WildteamAttack_results_*.json 2>/dev/null | head -1)
#             [ -n "$WILD_RESULT" ] && RESULT_FILES+=("$WILD_RESULT")
#         fi
#     else
#         echo "⚠️  WildteamAttack attack encountered issues"
#     fi
# else
#     echo "⚠️  WildteamAttack module not available, skipping..."
# fi

# echo ""
# echo "🔥 Running Jailbroken attack benchmark..."
# if python -c "import easyjailbreak.attacker.Jailbroken_wei_2023" 2>/dev/null; then
#     if python "${SCRIPT_DIR}/run_jailbroken.py" --target-model llama-3.2-3b --dataset-size 10; then
#         echo "✅ Jailbroken attack completed successfully"
#         # Find the most recent jailbroken result file
#         JAIL_RESULT=$(ls -t jailbroken_results_*.json 2>/dev/null | head -1)
#         [ -n "$JAIL_RESULT" ] && RESULT_FILES+=("$JAIL_RESULT")
#     else
#         echo "⚠️  Jailbroken attack encountered issues"
#     fi
# else
#     echo "⚠️  Jailbroken attack module not available, skipping..."
# fi

echo ""
echo "🎉 Benchmark execution completed!"
echo "📊 Check the results in the generated result directories"

# Show recent result files
echo ""
echo "📁 Generated result files:"
if [ ${#RESULT_FILES[@]} -gt 0 ]; then
    for file in "${RESULT_FILES[@]}"; do
        echo "  - $file"
    done
else
    echo "  No result files found"
fi

# Create summary if results exist
if [ ${#RESULT_FILES[@]} -gt 0 ]; then
    echo ""
    echo "📊 Creating benchmark summary..."
    
    # Run the Python analysis script with all result files
    python "${SCRIPT_DIR}/analyze_benchmark_results.py" "${RESULT_FILES[@]}"
else
    echo ""
    echo "📊 No result files found, running analysis on all available files..."
    
    # Run analysis on entire logs directory to capture all results
    python "${SCRIPT_DIR}/analyze_benchmark_results.py" --dir logs/ --limit 100
fi

# Show all recent result files in directory
echo ""
echo "📁 All recent result files in directory:"
ls -lt *.json 2>/dev/null | head -10 || echo "  No JSON files found"

echo ""
echo "💡 vLLM servers are still running for future use"
echo "   To stop servers manually, run: pkill -f vllm"
echo "   Or use individual server management commands"

echo "🏁 Script execution finished - servers remain active"
