#!/bin/bash

# Multi-model vLLM server startup script
# GPU 0,3 are occupied by gmk_multiseg_strategies, using clean GPUs only

echo "🚀 Starting multi-model vLLM servers..."
echo "⚠️  GPU 0,3 occupied by gmk_multiseg_strategies, using clean GPUs 1,2,4,5,6,7"

# Allow overriding max_model_len and set memory configuration
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Function to start a server in background
start_server() {
    local model=$1
    local port=$2
    local name=$3
    local gpus=$4
    local tp_size=$5
    local max_len=${6:-2048} # Default max_model_len
    local mem_util=${7:-0.95}  # Default gpu_memory_utilization

    echo "Starting $name on port $port using GPUs: $gpus (TP: $tp_size, MaxLen: $max_len, MemUtil: $mem_util)"
    CUDA_VISIBLE_DEVICES=$gpus nohup /home/ycyoon/anaconda3/envs/rgtnet/bin/python -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        --port $port \
        --host 0.0.0.0 \
        --trust-remote-code \
        --tensor-parallel-size $tp_size \
        --gpu-memory-utilization $mem_util \
        --max-model-len $max_len \
        --block-size 16 \
        --enforce-eager \
        --disable-custom-all-reduce > "server_${port}.log" 2>&1 &
    
    local pid=$!
    echo "Server $name (PID: $pid) starting on port $port..."
}

# GPU Allocation Plan (avoiding occupied GPU 0,3)
# GPU 1,2: DeepSeek-R1 (2 H100 GPUs)
# GPU 4: HarmBench (1 GPU)  
# GPU 6,7: Meta-Llama-3-70B (2 GPUs)

# Start DeepSeek-R1-Distill for PAIR attack model (Fix: Use 2 GPUs, reduce max_len)
start_server "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" 8002 "DeepSeek-R1-Distill" "1,2" 2 2048

# Start HarmBench-Llama-2-13b for PAIR eval model (Fix: correct max_model_len)
start_server "cais/HarmBench-Llama-2-13b-cls" 8003 "HarmBench-Llama-2-13b" "4" 1 2048

# Start Meta-Llama-3-70B for final evaluation (Fix: Use Meta-Llama-3-70B-Instruct)
start_server "meta-llama/Meta-Llama-3-70B-Instruct" 8006 "Meta-Llama-3-70B-Instruct" "6,7" 2

echo "🎯 All servers started! Checking status with improved health checks..."

# Robust health check function
check_server_health() {
    local port=$1
    local timeout=600 # 5 minutes timeout
    local start_time=$(date +%s)

    echo -n "Checking server on port $port..."
    while true; do
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))

        if [ $elapsed -ge $timeout ]; then
            echo " ❌ Timed out after ${timeout}s."
            return 1
        fi

        # Use curl to check the /v1/models endpoint which is a reliable indicator
        if curl -s -f "http://0.0.0.0:$port/v1/models" > /dev/null; then
            echo " ✅ Server is running."
            return 0
        fi
        
        sleep 10
        echo -n "."
    done
}

# Check all servers
all_running=true
for port in 8002 8003 8006; do
    if ! check_server_health $port; then
        all_running=false
        echo "❌ Server on port $port failed to start. Check server_${port}.log for details."
    fi
done

if $all_running; then
    echo "✅ All servers are up and running!"
else
    echo "❌ Some servers failed to start."
fi

echo "🔍 Server logs available at: server_8002.log, server_8003.log, server_8006.log"
echo "🌟 Multi-model servers ready for comprehensive attack testing!"
