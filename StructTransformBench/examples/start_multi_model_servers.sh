#!/bin/bash

# Multi-model vLLM server startup script
# GPU 0,3 are occupied by gmk_multiseg_strategies, using clean GPUs only

echo "🚀 Starting multi-model vLLM servers..."
echo "⚠️  GPU 0,3 occupied by gmk_multiseg_strategies, using clean GPUs 1,2,4,5,6,7"

# Allow overriding max_model_len and set memory configuration
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fix Python path to avoid loading development FlashInfer
export PYTHONPATH=""

# Disable FlashInfer to avoid compatibility issues
export VLLM_USE_FLASHINFER_SAMPLER=0
export FLASHINFER_FORCE_BUILD_FROM_SOURCE=0
export VLLM_ATTENTION_BACKEND="TORCH"  # Force PyTorch attention backend

# Add distributed training stability settings
export NCCL_TIMEOUT=1800  # 30 minutes timeout
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=OFF
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

# Function to check if server is already running and healthy
is_server_healthy() {
    local port=$1
    # Check if port is open and server responds to /v1/models endpoint
    if curl -s -f "http://0.0.0.0:$port/v1/models" > /dev/null 2>&1; then
        return 0  # Server is healthy
    else
        return 1  # Server is not healthy or not running
    fi
}

# Function to start a server in background
start_server() {
    local model=$1
    local port=$2
    local name=$3
    local gpus=$4
    local tp_size=$5
    local max_len=${6:-2048} # Default max_model_len
    local mem_util=${7:-0.95}  # Default gpu_memory_utilization

    # Check if server is already running and healthy
    if is_server_healthy $port; then
        echo "✅ $name on port $port is already running and healthy. Skipping..."
        return 0
    fi

    echo "Starting $name on port $port using GPUs: $gpus (TP: $tp_size, MaxLen: $max_len, MemUtil: $mem_util)"
    
    # Kill any existing unhealthy process on this port
    existing_pid=$(lsof -ti:$port 2>/dev/null)
    if [ ! -z "$existing_pid" ]; then
        echo "  Killing existing unhealthy process on port $port (PID: $existing_pid)"
        kill -9 $existing_pid 2>/dev/null
        sleep 2
    fi
    
    # Clear PYTHONPATH and set environment for each server
    PYTHONPATH="" \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    VLLM_ATTENTION_BACKEND="TORCH" \
    NCCL_TIMEOUT=1800 \
    VLLM_WORKER_MULTIPROC_METHOD="spawn" \
    CUDA_VISIBLE_DEVICES=$gpus \
    nohup /home/ycyoon/anaconda3/envs/rgtnet/bin/python -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        --port $port \
        --host 0.0.0.0 \
        --trust-remote-code \
        --tensor-parallel-size $tp_size \
        --gpu-memory-utilization $mem_util \
        --max-model-len $max_len \
        --block-size 16 \
        --enforce-eager \
        --disable-custom-all-reduce \
        --distributed-executor-backend "mp" > "server_${port}.log" 2>&1 &
    
    local pid=$!
    echo "Server $name (PID: $pid) starting on port $port..."
}

# GPU 1: Reserved for dynamic target models (started by run_PAIR.py)
# GPU 2: Llama-3.1-70B-Instruct attack model (0,2 GPU)
# GPU 4: HarmBench (1 GPU)  
# GPU 5: WildGuard (1 GPU)
# GPU 6,7: Meta-Llama-3-70B (2 GPUs)

# NOTE: Target model on port 8001 will be started dynamically by run_PAIR.py

# Start Llama-3.1-70B-Instruct for PAIR attack model (single GPU)
start_server "meta-llama/Llama-3.1-70B-Instruct" 8002 "Llama-3.1-70B-Instruct" "0,2" 2 8192 0.95

# Start HarmBench-Llama-2-13b for PAIR eval model
start_server "cais/HarmBench-Llama-2-13b-cls" 8003 "HarmBench-Llama-2-13b" "4" 1 8192

# Start WildGuard for refusal evaluation
start_server "allenai/wildguard" 8004 "WildGuard" "5" 1 8192

# Start Meta-Llama-3-70B for final evaluation (Fix: Use Meta-Llama-3-70B-Instruct)
start_server "meta-llama/Meta-Llama-3-70B-Instruct" 8006 "Meta-Llama-3-70B-Instruct" "6,7" 2 8192

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

# Check all servers (excluding port 8001 which is managed by run_PAIR.py)
all_running=true
for port in 8002 8003 8004 8006; do
    if ! check_server_health $port; then
        all_running=false
        echo "❌ Server on port $port failed to start. Check server_${port}.log for details."
    fi
done

if $all_running; then
    echo "✅ All core servers are up and running!"
    echo "ℹ️  Target model server (port 8001) will be started dynamically by run_PAIR.py"
else
    echo "❌ Some servers failed to start."
fi

echo "🔍 Server logs available at: server_8002.log, server_8003.log, server_8004.log, server_8006.log"
echo "🌟 Multi-model servers ready for comprehensive attack testing!"
echo "📋 To run PAIR attack: python run_PAIR.py --target-model <model_name> --dataset-size <size>"
