#!/bin/bash

# Multi-model vLLM server startup script
# This script starts multiple vLLM servers for different attack models

export HF_HOME="/ceph_data/ycyoon/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

echo "🚀 Starting multi-model vLLM servers for attack benchmarks..."

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠️  Port $port already in use, stopping existing process..."
        pkill -f "port $port" 2>/dev/null || true
        sleep 2
    fi
}

# Function to start vLLM server
start_vllm_server() {
    local model_path=$1
    local model_name=$2
    local port=$3
    local gpu_id=$4
    local log_file=$5
    # GPU memory utilization, default to 0.7 if not supplied
    local gpu_mem_util=${6:-0.8}
    
    echo "📡 Starting $model_name server on port $port (GPU $gpu_id)..."
    
    check_port $port
    
    # Force GPU selection with multiple methods
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    
    # Check if model exists locally
    echo "🔎 Checking for local model: $model_path"
    if [ -f "$model_path/config.json" ] || [ -f "$model_path/pytorch_model.bin" ] || [ -f "$model_path/model.safetensors" ]; then
        echo "✅ Model found locally: $model_path"
        local_model_path="$model_path"
    else
        echo "🔍 Model not found locally, using HuggingFace model: $model_name"
        local_model_path="$model_name"
    fi
    
    # Start vLLM with explicit GPU selection
    env CUDA_VISIBLE_DEVICES=$gpu_id python -m vllm.entrypoints.openai.api_server \
        --model "$local_model_path" \
        --tensor-parallel-size 1 \
        --port $port \
        --dtype auto \
        --max-model-len 2048 \
        --gpu-memory-utilization $gpu_mem_util \
        --api-key "" \
        --served-model-name "$model_name" \
        --enforce-eager \
        --disable-log-requests \
        --swap-space 4 \
        > "$log_file" 2>&1 &
    
    local pid=$!
    echo "  📍 $model_name server started with PID: $pid"
    echo "  📄 Logs: $log_file"
    
    # Wait a bit for server process to initialize
    sleep 5
    # Verify process is running
    if ps -p $pid > /dev/null; then
        echo "  ✅ $model_name server process is alive"
    else
        echo "  ❌ $model_name server failed to start"
        cat "$log_file" | tail -10
        return
    fi
    # Poll HTTP endpoint until it's responsive
    for i in {1..10}; do
        http_code=$(curl -s --connect-timeout 3 -o /dev/null -w "%{http_code}" "http://localhost:$port/v1/models")
        if [ "$http_code" = "200" ]; then
            echo "  🚀 HTTP endpoint ready at port $port"
            break
        fi
        sleep 2
    done
    if [ "$http_code" != "200" ]; then
        echo "  ⚠️  HTTP endpoint not responding on port $port after retries"
        tail -n 10 "$log_file"
    fi
}

# Check if vLLM is available
if ! python -c "import vllm" 2>/dev/null; then
    echo "❌ vLLM not installed! Please install with: pip install vllm"
    exit 1
fi

echo "✅ vLLM is available"

# Stop any existing vLLM processes
echo "🧹 Stopping existing vLLM processes..."
pkill -f vllm 2>/dev/null || true
sleep 3

# Model configurations
# Format: model_path:model_name:port:gpu_id:log_file

# Target model (Llama 3.2 3B) - GPU 1
start_vllm_server \
    "/ceph_data/ycyoon/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/*/." \
    "meta-llama/Llama-3.2-3B-Instruct" \
    8001 \
    1 \
    "vllm_llama32_3b_server.log"

# Attack model (DeepSeek R1) - GPU 5 with low memory utilization
start_vllm_server \
    "/ceph_data/ycyoon/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B/snapshots/*/." \
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" \
    8002 \
    5 \
    "vllm_deepseek_r1_server.log" \
    0.1

# Evaluation model (HarmBench Llama 2 13B) - GPU 4
start_vllm_server \
    "/ceph_data/ycyoon/.cache/huggingface/hub/models--cais--HarmBench-Llama-2-13b-cls/snapshots/*/." \
    "cais/HarmBench-Llama-2-13b-cls" \
    8003 \
    3 \
    "vllm_harmbench_server.log"

# WildGuard model - GPU 2
start_vllm_server \
    "/ceph_data/ycyoon/.cache/huggingface/hub/models--allenai--wildguard/snapshots/*/." \
    "allenai/wildguard" \
    8004 \
    2 \
    "vllm_wildguard_server.log"

# meta-llama--Llama-3-70B
start_vllm_server \
    "/ceph_data/ycyoon/.cache/huggingface/hub/models--meta-llama--Llama-3-70B-Instruct/snapshots/*/." \
    "meta-llama/Meta-Llama-3-70B-Instruct" \
    8006 \
    6 \
    "vllm_meta_llama_3_70B_server.log"

echo ""
echo "🎉 Multi-model server startup completed!"
echo "📊 Active servers:"
echo "  • Llama 3.2 3B (Target): http://localhost:8001 [GPU 1]"
echo "  • DeepSeek R1 (Attack): http://localhost:8002 [GPU 5]" 
echo "  • HarmBench Llama 2 13B (Eval): http://localhost:8003 [GPU 3]"
echo "  • WildGuard (Safety): http://localhost:8004 [GPU 2]"
echo "  • Llama 4 Scout 17B (Final Eval): http://localhost:8006 [GPU 6]"

echo ""
echo "💡 To stop all servers later, use: pkill -f vllm"

# Test server health using /health endpoint with retries
# Test server health using /v1/models endpoint with retries
echo ""
echo "🧪 Testing server health..."
# Wait for servers to bind ports
sleep 10
# Retries up to 5 times per port
# Test server health using /v1/models endpoint with retries
for port in 8001 8002 8003 8004 8006; do
    status="❌ Not responding"
    for i in {1..5}; do
        code=$(curl -s --connect-timeout 3 -o /dev/null -w "%{http_code}" "http://localhost:$port/v1/models")
        if [ "$code" = "200" ]; then
            status="✅ OK"
            break
        fi
        sleep 2
    done
    echo "  Port $port: $status"
done

echo ""
echo "✅ Multi-model vLLM servers are ready!"
