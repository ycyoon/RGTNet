#!/bin/bash

echo "Starting vLLM servers for Llama-3.2-3B benchmark..."

# Kill existing processes on these ports
echo "Cleaning up existing processes..."
pkill -f "vllm.entrypoints.openai.api_server"
sleep 5

# Check available GPU memory
echo "Checking GPU availability..."
nvidia-smi

# Start Llama-3.2-3B target model server
echo "Starting Llama-3.2-3B target model on port 8000..."
nohup python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.3 \
    --trust-remote-code > target_server.log 2>&1 &

sleep 20

# Start HarmBench evaluation model server  
echo "Starting HarmBench evaluation model on port 8001..."
nohup python -m vllm.entrypoints.openai.api_server \
    --model cais/HarmBench-Llama-2-13b-cls \
    --host 0.0.0.0 \
    --port 8001 \
    --tensor-parallel-size 1 \
    --max-model-len 1024 \
    --gpu-memory-utilization 0.3 \
    --trust-remote-code > eval_server.log 2>&1 &

sleep 20

# Start WildGuard refusal evaluation model server
echo "Starting WildGuard refusal evaluation model on port 8002..."
nohup python -m vllm.entrypoints.openai.api_server \
    --model allenai/wildguard \
    --host 0.0.0.0 \
    --port 8002 \
    --tensor-parallel-size 1 \
    --max-model-len 1024 \
    --gpu-memory-utilization 0.3 \
    --trust-remote-code > refusal_server.log 2>&1 &

echo "All servers starting... Waiting 60 seconds for initialization..."
sleep 60

# Test server connections
echo "Testing server connections..."
if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "✅ Target model server (port 8000): Ready"
else
    echo "❌ Target model server (port 8000): Not ready"
    echo "Check target_server.log for errors"
fi

if curl -s http://localhost:8001/v1/models > /dev/null 2>&1; then
    echo "✅ Eval model server (port 8001): Ready"
else
    echo "❌ Eval model server (port 8001): Not ready"
    echo "Check eval_server.log for errors"
fi

if curl -s http://localhost:8002/v1/models > /dev/null 2>&1; then
    echo "✅ Refusal eval model server (port 8002): Ready"
else
    echo "❌ Refusal eval model server (port 8002): Not ready" 
    echo "Check refusal_server.log for errors"
fi

echo "Servers initialization completed!"
echo "Check log files if any server failed to start:"
echo "  - target_server.log"
echo "  - eval_server.log"
echo "  - refusal_server.log"
