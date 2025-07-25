#!/bin/bash

# Foundation Model Server Manager
# 다양한 foundation 모델들을 위한 서버 관리 스크립트

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Model configurations
declare -A MODELS=(
    ["llama-3.2-1b"]="meta-llama/Llama-3.2-1B-Instruct:8000"
    ["llama-3.2-3b"]="meta-llama/Llama-3.2-3B-Instruct:8000"
    ["llama-3.1-8b"]="meta-llama/Llama-3.1-8B-Instruct:8003"
    ["qwen-2.5-7b"]="Qwen/Qwen2.5-7B-Instruct:8004"
    ["qwen-2.5-14b"]="Qwen/Qwen2.5-14B-Instruct:8005"
    ["mistral-7b"]="mistralai/Mistral-7B-Instruct-v0.3:8006"
    ["gemma-2-9b"]="google/gemma-2-9b-it:8007"
    ["phi-3.5-mini"]="microsoft/Phi-3.5-mini-instruct:8008"
)

# Evaluator models
declare -A EVALUATORS=(
    ["harmbench"]="cais/HarmBench-Llama-2-13b-cls:8001"
    ["wildguard"]="allenai/wildguard:8002"
)

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}Foundation Model Server Manager${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  start [model_key]     - Start specific model server"
    echo "  start-all            - Start all model servers"
    echo "  start-evaluators     - Start evaluation model servers"
    echo "  stop [port]          - Stop server on specific port"
    echo "  stop-all             - Stop all model servers"
    echo "  status               - Check status of all servers"
    echo "  list                 - List available models"
    echo "  benchmark            - Run multi-model benchmark"
    echo "  quick-test [model]   - Quick test of specific model"
    echo ""
    echo "Available Models:"
    for key in "${!MODELS[@]}"; do
        IFS=':' read -r model_name port <<< "${MODELS[$key]}"
        echo "  $key -> $model_name (port $port)"
    done
    echo ""
    echo "Evaluators:"
    for key in "${!EVALUATORS[@]}"; do
        IFS=':' read -r model_name port <<< "${EVALUATORS[$key]}"
        echo "  $key -> $model_name (port $port)"
    done
}

check_dependencies() {
    echo -e "${YELLOW}Checking dependencies...${NC}"
    
    # Check if vllm is installed
    if ! python -c "import vllm" 2>/dev/null; then
        echo -e "${RED}❌ vLLM not found. Please install with: pip install vllm${NC}"
        return 1
    fi
    
    # Check if transformers is installed
    if ! python -c "import transformers" 2>/dev/null; then
        echo -e "${RED}❌ transformers not found. Please install with: pip install transformers${NC}"
        return 1
    fi
    
    # Check GPU availability
    if ! nvidia-smi > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  NVIDIA GPU not detected. Models will run on CPU (slower)${NC}"
    else
        echo -e "${GREEN}✅ NVIDIA GPU detected${NC}"
    fi
    
    echo -e "${GREEN}✅ Dependencies check complete${NC}"
    return 0
}

check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

start_model_server() {
    local model_key=$1
    
    if [[ -z "${MODELS[$model_key]}" ]]; then
        echo -e "${RED}❌ Unknown model: $model_key${NC}"
        return 1
    fi
    
    IFS=':' read -r model_name port <<< "${MODELS[$model_key]}"
    
    echo -e "${BLUE}Starting $model_key...${NC}"
    echo -e "   Model: $model_name"
    echo -e "   Port: $port"
    
    # Check if port is already in use
    if check_port $port; then
        echo -e "${YELLOW}⚠️  Port $port is already in use${NC}"
        echo -e "   Checking if it's the same model..."
        
        # Try to query the existing server
        if curl -s "http://localhost:$port/v1/models" > /dev/null; then
            echo -e "${GREEN}✅ Server already running on port $port${NC}"
            return 0
        else
            echo -e "${RED}❌ Port $port is occupied by another service${NC}"
            return 1
        fi
    fi
    
    # Start the server
    echo -e "${YELLOW}🚀 Launching vLLM server...${NC}"
    
    # Create log directory
    mkdir -p logs
    
    # Determine GPU memory settings based on model size
    local gpu_memory_utilization="0.8"
    local max_model_len="4096"
    
    case $model_key in
        *"1b"*|*"mini"*)
            gpu_memory_utilization="0.3"
            max_model_len="2048"
            ;;
        *"3b"*)
            gpu_memory_utilization="0.5"
            max_model_len="4096"
            ;;
        *"7b"*|*"8b"*|*"9b"*)
            gpu_memory_utilization="0.8"
            max_model_len="4096"
            ;;
        *"14b"*)
            gpu_memory_utilization="0.9"
            max_model_len="4096"
            ;;
    esac
    
    # Launch vLLM server in background
    nohup python -m vllm.entrypoints.openai.api_server \
        --model "$model_name" \
        --port $port \
        --gpu-memory-utilization $gpu_memory_utilization \
        --max-model-len $max_model_len \
        --trust-remote-code \
        --disable-log-requests \
        > "logs/${model_key}_server.log" 2>&1 &
    
    local server_pid=$!
    echo $server_pid > "logs/${model_key}_server.pid"
    
    echo -e "${YELLOW}⏳ Waiting for server to start...${NC}"
    
    # Wait for server to be ready (up to 300 seconds)
    local attempts=0
    local max_attempts=60
    
    while [ $attempts -lt $max_attempts ]; do
        if curl -s "http://localhost:$port/v1/models" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Server started successfully!${NC}"
            echo -e "   PID: $server_pid"
            echo -e "   URL: http://localhost:$port"
            echo -e "   Logs: logs/${model_key}_server.log"
            return 0
        fi
        
        sleep 5
        attempts=$((attempts + 1))
        echo -n "."
    done
    
    echo -e "\n${RED}❌ Server failed to start within timeout${NC}"
    echo -e "   Check logs: logs/${model_key}_server.log"
    
    # Kill the process if it's still running
    if kill -0 $server_pid 2>/dev/null; then
        kill $server_pid
        rm -f "logs/${model_key}_server.pid"
    fi
    
    return 1
}

start_evaluator_server() {
    local eval_key=$1
    
    if [[ -z "${EVALUATORS[$eval_key]}" ]]; then
        echo -e "${RED}❌ Unknown evaluator: $eval_key${NC}"
        return 1
    fi
    
    IFS=':' read -r model_name port <<< "${EVALUATORS[$eval_key]}"
    
    echo -e "${BLUE}Starting evaluator $eval_key...${NC}"
    echo -e "   Model: $model_name"
    echo -e "   Port: $port"
    
    # Check if port is already in use
    if check_port $port; then
        echo -e "${YELLOW}⚠️  Evaluator server already running on port $port${NC}"
        return 0
    fi
    
    # Create log directory
    mkdir -p logs
    
    # Launch evaluator server
    nohup python -m vllm.entrypoints.openai.api_server \
        --model "$model_name" \
        --port $port \
        --gpu-memory-utilization 0.3 \
        --max-model-len 2048 \
        --trust-remote-code \
        --disable-log-requests \
        > "logs/${eval_key}_evaluator.log" 2>&1 &
    
    local server_pid=$!
    echo $server_pid > "logs/${eval_key}_evaluator.pid"
    
    echo -e "${YELLOW}⏳ Waiting for evaluator to start...${NC}"
    
    # Wait for server to be ready
    local attempts=0
    local max_attempts=30
    
    while [ $attempts -lt $max_attempts ]; do
        if curl -s "http://localhost:$port/v1/models" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Evaluator started successfully!${NC}"
            return 0
        fi
        
        sleep 5
        attempts=$((attempts + 1))
        echo -n "."
    done
    
    echo -e "\n${RED}❌ Evaluator failed to start${NC}"
    return 1
}

stop_server() {
    local port=$1
    
    echo -e "${YELLOW}Stopping server on port $port...${NC}"
    
    # Find and kill the process
    local pid=$(lsof -ti:$port)
    
    if [[ -n "$pid" ]]; then
        kill $pid
        sleep 2
        
        # Force kill if still running
        if kill -0 $pid 2>/dev/null; then
            kill -9 $pid
        fi
        
        echo -e "${GREEN}✅ Server stopped${NC}"
    else
        echo -e "${YELLOW}⚠️  No server running on port $port${NC}"
    fi
    
    # Clean up PID files
    rm -f logs/*_server.pid logs/*_evaluator.pid
}

stop_all_servers() {
    echo -e "${YELLOW}Stopping all model servers...${NC}"
    
    # Stop model servers
    for model_key in "${!MODELS[@]}"; do
        IFS=':' read -r model_name port <<< "${MODELS[$model_key]}"
        if check_port $port; then
            stop_server $port
        fi
    done
    
    # Stop evaluator servers
    for eval_key in "${!EVALUATORS[@]}"; do
        IFS=':' read -r model_name port <<< "${EVALUATORS[$eval_key]}"
        if check_port $port; then
            stop_server $port
        fi
    done
    
    echo -e "${GREEN}✅ All servers stopped${NC}"
}

check_server_status() {
    echo -e "${BLUE}Server Status:${NC}"
    echo ""
    
    echo "Model Servers:"
    for model_key in "${!MODELS[@]}"; do
        IFS=':' read -r model_name port <<< "${MODELS[$model_key]}"
        
        if check_port $port; then
            # Try to query the API
            if curl -s "http://localhost:$port/v1/models" > /dev/null 2>&1; then
                echo -e "  ${GREEN}✅${NC} $model_key ($port) - Running"
            else
                echo -e "  ${YELLOW}⚠️${NC} $model_key ($port) - Port occupied but not responding"
            fi
        else
            echo -e "  ${RED}❌${NC} $model_key ($port) - Stopped"
        fi
    done
    
    echo ""
    echo "Evaluator Servers:"
    for eval_key in "${!EVALUATORS[@]}"; do
        IFS=':' read -r model_name port <<< "${EVALUATORS[$eval_key]}"
        
        if check_port $port; then
            if curl -s "http://localhost:$port/v1/models" > /dev/null 2>&1; then
                echo -e "  ${GREEN}✅${NC} $eval_key ($port) - Running"
            else
                echo -e "  ${YELLOW}⚠️${NC} $eval_key ($port) - Port occupied but not responding"
            fi
        else
            echo -e "  ${RED}❌${NC} $eval_key ($port) - Stopped"
        fi
    done
}

quick_test_model() {
    local model_key=$1
    
    if [[ -z "${MODELS[$model_key]}" ]]; then
        echo -e "${RED}❌ Unknown model: $model_key${NC}"
        return 1
    fi
    
    IFS=':' read -r model_name port <<< "${MODELS[$model_key]}"
    
    echo -e "${BLUE}Quick testing $model_key...${NC}"
    
    # Check if server is running
    if ! check_port $port; then
        echo -e "${RED}❌ Server not running. Start it first with: $0 start $model_key${NC}"
        return 1
    fi
    
    # Test API call
    local test_prompt="Hello, how are you?"
    
    echo -e "${YELLOW}Sending test prompt: '$test_prompt'${NC}"
    
    local response=$(curl -s -X POST "http://localhost:$port/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$model_name\",
            \"prompt\": \"$test_prompt\",
            \"max_tokens\": 50,
            \"temperature\": 0.7
        }")
    
    if [[ $? -eq 0 ]] && [[ -n "$response" ]]; then
        echo -e "${GREEN}✅ Model response received:${NC}"
        echo "$response" | python -m json.tool
    else
        echo -e "${RED}❌ Failed to get response from model${NC}"
        return 1
    fi
}

run_benchmark() {
    echo -e "${BLUE}Running multi-model benchmark...${NC}"
    
    # Check if benchmark script exists
    if [[ ! -f "multi_model_benchmark.py" ]]; then
        echo -e "${RED}❌ multi_model_benchmark.py not found${NC}"
        return 1
    fi
    
    # Check if evaluators are running
    local evaluators_ready=true
    for eval_key in "${!EVALUATORS[@]}"; do
        IFS=':' read -r model_name port <<< "${EVALUATORS[$eval_key]}"
        if ! check_port $port; then
            echo -e "${YELLOW}⚠️  Evaluator $eval_key not running${NC}"
            evaluators_ready=false
        fi
    done
    
    if [[ "$evaluators_ready" == "false" ]]; then
        echo -e "${YELLOW}Starting evaluator servers...${NC}"
        start_evaluator_server "harmbench"
        start_evaluator_server "wildguard"
    fi
    
    # Run the benchmark
    echo -e "${GREEN}🚀 Starting benchmark...${NC}"
    python multi_model_benchmark.py
}

start_all_servers() {
    echo -e "${BLUE}Starting all model servers...${NC}"
    
    # Start evaluators first
    echo -e "${YELLOW}Starting evaluators...${NC}"
    start_evaluator_server "harmbench"
    start_evaluator_server "wildguard"
    
    # Start model servers
    echo -e "${YELLOW}Starting model servers...${NC}"
    for model_key in "${!MODELS[@]}"; do
        echo ""
        start_model_server "$model_key"
        
        # Add delay between server starts to avoid resource conflicts
        if [[ $? -eq 0 ]]; then
            echo -e "${GREEN}✅ $model_key started successfully${NC}"
            sleep 10  # Wait before starting next server
        else
            echo -e "${RED}❌ Failed to start $model_key${NC}"
        fi
    done
    
    echo ""
    echo -e "${GREEN}🎉 All servers startup process completed!${NC}"
    check_server_status
}

# Main execution
case "$1" in
    "start")
        print_header
        check_dependencies
        if [[ -n "$2" ]]; then
            start_model_server "$2"
        else
            echo -e "${RED}❌ Please specify a model to start${NC}"
            print_usage
        fi
        ;;
    "start-all")
        print_header
        check_dependencies
        start_all_servers
        ;;
    "start-evaluators")
        print_header
        check_dependencies
        start_evaluator_server "harmbench"
        start_evaluator_server "wildguard"
        ;;
    "stop")
        if [[ -n "$2" ]]; then
            stop_server "$2"
        else
            echo -e "${RED}❌ Please specify a port to stop${NC}"
        fi
        ;;
    "stop-all")
        stop_all_servers
        ;;
    "status")
        check_server_status
        ;;
    "list")
        print_usage
        ;;
    "benchmark")
        print_header
        run_benchmark
        ;;
    "quick-test")
        if [[ -n "$2" ]]; then
            quick_test_model "$2"
        else
            echo -e "${RED}❌ Please specify a model to test${NC}"
        fi
        ;;
    *)
        print_header
        print_usage
        ;;
esac
