#!/bin/bash

# Enable debugging
set -x  # Print each command before executing
set -e  # Exit on error

# Set environment variables to avoid external API calls
export OPENAI_API_KEY=""
export DEEPSEEK_PLATFORM_API_KEY=""
export OPENROUTER_API_KEY=""
export API_KEY=""

# Unset offline mode
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE

echo "🚀 HuggingFace Models Benchmark (DEBUG MODE)"
echo "==============================="
echo "🔧 Environment configured for local-only execution"

# Function to handle errors with more detail
handle_error() {
    local line=$1
    local exit_code=$?
    echo "❌ Error occurred in script at line $line with exit code $exit_code"
    echo "📍 Last command: $BASH_COMMAND"
    echo "🧹 Cleaning up any background processes..."
    pkill -f vllm 2>/dev/null || true
    exit 1
}

# Set up error handler
trap 'handle_error ${LINENO}' ERR

# Get script directory (where this script is located)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "📁 Working directory: $SCRIPT_DIR"

# Start servers
echo "📡 Starting vLLM servers for HuggingFace models..."
bash ./start_multi_model_servers.sh

echo ""
echo "⏳ Checking server status and waiting for readiness..."

# Function to check if server is responding
check_server() {
    local port=$1
    local name=$2
    
    echo "🔍 Checking server $name on port $port..."
    
    # Check if port is listening
    if netstat -tuln 2>/dev/null | grep ":$port " > /dev/null 2>&1; then
        echo "  ✓ Port $port is listening"
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
declare -a server_names=("DeepSeek-R1-Distill" "HarmBench-Llama-2-13b" "WildGuard" "Meta-Llama-3-70B")

# Test array access
echo "🔍 Testing array access..."
echo "Number of ports: ${#server_ports[@]}"
echo "Number of names: ${#server_names[@]}"

for i in "${!server_ports[@]}"; do
    echo "Index $i: port=${server_ports[$i]}, name=${server_names[$i]}"
done

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
    
    # Check each server
    for i in 0 1 2 3; do
        echo "🔍 Checking index $i..."
        port="${server_ports[$i]}"
        name="${server_names[$i]}"
        
        echo "  Port: $port, Name: $name"
        
        if check_server "$port" "$name"; then
            echo "  Server $name is ready"
            ready_count=$((ready_count + 1))
        else
            echo "  Server $name is not ready"
            all_ready=false
        fi
    done
    
    echo "📊 Ready servers: $ready_count/${#server_ports[@]}"
    
    if [ "$all_ready" = true ]; then
        echo ""
        echo "🎉 All servers are ready! Proceeding with benchmark..."
        break
    fi
    
    echo "⏳ Waiting ${wait_interval}s before next check..."
    sleep $wait_interval
    total_waited=$((total_waited + wait_interval))
done

echo "✅ Debug script completed successfully!"
