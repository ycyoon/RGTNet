#!/bin/bash

# Multi-model vLLM server startup script - unified configuration
echo "🚀 Starting vLLM servers for HuggingFace models..."

# Allow overriding max_model_len and set memory configuration
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Ensure HF_TOKEN is set and offline mode is disabled
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️ Warning: HF_TOKEN is not set. Some models may not be accessible."
    echo "💡 You can set it with: export HF_TOKEN=your_token_here"
else
    echo "✅ HF_TOKEN is set"
fi

# Ensure offline mode is disabled
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

# Check if vLLM is installed
echo "🔍 Checking vLLM installation..."
if python -c "import vllm" 2>/dev/null; then
    echo "✅ vLLM is available"
    VLLM_AVAILABLE=true
else
    echo "❌ vLLM is not installed"
    echo "💡 Options:"
    echo "   1. Install vLLM: pip install vllm"
    echo "   2. Continue with alternative HuggingFace server mode"
    echo ""
    read -p "Do you want to install vLLM now? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "📦 Installing vLLM..."
        pip install vllm
        if python -c "import vllm" 2>/dev/null; then
            echo "✅ vLLM installed successfully"
            VLLM_AVAILABLE=true
        else
            echo "❌ vLLM installation failed"
            VLLM_AVAILABLE=false
        fi
    else
        echo "⚠️  Continuing without vLLM (will use alternative mode)"
        VLLM_AVAILABLE=false
    fi
fi

# Check if ports are already in use
check_port() {
    local port=$1
    local model_name=$2
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠️  Port $port already in use - $model_name server may already be running"
        return 1
    fi
    return 0
}

# Function to start a vLLM server with unified configuration
start_vllm_server() {
    local model_name=$1
    local port=$2
    local gpu_ids=$3
    local display_name=$4
    local mem_util=${5:-0.6}  # Default gpu_memory_utilization
    local max_len=${6:-1024}  # Default max_model_len
    
    echo "📡 Starting $display_name server on port $port using GPUs: $gpu_ids..."
    
    if check_port $port "$display_name"; then
        if [ "$VLLM_AVAILABLE" = true ]; then
            # Use unified vLLM configuration
            CUDA_VISIBLE_DEVICES=$gpu_ids nohup python -m vllm.entrypoints.openai.api_server 
                --model "$model_name" 
                --port $port 
                --host 0.0.0.0 
                --trust-remote-code 
                --tensor-parallel-size $(echo $gpu_ids | tr ',' '\n' | wc -l) 
                --gpu-memory-utilization $mem_util 
                --max-model-len $max_len 
                --api-key "" 
                --served-model-name $display_name 
                --enforce-eager 
                --disable-custom-all-reduce 
                --disable-log-requests 
                > "vllm_${display_name,,}_server.log" 2>&1 &
        else
            # Use alternative HuggingFace server
            echo "  🔄 Using alternative HuggingFace server mode..."
            CUDA_VISIBLE_DEVICES=$gpu_ids python -c "
import os
import sys
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print('🔧 Loading model: $model_name')
tokenizer = AutoTokenizer.from_pretrained('$model_name')
model = AutoModelForCausalLM.from_pretrained('$model_name', torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

app = Flask(__name__)

@app.route('/v1/models', methods=['GET'])
def list_models():
    return jsonify({'data': [{'id': '$display_name', 'object': 'model'}]})

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    data = request.json
    messages = data.get('messages', [])
    if messages:
        prompt = messages[-1].get('content', '')
        inputs = tokenizer(prompt, return_tensors='pt')
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return jsonify({'choices': [{'message': {'content': response}}]})
    return jsonify({'error': 'No messages provided'})

print('✅ Server ready on port $port')
app.run(host='0.0.0.0', port=$port)
" > "hf_${display_name,,}_server.log" 2>&1 &
        fi
        
        local server_pid=$!
        echo "  📍 $display_name server started with PID: $server_pid"
        
        if [ "$VLLM_AVAILABLE" = true ]; then
            echo "  📄 Logs: vllm_${display_name,,}_server.log"
        else
            echo "  📄 Logs: hf_${display_name,,}_server.log"
        fi
        
        # Wait a bit for server to start
        sleep 5
        
        # Robust health check function with improved timeout
        local timeout=300  # 5 minutes timeout
        local start_time=$(date +%s)
        
        echo -n "  ⏳ Waiting for $display_name server to be ready..."
        while true; do
            current_time=$(date +%s)
            elapsed=$((current_time - start_time))

            if [ $elapsed -ge $timeout ]; then
                echo " ❌ Timed out after ${timeout}s."
                return 1
            fi

            # Use curl to check the /v1/models endpoint
            if curl -s -f "http://localhost:$port/v1/models" > /dev/null 2>&1; then
                echo " ✅ Server is ready!"
                return 0
            fi
            
            sleep 5
            echo -n "."
        done
        
        echo "  ❌ $display_name server failed to start properly"
        # Show last log lines for debugging
        if [ "$VLLM_AVAILABLE" = true ]; then
            LOG_FILE="vllm_${display_name,,}_server.log"
        else
            LOG_FILE="hf_${display_name,,}_server.log"
        fi
        
        if [ -f "$LOG_FILE" ]; then
            echo "   📑 Last log lines for $display_name server:"
            tail -n 20 "$LOG_FILE" | sed 's/^/      /'
        fi
        return 1
    else
        echo "  ✅ $display_name server already running on port $port"
        return 0
    fi
}

# Function to ensure model is downloaded and get its local path
ensure_model_is_local() {
    local model_repo_id=$1
    echo "🔎 Checking for local cache of $model_repo_id..." >&2

    # Try to find the model path from local cache first
    local model_path
    model_path=$(HF_HUB_OFFLINE=1 python -c "
import os
try:
    from huggingface_hub import snapshot_download
    path = snapshot_download('$model_repo_id', local_files_only=True)
    print(path)
except Exception as e:
    pass
" 2>/dev/null)

    # Parse the model path from the output
    local cached_path=$(echo "$model_path" | grep "^/ceph_data\|^/" | tail -1)
    
    if [ -n "$cached_path" ] && [ -d "$cached_path" ]; then
        echo "✅ Model found in local cache: $cached_path" >&2
        echo "$cached_path"
        return 0
    fi

    echo "⚠️  Model not found in local cache. Starting download from Hugging Face Hub..." >&2
    echo "   This may take a while depending on model size and network speed." >&2

    # Check and set up cache directory
    echo "🔧 Setting up cache directories..." >&2
    
    # Download the model with proper error handling and environment setup
    model_path=$(python -c "
import sys
import os

# Use the exact environment variables from .bashrc
required_vars = ['HF_HOME', 'HF_HUB_CACHE', 'TRANSFORMERS_CACHE', 'HF_DATASETS_CACHE']
for var in required_vars:
    if var in os.environ:
        print(f'✅ {var}: {os.environ[var]}')
    else:
        print(f'⚠️  {var}: Not set in environment')

# Use existing environment variables (should be set from .bashrc)
hf_home = os.environ.get('HF_HOME')
hf_hub_cache = os.environ.get('HF_HUB_CACHE')
transformers_cache = os.environ.get('TRANSFORMERS_CACHE')

if not hf_home:
    print('❌ HF_HOME not set! Please source ~/.bashrc')
    sys.exit(1)

# Ensure cache directories exist
print(f'📁 Ensuring cache directories exist...')
for cache_dir in [hf_home, hf_hub_cache, transformers_cache]:
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        print(f'   ✅ {cache_dir}')

# Remove offline environment variables for download
offline_vars = ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE']
for var in offline_vars:
    if var in os.environ:
        del os.environ[var]
        print(f'🌐 Removed {var} for online download')

print(f'� Starting download of $model_repo_id...')

try:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError
    
    print('📥 Downloading $model_repo_id...')
    path = snapshot_download('$model_repo_id', resume_download=True)
    print(f'✅ Download completed: {path}')
    print(path)
except HfHubHTTPError as e:
    if '401' in str(e):
        print('❌ Authentication required. Set HF_TOKEN environment variable.', file=sys.stderr)
        print('   Get token from: https://huggingface.co/settings/tokens', file=sys.stderr)
    elif '403' in str(e):
        print('❌ Access denied. Request access to this model first.', file=sys.stderr)
    elif '404' in str(e):
        print('❌ Model not found. Check model name: $model_repo_id', file=sys.stderr)
    else:
        print(f'❌ HTTP Error: {e}', file=sys.stderr)
except ImportError as e:
    print(f'❌ Missing package: {e}', file=sys.stderr)
    print('   Try: pip install huggingface_hub', file=sys.stderr)
except Exception as e:
    print(f'❌ Download error: {e}', file=sys.stderr)
    print(f'   Model: $model_repo_id', file=sys.stderr)
" 2>&1)

    # Parse the output to get the path
    download_path=$(echo "$model_path" | grep "^/ceph_data\|^/" | grep -v "✅\|⚠️\|📁\|🌐\|📥" | tail -1)
    
    if [ -n "$download_path" ] && [ -d "$download_path" ]; then
        echo "✅ Download complete. Model is at: $download_path"
        echo "$download_path"
        return 0
    else
        echo "❌ Failed to download model $model_repo_id."
        echo "📋 Debug information:"
        echo "$model_path" | sed 's/^/   /'
        
        echo ""
        echo "💡 Troubleshooting suggestions:"
        echo "   1. Check internet connection"
        echo "   2. Verify model name: $model_repo_id"
        echo "   3. Ensure ~/.bashrc is sourced: source ~/.bashrc"
        echo "   4. Check if cache directories exist and are writable:"
        echo "      ls -la /ceph_data/ycyoon/.cache/huggingface/"
        echo "   5. Set HF_TOKEN if authentication is required:"
        echo "      export HF_TOKEN=your_token_here"
        echo "   6. Try a different model (e.g., gpt2, distilgpt2)"
        
        return 1
    fi
}

# Model configuration - using a publicly available model
MODEL_REPO_ID="microsoft/DialoGPT-medium"  # Medium-sized, publicly available
echo "🤖 Target model: $MODEL_REPO_ID"

# Display current environment
echo "🔧 Environment check:"
echo "   HF_HOME: ${HF_HOME:-'(not set, will use default)'}"
echo "   HF_HUB_CACHE: ${HF_HUB_CACHE:-'(not set, will use HF_HOME/hub)'}"
echo "   TRANSFORMERS_CACHE: ${TRANSFORMERS_CACHE:-'(not set, will use HF_HOME/transformers)'}"
echo "   HF_DATASETS_CACHE: ${HF_DATASETS_CACHE:-'(not set, will use HF_HOME/datasets)'}"
echo "   HF_TOKEN: ${HF_TOKEN:+'(set)' || '(not set)'}"

# Ensure the model is downloaded and get the local path
echo "🔍 Preparing model..."
LOCAL_MODEL_PATH=$(ensure_model_is_local "$MODEL_REPO_ID")

if [ -z "$LOCAL_MODEL_PATH" ]; then
    echo "❌ Aborting: Could not get local path for $MODEL_REPO_ID."
    exit 1
fi

echo "📁 Using local model path: $LOCAL_MODEL_PATH"
echo ""

# Start DialoGPT server using the local path
start_vllm_server "$LOCAL_MODEL_PATH" 8001 "0" "DialoGPT-Medium"

echo ""
echo "🎉 Server startup completed!"
echo "📊 Active servers:"
if [ "$VLLM_AVAILABLE" = true ]; then
    echo "  • DialoGPT-Medium (vLLM): http://localhost:8001"
else
    echo "  • DialoGPT-Medium (HuggingFace): http://localhost:8001"
fi
echo ""
if [ "$VLLM_AVAILABLE" = true ]; then
    echo "💡 To stop servers later, use: pkill -f vllm"
else
    echo "💡 To stop servers later, use: pkill -f flask"
fi
