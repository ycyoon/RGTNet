#!/bin/bash

# Enable strict error handling
set -e

# Set environment variables to avoid external API calls
export OPENAI_API_KEY=""
export DEEPSEEK_PLATFORM_API_KEY=""
export OPENROUTER_API_KEY=""
export API_KEY=""

# Unset offline mode
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE

echo "🚀 HuggingFace Models Benchmark"
echo "==============================="
echo "🔧 Environment configured for local-only execution"

# Function to handle errors
handle_error() {
    echo "❌ Error occurred in script at line $1"
    echo "🧹 Cleaning up any background processes..."
    pkill -f vllm 2>/dev/null || true
    exit 1
}

# Set up error handler
trap 'handle_error ${LINENO}' ERR

# Check if required files exist
if [ ! -f "multi_model_config.yaml" ]; then
    echo "❌ multi_model_config.yaml not found!"
    exit 1
fi

if [ ! -f "multi_model_benchmark.py" ]; then
    echo "❌ multi_model_benchmark.py not found!"
    exit 1
fi

# Verify Python environment and dependencies
echo "🔍 Checking Python environment..."
python -c "import transformers; print(f'✅ Transformers version: {transformers.__version__}')" || {
    echo "❌ Transformers not available!"
    exit 1
}

# Check if easyjailbreak can be imported
python -c "
try:
    import sys
    import os
    sys.path.insert(0, os.path.abspath('.'))
    import easyjailbreak
    print('✅ EasyJailbreak module available')
except ImportError as e:
    print(f'⚠️  EasyJailbreak import issue: {e}')
    print('   Continuing with available modules...')
" || echo "⚠️  Some modules may not be available"

echo "✅ Configuration files found"
echo "✅ Using HuggingFace models with local inference"
echo "✅ Using local vLLM servers - no external API keys needed"

# Make server script executable
chmod +x start_deepseek_llama_servers.sh
chmod +x download_models.py

# Check if vLLM is available
if ! python -c "import vllm" 2>/dev/null; then
    echo "❌ vLLM not installed! Please install with: pip install vllm"
    exit 1
fi

echo "✅ vLLM is available"

# Pre-download models
echo "📥 Pre-downloading models to ensure availability..."
./download_models.py || {
    echo "⚠️ Model download encountered issues, but continuing..."
}

# Start servers
echo "📡 Starting vLLM servers for HuggingFace models..."
./start_deepseek_llama_servers.sh

echo ""
echo "⏳ Waiting for servers to be ready..."
sleep 10

echo ""
echo "🧪 Testing benchmark setup..."
./test_benchmark.sh

echo ""
echo "🎯 Starting benchmark with local HuggingFace models..."
echo "📁 Results will be saved to corresponding result directories"
echo ""

# First run the multi-model benchmark
echo "🔥 Running multi-model benchmark..."
if python multi_model_benchmark.py --models llama-3.2-3b --use-local; then
    echo "✅ Multi-model benchmark completed successfully"
else
    echo "⚠️  Multi-model benchmark encountered issues, continuing..."
fi

echo ""
# Run the benchmark with PAIR attack (if available)
echo "🔥 Running PAIR attack benchmark..."
if python -c "import easyjailbreak.attacker.PAIR_chao_2023" 2>/dev/null; then
    if python run_PAIR.py --target-model llama-3.2-3b --dataset-size 10; then
        echo "✅ PAIR attack completed successfully"
    else
        echo "⚠️  PAIR attack encountered issues"
    fi
else
    echo "⚠️  PAIR attack module not available, skipping..."
fi

echo ""
echo "🔥 Running WildteamAttack benchmark..."
if python -c "import easyjailbreak.attacker" 2>/dev/null; then
    if python run_WildteamAttack.py --target-model llama-3.2-3b --dataset-size 10 --action generate; then
        echo "✅ WildteamAttack generate completed successfully"
    else
        echo "⚠️  WildteamAttack generate encountered issues"
    fi
    
    if python run_WildteamAttack.py --target-model llama-3.2-3b --dataset-size 10 --action attack; then
        echo "✅ WildteamAttack attack completed successfully"
    else
        echo "⚠️  WildteamAttack attack encountered issues"
    fi
else
    echo "⚠️  WildteamAttack module not available, skipping..."
fi

echo ""
echo "🔥 Running Jailbroken attack benchmark..."
if python -c "import easyjailbreak.attacker.Jailbroken_wei_2023" 2>/dev/null; then
    if python run_jailbroken.py --target-model llama-3.2-3b --dataset-size 10; then
        echo "✅ Jailbroken attack completed successfully"
    else
        echo "⚠️  Jailbroken attack encountered issues"
    fi
else
    echo "⚠️  Jailbroken attack module not available, skipping..."
fi

echo ""
echo "🎉 Benchmark execution completed!"
echo "📊 Check the results in the generated result directories"

# Show recent result files
echo "📁 Generated result directories:"
ls -lt | grep -E "(results|benchmark|\.json|\.log)" | head -10 || echo "   No result files found yet"

echo ""
echo "🧹 Cleaning up background processes..."
echo "💡 Stopping vLLM servers..."
pkill -f vllm 2>/dev/null && echo "   ✅ vLLM processes stopped" || echo "   ℹ️  No vLLM processes found"

echo "✅ Cleanup completed!"
echo "🏁 Script execution finished"
