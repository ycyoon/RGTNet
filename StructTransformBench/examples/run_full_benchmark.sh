#!/bin/bash

echo "🚀 Llama-3.2-3B Benchmark with Pre-generated Attacks"
echo "================================================================"

# Check if required files exist
if [ ! -f "config_llama32.yaml" ]; then
    echo "❌ config_llama32.yaml not found!"
    exit 1
fi

if [ ! -f "run_llama32_prebuilt.py" ]; then
    echo "❌ run_llama32_prebuilt.py not found!"
    exit 1
fi

echo "✅ Configuration files found"
echo "✅ Using pre-generated attacks - no OpenAI API key needed"

# Make server script executable
chmod +x start_llama32_servers.sh

# Check if vLLM is available
if ! python -c "import vllm" 2>/dev/null; then
    echo "❌ vLLM not installed! Please install with: pip install vllm"
    exit 1
fi

echo "✅ vLLM is available"

# Start servers
echo "📡 Starting vLLM servers..."
./start_llama32_servers.sh

echo ""
echo "🎯 Starting Llama-3.2-3B benchmark..."
echo "📁 Results will be saved to: ./llama32_benchmark_results/"
echo ""

# Run the benchmark
python run_llama32_prebuilt.py

echo ""
echo "🎉 Benchmark completed!"
echo "📊 Check the results in ./llama32_benchmark_results/"

# Show recent result files
if [ -d "./llama32_benchmark_results" ]; then
    echo "📁 Recent result files:"
    ls -lt ./llama32_benchmark_results/ | head -5
fi
