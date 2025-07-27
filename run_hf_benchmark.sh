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
declare -a server_names=("DeepSeek-R1-Distill" "HarmBench-Llama-2-13b" "WildGuard" "Meta-Llama-3-70B")

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

# First run the multi-model benchmark
echo "🔥 Running multi-model benchmark..."
if python multi_model_benchmark.py --models llama-3.2-3b --use-local; then
    echo "✅ Multi-model benchmark completed successfully"
    # Find the most recent result file
    MULTI_RESULT=$(ls -t multi_model_results_*.json 2>/dev/null | head -1)
    [ -n "$MULTI_RESULT" ] && RESULT_FILES+=("$MULTI_RESULT")
else
    echo "⚠️  Multi-model benchmark encountered issues, continuing..."
fi

echo ""
# Run the benchmark with PAIR attack (if available)
echo "🔥 Running PAIR attack benchmark..."
if python -c "import easyjailbreak.attacker.PAIR_chao_2023" 2>/dev/null; then
    if python run_PAIR.py --target-model llama-3.2-3b --dataset-size 10; then
        echo "✅ PAIR attack completed successfully"
        # Find the most recent PAIR result file
        PAIR_RESULT=$(ls -t PAIR_results_*.json 2>/dev/null | head -1)
        [ -n "$PAIR_RESULT" ] && RESULT_FILES+=("$PAIR_RESULT")
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
        # Find the most recent WildteamAttack result file
        WILD_RESULT=$(ls -t WildteamAttack_results_*.json 2>/dev/null | head -1)
        [ -n "$WILD_RESULT" ] && RESULT_FILES+=("$WILD_RESULT")
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
        # Find the most recent jailbroken result file
        JAIL_RESULT=$(ls -t jailbroken_results_*.json 2>/dev/null | head -1)
        [ -n "$JAIL_RESULT" ] && RESULT_FILES+=("$JAIL_RESULT")
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
    
    # Create a Python script inline to analyze results
    cat > analyze_results_temp.py << 'EOF'
import json
import sys
from datetime import datetime

def analyze_file(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Extract metrics
        model = data.get('target_model', data.get('model', 'Unknown'))
        attack_type = 'Unknown'
        
        if 'PAIR' in filepath:
            attack_type = 'PAIR'
        elif 'Wildteam' in filepath:
            attack_type = 'WildteamAttack'
        elif 'jailbroken' in filepath:
            attack_type = 'Jailbroken'
        elif 'multi_model' in filepath:
            attack_type = 'Multi-Model'
            
        results = data.get('results', data.get('evaluation_results', []))
        dataset_size = len(results) if isinstance(results, list) else 0
        
        # Calculate success rate
        successful = 0
        if isinstance(results, list):
            for r in results:
                if r.get('jailbroken', r.get('success', r.get('is_jailbroken', False))):
                    successful += 1
        
        success_rate = (successful / dataset_size * 100) if dataset_size > 0 else 0
        
        return {
            'file': filepath,
            'model': model,
            'attack_type': attack_type,
            'dataset_size': dataset_size,
            'successful': successful,
            'success_rate': success_rate
        }
    except Exception as e:
        return None

# Analyze all files passed as arguments
results = []
for filepath in sys.argv[1:]:
    result = analyze_file(filepath)
    if result:
        results.append(result)

# Create summary table
if results:
    print("\n📊 Benchmark Results Summary")
    print("=" * 80)
    print(f"{'Attack Type':<20} {'Model':<20} {'Success Rate':<15} {'Dataset Size':<15}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['attack_type']:<20} {r['model'][:19]:<20} "
              f"{r['success_rate']:>6.1f}%        {r['dataset_size']:<15}")
    
    print("-" * 80)
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save to markdown file
    with open('benchmark_summary.md', 'w') as f:
        f.write("# Benchmark Results Summary\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("| Attack Type | Model | Success Rate | Dataset Size |\n")
        f.write("|-------------|-------|--------------|-------------|\n")
        
        for r in results:
            f.write(f"| {r['attack_type']} | {r['model']} | {r['success_rate']:.1f}% | {r['dataset_size']} |\n")
        
        f.write("\n## Result Files\n\n")
        for r in results:
            f.write(f"- `{r['file']}`\n")
    
    print("\n✅ Summary saved to: benchmark_summary.md")
else:
    print("No valid results found to analyze")
EOF

    # Run the analysis
    python analyze_results_temp.py "${RESULT_FILES[@]}"
    
    # Clean up temp file
    rm -f analyze_results_temp.py
fi

# Show all recent result files in directory
echo ""
echo "📁 All recent result files in directory:"
ls -lt *.json 2>/dev/null | head -10 || echo "  No JSON files found"

echo ""
echo "🧹 Cleaning up background processes..."
echo "💡 Stopping vLLM servers..."
pkill -f vllm 2>/dev/null && echo "   ✅ vLLM processes stopped" || echo "   ℹ️  No vLLM processes found"

echo "✅ Cleanup completed!"
echo "🏁 Script execution finished"
