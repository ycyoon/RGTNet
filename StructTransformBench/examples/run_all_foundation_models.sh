#!/bin/bash

# Array of all foundation models from multi_model_benchmark.py
MODELS=(
    "llama-3.2-1b"
    "llama-3.2-3b"
    "llama-3.1-8b"
    "qwen-2.5-7b"
    "qwen-2.5-14b"
    "mistral-7b"
    "gemma-2-9b"
    "phi-3.5-mini"
    "vicuna-7b"
    "falcon-7b"
)

# Attack method to run (PAIR, WildteamAttack, or jailbroken)
ATTACK_METHOD=${1:-"PAIR"}
DATASET_SIZE=${2:-"100"}  # Default to 100 samples for faster testing

# Create results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="foundation_models_results_${ATTACK_METHOD}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "Running ${ATTACK_METHOD} attack on all foundation models..."
echo "Dataset size: ${DATASET_SIZE}"
echo "Results directory: ${RESULTS_DIR}"
echo "=========================================="

# Log file for overall progress
LOG_FILE="${RESULTS_DIR}/benchmark_log.txt"

# Loop through all models
for model in "${MODELS[@]}"; do
    echo ""
    echo "Testing model: $model" | tee -a "$LOG_FILE"
    echo "Start time: $(date)" | tee -a "$LOG_FILE"
    
    # Run the attack
    if [ "$ATTACK_METHOD" == "WildteamAttack" ]; then
        # WildteamAttack has a different interface with --action parameter
        python run_${ATTACK_METHOD}.py --target-model "$model" --dataset-size "$DATASET_SIZE" --action generate 2>&1 | tee -a "${RESULTS_DIR}/${model}_${ATTACK_METHOD}.log"
        
        # Also run the attack phase
        python run_${ATTACK_METHOD}.py --target-model "$model" --dataset-size "$DATASET_SIZE" --action attack 2>&1 | tee -a "${RESULTS_DIR}/${model}_${ATTACK_METHOD}.log"
    else
        python run_${ATTACK_METHOD}.py --target-model "$model" --dataset-size "$DATASET_SIZE" 2>&1 | tee -a "${RESULTS_DIR}/${model}_${ATTACK_METHOD}.log"
    fi
    
    # Check exit status
    if [ $? -eq 0 ]; then
        echo "✓ Completed testing for $model" | tee -a "$LOG_FILE"
    else
        echo "✗ Failed testing for $model" | tee -a "$LOG_FILE"
    fi
    
    echo "End time: $(date)" | tee -a "$LOG_FILE"
    echo "-----------------------------------" | tee -a "$LOG_FILE"
    
    # Move results to organized directory
    if [ -d "${ATTACK_METHOD}-${model}-results" ]; then
        mv "${ATTACK_METHOD}-${model}-results" "${RESULTS_DIR}/"
    fi
done

echo ""
echo "=========================================="
echo "All models tested. Results saved in ${RESULTS_DIR}/"
echo "Summary log: ${LOG_FILE}"

# Generate summary report
echo ""
echo "Generating summary report..."
python -c "
import os
import json
import glob

results_dir = '${RESULTS_DIR}'
attack_method = '${ATTACK_METHOD}'
models = ${MODELS[@]}

print('\\nFoundation Models Benchmark Summary')
print('=' * 50)
print(f'Attack Method: {attack_method}')
print(f'Models Tested: {len(models)}')
print('=' * 50)

# Parse results
for model in models:
    result_dirs = glob.glob(f'{results_dir}/{attack_method}-{model}-results*')
    if result_dirs:
        print(f'\\n{model}:')
        # Look for JSONL files with results
        jsonl_files = glob.glob(f'{result_dirs[0]}/*.jsonl')
        if jsonl_files:
            print(f'  ✓ Results found: {os.path.basename(jsonl_files[0])}')
        else:
            print('  ⚠ No JSONL results found')
    else:
        print(f'\\n{model}: ✗ No results directory found')
"
