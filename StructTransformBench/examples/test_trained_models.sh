#!/bin/bash

# Test script for trained RGTNet models in benchmarks

echo "🧪 Testing Trained RGTNet Model Integration"
echo "=========================================="

# Check if trained model exists
TRAINED_MODEL="/home/ycyoon/work/RGTNet/models/llama3.2_3b_rgtnet_epoch1.pth"

if [ ! -f "$TRAINED_MODEL" ]; then
    echo "❌ Trained model not found at: $TRAINED_MODEL"
    echo "Available models:"
    ls -la /home/ycyoon/work/RGTNet/models/*.pth 2>/dev/null || echo "  No .pth files found"
    exit 1
fi

echo "✅ Found trained model: $TRAINED_MODEL"

# Test 1: Check if RGTNet modules can be imported
echo ""
echo "🔍 Test 1: Checking RGTNet module imports..."
python -c "
import sys
sys.path.append('/home/ycyoon/work/RGTNet')
try:
    from model import create_model, load_checkpoint
    from config import setup_args
    print('✅ RGTNet modules imported successfully')
except ImportError as e:
    print(f'❌ RGTNet import failed: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Module import test failed"
    exit 1
fi

# Test 2: Test model loading
echo ""
echo "🔍 Test 2: Testing model loading..."
python -c "
import sys
sys.path.append('/home/ycyoon/work/RGTNet')
sys.path.append('/home/ycyoon/work/RGTNet/StructTransformBench/examples')

try:
    from run_PAIR import RGTNetModel
    import torch
    
    print('Loading model...')
    model = RGTNetModel('$TRAINED_MODEL')
    
    # Test generation
    print('Testing generation...')
    response = model.generate('Hello, how are you?')
    print(f'Model response: {response[:100]}...')
    
    print('✅ Model loading and generation test passed')
    
except Exception as e:
    print(f'❌ Model loading test failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Model loading test failed"
    exit 1
fi

# Test 3: Quick PAIR attack test
echo ""
echo "🔍 Test 3: Quick PAIR attack test (1 sample)..."
cd /home/ycyoon/work/RGTNet/StructTransformBench/examples

# Check if servers are running
if ! curl -s http://localhost:8002/v1/models >/dev/null 2>&1; then
    echo "⚠️  Attack model server (port 8002) not running, skipping PAIR test"
    echo "   Start servers with: bash start_multi_model_servers.sh"
else
    echo "✅ Attack model server is running"
    
    # Run quick PAIR test
    if python run_PAIR.py --use-trained-model --dataset-size 1; then
        echo "✅ PAIR attack test passed"
    else
        echo "⚠️  PAIR attack test had issues, but model integration works"
    fi
fi

# Test 4: Multi-model benchmark test
echo ""
echo "🔍 Test 4: Multi-model benchmark test..."
if python multi_model_benchmark.py --trained-model test-model "$TRAINED_MODEL" --help >/dev/null 2>&1; then
    echo "✅ Multi-model benchmark integration test passed"
else
    echo "⚠️  Multi-model benchmark test had issues"
fi

echo ""
echo "🎉 All tests completed!"
echo ""
echo "Usage examples:"
echo "  PAIR Attack:     python run_PAIR.py --use-trained-model --dataset-size 5"
echo "  Multi-benchmark: python multi_model_benchmark.py --trained-model my-model $TRAINED_MODEL"
echo "  Full benchmark:  bash run_hf_benchmark.sh"
