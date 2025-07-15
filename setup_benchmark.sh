#!/bin/bash

# Script to download and set up StructTransform benchmark

echo "Setting up StructTransform benchmark..."

# Create benchmark directory
mkdir -p benchmark

# Clone the StructTransform benchmark repository
if [ ! -d "StructTransformBench" ]; then
    echo "Cloning StructTransform benchmark repository..."
    git clone https://github.com/qcri/StructTransformBench.git
    cd StructTransformBench
    
    # Install requirements
    pip install -r requirements.txt
    pip install -e .
    
    cd ..
fi

# Copy benchmark files to our benchmark directory
echo "Copying benchmark files..."
if [ -d "StructTransformBench/benchmark" ]; then
    cp -r StructTransformBench/benchmark/* benchmark/
    echo "Benchmark files copied successfully!"
else
    echo "Warning: Benchmark files not found in StructTransformBench repository"
    echo "You may need to download them manually from the repository"
fi

# Create a simple test to check if files exist
echo "Checking for benchmark files..."
for file in json_dataset.pkl sql_dataset.pkl cypher_dataset.pkl symlogix_dataset.pkl; do
    if [ -f "benchmark/$file" ]; then
        echo "✓ Found: benchmark/$file"
    else
        echo "✗ Missing: benchmark/$file"
    fi
done

echo "Setup complete!"
echo ""
echo "To run the benchmark evaluation:"
echo "python main.py --benchmark_dir benchmark"
echo ""
echo "Or with individual files:"
echo "python main.py --json_dataset benchmark/json_dataset.pkl --sql_dataset benchmark/sql_dataset.pkl"
