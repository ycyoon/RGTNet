# RGTNet with Instruction Fine-tuning and StructTransform Benchmark

This repository contains an implementation of Role-Gated Transformer (RGTNet) with instruction fine-tuning capabilities and integrated support for the StructTransform benchmark dataset.

## Overview

The RGTNet implementation includes:

1. **Role-Aware Transformer**: A transformer model that distinguishes between instruction and data tokens
2. **Instruction Fine-tuning**: Training on ShareGPT, Alpaca, Dolly, and FLAN datasets
3. **Pretrained Model Integration**: Support for LLaMA-2, Mistral, and other pretrained models
4. **StructTransform Benchmark**: Evaluation against structure transformation attacks
5. **Safety Training**: Additional training on adversarial prompts and refusal pairs

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Quick Training (Recommended for Testing)

```bash
python train_instruction.py --quick
```

This will:
- Download instruction datasets (ShareGPT, Alpaca, Dolly, FLAN)
- Train a small model (256 dim, 3 layers) for quick testing
- Save the trained model and results

### 3. Full Training (Recommended Settings)

```bash
python train_instruction.py --full
```

This will:
- Download instruction datasets
- Load pretrained weights (DialoGPT-medium by default)
- Train with paper-recommended settings (512 dim, 6 layers, 3 epochs)
- Save the trained model and results

### 4. Manual Training

```bash
python main.py --download_datasets --epochs 3 --batch_size 16 --lr 5e-5
```

## Dataset Information

The implementation automatically downloads and processes:

### Instruction Datasets
- **ShareGPT**: Human-AI conversations (~100k samples)
- **Alpaca**: Instruction-following dataset (~50k samples)
- **Dolly**: High-quality instruction dataset (~15k samples)
- **FLAN**: Multi-task instruction dataset (~100k samples)

### Safety Dataset
- **Adversarial Prompts**: Harmful requests that should be refused
- **Refusal Pairs**: Appropriate refusal responses
- **Structured Attacks**: JSON, SQL, Cypher, and SymLogix attack patterns

Total dataset size: ~500k samples (following paper recommendations)

## Model Architecture

### Role-Sensitive Embedding
- Applies orthogonal rotation to data tokens
- Preserves instruction token representations
- Helps distinguish between instruction and structured data

### Role-Gated Self-Attention
- Adds learnable bias (δ) to attention scores based on token roles
- Enhances model's ability to prioritize instruction tokens
- Prevents structured data from overwhelming instruction processing

### Pretrained Model Integration
- Supports loading weights from LLaMA-2, Mistral, DialoGPT, and other models
- Automatically adapts pretrained weights to RGT architecture
- Preserves pretrained knowledge while adding role-awareness

## Training Configuration

### Recommended Settings (from paper)
- **Base Model**: LLaMA-2 7B or Mistral 7B
- **Batch Size**: 16
- **Learning Rate**: 5e-5
- **Epochs**: 3
- **Sequence Length**: 512
- **Model Dimension**: 512
- **Attention Heads**: 8
- **Layers**: 6

### Quick Test Settings
- **Model Dimension**: 256
- **Attention Heads**: 4
- **Layers**: 3
- **Epochs**: 2
- **Batch Size**: 8

## Usage Examples

### Training with Custom Datasets

```bash
python main.py \
  --train_file data/my_train.jsonl \
  --val_file data/my_val.jsonl \
  --epochs 3 \
  --batch_size 16 \
  --lr 5e-5 \
  --save_path my_model.pth
```

### Training with Pretrained Model

```bash
python main.py \
  --download_datasets \
  --pretrained_model microsoft/DialoGPT-medium \
  --epochs 3 \
  --save_path rgt_dialogpt.pth
```

### Evaluation Only

```bash
python main.py \
  --eval_only \
  --save_path trained_model.pth \
  --benchmark_dir benchmark \
  --results_file evaluation_results.json
```

## StructTransform Benchmark

The implementation includes full support for the StructTransform benchmark:

### Setup Benchmark
```bash
./setup_benchmark.sh
```

### Run Evaluation
```bash
python main.py --eval_only --benchmark_dir benchmark
```

### Benchmark Metrics
- **Attack Success Rate (ASR)**: Percentage of attacks that succeeded
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **Structure-specific results**: JSON, SQL, Cypher, SymLogix

## File Structure

```
RGTNet/
├── main.py                    # Main training and evaluation script
├── train_instruction.py       # Quick training script
├── requirements.txt          # Python dependencies
├── setup_benchmark.sh        # StructTransform benchmark setup
├── README.md                 # This file
├── config.yaml              # Configuration file
├── data/                    # Downloaded datasets
│   ├── train_instruction.jsonl
│   ├── val_instruction.jsonl
│   └── test_instruction.jsonl
├── benchmark/               # StructTransform benchmark files
│   ├── json_dataset.pkl
│   ├── sql_dataset.pkl
│   ├── cypher_dataset.pkl
│   └── symlogix_dataset.pkl
└── results/                 # Training and evaluation results
    ├── training_results.json
    └── evaluation_results.json
```

## Key Features

1. **Automatic Dataset Download**: Downloads and processes instruction datasets automatically
2. **Pretrained Model Integration**: Supports various pretrained models (LLaMA, Mistral, DialoGPT)
3. **Role-Aware Training**: Distinguishes between instruction and data tokens during training
4. **Safety Training**: Includes adversarial prompts and refusal training
5. **Comprehensive Evaluation**: StructTransform benchmark integration
6. **Flexible Configuration**: Command-line arguments for easy experimentation

## Performance Notes

- **GPU Recommended**: Training is much faster with CUDA support
- **Memory Requirements**: Full model requires ~8GB GPU memory
- **Training Time**: 
  - Quick training: ~30 minutes on GPU
  - Full training: ~2-4 hours on GPU
- **Dataset Size**: ~500k samples, ~2GB storage

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or model dimensions
2. **Dataset Download Errors**: Check internet connection and datasets library
3. **Tokenizer Errors**: Ensure transformers library is up to date
4. **Import Errors**: Install all requirements with `pip install -r requirements.txt`

### Solutions

```bash
# For memory issues
python main.py --batch_size 8 --d_model 256

# For dataset issues
python main.py --download_datasets --max_length 256

# For evaluation only
python main.py --eval_only --save_path existing_model.pth
```

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{rgtnet2024,
  title={Role-Gated Transformer Networks for Instruction-Following Safety},
  author={[Authors]},
  journal={Under Review},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
