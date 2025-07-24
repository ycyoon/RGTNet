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

### 1. Setup Visual Studio Code with GitHub Pro+ (Recommended)

This repository is optimized for Visual Studio Code with GitHub Pro+ features including GitHub Copilot.

#### Prerequisites
- Visual Studio Code installed
- GitHub Pro+ subscription with Copilot access
- Git configured with your GitHub account

#### Setup Steps
1. **Open in VS Code**:
   ```bash
   code RGTNet.code-workspace
   ```
   Or open the folder directly in VS Code.

2. **Install recommended extensions**:
   VS Code will prompt to install recommended extensions. Accept to install:
   - GitHub Copilot & Copilot Chat (requires GitHub Pro+)
   - Python extension pack
   - GitLens for advanced Git features
   - Jupyter for notebook support

3. **Sign in to GitHub**:
   - Press `Ctrl+Shift+P` and run "GitHub: Sign in"
   - Authenticate with your GitHub Pro+ account
   - Verify Copilot is activated in the status bar

4. **Setup Python environment**:
   - Use `F1` → "Python: Select Interpreter" to choose your Python interpreter
   - Or run the "Setup Development Environment" task from the Command Palette

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Quick Training (Recommended for Testing)

```bash
python train_instruction.py --quick
```

This will:
- Download instruction datasets (ShareGPT, Alpaca, Dolly, FLAN)
- Train a small model (256 dim, 3 layers) for quick testing
- Save the trained model and results

### 4. Full Training (Recommended Settings)

```bash
python train_instruction.py --full
```

This will:
- Download instruction datasets
- Load pretrained weights (DialoGPT-medium by default)
- Train with paper-recommended settings (512 dim, 6 layers, 3 epochs)
- Save the trained model and results

### 5. Manual Training

```bash
python main.py --download_datasets --epochs 3 --batch_size 16 --lr 5e-5
```

## Visual Studio Code Integration

This repository includes comprehensive VS Code configuration to maximize your productivity with GitHub Pro+ features.

### GitHub Pro+ Features Available

1. **GitHub Copilot Integration**:
   - Code completion and suggestions for Python ML code
   - Copilot Chat for explaining complex transformer architectures
   - Inline suggestions optimized for PyTorch and ML patterns

2. **Advanced Git Features**:
   - GitLens for code history and blame annotations
   - GitHub Pull Request integration
   - Repository insights and analytics

3. **Debugging Configuration**:
   - Pre-configured launch configurations for training scripts
   - Debug profiles for quick testing, full training, and evaluation
   - Integrated terminal with proper environment setup

### VS Code Tasks Available

Access these via `Ctrl+Shift+P` → "Tasks: Run Task":

- **Install Dependencies**: Automatically install requirements.txt
- **Quick Train**: Run quick training with optimized settings
- **Full Train**: Run full training with paper settings
- **Setup Benchmark**: Initialize StructTransform benchmark
- **Format Code**: Apply Black formatting
- **Lint Code**: Run flake8 linting
- **Clean Cache**: Remove Python cache files

### Workspace Features

- **File Nesting**: Organized view of related files (*.py with *.pyc, etc.)
- **Smart Search**: Excludes data/, models/, and cache directories
- **Python Integration**: Auto-detection of virtual environments
- **Jupyter Support**: Built-in notebook support for experimentation

### Troubleshooting VS Code Setup

1. **GitHub Copilot not working**:
   ```bash
   # Check authentication
   # In VS Code: Ctrl+Shift+P → "GitHub: Sign in"
   # Verify Pro+ subscription in GitHub settings
   ```

2. **Python interpreter not found**:
   ```bash
   # In VS Code: Ctrl+Shift+P → "Python: Select Interpreter"
   # Choose the correct Python path (e.g., /usr/bin/python or venv/bin/python)
   ```

3. **Extensions not installing**:
   ```bash
   # Manually install key extensions:
   code --install-extension github.copilot
   code --install-extension ms-python.python
   code --install-extension eamodio.gitlens
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
├── RGTNet.code-workspace     # VS Code workspace configuration
├── .vscode/                  # VS Code configuration directory
│   ├── settings.json         # Python and GitHub Pro+ settings
│   ├── launch.json          # Debugging configurations
│   ├── tasks.json           # Development tasks
│   └── extensions.json      # Recommended extensions
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

## 한국어 사용자를 위한 VS Code 설정 가이드

GitHub Pro+ 구독을 Visual Studio Code에서 최대한 활용하기 위한 설정이 포함되어 있습니다.

### GitHub Pro+ 기능 활용하기

1. **GitHub Copilot 설정**:
   - `Ctrl+Shift+P` → "GitHub: Sign in"으로 GitHub 계정에 로그인
   - 상태바에서 Copilot이 활성화되었는지 확인
   - Pro+ 구독이 활성화되어 있는지 GitHub 설정에서 확인

2. **VS Code에서 프로젝트 열기**:
   ```bash
   code RGTNet.code-workspace
   ```

3. **권장 확장 프로그램 설치**:
   - GitHub Copilot (Pro+ 구독 필요)
   - Python 확장 팩
   - GitLens
   - Jupyter

### 문제 해결

- **Copilot이 작동하지 않는 경우**: GitHub 계정 재로그인 후 Pro+ 구독 상태 확인
- **Python 인터프리터를 찾을 수 없는 경우**: `F1` → "Python: Select Interpreter"에서 올바른 Python 경로 선택

## License

This project is licensed under the MIT License - see the LICENSE file for details.
