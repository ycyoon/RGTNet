import argparse
import os
from pathlib import Path

def setup_args():
    """Setup command line arguments"""
    parser = argparse.ArgumentParser(description='RGTNet Training')
    
    # Model parameters
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--max_seq_len', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--bias_delta', type=float, default=1.0, help='Role-gated attention bias delta')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Use gradient checkpointing to save memory')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients')
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision training')
    parser.add_argument('--enable_benchmark', action='store_true', help='Enable benchmarking during training')
    parser.add_argument('--benchmark_freq', type=int, default=10, help='Frequency of benchmark logging')
    parser.add_argument('--max_iters', type=int, default=None, help='Maximum number of training iterations (overrides epochs if set)')
    
    # Data parameters
    parser.add_argument('--download_datasets', action='store_true', help='Download datasets')
    parser.add_argument('--train_file', type=str, help='Training data file')
    parser.add_argument('--val_file', type=str, help='Validation data file')
    parser.add_argument('--tokenizer_name', type=str, default='bert-base-uncased', help='Tokenizer name')
    parser.add_argument('--pretrained_model_name', type=str, default=None, help='Pretrained model name for initialization')
    parser.add_argument('--benchmark_dir', type=str, default=None, help='Directory containing benchmark datasets')
    
    # Output parameters
    parser.add_argument('--save_path', type=str, default='model.pth', help='Model save path')
    parser.add_argument('--results_file', type=str, default='results.json', help='Results file path')
    
    # Execution parameters
    parser.add_argument('--train_only', action='store_true', help='Only run training')
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    return parser.parse_args()

def setup_environment():
    """Setup environment variables"""
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
def get_device(args):
    """Get the appropriate device"""
    import torch
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    return device

def create_directories(args):
    """Create necessary directories"""
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    # Create other commonly used directories
    Path('models').mkdir(exist_ok=True)
    Path('evaluation_results').mkdir(exist_ok=True)