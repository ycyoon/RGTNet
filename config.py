import argparse
import os
from pathlib import Path

def get_llm_config_from_pretrained(pretrained_model_name):
    from transformers import AutoConfig, AutoModelForCausalLM
    try:
        config = AutoConfig.from_pretrained(pretrained_model_name, trust_remote_code=True)
        d_model = getattr(config, 'hidden_size', getattr(config, 'n_embd', None))
        nhead = getattr(config, 'num_attention_heads', getattr(config, 'n_head', None))
        num_layers = getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', None))
        max_seq_len = getattr(config, 'max_position_embeddings', getattr(config, 'n_positions', 512))
        if None not in (d_model, nhead, num_layers, max_seq_len):
            return d_model, nhead, num_layers, max_seq_len
        print(f"[WARN] Config missing values: d_model={d_model}, nhead={nhead}, num_layers={num_layers}, max_seq_len={max_seq_len}. Trying to load model for fallback...")
    except Exception as e:
        print(f"[WARN] Could not load config for {pretrained_model_name}: {e}. Trying to load model for fallback...")
    # Fallback: load model and extract config
    try:
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name, trust_remote_code=True)
        config = model.config
        d_model = getattr(config, 'hidden_size', getattr(config, 'n_embd', None))
        nhead = getattr(config, 'num_attention_heads', getattr(config, 'n_head', None))
        num_layers = getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', None))
        max_seq_len = getattr(config, 'max_position_embeddings', getattr(config, 'n_positions', 512))
        if None in (d_model, nhead, num_layers, max_seq_len):
            raise ValueError(f"Could not extract all config values from model: d_model={d_model}, nhead={nhead}, num_layers={num_layers}, max_seq_len={max_seq_len}")
        return d_model, nhead, num_layers, max_seq_len
    except Exception as e:
        print(f"[ERROR] Could not load model or extract config for {pretrained_model_name}: {e}")
        exit(1)

def setup_args():
    """Setup command line arguments"""
    parser = argparse.ArgumentParser(description='RGTNet Training')
    
    # Model parameters
    parser.add_argument('--pretrained_model_name', type=str, default='meta-llama/Meta-Llama-3-10B', help='Pretrained model name for RoleAwareTransformerDecoder backbone (e.g., meta-llama/Meta-Llama-3-10B)')
    parser.add_argument('--use_dummy_model', action='store_true', help='Use a lightweight dummy model for quick testing, ignoring pretrained model settings.')
    parser.add_argument('--d_model', type=int, default=None, help='Model dimension (auto from pretrained if not set)')
    parser.add_argument('--nhead', type=int, default=None, help='Number of attention heads (auto from pretrained if not set)')
    parser.add_argument('--num_layers', type=int, default=None, help='Number of transformer layers (auto from pretrained if not set)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--max_seq_len', type=int, default=None, help='Maximum sequence length (auto from pretrained if not set)')
    parser.add_argument('--bias_delta', type=float, default=1.0, help='Role-gated attention bias delta')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients before updating weights.')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--use_amp', action='store_true', help='Use Automatic Mixed Precision (AMP) for training.')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Enable gradient checkpointing')
    
    # Benchmark parameters
    parser.add_argument('--benchmark_freq', type=int, default=5, help='Benchmark evaluation frequency (every N epochs)')
    parser.add_argument('--enable_benchmark', action='store_true', help='Enable benchmark evaluation during training')
    parser.add_argument('--benchmark_dir', type=str, default='StructTransformBench/benchmark', help='Benchmark directory')
    
    # Data parameters
    parser.add_argument('--download_datasets', action='store_true', help='Download instruction-following datasets')
    parser.add_argument('--train_file', type=str, default='data/train.json', help='Path to the training data file')
    parser.add_argument('--val_file', type=str, help='Validation data file')
    parser.add_argument('--tokenizer_name', type=str, default='bert-base-uncased', help='Tokenizer name')
    
    # Output parameters
    parser.add_argument('--save_path', type=str, default='model.pth', help='Model save path')
    parser.add_argument('--results_file', type=str, default='results.json', help='Results file path')
    
    # Execution parameters
    parser.add_argument('--train_only', action='store_true', help='Only run training')
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    args = parser.parse_args()
    # pretrained_model_name에서 자동 세팅
    if not args.use_dummy_model and (args.d_model is None or args.nhead is None or args.num_layers is None or args.max_seq_len is None):
        d_model, nhead, num_layers, max_seq_len = get_llm_config_from_pretrained(args.pretrained_model_name)
        if args.d_model is None and d_model is not None:
            args.d_model = d_model
        if args.nhead is None and nhead is not None:
            args.nhead = nhead
        if args.num_layers is None and num_layers is not None:
            args.num_layers = num_layers
        if args.max_seq_len is None and max_seq_len is not None:
            args.max_seq_len = max_seq_len
    return args

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