import argparse
import os
from pathlib import Path

def setup_args():
    """Setup command line arguments with improved checkpoint management"""
    parser = argparse.ArgumentParser(description='RGTNet Training with DeepSpeed')
    
    # Model parameters - removed d_model, nhead, num_layers as they will be auto-detected from pretrained model
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--max_seq_len', type=int, default=8192, help='Maximum sequence length')
    
    # Hybrid model parameters
    parser.add_argument('--enable_role_adapters', action='store_true', default=False, help='Enable RGTNet role-aware adapters')
    parser.add_argument('--use_explicit_bias', action='store_true', help='Use explicit role-gated attention with bias')
    parser.add_argument('--bias_delta', type=float, default=1.0, help='Bias delta parameter for role-gated attention')
    parser.add_argument('--use_quantization', action='store_true', help='Use 4-bit quantization for memory efficiency')
    parser.add_argument('--use_lora', action='store_true', help='Apply LoRA adapters to the base model')
    
    # PEFT/LoRA parameters (when use_lora is enabled)
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank (dimension of adapter)')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA scaling factor')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout rate')
    
    # Checkpoint saving options
    parser.add_argument('--save_deepspeed_checkpoint', action='store_true', help='Save DeepSpeed checkpoints (for training resumption)')
    parser.add_argument('--save_merged_model', action='store_true', help='Save merged model (converted from DeepSpeed checkpoint)')
    parser.add_argument('--lora_only', action='store_true', default=False, help='Save only LoRA adapters (default: False)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
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
    # 토크나이저는 사전 훈련된 모델의 이름을 사용하여 자동으로 로드
    parser.add_argument('--pretrained_model_name', type=str, default=None, help='Pretrained model name for initialization')
    parser.add_argument('--benchmark_dir', type=str, default=None, help='Directory containing benchmark datasets')
    
    # Output parameters
    parser.add_argument('--save_path', type=str, default='model.pth', help='Model save path')
    parser.add_argument('--results_file', type=str, default='results.json', help='Results file path')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    
    # Execution parameters
    parser.add_argument('--train_only', action='store_true', help='Only run training')
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    # DeepSpeed parameters
    parser.add_argument('--deepspeed', action='store_true', help='Use DeepSpeed for distributed training')
    parser.add_argument('--deepspeed_config', type=str, default='ds_config.json', help='DeepSpeed configuration file')
    
    # Distributed training parameters
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    
    # Checkpoint management
    parser.add_argument('--auto_merge_checkpoint', action='store_true', default=True,
                       help='Automatically merge DeepSpeed checkpoints after each save')
    parser.add_argument('--unified_checkpoint_dir', type=str, default=None,
                       help='Unified directory for all checkpoints')
    
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


def merge_deepspeed_checkpoint(checkpoint_dir, output_dir, logger=None):
    """Merge DeepSpeed checkpoint using zero_to_fp32.py"""
    import os
    import subprocess
    import sys
    
    try:
        # Check if zero_to_fp32.py exists in the checkpoint directory
        zero_script = os.path.join(checkpoint_dir, "zero_to_fp32.py")
        if not os.path.exists(zero_script):
            print(f"⚠️  zero_to_fp32.py not found in {checkpoint_dir}")
            return False
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Run zero_to_fp32.py
        cmd = [
            sys.executable, zero_script, 
            checkpoint_dir, output_dir, 
            "--safe_serialization"
        ]
        
        print(f"🔄 Merging DeepSpeed checkpoint...")
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=checkpoint_dir)
        
        if result.returncode == 0:
            print(f"✅ Successfully merged checkpoint to {output_dir}")
            if logger:
                logger.info(f"DeepSpeed checkpoint merged to {output_dir}")
            return True
        else:
            print(f"❌ Failed to merge checkpoint:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            if logger:
                logger.error(f"Failed to merge DeepSpeed checkpoint: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error during checkpoint merging: {e}")
        if logger:
            logger.error(f"Error during checkpoint merging: {e}")
        return False