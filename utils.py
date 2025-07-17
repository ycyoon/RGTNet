import os
import json
import torch
import random
import numpy as np
from pathlib import Path

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_results(results, filepath):
    """Save results to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {filepath}")

def load_results(filepath):
    """Load results from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def print_model_info(model):
    print("Model architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

def format_time(seconds):
    """Format time in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.2f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"

def check_gpu_memory():
    """Check GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_cached = torch.cuda.memory_reserved(i) / 1024**3
            #print(f"GPU {i}: {memory_allocated:.2f}GB allocated, {memory_cached:.2f}GB cached")
    else:
        print("No GPU available")

def cleanup_cache():
    """Clean up GPU cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared")