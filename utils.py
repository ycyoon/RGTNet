import os
import json
import torch
import random
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

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

def print_model_info(model, head):
    """Print model information"""
    model_params = count_parameters(model)
    head_params = count_parameters(head) if head is not None else 0
    total_params = model_params + head_params
    
    print("\n" + "="*50)
    print("MODEL INFORMATION")
    print("="*50)
    print(f"Model parameters: {model_params:,}")
    if head is not None:
        print(f"Head parameters: {head_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    print("="*50)

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
            print(f"GPU {i}: {memory_allocated:.2f}GB allocated, {memory_cached:.2f}GB cached")
    else:
        print("No GPU available")

def cleanup_cache():
    """Clean up GPU cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared")

def setup_logging(log_dir='logs', log_level=logging.INFO, is_main_process=True):
    """Setup logging configuration to save logs in the logs folder"""
    if not is_main_process:
        return None
    
    # Create logs directory
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamp for unique log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"rgtnet_training_{timestamp}.log"
    
    # Setup logging configuration
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    logger = logging.getLogger('RGTNet')
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger

def log_training_progress(logger, epoch, total_epochs, train_loss, val_loss, lr, epoch_time, best_val_loss=None):
    """Log training progress"""
    if logger is None:
        return
    
    message = (f"Epoch {epoch}/{total_epochs} - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"LR: {lr:.2e}, Time: {format_time(epoch_time)}")
    
    if best_val_loss is not None:
        message += f", Best Val Loss: {best_val_loss:.4f}"
    
    logger.info(message)

def log_final_performance(logger, results, model_path):
    """Log final performance summary"""
    if logger is None:
        return
    
    logger.info("=" * 80)
    logger.info("FINAL PERFORMANCE SUMMARY")
    logger.info("=" * 80)
    
    logger.info(f"Model saved to: {model_path}")
    
    # Safe access to results with None check
    if results is not None:
        logger.info(f"Best validation loss: {results.get('best_val_loss', 'N/A'):.4f}")
        logger.info(f"Final training loss: {results.get('final_train_loss', 'N/A'):.4f}")
        logger.info(f"Final validation loss: {results.get('final_val_loss', 'N/A'):.4f}")
        logger.info(f"Total epochs completed: {results.get('total_epochs', 'N/A')}")
        
        # Log benchmark results if available
        training_stats = results.get('training_stats', {})
        benchmark_results = training_stats.get('benchmark_results', [])
        
        if benchmark_results:
            logger.info("\nBenchmark Progress Summary:")
            logger.info("-" * 40)
            for result in benchmark_results:
                epoch = result.get('epoch', 'N/A')
                asr = result.get('overall_asr', 'N/A')
                if asr != 'N/A':
                    logger.info(f"Epoch {epoch}: ASR = {asr:.4f}")
                else:
                    logger.info(f"Epoch {epoch}: ASR = {asr}")
        
        # Log training statistics
        train_losses = training_stats.get('train_losses', [])
        val_losses = training_stats.get('val_losses', [])
        
        if train_losses and val_losses:
            logger.info(f"\nTraining Statistics:")
            logger.info(f"Initial train loss: {train_losses[0]:.4f}")
            logger.info(f"Final train loss: {train_losses[-1]:.4f}")
            logger.info(f"Train loss improvement: {train_losses[0] - train_losses[-1]:.4f}")
            logger.info(f"Initial val loss: {val_losses[0]:.4f}")
            logger.info(f"Final val loss: {val_losses[-1]:.4f}")
            logger.info(f"Val loss improvement: {val_losses[0] - val_losses[-1]:.4f}")
    else:
        logger.info("No training results available")
    
    logger.info("=" * 80)

def log_evaluation_results(logger, eval_results):
    """Log detailed evaluation results"""
    if logger is None:
        return
    
    logger.info("=" * 80)
    logger.info("DETAILED EVALUATION RESULTS")
    logger.info("=" * 80)
    
    # Log overall metrics
    overall_metrics = eval_results.get('overall_metrics', {})
    for metric, value in overall_metrics.items():
        if isinstance(value, float):
            logger.info(f"{metric}: {value:.4f}")
        else:
            logger.info(f"{metric}: {value}")
    
    # Log per-dataset results if available
    dataset_results = eval_results.get('dataset_results', {})
    if dataset_results:
        logger.info("\nPer-Dataset Results:")
        logger.info("-" * 40)
        for dataset, metrics in dataset_results.items():
            logger.info(f"\n{dataset}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    logger.info(f"  {metric}: {value:.4f}")
                else:
                    logger.info(f"  {metric}: {value}")
    
    logger.info("=" * 80)