#!/usr/bin/env python3
"""
RGTNet Main Entry Point
"""

import sys
import os
import time
from transformers import AutoTokenizer

# Import our modules
from config import setup_args, setup_environment, get_device, create_directories
from model import create_model
from data_loader import download_instruction_datasets, create_data_loaders, load_data_from_files
from trainer import train_model
from evaluator import evaluate_model_detailed, save_evaluation_results, print_evaluation_summary
from utils import set_seed, print_model_info, format_time, check_gpu_memory, save_results

def main():
    """Main execution function"""
    start_time = time.time()
    
    # Setup
    args = setup_args()
    setup_environment()
    set_seed(42)
    
    # Get device and create directories
    device = get_device(args)
    create_directories(args)
    
    # Print GPU info
    check_gpu_memory()
    
    # Initialize tokenizer
    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model
    print("Creating model...")
    model, head = create_model(args, tokenizer)
    print_model_info(model, head)
    
    # Data preparation
    train_data, val_data = None, None
    
    if args.download_datasets:
        print("Downloading datasets...")
        train_data, val_data = download_instruction_datasets()
    else:
        print("Loading data from files...")
        train_data, val_data = load_data_from_files(args.train_file, args.val_file, tokenizer, args)
    
    if not train_data or not val_data:
        print("Error: No training or validation data available")
        sys.exit(1)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(train_data, val_data, tokenizer, args)
    
    # Training
    if not args.eval_only:
        print("Starting training...")
        training_results = train_model(model, head, train_loader, val_loader, device, args)
        
        # Save training results
        training_results_file = args.results_file.replace('.json', '_training.json')
        save_results(training_results, training_results_file)
        
        print(f"Training completed in {format_time(time.time() - start_time)}")
    
    # Evaluation
    if not args.train_only:
        print("Starting evaluation...")
        eval_results = evaluate_model_detailed(model, head, val_loader, device, args)
        
        # Print and save evaluation results
        print_evaluation_summary(eval_results)
        eval_results_file = args.results_file.replace('.json', '_evaluation.json')
        save_evaluation_results(eval_results, eval_results_file)
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {format_time(total_time)}")
    print("✅ Process completed successfully!")

if __name__ == '__main__':
    main()
