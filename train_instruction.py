#!/usr/bin/env python3
"""
Quick training example for RGTNet with instruction datasets
"""

import subprocess
import os
import sys
import torch

def run_quick_training():
    """Run quick training with instruction datasets"""
    
    print("=== RGTNet Quick Training Example ===")
    
    # Step 1: Install dependencies
    print("1. Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✓ Dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not install all dependencies: {e}")
    
    # Step 2: Download and train
    print("\n2. Downloading datasets and training...")
    train_cmd = [
        sys.executable, "main.py",
        "--download_datasets",
        "--epochs", "2",
        "--batch_size", "8",
        "--lr", "1e-4",
        "--pretrained_model_name", "bert-base-uncased",  # Use pretrained model, dimensions auto-detected
        "--max_seq_len", "256",
        "--save_path", "rgt_instruction_model.pth",
        "--results_file", "instruction_training_results.json",
        "--device", "cuda" if torch.cuda.is_available() else "cpu"
    ]
    
    try:
        if torch.cuda.is_available():
            print("Using GPU for training")
        else:
            print("Using CPU for training")
    except:
        print("PyTorch not available")
    
    try:
        subprocess.run(train_cmd, check=True)
        print("✓ Training completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Training failed: {e}")
        return False
    
    # Step 3: Quick evaluation
    print("\n3. Running quick evaluation...")
    eval_cmd = [
        sys.executable, "main.py",
        "--eval_only",
        "--save_path", "rgt_instruction_model.pth",
        "--results_file", "instruction_eval_results.json"
    ]
    
    try:
        subprocess.run(eval_cmd, check=True)
        print("✓ Evaluation completed")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Evaluation failed: {e}")
        print("This is normal if benchmark files are not available")
    
    print("\n=== Training Complete ===")
    print("Generated files:")
    print("- rgt_instruction_model.pth: Trained model")
    print("- instruction_training_results.json: Training metrics")
    print("- data/: Downloaded instruction datasets")
    
    return True

def run_full_training():
    """Run full training with recommended settings"""
    
    print("=== RGTNet Full Training (Recommended Settings) ===")
    
    # Install dependencies
    print("1. Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✓ Dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not install all dependencies: {e}")
    
    # Full training with recommended parameters
    print("\n2. Starting full training (this may take a while)...")
    train_cmd = [
        sys.executable, "main.py",
        "--download_datasets",
        "--epochs", "3",
        "--batch_size", "16",
        "--lr", "5e-5",
        "--pretrained_model_name", "microsoft/DialoGPT-medium",  # Model dimensions auto-detected
        "--max_seq_len", "512",
        "--save_path", "rgt_full_model.pth",
        "--results_file", "full_training_results.json",
        "--device", "cuda" if torch.cuda.is_available() else "cpu"
    ]
    
    try:
        import torch
        if torch.cuda.is_available():
            print("Using GPU for training")
        else:
            print("Using CPU for training (this will be slower)")
    except ImportError:
        print("PyTorch not installed")
    
    try:
        subprocess.run(train_cmd, check=True)
        print("✓ Full training completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Training failed: {e}")
        return False
    
    # Evaluation
    print("\n3. Running evaluation...")
    eval_cmd = [
        sys.executable, "main.py",
        "--eval_only",
        "--save_path", "rgt_full_model.pth",
        "--results_file", "full_eval_results.json"
    ]
    
    try:
        subprocess.run(eval_cmd, check=True)
        print("✓ Evaluation completed")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Evaluation failed: {e}")
    
    print("\n=== Full Training Complete ===")
    print("Generated files:")
    print("- rgt_full_model.pth: Trained model")
    print("- full_training_results.json: Training metrics")
    print("- data/: Downloaded instruction datasets")
    
    return True

def show_usage():
    """Show usage information"""
    print("RGTNet Training Script")
    print("Usage:")
    print("  python train_instruction.py --quick    # Quick training (small model)")
    print("  python train_instruction.py --full     # Full training (recommended)")
    print("  python train_instruction.py --help     # Show this help")
    print()
    print("Quick training: Small model for testing (fast)")
    print("Full training: Recommended settings from paper (slower but better)")

def main():
    if len(sys.argv) < 2:
        show_usage()
        return
    
    if sys.argv[1] == "--quick":
        success = run_quick_training()
    elif sys.argv[1] == "--full":
        success = run_full_training()
    elif sys.argv[1] == "--help":
        show_usage()
        return
    else:
        print(f"Unknown option: {sys.argv[1]}")
        show_usage()
        return
    
    if success:
        print("\n🎉 Training completed successfully!")
    else:
        print("\n❌ Training failed. Check the error messages above.")

if __name__ == "__main__":
    main()
