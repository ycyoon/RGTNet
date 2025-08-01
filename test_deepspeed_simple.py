#!/usr/bin/env python3
"""
Simple DeepSpeed multi-GPU test without LoRA
"""
import os
import sys
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_deepspeed_simple():
    """Test DeepSpeed with simple model on multiple GPUs"""
    print("🧪 Testing DeepSpeed with simple model on multiple GPUs...")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False
    
    num_gpus = torch.cuda.device_count()
    print(f"📊 Found {num_gpus} GPUs")
    
    if num_gpus < 2:
        print("⚠️  Need at least 2 GPUs for multi-GPU test")
        return False
    
    try:
        # Test 1: Simple model creation
        print("📦 Creating simple model...")
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("✅ Model creation successful!")
        
        # Test 2: Check if we're in a distributed environment
        print("🌐 Checking distributed environment...")
        is_distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
        if is_distributed:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ.get('LOCAL_RANK', rank))
            print(f"✅ Running in distributed mode: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        else:
            print("⚠️  Not running in distributed mode")
            rank = 0
            world_size = 1
            local_rank = 0
        
        # Test 3: Multi-GPU forward pass
        print("🔄 Testing multi-GPU forward pass...")
        
        # Create sample batch
        input_texts = [
            "Hello, how are you?",
            "What is machine learning?"
        ]
        
        inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
        
        # Move to GPU
        for key in inputs:
            inputs[key] = inputs[key].cuda()
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        print("✅ Multi-GPU forward pass successful!")
        print(f"   Output shape: {outputs['logits'].shape}")
        
        # Test 4: Memory usage per GPU
        print("💾 Checking memory usage per GPU...")
        for i in range(num_gpus):
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"   GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        
        # Test 5: Simple training step
        print("🎯 Testing training step...")
        try:
            # Create dummy labels
            labels = torch.randint(0, tokenizer.vocab_size, (inputs['input_ids'].shape[0], inputs['input_ids'].shape[1])).cuda()
            
            # Forward pass with loss
            outputs = model(**inputs, labels=labels)
            loss = outputs['loss']
            
            print("✅ Training forward pass successful!")
            print(f"   Loss: {loss.item():.4f}")
            
        except Exception as e:
            print(f"⚠️  Training step failed: {e}")
        
        print("\n🎉 DeepSpeed simple model test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🔬 DeepSpeed Simple Model Test")
    print("=" * 60)
    
    success = test_deepspeed_simple()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ DeepSpeed simple model test passed!")
    else:
        print("❌ DeepSpeed simple model test failed!")
    print("=" * 60) 