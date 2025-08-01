#!/usr/bin/env python3
"""
Quick test script to verify LoRA functionality
"""
import os
import torch
import argparse
from transformers import AutoTokenizer
from model_hybrid import create_hybrid_model

def test_lora():
    """Test LoRA functionality"""
    print("🧪 Testing LoRA functionality...")
    
    # Test arguments
    args = argparse.Namespace()
    args.pretrained_model_name = "meta-llama/Llama-3.2-3B-Instruct"
    args.use_lora = True
    args.enable_role_adapters = True
    args.lora_r = 8
    args.lora_alpha = 16
    args.lora_dropout = 0.05
    args.use_quantization = False
    
    try:
        # Test 1: Model creation
        print("📦 Creating hybrid model with LoRA...")
        model, tokenizer = create_hybrid_model(args)
        print("✅ Model creation successful!")
        
        # Test 2: Check if LoRA is applied
        print("🔍 Checking LoRA application...")
        lora_applied = False
        for name, module in model.named_modules():
            if 'lora' in name.lower():
                lora_applied = True
                print(f"   Found LoRA module: {name}")
                break
        
        if lora_applied:
            print("✅ LoRA adapters found!")
        else:
            print("⚠️  No LoRA adapters found - checking PEFT modules...")
            for name, module in model.named_modules():
                if hasattr(module, 'peft_config'):
                    lora_applied = True
                    print(f"   Found PEFT module: {name}")
                    break
        
        if not lora_applied:
            print("❌ LoRA not properly applied!")
            return False
        
        # Test 3: Simple forward pass
        print("🚀 Testing forward pass...")
        input_text = "Hello, how are you?"
        inputs = tokenizer(input_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print("✅ Forward pass successful!")
        print(f"   Output shape: {outputs['logits'].shape}")
        
        # Test 4: Check model parameters
        print("📊 Checking model parameters...")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Trainable ratio: {trainable_params/total_params*100:.2f}%")
        
        # Test 5: Memory usage
        print("💾 Checking memory usage...")
        if torch.cuda.is_available():
            print(f"   GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            print(f"   GPU memory cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
        
        print("\n🎉 All tests passed! LoRA is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_without_lora():
    """Test without LoRA for comparison"""
    print("\n🧪 Testing without LoRA for comparison...")
    
    args = argparse.Namespace()
    args.pretrained_model_name = "meta-llama/Llama-3.2-3B-Instruct"
    args.use_lora = False
    args.enable_role_adapters = False
    args.use_quantization = False
    
    try:
        model, tokenizer = create_hybrid_model(args)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   Total parameters (no LoRA): {total_params:,}")
        print(f"   Trainable parameters (no LoRA): {trainable_params:,}")
        print(f"   Trainable ratio (no LoRA): {trainable_params/total_params*100:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Test without LoRA failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("🔬 RGTNet LoRA Quick Test")
    print("=" * 50)
    
    # Test with LoRA
    success = test_lora()
    
    if success:
        # Test without LoRA for comparison
        test_without_lora()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ LoRA test completed successfully!")
    else:
        print("❌ LoRA test failed!")
    print("=" * 50) 