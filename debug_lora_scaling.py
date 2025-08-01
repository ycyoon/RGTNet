#!/usr/bin/env python3
"""
Debug and control LoRA scaling values
"""
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

def debug_lora_scaling_detailed():
    """Debug LoRA scaling values in detail"""
    print("🔍 Debugging LoRA scaling values in detail...")
    
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    
    # Test different scaling values
    scaling_tests = [
        {"r": 8, "alpha": 8, "expected_scaling": 1.0},
        {"r": 8, "alpha": 16, "expected_scaling": 2.0},
        {"r": 16, "alpha": 16, "expected_scaling": 1.0},
        {"r": 8, "alpha": 4, "expected_scaling": 0.5},
    ]
    
    for test in scaling_tests:
        print(f"\n📊 Testing scaling: r={test['r']}, alpha={test['alpha']}, expected={test['expected_scaling']}")
        
        try:
            # Create LoRA config
            lora_config = LoraConfig(
                r=test['r'],
                lora_alpha=test['alpha'],
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            print(f"   Actual scaling from config: {lora_config.lora_alpha / lora_config.r}")
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            
            # Apply LoRA
            model = get_peft_model(model, lora_config)
            
            # Find and check LoRA modules
            for name, module in model.named_modules():
                if 'q_proj' in name and 'lora_A' in name and 'default' in name:
                    print(f"   Found LoRA module: {name}")
                    print(f"   LoRA A shape: {module.weight.shape}")
                    
                    # Check the corresponding lora_B module
                    b_name = name.replace('lora_A', 'lora_B')
                    if b_name in dict(model.named_modules()):
                        b_module = dict(model.named_modules())[b_name]
                        print(f"   LoRA B shape: {b_module.weight.shape}")
                        
                        # Calculate expected output dimension
                        input_dim = module.weight.shape[1]  # 3072
                        rank = module.weight.shape[0]       # r (8)
                        output_dim = b_module.weight.shape[0]  # 3072
                        
                        print(f"   Input dim: {input_dim}")
                        print(f"   Rank: {rank}")
                        print(f"   Output dim: {output_dim}")
                        print(f"   Expected intermediate: {rank}")
                        
                        # Check if there's a scaling issue
                        if input_dim != output_dim:
                            print(f"   ⚠️  Dimension mismatch: {input_dim} != {output_dim}")
                        
                        break
            
            # Test forward pass
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            input_text = "Hello"
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=16)
            
            # Move to GPU
            for key in inputs:
                inputs[key] = inputs[key].cuda()
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            print(f"   ✅ Forward pass successful!")
            print(f"   Output shape: {outputs['logits'].shape}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()

def test_custom_scaling():
    """Test with custom scaling values"""
    print("\n🔧 Testing custom scaling values...")
    
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    
    # Try with very small scaling
    try:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=1,  # Very small alpha for scaling=0.125
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        print(f"Testing with scaling = {lora_config.lora_alpha / lora_config.r}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        model = get_peft_model(model, lora_config)
        
        # Test forward pass
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        input_text = "Hello"
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=16)
        
        for key in inputs:
            inputs[key] = inputs[key].cuda()
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"✅ Custom scaling test successful!")
        print(f"Output shape: {outputs['logits'].shape}")
        
    except Exception as e:
        print(f"❌ Custom scaling test failed: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("🔬 LoRA Scaling Debug and Control")
    print("=" * 60)
    
    debug_lora_scaling_detailed()
    test_custom_scaling()
    
    print("\n" + "=" * 60)
    print("✅ Debug completed!")
    print("=" * 60) 