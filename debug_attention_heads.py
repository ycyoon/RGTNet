#!/usr/bin/env python3
"""
Debug attention heads distribution across GPUs
"""
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model

def debug_attention_heads():
    """Debug attention heads distribution"""
    print("🔍 Debugging attention heads distribution...")
    
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    
    # Load config
    config = AutoConfig.from_pretrained(model_name)
    print(f"Model config:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num attention heads: {config.num_attention_heads}")
    print(f"  Head dimension: {config.hidden_size // config.num_attention_heads}")
    
    # Calculate expected dimensions
    head_dim = config.hidden_size // config.num_attention_heads
    print(f"  Expected head dimension: {head_dim}")
    
    # Test with different GPU counts
    for num_gpus in [1, 2, 4]:
        print(f"\n📊 Testing with {num_gpus} GPUs...")
        
        try:
            # Set device map for multiple GPUs
            if num_gpus > 1:
                device_map = "auto"
            else:
                device_map = "cuda:0"
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device_map,
                trust_remote_code=True,
            )
            
            # Check first layer dimensions
            first_layer = model.model.layers[0]
            q_proj_shape = first_layer.self_attn.q_proj.weight.shape
            k_proj_shape = first_layer.self_attn.k_proj.weight.shape
            v_proj_shape = first_layer.self_attn.v_proj.weight.shape
            o_proj_shape = first_layer.self_attn.o_proj.weight.shape
            
            print(f"  q_proj shape: {q_proj_shape}")
            print(f"  k_proj shape: {k_proj_shape}")
            print(f"  v_proj shape: {v_proj_shape}")
            print(f"  o_proj shape: {o_proj_shape}")
            
            # Check if dimensions are affected by GPU count
            if q_proj_shape[0] != config.hidden_size or q_proj_shape[1] != config.hidden_size:
                print(f"  ⚠️  Dimension mismatch detected!")
                print(f"     Expected: {config.hidden_size} x {config.hidden_size}")
                print(f"     Actual: {q_proj_shape[0]} x {q_proj_shape[1]}")
                
                # Calculate the ratio
                ratio = config.hidden_size / q_proj_shape[0]
                print(f"     Ratio: {ratio}")
                print(f"     GPU count: {num_gpus}")
                print(f"     Ratio matches GPU count: {abs(ratio - num_gpus) < 0.1}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")

def test_lora_with_device_map():
    """Test LoRA with different device map strategies"""
    print("\n🔍 Testing LoRA with different device map strategies...")
    
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    
    # Test different device map strategies
    device_maps = [
        "auto",
        "sequential",
        {"": "cuda:0"},  # All on single GPU
    ]
    
    for device_map in device_maps:
        print(f"\n📊 Testing device_map: {device_map}")
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device_map,
                trust_remote_code=True,
            )
            
            # Apply LoRA
            lora_config = LoraConfig(
                r=8,
                lora_alpha=1,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            model = get_peft_model(model, lora_config)
            
            # Check LoRA dimensions
            for name, module in model.named_modules():
                if 'q_proj' in name and 'lora_A' in name and 'default' in name:
                    print(f"  LoRA A shape: {module.weight.shape}")
                    
                    b_name = name.replace('lora_A', 'lora_B')
                    if b_name in dict(model.named_modules()):
                        b_module = dict(model.named_modules())[b_name]
                        print(f"  LoRA B shape: {b_module.weight.shape}")
                        
                        a_shape = module.weight.shape
                        b_shape = b_module.weight.shape
                        
                        if a_shape[1] != b_shape[0]:
                            print(f"  ❌ Dimension mismatch: {a_shape[1]} != {b_shape[0]}")
                        else:
                            print(f"  ✅ Dimensions match")
                        break
            
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
            
            print(f"  ✅ Forward pass successful!")
            print(f"  Output shape: {outputs['logits'].shape}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("🔬 Attention Heads Debug")
    print("=" * 60)
    
    debug_attention_heads()
    test_lora_with_device_map()
    
    print("\n" + "=" * 60)
    print("✅ Debug completed!")
    print("=" * 60) 