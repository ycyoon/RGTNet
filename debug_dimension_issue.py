#!/usr/bin/env python3
"""
Debug dimension mismatch issue in detail
"""
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model

def analyze_model_dimensions():
    """Analyze model dimensions in detail"""
    print("🔍 Analyzing model dimensions in detail...")
    
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    
    # Load config
    config = AutoConfig.from_pretrained(model_name)
    print(f"Model config:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Intermediate size: {config.intermediate_size}")
    print(f"  Num attention heads: {config.num_attention_heads}")
    print(f"  Head dimension: {config.hidden_size // config.num_attention_heads}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Check first layer dimensions
    first_layer = model.model.layers[0]
    print(f"\nFirst layer dimensions:")
    print(f"  q_proj: {first_layer.self_attn.q_proj.weight.shape}")
    print(f"  k_proj: {first_layer.self_attn.k_proj.weight.shape}")
    print(f"  v_proj: {first_layer.self_attn.v_proj.weight.shape}")
    print(f"  o_proj: {first_layer.self_attn.o_proj.weight.shape}")
    
    # Check if there's any dimension doubling
    q_in, q_out = first_layer.self_attn.q_proj.weight.shape
    print(f"\nQ projection: {q_in} -> {q_out}")
    print(f"Expected: {config.hidden_size} -> {config.hidden_size}")
    
    if q_in != config.hidden_size or q_out != config.hidden_size:
        print(f"⚠️  Dimension mismatch detected!")
        print(f"   Expected: {config.hidden_size} x {config.hidden_size}")
        print(f"   Actual: {q_in} x {q_out}")

def test_lora_dimensions():
    """Test LoRA dimensions step by step"""
    print("\n🔍 Testing LoRA dimensions step by step...")
    
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    
    # Test 1: Load base model
    print("1. Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Check base model dimensions
    first_layer = model.model.layers[0]
    base_q_shape = first_layer.self_attn.q_proj.weight.shape
    print(f"   Base q_proj shape: {base_q_shape}")
    
    # Test 2: Apply LoRA
    print("2. Applying LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=1,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    # Test 3: Check LoRA modules
    print("3. Checking LoRA modules...")
    for name, module in model.named_modules():
        if 'q_proj' in name and 'lora_A' in name and 'default' in name:
            print(f"   Found LoRA A: {name}")
            print(f"   LoRA A shape: {module.weight.shape}")
            
            # Check corresponding B module
            b_name = name.replace('lora_A', 'lora_B')
            if b_name in dict(model.named_modules()):
                b_module = dict(model.named_modules())[b_name]
                print(f"   LoRA B shape: {b_module.weight.shape}")
                
                # Analyze dimensions
                a_shape = module.weight.shape
                b_shape = b_module.weight.shape
                
                print(f"   LoRA A: {a_shape[0]} x {a_shape[1]}")
                print(f"   LoRA B: {b_shape[0]} x {b_shape[1]}")
                print(f"   Expected: rank x input_dim -> output_dim x rank")
                print(f"   Actual: {a_shape[0]} x {a_shape[1]} -> {b_shape[0]} x {b_shape[1]}")
                
                # Check if dimensions match
                if a_shape[1] != b_shape[0]:
                    print(f"   ❌ Dimension mismatch: {a_shape[1]} != {b_shape[0]}")
                else:
                    print(f"   ✅ Dimensions match")
                
                break
    
    # Test 4: Try forward pass
    print("4. Testing forward pass...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        input_text = "Hello"
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=16)
        
        for key in inputs:
            inputs[key] = inputs[key].cuda()
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"   ✅ Forward pass successful!")
        print(f"   Output shape: {outputs['logits'].shape}")
        
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

def test_hybrid_model():
    """Test hybrid model specifically"""
    print("\n🔍 Testing hybrid model...")
    
    try:
        from model_hybrid import create_hybrid_model
        
        args = argparse.Namespace()
        args.pretrained_model_name = "meta-llama/Llama-3.2-3B-Instruct"
        args.use_lora = True
        args.enable_role_adapters = True
        args.lora_r = 8
        args.lora_alpha = 1
        args.lora_dropout = 0.05
        args.use_quantization = False
        
        print("Creating hybrid model...")
        model, tokenizer = create_hybrid_model(args)
        
        print("Checking LoRA modules in hybrid model...")
        for name, module in model.named_modules():
            if 'q_proj' in name and 'lora_A' in name and 'default' in name:
                print(f"   Found LoRA A: {name}")
                print(f"   LoRA A shape: {module.weight.shape}")
                
                b_name = name.replace('lora_A', 'lora_B')
                if b_name in dict(model.named_modules()):
                    b_module = dict(model.named_modules())[b_name]
                    print(f"   LoRA B shape: {b_module.weight.shape}")
                    
                    a_shape = module.weight.shape
                    b_shape = b_module.weight.shape
                    print(f"   LoRA A: {a_shape[0]} x {a_shape[1]}")
                    print(f"   LoRA B: {b_shape[0]} x {b_shape[1]}")
                    
                    if a_shape[1] != b_shape[0]:
                        print(f"   ❌ Dimension mismatch: {a_shape[1]} != {b_shape[0]}")
                    else:
                        print(f"   ✅ Dimensions match")
                    break
        
        # Test forward pass
        print("Testing hybrid model forward pass...")
        input_text = "Hello"
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=16)
        
        for key in inputs:
            inputs[key] = inputs[key].cuda()
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"✅ Hybrid model forward pass successful!")
        print(f"Output shape: {outputs['logits'].shape}")
        
    except Exception as e:
        print(f"❌ Hybrid model test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 60)
    print("🔬 Dimension Mismatch Debug")
    print("=" * 60)
    
    analyze_model_dimensions()
    test_lora_dimensions()
    test_hybrid_model()
    
    print("\n" + "=" * 60)
    print("✅ Debug completed!")
    print("=" * 60) 