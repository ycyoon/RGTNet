#!/usr/bin/env python3
"""
Debug LoRA scaling and dimensions
"""
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

def debug_lora_scaling():
    """Debug LoRA scaling values and dimensions"""
    print("🔍 Debugging LoRA scaling and dimensions...")
    
    # Test different LoRA configurations
    configs = [
        {"r": 8, "alpha": 8, "name": "r=8, alpha=8 (scaling=1)"},
        {"r": 8, "alpha": 16, "name": "r=8, alpha=16 (scaling=2)"},
        {"r": 16, "alpha": 16, "name": "r=16, alpha=16 (scaling=1)"},
    ]
    
    for config in configs:
        print(f"\n📊 Testing: {config['name']}")
        print(f"   R: {config['r']}")
        print(f"   Alpha: {config['alpha']}")
        print(f"   Scaling: {config['alpha'] / config['r']}")
        
        try:
            # Create LoRA config
            lora_config = LoraConfig(
                r=config['r'],
                lora_alpha=config['alpha'],
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            # Load model
            model_name = "meta-llama/Llama-3.2-3B-Instruct"
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            
            # Apply LoRA
            model = get_peft_model(model, lora_config)
            
            # Check LoRA modules
            lora_modules = []
            for name, module in model.named_modules():
                if 'lora' in name.lower():
                    lora_modules.append(name)
                    if 'q_proj' in name and 'lora_A' in name:
                        print(f"   Found LoRA module: {name}")
                        print(f"   Module shape: {module.weight.shape if hasattr(module, 'weight') else 'No weight'}")
            
            print(f"   Total LoRA modules: {len(lora_modules)}")
            
            # Test forward pass
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            input_text = "Hello, how are you?"
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=32)
            
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

def debug_model_dimensions():
    """Debug model dimensions"""
    print("\n🔍 Debugging model dimensions...")
    
    try:
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        
        # Load config
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        
        print(f"Model: {model_name}")
        print(f"Hidden size: {config.hidden_size}")
        print(f"Intermediate size: {config.intermediate_size}")
        print(f"Num attention heads: {config.num_attention_heads}")
        print(f"Head dimension: {config.hidden_size // config.num_attention_heads}")
        print(f"Num hidden layers: {config.num_hidden_layers}")
        
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
        print(f"  gate_proj: {first_layer.mlp.gate_proj.weight.shape}")
        print(f"  up_proj: {first_layer.mlp.up_proj.weight.shape}")
        print(f"  down_proj: {first_layer.mlp.down_proj.weight.shape}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 60)
    print("🔬 LoRA Scaling Debug")
    print("=" * 60)
    
    debug_model_dimensions()
    debug_lora_scaling()
    
    print("\n" + "=" * 60)
    print("✅ Debug completed!")
    print("=" * 60) 