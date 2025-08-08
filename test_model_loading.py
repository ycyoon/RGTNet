#!/usr/bin/env python3
"""
Test RGTNet model loading
"""

import os
import torch
from transformers import AutoTokenizer

# Set environment variables to prevent NCCL issues
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_TIMEOUT'] = '1800'

def test_model_loading():
    model_path = "models/rgtnet_llama-3.2-1b-instruct_20250807_1044/merged_epoch_0"
    
    print(f"🔧 Testing model loading from: {model_path}")
    print(f"🔧 Using device: cuda:0")
    
    # Check if RGTNet model info exists
    rgtnet_info_path = os.path.join(model_path, "rgtnet_model_info.json")
    if os.path.exists(rgtnet_info_path):
        print("✅ RGTNet model info found")
        
        # Load tokenizer
        print("🔧 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer loaded successfully")
        
        # Load model
        print("🔧 Loading RGTNet model...")
        try:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map='cuda:0',
                trust_remote_code=True
            )
            model.eval()
            print("✅ RGTNet model loaded successfully")
            
            # Test generation
            print("🔧 Testing generation...")
            test_prompt = "Human: What is 2+2?\n\nAssistant:"
            
            inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to('cuda:0') for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Decode response
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print(f"✅ Generation successful: {response}")
            
        except Exception as e:
            print(f"❌ Failed to load RGTNet model: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ RGTNet model info not found")

if __name__ == "__main__":
    test_model_loading()
