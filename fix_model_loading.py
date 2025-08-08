#!/usr/bin/env python3
"""
Fix RGTNet model loading by creating a proper model wrapper
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoConfig

# Set environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_TIMEOUT'] = '1800'

class RGTNetWrapper:
    """Wrapper for RGTNet models to handle the custom architecture"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device('cuda:0')
        
        print(f"(유🔧 Loading RGTNet model from: {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load and fix the model weights
        self.model = self._load_and_fix_model()
        self.model.eval()
        
        print("✅ RGTNet model loaded successfully")
    
    def _load_and_fix_model(self):
        """Load model and fix the weight mapping from model.base_model.* to model.*"""
        import json
        from transformers import LlamaForCausalLM
        
        # First, load the state dict directly
        print("🔧 Loading state dict directly...")
        
        # Load pytorch_model.bin directly
        state_dict_path = os.path.join(self.model_path, "pytorch_model.bin")
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location='cpu')
            print(f"📦 Loaded state dict with {len(state_dict)} keys")
        else:
            # Try to load using safetensors
            from safetensors import safe_open
            safetensors_path = os.path.join(self.model_path, "model.safetensors")
            if os.path.exists(safetensors_path):
                state_dict = {}
                with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
                print(f"📦 Loaded state dict from safetensors with {len(state_dict)} keys")
            else:
                raise FileNotFoundError("No pytorch_model.bin or model.safetensors found")
        
        # Create a mapping from base_model.* to model.*
        print("🔧 Fixing weight key mapping...")
        fixed_state_dict = {}
        
        for key, value in state_dict.items():
            if key.startswith('base_model.model.'):
                # Remove 'base_model.' prefix to get 'model.*'
                new_key = key.replace('base_model.', '')
                fixed_state_dict[new_key] = value
                print(f"📝 Mapped: {key} -> {new_key}")
            elif key.startswith('base_model.lm_head.'):
                # Map base_model.lm_head.* to lm_head.*
                new_key = key.replace('base_model.', '')
                fixed_state_dict[new_key] = value
                print(f"📝 Mapped: {key} -> {new_key}")
            elif key.startswith('base_model.'):
                # For other base_model keys, remove base_model prefix
                new_key = key.replace('base_model.', '')
                fixed_state_dict[new_key] = value
                print(f"📝 Mapped: {key} -> {new_key}")
            else:
                # Keep other keys as-is
                fixed_state_dict[key] = value
        
        print(f"✅ Fixed state dict with {len(fixed_state_dict)} keys")
        
        # Load config and create model
        config_path = os.path.join(self.model_path, "config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        from transformers import LlamaConfig
        config = LlamaConfig(**config_dict)
        
        # Create empty model
        print("🔧 Creating empty LlamaForCausalLM model...")
        model = LlamaForCausalLM(config)
        
        # Load the fixed weights
        print("🔧 Loading fixed weights...")
        missing_keys, unexpected_keys = model.load_state_dict(fixed_state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️ Missing keys: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            # Check if lm_head.weight is missing - it might need to be tied to embed_tokens
            if 'lm_head.weight' in missing_keys and 'model.embed_tokens.weight' in fixed_state_dict:
                print("🔧 Tying lm_head.weight to embed_tokens.weight...")
                fixed_state_dict['lm_head.weight'] = fixed_state_dict['model.embed_tokens.weight']
                # Reload with the tied weight
                missing_keys, unexpected_keys = model.load_state_dict(fixed_state_dict, strict=False)
                print(f"✅ After tying weights - Missing keys: {len(missing_keys)}")
        else:
            print("✅ No missing keys!")
        
        if unexpected_keys:
            print(f"⚠️ Unexpected keys: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
        
        # Move to device
        model = model.to(self.device, dtype=torch.float16)
        
        return model
    
    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7):
        """Generate response for a given prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode only the new tokens
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip()

def test_rgtnet_model():
    model_path = "models/rgtnet_llama-3.2-1b-instruct_20250807_1044/merged_epoch_0"
    
    print("🔧 Testing RGTNet model loading and generation...")
    
    try:
        # Create wrapper
        model = RGTNetWrapper(model_path)
        
        # Test generation
        test_prompts = [
            "Human: What is 2+2?\n\nAssistant:",
            "Human: Explain quantum computing in simple terms.\n\nAssistant:",
            "Human: Write a short poem about AI.\n\nAssistant:"
        ]
        
        for i, prompt in enumerate(test_prompts):
            print(f"\n📝 Test {i+1}:")
            print(f"Prompt: {prompt}")
            
            response = model.generate(prompt, max_new_tokens=100, temperature=0.7)
            print(f"Response: {response}")
            
            # Check if response is reasonable
            if len(response.strip()) > 10 and not any(char in response for char in ['探', 'glitch', 'FNcef']):
                print("✅ Response looks reasonable")
            else:
                print("⚠️ Response may have issues")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rgtnet_model()
    if success:
        print("\n✅ RGTNet model test completed successfully!")
    else:
        print("\n❌ RGTNet model test failed!")
