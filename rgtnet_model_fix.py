#!/usr/bin/env python3
"""
RGTNet Model Fix for multi_model_benchmark.py
This script provides the corrected RGTNet model loading logic.
"""

import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
from easyjailbreak.models.model_base import ModelBase

class FixedRGTNetModel(ModelBase):
    """Fixed RGTNet model with proper weight mapping"""
    
    def __init__(self, model_path: str, pretrained_model_name: str = None):
        super().__init__()
        self.model_path = model_path
        self.pretrained_model_name = pretrained_model_name or "meta-llama/Llama-3.2-3B-Instruct"
        
        # Force single GPU usage to prevent device mismatch
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        print(f"🔧 Loading RGTNet model from: {model_path}")
        print(f"🔧 Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create model configuration
        self._setup_model_config()
        
        # Load the trained model
        self.model = self._load_trained_model()
        self.model.eval()
        
        print(f"✅ RGTNet model loaded successfully on {self.device}")
    
    def _setup_model_config(self):
        """Setup model configuration based on pretrained model"""
        from transformers import AutoConfig
        
        # Check if model_path is a directory (merged model) or file
        if os.path.isdir(self.model_path):
            print(f"📁 Model path is a directory: {self.model_path}")
            # Check if it's a HuggingFace model directory
            if os.path.exists(os.path.join(self.model_path, 'config.json')):
                print("📋 Detected HuggingFace model directory")
                self.merged_model_path = self.model_path
            else:
                print("⚠️ No config.json found in directory")
                self.merged_model_path = None
        else:
            print(f"📄 Model path is a file: {self.model_path}")
            self.merged_model_path = self.model_path
        
        # Get pretrained model config
        pretrained_config = AutoConfig.from_pretrained(self.pretrained_model_name)
        
        # Create args object for model creation
        class ModelArgs:
            def __init__(self, pretrained_model_name):
                self.d_model = pretrained_config.hidden_size
                self.nhead = pretrained_config.num_attention_heads
                self.num_layers = pretrained_config.num_hidden_layers
                self.dropout = 0.1
                self.max_seq_len = getattr(pretrained_config, 'max_position_embeddings', 2048)
                self.bias_delta = 1.0
                self.vocab_size = pretrained_config.vocab_size
                self.pretrained_model_name = pretrained_model_name
        
        self.args = ModelArgs(self.pretrained_model_name)
        print(f"📊 Model config: d_model={self.args.d_model}, layers={self.args.num_layers}, heads={self.args.nhead}")
    
    def _load_trained_model(self):
        """Load the trained RGTNet model with proper weight mapping"""
        # Determine the actual model directory to load
        if hasattr(self, 'merged_model_path') and self.merged_model_path:
            model_dir = self.merged_model_path
            print(f"📋 Loading from merged model directory: {model_dir}")
        else:
            model_dir = self.model_path
            print(f"📋 Loading from model path: {model_dir}")
        
        # Check if it's a HuggingFace model directory (merged model)
        if os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, 'config.json')):
            print("📁 Detected HuggingFace model directory, loading with fixed weight mapping...")
            try:
                # Check if this is a RGTNet model by looking for rgtnet_model_info.json
                rgtnet_info_path = os.path.join(model_dir, "rgtnet_model_info.json")
                
                if os.path.exists(rgtnet_info_path):
                    print("🔧 Detected RGTNet model, applying weight mapping fix...")
                    return self._load_rgtnet_with_mapping_fix(model_dir)
                else:
                    print("🔧 Loading as standard HuggingFace model...")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_dir,
                        torch_dtype=torch.float16,
                        device_map=self.device,
                        trust_remote_code=True
                    )
                    print("✅ HuggingFace model loaded successfully")
                    return model
            except Exception as e:
                print(f"❌ Failed to load model: {e}")
                raise
        else:
            print("❌ Expected HuggingFace model directory with config.json")
            raise FileNotFoundError(f"No config.json found in {model_dir}")
    
    def _load_rgtnet_with_mapping_fix(self, model_dir):
        """Load RGTNet model with proper weight mapping from base_model.* to model.*"""
        
        # Load state dict directly
        print("🔧 Loading state dict directly...")
        state_dict_path = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location='cpu')
            print(f"📦 Loaded state dict with {len(state_dict)} keys")
        else:
            # Try to load using safetensors
            try:
                from safetensors import safe_open
                safetensors_path = os.path.join(model_dir, "model.safetensors")
                if os.path.exists(safetensors_path):
                    state_dict = {}
                    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            state_dict[key] = f.get_tensor(key)
                    print(f"📦 Loaded state dict from safetensors with {len(state_dict)} keys")
                else:
                    raise FileNotFoundError("No pytorch_model.bin or model.safetensors found")
            except ImportError:
                raise FileNotFoundError("No pytorch_model.bin found and safetensors not available")
        
        # Create a mapping from base_model.* to model.*
        print("🔧 Fixing weight key mapping...")
        fixed_state_dict = {}
        
        for key, value in state_dict.items():
            if key.startswith('base_model.model.'):
                # Remove 'base_model.' prefix to get 'model.*'
                new_key = key.replace('base_model.', '')
                fixed_state_dict[new_key] = value
            elif key.startswith('base_model.lm_head.'):
                # Map base_model.lm_head.* to lm_head.*
                new_key = key.replace('base_model.', '')
                fixed_state_dict[new_key] = value
            elif key.startswith('base_model.'):
                # For other base_model keys, remove base_model prefix
                new_key = key.replace('base_model.', '')
                fixed_state_dict[new_key] = value
            else:
                # Keep other keys as-is
                fixed_state_dict[key] = value
        
        print(f"✅ Fixed state dict with {len(fixed_state_dict)} keys")
        
        # Load config and create model
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
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
        print("✅ RGTNet model loaded successfully with weight mapping fix")
        
        return model
    
    def generate(self, prompts, **kwargs):
        """Generate responses for given prompts"""
        if isinstance(prompts, str):
            prompts = [prompts]
        
        responses = []
        
        with torch.no_grad():
            for prompt in prompts:
                # Tokenize input
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=self.args.max_seq_len - 512,  # Leave room for generation
                    padding=True
                )
                
                # Move inputs to device explicitly
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate
                try:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model.generate(
                            inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            max_new_tokens=kwargs.get('max_tokens', 100),
                            min_new_tokens=20,  # 최소 20개 토큰 생성
                            temperature=kwargs.get('temperature', 0.8),
                            do_sample=True,  # 샘플링 강제 활성화
                            top_p=0.9,
                            repetition_penalty=1.1,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            use_cache=True
                        )
                    
                    # Decode response
                    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                    response = self.tokenizer.decode(
                        generated_tokens, 
                        skip_special_tokens=True
                    ).strip()
                    
                    responses.append(response)
                    
                except Exception as e:
                    print(f"⚠️ Generation failed for prompt: {e}")
                    responses.append("Error: Generation failed")
        
        return responses if len(responses) > 1 else responses[0]


# Usage example:
if __name__ == "__main__":
    model_path = "/home/ycyoon/work/RGTNet/models/rgtnet_llama-3.2-1b-instruct_20250807_1044/merged_epoch_0"
    model = FixedRGTNetModel(model_path)
    
    test_prompt = "Hello, how are you?"
    response = model.generate(test_prompt)
    print(f"Response: {response}")

