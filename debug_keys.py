#!/usr/bin/env python3
"""
Debug script to inspect the actual keys in the model state dict
"""

import torch
import os

# Set environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def debug_model_keys():
    model_path = "models/rgtnet_llama-3.2-1b-instruct_20250807_1044/merged_epoch_0"
    state_dict_path = os.path.join(model_path, "pytorch_model.bin")
    
    print(f"🔧 Loading state dict from: {state_dict_path}")
    state_dict = torch.load(state_dict_path, map_location='cpu')
    
    print(f"📦 Total keys: {len(state_dict)}")
    
    # Print first 20 keys to understand the structure
    print("\n🔍 First 20 keys:")
    for i, key in enumerate(list(state_dict.keys())[:20]):
        print(f"  {i+1:2d}. {key}")
    
    # Check key patterns
    model_base_model_keys = [k for k in state_dict.keys() if k.startswith('model.base_model.')]
    base_model_keys = [k for k in state_dict.keys() if k.startswith('base_model.')]
    model_keys = [k for k in state_dict.keys() if k.startswith('model.') and not k.startswith('model.base_model.')]
    other_keys = [k for k in state_dict.keys() if not any(k.startswith(prefix) for prefix in ['model.base_model.', 'base_model.', 'model.'])]
    
    print(f"\n📊 Key patterns:")
    print(f"  • model.base_model.*: {len(model_base_model_keys)}")
    print(f"  • base_model.*: {len(base_model_keys)}")
    print(f"  • model.* (not base_model): {len(model_keys)}")
    print(f"  • Other: {len(other_keys)}")
    
    if model_base_model_keys:
        print(f"\n📝 Sample model.base_model.* keys:")
        for key in model_base_model_keys[:5]:
            print(f"  • {key}")
    
    if base_model_keys:
        print(f"\n📝 Sample base_model.* keys:")
        for key in base_model_keys[:5]:
            print(f"  • {key}")
    
    if model_keys:
        print(f"\n📝 Sample model.* keys:")
        for key in model_keys[:5]:
            print(f"  • {key}")
    
    if other_keys:
        print(f"\n📝 Other keys:")
        for key in other_keys:
            print(f"  • {key}")

if __name__ == "__main__":
    debug_model_keys()

