#!/usr/bin/env python3
"""
병합된 체크포인트의 구조를 확인하는 스크립트
"""

import torch
from collections import OrderedDict

def check_merged_checkpoint():
    """병합된 체크포인트의 구조 확인"""
    model_path = "/home/ycyoon/work/RGTNet/models/llama3.2_3b_rgtnet.pth_merged"
    
    print("Loading merged checkpoint...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    print(f"Total parameters: {len(state_dict)}")
    
    # 주요 파라미터들 확인
    key_params = [
        'embedding.weight',
        'pos_encoder.weight', 
        'lm_head.weight',
        'layers.0.self_attn.q_proj.weight',
        'layers.0.linear1.weight',
        'layers.0.linear2.weight',
        'norm.weight'
    ]
    
    print("\nKey parameter shapes:")
    print("-" * 60)
    
    for param_name in key_params:
        if param_name in state_dict:
            shape = state_dict[param_name].shape
            print(f"{param_name:<35} {shape}")
        else:
            print(f"{param_name:<35} MISSING")
    
    # embedding 관련 파라미터들 확인
    print("\nEmbedding-related parameters:")
    print("-" * 60)
    
    embedding_params = [key for key in state_dict.keys() if 'embedding' in key]
    for param_name in embedding_params:
        shape = state_dict[param_name].shape
        print(f"{param_name:<35} {shape}")
    
    # pos_encoder 관련 파라미터들 확인
    print("\nPosition encoder parameters:")
    print("-" * 60)
    
    pos_params = [key for key in state_dict.keys() if 'pos_encoder' in key]
    for param_name in pos_params:
        shape = state_dict[param_name].shape
        print(f"{param_name:<35} {shape}")
    
    # lm_head 관련 파라미터들 확인
    print("\nLM head parameters:")
    print("-" * 60)
    
    lm_params = [key for key in state_dict.keys() if 'lm_head' in key]
    for param_name in lm_params:
        shape = state_dict[param_name].shape
        print(f"{param_name:<35} {shape}")
    
    # 첫 번째 layer의 모든 파라미터 확인
    print("\nFirst layer parameters:")
    print("-" * 60)
    
    layer0_params = [key for key in state_dict.keys() if key.startswith('layers.0.')]
    for param_name in sorted(layer0_params):
        shape = state_dict[param_name].shape
        print(f"{param_name:<35} {shape}")

if __name__ == "__main__":
    check_merged_checkpoint() 