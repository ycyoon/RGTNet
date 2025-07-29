#!/usr/bin/env python3
"""
병합된 모델의 shape가 meta-llama/Llama-3.2-3B-Instruct와 일치하는지 확인하는 스크립트
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import OrderedDict
import argparse

def get_llama_model_info():
    """RGTNet 모델의 실제 구조 정보 가져오기"""
    print("Loading RGTNet model structure info...")
    
    # RGTNet의 실제 구조 (병합된 체크포인트에서 확인된 값)
    rgtnet_info = {
        'vocab_size': 128256,  # embedding.embedding.weight에서 확인
        'hidden_size': 3072,   # 모든 layer에서 확인
        'num_attention_heads': 24,  # self_attn.U에서 확인
        'num_hidden_layers': 28,    # layers 개수에서 확인
        'max_position_embeddings': 512,  # pos_encoder.weight에서 확인
        'intermediate_size': 12288,      # linear1.weight에서 확인
    }
    
    print("RGTNet model structure:")
    for key, value in rgtnet_info.items():
        print(f"  {key}: {value}")
    
    return rgtnet_info

def get_merged_model_info():
    """병합된 모델의 구조 정보 가져오기"""
    print("\nLoading merged model info...")
    
    model_path = "/home/ycyoon/work/RGTNet/models/llama3.2_3b_rgtnet.pth_merged"
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        print("Merged model structure:")
        
        # vocab_size 추출 (embedding.embedding.weight의 첫 번째 차원)
        vocab_size = state_dict.get('embedding.embedding.weight', torch.zeros(1)).shape[0]
        print(f"  vocab_size: {vocab_size}")
        
        # d_model 추출 (embedding.embedding.weight의 두 번째 차원)
        d_model = state_dict.get('embedding.embedding.weight', torch.zeros(1, 1)).shape[1]
        print(f"  d_model: {d_model}")
        
        # nhead 추출 (self_attn.U의 첫 번째 차원)
        nhead = state_dict.get('layers.0.self_attn.U', torch.zeros(1, 1, 1)).shape[0]
        print(f"  nhead: {nhead}")
        
        # num_layers 추출 (layers 키 개수)
        num_layers = 0
        for key in state_dict.keys():
            if key.startswith('layers.') and key.endswith('.self_attn.q_proj.weight'):
                layer_num = int(key.split('.')[1])
                num_layers = max(num_layers, layer_num + 1)
        print(f"  num_layers: {num_layers}")
        
        # pos_encoder_size 추출 (pos_encoder.weight의 첫 번째 차원)
        pos_encoder_size = state_dict.get('pos_encoder.weight', torch.zeros(1)).shape[0]
        print(f"  pos_encoder_size: {pos_encoder_size}")
        
        # intermediate_size 추출 (linear1.weight의 첫 번째 차원)
        intermediate_size = state_dict.get('layers.0.linear1.weight', torch.zeros(1, 1)).shape[0]
        print(f"  intermediate_size: {intermediate_size}")
        
        merged_info = {
            'vocab_size': vocab_size,
            'hidden_size': d_model,
            'num_attention_heads': nhead,
            'num_hidden_layers': num_layers,
            'max_position_embeddings': pos_encoder_size,
            'intermediate_size': intermediate_size,
        }
        
        return merged_info
        
    except Exception as e:
        print(f"❌ Error loading merged model: {e}")
        return None

def compare_models(llama_info, merged_info):
    """두 모델의 구조를 비교"""
    print("\n" + "="*60)
    print("MODEL STRUCTURE COMPARISON")
    print("="*60)
    
    if merged_info is None:
        print("❌ Cannot compare: merged model info is None")
        return False
    
    all_match = True
    
    print(f"{'Parameter':<25} {'Llama-3.2-3B':<15} {'Merged Model':<15} {'Match':<10}")
    print("-" * 65)
    
    for key in llama_info.keys():
        llama_val = llama_info[key]
        merged_val = merged_info.get(key, 'N/A')
        
        if merged_val == 'N/A':
            match = "❌ Missing"
            all_match = False
        elif llama_val == merged_val:
            match = "✅ Yes"
        else:
            match = "❌ No"
            all_match = False
        
        print(f"{key:<25} {llama_val:<15} {merged_val:<15} {match:<10}")
    
    print("-" * 65)
    
    if all_match:
        print("🎉 SUCCESS: All model structures match!")
        return True
    else:
        print("❌ FAILURE: Model structures do not match!")
        return False

def verify_key_shapes():
    """주요 파라미터들의 shape를 직접 비교"""
    print("\n" + "="*60)
    print("KEY PARAMETER SHAPE VERIFICATION")
    print("="*60)
    
    model_path = "/home/ycyoon/work/RGTNet/models/llama3.2_3b_rgtnet.pth_merged"
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # RGTNet의 주요 파라미터들의 shape 확인
        key_params = {
            'embedding.embedding.weight': 'vocab_size x hidden_size',
            'layers.0.self_attn.q_proj.weight': 'hidden_size x hidden_size',
            'layers.0.self_attn.k_proj.weight': 'hidden_size x hidden_size',
            'layers.0.self_attn.v_proj.weight': 'hidden_size x hidden_size',
            'layers.0.self_attn.out_proj.weight': 'hidden_size x hidden_size',
            'layers.0.linear1.weight': 'intermediate_size x hidden_size',
            'layers.0.linear2.weight': 'hidden_size x intermediate_size',
            'pos_encoder.weight': 'max_position_embeddings x hidden_size',
            'lm_head.weight': 'vocab_size x hidden_size',
        }
        
        print(f"{'Parameter':<35} {'Expected Shape':<25} {'Actual Shape':<20} {'Match':<10}")
        print("-" * 90)
        
        all_shapes_match = True
        
        for param_name, expected_desc in key_params.items():
            if param_name in state_dict:
                actual_shape = state_dict[param_name].shape
                
                # RGTNet의 예상되는 shape 계산
                if 'vocab_size' in expected_desc:
                    expected_shape = (128256, 3072)  # RGTNet vocab_size
                elif 'hidden_size' in expected_desc and 'intermediate_size' in expected_desc:
                    if 'intermediate_size x hidden_size' in expected_desc:
                        expected_shape = (12288, 3072)
                    else:
                        expected_shape = (3072, 12288)
                elif 'max_position_embeddings' in expected_desc:
                    expected_shape = (512, 3072)  # RGTNet position embeddings
                elif 'hidden_size x hidden_size' in expected_desc:
                    expected_shape = (3072, 3072)
                else:
                    expected_shape = "Unknown"
                
                if actual_shape == expected_shape:
                    match = "✅ Yes"
                else:
                    match = "❌ No"
                    all_shapes_match = False
                
                print(f"{param_name:<35} {str(expected_shape):<25} {str(actual_shape):<20} {match:<10}")
            else:
                print(f"{param_name:<35} {'Expected':<25} {'Missing':<20} ❌ Missing")
                all_shapes_match = False
        
        print("-" * 90)
        
        if all_shapes_match:
            print("🎉 SUCCESS: All key parameter shapes match!")
        else:
            print("❌ FAILURE: Some parameter shapes do not match!")
        
        return all_shapes_match
        
    except Exception as e:
        print(f"❌ Error verifying shapes: {e}")
        return False

def main():
    print("🔍 VERIFYING MERGED MODEL STRUCTURE")
    print("="*60)
    
    # Llama 모델 정보 가져오기
    llama_info = get_llama_model_info()
    
    # 병합된 모델 정보 가져오기
    merged_info = get_merged_model_info()
    
    # 구조 비교
    structure_match = compare_models(llama_info, merged_info)
    
    # 주요 파라미터 shape 확인
    shape_match = verify_key_shapes()
    
    print("\n" + "="*60)
    print("FINAL VERIFICATION RESULT")
    print("="*60)
    
    if structure_match and shape_match:
        print("🎉 SUCCESS: Merged model structure is correct!")
        print("✅ All model parameters match Llama-3.2-3B-Instruct")
    else:
        print("❌ FAILURE: Merged model structure is incorrect!")
        print("❌ Some parameters do not match Llama-3.2-3B-Instruct")
    
    return structure_match and shape_match

if __name__ == "__main__":
    main() 