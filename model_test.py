#rgtnet 모델 테스트

import torch
from model import create_model, load_checkpoint
from transformers import AutoTokenizer
import torch.nn as nn
import argparse
import os
import pickle
from collections import OrderedDict

def use_fsdp_consolidation(base_path, rank_count, output_path):
    """FSDP의 내장 병합 기능을 사용하여 shard 파일들을 병합"""
    print(f"\n--- Using FSDP consolidation to merge {rank_count} sharded checkpoints ---")
    
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig
    import torch.distributed as dist
    import os
    
    # 분산 환경 초기화 (단일 프로세스)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    
    print("  Initializing single-process distributed environment...")
    dist.init_process_group(backend='gloo', rank=0, world_size=1)
    
    try:
        # 모든 shard 파일 확인
        shard_files = []
        for rank in range(rank_count):
            shard_path = f"{base_path}.rank{rank}.pt"
            if os.path.exists(shard_path):
                shard_files.append(shard_path)
                print(f"  Found shard: {shard_path}")
            else:
                print(f"  ⚠️  Missing shard: {shard_path}")
        
        if not shard_files:
            print("❌ No shard files found!")
            return False
        
        # 첫 번째 shard에서 기본 정보 가져오기
        print(f"  Loading first shard to check structure...")
        
        # 단일 프로세스 환경에서 로드 시도
        first_shard = torch.load(shard_files[0], map_location='cpu', weights_only=False)
        print(f"  Successfully loaded first shard")
        print(f"  First shard keys: {list(first_shard.keys())}")
        
        # 병합된 state_dict 생성
        merged_state_dict = OrderedDict()
        epoch = -1
        
        # 각 shard에서 파라미터 수집
        for rank, shard_path in enumerate(shard_files):
            print(f"  Loading shard {rank}: {shard_path}")
            
            try:
                # 단일 프로세스 환경에서 로드
                checkpoint = torch.load(shard_path, map_location='cpu', weights_only=False)
                
                if 'model' in checkpoint:
                    # FSDP sharded state dict
                    sharded_state = checkpoint['model']
                    print(f"    Shard {rank} has {len(sharded_state)} parameters")
                    
                    for key, value in sharded_state.items():
                        if key not in merged_state_dict:
                            merged_state_dict[key] = value
                        else:
                            # 이미 있는 경우 덮어쓰기 (마지막 shard가 우선)
                            merged_state_dict[key] = value
                    
                    if epoch == -1:
                        epoch = checkpoint.get('epoch', 0)
                else:
                    print(f"    ⚠️  No 'model' key in shard {rank}")
                    
            except Exception as e:
                print(f"    ❌ Error loading shard {rank}: {e}")
                continue
        
        print(f"  Total merged parameters: {len(merged_state_dict)}")
        
        # 병합된 체크포인트 저장
        merged_checkpoint = {
            'model_state_dict': merged_state_dict,
            'epoch': epoch
        }
        
        torch.save(merged_checkpoint, output_path)
        print(f"✅ Merged checkpoint saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error merging shards: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 분산 환경 정리
        if dist.is_initialized():
            dist.destroy_process_group()
            print("  Cleaned up distributed environment")

def extract_model_config_from_checkpoint(checkpoint_path):
    """체크포인트에서 모델 설정을 정확히 추출"""
    print("Loading merged checkpoint...")
    
    # 병합된 체크포인트에서 모델 정보 추출
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 모델 state_dict에서 구조 정보 추출
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    print("Extracted model config:")
    
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
    
    return {
        'vocab_size': vocab_size,
        'd_model': d_model,
        'nhead': nhead,
        'num_layers': num_layers,
        'max_seq_len': pos_encoder_size,
        'intermediate_size': intermediate_size,
    }

def create_model_args(config):
    """추출된 설정으로 args 생성"""
    args = argparse.Namespace(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dropout=0.1,
        bias_delta=1.0,
        max_seq_len=config['max_seq_len'],
        pretrained_model_name=None,
        gradient_checkpointing=False,
    )
    
    return args

def test_model_generation(model, tokenizer, device, prompts=None):
    """모델 생성 테스트"""
    if prompts is None:
        prompts = [
            "Hello, how are you?",
            "What is artificial intelligence?",
            "Explain quantum computing in simple terms.",
            "Write a short story about a robot.",
        ]
    
    print("\n" + "="*60)
    print("MODEL GENERATION TEST")
    print("="*60)
    
    model.eval()
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Test {i}: {prompt} ---")
        
        try:
            # 토큰화
            tokens = tokenizer.encode(prompt, return_tensors="pt")
            tokens = tokens.to(device)
            
            print(f"Input tokens shape: {tokens.shape}")
            print(f"Input text: {prompt}")
            
            # 생성
            with torch.no_grad():
                output = model.generate(
                    tokens, 
                    max_length=100, 
                    do_sample=True, 
                    temperature=0.8,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # 디코딩
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"Generated text: {generated_text}")
            
        except Exception as e:
            print(f"❌ Error in generation: {e}")
            import traceback
            traceback.print_exc()

def find_latest_model():
    """Find the latest timestamped model directory"""
    base_dir = "/home/ycyoon/work/RGTNet/models"
    model_pattern = "llama3.2_3b_rgtnet_*"
    
    # Find all matching directories
    import glob
    model_dirs = glob.glob(os.path.join(base_dir, model_pattern))
    
    if not model_dirs:
        # Fallback to old merged file
        old_path = "/home/ycyoon/work/RGTNet/models/llama3.2_3b_rgtnet.pth_merged"
        if os.path.exists(old_path):
            return old_path, None
        return None, None
    
    # Sort by creation time (newest first)
    model_dirs.sort(key=os.path.getctime, reverse=True)
    latest_dir = model_dirs[0]
    
    # Look for the model file in the directory
    model_file = os.path.join(latest_dir, "llama3.2_3b_rgtnet.pth")
    if os.path.exists(model_file):
        return model_file, latest_dir
    
    # If no .pth file, check if the directory itself is a file (old format)
    if os.path.isfile(latest_dir):
        return latest_dir, os.path.dirname(latest_dir)
    
    # If still no file found, return None
    return None, latest_dir

def main():
    """메인 함수"""
    print("🚀 RGTNet Model Test")
    print("="*60)
    
    # 최신 모델 찾기
    model_path, model_dir = find_latest_model()
    
    if model_path is None:
        print("❌ No model found!")
        print("Please ensure a trained model exists in the models directory.")
        return
    
    print(f"📁 Using model: {model_path}")
    if model_dir:
        print(f"📂 Model directory: {model_dir}")
    
    # 체크포인트 파일 존재 확인
    if not os.path.exists(model_path):
        print(f"❌ Checkpoint file not found: {model_path}")
        print("Please ensure the model file exists.")
        return
    
    # 체크포인트에서 모델 설정 추출
    config = extract_model_config_from_checkpoint(model_path)
    
    # args 생성
    args = create_model_args(config)
    
    print(f"\nUsing extracted model config:")
    print(f"  d_model: {args.d_model}")
    print(f"  nhead: {args.nhead}")
    print(f"  num_layers: {args.num_layers}")
    print(f"  vocab_size: {args.vocab_size}")
    print(f"  max_seq_len: {args.max_seq_len}")
    
    # 모델 생성 및 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    try:
        model = create_model(args, args.vocab_size - 1)  # pad_idx = vocab_size - 1
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        
        # 체크포인트 로드
        print("\nLoading checkpoint...")
        load_checkpoint(model, optimizer, model_path, device)
        print("✅ Checkpoint loaded successfully!")
        
        # 모델 정보 출력
        print(f"\nModel structure:")
        print(model)
        
        # 토크나이저 로드
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
        print("✅ Tokenizer loaded successfully!")
        
        # 기본 forward pass 테스트
        print("\n" + "="*60)
        print("BASIC FORWARD PASS TEST")
        print("="*60)
        
        model.eval()
        test_input = torch.randint(0, 1000, (1, 10)).to(device)
        test_role_mask = torch.randint(0, 2, (1, 10)).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=test_input, role_mask=test_role_mask)
            print(f"✅ Forward pass successful!")
            print(f"Output shape: {outputs['logits'].shape}")
            print(f"Expected shape: (1, 10, {args.vocab_size})")
        
        # 생성 테스트
        test_model_generation(model, tokenizer, device)
        
    except Exception as e:
        print(f"❌ Error during model setup: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()







