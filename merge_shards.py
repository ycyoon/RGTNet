#!/usr/bin/env python3
"""
FSDP Shard 병합 스크립트
torchrun --nproc_per_node=8 merge_shards.py
"""

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
import os
import argparse
from collections import OrderedDict

def merge_fsdp_shards():
    """FSDP shard 파일들을 병합"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print(f"Starting FSDP shard merge with {world_size} processes")
    
    # shard 체크포인트 경로 설정
    base_path = "/home/ycyoon/work/RGTNet/models/llama3.2_3b_rgtnet.pth"
    shard_path = f"{base_path}.rank{rank}.pt"
    output_path = base_path + "_merged"
    
    print(f"[Rank {rank}] Loading shard {rank}: {shard_path}")
    
    # 각 rank가 자신의 shard만 로드
    checkpoint = torch.load(shard_path, map_location='cpu', weights_only=False)
    
    if 'model' in checkpoint:
        sharded_state = checkpoint['model']
        print(f"[Rank {rank}] Shard {rank} has {len(sharded_state)} parameters")
        
        # epoch 정보 수집 (첫 번째 shard에서)
        epoch = checkpoint.get('epoch', 0) if rank == 0 else None
    else:
        print(f"[Rank {rank}] ⚠️  No 'model' key in shard {rank}")
        return
    
    # 모든 프로세스 동기화
    dist.barrier()
    
    # 모든 rank가 자신의 데이터를 임시 파일에 저장
    temp_file = f"/tmp/shard_data_rank_{rank}.pt"
    
    # ShardedTensor를 일반 Tensor로 변환하여 저장 (전체 구조 보존)
    converted_state = {}
    for key, value in sharded_state.items():
        if hasattr(value, 'local_shards'):
            # ShardedTensor의 전체 구조 정보 수집
            local_shards = value.local_shards()
            if not local_shards:
                # 이 rank에 shard가 없음, 건너뜀
                continue
            
            # 전체 텐서 크기 계산
            total_size = value.size()
            print(f"[Rank {rank}] {key}: total_size={total_size}, local_shards={len(local_shards)}")
            
            # 각 shard의 정보 저장
            shard_info = {
                'total_size': total_size,
                'local_shards': []
            }
            
            for shard in local_shards:
                shard_info['local_shards'].append({
                    'tensor': shard.tensor,
                    'metadata': shard.metadata
                })
            
            converted_state[key] = shard_info
            
        elif hasattr(value, 'local_tensor'):
            converted_state[key] = value.local_tensor()
        else:
            converted_state[key] = value
    
    torch.save(converted_state, temp_file)
    print(f"[Rank {rank}] Saved data to {temp_file}")
    
    # 모든 rank가 파일 저장을 완료할 때까지 대기
    dist.barrier()
    
    if rank == 0:
        print("All shards loaded and saved successfully")
        print("Merging all shard data...")
        
        # 병합된 state_dict 생성
        merged_state_dict = OrderedDict()
        
        # 모든 rank의 데이터를 수집
        for r in range(world_size):
            temp_file = f"/tmp/shard_data_rank_{r}.pt"
            print(f"  Loading data from rank {r}: {temp_file}")
            
            try:
                rank_data = torch.load(temp_file, map_location='cpu', weights_only=False)
                print(f"    Rank {r} data loaded with {len(rank_data)} parameters")
                
                for key, value in rank_data.items():
                    if key not in merged_state_dict:
                        merged_state_dict[key] = value
                    else:
                        # 이미 있는 경우 shard 정보 병합
                        if isinstance(value, dict) and 'local_shards' in value:
                            # ShardedTensor 정보 병합
                            existing = merged_state_dict[key]
                            if isinstance(existing, dict) and 'local_shards' in existing:
                                existing['local_shards'].extend(value['local_shards'])
                            else:
                                merged_state_dict[key] = value
                        else:
                            # 일반 텐서는 덮어쓰기
                            merged_state_dict[key] = value
            except Exception as e:
                print(f"    ❌ Error loading rank {r} data: {e}")
        
        print(f"  Total merged parameters: {len(merged_state_dict)}")
        
        # ShardedTensor 정보를 실제 텐서로 변환
        print("  Reconstructing full tensors from shards...")
        final_state_dict = OrderedDict()
        
        for key, value in merged_state_dict.items():
            if isinstance(value, dict) and 'local_shards' in value:
                # ShardedTensor 재구성
                total_size = value['total_size']
                local_shards = value['local_shards']
                
                print(f"    Reconstructing {key}: {total_size} from {len(local_shards)} shards")
                
                # 전체 텐서 생성
                if len(total_size) == 1:
                    full_tensor = torch.zeros(total_size[0], dtype=local_shards[0]['tensor'].dtype)
                elif len(total_size) == 2:
                    full_tensor = torch.zeros(total_size[0], total_size[1], dtype=local_shards[0]['tensor'].dtype)
                else:
                    full_tensor = torch.zeros(*total_size, dtype=local_shards[0]['tensor'].dtype)
                
                # 각 shard를 올바른 위치에 복사
                for shard_info in local_shards:
                    tensor = shard_info['tensor']
                    metadata = shard_info['metadata']
                    
                    # shard의 위치 정보 사용
                    if hasattr(metadata, 'shard_offsets') and hasattr(metadata, 'shard_sizes'):
                        offsets = metadata.shard_offsets
                        sizes = metadata.shard_sizes
                        
                        if len(total_size) == 1:
                            full_tensor[offsets[0]:offsets[0]+sizes[0]] = tensor
                        elif len(total_size) == 2:
                            full_tensor[offsets[0]:offsets[0]+sizes[0], 
                                      offsets[1]:offsets[1]+sizes[1]] = tensor
                        else:
                            # 다차원 텐서 처리
                            slices = tuple(slice(offsets[i], offsets[i]+sizes[i]) for i in range(len(total_size)))
                            full_tensor[slices] = tensor
                    else:
                        # 메타데이터가 없는 경우 단순 복사
                        full_tensor = tensor
                        break
                
                final_state_dict[key] = full_tensor
            else:
                # 일반 텐서는 그대로 사용
                final_state_dict[key] = value
        
        # 병합된 체크포인트 저장
        merged_checkpoint = {
            'model_state_dict': final_state_dict,
            'epoch': epoch
        }
        
        torch.save(merged_checkpoint, output_path)
        print(f"✅ Merged checkpoint saved to: {output_path}")
    
    # 모든 프로세스가 완료될 때까지 대기
    dist.barrier()
    
    # 각 rank가 자신의 임시 파일만 정리
    if os.path.exists(temp_file):
        os.remove(temp_file)
        print(f"[Rank {rank}] Cleaned up {temp_file}")
    
    print(f"[Rank {rank}] Merge process completed")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    
    # 분산 환경 초기화
    dist.init_process_group(backend='nccl')
    
    try:
        merge_fsdp_shards()
    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()