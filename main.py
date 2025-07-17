#!/usr/bin/env python3
"""
RGTNet Main Entry Point
"""
import os
import sys
import time
from transformers import AutoTokenizer

# Import our modules
from config import setup_args, setup_environment, get_device, create_directories
from model import create_model, CheckpointedRoleAwareTransformerLayer
from data_loader import download_instruction_datasets, create_data_loaders, load_data_from_files, GenerationDataset
from trainer import train_model
from evaluator import evaluate_model_detailed, save_evaluation_results, print_evaluation_summary
from utils import set_seed, print_model_info, format_time, check_gpu_memory, save_results
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, CPUOffload
from torch.utils.data.distributed import DistributedSampler


def main():
    """Main execution function"""
    start_time = time.time()
    
    # Setup
    args = setup_args()
    setup_environment()
    set_seed(42)
    
    # DDP 환경 변수 및 초기화
    is_ddp = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if is_ddp:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        world_size = int(os.environ['WORLD_SIZE'])
        
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        # Limit initial reserved memory to reduce fragmentation
        try:
            torch.cuda.set_per_process_memory_fraction(0.9, device)
        except AttributeError:
            pass  # Older PyTorch versions may not have this API
        
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            rank=rank,
            world_size=world_size
        )
        if rank == 0:
            print(f"DDP enabled. World size: {world_size}")
    else:
        # non-DDP case
        device = get_device(args)
        # Limit initial reserved memory in single-GPU case as well
        if torch.cuda.is_available():
            try:
                torch.cuda.set_per_process_memory_fraction(0.9, device)
            except AttributeError:
                pass
        rank = 0
        local_rank = 0
    
    # rank==0에서만 디렉토리 생성
    if rank == 0:
        create_directories(args)
    if is_ddp:
        dist.barrier()  # 모든 프로세스가 디렉토리 생성 후 진행
    
    # Print GPU info
    if rank == 0:
        check_gpu_memory()
    
    # Initialize tokenizer
    if rank == 0:
        print(f"Loading tokenizer: {args.pretrained_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model
    if rank == 0:
        print("Creating model...")
    model, tokenizer = create_model(args, tokenizer)
    if rank == 0:
        print_model_info(model)
    model = model.to(device)
    if is_ddp:
        # Configure FSDP with full sharding and mixed precision
        mp_policy = MixedPrecision(param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16)
        auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={CheckpointedRoleAwareTransformerLayer})
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mp_policy,
            cpu_offload=CPUOffload(offload_params=True),
            device_id=device
        )
    
    # Data preparation
    train_data, val_data = None, None

    if args.download_datasets:
        print("Downloading datasets...")
        train_data, val_data = download_instruction_datasets()
    else:
        print("Loading data from files...")
        train_data, val_data = load_data_from_files(args.train_file, args.val_file, tokenizer, args)

    if not train_data or not val_data:
        print("Error: No training or validation data available")
        sys.exit(1)

    # GenerationDataset 생성
    train_dataset = GenerationDataset(train_data, tokenizer, args.max_seq_len)
    val_dataset = GenerationDataset(val_data, tokenizer, args.max_seq_len)

    # DataLoader에 DistributedSampler 적용
    if is_ddp:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)
        # In DDP, num_workers should often be 0 to avoid issues with shared memory.
        args.num_workers = getattr(args, 'num_workers', 0)
        train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, tokenizer, args, train_sampler=train_sampler, val_sampler=val_sampler)
    else:
        # For non-DDP, we don't need samplers.
        train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, tokenizer, args)
    
    # Training
    if not args.eval_only:
        if rank == 0:
            print("Starting training...")
        training_results = train_model(model, train_loader, val_loader, device, args)
        
        # Save training results
        if rank == 0:
            training_results_file = args.results_file.replace('.json', '_training.json')
            save_results(training_results, training_results_file)
            
            print(f"Training completed in {format_time(time.time() - start_time)}")
    
    # Evaluation
    if not args.train_only:
        if rank == 0:
            print("Starting evaluation...")
        eval_results = evaluate_model_detailed(model, val_loader, device, args)
        
        # Print and save evaluation results
        if rank == 0:
            print_evaluation_summary(eval_results)
            eval_results_file = args.results_file.replace('.json', '_evaluation.json')
            save_evaluation_results(eval_results, eval_results_file)
    
    # Final summary
    if rank == 0:
        total_time = time.time() - start_time
        print(f"\nTotal execution time: {format_time(total_time)}")
        print("✅ Process completed successfully!")
    
    # Clean up the DDP process group at the very end
    if is_ddp:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
