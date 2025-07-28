#!/usr/bin/env python3
"""
RGTNet Main Entry Point
"""
import os
import sys
import time
from datetime import timedelta
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data.distributed import DistributedSampler
from functools import partial

# Import our modules
from config import setup_args, setup_environment, get_device, create_directories
from model import create_model, RoleAwareTransformerLayer, CheckpointedRoleAwareTransformerLayer
from data_loader import download_instruction_datasets, create_data_loaders, load_data_from_files, InstructionDataset, GenerationDataset
from trainer import train_model
from evaluator import evaluate_model_detailed, save_evaluation_results, print_evaluation_summary
from utils import set_seed, print_model_info, format_time, check_gpu_memory, save_results, setup_logging, log_final_performance, log_evaluation_results


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
        
        # Enhanced NCCL configuration for H100 stability
        os.environ.setdefault('NCCL_SOCKET_IFNAME', 'lo,eth0,ib0')
        os.environ.setdefault('NCCL_IB_DISABLE', '0')
        os.environ.setdefault('NCCL_P2P_DISABLE', '0')
        os.environ.setdefault('NCCL_SHM_DISABLE', '0')
        os.environ.setdefault('NCCL_TREE_THRESHOLD', '0')
        os.environ.setdefault('NCCL_ALGO', 'Tree')
        os.environ.setdefault('NCCL_PROTO', 'Simple')
        os.environ.setdefault('NCCL_BUFFSIZE', '8388608')
        os.environ.setdefault('NCCL_NTHREADS', '64')
        os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')
        os.environ.setdefault('NCCL_TIMEOUT', '3600')  # 1 hour timeout
        
        # Limit initial reserved memory to reduce fragmentation
        try:
            # Use device index (local_rank) instead of device object
            torch.cuda.set_per_process_memory_fraction(0.85, local_rank)  # More conservative
        except (AttributeError, ValueError):
            pass
        
        # Initialize with extended timeout
        try:
            if rank == 0:
                print(f"Initializing NCCL with enhanced configuration for {world_size} H100 GPUs")
            
            # Set optimal NCCL environment variables for H100
            os.environ['NCCL_ALGO'] = 'Ring'  # Use Ring instead of Tree for better stability
            os.environ['NCCL_PROTO'] = 'Simple'  # Keep Simple protocol
            os.environ['NCCL_MIN_NCHANNELS'] = '4'  # Optimize for H100
            os.environ['NCCL_MAX_NCHANNELS'] = '16'
            os.environ['NCCL_P2P_DISABLE'] = '1'  # Disable P2P for stability
            os.environ['NCCL_IB_DISABLE'] = '1'  # Disable InfiniBand if causing issues
            os.environ['NCCL_SOCKET_IFNAME'] = 'lo'  # Use loopback for local communication
            
            timeout = timedelta(minutes=30)
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                rank=rank,
                world_size=world_size,
                timeout=timeout
            )
            
            # Warm up communication
            test_tensor = torch.ones(1).cuda(local_rank)
            dist.all_reduce(test_tensor)
            
            if rank == 0:
                print(f"NCCL initialization successful. Communication test passed.")
            
        except Exception as e:
            if rank == 0:
                print(f"NCCL initialization failed: {e}")
            raise
        
        # Synchronize all processes
        dist.barrier()
        if rank == 0:
            print(f"DDP enabled. World size: {world_size}")
    else:
        # non-DDP case
        device = get_device(args)
        # Limit initial reserved memory in single-GPU case as well
        if torch.cuda.is_available():
            try:
                # Extract device index from device object
                device_index = device.index if device.index is not None else 0
                torch.cuda.set_per_process_memory_fraction(0.9, device_index)
            except (AttributeError, ValueError):
                pass
        rank = 0
        local_rank = 0

    # rank==0에서만 디렉토리 생성
    if rank == 0:
        create_directories(args)
    if is_ddp:
        dist.barrier()  # 모든 프로세스가 디렉토리 생성 후 진행
    
    # Setup logging (only on main process)
    logger = setup_logging('logs', is_main_process=(rank == 0))
    if logger:
        logger.info("RGTNet training started")
        logger.info(f"DDP enabled: {is_ddp}, World size: {world_size if is_ddp else 1}")
    
    # Print GPU info
    if rank == 0:
        check_gpu_memory()
    
    # Initialize tokenizer and set pad_token
    if rank == 0:
        print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            if rank == 0:
                print("Tokenizer has no pad_token, using eos_token as pad_token.")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            if rank == 0:
                print("Tokenizer has no pad_token or eos_token, adding a new [PAD] token.")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
    # Resize model embeddings if a new token was added
    if 'pad_token' in tokenizer.special_tokens_map and tokenizer.added_tokens_decoder:
        args.vocab_size = len(tokenizer)
    else:
        args.vocab_size = tokenizer.vocab_size

    # Create model
    if rank == 0:
        print("Creating model...")
    model = create_model(args, tokenizer.pad_token_id)
    if rank == 0:
        print_model_info(model, None)  # Pass None for head since it's integrated in the model
    model = model.to(device)
    if is_ddp:
        # Enhanced FSDP configuration for H100 stability
        if rank == 0:
            print("Configuring FSDP for distributed training...")
        
        # More conservative mixed precision for stability
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,  # Better numerical stability than float16
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.float32   # Keep buffers in float32
        )
        
        # Auto wrap policy for transformer layers
        # Use RoleAwareTransformerLayer instead of CheckpointedRoleAwareTransformerLayer to avoid gradient checkpointing issues
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy, 
            transformer_layer_cls={RoleAwareTransformerLayer}
        )
        
        # FSDP with conservative settings for NCCL stability
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mp_policy,
            cpu_offload=CPUOffload(offload_params=False),  # Disable CPU offload for speed
            device_id=local_rank,
            sync_module_states=True,  # Ensure consistent initialization
            param_init_fn=None,
            use_orig_params=True,   # Use original parameters for better NCCL compatibility
            limit_all_gathers=True,  # Reduce memory usage
            forward_prefetch=True,   # Enable prefetching for better performance
            backward_prefetch=True   # Enable backward prefetching
        )
        
        if rank == 0:
            print("FSDP model wrapping completed successfully")
    
    # Data preparation
    train_data, val_data = None, None

    if args.download_datasets:
        if rank == 0:
            print("Downloading datasets...")
        train_data, val_data = download_instruction_datasets()
    else:
        if rank == 0:
            print("Loading data from files...")
        train_data, val_data = load_data_from_files(args.train_file, args.val_file, tokenizer, args)

    if not train_data or not val_data:
        if rank == 0:
            print("Error: No training or validation data available")
            sys.exit(1)

    # GenerationDataset 생성
    train_dataset = GenerationDataset(train_data, tokenizer, args.max_seq_len)
    val_dataset = GenerationDataset(val_data, tokenizer, args.max_seq_len)

    # DataLoader에 DistributedSampler 적용
    if is_ddp:
        if rank == 0:
            print("Setting up distributed data loaders...")
        
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False  # Changed to False to prevent batch size inconsistency
        )
        val_sampler = DistributedSampler(
            val_dataset, 
            num_replicas=world_size,
            rank=rank,
            shuffle=False, 
            drop_last=False  # Changed to False to prevent batch size inconsistency
        )
        
        # Conservative num_workers for H100 stability
        args.num_workers = min(getattr(args, 'num_workers', 4), 4)
        
        # Set batch size per GPU
        effective_batch_size = getattr(args, 'batch_size', 8) // world_size
        effective_batch_size = max(effective_batch_size, 1)  # Ensure at least 1
        
        if rank == 0:
            print(f"Effective batch size per GPU: {effective_batch_size}")
            print(f"Total effective batch size: {effective_batch_size * world_size}")
        
        train_loader, val_loader = create_data_loaders(
            train_dataset, val_dataset, tokenizer, args, 
            train_sampler=train_sampler, val_sampler=val_sampler
        )
    else:
        # For non-DDP, we don't need samplers.
        train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, tokenizer, args)
    
    # Training
    if not args.eval_only:
        if rank == 0:
            print("Starting training...")
            if logger:
                logger.info("Starting model training...")
        training_results = train_model(model, train_loader, val_loader, device, args, local_rank=local_rank, logger=logger)
        
        # Save training results and log final performance
        if rank == 0:
            training_results_file = args.results_file.replace('.json', '_training.json')
            save_results(training_results, training_results_file)
            
            # Log final performance summary
            if logger:
                log_final_performance(logger, training_results, args.save_path)
            
            print(f"Training completed in {format_time(time.time() - start_time)}")
    
    # Evaluation
    if not args.train_only:
        # Wait for all processes to finish training before starting evaluation
        if is_ddp:
            dist.barrier()
            
        if rank == 0:
            print("Starting detailed evaluation...")
            if logger:
                logger.info("Starting detailed evaluation...")
        
        # Run evaluation on all processes, but save/print results only on rank 0
        eval_results = evaluate_model_detailed(model, val_loader, tokenizer, device, args)
        
        # Print and save evaluation results
        if rank == 0:
            print_evaluation_summary(eval_results)
            eval_results_file = args.results_file.replace('.json', '_evaluation.json')
            save_evaluation_results(eval_results, eval_results_file)
            
            # Log evaluation results
            if logger:
                log_evaluation_results(logger, eval_results)
    
    # Final summary
    if rank == 0:
        total_time = time.time() - start_time
        print(f"\nTotal execution time: {format_time(total_time)}")
        print("✅ Process completed successfully!")
        
        if logger:
            logger.info(f"Total execution time: {format_time(total_time)}")
            logger.info("✅ Process completed successfully!")
    
    # Clean up the DDP process group at the very end
    if is_ddp:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
