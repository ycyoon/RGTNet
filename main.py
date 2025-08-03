#!/usr/bin/env python3
"""
RGTNet Main Entry Point with DeepSpeed
"""
import os
import time
from transformers import AutoTokenizer
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import argparse

# DeepSpeed imports
import deepspeed

# Import our modules
from config import setup_args, setup_environment, create_directories
from model_hybrid import create_hybrid_model, create_model  # ✅ LoRA 지원 복원
from data_loader import download_instruction_datasets, create_data_loaders, load_data_from_files
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
    
    try:
        # DDP 환경 변수 및 초기화 - 원래 설정으로 복원
        is_ddp = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
        if is_ddp:
            rank = int(os.environ['RANK'])
            local_rank = int(os.environ.get('LOCAL_RANK', rank))
            world_size = int(os.environ['WORLD_SIZE'])
            
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
            
            # 원래 NCCL 설정으로 복원 (문제가 되었던 부분 제거)
            dist.init_process_group(backend='nccl')
            dist.barrier()
            
            if rank == 0:
                print(f"DDP enabled. World size: {world_size}")
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            rank = 0
            world_size = 1
        
        # rank==0에서만 디렉토리 생성
        if rank == 0:
            create_directories(args)
            
            # 🔧 FIX: 3개 반환값 모두 받기
            from trainer import create_timestamped_save_path
            args.timestamped_dir = create_timestamped_save_path(args.save_path, args.pretrained_model_name)
            
            print(f"📁 Timestamped directory: {args.timestamped_dir}")

        if is_ddp:
            dist.barrier()
            
            # 🔧 FIX: 경로를 모든 rank에 브로드캐스트
            if rank != 0:
                # 더미 값으로 초기화 (브로드캐스트로 실제 값 받을 예정)
                args.timestamped_dir = ""
            
            # 문자열 브로드캐스트 (간단한 방법)
            import pickle
            if rank == 0:
                path_data = {
                    'timestamped_dir': args.timestamped_dir,
                }
                path_bytes = pickle.dumps(path_data)
                path_size = len(path_bytes)
            else:
                path_size = 0
            
            # 크기 브로드캐스트
            size_tensor = torch.tensor(path_size, dtype=torch.int64, device=device)
            dist.broadcast(size_tensor, src=0)
            
            # 데이터 브로드캐스트
            if rank == 0:
                data_tensor = torch.tensor(list(path_bytes), dtype=torch.uint8, device=device)
            else:
                data_tensor = torch.zeros(size_tensor.item(), dtype=torch.uint8, device=device)
            
            dist.broadcast(data_tensor, src=0)
            
            if rank != 0:
                path_data = pickle.loads(bytes(data_tensor.cpu().numpy()))
                args.timestamped_dir = path_data['timestamped_dir']
        
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
            print(f"Loading tokenizer from pretrained model: {args.pretrained_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)

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
        
        # Use hybrid model for LoRA support
        if getattr(args, 'use_lora', False) or getattr(args, 'enable_role_adapters', False):
            if rank == 0:
                print("Using hybrid model with LoRA support...")
            model, _ = create_hybrid_model(args)  # create_hybrid_model returns (model, tokenizer)
        else:
            if rank == 0:
                print("Using standard RGTNet model...")
            model = create_model(args, tokenizer.pad_token_id)
        
        # 🔧 FIX: Ensure all model parameters are float32 before DeepSpeed initialization
        for param in model.parameters():
            if param.dtype != torch.float32:
                param.data = param.data.to(torch.float32)
        
        if rank == 0:
            print_model_info(model, None)  # Pass None for head since it's integrated in the model
        model = model.to(device)
        
        # DeepSpeed initialization with improved checkpoint handling
        if args.deepspeed:
            if rank == 0:
                print("Initializing DeepSpeed...")
            
            # Parse DeepSpeed arguments
            parser = argparse.ArgumentParser()
            parser = deepspeed.add_config_arguments(parser)
            ds_args = parser.parse_args([])
            

            
            # Initialize DeepSpeed
            model, _, _, _ = deepspeed.initialize(
                args=ds_args,
                model=model,
                model_parameters=model.parameters(),
                config=args.deepspeed_config
            )
            
            if rank == 0:
                print("DeepSpeed initialization completed successfully")
                
                # Copy zero_to_fp32.py to unified checkpoint directory for later use
                import shutil
                zero_script_src = os.path.join(os.path.dirname(deepspeed.__file__), 
                                             "utils", "zero_to_fp32.py")
                if os.path.exists(zero_script_src):
                    zero_script_dst = os.path.join(args.timestamped_dir, "zero_to_fp32.py")
                    shutil.copy2(zero_script_src, zero_script_dst)
                    print(f"📋 Copied zero_to_fp32.py to {zero_script_dst}")
                else:
                    print("⚠️  Could not find zero_to_fp32.py in",zero_script_src)

        
        # Data preparation
        train_data, val_data = None, None

        if args.download_datasets:
            if rank == 0:
                print("Downloading datasets...")
            download_instruction_datasets()
            if is_ddp:
                dist.barrier()
        
        if args.train_file and args.val_file:
            if rank == 0:
                print(f"Loading data from files: {args.train_file}, {args.val_file}")
            train_data, val_data = load_data_from_files(args.train_file, args.val_file)
        else:
            if rank == 0:
                print("No train/val files specified, using default datasets")
            train_data, val_data = download_instruction_datasets()
        
        # Create datasets
        if rank == 0:
            print("Creating datasets...")
        
        # 🔧 FIX: Use GenerationDataset instead of InstructionDataset
        from data_loader import GenerationDataset
        train_dataset = GenerationDataset(train_data, tokenizer, args.max_seq_len)
        val_dataset = GenerationDataset(val_data, tokenizer, args.max_seq_len)
        
        if rank == 0:
            print(f"Train dataset size: {len(train_dataset)}")
            print(f"Val dataset size: {len(val_dataset)}")
        
        # Create data loaders
        if is_ddp:
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
            
            # 🔧 FIX: DeepSpeed가 배치 크기를 관리하므로 로그 메시지만 출력
            if rank == 0:
                print("Batch size managed by DeepSpeed configuration (ds_config.json)")
            
            train_loader, val_loader = create_data_loaders(
                train_dataset, val_dataset, tokenizer, args, 
                train_sampler=train_sampler, val_sampler=val_sampler
            )
        else:
            # For non-DDP, we don't need samplers.
            train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, tokenizer, args)
        
        # Training
        training_results = None
        if not args.eval_only:
            if rank == 0:
                print("Starting training...")
                if logger:
                    logger.info("Starting model training...")
            # Enable auto-merge by default for better integration
            if not hasattr(args, 'auto_merge_checkpoint'):
                args.auto_merge_checkpoint = True
            
            training_results = train_model(model, train_loader, val_loader, device, args, local_rank=local_rank, logger=logger, tokenizer=tokenizer)
            
            # Save training results and log final performance
            if rank == 0:
                # Create timestamped results directory
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_dir = os.path.join(os.path.dirname(args.results_file), f"results_{timestamp}")
                os.makedirs(results_dir, exist_ok=True)
                
                # Save training results in timestamped directory
                training_results_file = os.path.join(results_dir, "training_results.json")
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
                # Use the same timestamped results directory
                eval_results_file = os.path.join(results_dir, "evaluation_results.json")
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
            try:
                # 🔧 FIX: DDP 정리 개선 - 모든 rank가 동기화된 후 정리
                dist.barrier()
                dist.destroy_process_group()
                if rank == 0:
                    print("✅ DDP process group cleaned up successfully")
            except Exception as e:
                if rank == 0:
                    print(f"⚠️  Warning: Error during DDP cleanup: {e}")
                # Force cleanup even if there's an error
                try:
                    dist.destroy_process_group()
                except:
                    pass
    
    except Exception as e:
        # Handle any unexpected errors
        if rank == 0:
            print(f"\n❌ Unexpected error occurred: {e}")
        
        # Re-raise the exception
        raise

if __name__ == '__main__':
    main()
