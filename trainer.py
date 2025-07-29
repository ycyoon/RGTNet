import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import GradScaler, autocast
import json
import time
from tqdm import tqdm
import os
import torch.distributed as dist
from datetime import datetime
from model import load_checkpoint, load_sharded_checkpoint


def cleanup_memory():
    """Clean up GPU memory between epochs"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def create_timestamped_save_path(base_path):
    """Create timestamped save path with date and time"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Extract directory and filename from base_path
    base_dir = os.path.dirname(base_path)
    base_name = os.path.basename(base_path)
    name_without_ext = os.path.splitext(base_name)[0]
    ext = os.path.splitext(base_name)[1]
    
    # Create timestamped directory
    timestamped_dir = os.path.join(base_dir, f"{name_without_ext}_{timestamp}")
    os.makedirs(timestamped_dir, exist_ok=True)
    
    # Return the full path for the model file
    return os.path.join(timestamped_dir, f"{name_without_ext}{ext}"), timestamped_dir

def train_model(model, train_loader, val_loader, device, args, local_rank=0, logger=None):
    """Train the decoder-only (causal LM) model"""
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    total_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # Initialize GradScaler for AMP
    scaler = GradScaler(enabled=args.use_amp) if torch.cuda.is_available() else None
    
    model.train()
    best_val_loss = float('inf')
    training_stats = {
        'train_losses': [],
        'val_losses': [],
        'learning_rates': [],
        'epochs': [],
        'benchmark_results': []
    }
    
    # Create timestamped save path
    timestamped_save_path, timestamped_dir = create_timestamped_save_path(args.save_path)
    args.timestamped_save_path = timestamped_save_path
    args.timestamped_dir = timestamped_dir
    
    if is_main_process():
        print(f"📁 Model will be saved to: {timestamped_dir}")
        print(f"📄 Model file path: {timestamped_save_path}")
        if logger:
            logger.info(f"Model save directory: {timestamped_dir}")
            logger.info(f"Model file path: {timestamped_save_path}")
    
    start_epoch = 0
    if getattr(args, 'resume_from_checkpoint', None) and os.path.exists(args.resume_from_checkpoint):
        try:
            # Try to load as merged checkpoint first
            from model import load_checkpoint
            start_epoch = load_checkpoint(model, optimizer, args.resume_from_checkpoint, device, is_main_process())
            if is_main_process():
                print(f"✅ Resuming training from merged checkpoint: {args.resume_from_checkpoint}")
                print(f"Starting from epoch {start_epoch}...")
        except Exception as e:
            if is_main_process():
                print(f"⚠️  Failed to load merged checkpoint: {e}")
                print("   Trying to load as sharded checkpoint...")
            
            try:
                # Try to load as sharded checkpoint
                from model import load_sharded_checkpoint
                start_epoch = load_sharded_checkpoint(model, optimizer, args.resume_from_checkpoint, device)
                if is_main_process():
                    print(f"✅ Resuming training from sharded checkpoint: {args.resume_from_checkpoint}")
                    print(f"Starting from epoch {start_epoch}...")
            except Exception as e2:
                if is_main_process():
                    print(f"⚠️  Failed to load sharded checkpoint: {e2}. Starting from scratch.")
                start_epoch = 0
    elif getattr(args, 'resume', False) and os.path.exists(args.save_path):
        try:
            # Pass is_main_process() to prevent duplicate logging
            start_epoch = load_sharded_checkpoint(model, optimizer, args.save_path, device, is_main_process())
            if is_main_process():
                print(f"Resuming training from epoch {start_epoch}...")
        except Exception as e:
            if is_main_process():
                print(f"⚠️  Failed to load checkpoint for resuming: {e}. Starting from scratch.")
            start_epoch = 0

    # Adjust total_steps and scheduler if resuming to ensure LR warmup continues correctly
    if start_epoch > 0:
        completed_steps = (len(train_loader) // args.gradient_accumulation_steps) * start_epoch
        scheduler.last_epoch = completed_steps - 1  # set to last finished step
    
    if is_main_process():
        print(f"Starting training for {args.epochs} epochs...")
        if logger:
            logger.info(f"Starting training for {args.epochs} epochs...")
            logger.info(f"Learning rate: {args.lr}, Batch size: {args.batch_size}")
            # Model parameters are now auto-detected from pretrained model
            logger.info(f"Pretrained model: {getattr(args, 'pretrained_model_name', 'None')}")
            logger.info(f"Model parameters will be auto-detected from pretrained model config")
    
    for epoch in range(start_epoch, args.epochs):
        # Clean up memory at the start of each epoch
        cleanup_memory()
        
        if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        epoch_start_time = time.time()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not is_main_process())
        
        should_break_epoch = False
        for batch_idx, batch in enumerate(progress_bar):
            if hasattr(args, 'max_iters') and args.max_iters is not None and batch_idx >= args.max_iters:
                if is_main_process():
                    print(f"Reached max_iters ({args.max_iters}), stopping training for this epoch.")
                should_break_epoch = True
                break

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            role_mask = batch['role_mask'].to(device)
            
            # AMP: autocast context manager
            with autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                outputs = model(input_ids=input_ids, role_mask=role_mask, labels=labels)
                loss = outputs['loss']
                if loss.dim() > 0:
                    loss = loss.mean()
                
                # Scale loss for gradient accumulation
                loss = loss / args.gradient_accumulation_steps

            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            total_loss += loss.item() * args.gradient_accumulation_steps # Unscale for logging
            num_batches += 1
            
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0 and is_main_process():
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * args.gradient_accumulation_steps:.4f}',
                    'avg_loss': f'{total_loss/num_batches:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
        
        # After each epoch, evaluate
        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # Only run validation if not in train_only mode
        if not getattr(args, 'train_only', False):
            val_loss = evaluate_model(model, val_loader, device, args)
        else:
            val_loss = float('inf')  # Set to infinity when skipping validation
            if is_main_process():
                print("Skipping validation (train_only mode)")
        
        if is_main_process():
            training_stats['train_losses'].append(avg_train_loss)
            training_stats['val_losses'].append(val_loss)
            training_stats['learning_rates'].append(scheduler.get_last_lr()[0])
            training_stats['epochs'].append(epoch + 1)
        
        # Save checkpoint logic - 매 epoch마다 완전한 checkpoint 저장
        should_save = False
        
        # rank 0에서만 저장 여부 결정
        if is_main_process():
            if not getattr(args, 'train_only', False) and val_loss < best_val_loss:
                best_val_loss = val_loss
                should_save = True
                print(f"New best validation loss: {val_loss:.4f}")
            elif getattr(args, 'train_only', False):
                # train_only 모드에서는 매 epoch 저장
                should_save = True
        
        # 저장 여부와 best_val_loss를 모든 rank에 브로드캐스트
        if dist.is_initialized():
            should_save_tensor = torch.tensor(should_save, device=device, dtype=torch.bool)
            dist.broadcast(should_save_tensor, src=0)
            should_save = should_save_tensor.item()
            
            # best_val_loss 브로드캐스트 (validation 모드일 때만)
            if not getattr(args, 'train_only', False):
                best_val_loss_tensor = torch.tensor(best_val_loss, device=device, dtype=torch.float32)
                dist.broadcast(best_val_loss_tensor, src=0)
                best_val_loss = best_val_loss_tensor.item()
        
        # 모든 rank가 참여하여 저장
        if should_save:
            from model import save_checkpoint, save_sharded_checkpoint
            save_start_time = time.time()
            
            # Save sharded checkpoint and automatically merge
            save_sharded_checkpoint(model, optimizer, epoch, args)
            
            # Wait for all processes to complete saving
            if dist.is_initialized():
                dist.barrier()
            
            save_duration = time.time() - save_start_time
            
            if is_main_process():
                print(f"✅ Sharded checkpoint saved and merged at epoch {epoch+1} (took {save_duration:.2f}s)")
                print(f"📁 Checkpoint location: {args.timestamped_dir}")
                if logger:
                    logger.info(f"Checkpoint saved at epoch {epoch+1} in {args.timestamped_dir}")
        
        # Check if max_iters limit is reached
        if hasattr(args, 'max_iters') and args.max_iters is not None and args.max_iters > 0:
            total_iters = (epoch * len(train_loader)) + batch_idx + 1
            if total_iters >= args.max_iters:
                if is_main_process():
                    print(f"\n⚠️  Reached max_iters limit ({args.max_iters}). Stopping training.")
                    print("Saving final checkpoint before early termination...")
                
                # Save checkpoint on early termination
                should_save_early = True
                
                # Broadcast early termination save decision
                if dist.is_initialized():
                    should_save_early_tensor = torch.tensor(should_save_early, device=device, dtype=torch.bool)
                    dist.broadcast(should_save_early_tensor, src=0)
                    should_save_early = should_save_early_tensor.item()
                
                # All ranks participate in checkpoint saving
                if should_save_early:
                    from model import save_checkpoint, save_sharded_checkpoint
                    save_sharded_checkpoint(model, optimizer, epoch, args, force_merge=True)
                    
                    # Wait for all processes to complete saving
                    if dist.is_initialized():
                        dist.barrier()
                    
                    if is_main_process():
                        print("✅ Early termination checkpoint saved and merged")
                        print(f"📁 Checkpoint location: {args.timestamped_dir}")
                        if logger:
                            logger.info(f"Early termination checkpoint saved in {args.timestamped_dir}")
                
                should_break_epoch = True
                break
        
        epoch_time = time.time() - epoch_start_time
        
        if is_main_process():
            # Import log_training_progress here to avoid circular import
            from utils import log_training_progress
            
            print(f"Epoch {epoch+1}/{args.epochs} completed in {epoch_time:.2f}s")
            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
            print("-" * 50)
            
            # Log training progress
            if logger:
                log_training_progress(logger, epoch+1, args.epochs, avg_train_loss, val_loss, 
                                    scheduler.get_last_lr()[0], epoch_time, best_val_loss)
            
        if should_break_epoch:
            if dist.is_initialized():
                dist.barrier(device_ids=[local_rank])
            break

    results = {
        'best_val_loss': best_val_loss,
        'final_train_loss': training_stats['train_losses'][-1] if training_stats['train_losses'] else 0,
        'final_val_loss': training_stats['val_losses'][-1] if training_stats['val_losses'] else 0,
        'training_stats': training_stats,
        'total_epochs': args.epochs,
        'model_path': args.timestamped_save_path,
        'model_dir': args.timestamped_dir
    }
    if is_main_process():
        print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
        if logger:
            logger.info(f"Training completed! Best validation loss: {best_val_loss:.4f}")
        if training_stats['benchmark_results']:
            print("\n📈 BENCHMARK PROGRESS SUMMARY")
            print("="*50)
            if logger:
                logger.info("BENCHMARK PROGRESS SUMMARY")
            for result in training_stats['benchmark_results']:
                msg = f"Epoch {result['epoch']}: ASR = {result.get('overall_asr', 'N/A'):.4f}"
                print(msg)
                if logger:
                    logger.info(msg)
    return results

def evaluate_model(model, val_loader, device, args):
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation", disable=not is_main_process())
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            role_mask = batch['role_mask'].to(device)
            outputs = model(input_ids=input_ids, role_mask=role_mask, labels=labels)
            loss = outputs['loss']
            total_loss += loss.item()
            num_batches += 1
            
    # Synchronize across all processes
    if dist.is_initialized():
        total_loss_tensor = torch.tensor(total_loss, device=device, dtype=torch.float32)
        num_batches_tensor = torch.tensor(num_batches, device=device, dtype=torch.int64)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
        avg_loss = total_loss_tensor.item() / num_batches_tensor.item() if num_batches_tensor.item() > 0 else 0.0
    else:
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
    model.train()
    return avg_loss