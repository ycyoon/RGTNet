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
from model import load_checkpoint


def cleanup_memory():
    """Clean up GPU memory between epochs"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

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
    
    start_epoch = 0
    if getattr(args, 'resume_from_checkpoint', None) and os.path.exists(args.resume_from_checkpoint):
        try:
            # Pass is_main_process() to prevent duplicate logging
            start_epoch = load_checkpoint(model, optimizer, args.resume_from_checkpoint, device, is_main_process())
            if is_main_process():
                print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
                print(f"Starting from epoch {start_epoch}...")
        except Exception as e:
            if is_main_process():
                print(f"⚠️  Failed to load checkpoint: {e}. Starting from scratch.")
            start_epoch = 0
    elif getattr(args, 'resume', False) and os.path.exists(args.save_path):
        try:
            # Pass is_main_process() to prevent duplicate logging
            start_epoch = load_checkpoint(model, optimizer, args.save_path, device, is_main_process())
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
        val_loss = evaluate_model(model, val_loader, device, args)
        
        if is_main_process():
            training_stats['train_losses'].append(avg_train_loss)
            training_stats['val_losses'].append(val_loss)
            training_stats['learning_rates'].append(scheduler.get_last_lr()[0])
            training_stats['epochs'].append(epoch + 1)
        
        # Save checkpoint every epoch
        if is_main_process():
            from model import save_checkpoint
            epoch_ckpt_path = args.save_path.replace('.pth', f'_epoch{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch, epoch_ckpt_path)

        if is_main_process() and val_loss < best_val_loss:
            best_val_loss = val_loss
            from model import save_checkpoint
            # Corrected the arguments for save_checkpoint
            save_checkpoint(model, optimizer, epoch, args.save_path)

        # Run benchmark evaluation if enabled
        if is_main_process() and getattr(args, 'enable_benchmark', False) and (epoch + 1) % getattr(args, 'benchmark_freq', 10) == 0:
            if logger:
                logger.info(f"Running benchmark evaluation at epoch {epoch + 1}")
            try:
                # Placeholder for benchmark evaluation - function not yet implemented
                # benchmark_result = run_benchmark_evaluation(model, epoch + 1, device, args, logger)
                # if benchmark_result:
                #     training_stats['benchmark_results'].append(benchmark_result)
                #     if logger:
                #         logger.info(f"Benchmark completed: ASR = {benchmark_result.get('overall_asr', 'N/A')}")
                if logger:
                    logger.info(f"Benchmark evaluation placeholder - epoch {epoch + 1}")
            except Exception as e:
                if logger:
                    logger.warning(f"Benchmark evaluation failed: {e}")
                print(f"Warning: Benchmark evaluation failed: {e}")

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
        'model_path': args.save_path
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