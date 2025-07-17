import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import GradScaler, autocast
import json
import time
from tqdm import tqdm
import os

def run_benchmark_evaluation(model, head, tokenizer, device, args, epoch):
    """Run StructTransform benchmark evaluation during training (논문 방식: HarmBench judge + Refusal judge)"""
    try:
        from structtransform_benchmark import StructTransformEvaluator, EvaluationConfig, StructTransformDataset
        import os
        # Create benchmark config
        benchmark_config = EvaluationConfig(
            model_path="",  # Not needed for evaluation during training
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
            tokenizer_name=args.tokenizer_name,
            bias_delta=args.bias_delta,
            batch_size=8,  # Smaller batch size for faster evaluation
            device=device,
            benchmark_dir=args.benchmark_dir if hasattr(args, 'benchmark_dir') else "StructTransformBench/benchmark"
        )
        # 논문 방식 evaluator (HarmBench judge + Refusal judge)
        evaluator = StructTransformEvaluator(model, tokenizer, device, benchmark_config)
        print(f"\n{'='*60}")
        print(f"BENCHMARK EVALUATION (논문 방식, HarmBench judge + Refusal judge) - Epoch {epoch+1}")
        print(f"{'='*60}")
        # 빠른 평가를 위해 대표 구조만 사용하거나 전체 구조 평가
        results = evaluator.evaluate_all_structures()
        # Save intermediate results
        benchmark_results = {
            'epoch': epoch + 1,
            'results': results,
            'timestamp': time.time()
        }
        benchmark_file = f"benchmark_results_epoch_{epoch+1}_full.json"
        with open(benchmark_file, 'w') as f:
            import json
            json.dump(benchmark_results, f, indent=2)
        print(f"Benchmark results saved to {benchmark_file}")
        print(f"{'='*60}\n")
        return benchmark_results
    except Exception as e:
        print(f"⚠️  Benchmark evaluation failed: {e}")
        return None

def train_model(model, train_loader, val_loader, device, args):
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
    
    print(f"Starting training for {args.epochs} epochs...")
    benchmark_freq = getattr(args, 'benchmark_freq', 5)
    
    for epoch in range(args.epochs):
        if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        epoch_start_time = time.time()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
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
            
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * args.gradient_accumulation_steps:.4f}',
                    'avg_loss': f'{total_loss/num_batches:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
        
        # After each epoch, evaluate
        avg_train_loss = total_loss / num_batches
        val_loss = evaluate_model(model, val_loader, device)
        training_stats['train_losses'].append(avg_train_loss)
        training_stats['val_losses'].append(val_loss)
        training_stats['learning_rates'].append(scheduler.get_last_lr()[0])
        training_stats['epochs'].append(epoch + 1)
        benchmark_result = None
        if getattr(args, 'enable_benchmark', False) and ((epoch + 1) % benchmark_freq == 0 or epoch == 0):
            print(f"\n🔄 Running benchmark evaluation at epoch {epoch+1}...")
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                benchmark_result = run_benchmark_evaluation(model, tokenizer, device, args, epoch)
                if benchmark_result:
                    training_stats['benchmark_results'].append(benchmark_result)
            except Exception as e:
                print(f"⚠️  Could not run benchmark evaluation: {e}")
        if val_loss < best_val_loss:
            from model import save_checkpoint
            save_checkpoint(model, None, optimizer, scheduler, epoch, val_loss, args.save_path)
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{args.epochs} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
        if benchmark_result:
            print(f"Benchmark ASR: {benchmark_result.get('overall_asr', 0):.4f}")
        print("-" * 50)
    results = {
        'best_val_loss': best_val_loss,
        'final_train_loss': training_stats['train_losses'][-1],
        'final_val_loss': training_stats['val_losses'][-1],
        'training_stats': training_stats,
        'total_epochs': args.epochs,
        'model_path': args.save_path
    }
    print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    if training_stats['benchmark_results']:
        print("\n📈 BENCHMARK PROGRESS SUMMARY")
        print("="*50)
        for result in training_stats['benchmark_results']:
            print(f"Epoch {result['epoch']}: ASR = {result.get('overall_asr', 0):.4f}")
    return results

def evaluate_model(model, val_loader, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            role_mask = batch['role_mask'].to(device)
            outputs = model(input_ids=input_ids, role_mask=role_mask, labels=labels)
            loss = outputs['loss']
            total_loss += loss.item()
            num_batches += 1
    model.train()
    return total_loss / num_batches if num_batches > 0 else 0.0