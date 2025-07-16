import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import json
import time
from tqdm import tqdm

def train_model(model, head, train_loader, val_loader, device, args):
    """Train the RoleAwareTransformer model"""
    
    # Move models to device
    model = model.to(device)
    head = head.to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    model.train()
    head.train()
    
    best_val_loss = float('inf')
    training_stats = {
        'train_losses': [],
        'val_losses': [],
        'learning_rates': [],
        'epochs': []
    }
    
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        total_loss = 0
        num_batches = 0
        
        # Training phase
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Get model output
            outputs = model(input_ids, attention_mask=attention_mask)
            
            # Use pooled output (mean of sequence)
            pooled_output = outputs.mean(dim=1)
            logits = head(pooled_output)
            
            # Calculate loss (using dummy loss for now)
            loss = nn.MSELoss()(logits.squeeze(), labels.float())
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        # Calculate average training loss
        avg_train_loss = total_loss / num_batches
        
        # Validation phase
        val_loss = evaluate_model(model, head, val_loader, device)
        
        # Update training stats
        training_stats['train_losses'].append(avg_train_loss)
        training_stats['val_losses'].append(val_loss)
        training_stats['learning_rates'].append(scheduler.get_last_lr()[0])
        training_stats['epochs'].append(epoch + 1)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            from model import save_checkpoint
            save_checkpoint(model, head, optimizer, scheduler, epoch, val_loss, args.save_path)
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{args.epochs} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
        print("-" * 50)
    
    # Final results
    results = {
        'best_val_loss': best_val_loss,
        'final_train_loss': training_stats['train_losses'][-1],
        'final_val_loss': training_stats['val_losses'][-1],
        'training_stats': training_stats,
        'total_epochs': args.epochs,
        'model_path': args.save_path
    }
    
    print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    return results

def evaluate_model(model, head, val_loader, device):
    """Evaluate the model on validation set"""
    model.eval()
    head.eval()
    
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            pooled_output = outputs.mean(dim=1)
            logits = head(pooled_output)
            
            # Calculate loss
            loss = nn.MSELoss()(logits.squeeze(), labels.float())
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    head.train()
    
    return total_loss / num_batches if num_batches > 0 else 0.0