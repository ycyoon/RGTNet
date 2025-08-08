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
import deepspeed
import shutil
import json
from transformers import AutoTokenizer, AutoConfig
from datetime import timezone, timedelta
# PEFT models handle checkpointing through the transformers/peft libraries


def cleanup_memory():
    """Clean up GPU memory between epochs"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def create_huggingface_config_files(model, tokenizer, output_dir, logger=None):
    """
    Create HuggingFace-style config files for the trained model
    
    Args:
        model: The trained hybrid model
        tokenizer: The tokenizer used for training
        output_dir: Directory to save config files
        logger: Optional logger for logging
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base model config (handle DeepSpeed wrapped models)
        actual_model = model
        if hasattr(model, 'module'):  # DeepSpeed wrapped model
            actual_model = model.module
            print("🔧 Detected DeepSpeed wrapped model, accessing underlying model")
        
        if hasattr(actual_model, 'base_model') and hasattr(actual_model.base_model, 'config'):
            base_config = actual_model.base_model.config
            print(f"✅ Found base model config: {base_config._name_or_path}")
        elif hasattr(actual_model, 'config'):
            base_config = actual_model.config
            print(f"✅ Found model config: {base_config._name_or_path}")
        else:
            print("⚠️ Cannot find model config, using default Llama-3.2-3B config")
            base_config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
        
        # 1. Save model config.json
        config_path = os.path.join(output_dir, "config.json")
        base_config.save_pretrained(output_dir)
        print(f"✅ Saved config.json to {config_path}")
        
        # 2. Save tokenizer files
        try:
            # Try to save tokenizer files directly
            tokenizer.save_pretrained(output_dir)
            print(f"✅ Saved tokenizer files to {output_dir}")
        except Exception as e:
            print(f"⚠️ Error saving tokenizer with save_pretrained: {e}")
            # Fallback: copy from cache
            fallback_success = _copy_tokenizer_files_from_cache(tokenizer, output_dir, logger)
            if not fallback_success:
                print(f"⚠️ Failed to copy tokenizer files from cache, continuing without tokenizer files")
        
        # 3. Save generation_config.json
        try:
            if hasattr(base_config, 'to_dict'):
                generation_config = {
                    "bos_token_id": getattr(base_config, 'bos_token_id', 128000),
                    "eos_token_id": getattr(base_config, 'eos_token_id', 128001),
                    "max_length": 2048,
                    "pad_token_id": getattr(base_config, 'pad_token_id', 128001),
                    "do_sample": True,
                    "temperature": 0.6,
                    "top_p": 0.9
                }
                
                generation_config_path = os.path.join(output_dir, "generation_config.json")
                with open(generation_config_path, 'w') as f:
                    json.dump(generation_config, f, indent=2)
                print(f"✅ Saved generation_config.json to {generation_config_path}")
        except Exception as e:
            print(f"⚠️ Error creating generation_config.json: {e}")
        
        # 4. Create a model info file for RGTNet specifics
        rgtnet_info = {
            "model_type": "RGTNet-Hybrid",
            "base_model": getattr(base_config, '_name_or_path', 'meta-llama/Llama-3.2-3B-Instruct'),
            "enable_role_adapters": getattr(actual_model, 'enable_role_adapters', True),
            "has_lora_adapters": hasattr(actual_model, 'base_model') and hasattr(actual_model.base_model, 'peft_config'),
            "created_at": datetime.now().isoformat(),
            "transformers_version": "4.x",
            "torch_dtype": "float32",
            "training_framework": "DeepSpeed" if hasattr(model, 'module') else "PyTorch"
        }
        
        rgtnet_info_path = os.path.join(output_dir, "rgtnet_model_info.json")
        with open(rgtnet_info_path, 'w') as f:
            json.dump(rgtnet_info, f, indent=2)
        print(f"✅ Saved RGTNet model info to {rgtnet_info_path}")
        
        # 5. Verify created files
        verification_success = _verify_config_files(output_dir, logger)
        
        if logger:
            logger.info(f"HuggingFace config files created in {output_dir}")
        
        return verification_success
        
    except Exception as e:
        print(f"❌ Error creating HuggingFace config files: {e}")
        if logger:
            logger.error(f"Error creating HuggingFace config files: {e}")
        return False

def _copy_tokenizer_files_from_cache(tokenizer, output_dir, logger=None):
    """
    Fallback: Copy tokenizer files from HuggingFace cache
    """
    try:
        # Get the model name from tokenizer
        model_name = getattr(tokenizer, 'name_or_path', 'meta-llama/Llama-3.2-3B-Instruct')
        
        # Common cache paths
        cache_paths = [
            f"/ceph_data/ycyoon/.cache/huggingface/transformers/models--{model_name.replace('/', '--')}/snapshots",
            f"~/.cache/huggingface/transformers/models--{model_name.replace('/', '--')}/snapshots"
        ]
        
        tokenizer_files = [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json"
        ]
        
        for cache_base in cache_paths:
            cache_base = os.path.expanduser(cache_base)
            if os.path.exists(cache_base):
                # Find the latest snapshot
                snapshots = [d for d in os.listdir(cache_base) if os.path.isdir(os.path.join(cache_base, d))]
                if snapshots:
                    latest_snapshot = sorted(snapshots)[-1]
                    snapshot_dir = os.path.join(cache_base, latest_snapshot)
                    
                    for file_name in tokenizer_files:
                        src_file = os.path.join(snapshot_dir, file_name)
                        dst_file = os.path.join(output_dir, file_name)
                        
                        if os.path.exists(src_file):
                            shutil.copy2(src_file, dst_file)
                            print(f"✅ Copied {file_name} from cache")
                        elif os.path.islink(src_file):
                            # Handle symlinks in HuggingFace cache
                            link_target = os.readlink(src_file)
                            if not os.path.isabs(link_target):
                                link_target = os.path.join(snapshot_dir, link_target)
                            if os.path.exists(link_target):
                                shutil.copy2(link_target, dst_file)
                                print(f"✅ Copied {file_name} from cache (via symlink)")
                    return True
        
        print("⚠️ Could not find tokenizer files in cache")
        return False
        
    except Exception as e:
        print(f"⚠️ Error copying tokenizer files from cache: {e}")
        if logger:
            logger.warning(f"Error copying tokenizer files from cache: {e}")
        return False

def _verify_config_files(output_dir, logger=None):
    """
    Verify that all necessary HuggingFace config files were created successfully
    """
    required_files = [
        "config.json",
        "rgtnet_model_info.json"
    ]
    
    optional_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json"
    ]
    
    all_success = True
    
    # Check required files
    for file_name in required_files:
        file_path = os.path.join(output_dir, file_name)
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            print(f"✅ Verified: {file_name} ({os.path.getsize(file_path)} bytes)")
        else:
            print(f"❌ Missing or empty: {file_name}")
            all_success = False
    
    # Check optional files
    optional_success = 0
    for file_name in optional_files:
        file_path = os.path.join(output_dir, file_name)
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            print(f"✅ Verified: {file_name} ({os.path.getsize(file_path)} bytes)")
            optional_success += 1
        else:
            print(f"⚠️ Optional file missing: {file_name}")
    
    # Summary
    print(f"📊 Config file verification:")
    print(f"   • Required files: {len([f for f in required_files if os.path.exists(os.path.join(output_dir, f))])}/{len(required_files)} ✅")
    print(f"   • Optional files: {optional_success}/{len(optional_files)} ✅")
    
    if logger:
        logger.info(f"Config verification: {len(required_files)} required, {optional_success} optional files created")
    
    return all_success

def create_timestamped_save_path(base_path, pretrained_model_name):
    """Create a timestamped save path for checkpoints with pretrained model name"""
    import os
    from datetime import datetime, timezone, timedelta
    
    # Create model name for directory
    # Extract model name from full path (e.g., "meta-llama/Llama-3.2-3B-Instruct" -> "llama-3.2-3b")
    model_name = pretrained_model_name.split('/')[-1].lower()
    # Clean up model name for folder use
    model_name = model_name.replace('_', '-').replace(' ', '-')
    
    # Add timestamp to folder name
    # korean time zone
    timestamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M")
    
    # Construct final folder name with model info and timestamp
    folder_name = f"rgtnet_{model_name}_{timestamp}"
    timestamped_dir = os.path.join(base_path, folder_name)
        
    os.makedirs(timestamped_dir, exist_ok=True)
    
    print(f"📁 Created save directory: {timestamped_dir}")
    return timestamped_dir

def merge_deepspeed_checkpoint(checkpoint_dir, output_dir, logger=None):
    """Merge DeepSpeed checkpoint using zero_to_fp32.py"""
    import os
    import subprocess
    import sys
    
    try:
        # 🔧 FIX: latest 파일에서 실제 체크포인트 태그 읽기 (fallback 로직 포함)
        latest_file = os.path.join(checkpoint_dir, "latest")
        tag = None
        
        if os.path.exists(latest_file):
            with open(latest_file, 'r') as f:
                tag = f.read().strip()
            print(f"✅ Found latest file: {tag}")
        else:
            print(f"⚠️  latest file not found in {checkpoint_dir}, searching for epoch folders...")
            # Fallback: 가장 최근 epoch 폴더 찾기
            epoch_dirs = []
            for item in os.listdir(checkpoint_dir):
                if item.startswith('epoch_') and os.path.isdir(os.path.join(checkpoint_dir, item)):
                    try:
                        epoch_num = int(item.split('_')[1])
                        epoch_dirs.append((epoch_num, item))
                    except (IndexError, ValueError):
                        continue
            
            if epoch_dirs:
                # 가장 높은 epoch 번호 찾기
                epoch_dirs.sort(reverse=True)
                tag = epoch_dirs[0][1]
                print(f"✅ Found most recent epoch folder: {tag}")
            else:
                print(f"❌ No epoch folders found in {checkpoint_dir}")
                return False
        
        # 실제 체크포인트가 있는 디렉토리
        actual_checkpoint_dir = os.path.join(checkpoint_dir, tag)
        
        # Check if zero_to_fp32.py exists in the checkpoint directory
        zero_script = os.path.join(checkpoint_dir, "zero_to_fp32.py")
        if not os.path.exists(zero_script):
            print(f"⚠️  zero_to_fp32.py not found in {checkpoint_dir}")
            return False
        
        # 🔧 FIX: 실제 체크포인트 파일들이 있는지 확인
        required_files = ["mp_rank_00_model_states.pt"]
        for file in required_files:
            if not os.path.exists(os.path.join(actual_checkpoint_dir, file)):
                print(f"⚠️  Required checkpoint file not found: {file}")
                return False
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # 🔧 FIX: latest 파일이 있는 상위 디렉토리를 사용 (zero_to_fp32.py가 latest 파일을 읽어서 epoch 폴더를 찾음)
        # Note: safetensors has issues with tied weights (lm_head.weight & embed_tokens.weight), so we use PyTorch serialization
        cmd = [
            sys.executable, zero_script, 
            checkpoint_dir, output_dir  # latest 파일이 있는 디렉토리 지정
            # Removed --safe_serialization due to tied weights issue in Llama models
        ]
        
        print(f"🔄 Merging DeepSpeed checkpoint...")
        print(f"Base checkpoint directory: {checkpoint_dir}")
        print(f"Target epoch directory: {actual_checkpoint_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Command: {' '.join(cmd)}")
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Successfully merged checkpoint to {output_dir}")
            if logger:
                logger.info(f"DeepSpeed checkpoint merged to {output_dir}")
            return True
        else:
            print(f"❌ Failed to merge checkpoint:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            if logger:
                logger.error(f"Failed to merge DeepSpeed checkpoint: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error during checkpoint merging: {e}")
        if logger:
            logger.error(f"Error during checkpoint merging: {e}")
        return False

def train_model(model, train_loader, val_loader, device, args, local_rank=0, logger=None, tokenizer=None):
    """Train the model"""
    if logger:
        logger.info("Starting training...")
    
    # Training setup
    best_val_loss = float('inf')
    training_stats = {
        'train_losses': [],
        'val_losses': [],
        'learning_rates': [],  # 🔧 FIX: 추가 필요
        'epochs': []
    }
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        if is_main_process():
            print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        model.train()
        total_loss = 0
        num_batches = 0

        
        # DeepSpeed wraps model in engine.module
        if args.use_lora:
            actual_model = model.module if hasattr(model, 'module') else model
            adapter_save_dir = os.path.join(args.timestamped_dir, f"lora_adapters_epoch_{epoch}")
        
        # Create progress bar for main process only
        if is_main_process():
            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        else:
            progress_bar = train_loader
        
        for batch_idx, batch in enumerate(progress_bar):
            # 🔧 FIX: Early termination 체크를 올바른 위치로 이동
            if hasattr(args, 'max_iters') and args.max_iters is not None and batch_idx >= args.max_iters:
                if is_main_process():
                    print(f"Reached max_iters ({args.max_iters}), stopping training")
                
                # Save checkpoint on early termination
                if hasattr(model, 'save_checkpoint'):  # DeepSpeed
                    tag = f"epoch_{epoch}"
                    model.save_checkpoint(args.timestamped_dir, tag=tag)
                    # 🔧 FIX: barrier 제거 - hang 방지
                    if is_main_process():
                        print("✅ Early termination checkpoint saved")
                        print(f"📁 DeepSpeed checkpoint location: {args.timestamped_dir}")
                        
                        # 🔧 FIX: Create latest file for early termination checkpoint
                        latest_file_path = os.path.join(args.timestamped_dir, "latest")
                        try:
                            with open(latest_file_path, 'w') as f:
                                f.write(tag)
                            print(f"✅ Created latest file: {latest_file_path} -> {tag}")
                            if logger:
                                logger.info(f"Created latest file pointing to {tag}")
                        except Exception as e:
                            print(f"⚠️ Failed to create latest file: {e}")
                            if logger:
                                logger.warning(f"Failed to create latest file: {e}")
                        
                        print("💡 You can convert this checkpoint to a merged model later using:")
                        print(f"   python -m deepspeed.checkpoint.zero_to_fp32 {args.timestamped_dir}/epoch_{epoch} merged_model.bin")
                        
                        if logger:
                            logger.info(f"Early termination checkpoint saved to: {args.timestamped_dir}/epoch_{epoch}")

                if args.use_lora:
                    if hasattr(actual_model, 'base_model') and hasattr(actual_model.base_model, 'peft_config'):
                        print(f"🎯 Found PEFT model, saving LoRA adapters...")
                        actual_model.base_model.save_pretrained(adapter_save_dir)
                        print(f"✅ LoRA adapters saved to: {adapter_save_dir}")
                        print(f"📁 LoRA location: {adapter_save_dir}")
                        
                        if logger:
                            logger.info(f"LoRA adapters saved to: {adapter_save_dir}")
                    else:
                        print(f"⚠️ No PEFT model found! Cannot save LoRA adapters.")
                        print(f"   Model has module: {hasattr(model, 'module')}")
                        print(f"   Actual model has base_model: {hasattr(actual_model, 'base_model')}")
                        if hasattr(actual_model, 'base_model'):
                            print(f"   Base model has peft_config: {hasattr(actual_model.base_model, 'peft_config')}")
                        
                        if logger:
                            logger.warning(f"No PEFT model found - cannot save LoRA adapters")
                
                # Early return with results
                return {
                    'best_val_loss': best_val_loss,
                    'final_train_loss': total_loss / num_batches if num_batches > 0 else 0,
                    'final_val_loss': float('inf'),
                    'training_stats': training_stats,
                    'total_epochs': epoch + 1,
                    'early_termination': True
                }

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            role_mask = batch['role_mask'].to(device)
            
            # Forward pass with RGTNet role-aware attention
            outputs = model(input_ids=input_ids, role_mask=role_mask, labels=labels)
            loss = outputs['loss']
            if loss.dim() > 0:
                loss = loss.mean()
            
            # Backward pass and optimization
            if hasattr(model, 'backward'):  # DeepSpeed
                model.backward(loss)
                model.step()

            total_loss += loss.item() * args.gradient_accumulation_steps
            num_batches += 1
            
            if is_main_process():
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/num_batches:.4f}',
                    'lr': f'{model.get_lr()[0]:.2e}' if hasattr(model, 'get_lr') else 'N/A'
                })
        
        # After each epoch, evaluate
        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # Only run validation if not in train_only mode
        if not getattr(args, 'train_only', False):
            val_loss = evaluate_model(model, val_loader, device, args)
        else:
            val_loss = float('inf')
            if is_main_process():
                print("Skipping validation (train_only mode)")
        
        if is_main_process():
            training_stats['train_losses'].append(avg_train_loss)
            training_stats['val_losses'].append(val_loss)
            training_stats['learning_rates'].append(model.get_lr()[0] if hasattr(model, 'get_lr') else 0.0)
            training_stats['epochs'].append(epoch + 1)
        
        # 🔧 FIX: 매 epoch마다 저장하도록 수정
        should_save = True  # 매 epoch마다 저장
        
        if is_main_process():
            if not getattr(args, 'train_only', False) and val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"New best validation loss: {val_loss:.4f}")
            elif getattr(args, 'train_only', False):
                print(f"Training only mode - saving checkpoint")
            else:
                print(f"Regular checkpoint save at epoch {epoch+1}")
        
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
        save_start_time = time.time()
        
        # 🔧 ADD: 디버깅 - 저장 디렉토리 확인
        if is_main_process():
            print(f"🔍 Debug: Saving to directory: {args.timestamped_dir}")
            print(f"🔍 Debug: Directory exists: {os.path.exists(args.timestamped_dir)}")
            if not os.path.exists(args.timestamped_dir):
                print(f"🔧 Creating directory: {args.timestamped_dir}")
                os.makedirs(args.timestamped_dir, exist_ok=True)
        
        # Conditional checkpoint saving based on args
        if hasattr(model, 'save_checkpoint'):  
            # Save DeepSpeed checkpoint only if explicitly requested
            tag = f"epoch_{epoch}"
            model.save_checkpoint(args.timestamped_dir, tag=tag)
            
            # Create latest file for DeepSpeed checkpoint
            if is_main_process():
                print(f"✅ DeepSpeed checkpoint saved to: {args.timestamped_dir}/{tag}")
                
                # 🔧 FIX: Create latest file that points to the current checkpoint
                latest_file_path = os.path.join(args.timestamped_dir, "latest")
                try:
                    with open(latest_file_path, 'w') as f:
                        f.write(tag)
                    print(f"✅ Created latest file: {latest_file_path} -> {tag}")
                    if logger:
                        logger.info(f"Created latest file pointing to {tag}")
                except Exception as e:
                    print(f"⚠️ Failed to create latest file: {e}")
                    if logger:
                        logger.warning(f"Failed to create latest file: {e}")

                # 🔧 FIX: 스크립트를 체크포인트가 저장된 폴더에 바로 복사합니다.
                import shutil
                try:
                    zero_script_src = os.path.join(os.path.dirname(deepspeed.__file__), 
                                                    "checkpoint", "zero_to_fp32.py")
                    if os.path.exists(zero_script_src):
                        shutil.copy2(zero_script_src, args.timestamped_dir)
                        if logger:
                            logger.info(f"Copied zero_to_fp32.py to {args.timestamped_dir}")
                except Exception as e:
                    if logger:
                        logger.warning(f"Could not copy zero_to_fp32.py: {e}")

        # 🔧 FIX: barrier와 sleep 제거 - hang 방지
        # DeepSpeed가 자체적으로 동기화를 처리하므로 명시적 동기화 불필요
        
        save_duration = time.time() - save_start_time
        
        # Conditional model merging (only if DeepSpeed checkpoint was saved)
        if not args.use_lora and is_main_process():
            checkpoint_type = "DeepSpeed"
            print(f"✅ {checkpoint_type} checkpoint saved at epoch {epoch+1} (took {save_duration:.2f}s)")
            print(f"📁 Checkpoint location: {args.timestamped_dir}")
            if logger:
                logger.info(f"Checkpoint saved at epoch {epoch+1} in {args.timestamped_dir}")
        
            # 모델 병합 - timestamped_dir 밑에 저장
            merge_output_dir = os.path.join(args.timestamped_dir, f"merged_epoch_{epoch}")
            
            success = merge_deepspeed_checkpoint(args.timestamped_dir, merge_output_dir, logger)
            if success:
                print(f"✅ Merged model saved to: {merge_output_dir}")
                args.latest_merged_checkpoint = merge_output_dir
                
                # Create HuggingFace config files
                if tokenizer is not None:
                    config_success = create_huggingface_config_files(model, tokenizer, merge_output_dir, logger)
                    if config_success:
                        print(f"✅ HuggingFace config files created in: {merge_output_dir}")
                    else:
                        print(f"⚠️ Failed to create some HuggingFace config files in: {merge_output_dir}")
                else:
                    print("⚠️ Tokenizer not provided, skipping HuggingFace config file creation")
        
            # 🔧 FIX: 병합은 rank 0에서만 실행되므로 동기화 불필요
            # DeepSpeed save_checkpoint가 이미 모든 프로세스 동기화를 처리함
        
        # 🔧 MODIFIED: Save LoRA adapters (default behavior when LoRA is used) 
        if args.use_lora and is_main_process():
            if hasattr(actual_model, 'base_model'):
                print(f"🔍 Debug: hasattr(actual_model.base_model, 'peft_config'): {hasattr(actual_model.base_model, 'peft_config')}")
                print(f"🔍 Debug: Base model type: {type(actual_model.base_model)}")
            
            try:
                # Save LoRA adapters
                
                if hasattr(actual_model, 'base_model') and hasattr(actual_model.base_model, 'peft_config'):
                    print(f"🎯 Found PEFT model, saving LoRA adapters...")
                    actual_model.base_model.save_pretrained(adapter_save_dir)
                    print(f"✅ LoRA adapters saved to: {adapter_save_dir}")
                    print(f"📁 LoRA location: {adapter_save_dir}")
                    
                    if logger:
                        logger.info(f"LoRA adapters saved to: {adapter_save_dir}")
                else:
                    print(f"⚠️ No PEFT model found! Cannot save LoRA adapters.")
                    print(f"   Model has module: {hasattr(model, 'module')}")
                    print(f"   Actual model has base_model: {hasattr(actual_model, 'base_model')}")
                    if hasattr(actual_model, 'base_model'):
                        print(f"   Base model has peft_config: {hasattr(actual_model.base_model, 'peft_config')}")
                    
                    if logger:
                        logger.warning(f"No PEFT model found - cannot save LoRA adapters")
                        
            except Exception as e:
                print(f"⚠️ Failed to save LoRA adapters: {e}")
                import traceback
                traceback.print_exc()
                if logger:
                    logger.warning(f"Failed to save LoRA adapters: {e}")
            
        # 🔧 FIX: barrier 제거 - hang 방지

        # 🔧 FIX: epoch 로그를 올바른 위치로 이동
        epoch_time = time.time() - epoch_start_time
        
        if is_main_process():
            from utils import log_training_progress
            
            print(f"Epoch {epoch+1}/{args.epochs} completed in {epoch_time:.2f}s")
            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {model.get_lr()[0]:.2e}" if hasattr(model, 'get_lr') else "Learning Rate: N/A")
            print("-" * 50)
            
            if logger:
                log_training_progress(logger, epoch+1, args.epochs, avg_train_loss, val_loss, 
                                    model.get_lr()[0] if hasattr(model, 'get_lr') else 0.0, epoch_time, best_val_loss)
    
    # 🔧 FIX: 정상 완료 시 올바른 결과 반환
    results = {
        'best_val_loss': best_val_loss,
        'final_train_loss': training_stats['train_losses'][-1] if training_stats['train_losses'] else 0,
        'final_val_loss': training_stats['val_losses'][-1] if training_stats['val_losses'] else 0,
        'training_stats': training_stats,
        'total_epochs': args.epochs,
        'early_termination': False
    }
    
    if is_main_process():
        print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
        if logger:
            logger.info(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    
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
            
            # RGTNet forward pass with role-aware attention
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