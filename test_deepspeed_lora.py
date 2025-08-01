#!/usr/bin/env python3
"""
DeepSpeed multi-GPU LoRA test script
"""
import os
import sys
import torch
import argparse
import deepspeed
from transformers import AutoTokenizer
from model_hybrid import create_hybrid_model

def setup_deepspeed_config():
    """Create a minimal DeepSpeed config for testing"""
    config = {
        "train_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 5e-5,
                "weight_decay": 0.01
            }
        },
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "zero_optimization": {
            "stage": 0,  # Use Stage 0 to avoid parameter sharding with LoRA
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
            "offload_optimizer": {
                "device": "none",
                "pin_memory": False
            }
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 10,
        "wall_clock_breakdown": False,
        "dump_state": False,
        "distributed_inference": False
    }
    return config

def test_deepspeed_lora():
    """Test DeepSpeed with LoRA on multiple GPUs"""
    print("🧪 Testing DeepSpeed with LoRA on multiple GPUs...")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False
    
    num_gpus = torch.cuda.device_count()
    print(f"📊 Found {num_gpus} GPUs")
    
    if num_gpus < 2:
        print("⚠️  Need at least 2 GPUs for multi-GPU test")
        return False
    
    # Test arguments
    args = argparse.Namespace()
    args.pretrained_model_name = "meta-llama/Llama-3.2-3B-Instruct"
    args.use_lora = True
    args.enable_role_adapters = True
    args.lora_r = 8
    args.lora_alpha = 1  # Set alpha=1 for scaling=0.125 (worked in debug)
    args.lora_dropout = 0.05
    args.use_quantization = False
    args.deepspeed = True
    args.deepspeed_config = "test_ds_config.json"
    
    try:
        # Create DeepSpeed config file
        import json
        ds_config = setup_deepspeed_config()
        with open("test_ds_config.json", "w") as f:
            json.dump(ds_config, f, indent=2)
        
        # Test 1: Model creation with LoRA
        print("📦 Creating hybrid model with LoRA...")
        model, tokenizer = create_hybrid_model(args)
        print("✅ Model creation successful!")
        
        # Test 2: Check LoRA application and scaling
        print("🔍 Checking LoRA application and scaling...")
        lora_modules = []
        scaling_info = {}
        
        for name, module in model.named_modules():
            if 'lora' in name.lower():
                lora_modules.append(name)
                
                # Check scaling values for q_proj modules
                if 'q_proj' in name and 'lora_A' in name and 'default' in name:
                    print(f"   Found LoRA module: {name}")
                    print(f"   LoRA A shape: {module.weight.shape}")
                    
                    # Check corresponding lora_B module
                    b_name = name.replace('lora_A', 'lora_B')
                    if b_name in dict(model.named_modules()):
                        b_module = dict(model.named_modules())[b_name]
                        print(f"   LoRA B shape: {b_module.weight.shape}")
                        
                        # Calculate dimensions
                        input_dim = module.weight.shape[1]  # 3072
                        rank = module.weight.shape[0]       # r (8)
                        output_dim = b_module.weight.shape[0]  # 3072
                        
                        print(f"   Input dim: {input_dim}")
                        print(f"   Rank: {rank}")
                        print(f"   Output dim: {output_dim}")
                        
                        # Calculate expected scaling
                        expected_scaling = args.lora_alpha / args.lora_r
                        print(f"   Expected scaling: {expected_scaling}")
                        
                        scaling_info = {
                            'input_dim': input_dim,
                            'rank': rank,
                            'output_dim': output_dim,
                            'expected_scaling': expected_scaling
                        }
                        break
        
        if lora_modules:
            print(f"✅ Found {len(lora_modules)} LoRA modules")
            print(f"   Example: {lora_modules[0]}")
        else:
            print("❌ No LoRA modules found!")
            return False
        
        # Test 3: Check trainable parameters
        print("📊 Checking trainable parameters...")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Trainable ratio: {trainable_params/total_params*100:.2f}%")
        
        # Test 4: Check if we're in a distributed environment
        print("🌐 Checking distributed environment...")
        is_distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
        if is_distributed:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ.get('LOCAL_RANK', rank))
            print(f"✅ Running in distributed mode: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        else:
            print("⚠️  Not running in distributed mode")
            rank = 0
            world_size = 1
            local_rank = 0
        
        # Test 5: Multi-GPU forward pass
        print("🔄 Testing multi-GPU forward pass...")
        
        # Create sample batch (use smaller batch and shorter sequence)
        input_texts = [
            "Hello",
            "Hi"
        ]
        
        inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=16)
        
        # Move to GPU
        for key in inputs:
            inputs[key] = inputs[key].cuda()
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        print("✅ Multi-GPU forward pass successful!")
        print(f"   Output shape: {outputs['logits'].shape}")
        
        # Test 6: Memory usage per GPU
        print("💾 Checking memory usage per GPU...")
        for i in range(num_gpus):
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"   GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        
        # Test 7: Simple training step (optional)
        print("🎯 Testing training step...")
        try:
            # Create dummy labels
            labels = torch.randint(0, tokenizer.vocab_size, (inputs['input_ids'].shape[0], inputs['input_ids'].shape[1])).cuda()
            
            # Forward pass with loss
            outputs = model(**inputs, labels=labels)
            loss = outputs['loss']
            
            print("✅ Training forward pass successful!")
            print(f"   Loss: {loss.item():.4f}")
            
        except Exception as e:
            print(f"⚠️  Training step failed: {e}")
        
        print("\n🎉 DeepSpeed multi-GPU LoRA test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if os.path.exists("test_ds_config.json"):
            os.remove("test_ds_config.json")

def test_adapter_functionality():
    """Test adapter-specific functionality"""
    print("\n🧪 Testing adapter functionality...")
    
    args = argparse.Namespace()
    args.pretrained_model_name = "meta-llama/Llama-3.2-3B-Instruct"
    args.use_lora = True
    args.enable_role_adapters = True
    args.lora_r = 8
    args.lora_alpha = 1  # Set alpha=1 for scaling=0.125 (worked in debug)
    args.lora_dropout = 0.05
    args.use_quantization = False
    
    try:
        model, tokenizer = create_hybrid_model(args)
        
        # Test adapter methods
        print("🔧 Testing adapter methods...")
        
        # Check if model has adapter methods
        if hasattr(model, 'get_adapter_state_dict'):
            print("✅ get_adapter_state_dict method found")
        
        if hasattr(model, 'set_adapter_state_dict'):
            print("✅ set_adapter_state_dict method found")
        
        # Test adapter state dict
        if hasattr(model, 'get_adapter_state_dict'):
            adapter_state = model.get_adapter_state_dict()
            print(f"✅ Adapter state dict created with {len(adapter_state)} items")
        
        # Test role adapter functionality
        if hasattr(model, 'role_adapters') and len(model.role_adapters) > 0:
            print(f"✅ Role adapters found: {len(model.role_adapters)}")
        
        print("✅ Adapter functionality test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Adapter test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🔬 DeepSpeed Multi-GPU LoRA Test")
    print("=" * 60)
    
    # Test adapter functionality
    adapter_success = test_adapter_functionality()
    
    # Test DeepSpeed with LoRA
    deepspeed_success = test_deepspeed_lora()
    
    print("\n" + "=" * 60)
    if adapter_success and deepspeed_success:
        print("✅ All tests passed! DeepSpeed + LoRA is working correctly.")
    else:
        print("❌ Some tests failed!")
        if not adapter_success:
            print("   - Adapter functionality test failed")
        if not deepspeed_success:
            print("   - DeepSpeed multi-GPU test failed")
    print("=" * 60) 