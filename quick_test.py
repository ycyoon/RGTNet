import torch
import os
import sys
from transformers import AutoTokenizer

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Now import project modules
from config import setup_args
from model import create_model
from utils import set_seed

def quick_test():
    """
    A lightweight script to quickly test model forward and backward passes
    without loading any datasets.
    """
    # --- Setup ---
    # We need to parse args to get model configuration
    args = setup_args()
    set_seed(42)

    # --- DDP Initialization ---
    is_ddp = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if is_ddp:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        device = f'cuda:{local_rank}'
        print(f"DDP enabled on RANK {rank} / {world_size} on device {device}")
    else:
        rank = 0
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"DDP not enabled. Running on {device}")

    # --- Model Creation ---
    if rank == 0:
        print("Creating tokenizer and model...")
    
    # Always use a lightweight tokenizer for testing
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size
    pad_idx = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    if args.use_dummy_model:
        if rank == 0:
            print("Using dummy model configuration for testing.")
        # Override args with lightweight settings for a dummy model
        args.d_model = 64
        args.nhead = 4
        args.num_layers = 2
        args.max_seq_len = 256 # Smaller sequence length for faster testing
        args.pretrained_model_name = None # Ensure we don't try to load weights
    
    model = create_model(args, tokenizer)[0].to(device)

    if is_ddp:
        # Important: set find_unused_parameters=False if not all params are used in forward pass
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if rank == 0:
        print("Model created successfully.")
        # print(model) # Optional: print model architecture

    # --- Dummy Data Creation ---
    batch_size = 4
    seq_len = args.max_seq_len
    
    if rank == 0:
        print(f"Creating dummy data: B={batch_size}, L={seq_len}, V={vocab_size}")

    # Create random input_ids and role_mask
    dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    dummy_role_mask = torch.randint(0, 2, (batch_size, seq_len), device=device)
    
    if rank == 0:
        print("Dummy data created.")

    # --- Test Forward and Backward Pass ---
    try:
        if rank == 0:
            print("Attempting forward pass...")
        
        # model.forward() returns a dictionary. Pass labels to calculate loss.
        outputs = model(input_ids=dummy_input_ids, role_mask=dummy_role_mask, labels=dummy_input_ids)
        loss = outputs['loss']
        
        if rank == 0:
            print(f"Forward pass successful. Loss: {loss.item()}")
        
        if rank == 0:
            print("Attempting backward pass...")
        
        loss.backward()
        
        if rank == 0:
            print("Backward pass successful.")

    except Exception as e:
        print(f"[ERROR on RANK {rank}] Test failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # --- DDP Cleanup ---
        if is_ddp:
            dist.destroy_process_group()
        
        if rank == 0:
            print("Test finished.")

if __name__ == '__main__':
    quick_test() 