import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import time
import os
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)

# -------------------- Core Model Definitions --------------------

def _create_single_orthogonal_matrix(dim, device=None):
    """Generates a single random orthogonal matrix of shape (dim, dim)."""
    random_matrix = torch.randn(dim, dim, device=device)
    q, r = torch.linalg.qr(random_matrix)
    return q

def _create_stacked_orthogonal_matrices(num_matrices, dim, device=None):
    """Generates a stack of random orthogonal matrices of shape (num_matrices, dim, dim)."""
    return torch.stack([_create_single_orthogonal_matrix(dim, device) for _ in range(num_matrices)])

class LegacyRoleSensitiveEmbedding(nn.Module):
    """
    Legacy role-sensitive token embedding using R matrix.
    Compatible with older checkpoints that use embedding.R instead of role_transformers.
    """
    def __init__(self, vocab_size, d_model, pad_idx, device=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        # Legacy R matrix for role transformation
        self.R = nn.Parameter(torch.randn(d_model, d_model))

    def forward(self, input_ids, role_mask):
        """
        input_ids: (B, L)
        role_mask: (B, L) with 0 for user, 1 for agent
        """
        x = self.embedding(input_ids)
        
        # Apply R matrix transformation based on role
        # For legacy compatibility, apply R transformation to agent tokens
        agent_mask = (role_mask == 1).unsqueeze(-1)  # (B, L, 1)
        transformed_x = x.clone()
        
        # Apply R transformation to agent tokens
        agent_tokens = x[role_mask == 1]  # Get all agent tokens
        if agent_tokens.numel() > 0:
            transformed_agent = torch.matmul(agent_tokens, self.R.T)
            transformed_x[role_mask == 1] = transformed_agent
            
        return transformed_x


class RoleSensitiveEmbedding(nn.Module):
    """
    Role-sensitive token embedding.
    A role-specific linear transformation is applied to the standard token embedding.
    """
    def __init__(self, vocab_size, d_model, pad_idx, device=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        # 0: user, 1: agent
        self.role_transformers = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(2) 
        ])

    def forward(self, input_ids, role_mask):
        """
        input_ids: (B, L)
        role_mask: (B, L) with 0 for user, 1 for agent
        """
        x = self.embedding(input_ids)
        
        # Apply role-specific transformation
        # Create a tensor to store the output
        transformed_x = torch.zeros_like(x)
        
        # Apply transformation for each role
        for role_idx, transformer in enumerate(self.role_transformers):
            # Create a mask for the current role
            role_specific_mask = (role_mask == role_idx)
            # Apply the transformation where the mask is True, ensuring dtype matches
            transformed_x[role_specific_mask] = transformer(x[role_specific_mask]).to(transformed_x.dtype)
            
        return transformed_x

class LegacyRoleGatedSelfAttention(nn.Module):
    """
    Legacy role-gated self-attention mechanism.
    Compatible with older checkpoints that use scalar delta and no U matrix.
    """
    def __init__(self, d_model, nhead, dropout=0.1, delta=1.0):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        # Legacy scalar delta parameter
        self.delta = nn.Parameter(torch.tensor(delta))
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, role_mask, causal_mask=None):
        bsz, seq_len, _ = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Legacy role-based gating without U matrix rotation
        if self.delta > 0:
            # Simple role-based scaling/gating
            role_gate = torch.sigmoid(self.delta * (role_mask.float() - 0.5))
            role_gate = role_gate.unsqueeze(-1)  # (B, L, 1)
            k = k * role_gate  # Apply role gating to keys

        # Use scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            q.view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2),
            k.view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2),
            v.view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2),
            attn_mask=causal_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False # Explicitly set to False when using attn_mask
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(attn_output)


class RoleGatedSelfAttention(nn.Module):
    """
    Role-gated self-attention mechanism.
    It rotates the key vector based on its role.
    """
    def __init__(self, d_model, nhead, dropout=0.1, delta=1.0):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.delta = nn.Parameter(torch.tensor([delta]))
        
        # Orthogonal matrix for rotation
        self.register_buffer("U", _create_stacked_orthogonal_matrices(self.nhead, self.head_dim))
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, role_mask, causal_mask=None):
        bsz, seq_len, _ = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Role-based rotation using the pre-computed buffer
        if self.delta > 0:
            # Expand the buffer to the batch dimension and ensure matching dtype
            U_batch = self.U.unsqueeze(0).expand(bsz, -1, -1, -1).to(dtype=k.dtype, device=k.device)
            
            # Reshape K for rotation
            k_reshaped = k.view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2).contiguous()
            
            # Create a temporary tensor to hold the rotated values
            rotated_k_part = torch.matmul(k_reshaped, U_batch)

            # Explicitly expand role_gate to match k_reshaped's dimensions for broadcasting
            role_gate = torch.sigmoid(self.delta[0] * (role_mask.float() - 0.5))
            role_gate = role_gate.unsqueeze(1).unsqueeze(-1).expand(-1, self.nhead, -1, self.head_dim)

            # Perform the final computation in-place on k_reshaped
            k_reshaped.add_((rotated_k_part - k_reshaped) * role_gate)

            k = k_reshaped.transpose(1, 2).reshape(bsz, seq_len, self.d_model)

        # Use scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            q.view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2),
            k.view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2),
            v.view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2),
            attn_mask=causal_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False # Explicitly set to False when using attn_mask
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(attn_output)

class RoleAwareTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, bias_delta=1.0):
        super().__init__()
        self.self_attn = RoleGatedSelfAttention(d_model, nhead, dropout, bias_delta)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, role_mask, src_mask=None, src_key_padding_mask=None):
        # Self-attention
        attn_out = self.self_attn(src, role_mask, causal_mask=src_mask)
        
        # First residual connection and normalization
        src = src + self.dropout1(attn_out)
        src = self.norm1(src)

        # Feed-forward network
        ff_out = self.linear2(self.dropout(F.relu(self.linear1(src))))
        
        # Second residual connection and normalization
        src = src + self.dropout2(ff_out)
        src = self.norm2(src)
        
        return src

class CheckpointedRoleAwareTransformerLayer(RoleAwareTransformerLayer):
    def forward(self, src, role_mask, src_mask=None, src_key_padding_mask=None):
        return checkpoint(super().forward, src, role_mask, src_mask, src_key_padding_mask, use_reentrant=False)

class RoleAwareTransformerDecoder(nn.Module):
    """
    Role-aware Transformer Decoder with LLM capabilities (generation, causal mask, LMHead).
    Initializes with pretrained embeddings/LMHead from a specified model.
    """
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1, bias_delta=1.0, pad_idx=0, max_seq_len=8192, device=None, pretrained_model_name=None, use_gradient_checkpointing=False):
        super().__init__()
        self.embedding = RoleSensitiveEmbedding(vocab_size, d_model, pad_idx, device)
        # Increase positional encoding size to support longer sequences
        self.pos_encoder = nn.Embedding(512, d_model)  # Support up to 512 positions
        
        layer_class = CheckpointedRoleAwareTransformerLayer if use_gradient_checkpointing else RoleAwareTransformerLayer
        self.layers = nn.ModuleList([
            layer_class(d_model, nhead, dim_feedforward, dropout, bias_delta)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Pretrained Llama-3.2-90B embedding/LMHead initialization
        if pretrained_model_name is not None:
            try:
                print(f"Loading pretrained weights from {pretrained_model_name}...")
                llama = AutoModelForCausalLM.from_pretrained(pretrained_model_name, trust_remote_code=True)
                with torch.no_grad():
                    # Adapt embedding weights
                    llama_embed = llama.get_input_embeddings().weight
                    my_embed = self.embedding.embedding.weight
                    n = min(my_embed.size(0), llama_embed.size(0))
                    my_embed[:n].copy_(llama_embed[:n])
                    
                    # Adapt LM head weights, trying common attribute names
                    lm_head_candidates = [
                        ('lm_head',),
                        ('cls', 'predictions', 'decoder'),
                        ('cls', 'decoder')
                    ]
                    llama_lm_head_weight = None
                    for names in lm_head_candidates:
                        parent = llama
                        found = True
                        for name in names:
                            if hasattr(parent, name):
                                parent = getattr(parent, name)
                            else:
                                found = False
                                break
                        if found and hasattr(parent, 'weight'):
                            llama_lm_head_weight = parent.weight
                            break
                    
                    if llama_lm_head_weight is not None:
                        my_lm_head = self.lm_head.weight
                        n_lm = min(my_lm_head.size(0), llama_lm_head_weight.size(0))
                        my_lm_head[:n_lm].copy_(llama_lm_head_weight[:n_lm])
                        print("Successfully copied LM head weights.")
                    else:
                        print("[WARN] Could not find a suitable LM head in the pretrained model. LM head is randomly initialized.")

                print("Finished loading and adapting pretrained weights.")
            except Exception as e:
                print(f"[ERROR] Failed to load or adapt pretrained model '{pretrained_model_name}': {e}. Model will be randomly initialized.")

    def forward(self, input_ids, role_mask, labels=None):
        B, L = input_ids.size()
        x = self.embedding(input_ids, role_mask)
        
        # Critical fix: Ensure position indices are within bounds to prevent CUDA indexing errors
        max_pos = self.pos_encoder.num_embeddings - 1  # 127 for embedding size 128
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        
        # Clamp positions to prevent out-of-bounds access
        positions = torch.clamp(positions, 0, max_pos)
        
        # Add positional encoding with bounds checking
        try:
            pos_embeddings = self.pos_encoder(positions)
            x = x + pos_embeddings
        except Exception as e:
            print(f"Warning: Positional encoding failed with error: {e}. Using zero positional encoding.")
            # Fallback: use zero positional encoding
            x = x + torch.zeros_like(x)

        # Create a causal mask for the decoder
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        causal_mask = causal_mask.to(dtype=x.dtype)

        for layer in self.layers:
            x = layer(x, role_mask, src_mask=causal_mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            # Flatten the tokens
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        
        return {'loss': loss, 'logits': logits}

    @torch.no_grad()
    def generate(self, input_ids, max_length=100, max_new_tokens=None, num_return_sequences=1,
                 do_sample=False, top_k=50, top_p=0.95, pad_token_id=None, **kwargs):
        """
        Generate sequences using the model with robust error handling and bounds checking.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_length: Maximum total sequence length
            max_new_tokens: Maximum number of new tokens to generate
            num_return_sequences: Number of sequences to return (only 1 supported)
            do_sample: Whether to use sampling
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            pad_token_id: Padding token ID
            
        Returns:
            Generated sequences [batch_size, new_seq_len]
        """
        self.eval()
        
        # Get actual vocabulary size from embedding layer
        actual_vocab_size = self.embedding.embedding.num_embeddings
        
        # Validate and clamp input tokens to prevent CUDA indexing errors
        input_ids = torch.clamp(input_ids, 0, actual_vocab_size - 1)
        
        # Handle max_new_tokens parameter
        if max_new_tokens is not None:
            max_length = input_ids.size(1) + max_new_tokens
            
        sequences = [input_ids.clone()]  # only supports num_return_sequences==1 for now
        
        # Set default pad_token_id if not provided
        if pad_token_id is None:
            pad_token_id = actual_vocab_size - 1  # Use last token as pad
        
        # Ensure pad_token_id is within bounds
        pad_token_id = min(max(0, pad_token_id), actual_vocab_size - 1)
            
        # Limit maximum generation steps to prevent infinite loops
        max_generation_steps = min(max_length - input_ids.size(1), 20)  # Reduced to 20 for stability
        
        with torch.no_grad():  # Ensure no gradients during generation
            for step in range(max_generation_steps):
                seq = sequences[0]
                
                # Early termination if sequence is getting too long
                if seq.size(1) >= max_length:
                    break
                    
                # Validate ALL sequence tokens are within range before forward pass
                seq = torch.clamp(seq, 0, actual_vocab_size - 1)
                sequences[0] = seq  # Update the sequence with clamped values
                
                # Create a dummy role mask (all zeros for simplicity)
                role_mask = torch.zeros_like(seq, dtype=torch.long)
                
                try:
                    # Forward pass with comprehensive error handling
                    outputs = self.forward(seq, role_mask)
                    
                    if 'logits' not in outputs:
                        print("Warning: No logits in model output. Using fallback.")
                        break
                        
                    next_token_logits = outputs['logits'][:, -1, :]
                    
                    # Critical: Ensure logits size exactly matches vocabulary size
                    if next_token_logits.size(-1) != actual_vocab_size:
                        if next_token_logits.size(-1) > actual_vocab_size:
                            next_token_logits = next_token_logits[:, :actual_vocab_size]
                        else:
                            # Pad with very negative values (will have near-zero probability)
                            padding_size = actual_vocab_size - next_token_logits.size(-1)
                            padding = torch.full((next_token_logits.size(0), padding_size), 
                                               -1e9, device=next_token_logits.device, dtype=next_token_logits.dtype)
                            next_token_logits = torch.cat([next_token_logits, padding], dim=-1)
                    
                    # Apply temperature for more stable generation
                    temperature = 0.8
                    next_token_logits = next_token_logits / temperature
                    
                    # Apply sampling or greedy decoding
                    if do_sample and top_k > 0:
                        # Simplified top-k sampling for stability
                        top_k_actual = min(top_k, next_token_logits.size(-1))
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k_actual)
                        
                        # Sample from top-k
                        probs = torch.softmax(top_k_logits, dim=-1)
                        sampled_idx = torch.multinomial(probs, num_samples=1)
                        next_token = top_k_indices.gather(-1, sampled_idx)
                    else:
                        # Greedy decoding (most stable)
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    # Critical: Ensure generated token is within vocabulary bounds
                    next_token = torch.clamp(next_token, 0, actual_vocab_size - 1)
                    
                    # Verify token is valid before appending
                    if torch.any(next_token >= actual_vocab_size) or torch.any(next_token < 0):
                        print(f"Warning: Generated invalid token {next_token.item()}, using pad token")
                        next_token = torch.full_like(next_token, pad_token_id)
                    
                    # Append the new token
                    sequences[0] = torch.cat([seq, next_token], dim=1)
                    
                    # Stop if we hit pad token - FIX: Proper batch-wise checking
                    if pad_token_id is not None:
                        # Check if ANY token in the batch is a pad token
                        pad_mask = (next_token == pad_token_id)
                        if torch.any(pad_mask):
                            break
                            
                    if sequences[0].size(1) >= max_length:
                        break
                        
                except Exception as e:
                    # Comprehensive fallback: use a safe token
                    print(f"Warning: Generation step {step} failed with error: {e}. Using fallback.")
                    fallback_token = torch.full((input_ids.size(0), 1), pad_token_id, 
                                               device=input_ids.device, dtype=input_ids.dtype)
                    fallback_token = torch.clamp(fallback_token, 0, actual_vocab_size - 1)
                    
                    # Ensure we don't exceed max length
                    if seq.size(1) < max_length:
                        sequences[0] = torch.cat([seq, fallback_token], dim=1)
                    break
        
        # Final validation: ensure all tokens in the generated sequence are within bounds
        final_sequence = torch.clamp(sequences[0], 0, actual_vocab_size - 1)
        return final_sequence

# -------------------- Model Creation and Loading --------------------

def create_model(args, pad_idx):
    """Creates the RoleAwareTransformerDecoder model."""
    # Auto-detect model parameters from pretrained model if specified
    if args.pretrained_model_name:
        from transformers import AutoConfig
        try:
            pretrained_config = AutoConfig.from_pretrained(args.pretrained_model_name)
            d_model = pretrained_config.hidden_size
            nhead = pretrained_config.num_attention_heads
            num_layers = pretrained_config.num_hidden_layers
            print(f"Auto-detected model config from {args.pretrained_model_name}:")
            print(f"  d_model: {d_model}")
            print(f"  nhead: {nhead}")
            print(f"  num_layers: {num_layers}")
        except Exception as e:
            print(f"Warning: Could not auto-detect config from {args.pretrained_model_name}: {e}")
            print("Using args parameters")
            d_model = args.d_model
            nhead = args.nhead
            num_layers = args.num_layers
    else:
        # Use parameters from args when no pretrained model is specified
        d_model = args.d_model
        nhead = args.nhead
        num_layers = args.num_layers
        print("Using args model config:")
        print(f"  d_model: {d_model}")
        print(f"  nhead: {nhead}")
        print(f"  num_layers: {num_layers}")
    
    model = RoleAwareTransformerDecoder(
        vocab_size=args.vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=d_model * 4,
        dropout=args.dropout,
        bias_delta=args.bias_delta,
        pad_idx=pad_idx,
        max_seq_len=args.max_seq_len,
        pretrained_model_name=args.pretrained_model_name,
        use_gradient_checkpointing=getattr(args, 'gradient_checkpointing', False)
    )
    return model

# trainer.py의 save_checkpoint 함수 수정
def use_fsdp_consolidation(base_path, rank_count, output_path):
    """FSDP의 내장 병합 기능을 사용하여 shard 파일들을 병합"""
    print(f"\n--- Using FSDP consolidation to merge {rank_count} sharded checkpoints ---")
    
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig
    import torch.distributed as dist
    import os
    
    # 분산 환경 초기화 (단일 프로세스)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['RANK'] = '0'
    os.environ['LOCAL_RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    
    print("  Initializing single-process distributed environment...")
    dist.init_process_group(backend='gloo', rank=0, world_size=1)
    
    try:
        # 모든 shard 파일 확인
        shard_files = []
        for rank in range(rank_count):
            shard_path = f"{base_path}.rank{rank}.pt"
            if os.path.exists(shard_path):
                shard_files.append(shard_path)
                print(f"  Found shard: {shard_path}")
            else:
                print(f"  ⚠️  Missing shard: {shard_path}")
        
        if not shard_files:
            print("❌ No shard files found!")
            return False
        
        # 첫 번째 shard에서 기본 정보 가져오기
        print(f"  Loading first shard to check structure...")
        
        # 단일 프로세스 환경에서 로드 시도
        first_shard = torch.load(shard_files[0], map_location='cpu', weights_only=False)
        print(f"  Successfully loaded first shard")
        print(f"  First shard keys: {list(first_shard.keys())}")
        
        # 병합된 state_dict 생성
        merged_state_dict = OrderedDict()
        epoch = -1
        
        # 각 shard에서 파라미터 수집
        for rank, shard_path in enumerate(shard_files):
            print(f"  Loading shard {rank}: {shard_path}")
            
            try:
                # 단일 프로세스 환경에서 로드
                checkpoint = torch.load(shard_path, map_location='cpu', weights_only=False)
                
                if 'model' in checkpoint:
                    # FSDP sharded state dict
                    sharded_state = checkpoint['model']
                    print(f"    Shard {rank} has {len(sharded_state)} parameters")
                    
                    for key, value in sharded_state.items():
                        if key not in merged_state_dict:
                            merged_state_dict[key] = value
                        else:
                            # 이미 있는 경우 덮어쓰기 (마지막 shard가 우선)
                            merged_state_dict[key] = value
                    
                    if epoch == -1:
                        epoch = checkpoint.get('epoch', 0)
                else:
                    print(f"    ⚠️  No 'model' key in shard {rank}")
                    
            except Exception as e:
                print(f"    ❌ Error loading shard {rank}: {e}")
                continue
        
        print(f"  Total merged parameters: {len(merged_state_dict)}")
        
        # 병합된 체크포인트 저장
        merged_checkpoint = {
            'model_state_dict': merged_state_dict,
            'epoch': epoch
        }
        
        torch.save(merged_checkpoint, output_path)
        print(f"✅ Merged checkpoint saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error merging shards: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 분산 환경 정리
        if dist.is_initialized():
            dist.destroy_process_group()
            print("  Cleaned up distributed environment")

def merge_sharded_checkpoints(base_path, rank_count, output_path):
    """Merge sharded checkpoints into a single file"""
    from collections import OrderedDict
    import torch.distributed as dist
    
    print(f"\n--- Merging {rank_count} sharded checkpoints ---")
    
    # Load all shards
    full_state_dict = OrderedDict()
    epoch = -1
    
    for rank in range(rank_count):
        shard_path = f"{base_path}.rank{rank}.pt"
        if os.path.exists(shard_path):
            print(f"  Loading shard: {shard_path}")
            try:
                # Set environment variables to match the rank that saved this shard
                original_rank = os.environ.get('RANK', '0')
                original_local_rank = os.environ.get('LOCAL_RANK', '0')
                
                # Temporarily set rank to match the shard being loaded
                os.environ['RANK'] = str(rank)
                os.environ['LOCAL_RANK'] = str(rank)
                
                # Load the checkpoint
                checkpoint = torch.load(shard_path, map_location='cpu', weights_only=False)
                
                # Restore original rank
                os.environ['RANK'] = original_rank
                os.environ['LOCAL_RANK'] = original_local_rank
                
                if 'model' in checkpoint:
                    full_state_dict.update(checkpoint['model'])
                    if epoch == -1:
                        epoch = checkpoint.get('epoch', 0)
                        
            except Exception as e:
                print(f"  ❌ Error loading shard {rank}: {e}")
                # Try alternative loading method
                try:
                    print(f"  🔄 Trying alternative loading method for shard {rank}...")
                    checkpoint = torch.load(shard_path, map_location='cpu', weights_only=True)
                    if 'model' in checkpoint:
                        full_state_dict.update(checkpoint['model'])
                        if epoch == -1:
                            epoch = checkpoint.get('epoch', 0)
                except Exception as e2:
                    print(f"  ❌ Alternative loading also failed for shard {rank}: {e2}")
                    continue
        else:
            print(f"  ⚠️  Warning: Shard not found: {shard_path}")
    
    if full_state_dict:
        # Save merged checkpoint
        merged_checkpoint = {
            'model_state_dict': full_state_dict,
            'epoch': epoch
        }
        
        # Try to preserve optimizer state from rank 0
        rank0_shard_path = f"{base_path}.rank0.pt"
        if os.path.exists(rank0_shard_path):
            try:
                # Set environment variables for rank 0
                original_rank = os.environ.get('RANK', '0')
                original_local_rank = os.environ.get('LOCAL_RANK', '0')
                os.environ['RANK'] = '0'
                os.environ['LOCAL_RANK'] = '0'
                
                rank0_checkpoint = torch.load(rank0_shard_path, map_location='cpu', weights_only=False)
                
                # Restore original rank
                os.environ['RANK'] = original_rank
                os.environ['LOCAL_RANK'] = original_local_rank
                
                if 'optimizer' in rank0_checkpoint:
                    merged_checkpoint['optimizer_state_dict'] = rank0_checkpoint['optimizer']
                    print("  ✅ Preserved optimizer state from rank 0")
            except Exception as e:
                print(f"  ⚠️  Could not preserve optimizer state: {e}")
        
        torch.save(merged_checkpoint, output_path)
        print(f"✅ Merged checkpoint saved to: {output_path}")
        
        # Clean up shard files
        for rank in range(rank_count):
            shard_path = f"{base_path}.rank{rank}.pt"
            if os.path.exists(shard_path):
                os.remove(shard_path)
                print(f"  Removed shard: {shard_path}")
        
        return True
    else:
        print("❌ No shards found to merge!")
        return False

def verify_checkpoint(checkpoint_path):
    """Verify if checkpoint is valid"""
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        # weights_only=False 추가
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint or 'model' in checkpoint:
            print(f"✅ Valid checkpoint verified: {checkpoint_path}")
            return True
        else:
            print(f"❌ Invalid checkpoint format: {checkpoint_path}")
            return False
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False

def load_checkpoint(model, optimizer, checkpoint_path, device, is_main=True):
    """Load checkpoint from file"""
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # weights_only=False 추가
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        model_state = checkpoint['model']
    else:
        # Assume the checkpoint is the state dict itself
        model_state = checkpoint
    
    # Load model state
    if isinstance(model, FSDP):
        # For FSDP models, we need to use the appropriate loading strategy
        # Check if this is a merged checkpoint (non-sharded state dict)
        if 'model_state_dict' in checkpoint:
            # This is a merged checkpoint, use FULL_STATE_DICT
            from torch.distributed.fsdp import StateDictType, FullStateDictConfig
            cfg = FullStateDictConfig(offload_to_cpu=False)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
                model.load_state_dict(model_state)
        else:
            # This is a sharded checkpoint, use SHARDED_STATE_DICT
            from torch.distributed.fsdp import StateDictType, ShardedStateDictConfig
            cfg = ShardedStateDictConfig(offload_to_cpu=False)
            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, cfg):
                model.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)
    
    # Load optimizer state if available
    if optimizer and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if is_main:
                print("✅ Optimizer state loaded successfully")
        except Exception as e:
            if is_main:
                print(f"⚠️  Warning: Could not load optimizer state: {e}")
                print("   Initializing fresh optimizer state")
    
    # Get epoch
    epoch = checkpoint.get('epoch', 0)
    
    if is_main:
        print(f"✅ Checkpoint loaded from epoch {epoch}")
    
    return epoch

import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType


def save_checkpoint(model, optimizer, epoch, args, save_path=None, force_merge=False):
    """
    Save checkpoint with automatic merging after each epoch
    
    Args:
        model: Model to save
        optimizer: Optimizer to save (optional)
        epoch: Current epoch
        args: Training arguments
        save_path: Path to save checkpoint
        force_merge: Force merging even if not final epoch
    """
    import torch.distributed as dist
    import time
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        StateDictType,
        ShardedStateDictConfig,
    )

    if save_path is None:
        save_path = args.save_path

    is_fsdp = isinstance(model, FSDP)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank0 = rank == 0

    if rank0:
        print(f"\n--- Starting Checkpoint Save (Epoch {epoch + 1}) ---")
    
    total_start_time = time.time()

    if is_fsdp:
        # Use old API with deprecation warning suppression
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            
            # 1. state_dict 생성
            state_dict_start_time = time.time()
            
            cfg = ShardedStateDictConfig(offload_to_cpu=False)
            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, cfg):
                model_state_dict = model.state_dict()
            
            state_dict_duration = time.time() - state_dict_start_time
            if rank0:
                print(f"  [1/2] Model state_dict creation took: {state_dict_duration:.4f}s")
            
            # 2. 파일 저장 (optimizer 제외하여 속도 향상)
            save_start_time = time.time()
            shard_path = f"{save_path}.rank{rank}.pt"
            torch.save({
                'model': model_state_dict,
                'epoch': epoch
            }, shard_path)
            save_duration = time.time() - save_start_time
            
            if rank0:
                print(f"  [2/2] torch.save (rank {rank}) took: {save_duration:.4f}s")
            
    else:
        # Standard saving for non-FSDP models
        if rank0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                'epoch': epoch
            }, save_path)
            print(f"✅ Standard checkpoint saved to {save_path}")

    # Synchronize all ranks before merging
    if dist.is_initialized():
        dist.barrier()
    
    # 모든 rank에서 병합 수행
    if is_fsdp:
        if rank0:
            merge_start_time = time.time()
            success = merge_sharded_checkpoints(save_path, world_size, save_path)
            if success:
                merge_duration = time.time() - merge_start_time
                print(f"✅ Merging completed in {merge_duration:.4f}s")
                print(f"✅ Complete checkpoint saved to: {save_path}")
        # synchronize so that non-zero ranks wait until merge is done
        dist.barrier()

    total_duration = time.time() - total_start_time
    if rank0:
        print(f"✅ Total checkpoint save time: {total_duration:.4f}s\n")

 
def save_sharded_checkpoint(model: FSDP, optimizer, epoch: int, args, force_merge=False):
    """
    각 rank가 자기 shard만 .rank{r}.pt 파일로 저장하고 자동으로 병합합니다.
    """
    assert dist.is_initialized(), "Distributed not initialized"
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Use timestamped save path if available
    if hasattr(args, 'timestamped_save_path'):
        save_base = args.timestamped_save_path.rstrip(".pt")
    else:
        save_base = args.save_path.rstrip(".pt")
    
    shard_path = f"{save_base}.rank{rank}.pt"

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        sharded_state = model.state_dict()

    checkpoint = {
        'model':     sharded_state,
        'optimizer': optimizer.state_dict(),
        'epoch':     epoch,
    }
    torch.save(checkpoint, shard_path)
    print(f"(rank {rank}) ✅ Sharded checkpoint saved to: {shard_path}")

    # Wait for all ranks to finish saving
    dist.barrier()
    
    # Automatically merge shards on rank 0
    if rank == 0:
        print(f"🔄 Starting automatic merge of {world_size} shards...")
        merge_start_time = time.time()
        
        # Determine output path
        if hasattr(args, 'timestamped_save_path'):
            output_path = args.timestamped_save_path
        else:
            output_path = args.save_path
        
        try:
            # Try FSDP consolidation first (more reliable for FSDP models)
            success = use_fsdp_consolidation(save_base, world_size, output_path)
            
            if not success:
                # Fallback to manual merge
                print("🔄 FSDP consolidation failed, trying manual merge...")
                success = merge_sharded_checkpoints(save_base, world_size, output_path)
            
            if success:
                merge_duration = time.time() - merge_start_time
                print(f"✅ Automatic merge completed in {merge_duration:.2f}s")
                print(f"📁 Merged checkpoint saved to: {output_path}")
                
                # Clean up shard files after successful merge
                print("🧹 Cleaning up shard files...")
                for r in range(world_size):
                    shard_file = f"{save_base}.rank{r}.pt"
                    if os.path.exists(shard_file):
                        os.remove(shard_file)
                        print(f"  Removed: {shard_file}")
            else:
                print("❌ Automatic merge failed!")
                print("⚠️  Shard files will be preserved for manual merge")
                
        except Exception as e:
            print(f"❌ Error during automatic merge: {e}")
            print("⚠️  Shard files will be preserved for manual merge")
    
    # Wait for rank 0 to complete merging
    dist.barrier()


def load_sharded_checkpoint(model: FSDP, optimizer, save_path, map_location='cpu'):
    """
    각 rank가 자기 shard만 로드합니다.
    """
    assert dist.is_initialized(), "Distributed not initialized"
    rank = dist.get_rank()
    save_base = save_path.rstrip(".pt")
    shard_path = f"{save_base}.rank{rank}.pt"

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        checkpoint = torch.load(
            shard_path,
            map_location=map_location,
            weights_only=False
        )
        model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint.get('epoch', 0)
    print(f"(rank {rank}) ✅ Sharded checkpoint loaded from: {shard_path} (epoch={epoch})")

    dist.barrier()
    return epoch