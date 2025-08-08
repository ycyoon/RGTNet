#!/usr/bin/env python3
"""
RGTNet Model with Role-Aware Architecture and Token Democracy
Preserves core RGTNet ideas: RoleSensitiveEmbedding, orthogonal transformations, RoleGatedSelfAttention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import time
import os

# -------------------- Core RGTNet Components --------------------

def _create_single_orthogonal_matrix(dim, device=None):
    """Generates a single random orthogonal matrix of shape (dim, dim)."""
    random_matrix = torch.randn(dim, dim, device=device)
    q, r = torch.linalg.qr(random_matrix)
    return q

def _create_stacked_orthogonal_matrices(num_matrices, dim, device=None):
    """Generates a stack of random orthogonal matrices of shape (num_matrices, dim, dim)."""
    return torch.stack([_create_single_orthogonal_matrix(dim, device) for _ in range(num_matrices)])

class RoleSensitiveEmbedding(nn.Module):
    """
    Role-sensitive token embedding with token democracy.
    Core RGTNet component that applies role-specific transformations.
    """
    def __init__(self, vocab_size, d_model, pad_idx, device=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        # 0: user, 1: agent
        self.role_transformers = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(2) 
        ])
        
        # CRITICAL: Initialize role transformers to identity matrices
        # This preserves the pretrained embeddings at the start of training.
        with torch.no_grad():
            for transformer in self.role_transformers:
                transformer.weight.copy_(torch.eye(d_model))

    def forward(self, input_ids, role_mask):
        """
        input_ids: (B, L)
        role_mask: (B, L) with 0 for user, 1 for agent
        """
        x = self.embedding(input_ids)
        
        # Apply role-specific transformation for token democracy
        transformed_x = torch.zeros_like(x)
        
        # Apply transformation for each role
        for role_idx, transformer in enumerate(self.role_transformers):
            # Create a mask for the current role
            role_specific_mask = (role_mask == role_idx)
            # Apply the transformation where the mask is True
            transformed_x[role_specific_mask] = transformer(x[role_specific_mask]).to(transformed_x.dtype)
            
        return transformed_x

class RoleGatedSelfAttention(nn.Module):
    """
    Role-gated self-attention mechanism with orthogonal matrix transformations.
    Core RGTNet component implementing token democracy through key rotation.
    """
    def __init__(self, d_model, nhead, dropout=0.1, delta=1.0):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.delta = nn.Parameter(torch.tensor([delta]))
        
        # Orthogonal matrix for rotation - core RGTNet feature
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
        
        # Role-based orthogonal rotation for token democracy
        if self.delta > 0:
            # Expand the buffer to the batch dimension and ensure matching dtype
            U_batch = self.U.unsqueeze(0).expand(bsz, -1, -1, -1).to(dtype=k.dtype, device=k.device)
            
            # Reshape K for rotation
            k_reshaped = k.view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2).contiguous()
            
            # Create a temporary tensor to hold the rotated values
            rotated_k_part = torch.matmul(k_reshaped, U_batch)

            # Apply role-gated rotation
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
            is_causal=False
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(attn_output)

class RoleAwareTransformerLayer(nn.Module):
    """
    Single transformer layer with role-aware attention.
    Core RGTNet component combining role-gated attention with standard FFN.
    """
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
        # Self-attention with role awareness
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
    """Gradient checkpointed version for memory efficiency"""
    def forward(self, src, role_mask, src_mask=None, src_key_padding_mask=None):
        return checkpoint(super().forward, src, role_mask, src_mask, src_key_padding_mask, use_reentrant=False)

class RoleAwareTransformerDecoder(nn.Module):
    """
    RGTNet's main model: Role-aware Transformer Decoder with token democracy.
    Combines role-sensitive embeddings, orthogonal transformations, and causal LM capabilities.
    """
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, 
                 dropout=0.1, bias_delta=1.0, pad_idx=0, max_seq_len=8192, device=None, 
                 pretrained_model_name=None, use_gradient_checkpointing=False):
        super().__init__()
        
        # Core RGTNet components
        self.embedding = RoleSensitiveEmbedding(vocab_size, d_model, pad_idx, device)
        self.pos_encoder = nn.Embedding(512, d_model)  # Support up to 512 positions
        
        layer_class = CheckpointedRoleAwareTransformerLayer if use_gradient_checkpointing else RoleAwareTransformerLayer
        self.layers = nn.ModuleList([
            layer_class(d_model, nhead, dim_feedforward, dropout, bias_delta)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Initialize with pretrained weights if specified
        if pretrained_model_name is not None:
            self._load_pretrained_weights(pretrained_model_name)

    def _load_pretrained_weights(self, pretrained_model_name):
        """Load pretrained weights while preserving RGTNet architecture"""
        try:
            print(f"Loading pretrained weights from {pretrained_model_name}...")
            llama = AutoModelForCausalLM.from_pretrained(pretrained_model_name, trust_remote_code=True)
            
            with torch.no_grad():
                # Copy embedding weights
                llama_embed = llama.get_input_embeddings().weight
                my_embed = self.embedding.embedding.weight
                n = min(my_embed.size(0), llama_embed.size(0))
                my_embed[:n].copy_(llama_embed[:n])
                print(f"✅ Copied embedding weights ({n} tokens)")
                
                # Copy LM head weights
                if hasattr(llama, 'lm_head') and hasattr(llama.lm_head, 'weight'):
                    llama_lm_head_weight = llama.lm_head.weight
                    my_lm_head = self.lm_head.weight
                    n_lm = min(my_lm_head.size(0), llama_lm_head_weight.size(0))
                    my_lm_head[:n_lm].copy_(llama_lm_head_weight[:n_lm])
                    print(f"✅ Copied LM head weights ({n_lm} tokens)")
                else:
                    print("⚠️ Could not find LM head in pretrained model")

            print("✅ Pretrained weights loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to load pretrained weights: {e}")
            print("🔄 Continuing with random initialization")

    def forward(self, input_ids, role_mask, labels=None):
        B, L = input_ids.size()
        x = self.embedding(input_ids, role_mask)
        
        # Add positional encoding with bounds checking
        max_pos = self.pos_encoder.num_embeddings - 1
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        positions = torch.clamp(positions, 0, max_pos)
        
        try:
            pos_embeddings = self.pos_encoder(positions)
            x = x + pos_embeddings
        except Exception as e:
            print(f"Warning: Positional encoding failed: {e}")
            x = x + torch.zeros_like(x)

        # Create causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        causal_mask = causal_mask.to(dtype=x.dtype)

        # Apply role-aware transformer layers
        for layer in self.layers:
            x = layer(x, role_mask, src_mask=causal_mask)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        
        return {'loss': loss, 'logits': logits}

    @torch.no_grad()
    def generate(self, input_ids, max_length=100, max_new_tokens=None, num_return_sequences=1,
                 do_sample=False, top_k=50, top_p=0.95, pad_token_id=None, **kwargs):
        """Generate sequences with role-aware attention"""
        self.eval()
        
        actual_vocab_size = self.embedding.embedding.num_embeddings
        input_ids = torch.clamp(input_ids, 0, actual_vocab_size - 1)
        
        if max_new_tokens is not None:
            max_length = input_ids.size(1) + max_new_tokens
            
        sequences = [input_ids.clone()]
        
        if pad_token_id is None:
            pad_token_id = actual_vocab_size - 1
        pad_token_id = min(max(0, pad_token_id), actual_vocab_size - 1)
            
        max_generation_steps = min(max_length - input_ids.size(1), 20)
        
        with torch.no_grad():
            for step in range(max_generation_steps):
                seq = sequences[0]
                
                if seq.size(1) >= max_length:
                    break
                    
                seq = torch.clamp(seq, 0, actual_vocab_size - 1)
                sequences[0] = seq
                
                # Create dynamic role mask: input=user(0), generated=assistant(1)
                role_mask = torch.zeros_like(seq, dtype=torch.long)
                if seq.size(1) > input_ids.size(1):
                    role_mask[:, input_ids.size(1):] = 1
                
                try:
                    outputs = self.forward(seq, role_mask)
                    
                    if 'logits' not in outputs:
                        break
                        
                    next_token_logits = outputs['logits'][:, -1, :]
                    
                    # Ensure logits match vocab size
                    if next_token_logits.size(-1) != actual_vocab_size:
                        if next_token_logits.size(-1) > actual_vocab_size:
                            next_token_logits = next_token_logits[:, :actual_vocab_size]
                        else:
                            padding_size = actual_vocab_size - next_token_logits.size(-1)
                            padding = torch.full((next_token_logits.size(0), padding_size), 
                                               -1e9, device=next_token_logits.device, dtype=next_token_logits.dtype)
                            next_token_logits = torch.cat([next_token_logits, padding], dim=-1)
                    
                    # Apply temperature
                    temperature = 0.8
                    next_token_logits = next_token_logits / temperature
                    
                    # Generate next token
                    if do_sample and top_k > 0:
                        top_k_actual = min(top_k, next_token_logits.size(-1))
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k_actual)
                        probs = torch.softmax(top_k_logits, dim=-1)
                        sampled_idx = torch.multinomial(probs, num_samples=1)
                        next_token = top_k_indices.gather(-1, sampled_idx)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    next_token = torch.clamp(next_token, 0, actual_vocab_size - 1)
                    
                    # Append new token
                    sequences[0] = torch.cat([seq, next_token], dim=1)
                    
                    # Check for pad token
                    if pad_token_id is not None and torch.any(next_token == pad_token_id):
                        break
                        
                    if sequences[0].size(1) >= max_length:
                        break
                        
                except Exception as e:
                    print(f"Warning: Generation step {step} failed: {e}")
                    fallback_token = torch.full((input_ids.size(0), 1), pad_token_id, 
                                               device=input_ids.device, dtype=input_ids.dtype)
                    fallback_token = torch.clamp(fallback_token, 0, actual_vocab_size - 1)
                    
                    if seq.size(1) < max_length:
                        sequences[0] = torch.cat([seq, fallback_token], dim=1)
                    break
        
        final_sequence = torch.clamp(sequences[0], 0, actual_vocab_size - 1)
        return final_sequence

# -------------------- Model Creation --------------------

def create_model(args, pad_idx):
    """
    Creates the RoleAwareTransformerDecoder model with RGTNet's core features.
    Preserves role-sensitive embeddings, orthogonal transformations, and token democracy.
    """
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
            print(f"Warning: Could not auto-detect config: {e}")
            d_model = getattr(args, 'd_model', 512)
            nhead = getattr(args, 'nhead', 8)
            num_layers = getattr(args, 'num_layers', 6)
    else:
        d_model = getattr(args, 'd_model', 512)
        nhead = getattr(args, 'nhead', 8)
        num_layers = getattr(args, 'num_layers', 6)
    
    dim_feedforward = getattr(args, 'dim_feedforward', d_model * 4)
    
    model = RoleAwareTransformerDecoder(
        vocab_size=args.vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=args.dropout,
        bias_delta=args.bias_delta,
        pad_idx=pad_idx,
        max_seq_len=args.max_seq_len,
        pretrained_model_name=args.pretrained_model_name,
        use_gradient_checkpointing=getattr(args, 'gradient_checkpointing', False)
    )
    
    print("✅ RGTNet model created with core features:")
    print("   - RoleSensitiveEmbedding (token democracy)")
    print("   - RoleGatedSelfAttention (orthogonal transformations)")
    print("   - Pretrained weight initialization")
    
    return model

def load_checkpoint(checkpoint_path, model, device='cuda'):
    """
    Load a trained model checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint directory or file
        model: The model to load weights into
        device: Device to load the model on
    
    Returns:
        The loaded model
    """
    import os
    import json
    
    print(f"🔄 Loading checkpoint from: {checkpoint_path}")
    
    # Check if it's a directory (DeepSpeed checkpoint) or file
    if os.path.isdir(checkpoint_path):
        # DeepSpeed checkpoint directory
        print("📁 Detected DeepSpeed checkpoint directory")
        
        # Look for latest checkpoint
        latest_file = os.path.join(checkpoint_path, "latest")
        if os.path.exists(latest_file):
            with open(latest_file, 'r') as f:
                latest_tag = f.read().strip()
            checkpoint_dir = os.path.join(checkpoint_path, latest_tag)
            print(f"📋 Using latest checkpoint: {latest_tag}")
        else:
            # Fallback: look for epoch directories
            epoch_dirs = [d for d in os.listdir(checkpoint_path) if d.startswith('epoch_')]
            if epoch_dirs:
                latest_epoch = sorted(epoch_dirs, key=lambda x: int(x.split('_')[1]))[-1]
                checkpoint_dir = os.path.join(checkpoint_path, latest_epoch)
                print(f"📋 Using latest epoch: {latest_epoch}")
            else:
                checkpoint_dir = checkpoint_path
                print("📋 Using checkpoint directory directly")
        
        # Load DeepSpeed checkpoint
        try:
            from deepspeed.checkpoint import load_checkpoint as ds_load_checkpoint
            _, _, _, _ = ds_load_checkpoint(checkpoint_dir, model)
            print("✅ DeepSpeed checkpoint loaded successfully")
        except ImportError:
            print("⚠️ DeepSpeed not available, trying manual loading...")
            # Manual loading for DeepSpeed checkpoints
            model_states_file = os.path.join(checkpoint_dir, "mp_rank_00_model_states.pt")
            if os.path.exists(model_states_file):
                checkpoint = torch.load(model_states_file, map_location=device)
                if 'module' in checkpoint:
                    model.load_state_dict(checkpoint['module'])
                else:
                    model.load_state_dict(checkpoint)
                print("✅ Manual checkpoint loading successful")
            else:
                raise FileNotFoundError(f"Model states file not found: {model_states_file}")
    
    else:
        # Single file checkpoint
        print("📄 Detected single file checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        print("✅ Single file checkpoint loaded successfully")
    
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded and moved to {device}")
    return model