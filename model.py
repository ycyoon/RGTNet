import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import math
import os
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# -------------------- Core Model Definitions --------------------

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
            # Expand the buffer to the batch dimension
            U_batch = self.U.unsqueeze(0).expand(bsz, -1, -1, -1)
            
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
        self.pos_encoder = nn.Embedding(max_seq_len, d_model)
        
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
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = x + self.pos_encoder(positions)

        # Create a causal mask for the decoder
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)

        for layer in self.layers:
            x = layer(x, role_mask, src_mask=causal_mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            # Flatten the tokens
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        
        return {'loss': loss, 'logits': logits}

# -------------------- Model Creation and Loading --------------------

def create_model(args, tokenizer):
    vocab_size = tokenizer.vocab_size
    pad_idx = tokenizer.pad_token_id
    if pad_idx is None or pad_idx >= vocab_size or pad_idx < 0:
        pad_idx = 0 # Default pad token if not set or invalid
        print(f"[WARN] pad_token_id not found or invalid in tokenizer. Using default pad_idx={pad_idx}")

    model = RoleAwareTransformerDecoder(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.d_model * 4,
        dropout=args.dropout,
        bias_delta=args.bias_delta,
        pad_idx=pad_idx,
        max_seq_len=args.max_seq_len,
        pretrained_model_name=args.pretrained_model_name,
        use_gradient_checkpointing=args.gradient_checkpointing
    )
    return model, tokenizer

def save_checkpoint(model, optimizer, epoch, save_path):
    """Save model checkpoint."""
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel, FSDP)):
        model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
    print(f"Checkpoint saved to {save_path}")

def load_checkpoint(model, optimizer, save_path, device):
    """Load model checkpoint."""
    if not os.path.exists(save_path):
        print(f"No checkpoint found at {save_path}. Starting from scratch.")
        return 0
        
    checkpoint = torch.load(save_path, map_location=device)
    
    # Handle DDP-saved models by removing 'module.' prefix
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict)
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {save_path}, resuming from epoch {epoch+1}")
    return epoch + 1