import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import AutoTokenizer, AutoModel
import math

# -------------------- Core Model Definitions --------------------

def orthogonal_rotation_matrix(d_model, device=None):
    """Generate an orthogonal rotation matrix for role-sensitive embeddings"""
    # Use a more stable initialization
    A = torch.randn(d_model, d_model, device=device)
    # Add small regularization to prevent singular matrices
    A = A + 0.01 * torch.eye(d_model, device=device)
    
    # Use SVD instead of QR for better numerical stability
    try:
        U, S, V = torch.svd(A)
        # Ensure proper orthogonality
        Q = U @ V.t()
        # Ensure determinant is 1 (proper rotation)
        if torch.det(Q) < 0:
            Q[:, -1] *= -1
        return Q
    except:
        # Fallback to identity if SVD fails
        print("Warning: SVD failed, using identity matrix")
        return torch.eye(d_model, device=device)

class RoleSensitiveEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, pad_idx=0, device=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        R = orthogonal_rotation_matrix(d_model, device=device)
        self.register_buffer('R', R)

    def forward(self, token_ids, role_mask):
        h = self.embedding(token_ids)
        B, L, D = h.size()
        h_flat = h.view(-1, D)
        data_mask = ~role_mask.view(-1)
        if data_mask.any():
            rotated = h_flat[data_mask] @ self.R.t()
            h_flat[data_mask] = rotated
        return h_flat.view(B, L, D)

class RoleGatedSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, bias_delta=1.0):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.delta = nn.Parameter(torch.tensor(bias_delta))
        
        # Initialize weights properly
        self._init_weights()
        
    def _init_weights(self):
        # Xavier initialization for better numerical stability
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        # Initialize biases to zero
        nn.init.constant_(self.q_proj.bias, 0.0)
        nn.init.constant_(self.k_proj.bias, 0.0)
        nn.init.constant_(self.v_proj.bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x, role_mask, attn_mask=None, key_padding_mask=None):
        B, L, _ = x.size()
        
        # QKV projection
        Q = self.q_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1,2)
        K = self.k_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1,2)
        V = self.v_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1,2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2,-1)) * self.scale
        
        # Apply role-based bias: same roles get positive bias (optimized)
        # Create role comparison matrix more efficiently
        role_mask_expanded = role_mask.unsqueeze(2)  # [B, L, 1]
        role_mask_transposed = role_mask.unsqueeze(1)  # [B, 1, L]
        
        # Create same-role mask: True where roles match
        same_role_mask = (role_mask_expanded == role_mask_transposed).float()
        
        # Apply role bias where roles match
        role_bias = same_role_mask * self.delta  # [B, L, L]
        
        # Expand bias to match number of heads
        role_bias = role_bias.unsqueeze(1).expand(B, self.nhead, L, L)
        scores = scores + role_bias
        
        # Apply masks
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.bool(), float('-inf'))
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask.bool(), float('-inf'))
        
        # Apply softmax with numerical stability
        attn = torch.softmax(scores, dim=-1)
        # Handle NaN values that can occur during training
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply attention to values
        out = torch.matmul(attn, V)
        out = out.transpose(1,2).contiguous().view(B,L,-1)
        out = self.out_proj(out)
        
        return out

class RoleAwareTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, bias_delta=1.0):
        super().__init__()
        self.self_attn = RoleGatedSelfAttention(d_model, nhead, bias_delta)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, role_mask, src_mask=None, src_key_padding_mask=None):
        # Self-attention
        attn_out = self.self_attn(src, role_mask, src_mask, src_key_padding_mask)
        
        # First residual connection and normalization
        src = src + self.dropout1(attn_out)
        src = self.norm1(src)
        
        # Feed-forward network
        ff = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        
        # Second residual connection and normalization
        src = src + self.dropout2(ff)
        result = self.norm2(src)
        
        return result

class RoleAwareTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6,
                 dim_feedforward=2048, dropout=0.1, bias_delta=1.0, pad_idx=0, device=None):
        super().__init__()
        self.embedding = RoleSensitiveEmbedding(vocab_size, d_model, pad_idx, device)
        self.pos_encoder = nn.Embedding(5000, d_model)
        self.layers = nn.ModuleList([
            RoleAwareTransformerLayer(d_model, nhead, dim_feedforward, dropout, bias_delta)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, token_ids, role_mask, src_key_padding_mask=None):
        B, L = token_ids.size()
        
        # Embedding
        x = self.embedding(token_ids, role_mask)
        
        # Position encoding
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B,L)
        x = x + self.pos_encoder(positions)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, role_mask, src_key_padding_mask=src_key_padding_mask)
        
        # Final normalization
        return self.norm(x)

# -------------------- Helper Functions --------------------

def create_model(args, tokenizer):
    """Create the model and classification head"""
    # Get embedding dimension from tokenizer
    if hasattr(tokenizer, 'model_max_length'):
        max_seq_len = min(tokenizer.model_max_length, args.max_seq_len)
    else:
        max_seq_len = args.max_seq_len
    
    # Get vocab size from tokenizer
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer.get_vocab())
    
    # Get pad token id
    pad_idx = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    # Get bias_delta from args
    bias_delta = getattr(args, 'bias_delta', 1.0)
    
    # Create model
    model = RoleAwareTransformerEncoder(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.d_model * 4,  # Standard transformer feedforward size
        dropout=args.dropout,
        bias_delta=bias_delta,
        pad_idx=pad_idx
    )
    
    # Create classification head
    head = nn.Linear(args.d_model, 1)  # For binary classification or regression
    
    return model, head

def adapt_pretrained_weights(rgt_model, pretrained_model):
    """Adapt pretrained weights to RGT model"""
    try:
        pretrained_dict = pretrained_model.state_dict()
        rgt_dict = rgt_model.state_dict()
        
        # Filter out unnecessary keys and adapt dimensions
        adapted_dict = {}
        for k, v in pretrained_dict.items():
            if k in rgt_dict and rgt_dict[k].shape == v.shape:
                adapted_dict[k] = v
        
        # Load adapted weights
        rgt_model.load_state_dict(adapted_dict, strict=False)
        print(f"Successfully adapted {len(adapted_dict)} pretrained parameters")
        return True
        
    except Exception as e:
        print(f"Error adapting pretrained weights: {e}")
        return False

def save_checkpoint(model, head, optimizer, scheduler, epoch, loss, path):
    """Save model checkpoint with DataParallel support"""
    # Extract the actual model from DataParallel wrapper if needed
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    head_state = head.module.state_dict() if hasattr(head, 'module') else head.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'head_state_dict': head_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, head, optimizer, scheduler, path):
    """Load model checkpoint with DataParallel support"""
    checkpoint = torch.load(path)
    
    # Handle DataParallel wrapped models
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        
    if hasattr(head, 'module'):
        head.module.load_state_dict(checkpoint['head_state_dict'])
    else:
        head.load_state_dict(checkpoint['head_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']