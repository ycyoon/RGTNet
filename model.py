import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import AutoTokenizer, AutoModel
import math

class RoleAwareTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Position embedding
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Role embedding (for different roles like user, assistant, etc.)
        self.role_embedding = nn.Embedding(10, d_model)  # Support up to 10 roles
        
        # Transformer layers
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu'
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, role_mask=None, attention_mask=None):
        batch_size, seq_len = x.size()
        
        # First convert input_ids to embeddings
        x = self.embedding(x)
        
        # Add positional encoding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        
        # Add role embedding if provided
        if role_mask is not None:
            role_emb = self.role_embedding(role_mask)
            x = x + pos_emb + role_emb
        else:
            x = x + pos_emb
        
        # Apply transformer
        if attention_mask is not None:
            # Convert attention mask to key_padding_mask (inverted)
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None
        
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        return self.norm(x)

def create_model(args, tokenizer):
    """Create the model and classification head"""
    # Get embedding dimension from tokenizer
    if hasattr(tokenizer, 'model_max_length'):
        max_seq_len = min(tokenizer.model_max_length, args.max_seq_len)
    else:
        max_seq_len = args.max_seq_len
    
    # Get vocab size from tokenizer
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 30522
    
    # Create model
    model = RoleAwareTransformerEncoder(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_seq_len=max_seq_len
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
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'head_state_dict': head.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, head, optimizer, scheduler, path):
    """Load model checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    head.load_state_dict(checkpoint['head_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']