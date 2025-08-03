#!/usr/bin/env python3
"""
Hybrid RGTNet: Combines Llama's pretrained architecture with RGTNet's role-aware features
Strategy: Keep Llama's core architecture intact, add role-aware adapters
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

# -------------------- RGTNet Components as Adapters --------------------

def _create_single_orthogonal_matrix(dim, device=None):
    """Generates a single random orthogonal matrix of shape (dim, dim)."""
    random_matrix = torch.randn(dim, dim, device=device)
    q, r = torch.linalg.qr(random_matrix)
    return q

def _create_stacked_orthogonal_matrices(num_matrices, dim, device=None):
    """Generates a stack of random orthogonal matrices of shape (num_matrices, dim, dim)."""
    return torch.stack([_create_single_orthogonal_matrix(dim, device) for _ in range(num_matrices)])

# ---- Modified Attention with Explicit Role-Based Bias ----
class RoleGatedAttention(LlamaAttention):
    """
    Role-gated attention that adds explicit bias to attention scores based on role mask
    """
    def __init__(self, config, layer_idx=None, delta=1.0):
        super().__init__(config, layer_idx)
        
        # Ensure required attributes are properly set (in case parent class doesn't set them)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        # Learnable bias scalar for role-based attention
        self.delta = nn.Parameter(torch.tensor(delta))
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        role_mask=None,
        **kwargs,
    ):
        """
        Forward pass with role-based attention bias
        """
        # If no role_mask provided, use standard LlamaAttention
        if role_mask is None:
            result = super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs
            )
            # Ensure we return only 2 values for compatibility
            if len(result) == 3:
                return result[0], result[1]  # (attn_output, attn_weights)
            else:
                return result
        
        # Apply role-based attention with manual implementation
        bsz, q_len, _ = hidden_states.size()

        # Standard projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Handle grouped-query attention
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Apply role-based bias
        # role_mask: [B, L] -> [B, 1, 1, L] for broadcasting to attention weights
        role_bias = self.delta * role_mask.unsqueeze(1).unsqueeze(2)
        attn_weights = attn_weights + role_bias

        # Apply attention mask if provided
        if attention_mask is not None:
            # Ensure attention_mask has correct shape for broadcasting
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights + attention_mask

        # Apply softmax and dropout
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Apply dropout if training
        if self.training and hasattr(self, 'attention_dropout'):
            attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout)
        
        # Compute output
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        # Most Llama layers expect only 2 return values: (hidden_states, attention_weights)
        # For backward compatibility, return only 2 values like standard LlamaAttention
        return attn_output, attn_weights

    def _repeat_kv(self, hidden_states, n_rep):
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class RoleAwareAdapter(nn.Module):
    """
    Role-aware adapter that can be attached to Llama layers
    Preserves original Llama functionality while adding role-awareness
    """
    def __init__(self, d_model, nhead, bias_delta=1.0):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.bias_delta = bias_delta
        
        # Role-aware gating parameter
        self.delta = nn.Parameter(torch.tensor([bias_delta]))
        
        # Orthogonal matrices for token democracy
        self.register_buffer("U", _create_stacked_orthogonal_matrices(self.nhead, self.head_dim))
        
        # Small adaptation layers (much smaller than full transformation)
        self.role_gate = nn.Linear(d_model, 1, bias=False)
        
        # Initialize to minimal impact
        with torch.no_grad():
            self.role_gate.weight.fill_(0.0)
    
    def forward(self, hidden_states, role_mask, attention_mask=None):
        """
        Apply role-aware adaptation to hidden states
        """
        B, L, D = hidden_states.shape
        
        if self.delta.item() == 0 or role_mask is None:
            # No role adaptation
            return hidden_states
        
        # Ensure consistent dtype for all operations
        original_dtype = hidden_states.dtype
        
        # Compute role-based gating
        role_gate_logits = self.role_gate(hidden_states)  # [B, L, 1]
        role_influence = torch.sigmoid(self.delta * role_gate_logits).squeeze(-1)  # [B, L]
        
        # Apply role mask weighting
        role_weight = role_mask.float().to(original_dtype) * role_influence
        
        # Minimal adaptation: small perturbation based on role
        # This preserves most of Llama's original computation
        adaptation = torch.tanh(role_weight.unsqueeze(-1)) * 0.1  # Small scale
        adapted_states = hidden_states * (1.0 + adaptation)
        
        return adapted_states.to(original_dtype)

class HybridLlamaRGTNet(nn.Module):
    """
    Hybrid model: Llama architecture + RGTNet role-aware features
    Supports both role-aware adapters and explicit role-gated attention
    """
    def __init__(self, pretrained_model_name, enable_role_adapters=True, use_explicit_bias=False, 
                 bias_delta=1.0, use_quantization=False):
        super().__init__()
        
        # Load base Llama model
        if use_quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.base_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            self.base_model = prepare_model_for_kbit_training(self.base_model)
        else:
            # Check if we're in a DeepSpeed environment
            is_deepspeed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
            device_map = None if is_deepspeed else "auto"
            
            self.base_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name,
                torch_dtype=torch.float16,
                device_map=device_map,
                trust_remote_code=True,
            )
            
            # If in DeepSpeed environment, move model to CUDA manually
            if is_deepspeed and torch.cuda.is_available():
                local_rank = int(os.environ.get('LOCAL_RANK', 0))
                self.base_model = self.base_model.to(f'cuda:{local_rank}')
        
        self.config = self.base_model.config
        self.enable_role_adapters = enable_role_adapters
        self.use_explicit_bias = use_explicit_bias
        self.bias_delta = bias_delta
        
        # Replace attention modules with role-gated attention if requested
        if use_explicit_bias:
            for i, layer in enumerate(self.base_model.model.layers):
                layer.self_attn = RoleGatedAttention(self.config, layer_idx=i, delta=bias_delta)
        
        # Add role-aware adapters to each layer
        if enable_role_adapters:
            self.role_adapters = nn.ModuleList([
                RoleAwareAdapter(
                    d_model=self.config.hidden_size,
                    nhead=self.config.num_attention_heads,
                    bias_delta=bias_delta
                )
                for _ in range(self.config.num_hidden_layers)
            ])
        
        print(f"✅ Hybrid Llama-RGTNet created:")
        print(f"   - Base model: {pretrained_model_name}")
        print(f"   - Role adapters: {'Enabled' if enable_role_adapters else 'Disabled'}")
        print(f"   - Explicit bias: {'Enabled' if use_explicit_bias else 'Disabled'}")
        print(f"   - Bias delta: {bias_delta}")
        print(f"   - Quantization: {'Enabled' if use_quantization else 'Disabled'}")
    
    def forward(self, input_ids, attention_mask=None, role_mask=None, labels=None, **kwargs):
        """
        Forward pass with optional role-aware adaptation
        Supports both explicit bias in attention and adapter-based approach
        """
        # Handle explicit bias case
        if self.use_explicit_bias and role_mask is not None:
            # Pass role_mask to the base model's attention layers
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,  # Compute loss separately
                output_hidden_states=True,
                role_mask=role_mask,  # This will be passed to RoleGatedAttention
                **kwargs
            )
            hidden_states = outputs.hidden_states[-1]
            
            # Apply adapter-based role adaptation if also enabled
            if self.enable_role_adapters and len(self.role_adapters) > 0:
                adapted_hidden = self.role_adapters[-1](
                    hidden_states,
                    role_mask,
                    attention_mask
                )
                # Ensure dtype consistency for DeepSpeed FP16
                if hasattr(self.base_model.lm_head.weight, 'dtype'):
                    adapted_hidden = adapted_hidden.to(self.base_model.lm_head.weight.dtype)
                logits = self.base_model.lm_head(adapted_hidden)
            else:
                logits = self.base_model.lm_head(hidden_states)
                
        elif self.enable_role_adapters and role_mask is not None:
            # Adapter-only approach (original implementation)
            with torch.no_grad() if not self.training else torch.enable_grad():
                # Get standard Llama outputs
                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=None,  # Don't compute loss yet
                    output_hidden_states=True,
                    **kwargs
                )
                
                # Apply role-aware adaptation to hidden states
                hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
                
                # Apply role adaptation (simplified for now)
                if len(self.role_adapters) > 0:
                    # Use the last adapter for simplicity
                    adapted_hidden = self.role_adapters[-1](
                        hidden_states,
                        role_mask,
                        attention_mask
                    )
                    
                    # Ensure dtype consistency for DeepSpeed FP16
                    if hasattr(self.base_model.lm_head.weight, 'dtype'):
                        adapted_hidden = adapted_hidden.to(self.base_model.lm_head.weight.dtype)
                    
                    # Recompute logits with adapted hidden states
                    logits = self.base_model.lm_head(adapted_hidden)
                else:
                    logits = outputs.logits
        else:
            # Standard Llama forward pass
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
            return outputs
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits
        }
    
    def generate(self, input_ids, role_mask=None, **generation_kwargs):
        """
        Generation with role awareness
        """
        if not self.enable_role_adapters or role_mask is None:
            # Use standard Llama generation
            return self.base_model.generate(input_ids, **generation_kwargs)
        
        # Custom generation with role awareness
        # For now, fall back to standard generation
        # TODO: Implement role-aware generation
        return self.base_model.generate(input_ids, **generation_kwargs)

def create_hybrid_model(args):
    """
    Create hybrid Llama-RGTNet model
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create hybrid model
    model = HybridLlamaRGTNet(
        pretrained_model_name=args.pretrained_model_name,
        enable_role_adapters=getattr(args, 'enable_role_adapters', True),
        use_explicit_bias=getattr(args, 'use_explicit_bias', False),
        bias_delta=getattr(args, 'bias_delta', 1.0),
        use_quantization=getattr(args, 'use_quantization', False)
    )
    
    # Apply LoRA if requested
    if getattr(args, 'use_lora', False):
        print(f"🔍 Debug: Applying LoRA to model...")
        print(f"🔍 Debug: Model type before LoRA: {type(model)}")
        print(f"🔍 Debug: Base model type: {type(model.base_model)}")
        print(f"🔍 Debug: Has PEFT config: {hasattr(model.base_model, 'peft_config')}")
        
        # Check if LoRA is already applied
        if not hasattr(model.base_model, 'peft_config'):
            lora_r = getattr(args, 'lora_r', 8)
            lora_alpha = getattr(args, 'lora_alpha', 1)  # Set alpha to 1 for very small scaling=0.125
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=getattr(args, 'lora_dropout', 0.05),
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            print(f"🔧 Applying LoRA config: r={lora_r}, alpha={lora_alpha}")
            model.base_model = get_peft_model(model.base_model, lora_config)
            print(f"✅ LoRA adapters applied to base model (r={lora_r}, alpha={lora_alpha}, scaling={lora_alpha/lora_r})")
            print(f"🔍 Debug: Base model type after LoRA: {type(model.base_model)}")
            print(f"🔍 Debug: Has PEFT config after: {hasattr(model.base_model, 'peft_config')}")
        else:
            print("✅ LoRA adapters already applied")
    else:
        print("⚠️  LoRA not requested (use_lora=False)")
    
    return model, tokenizer

# Alternative creation function for explicit bias models
def create_model_variant(args):
    """
    Create model variants with explicit bias support
    Based on the provided example code
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model with explicit bias support
    model = HybridLlamaRGTNet(
        pretrained_model_name=args.pretrained_model_name,
        enable_role_adapters=getattr(args, 'enable_role_adapters', True),
        use_explicit_bias=getattr(args, 'use_explicit_bias', False),
        bias_delta=getattr(args, 'bias_delta', 1.0),
        use_quantization=getattr(args, 'use_quantization', False)
    )
    
    return model, tokenizer

# For backward compatibility
def create_model(args, pad_idx):
    """Backward compatibility wrapper"""
    model, tokenizer = create_hybrid_model(args)
    return model