#!/usr/bin/env python3
"""
Hybrid RGTNet: Combines Llama's pretrained architecture with RGTNet's role-aware features
Strategy: Keep Llama's core architecture intact, add role-aware adapters
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
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
    """
    def __init__(self, pretrained_model_name, enable_role_adapters=True, use_quantization=False):
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
        
        # Add role-aware adapters to each layer
        if enable_role_adapters:
            self.role_adapters = nn.ModuleList([
                RoleAwareAdapter(
                    d_model=self.config.hidden_size,
                    nhead=self.config.num_attention_heads,
                    bias_delta=1.0
                )
                for _ in range(self.config.num_hidden_layers)
            ])
        
        print(f"✅ Hybrid Llama-RGTNet created:")
        print(f"   - Base model: {pretrained_model_name}")
        print(f"   - Role adapters: {'Enabled' if enable_role_adapters else 'Disabled'}")
        print(f"   - Quantization: {'Enabled' if use_quantization else 'Disabled'}")
    
    def forward(self, input_ids, attention_mask=None, role_mask=None, labels=None, **kwargs):
        """
        Forward pass with optional role-aware adaptation
        """
        if not self.enable_role_adapters or role_mask is None:
            # Standard Llama forward pass
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
        
        # For now, use standard forward pass and apply role adaptation afterwards
        # This is safer and avoids position embedding issues
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

# For backward compatibility
def create_model(args, pad_idx):
    """Backward compatibility wrapper"""
    model, tokenizer = create_hybrid_model(args)
    return model