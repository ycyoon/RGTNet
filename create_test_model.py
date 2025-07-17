#!/usr/bin/env python3
"""
Test script for benchmark evaluation
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer

# Create a simple test model
class SimpleTestModel(nn.Module):
    def __init__(self, vocab_size, d_model=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True)
            for _ in range(3)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, input_ids, role_mask=None, src_key_padding_mask=None):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return self.norm(x)

class SimpleHead(nn.Module):
    def __init__(self, d_model=256, num_classes=2):
        super().__init__()
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        return self.classifier(x)

def create_test_checkpoint():
    """Create a test checkpoint for debugging"""
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size
    
    # Create test model and head
    model = SimpleTestModel(vocab_size)
    head = SimpleHead()
    
    # Create checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'head_state_dict': head.state_dict(),
        'epoch': 1,
        'best_accuracy': 0.85
    }
    
    # Save checkpoint
    torch.save(checkpoint, 'test_model.pth')
    print("Test checkpoint created: test_model.pth")

if __name__ == '__main__':
    create_test_checkpoint()
