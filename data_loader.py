import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import json
import os
from pathlib import Path

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Prepare input text
        if isinstance(item, dict):
            if 'conversations' in item:
                text = self._format_conversation(item['conversations'])
            elif 'instruction' in item:
                text = f"Instruction: {item['instruction']}\nResponse: {item.get('output', '')}"
            else:
                text = str(item)
        else:
            text = str(item)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(0, dtype=torch.long),  # Dummy label
            'original_text': text
        }
    
    def _format_conversation(self, conversations):
        """Format conversation data"""
        formatted = []
        for conv in conversations:
            role = conv.get('from', 'user')
            content = conv.get('value', '')
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)

def collate_instruction_batch(batch):
    """Collate function for instruction dataset"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    original_texts = [item['original_text'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'original_texts': original_texts
    }

def download_instruction_datasets():
    """Download and prepare instruction-following datasets"""
    try:
        from datasets import load_dataset
        print("Downloading instruction-following datasets...")
    except ImportError:
        print("Please install required packages: pip install datasets")
        return None, None
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    all_data = []
    
    # Download datasets with error handling
    try:
        print("Loading ShareGPT dataset...")
        dataset = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train")
        all_data.extend(dataset[:10000])  # Use first 10k samples
        print(f"Loaded {len(dataset)} samples from ShareGPT")
    except Exception as e:
        print(f"Error loading ShareGPT dataset: {e}")
    
    try:
        print("Loading Alpaca dataset...")
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        all_data.extend(dataset[:10000])  # Use first 10k samples
        print(f"Loaded {len(dataset)} samples from Alpaca")
    except Exception as e:
        print(f"Error loading Alpaca dataset: {e}")
    
    if not all_data:
        print("No datasets were successfully loaded")
        return None, None
    
    # Split into train and validation
    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    print(f"Total samples: {len(all_data)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    return train_data, val_data

def create_data_loaders(train_data, val_data, tokenizer, args):
    """Create data loaders"""
    train_dataset = InstructionDataset(train_data, tokenizer, args.max_seq_len)
    val_dataset = InstructionDataset(val_data, tokenizer, args.max_seq_len)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_instruction_batch,
        num_workers=0,  # Disable multiprocessing to avoid deadlock
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_instruction_batch,
        num_workers=0,  # Disable multiprocessing to avoid deadlock
        pin_memory=False
    )
    
    return train_loader, val_loader

def load_data_from_files(train_file, val_file, tokenizer, args):
    """Load data from files"""
    train_data = []
    val_data = []
    
    # Load training data
    if train_file and os.path.exists(train_file):
        with open(train_file, 'r', encoding='utf-8') as f:
            if train_file.endswith('.json'):
                train_data = json.load(f)
            elif train_file.endswith('.jsonl'):
                train_data = [json.loads(line) for line in f]
    
    # Load validation data
    if val_file and os.path.exists(val_file):
        with open(val_file, 'r', encoding='utf-8') as f:
            if val_file.endswith('.json'):
                val_data = json.load(f)
            elif val_file.endswith('.jsonl'):
                val_data = [json.loads(line) for line in f]
    
    return train_data, val_data