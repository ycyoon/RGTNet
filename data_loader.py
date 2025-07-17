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
        
        # Prepare input text and role information
        if isinstance(item, dict):
            if 'conversations' in item:
                text, role_info = self._format_conversation_with_roles(item['conversations'])
            elif 'instruction' in item:
                text, role_info = self._format_instruction_with_roles(item)
            else:
                text = str(item)
                role_info = None
        else:
            text = str(item)
            role_info = None
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create role mask for tokens
        role_mask = self._create_role_mask(text, role_info, encoding)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'role_mask': role_mask,
            'labels': torch.tensor(0, dtype=torch.long),  # Dummy label
            'original_text': text
        }
    
    def _format_conversation_with_roles(self, conversations):
        """Format conversation data with role information"""
        formatted = []
        role_segments = []
        
        for conv in conversations:
            role = conv.get('from', 'user')
            content = conv.get('value', '')
            
            # Map roles: human/user = instruction (True), assistant/gpt = data (False)
            is_instruction = role in ['human', 'user']
            
            formatted.append(f"{role}: {content}")
            role_segments.append((len("\n".join(formatted)), is_instruction))
        
        return "\n".join(formatted), role_segments
    
    def _format_instruction_with_roles(self, item):
        """Format instruction data with role information"""
        instruction = item.get('instruction', '')
        output = item.get('output', '')
        
        text = f"Instruction: {instruction}\nResponse: {output}"
        
        # Instruction part is True, response part is False
        instr_len = len(f"Instruction: {instruction}\n")
        role_segments = [(instr_len, True), (len(text), False)]
        
        return text, role_segments
    
    def _create_role_mask(self, text, role_info, encoding):
        """Create token-level role mask"""
        # Initialize all tokens as instruction tokens (True)
        role_mask = torch.ones(self.max_length, dtype=torch.bool)
        
        if role_info is not None:
            # Simple heuristic: split at response indicators
            response_start = -1
            text_lower = text.lower()
            
            for indicator in ['response:', 'assistant:', 'gpt:']:
                pos = text_lower.find(indicator)
                if pos != -1:
                    response_start = pos
                    break
            
            if response_start != -1:
                # Estimate token position where response starts
                pre_response = text[:response_start]
                post_response = text[response_start:]
                
                # Rough estimation: characters to tokens ratio
                char_to_token_ratio = len(encoding['input_ids'].squeeze()) / len(text) if len(text) > 0 else 0
                response_token_start = int(response_start * char_to_token_ratio)
                
                # Set tokens after response start as data tokens (False)
                if response_token_start < self.max_length:
                    role_mask[response_token_start:] = False
        
        return role_mask
    
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
    role_mask = torch.stack([item['role_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    original_texts = [item['original_text'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'role_mask': role_mask,
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
    
    # Try multiple datasets with error handling
    datasets_to_try = [
        {
            'name': 'Alpaca',
            'id': 'tatsu-lab/alpaca',
            'samples': 10000
        },
        {
            'name': 'OpenAssistant Conversations',
            'id': 'OpenAssistant/oasst1',
            'samples': 8000
        },
        {
            'name': 'Databricks Dolly',
            'id': 'databricks/databricks-dolly-15k',
            'samples': 7000
        },
        {
            'name': 'Stanford Human Preferences',
            'id': 'Anthropic/hh-rlhf',
            'samples': 5000
        },
        {
            'name': 'WizardLM Evol Instruct',
            'id': 'WizardLM/WizardLM_evol_instruct_70k',
            'samples': 5000
        }
    ]
    
    for dataset_info in datasets_to_try:
        try:
            print(f"Loading {dataset_info['name']} dataset...")
            dataset = load_dataset(dataset_info['id'], split="train")
            sample_count = min(len(dataset), dataset_info['samples'])
            all_data.extend(dataset[:sample_count])
            print(f"✅ Loaded {sample_count} samples from {dataset_info['name']}")
        except Exception as e:
            print(f"❌ Error loading {dataset_info['name']} dataset: {str(e)[:100]}...")
            continue
    
    # If we have some data but not enough, add more dummy data
    if len(all_data) < 1000:
        print(f"⚠️  Only {len(all_data)} samples loaded, adding dummy data for testing...")
        dummy_data = [
            {"instruction": "What is the capital of France?", "output": "The capital of France is Paris."},
            {"instruction": "Explain machine learning", "output": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."},
            {"instruction": "Write a simple Python function", "output": "def greet(name):\n    return f'Hello, {name}!'"},
            {"instruction": "What is 2 + 2?", "output": "2 + 2 equals 4."},
            {"instruction": "Tell me about the weather", "output": "I don't have access to real-time weather data, but I can help you understand weather patterns and concepts."},
            {"instruction": "How do you make tea?", "output": "To make tea: 1) Boil water, 2) Add tea leaves or tea bag, 3) Steep for 3-5 minutes, 4) Remove tea leaves, 5) Add milk/sugar if desired."},
            {"instruction": "What is Python?", "output": "Python is a high-level, interpreted programming language known for its simplicity and readability."},
            {"instruction": "Explain photosynthesis", "output": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen."},
            {"instruction": "What is the meaning of life?", "output": "The meaning of life is a philosophical question that has been debated for centuries, with different people finding different answers."},
            {"instruction": "How do computers work?", "output": "Computers work by processing binary data (0s and 1s) through electronic circuits, following programmed instructions to perform calculations and tasks."}
        ]
        # Add enough dummy data to reach at least 1000 samples
        needed_samples = max(1000 - len(all_data), 0)
        dummy_repeats = (needed_samples // len(dummy_data)) + 1
        all_data.extend(dummy_data * dummy_repeats)
        print(f"✅ Added {len(dummy_data * dummy_repeats)} dummy samples")
    
    # If still no data, create dummy data for testing
    if not all_data:
        print("❌ No datasets were successfully loaded, creating dummy data for testing...")
        dummy_data = [
            {"instruction": "What is the capital of France?", "output": "The capital of France is Paris."},
            {"instruction": "Explain machine learning", "output": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."},
            {"instruction": "Write a simple Python function", "output": "def greet(name):\n    return f'Hello, {name}!'"},
            {"instruction": "What is 2 + 2?", "output": "2 + 2 equals 4."},
            {"instruction": "Tell me about the weather", "output": "I don't have access to real-time weather data, but I can help you understand weather patterns and concepts."}
        ]
        # Repeat dummy data to have enough samples
        all_data = dummy_data * 200  # 1000 samples total
    
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
    
    # Adjust batch size for DataParallel
    # DataParallel splits the batch across available GPUs
    effective_batch_size = args.batch_size
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    if gpu_count > 1:
        print(f"DataParallel detected: Using {gpu_count} GPUs")
        print(f"Effective batch size per GPU: {effective_batch_size}")
        print(f"Total effective batch size: {effective_batch_size * gpu_count}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        collate_fn=collate_instruction_batch,
        num_workers=0,  # Disable multiprocessing to avoid deadlock
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=effective_batch_size,
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