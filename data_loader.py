import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import json
import os
from pathlib import Path

class InstructionDataset(Dataset):
    """
    Dataset for instruction-following fine-tuning using Llama-3 chat template.
    Optimized for PEFT/LoRA training - no role masking needed.
    """
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab_size = len(tokenizer)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        response = item.get('response', item.get('output', ''))  # Support both field names
        
        # Create conversation using Llama-3 chat template
        if input_text:
            user_message = f"{instruction}\n\n{input_text}"
        else:
            user_message = instruction
            
        # Format using chat template
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response}
        ]
        
        # Apply chat template (this handles special tokens automatically)
        full_text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        # Tokenize the full conversation
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Validate tokens to prevent CUDA errors
        input_ids = torch.clamp(input_ids, 0, self.vocab_size - 1)
        
        # Create labels (copy of input_ids)
        labels = input_ids.clone()
        
        # Mask the user prompt part (only train on assistant response)
        user_only_messages = [{"role": "user", "content": user_message}]
        user_text = self.tokenizer.apply_chat_template(
            user_only_messages,
            tokenize=False,
            add_generation_prompt=True  # This adds the assistant prompt
        )
        
        user_encoding = self.tokenizer(
            user_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Find the length of the user prompt to mask it
        user_prompt_len = (user_encoding['attention_mask'].squeeze() == 1).sum().item()
        
        # Mask prompt tokens in labels (set to -100 to ignore in loss)
        if user_prompt_len < self.max_length:
            labels[:user_prompt_len] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class GenerationDataset(Dataset):
    """
    Simplified dataset for causal language modeling with PEFT/LoRA.
    Uses Llama-3 chat template format.
    """
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab_size = len(tokenizer)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        response = item.get('response', item.get('output', ''))
        
        # Create conversation using Llama-3 chat template
        if input_text:
            user_message = f"{instruction}\n\n{input_text}"
        else:
            user_message = instruction
            
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response}
        ]
        
        # Apply chat template
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        
        # Validate tokens
        input_ids = torch.clamp(input_ids, 0, self.vocab_size - 1)
        
        # Create labels
        labels = input_ids.clone()
        
        # Calculate user prompt length for masking
        user_only_messages = [{"role": "user", "content": user_message}]
        user_text = self.tokenizer.apply_chat_template(
            user_only_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        user_encoding = self.tokenizer(
            user_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        user_prompt_len = (user_encoding['attention_mask'].squeeze() == 1).sum().item()
        
        # Mask prompt tokens in labels
        if user_prompt_len < self.max_length:
            labels[:user_prompt_len] = -100
        
        # Create role mask: user=0, assistant=1
        role_mask = torch.zeros(self.max_length, dtype=torch.long)
        if user_prompt_len < self.max_length:
            role_mask[user_prompt_len:] = 1
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'role_mask': role_mask
        }

def collate_generation_batch(batch):
    """Collate function for generation dataset with role_mask"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    role_mask = torch.stack([item['role_mask'] for item in batch])
    return {'input_ids': input_ids, 'labels': labels, 'role_mask': role_mask}

def collate_instruction_batch(batch):
    """Collate function for instruction dataset"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    if 'attention_mask' in batch[0]:
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
    else:
        return {'input_ids': input_ids, 'labels': labels}

def pad_to_multiple(data, multiple):
    n = len(data)
    if n % multiple != 0:
        extra = multiple - (n % multiple)
        data += [data[0].copy() for _ in range(extra)]
    return data

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
            dataset = dataset.select(range(sample_count))
            
            if hasattr(dataset, 'keys') and 'train' in dataset:
                split_data = dataset['train']
            else:
                split_data = dataset
            
            try:
                split_data_list = list(split_data)
                n_samples = len(split_data_list)
                all_data.extend(split_data_list)
                print(f"✅ Loaded {n_samples} samples from {dataset_info['name']}")
            except Exception as e:
                print(f"❌ Could not process {dataset_info['name']} samples: {e}")
                continue
        except Exception as e:
            print(f"❌ Error loading {dataset_info['name']} dataset: {str(e)[:100]}...")
            continue
    
    # Add dummy data if needed
    if len(all_data) < 1000:
        print(f"⚠️  Only {len(all_data)} samples loaded, adding dummy data for testing...")
        dummy_data = [
            {"instruction": "What is the capital of France?", "response": "The capital of France is Paris."},
            {"instruction": "Explain machine learning", "response": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."},
            {"instruction": "Write a simple Python function", "response": "def greet(name):\n    return f'Hello, {name}!'"},
            {"instruction": "What is 2 + 2?", "response": "2 + 2 equals 4."},
            {"instruction": "Tell me about the weather", "response": "I don't have access to real-time weather data, but I can help you understand weather patterns and concepts."},
            {"instruction": "How do you make tea?", "response": "To make tea: 1) Boil water, 2) Add tea leaves or tea bag, 3) Steep for 3-5 minutes, 4) Remove tea leaves, 5) Add milk/sugar if desired."},
            {"instruction": "What is Python?", "response": "Python is a high-level, interpreted programming language known for its simplicity and readability."},
            {"instruction": "Explain photosynthesis", "response": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen."},
            {"instruction": "What is the meaning of life?", "response": "The meaning of life is a philosophical question that has been debated for centuries, with different people finding different answers."},
            {"instruction": "How do computers work?", "response": "Computers work by processing binary data (0s and 1s) through electronic circuits, following programmed instructions to perform calculations and tasks."}
        ]
        
        needed_samples = max(1000 - len(all_data), 0)
        dummy_repeats = (needed_samples // len(dummy_data)) + 1
        all_data.extend(dummy_data * dummy_repeats)
        print(f"✅ Added {len(dummy_data * dummy_repeats)} dummy samples")
    
    # Create dummy data if still no data
    if not all_data:
        print("❌ No datasets were successfully loaded, creating dummy data for testing...")
        dummy_data = [
            {"instruction": "What is the capital of France?", "response": "The capital of France is Paris."},
            {"instruction": "Explain machine learning", "response": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."},
            {"instruction": "Write a simple Python function", "response": "def greet(name):\n    return f'Hello, {name}!'"},
            {"instruction": "What is 2 + 2?", "response": "2 + 2 equals 4."},
            {"instruction": "Tell me about the weather", "response": "I don't have access to real-time weather data, but I can help you understand weather patterns and concepts."}
        ]
        all_data = dummy_data * 200  # 1000 samples total
    
    # Split into train and validation
    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    # Pad to multiple of world_size for DDP
    world_size = 8
    train_data = pad_to_multiple(train_data, world_size)
    val_data = pad_to_multiple(val_data, world_size)
    
    print(f"Total samples: {len(all_data)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    return train_data, val_data

def create_data_loaders(train_dataset, val_dataset, tokenizer, args, train_sampler=None, val_sampler=None):
    """Create data loaders for PEFT training"""
    
    # DeepSpeed 설정 파일에서 배치 크기 정보 읽기
    if hasattr(args, 'deepspeed_config') and args.deepspeed_config:
        import json
        import torch.distributed as dist
        
        try:
            with open(args.deepspeed_config, 'r') as f:
                ds_config = json.load(f)
            
            train_batch_size = ds_config.get('train_batch_size', 8)
            gradient_accumulation_steps = ds_config.get('gradient_accumulation_steps', 4)
            
            # world_size 계산 (분산 학습 시)
            if dist.is_initialized():
                world_size = dist.get_world_size()
            else:
                world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
            
            # micro_batch_size 계산
            effective_batch_size = train_batch_size // (gradient_accumulation_steps * world_size)
            
            print(f"DeepSpeed batch configuration:")
            print(f"  - train_batch_size: {train_batch_size}")
            print(f"  - gradient_accumulation_steps: {gradient_accumulation_steps}")
            print(f"  - world_size: {world_size}")
            print(f"  - calculated micro_batch_size: {effective_batch_size}")
            
        except Exception as e:
            print(f"Failed to read DeepSpeed config: {e}")
            effective_batch_size = 1  # 폴백
    else:
        effective_batch_size = 1  # DeepSpeed 없을 때 기본값
    
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    num_workers = getattr(args, 'num_workers', 8)
    
    if gpu_count > 1:
        print(f"DataParallel detected: Using {gpu_count} GPUs")
        print(f"DataLoader batch_size=1, actual batch size controlled by DeepSpeed config")
    
    # Use RGTNet collate function with role_mask
    collate_fn = collate_generation_batch
    print("Using RGTNet GenerationDataset with role-aware batch collation")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return train_loader, val_loader

def load_data_from_files(train_file, val_file):
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