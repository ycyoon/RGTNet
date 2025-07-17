import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import json
import os
from pathlib import Path

class GenerationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512, num_roles=2):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_roles = num_roles
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item.get('instruction', '')
        output = item.get('output', '')
        # 프롬프트와 정답을 이어붙여 causal LM 학습
        full_text = prompt + self.tokenizer.eos_token + output + self.tokenizer.eos_token
        encoding = self.tokenizer(full_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        # input_ids가 vocab_size 이상인 경우 unk_token_id로 대체
        unk_token_id = self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else 0
        input_ids[input_ids >= self.tokenizer.vocab_size] = unk_token_id
        # assert input_ids.max().item() < self.tokenizer.vocab_size, f"input_ids max {input_ids.max().item()} >= vocab_size {self.tokenizer.vocab_size}"
        labels = input_ids.clone()
        # 역할 정보: instruction 부분은 0(user), output 부분은 1(assistant)로 마킹
        # 간단하게 prompt 길이까지 0, 그 이후 1로 설정
        prompt_encoding = self.tokenizer(prompt + self.tokenizer.eos_token, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        prompt_len = (prompt_encoding['attention_mask'].squeeze() == 1).sum().item()
        role_mask = torch.zeros(self.max_length, dtype=torch.long)
        if prompt_len < self.max_length:
            role_mask[prompt_len:] = 1
        # 안전하게 0~num_roles-1로 clamp
        role_mask = torch.clamp(role_mask, 0, self.num_roles-1)
        return {'input_ids': input_ids, 'labels': labels, 'role_mask': role_mask}

def collate_generation_batch(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    role_mask = torch.stack([item['role_mask'] for item in batch])
    return {'input_ids': input_ids, 'labels': labels, 'role_mask': role_mask}

def pad_to_multiple(data, multiple):
    n = len(data)
    if n % multiple != 0:
        extra = multiple - (n % multiple)
        # deepcopy 대신 shallow copy로 충분 (dict)
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
            # all_data.extend(dataset[:sample_count])
            if hasattr(dataset, 'keys') and 'train' in dataset:
                # DatasetDict인 경우 train split만 사용
                split_data = dataset['train']
            else:
                split_data = dataset
            # DatasetDict, IterableDatasetDict, IterableColumn 등은 list로 변환
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
    # DDP 환경에서 world_size(8)의 배수로 맞추기
    world_size = 8
    train_data = pad_to_multiple(train_data, world_size)
    val_data = pad_to_multiple(val_data, world_size)
    
    print(f"Total samples: {len(all_data)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    return train_data, val_data

def create_data_loaders(train_dataset, val_dataset, tokenizer, args, train_sampler=None, val_sampler=None):
    effective_batch_size = args.batch_size
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    num_workers = getattr(args, 'num_workers', 8)
    if gpu_count > 1:
        print(f"DataParallel detected: Using {gpu_count} GPUs")
        print(f"Effective batch size per GPU: {effective_batch_size}")
        print(f"Total effective batch size: {effective_batch_size * gpu_count}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_generation_batch,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_generation_batch,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
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