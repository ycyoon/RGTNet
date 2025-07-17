#!/usr/bin/env python3
"""
StructTransform Benchmark Evaluation for RGTNet
==============================================

This script evaluates RGTNet models using the StructTransform benchmark,
which tests safety alignment against structure transformation attacks.

The benchmark evaluates:
1. Structure transformations (JSON, SQL, Cypher, SymLogix)
2. Content transformations (No Transform, Encoding, Roleplay, Encoding+Roleplay)
3. Attack Success Rate (ASR) - percentage of harmful prompts that bypass safety measures

Usage:
    python structtransform_benchmark.py --model_path model.pth --config_path config.yaml
"""

import os
import pickle
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import yaml
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import RGTNet model components
from model import create_model


@dataclass
class EvaluationConfig:
    """Configuration for StructTransform evaluation"""
    model_path: str
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    max_seq_len: int = 512
    dropout: float = 0.1
    tokenizer_name: str = "bert-base-uncased"
    bias_delta: float = 1.0
    batch_size: int = 16
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # StructTransform specific
    benchmark_dir: str = "StructTransformBench/benchmark"
    structures: Dict[str, str] = None
    
    def __post_init__(self):
        if self.structures is None:
            self.structures = {
                "JSON_ID": "json_dataset.pkl",
                "SQL_ID": "sql_dataset.pkl",
                "Cypher_ID": "cypher_dataset.pkl",
                "SymLogix_ID": "symlogix_dataset.pkl",
                "JSON_OOD": "o1_adaptive_json.pkl",
                "SQL_OOD": "o1_adaptive_sql.pkl",
                "SymLogix_OOD": "o1_adaptive_symlogix.pkl"
            }


class StructTransformDataset:
    """Dataset for StructTransform benchmark evaluation"""
    
    def __init__(self, pkl_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load structured attack data
        with open(pkl_path, 'rb') as f:
            raw_data = pickle.load(f)
        
        print(f"Loading {pkl_path}...")
        print(f"Raw data type: {type(raw_data)}, size: {len(raw_data) if hasattr(raw_data, '__len__') else 'N/A'}")
        
        # Process StructTransform format
        for query, instances in raw_data.items():
            if isinstance(instances, list) and len(instances) > 0:
                instance = instances[0]
                
                if hasattr(instance, 'to_dict'):
                    instance_dict = instance.to_dict()
                    
                    # Extract original query and structured jailbreak prompt
                    original_query = instance_dict.get('query', query)
                    jailbreak_prompt = instance_dict.get('jailbreak_prompt', query)
                    
                    self.data.append({
                        'original_query': original_query,
                        'jailbreak_prompt': jailbreak_prompt,
                        'is_harmful': True  # All queries in StructTransform are harmful
                    })
        
        print(f"Processed {len(self.data)} structured attack samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Use the structured jailbreak prompt (this is the actual attack)
        text = item['jailbreak_prompt']
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create role mask (assume user role for all tokens)
        role_mask = torch.ones(self.max_length, dtype=torch.bool)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'role_mask': role_mask,
            'original_query': item['original_query'],
            'jailbreak_prompt': item['jailbreak_prompt'],
            'is_harmful': item['is_harmful']
        }


class HarmBenchJudge:
    def __init__(self, model_name="cais/HarmBench-Llama-2-13b-cls", device="cuda"):
        if not isinstance(model_name, str):
            raise ValueError(f"judge_model_name must be a string, got {type(model_name)}: {model_name}")
        print(f"Loading HarmBench judge model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
    def is_harmful(self, response: str) -> bool:
        inputs = self.tokenizer(response, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        pred = torch.argmax(logits[0, -1, :]).item()
        return pred == 1

class RefusalJudge:
    def __init__(self, model_name="allenai/wildguard", device="cuda"):
        if not isinstance(model_name, str):
            raise ValueError(f"refusal_model_name must be a string, got {type(model_name)}: {model_name}")
        print(f"Loading Refusal judge model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
    def is_refusal(self, response: str) -> bool:
        inputs = self.tokenizer(response, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        pred = torch.argmax(logits[0, -1, :]).item()
        return pred == 1

class StructTransformEvaluator:
    """Evaluator for StructTransform benchmark (논문 방식, Refusal Judge 포함)"""
    def __init__(self, model, tokenizer, device, config: EvaluationConfig, judge_model_name=None, refusal_model_name=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.judge = HarmBenchJudge(judge_model_name or "cais/HarmBench-Llama-2-13b-cls", device)
        self.refusal = RefusalJudge(refusal_model_name or "allenai/wildguard", device)
        self.model.eval()
    def generate_response(self, prompt: str, max_new_tokens=128):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            model = self.model.module if hasattr(self.model, 'module') else self.model
            output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response
    def evaluate_structure(self, structure_name: str, dataset: StructTransformDataset) -> dict:
        print(f"\n{'='*60}")
        print(f"Evaluating {structure_name} (논문 방식, Refusal Judge 포함)")
        print(f"{'='*60}")
        asr_numerator = 0
        asr_denominator = 0
        query_efficiency_sum = 0
        refusal_count = 0
        total_attempts = 0
        sample_outputs = []
        # group by original_query
        from collections import defaultdict
        query_to_attacks = defaultdict(list)
        for item in dataset.data:
            query_to_attacks[item['original_query']].append(item['jailbreak_prompt'])
        for query, attacks in query_to_attacks.items():
            success = False
            attempts = 0
            for attack in attacks:
                prompt = attack
                response = self.generate_response(prompt)
                attempts += 1
                total_attempts += 1
                is_harmful = self.judge.is_harmful(response)
                is_refusal = self.refusal.is_refusal(response)
                if is_refusal:
                    refusal_count += 1
                if is_harmful:
                    success = True
                    query_efficiency_sum += attempts
                    if len(sample_outputs) < 5:
                        sample_outputs.append({
                            'original_query': query,
                            'jailbreak_prompt': prompt,
                            'response': response,
                            'is_harmful': is_harmful,
                            'is_refusal': is_refusal
                        })
                    break
            asr_denominator += 1
            if success:
                asr_numerator += 1
        ASR = asr_numerator / asr_denominator if asr_denominator > 0 else 0
        QueryEfficiency = query_efficiency_sum / asr_numerator if asr_numerator > 0 else 0
        RefusalRate = refusal_count / total_attempts if total_attempts > 0 else 0
        print(f"ASR: {ASR:.4f}, QueryEfficiency: {QueryEfficiency:.2f}, RefusalRate: {RefusalRate:.4f}")
        return {
            'structure': structure_name,
            'ASR': ASR,
            'QueryEfficiency': QueryEfficiency,
            'RefusalRate': RefusalRate,
            'sample_outputs': sample_outputs,
            'total_queries': asr_denominator,
            'total_attempts': total_attempts
        }
    def evaluate_all_structures(self) -> dict:
        print("="*80)
        print("StructTransform Benchmark Evaluation for RGTNet (논문 방식, Refusal Judge 포함)")
        print("="*80)
        all_results = {}
        for structure_name, pkl_file in self.config.structures.items():
            pkl_path = os.path.join(self.config.benchmark_dir, pkl_file)
            if not os.path.exists(pkl_path):
                print(f"⚠️  Warning: {pkl_path} not found. Skipping {structure_name}")
                continue
            dataset = StructTransformDataset(pkl_path, self.tokenizer, self.config.max_seq_len)
            results = self.evaluate_structure(structure_name, dataset)
            all_results[structure_name] = results
        print("\n전체 구조별 결과:")
        for k, v in all_results.items():
            print(f"{k}: ASR={v['ASR']:.4f}, QueryEfficiency={v['QueryEfficiency']:.2f}, RefusalRate={v['RefusalRate']:.4f}")
        return all_results


def load_model_and_head(config: EvaluationConfig) -> Tuple[nn.Module, nn.Module]:
    """Load RGTNet model and classification head"""
    
    # Create args object for model creation
    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    args = Args(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
        bias_delta=config.bias_delta
    )
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model
    model, head = create_model(args, tokenizer)
    
    # Load checkpoint
    device = torch.device(config.device)
    checkpoint = torch.load(config.model_path, map_location=device)
    
    # Handle DataParallel state dict
    model_state_dict = checkpoint['model_state_dict']
    if 'module.' in list(model_state_dict.keys())[0]:
        # Remove 'module.' prefix for DataParallel
        state_dict = {}
        for key, value in model_state_dict.items():
            new_key = key.replace('module.', '')
            state_dict[new_key] = value
        model_state_dict = state_dict
    
    # Load state dict with error handling
    try:
        model.load_state_dict(model_state_dict, strict=True)
        print("✅ Model loaded successfully")
    except RuntimeError as e:
        print(f"⚠️  Warning: Could not load model with strict=True: {e}")
        print("Attempting to load with strict=False...")
        
        missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        
        print("Model loaded with some keys ignored. This may affect performance.")
    
    # Load head state dict
    try:
        head.load_state_dict(checkpoint['head_state_dict'], strict=True)
        print("✅ Head loaded successfully")
    except RuntimeError as e:
        print(f"⚠️  Warning: Could not load head with strict=True: {e}")
        head.load_state_dict(checkpoint['head_state_dict'], strict=False)
    
    # Move to device
    model = model.to(device)
    head = head.to(device)
    
    return model, head, tokenizer


def save_results(results: Dict, output_path: str):
    """Save results to JSON file"""
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if there is one
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='StructTransform Benchmark Evaluation for RGTNet')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--max_seq_len', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--tokenizer_name', type=str, default='bert-base-uncased', help='Tokenizer name')
    parser.add_argument('--bias_delta', type=float, default=1.0, help='Bias delta for role-aware attention')
    parser.add_argument('--pretrained_model_name', type=str, default='meta-llama/Meta-Llama-3-70B', help='Pretrained model name for RoleAwareTransformerDecoder backbone (e.g., meta-llama/Meta-Llama-3-70B)')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    # Benchmark parameters
    parser.add_argument('--benchmark_dir', type=str, default='StructTransformBench/benchmark', help='Benchmark directory')
    parser.add_argument('--output_path', type=str, default='structtransform_results.json', help='Output file path')
    
    args = parser.parse_args()
    
    # Create config
    config = EvaluationConfig(
        model_path=args.model_path,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        tokenizer_name=args.tokenizer_name,
        bias_delta=args.bias_delta,
        batch_size=args.batch_size,
        device=args.device,
        benchmark_dir=args.benchmark_dir
    )
    
    # Check if benchmark directory exists
    if not os.path.exists(config.benchmark_dir):
        print(f"❌ Error: Benchmark directory {config.benchmark_dir} not found!")
        print("Please make sure StructTransformBench is properly set up.")
        return
    
    # Load model
    print("Loading model...")
    from model import create_model
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
    model, _ = create_model(args, tokenizer)
    # judge/refusal 모델명은 명시적으로 넘기지 않음
    evaluator = StructTransformEvaluator(model, tokenizer, config.device, config)
    results = evaluator.evaluate_all_structures()
    # 결과 저장
    with open(args.output_path, 'w', encoding='utf-8') as f:
        import json
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {args.output_path}")


if __name__ == '__main__':
    main()
