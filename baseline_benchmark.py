# OFFLINE_MODE_PATCH_APPLIED
# Hugging Face 오프라인 모드 설정
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/ceph_data/ycyoon/.cache/huggingface"


#!/usr/bin/env python3
"""
Baseline Foundation Model Benchmark Evaluation
==============================================

This script evaluates baseline foundation models (DialoGPT-medium, Meta-Llama-3-8B, etc.)
using the StructTransform benchmark to compare their safety performance against RGTNet.

Usage:
    python baseline_benchmark.py --model_name microsoft/DialoGPT-medium
    python baseline_benchmark.py --model_name meta-llama/Meta-Llama-3-8B --device cuda:0
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
import time
import logging
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from collections import defaultdict

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠️ PEFT not available, LoRA support disabled")

# Import judge models from the existing benchmark
from structtransform_benchmark import HarmBenchJudge, RefusalJudge, StructTransformDataset


def setup_detailed_logging(model_name: str, log_dir: str = "benchmark_logs") -> tuple:
    """Setup detailed logging for responses and judge evaluations"""
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Clean model name for filename
    clean_model_name = model_name.replace("/", "_").replace(":", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup main logger
    main_log_file = os.path.join(log_dir, f"{clean_model_name}_{timestamp}_main.log")
    main_logger = logging.getLogger(f"benchmark_main_{clean_model_name}")
    main_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    main_logger.handlers.clear()
    
    # File handler for main log
    main_handler = logging.FileHandler(main_log_file, encoding='utf-8')
    main_handler.setLevel(logging.INFO)
    main_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main_handler.setFormatter(main_formatter)
    main_logger.addHandler(main_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    main_logger.addHandler(console_handler)
    
    # Setup detailed response logger (for responses and judge evaluations)
    detail_log_file = os.path.join(log_dir, f"{clean_model_name}_{timestamp}_detailed.jsonl")
    detail_logger = logging.getLogger(f"benchmark_detail_{clean_model_name}")
    detail_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    detail_logger.handlers.clear()
    
    # File handler for detailed log (JSON format)
    detail_handler = logging.FileHandler(detail_log_file, encoding='utf-8')
    detail_handler.setLevel(logging.INFO)
    detail_formatter = logging.Formatter('%(message)s')
    detail_handler.setFormatter(detail_formatter)
    detail_logger.addHandler(detail_handler)
    
    return main_logger, detail_logger, main_log_file, detail_log_file


@dataclass
class BaselineBenchmarkConfig:
    """Configuration for baseline foundation model evaluation"""
    model_name: str  # e.g., "microsoft/DialoGPT-medium", "meta-llama/Meta-Llama-3-8B"
    max_seq_len: int = 512
    max_new_tokens: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 1  # Usually 1 for foundation models to avoid OOM
    
    # LoRA adapter settings
    lora_adapter_path: Optional[str] = None  # Path to LoRA adapters
    use_lora: bool = False  # Whether to use LoRA adapters
    
    # StructTransform benchmark settings
    benchmark_dir: str = "StructTransformBench/benchmark"
    structures: Dict[str, str] = None
    
    # Judge model settings
    judge_model_name: str = "cais/HarmBench-Llama-2-13b-cls"
    refusal_model_name: str = "allenai/wildguard"
    judge_device: str = "cpu"  # Use CPU for judge models to save GPU memory
    
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


class BaselineFoundationModelEvaluator:
    """Evaluator for baseline foundation models on StructTransform benchmark"""
    
    def __init__(self, config: BaselineBenchmarkConfig):
        self.config = config
        print(f"🔧 Initializing baseline evaluator for {config.model_name}")
        
        # Setup detailed logging
        self.main_logger, self.detail_logger, self.main_log_file, self.detail_log_file = setup_detailed_logging(config.model_name)
        self.main_logger.info(f"Starting benchmark evaluation for {config.model_name}")
        self.main_logger.info(f"Detailed logs will be saved to: {self.detail_log_file}")
        
        # Load foundation model and tokenizer
        self._load_foundation_model()
        
        # Initialize judge models
        print(f"🔧 Loading judge models on {config.judge_device}...")
        self.main_logger.info(f"Loading judge models on {config.judge_device}")
        self.judge = HarmBenchJudge(config.judge_model_name, config.judge_device)
        self.refusal = RefusalJudge(config.refusal_model_name, config.judge_device)
        
        print("✅ Baseline evaluator initialized successfully!")
        self.main_logger.info("Baseline evaluator initialized successfully")
    
    def _load_foundation_model(self):
        """Load the baseline foundation model"""
        print(f"🔧 Loading foundation model: {self.config.model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate settings
            model_kwargs = {
                "torch_dtype": torch.float16 if self.config.device.startswith("cuda") else torch.float32,
                "device_map": "auto" if self.config.device.startswith("cuda") else None,
                "low_cpu_mem_usage": True,
            }
            
            # For large models, use additional memory optimizations
            if "llama" in self.config.model_name.lower() or "gpt" in self.config.model_name.lower():
                model_kwargs.update({
                    "load_in_8bit": False,  # Set to True if you want 8-bit quantization
                    "trust_remote_code": True
                })
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name, 
                **model_kwargs
            )
            
            # Load LoRA adapters if specified
            if self.config.use_lora and self.config.lora_adapter_path and PEFT_AVAILABLE:
                print(f"🎯 Loading LoRA adapters from: {self.config.lora_adapter_path}")
                try:
                    self.model = PeftModel.from_pretrained(base_model, self.config.lora_adapter_path)
                    # Merge adapters for faster inference
                    print("🔧 Merging adapters for faster inference...")
                    self.model = self.model.merge_and_unload()
                    print("✅ LoRA adapters loaded and merged successfully!")
                except Exception as e:
                    print(f"⚠️ Failed to load LoRA adapters: {e}")
                    print("🔄 Using base model without adapters")
                    self.model = base_model
            else:
                self.model = base_model
            
            # Move to device if device_map is not used
            if model_kwargs["device_map"] is None:
                self.model = self.model.to(self.config.device)
            
            self.model.eval()
            
            # Setup generation config
            self.generation_config = GenerationConfig(
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
            
            print(f"✅ Foundation model loaded successfully!")
            print(f"   Model dtype: {self.model.dtype}")
            print(f"   Device: {next(self.model.parameters()).device}")
            
        except Exception as e:
            print(f"❌ Error loading foundation model: {e}")
            raise
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using the foundation model"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=self.config.max_seq_len
            )
            
            # Move inputs to device
            if hasattr(self.model, 'device'):
                device = self.model.device
            else:
                device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode response (remove input tokens)
            input_length = inputs['input_ids'].shape[1]
            response_ids = output_ids[0][input_length:]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            print(f"⚠️ Error generating response: {e}")
            return "Error: Could not generate response"
    
    def evaluate_structure(self, structure_name: str, dataset: StructTransformDataset) -> dict:
        """Evaluate the foundation model on a specific structure"""
        print(f"\n{'='*60}")
        print(f"📊 Evaluating {self.config.model_name} on {structure_name}")
        print(f"{'='*60}")
        
        self.main_logger.info(f"Starting evaluation on {structure_name}")
        self.main_logger.info(f"Dataset size: {len(dataset.data)} items")
        
        asr_numerator = 0  # Successful attacks
        asr_denominator = 0  # Total queries
        query_efficiency_sum = 0  # Sum of attempts per successful attack
        refusal_count = 0  # Number of refusals
        total_attempts = 0  # Total attempts
        judge_elapsed = 0.0  # Time spent on judge evaluation
        refusal_elapsed = 0.0  # Time spent on refusal evaluation
        sample_outputs = []  # Sample outputs for analysis
        
        # Group attacks by original query
        query_to_attacks = defaultdict(list)
        for item in dataset.data:
            query_to_attacks[item['original_query']].append(item['jailbreak_prompt'])
        
        self.main_logger.info(f"Grouped into {len(query_to_attacks)} unique queries")
        
        # Evaluate each query
        for query_idx, (query, attacks) in enumerate(tqdm(
            query_to_attacks.items(), 
            desc=f"Evaluating {structure_name}", 
            leave=False
        )):
            success = False
            attempts = 0
            query_start_time = time.perf_counter()
            
            # Log query start
            self.detail_logger.info(json.dumps({
                "event": "query_start",
                "structure": structure_name,
                "query_idx": query_idx,
                "original_query": query,
                "attack_variants": len(attacks),
                "timestamp": datetime.now().isoformat()
            }, ensure_ascii=False))
            
            # Try each attack variant for this query
            for attack_idx, attack in enumerate(attacks):
                prompt = attack
                attempt_start_time = time.perf_counter()
                
                # Generate response using foundation model
                response = self.generate_response(prompt)
                attempts += 1
                total_attempts += 1
                response_time = time.perf_counter() - attempt_start_time
                
                # Evaluate response with judge models
                start_t = time.perf_counter()
                is_harmful = self.judge.is_harmful(response)
                judge_time = time.perf_counter() - start_t
                judge_elapsed += judge_time
                
                start_t = time.perf_counter()
                is_refusal = self.refusal.is_refusal(response)
                refusal_time = time.perf_counter() - start_t
                refusal_elapsed += refusal_time
                
                # Log detailed evaluation result
                self.detail_logger.info(json.dumps({
                    "event": "response_evaluation",
                    "structure": structure_name,
                    "query_idx": query_idx,
                    "attack_idx": attack_idx,
                    "attempt": attempts,
                    "original_query": query,
                    "jailbreak_prompt": prompt,
                    "model_response": response,
                    "response_time_sec": response_time,
                    "judge_evaluation": {
                        "is_harmful": is_harmful,
                        "judge_time_sec": judge_time,
                        "judge_model": self.config.judge_model_name
                    },
                    "refusal_evaluation": {
                        "is_refusal": is_refusal,
                        "refusal_time_sec": refusal_time,
                        "refusal_model": self.config.refusal_model_name
                    },
                    "timestamp": datetime.now().isoformat()
                }, ensure_ascii=False))
                
                if is_refusal:
                    refusal_count += 1
                
                # If harmful response generated, attack succeeded
                if is_harmful:
                    success = True
                    query_efficiency_sum += attempts
                    
                    # Log successful attack
                    self.detail_logger.info(json.dumps({
                        "event": "attack_success",
                        "structure": structure_name,
                        "query_idx": query_idx,
                        "attack_idx": attack_idx,
                        "attempts_to_success": attempts,
                        "original_query": query,
                        "successful_attack": prompt,
                        "harmful_response": response,
                        "timestamp": datetime.now().isoformat()
                    }, ensure_ascii=False))
                    
                    # Store sample for analysis
                    if len(sample_outputs) < 5:
                        sample_outputs.append({
                            'original_query': query,
                            'jailbreak_prompt': prompt,
                            'response': response[:200] + "..." if len(response) > 200 else response,
                            'is_harmful': is_harmful,
                            'is_refusal': is_refusal
                        })
                    break  # Stop trying other attacks for this query
            
            query_total_time = time.perf_counter() - query_start_time
            
            # Log query completion
            self.detail_logger.info(json.dumps({
                "event": "query_complete",
                "structure": structure_name,
                "query_idx": query_idx,
                "original_query": query,
                "success": success,
                "attempts": attempts,
                "query_total_time_sec": query_total_time,
                "timestamp": datetime.now().isoformat()
            }, ensure_ascii=False))
            
            asr_denominator += 1
            if success:
                asr_numerator += 1
        
        # Calculate metrics
        ASR = asr_numerator / asr_denominator if asr_denominator > 0 else 0
        QueryEfficiency = query_efficiency_sum / asr_numerator if asr_numerator > 0 else 0
        RefusalRate = refusal_count / total_attempts if total_attempts > 0 else 0
        
        avg_judge_latency = judge_elapsed / total_attempts if total_attempts > 0 else 0
        avg_refusal_latency = refusal_elapsed / total_attempts if total_attempts > 0 else 0
        
        # Log structure evaluation completion
        self.main_logger.info(f"Completed evaluation for {structure_name}")
        self.main_logger.info(f"ASR: {ASR:.4f} ({asr_numerator}/{asr_denominator})")
        self.main_logger.info(f"Query Efficiency: {QueryEfficiency:.2f}")
        self.main_logger.info(f"Refusal Rate: {RefusalRate:.4f} ({refusal_count}/{total_attempts})")
        self.main_logger.info(f"Avg Judge Latency: {avg_judge_latency:.3f}s")
        self.main_logger.info(f"Avg Refusal Latency: {avg_refusal_latency:.3f}s")
        
        # Log structure results in detail log
        self.detail_logger.info(json.dumps({
            "event": "structure_complete",
            "structure": structure_name,
            "results": {
                "ASR": ASR,
                "QueryEfficiency": QueryEfficiency,
                "RefusalRate": RefusalRate,
                "successful_attacks": asr_numerator,
                "total_queries": asr_denominator,
                "total_attempts": total_attempts,
                "refusal_count": refusal_count,
                "avg_judge_latency_sec": avg_judge_latency,
                "avg_refusal_latency_sec": avg_refusal_latency,
                "total_judge_time_sec": judge_elapsed,
                "total_refusal_time_sec": refusal_elapsed
            },
            "timestamp": datetime.now().isoformat()
        }, ensure_ascii=False))
        
        print(f"📈 Results for {structure_name}:")
        print(f"   ASR: {ASR:.4f} ({asr_numerator}/{asr_denominator})")
        print(f"   Query Efficiency: {QueryEfficiency:.2f}")
        print(f"   Refusal Rate: {RefusalRate:.4f} ({refusal_count}/{total_attempts})")
        print(f"   Avg Judge Latency: {avg_judge_latency:.3f}s")
        print(f"   Avg Refusal Latency: {avg_refusal_latency:.3f}s")
        
        return {
            'model_name': self.config.model_name,
            'structure': structure_name,
            'ASR': ASR,
            'QueryEfficiency': QueryEfficiency,
            'RefusalRate': RefusalRate,
            'successful_attacks': asr_numerator,
            'total_queries': asr_denominator,
            'total_attempts': total_attempts,
            'refusal_count': refusal_count,
            'avg_judge_latency_sec': avg_judge_latency,
            'avg_refusal_latency_sec': avg_refusal_latency,
            'sample_outputs': sample_outputs
        }
    
    def evaluate_all_structures(self) -> dict:
        """Evaluate the foundation model on all StructTransform structures"""
        print("="*80)
        print(f"🚀 StructTransform Benchmark Evaluation for {self.config.model_name}")
        print("="*80)
        
        all_results = {}
        
        for structure_name, pkl_file in self.config.structures.items():
            pkl_path = os.path.join(self.config.benchmark_dir, pkl_file)
            
            if not os.path.exists(pkl_path):
                print(f"⚠️  Warning: {pkl_path} not found. Skipping {structure_name}")
                continue
            
            # Load dataset
            dataset = StructTransformDataset(pkl_path, self.tokenizer, self.config.max_seq_len)
            
            # Evaluate structure
            results = self.evaluate_structure(structure_name, dataset)
            all_results[structure_name] = results
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"📊 SUMMARY: {self.config.model_name} Benchmark Results")
        print(f"{'='*80}")
        
        self.main_logger.info("="*60)
        self.main_logger.info(f"FINAL SUMMARY: {self.config.model_name}")
        self.main_logger.info("="*60)
        
        total_asr = 0
        total_refusal_rate = 0
        count = 0
        
        for structure_name, results in all_results.items():
            asr = results['ASR']
            refusal_rate = results['RefusalRate']
            print(f"{structure_name:15s}: ASR={asr:.4f}, RefusalRate={refusal_rate:.4f}")
            self.main_logger.info(f"{structure_name:15s}: ASR={asr:.4f}, RefusalRate={refusal_rate:.4f}")
            total_asr += asr
            total_refusal_rate += refusal_rate
            count += 1
        
        if count > 0:
            avg_asr = total_asr / count
            avg_refusal_rate = total_refusal_rate / count
            print(f"{'AVERAGE':15s}: ASR={avg_asr:.4f}, RefusalRate={avg_refusal_rate:.4f}")
            self.main_logger.info(f"{'AVERAGE':15s}: ASR={avg_asr:.4f}, RefusalRate={avg_refusal_rate:.4f}")
            
            # Log final summary in detail log
            self.detail_logger.info(json.dumps({
                "event": "evaluation_complete",
                "model_name": self.config.model_name,
                "summary": {
                    "average_ASR": avg_asr,
                    "average_RefusalRate": avg_refusal_rate,
                    "structures_evaluated": count,
                    "structure_results": {name: {"ASR": r["ASR"], "RefusalRate": r["RefusalRate"]} 
                                        for name, r in all_results.items()},
                    "evaluation_timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
                },
                "log_files": {
                    "main_log": self.main_log_file,
                    "detailed_log": self.detail_log_file
                },
                "timestamp": datetime.now().isoformat()
            }, ensure_ascii=False))
            
            # Add overall statistics
            all_results['_overall_statistics'] = {
                'model_name': self.config.model_name,
                'average_ASR': avg_asr,
                'average_RefusalRate': avg_refusal_rate,
                'structures_evaluated': count,
                'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'log_files': {
                    'main_log': self.main_log_file,
                    'detailed_log': self.detail_log_file
                }
            }
        
        self.main_logger.info("Evaluation completed successfully")
        self.main_logger.info(f"Detailed logs saved to: {self.detail_log_file}")
        print(f"\n📁 Detailed logs saved to: {self.detail_log_file}")
        
        return all_results
    
    def cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up resources...")
        
        # Clean up judge models
        if hasattr(self, 'judge'):
            del self.judge
        if hasattr(self, 'refusal'):
            del self.refusal
        
        # Clean up foundation model
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✅ Cleanup complete!")


def save_results(results: Dict, output_path: str, model_name: str):
    """Save benchmark results to JSON file"""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Add metadata
    results_with_metadata = {
        'metadata': {
            'model_name': model_name,
            'evaluation_type': 'baseline_foundation_model',
            'benchmark': 'StructTransform',
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_structures': len([k for k in results.keys() if not k.startswith('_')])
        },
        'results': results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_with_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Baseline Foundation Model Benchmark Evaluation')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, required=True, 
                       help='Foundation model name (e.g., microsoft/DialoGPT-medium, meta-llama/Meta-Llama-3-8B)')
    parser.add_argument('--max_seq_len', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--max_new_tokens', type=int, default=128, help='Maximum new tokens to generate')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (usually 1 for foundation models)')
    
    # LoRA parameters
    parser.add_argument('--lora_adapter_path', type=str, default=None, 
                       help='Path to LoRA adapter directory (e.g., ./models/lora_adapters_epoch_2)')
    parser.add_argument('--use_lora', action='store_true', 
                       help='Use LoRA adapters if lora_adapter_path is provided')
    
    # Benchmark parameters
    parser.add_argument('--benchmark_dir', type=str, default='StructTransformBench/benchmark', 
                       help='StructTransform benchmark directory')
    parser.add_argument('--output_dir', type=str, default='baseline_results', 
                       help='Output directory for results')
    
    # Judge model parameters
    parser.add_argument('--judge_model_name', type=str, default='cais/HarmBench-Llama-2-13b-cls',
                       help='Judge model for harmfulness detection')
    parser.add_argument('--refusal_model_name', type=str, default='allenai/wildguard',
                       help='Judge model for refusal detection')
    parser.add_argument('--judge_device', type=str, default='cpu',
                       help='Device for judge models (cpu recommended to save GPU memory)')
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device == 'auto':
        if torch.cuda.is_available():
            args.device = 'cuda'
        else:
            args.device = 'cpu'
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filename
    model_name_safe = args.model_name.replace('/', '_').replace('-', '_')
    output_path = os.path.join(args.output_dir, f'{model_name_safe}_benchmark_results.json')
    
    # Create config
    config = BaselineBenchmarkConfig(
        model_name=args.model_name,
        max_seq_len=args.max_seq_len,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        batch_size=args.batch_size,
        lora_adapter_path=args.lora_adapter_path,
        use_lora=args.use_lora,
        benchmark_dir=args.benchmark_dir,
        judge_model_name=args.judge_model_name,
        refusal_model_name=args.refusal_model_name,
        judge_device=args.judge_device
    )
    
    # Check if benchmark directory exists
    if not os.path.exists(config.benchmark_dir):
        print(f"❌ Error: Benchmark directory {config.benchmark_dir} not found!")
        print("Please run ./setup_benchmark.sh first to set up the StructTransform benchmark.")
        return
    
    # Initialize evaluator
    evaluator = None
    try:
        evaluator = BaselineFoundationModelEvaluator(config)
        
        # Run evaluation
        results = evaluator.evaluate_all_structures()
        
        # Save results
        save_results(results, output_path, args.model_name)
        
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"📄 Results saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        raise
    
    finally:
        # Cleanup
        if evaluator:
            evaluator.cleanup()


if __name__ == '__main__':
    main()
