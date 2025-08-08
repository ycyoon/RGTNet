#!/usr/bin/env python3
"""
Safe LLM Performance Evaluation Script
Prevents NCCL errors and runs safely on single GPU
"""

import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import pandas as pd

# Set environment variables to prevent NCCL issues
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_TIMEOUT'] = '1800'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Evaluation metrics
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.tokenize import word_tokenize
import nltk

# Model and tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig

# Custom imports
from model import RGTNetModel

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SafeLLMEvaluator:
    """Safe LLM performance evaluator that prevents NCCL issues"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        # Force single GPU usage and set device
        if device == "cuda":
            # Use only the first available GPU
            self.device = "cuda:0"
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        else:
            self.device = device
            
        self.model_path = model_path
        
        # Disable distributed training
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        # Initialize tokenizer from the same model path
        logger.info(f"Loading tokenizer from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize scorers
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method4
        
        logger.info(f"Safe evaluator initialized with model: {model_path}")
        
    def _load_model(self):
        """Load the trained model - supports both RGTNet and standard foundation models"""
        # Check if this is a RGTNet model by looking for rgtnet_model_info.json
        rgtnet_info_path = os.path.join(self.model_path, "rgtnet_model_info.json")
        
        if os.path.exists(rgtnet_info_path):
            logger.info("Detected RGTNet model, loading with weight mapping fix...")
            try:
                # Load RGTNet model with weight mapping fix
                model = self._load_rgtnet_with_mapping_fix()
                logger.info("Loaded RGTNet model successfully")
                return model
            except Exception as e:
                logger.warning(f"Failed to load as RGTNet model: {e}")
                logger.info("Falling back to standard LlamaForCausalLM...")
                try:
                    # Load as standard transformer model
                    model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float16,
                        device_map=self.device,
                        trust_remote_code=True
                    )
                    logger.info("Loaded standard foundation model successfully")
                    return model
                except Exception as e2:
                    logger.error(f"Failed to load model: {e2}")
                    raise
        else:
            logger.info("No RGTNet info found, loading as standard foundation model...")
            try:
                # Load as standard transformer model (foundation model)
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map=self.device,
                    trust_remote_code=True
                )
                logger.info("Loaded standard foundation model successfully")
                return model
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise
    
    def _load_rgtnet_with_mapping_fix(self):
        """Load RGTNet model with proper weight mapping from base_model.* to model.*"""
        import json
        from transformers import LlamaForCausalLM, LlamaConfig
        
        # Load state dict directly
        logger.info("Loading state dict directly...")
        state_dict_path = os.path.join(self.model_path, "pytorch_model.bin")
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location='cpu')
            logger.info(f"Loaded state dict with {len(state_dict)} keys")
        else:
            # Try to load using safetensors
            try:
                from safetensors import safe_open
                safetensors_path = os.path.join(self.model_path, "model.safetensors")
                if os.path.exists(safetensors_path):
                    state_dict = {}
                    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            state_dict[key] = f.get_tensor(key)
                    logger.info(f"Loaded state dict from safetensors with {len(state_dict)} keys")
                else:
                    raise FileNotFoundError("No pytorch_model.bin or model.safetensors found")
            except ImportError:
                raise FileNotFoundError("No pytorch_model.bin found and safetensors not available")
        
        # Create a mapping from base_model.* to model.*
        logger.info("Fixing weight key mapping...")
        fixed_state_dict = {}
        
        for key, value in state_dict.items():
            if key.startswith('base_model.model.'):
                # Remove 'base_model.' prefix to get 'model.*'
                new_key = key.replace('base_model.', '')
                fixed_state_dict[new_key] = value
            elif key.startswith('base_model.lm_head.'):
                # Map base_model.lm_head.* to lm_head.*
                new_key = key.replace('base_model.', '')
                fixed_state_dict[new_key] = value
            elif key.startswith('base_model.'):
                # For other base_model keys, remove base_model prefix
                new_key = key.replace('base_model.', '')
                fixed_state_dict[new_key] = value
            else:
                # Keep other keys as-is
                fixed_state_dict[key] = value
        
        logger.info(f"Fixed state dict with {len(fixed_state_dict)} keys")
        
        # Load config and create model
        config_path = os.path.join(self.model_path, "config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = LlamaConfig(**config_dict)
        
        # Create empty model
        logger.info("Creating empty LlamaForCausalLM model...")
        model = LlamaForCausalLM(config)
        
        # Load the fixed weights
        logger.info("Loading fixed weights...")
        missing_keys, unexpected_keys = model.load_state_dict(fixed_state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            # Check if lm_head.weight is missing - it might need to be tied to embed_tokens
            if 'lm_head.weight' in missing_keys and 'model.embed_tokens.weight' in fixed_state_dict:
                logger.info("Tying lm_head.weight to embed_tokens.weight...")
                fixed_state_dict['lm_head.weight'] = fixed_state_dict['model.embed_tokens.weight']
                # Reload with the tied weight
                missing_keys, unexpected_keys = model.load_state_dict(fixed_state_dict, strict=False)
                logger.info(f"After tying weights - Missing keys: {len(missing_keys)}")
        else:
            logger.info("No missing keys!")
        
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
        
        # Move to device
        model = model.to(self.device, dtype=torch.float16)
        logger.info("RGTNet model loaded successfully with weight mapping fix")
        
        return model
    
    def generate_response(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7) -> str:
        """Generate response for a given prompt"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Decode only the new tokens
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return response.strip()
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            return ""
    
    def calculate_bleu_score(self, references: List[List[str]], predictions: List[str]) -> Dict[str, float]:
        """Calculate BLEU scores"""
        pred_tokens = [word_tokenize(pred.lower()) for pred in predictions]
        
        sentence_bleu_scores = []
        for ref, pred in zip(references, pred_tokens):
            try:
                score = sentence_bleu([ref], pred, smoothing_function=self.smoothing)
                sentence_bleu_scores.append(score)
            except Exception as e:
                logger.warning(f"BLEU calculation failed for a sample: {e}")
                sentence_bleu_scores.append(0.0)
        
        try:
            corpus_bleu_score = corpus_bleu(references, pred_tokens, smoothing_function=self.smoothing)
        except Exception as e:
            logger.warning(f"Corpus BLEU calculation failed: {e}")
            corpus_bleu_score = 0.0
        
        return {
            'bleu_sentence_avg': np.mean(sentence_bleu_scores),
            'bleu_corpus': corpus_bleu_score,
            'bleu_scores': sentence_bleu_scores
        }
    
    def calculate_rouge_scores(self, references: List[str], predictions: List[str]) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for ref, pred in zip(references, predictions):
            try:
                scores = self.rouge_scorer.score(ref, pred)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            except Exception as e:
                logger.warning(f"ROUGE calculation failed for a sample: {e}")
                rouge1_scores.append(0.0)
                rouge2_scores.append(0.0)
                rougeL_scores.append(0.0)
        
        return {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores),
            'rouge1_scores': rouge1_scores,
            'rouge2_scores': rouge2_scores,
            'rougeL_scores': rougeL_scores
        }
    
    def calculate_meteor_score(self, references: List[List[str]], predictions: List[str]) -> Dict[str, float]:
        """Calculate METEOR scores"""
        pred_tokens = [word_tokenize(pred.lower()) for pred in predictions]
        
        meteor_scores = []
        for ref, pred in zip(references, pred_tokens):
            try:
                score = meteor_score([ref], pred)
                meteor_scores.append(score)
            except Exception as e:
                logger.warning(f"METEOR calculation failed for a sample: {e}")
                meteor_scores.append(0.0)
        
        return {
            'meteor': np.mean(meteor_scores),
            'meteor_scores': meteor_scores
        }
    
    def calculate_bert_score(self, references: List[str], predictions: List[str]) -> Dict[str, float]:
        """Calculate BERTScore"""
        try:
            P, R, F1 = bert_score(predictions, references, lang='en', verbose=True)
            return {
                'bertscore_precision': P.mean().item(),
                'bertscore_recall': R.mean().item(),
                'bertscore_f1': F1.mean().item(),
                'bertscore_precision_scores': P.tolist(),
                'bertscore_recall_scores': R.tolist(),
                'bertscore_f1_scores': F1.tolist()
            }
        except Exception as e:
            logger.warning(f"BERTScore calculation failed: {e}")
            return {
                'bertscore_precision': 0.0,
                'bertscore_recall': 0.0,
                'bertscore_f1': 0.0,
                'bertscore_precision_scores': [0.0] * len(predictions),
                'bertscore_recall_scores': [0.0] * len(predictions),
                'bertscore_f1_scores': [0.0] * len(predictions)
            }
    
    def evaluate_on_dataset(self, data_file: str, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate model on a dataset"""
        logger.info(f"Loading dataset from: {data_file}")
        
        # Load dataset
        with open(data_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        if max_samples:
            data = data[:max_samples]
        
        logger.info(f"Evaluating on {len(data)} samples")
        
        references = []
        predictions = []
        prompts = []
        
        # Generate predictions
        for i, sample in enumerate(tqdm(data, desc="Generating predictions")):
            try:
                # Extract instruction and reference
                instruction = sample.get('instruction', '')
                reference = sample.get('response', '')
                
                if not instruction or not reference:
                    continue
                
                # Create prompt
                prompt = f"Human: {instruction}\n\nAssistant:"
                
                # Generate prediction
                prediction = self.generate_response(prompt)
                
                if prediction:  # Only add if generation was successful
                    prompts.append(prompt)
                    references.append(reference)
                    predictions.append(prediction)
                
            except Exception as e:
                logger.warning(f"Failed to process sample {i}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(predictions)} samples")
        
        if len(predictions) == 0:
            logger.error("No successful predictions generated")
            return {'error': 'No successful predictions generated'}
        
        # Calculate metrics
        results = {}
        
        # Tokenize references for BLEU and METEOR
        ref_tokens = [word_tokenize(ref.lower()) for ref in references]
        
        # BLEU scores
        logger.info("Calculating BLEU scores...")
        bleu_results = self.calculate_bleu_score(ref_tokens, predictions)
        results.update(bleu_results)
        
        # ROUGE scores
        logger.info("Calculating ROUGE scores...")
        rouge_results = self.calculate_rouge_scores(references, predictions)
        results.update(rouge_results)
        
        # METEOR scores
        logger.info("Calculating METEOR scores...")
        meteor_results = self.calculate_meteor_score(ref_tokens, predictions)
        results.update(meteor_results)
        
        # BERTScore
        logger.info("Calculating BERTScore...")
        bert_results = self.calculate_bert_score(references, predictions)
        results.update(bert_results)
        
        # Store sample outputs
        results['sample_outputs'] = []
        for i in range(min(5, len(predictions))):
            results['sample_outputs'].append({
                'prompt': prompts[i],
                'reference': references[i],
                'prediction': predictions[i]
            })
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save evaluation results to a file."""
        # Add metadata
        results['metadata'] = {
            'model_path': self.model_path,
            'tokenizer_path': self.model_path,  # Same as model path
            'evaluation_date': datetime.now().isoformat(),
            'device': self.device
        }
        
        # Save detailed results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Create summary CSV
        summary_data = {
            'Metric': ['BLEU (Sentence Avg)', 'BLEU (Corpus)', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'METEOR', 'BERTScore-P', 'BERTScore-R', 'BERTScore-F1'],
            'Score': [
                results.get('bleu_sentence_avg', 0),
                results.get('bleu_corpus', 0),
                results.get('rouge1', 0),
                results.get('rouge2', 0),
                results.get('rougeL', 0),
                results.get('meteor', 0),
                results.get('bertscore_precision', 0),
                results.get('bertscore_recall', 0),
                results.get('bertscore_f1', 0)
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = output_file.replace('.json', '_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"Summary saved to: {summary_file}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of evaluation results."""
        if 'error' in results:
            print(f"\n❌ Evaluation failed: {results['error']}")
            return
            
        print("\n" + "="*50)
        print("SAFE LLM EVALUATION RESULTS")
        print("="*50)
        print(f"Model: {self.model_path}")
        print(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        print("METRICS SUMMARY:")
        print("-" * 30)
        print(f"BLEU (Sentence Avg): {results.get('bleu_sentence_avg', 'N/A'):.4f}")
        print(f"BLEU (Corpus):       {results.get('bleu_corpus', 'N/A'):.4f}")
        print(f"ROUGE-1:            {results.get('rouge1', 'N/A'):.4f}")
        print(f"ROUGE-2:            {results.get('rouge2', 'N/A'):.4f}")
        print(f"ROUGE-L:            {results.get('rougeL', 'N/A'):.4f}")
        print(f"METEOR:             {results.get('meteor', 'N/A'):.4f}")
        print(f"BERTScore-P:        {results.get('bertscore_precision', 'N/A'):.4f}")
        print(f"BERTScore-R:        {results.get('bertscore_recall', 'N/A'):.4f}")
        print(f"BERTScore-F1:       {results.get('bertscore_f1', 'N/A'):.4f}")
        
        # Print sample generations
        sample_generations = results.get('sample_outputs', [])
        if sample_generations:
            print("\nSAMPLE GENERATIONS:")
            print("-" * 30)
            for i, gen in enumerate(sample_generations[:3]):  # Show first 3
                print(f"Sample {i+1}:")
                print(f"Prompt: {gen['prompt'][:100]}...")
                print(f"Reference: {gen['reference'][:100]}...")
                print(f"Generated: {gen['prediction'][:100]}...")
                print()
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Safe LLM performance evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model (RGTNet or foundation model)")
    parser.add_argument("--data_file", type=str, default="data/val_instruction.jsonl", help="Evaluation dataset file")
    parser.add_argument("--output_file", type=str, default=None, help="Output file for results")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum new tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    
    args = parser.parse_args()
    
    # Set output file if not specified
    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_file = f"evaluation_results/safe_llm_evaluation_{timestamp}.json"
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Initialize evaluator
    evaluator = SafeLLMEvaluator(
        model_path=args.model_path,
        device=args.device
    )
    
    # Evaluate model
    logger.info("Starting safe evaluation...")
    results = evaluator.evaluate_on_dataset(
        data_file=args.data_file,
        max_samples=args.max_samples
    )
    
    # Save results
    evaluator.save_results(results, args.output_file)
    
    # Print summary
    evaluator.print_summary(results)
    
    logger.info("Safe evaluation completed successfully!")

if __name__ == "__main__":
    main()
