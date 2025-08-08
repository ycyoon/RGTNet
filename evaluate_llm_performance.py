#!/usr/bin/env python3
"""
LLM Performance Evaluation Script
Evaluates trained models using multiple metrics: BLEU, ROUGE, METEOR, BERTScore, etc.
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
from data_loader import create_dataloader
from model import RGTNetModel
from config import get_config

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

class LLMEvaluator:
    """Comprehensive LLM performance evaluator"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.model_path = model_path
        
        # Initialize tokenizer from the same model path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        self.model = self._load_model()
        self.model.to(device)
        self.model.eval()
        
        # Initialize scorers
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method4
        
        logger.info(f"Evaluator initialized with model: {model_path}")
        
    def _load_model(self):
        """Load the trained model - supports both RGTNet and standard foundation models"""
        try:
            # Try to load as RGTNet model first
            model = RGTNetModel.from_pretrained(self.model_path)
            logger.info("Loaded RGTNet model successfully")
            return model
        except Exception as e:
            logger.info(f"Not a RGTNet model, trying as standard foundation model: {e}")
            try:
                # Load as standard transformer model (foundation model)
                model = AutoModelForCausalLM.from_pretrained(self.model_path)
                logger.info("Loaded standard foundation model successfully")
                return model
            except Exception as e2:
                logger.error(f"Failed to load model: {e2}")
                raise
    
    def generate_response(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7) -> str:
        """Generate response for a given prompt"""
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
    
    def calculate_bleu_score(self, references: List[List[str]], predictions: List[str]) -> Dict[str, float]:
        """Calculate BLEU scores"""
        # Tokenize predictions
        pred_tokens = [word_tokenize(pred.lower()) for pred in predictions]
        
        # Calculate sentence-level BLEU
        sentence_bleu_scores = []
        for ref, pred in zip(references, pred_tokens):
            try:
                score = sentence_bleu([ref], pred, smoothing_function=self.smoothing)
                sentence_bleu_scores.append(score)
            except Exception as e:
                logger.warning(f"BLEU calculation failed for a sample: {e}")
                sentence_bleu_scores.append(0.0)
        
        # Calculate corpus-level BLEU
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
        # Tokenize predictions
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
                
                # Store results
                prompts.append(prompt)
                references.append(reference)
                predictions.append(prediction)
                
            except Exception as e:
                logger.warning(f"Failed to process sample {i}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(predictions)} samples")
        
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
        """Save evaluation results to file"""
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
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("LLM PERFORMANCE EVALUATION RESULTS")
        print("="*60)
        print(f"Model: {self.model_path}")
        print(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        print("METRICS SUMMARY:")
        print("-" * 40)
        print(f"BLEU (Sentence Avg): {results.get('bleu_sentence_avg', 0):.4f}")
        print(f"BLEU (Corpus):       {results.get('bleu_corpus', 0):.4f}")
        print(f"ROUGE-1:            {results.get('rouge1', 0):.4f}")
        print(f"ROUGE-2:            {results.get('rouge2', 0):.4f}")
        print(f"ROUGE-L:            {results.get('rougeL', 0):.4f}")
        print(f"METEOR:             {results.get('meteor', 0):.4f}")
        print(f"BERTScore-P:        {results.get('bertscore_precision', 0):.4f}")
        print(f"BERTScore-R:        {results.get('bertscore_recall', 0):.4f}")
        print(f"BERTScore-F1:       {results.get('bertscore_f1', 0):.4f}")
        print()
        
        print("SAMPLE OUTPUTS:")
        print("-" * 40)
        for i, sample in enumerate(results.get('sample_outputs', [])[:3]):
            print(f"Sample {i+1}:")
            print(f"Prompt:     {sample['prompt'][:100]}...")
            print(f"Reference:  {sample['reference'][:100]}...")
            print(f"Prediction: {sample['prediction'][:100]}...")
            print()
        
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM performance on golden standard dataset")
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
        args.output_file = f"evaluation_results/llm_evaluation_{timestamp}.json"
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Initialize evaluator
    evaluator = LLMEvaluator(
        model_path=args.model_path,
        device=args.device
    )
    
    # Evaluate model
    logger.info("Starting evaluation...")
    results = evaluator.evaluate_on_dataset(
        data_file=args.data_file,
        max_samples=args.max_samples
    )
    
    # Save results
    evaluator.save_results(results, args.output_file)
    
    # Print summary
    evaluator.print_summary(results)
    
    logger.info("Evaluation completed successfully!")

if __name__ == "__main__":
    main()
