#!/usr/bin/env python3
"""
Advanced LLM Performance Evaluation Script
Includes additional metrics, visualizations, and support for multiple datasets
"""

import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

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

class AdvancedLLMEvaluator:
    """Advanced LLM performance evaluator with additional metrics and visualizations"""
    
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
        
        logger.info(f"Advanced evaluator initialized with model: {model_path}")
        
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
    
    def calculate_length_metrics(self, texts: List[str]) -> Dict[str, float]:
        """Calculate length-based metrics"""
        lengths = [len(text.split()) for text in texts]
        return {
            'avg_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths),
            'lengths': lengths
        }
    
    def calculate_diversity_metrics(self, texts: List[str]) -> Dict[str, float]:
        """Calculate text diversity metrics"""
        # Unique words
        all_words = []
        for text in texts:
            words = word_tokenize(text.lower())
            all_words.extend(words)
        
        unique_words = set(all_words)
        total_words = len(all_words)
        
        # Type-token ratio
        ttr = len(unique_words) / total_words if total_words > 0 else 0
        
        # Lexical diversity (Simpson's D)
        word_counts = defaultdict(int)
        for word in all_words:
            word_counts[word] += 1
        
        simpson_d = 0
        for count in word_counts.values():
            simpson_d += (count * (count - 1)) / (total_words * (total_words - 1))
        simpson_d = 1 - simpson_d if total_words > 1 else 0
        
        return {
            'type_token_ratio': ttr,
            'simpson_diversity': simpson_d,
            'unique_words': len(unique_words),
            'total_words': total_words
        }
    
    def calculate_fluency_metrics(self, texts: List[str]) -> Dict[str, float]:
        """Calculate fluency metrics (simplified)"""
        # Count sentences (simple period-based)
        sentence_counts = []
        for text in texts:
            sentences = text.split('.')
            sentence_counts.append(len([s for s in sentences if s.strip()]))
        
        avg_sentences = np.mean(sentence_counts)
        
        # Average words per sentence
        total_words = sum(len(text.split()) for text in texts)
        total_sentences = sum(sentence_counts)
        avg_words_per_sentence = total_words / total_sentences if total_sentences > 0 else 0
        
        return {
            'avg_sentences': avg_sentences,
            'avg_words_per_sentence': avg_words_per_sentence,
            'sentence_counts': sentence_counts
        }
    
    def calculate_standard_metrics(self, references: List[str], predictions: List[str]) -> Dict[str, Any]:
        """Calculate standard evaluation metrics"""
        # Tokenize references for BLEU and METEOR
        ref_tokens = [word_tokenize(ref.lower()) for ref in references]
        
        # BLEU scores
        logger.info("Calculating BLEU scores...")
        bleu_results = self.calculate_bleu_score(ref_tokens, predictions)
        
        # ROUGE scores
        logger.info("Calculating ROUGE scores...")
        rouge_results = self.calculate_rouge_scores(references, predictions)
        
        # METEOR scores
        logger.info("Calculating METEOR scores...")
        meteor_results = self.calculate_meteor_score(ref_tokens, predictions)
        
        # BERTScore
        logger.info("Calculating BERTScore...")
        bert_results = self.calculate_bert_score(references, predictions)
        
        return {**bleu_results, **rouge_results, **meteor_results, **bert_results}
    
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
        """Evaluate model on a dataset with comprehensive metrics"""
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
                instruction = sample.get('instruction', '')
                reference = sample.get('response', '')
                
                if not instruction or not reference:
                    continue
                
                prompt = f"Human: {instruction}\n\nAssistant:"
                prediction = self.generate_response(prompt)
                
                prompts.append(prompt)
                references.append(reference)
                predictions.append(prediction)
                
            except Exception as e:
                logger.warning(f"Failed to process sample {i}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(predictions)} samples")
        
        # Calculate all metrics
        results = {}
        
        # Standard metrics
        standard_metrics = self.calculate_standard_metrics(references, predictions)
        results.update(standard_metrics)
        
        # Length metrics
        logger.info("Calculating length metrics...")
        ref_length_metrics = self.calculate_length_metrics(references)
        pred_length_metrics = self.calculate_length_metrics(predictions)
        results['reference_length_metrics'] = ref_length_metrics
        results['prediction_length_metrics'] = pred_length_metrics
        
        # Diversity metrics
        logger.info("Calculating diversity metrics...")
        ref_diversity = self.calculate_diversity_metrics(references)
        pred_diversity = self.calculate_diversity_metrics(predictions)
        results['reference_diversity_metrics'] = ref_diversity
        results['prediction_diversity_metrics'] = pred_diversity
        
        # Fluency metrics
        logger.info("Calculating fluency metrics...")
        ref_fluency = self.calculate_fluency_metrics(references)
        pred_fluency = self.calculate_fluency_metrics(predictions)
        results['reference_fluency_metrics'] = ref_fluency
        results['prediction_fluency_metrics'] = pred_fluency
        
        # Store sample outputs
        results['sample_outputs'] = []
        for i in range(min(5, len(predictions))):
            results['sample_outputs'].append({
                'prompt': prompts[i],
                'reference': references[i],
                'prediction': predictions[i]
            })
        
        return results
    
    def create_visualizations(self, results: Dict[str, Any], output_dir: str):
        """Create visualization plots for evaluation results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Main metrics comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        metrics = ['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'METEOR', 'BERTScore-F1']
        scores = [
            results.get('bleu_sentence_avg', 0),
            results.get('rouge1', 0),
            results.get('rouge2', 0),
            results.get('rougeL', 0),
            results.get('meteor', 0),
            results.get('bertscore_f1', 0)
        ]
        
        bars = ax.bar(metrics, scores, color='skyblue', alpha=0.7)
        ax.set_title('LLM Evaluation Metrics', fontsize=16, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/main_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Length distribution comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ref_lengths = results.get('reference_length_metrics', {}).get('lengths', [])
        pred_lengths = results.get('prediction_length_metrics', {}).get('lengths', [])
        
        ax1.hist(ref_lengths, bins=30, alpha=0.7, label='Reference', color='blue')
        ax1.hist(pred_lengths, bins=30, alpha=0.7, label='Prediction', color='red')
        ax1.set_title('Text Length Distribution')
        ax1.set_xlabel('Number of Words')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        # Box plot
        data_to_plot = [ref_lengths, pred_lengths]
        ax2.boxplot(data_to_plot, labels=['Reference', 'Prediction'])
        ax2.set_title('Text Length Box Plot')
        ax2.set_ylabel('Number of Words')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/length_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Score distributions
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # BLEU scores
        bleu_scores = results.get('bleu_scores', [])
        axes[0, 0].hist(bleu_scores, bins=20, alpha=0.7, color='green')
        axes[0, 0].set_title('BLEU Score Distribution')
        axes[0, 0].set_xlabel('BLEU Score')
        axes[0, 0].set_ylabel('Frequency')
        
        # ROUGE-1 scores
        rouge1_scores = results.get('rouge1_scores', [])
        axes[0, 1].hist(rouge1_scores, bins=20, alpha=0.7, color='orange')
        axes[0, 1].set_title('ROUGE-1 Score Distribution')
        axes[0, 1].set_xlabel('ROUGE-1 Score')
        axes[0, 1].set_ylabel('Frequency')
        
        # METEOR scores
        meteor_scores = results.get('meteor_scores', [])
        axes[1, 0].hist(meteor_scores, bins=20, alpha=0.7, color='purple')
        axes[1, 0].set_title('METEOR Score Distribution')
        axes[1, 0].set_xlabel('METEOR Score')
        axes[1, 0].set_ylabel('Frequency')
        
        # BERTScore F1
        bert_f1_scores = results.get('bertscore_f1_scores', [])
        axes[1, 1].hist(bert_f1_scores, bins=20, alpha=0.7, color='brown')
        axes[1, 1].set_title('BERTScore F1 Distribution')
        axes[1, 1].set_xlabel('BERTScore F1')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/score_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to: {output_dir}")
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save evaluation results to file with visualizations"""
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
        
        # Create comprehensive summary CSV
        summary_data = {
            'Metric': [
                'BLEU (Sentence Avg)', 'BLEU (Corpus)', 
                'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 
                'METEOR', 'BERTScore-P', 'BERTScore-R', 'BERTScore-F1',
                'Ref Avg Length', 'Pred Avg Length',
                'Ref Type-Token Ratio', 'Pred Type-Token Ratio',
                'Ref Simpson Diversity', 'Pred Simpson Diversity'
            ],
            'Score': [
                results.get('bleu_sentence_avg', 0),
                results.get('bleu_corpus', 0),
                results.get('rouge1', 0),
                results.get('rouge2', 0),
                results.get('rougeL', 0),
                results.get('meteor', 0),
                results.get('bertscore_precision', 0),
                results.get('bertscore_recall', 0),
                results.get('bertscore_f1', 0),
                results.get('reference_length_metrics', {}).get('avg_length', 0),
                results.get('prediction_length_metrics', {}).get('avg_length', 0),
                results.get('reference_diversity_metrics', {}).get('type_token_ratio', 0),
                results.get('prediction_diversity_metrics', {}).get('type_token_ratio', 0),
                results.get('reference_diversity_metrics', {}).get('simpson_diversity', 0),
                results.get('prediction_diversity_metrics', {}).get('simpson_diversity', 0)
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = output_file.replace('.json', '_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        
        # Create visualizations
        viz_dir = output_file.replace('.json', '_visualizations')
        self.create_visualizations(results, viz_dir)
        
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"Summary saved to: {summary_file}")
        logger.info(f"Visualizations saved to: {viz_dir}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print comprehensive evaluation summary"""
        print("\n" + "="*70)
        print("ADVANCED LLM PERFORMANCE EVALUATION RESULTS")
        print("="*70)
        print(f"Model: {self.model_path}")
        print(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        print("STANDARD METRICS:")
        print("-" * 50)
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
        
        print("LENGTH METRICS:")
        print("-" * 50)
        ref_len = results.get('reference_length_metrics', {})
        pred_len = results.get('prediction_length_metrics', {})
        print(f"Reference Avg Length: {ref_len.get('avg_length', 0):.2f} ± {ref_len.get('std_length', 0):.2f}")
        print(f"Prediction Avg Length: {pred_len.get('avg_length', 0):.2f} ± {pred_len.get('std_length', 0):.2f}")
        print()
        
        print("DIVERSITY METRICS:")
        print("-" * 50)
        ref_div = results.get('reference_diversity_metrics', {})
        pred_div = results.get('prediction_diversity_metrics', {})
        print(f"Reference Type-Token Ratio: {ref_div.get('type_token_ratio', 0):.4f}")
        print(f"Prediction Type-Token Ratio: {pred_div.get('type_token_ratio', 0):.4f}")
        print(f"Reference Simpson Diversity: {ref_div.get('simpson_diversity', 0):.4f}")
        print(f"Prediction Simpson Diversity: {pred_div.get('simpson_diversity', 0):.4f}")
        print()
        
        print("SAMPLE OUTPUTS:")
        print("-" * 50)
        for i, sample in enumerate(results.get('sample_outputs', [])[:3]):
            print(f"Sample {i+1}:")
            print(f"Prompt:     {sample['prompt'][:100]}...")
            print(f"Reference:  {sample['reference'][:100]}...")
            print(f"Prediction: {sample['prediction'][:100]}...")
            print()
        
        print("="*70)

def main():
    parser = argparse.ArgumentParser(description="Advanced LLM performance evaluation")
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
        args.output_file = f"evaluation_results/advanced_llm_evaluation_{timestamp}.json"
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Initialize evaluator
    evaluator = AdvancedLLMEvaluator(
        model_path=args.model_path,
        device=args.device
    )
    
    # Evaluate model
    logger.info("Starting advanced evaluation...")
    results = evaluator.evaluate_on_dataset(
        data_file=args.data_file,
        max_samples=args.max_samples
    )
    
    # Save results
    evaluator.save_results(results, args.output_file)
    
    # Print summary
    evaluator.print_summary(results)
    
    logger.info("Advanced evaluation completed successfully!")

if __name__ == "__main__":
    main()
