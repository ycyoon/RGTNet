#!/usr/bin/env python3
"""
Multiple Model Evaluation Script
Evaluates multiple models (RGTNet and foundation models) and compares their performance
"""

import os
import json
import argparse
import subprocess
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiModelEvaluator:
    """Evaluates multiple models and compares their performance"""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def evaluate_model(self, model_path: str, model_name: str, data_file: str, 
                      max_samples: int = 100, device: str = "cuda") -> Dict[str, Any]:
        """Evaluate a single model"""
        logger.info(f"Evaluating model: {model_name} at {model_path}")
        
        # Create output file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{self.output_dir}/{model_name}_evaluation_{timestamp}.json"
        
        # Run evaluation command
        cmd = [
            "python", "evaluate_llm_performance.py",
            "--model_path", model_path,
            "--data_file", data_file,
            "--output_file", output_file,
            "--max_samples", str(max_samples),
            "--device", device
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Successfully evaluated {model_name}")
            
            # Load results
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            return {
                'model_name': model_name,
                'model_path': model_path,
                'results': results,
                'output_file': output_file
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
            logger.error(f"Error output: {e.stderr}")
            return {
                'model_name': model_name,
                'model_path': model_path,
                'error': str(e),
                'results': None
            }
    
    def evaluate_models(self, models: List[Dict[str, str]], data_file: str, 
                       max_samples: int = 100, device: str = "cuda") -> List[Dict[str, Any]]:
        """Evaluate multiple models"""
        results = []
        
        for model_config in models:
            model_name = model_config['name']
            model_path = model_config['path']
            
            result = self.evaluate_model(
                model_path=model_path,
                model_name=model_name,
                data_file=data_file,
                max_samples=max_samples,
                device=device
            )
            
            results.append(result)
        
        return results
    
    def create_comparison_report(self, evaluation_results: List[Dict[str, Any]]) -> str:
        """Create a comparison report of all evaluated models"""
        # Filter successful evaluations
        successful_results = [r for r in evaluation_results if r.get('results') is not None]
        
        if not successful_results:
            logger.error("No successful evaluations to compare")
            return None
        
        # Extract key metrics
        comparison_data = []
        for result in successful_results:
            model_name = result['model_name']
            results = result['results']
            
            comparison_data.append({
                'Model': model_name,
                'BLEU (Sentence Avg)': results.get('bleu_sentence_avg', 0),
                'BLEU (Corpus)': results.get('bleu_corpus', 0),
                'ROUGE-1': results.get('rouge1', 0),
                'ROUGE-2': results.get('rouge2', 0),
                'ROUGE-L': results.get('rougeL', 0),
                'METEOR': results.get('meteor', 0),
                'BERTScore-P': results.get('bertscore_precision', 0),
                'BERTScore-R': results.get('bertscore_recall', 0),
                'BERTScore-F1': results.get('bertscore_f1', 0)
            })
        
        # Create DataFrame
        df = pd.DataFrame(comparison_data)
        
        # Save comparison report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = f"{self.output_dir}/model_comparison_{timestamp}.csv"
        df.to_csv(comparison_file, index=False)
        
        # Print comparison
        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*80)
        print(df.to_string(index=False, float_format='%.4f'))
        print("\n" + "="*80)
        
        return comparison_file
    
    def create_advanced_comparison(self, evaluation_results: List[Dict[str, Any]]) -> str:
        """Create advanced comparison with additional metrics"""
        successful_results = [r for r in evaluation_results if r.get('results') is not None]
        
        if not successful_results:
            return None
        
        # Extract comprehensive metrics
        comparison_data = []
        for result in successful_results:
            model_name = result['model_name']
            results = result['results']
            
            # Standard metrics
            data = {
                'Model': model_name,
                'BLEU (Sentence Avg)': results.get('bleu_sentence_avg', 0),
                'BLEU (Corpus)': results.get('bleu_corpus', 0),
                'ROUGE-1': results.get('rouge1', 0),
                'ROUGE-2': results.get('rouge2', 0),
                'ROUGE-L': results.get('rougeL', 0),
                'METEOR': results.get('meteor', 0),
                'BERTScore-P': results.get('bertscore_precision', 0),
                'BERTScore-R': results.get('bertscore_recall', 0),
                'BERTScore-F1': results.get('bertscore_f1', 0)
            }
            
            # Length metrics
            ref_len = results.get('reference_length_metrics', {})
            pred_len = results.get('prediction_length_metrics', {})
            data.update({
                'Ref Avg Length': ref_len.get('avg_length', 0),
                'Pred Avg Length': pred_len.get('avg_length', 0),
                'Length Difference': abs(ref_len.get('avg_length', 0) - pred_len.get('avg_length', 0))
            })
            
            # Diversity metrics
            ref_div = results.get('reference_diversity_metrics', {})
            pred_div = results.get('prediction_diversity_metrics', {})
            data.update({
                'Ref Type-Token Ratio': ref_div.get('type_token_ratio', 0),
                'Pred Type-Token Ratio': pred_div.get('type_token_ratio', 0),
                'Ref Simpson Diversity': ref_div.get('simpson_diversity', 0),
                'Pred Simpson Diversity': pred_div.get('simpson_diversity', 0)
            })
            
            comparison_data.append(data)
        
        # Create DataFrame
        df = pd.DataFrame(comparison_data)
        
        # Save advanced comparison
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        advanced_file = f"{self.output_dir}/advanced_model_comparison_{timestamp}.csv"
        df.to_csv(advanced_file, index=False)
        
        return advanced_file

def load_model_config(config_file: str) -> List[Dict[str, str]]:
    """Load model configuration from JSON file"""
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config['models']

def create_sample_config():
    """Create a sample configuration file"""
    sample_config = {
        "models": [
            {
                "name": "RGTNet_Trained",
                "path": "models",
                "description": "Trained RGTNet model"
            },
            {
                "name": "Llama_3.2_1B_Base",
                "path": "meta-llama/Llama-3.2-1B-Instruct",
                "description": "Base Llama 3.2 1B Instruct model"
            },
            {
                "name": "Llama_3.2_3B_Base",
                "path": "meta-llama/Llama-3.2-3B-Instruct",
                "description": "Base Llama 3.2 3B Instruct model"
            }
        ]
    }
    
    with open("model_config.json", 'w', encoding='utf-8') as f:
        json.dump(sample_config, f, indent=2)
    
    print("Sample configuration file created: model_config.json")
    print("Please edit this file to specify your models for evaluation.")

def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple LLM models")
    parser.add_argument("--config_file", type=str, default="model_config.json", 
                       help="JSON file containing model configurations")
    parser.add_argument("--data_file", type=str, default="data/val_instruction.jsonl", 
                       help="Evaluation dataset file")
    parser.add_argument("--max_samples", type=int, default=100, 
                       help="Maximum number of samples to evaluate per model")
    parser.add_argument("--device", type=str, default="cuda", 
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--create_sample_config", action="store_true", 
                       help="Create a sample configuration file")
    
    args = parser.parse_args()
    
    if args.create_sample_config:
        create_sample_config()
        return
    
    # Check if config file exists
    if not os.path.exists(args.config_file):
        print(f"Configuration file not found: {args.config_file}")
        print("Creating sample configuration file...")
        create_sample_config()
        return
    
    # Load model configurations
    try:
        models = load_model_config(args.config_file)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return
    
    logger.info(f"Loaded {len(models)} models for evaluation")
    
    # Initialize evaluator
    evaluator = MultiModelEvaluator()
    
    # Evaluate all models
    logger.info("Starting multi-model evaluation...")
    results = evaluator.evaluate_models(
        models=models,
        data_file=args.data_file,
        max_samples=args.max_samples,
        device=args.device
    )
    
    # Create comparison reports
    logger.info("Creating comparison reports...")
    comparison_file = evaluator.create_comparison_report(results)
    advanced_file = evaluator.create_advanced_comparison(results)
    
    # Print summary
    successful_count = len([r for r in results if r.get('results') is not None])
    failed_count = len(results) - successful_count
    
    print(f"\n📊 Evaluation Summary:")
    print(f"✅ Successful evaluations: {successful_count}")
    print(f"❌ Failed evaluations: {failed_count}")
    
    if comparison_file:
        print(f"📈 Comparison report: {comparison_file}")
    if advanced_file:
        print(f"📊 Advanced comparison: {advanced_file}")
    
    # Print failed evaluations
    failed_models = [r for r in results if r.get('results') is None]
    if failed_models:
        print(f"\n❌ Failed Models:")
        for model in failed_models:
            print(f"  - {model['model_name']}: {model.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
