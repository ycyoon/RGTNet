#!/usr/bin/env python3
"""
Analyze benchmark results and create a summary table in Markdown format
"""

import json
import os
import glob
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any

class BenchmarkAnalyzer:
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.results = {}
        
    def find_result_files(self) -> List[Path]:
        """Find all result files in the directory"""
        patterns = [
            "**/*results*.json",
            "**/*benchmark*.json",
            "**/*attack*.json",
            "**/*evaluation*.json",
            "**/results_*.json",
            "PAIR_results_*.json",
            "WildteamAttack_results_*.json",
            "jailbroken_results_*.json",
            "multi_model_results_*.json"
        ]
        
        result_files = []
        for pattern in patterns:
            files = list(self.base_dir.glob(pattern))
            result_files.extend(files)
            
        # Remove duplicates and filter out cache/temp files
        result_files = list(set(result_files))
        result_files = [f for f in result_files if '.cache' not in str(f) and '__pycache__' not in str(f)]
        
        print(f"Found {len(result_files)} result files")
        return sorted(result_files)
    
    def parse_result_file(self, filepath: Path) -> Dict[str, Any]:
        """Parse a single result file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return {}
    
    def extract_metrics(self, data: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Extract relevant metrics from result data"""
        metrics = {
            'filename': filename,
            'model': 'Unknown',
            'attack_type': 'Unknown',
            'dataset_size': 0,
            'success_rate': 0.0,
            'avg_queries': 0,
            'total_time': 0,
            'timestamp': None
        }
        
        # Try to extract model name
        if 'model' in data:
            metrics['model'] = data['model']
        elif 'target_model' in data:
            metrics['model'] = data['target_model']
        elif 'llama' in filename.lower():
            metrics['model'] = 'llama-3.2-3b'
            
        # Try to extract attack type
        if 'attack_type' in data:
            metrics['attack_type'] = data['attack_type']
        elif 'attacker' in data:
            metrics['attack_type'] = data['attacker']
        elif 'PAIR' in filename:
            metrics['attack_type'] = 'PAIR'
        elif 'Wildteam' in filename:
            metrics['attack_type'] = 'WildteamAttack'
        elif 'jailbroken' in filename:
            metrics['attack_type'] = 'Jailbroken'
        elif 'multi_model' in filename:
            metrics['attack_type'] = 'Multi-Model'
            
        # Extract numeric metrics
        if 'results' in data and isinstance(data['results'], list):
            results = data['results']
            metrics['dataset_size'] = len(results)
            
            # Calculate success rate
            successful = sum(1 for r in results if r.get('jailbroken', False) or r.get('success', False))
            metrics['success_rate'] = (successful / len(results) * 100) if results else 0
            
            # Average queries
            queries = [r.get('queries', 0) or r.get('num_queries', 0) for r in results]
            metrics['avg_queries'] = sum(queries) / len(queries) if queries else 0
            
        elif 'evaluation_results' in data:
            eval_results = data['evaluation_results']
            if isinstance(eval_results, list):
                metrics['dataset_size'] = len(eval_results)
                successful = sum(1 for r in eval_results if r.get('is_jailbroken', False))
                metrics['success_rate'] = (successful / len(eval_results) * 100) if eval_results else 0
                
        # Extract timing
        if 'total_time' in data:
            metrics['total_time'] = data['total_time']
        elif 'elapsed_time' in data:
            metrics['total_time'] = data['elapsed_time']
            
        # Extract timestamp
        if 'timestamp' in data:
            metrics['timestamp'] = data['timestamp']
        elif 'date' in data:
            metrics['timestamp'] = data['date']
            
        return metrics
    
    def create_summary_table(self) -> pd.DataFrame:
        """Create a summary table of all results"""
        all_metrics = []
        
        result_files = self.find_result_files()
        for filepath in result_files:
            data = self.parse_result_file(filepath)
            if data:
                metrics = self.extract_metrics(data, filepath.name)
                all_metrics.append(metrics)
                
        if not all_metrics:
            print("No valid metrics found")
            return pd.DataFrame()
            
        df = pd.DataFrame(all_metrics)
        
        # Group by model and attack type
        summary = df.groupby(['model', 'attack_type']).agg({
            'success_rate': 'mean',
            'avg_queries': 'mean',
            'dataset_size': 'max',
            'total_time': 'mean'
        }).round(2)
        
        return summary
    
    def generate_markdown_report(self, output_file: str = "benchmark_results_summary.md"):
        """Generate a comprehensive markdown report"""
        summary_df = self.create_summary_table()
        
        with open(output_file, 'w') as f:
            f.write("# RGTNet Benchmark Results Summary\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall summary table
            f.write("## Overall Performance Summary\n\n")
            
            if not summary_df.empty:
                # Convert to markdown table
                f.write("| Model | Attack Type | Success Rate (%) | Avg Queries | Dataset Size | Avg Time (s) |\n")
                f.write("|-------|-------------|------------------|-------------|--------------|-------------|\n")
                
                for (model, attack), row in summary_df.iterrows():
                    f.write(f"| {model} | {attack} | {row['success_rate']:.1f} | "
                           f"{row['avg_queries']:.1f} | {int(row['dataset_size'])} | "
                           f"{row['total_time']:.1f} |\n")
            else:
                f.write("No results found.\n")
                
            # Detailed results by attack type
            f.write("\n## Detailed Results by Attack Type\n\n")
            
            result_files = self.find_result_files()
            attack_types = {}
            
            for filepath in result_files:
                data = self.parse_result_file(filepath)
                if data:
                    metrics = self.extract_metrics(data, filepath.name)
                    attack = metrics['attack_type']
                    if attack not in attack_types:
                        attack_types[attack] = []
                    attack_types[attack].append(metrics)
                    
            for attack, results in attack_types.items():
                f.write(f"\n### {attack}\n\n")
                f.write("| Model | Success Rate (%) | Queries | Time (s) | File |\n")
                f.write("|-------|------------------|---------|----------|------|\n")
                
                for r in results:
                    f.write(f"| {r['model']} | {r['success_rate']:.1f} | "
                           f"{r['avg_queries']:.1f} | {r['total_time']:.1f} | "
                           f"{r['filename']} |\n")
                    
            # File listing
            f.write("\n## Result Files Found\n\n")
            for i, filepath in enumerate(result_files, 1):
                f.write(f"{i}. `{filepath}`\n")
                
        print(f"✅ Report generated: {output_file}")
        
        # Also create a CSV for easy analysis
        if not summary_df.empty:
            csv_file = output_file.replace('.md', '.csv')
            summary_df.to_csv(csv_file)
            print(f"📊 CSV file created: {csv_file}")

def main():
    analyzer = BenchmarkAnalyzer()
    analyzer.generate_markdown_report()
    
    # Also print summary to console
    print("\n📊 Quick Summary:")
    summary = analyzer.create_summary_table()
    if not summary.empty:
        print(summary.to_string())
    else:
        print("No results found to summarize.")

if __name__ == "__main__":
    main()
