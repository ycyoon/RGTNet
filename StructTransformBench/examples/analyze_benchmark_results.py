#!/usr/bin/env python3
"""
Benchmark Results Analysis Script
Analyzes JSON result files from various attack methods and generates summary reports.
"""

import json
import sys
import os
import glob
from datetime import datetime
from pathlib import Path


def analyze_file(filepath):
    """Analyze a single result file and extract key metrics"""
    try:
        if not os.path.exists(filepath):
            return None
        
        filename = os.path.basename(filepath)
        
        # Check file type and use appropriate parser
        if filepath.endswith('.log'):
            return analyze_log_file(filepath)
        elif filepath.endswith('.md'):
            return analyze_markdown_file(filepath)
        elif filepath.endswith('.json') or filepath.endswith('.jsonl'):
            return analyze_json_file(filepath)
        else:
            # Try to detect format by content
            with open(filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if first_line.startswith('#'):
                    return analyze_markdown_file(filepath)
                elif first_line.startswith('{'):
                    return analyze_json_file(filepath)
                else:
                    return analyze_log_file(filepath)
        
    except Exception as e:
        print(f'Warning: Could not analyze {filepath}: {e}', file=sys.stderr)
        return None


def analyze_json_file(filepath):
    """Analyze a JSON result file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        filename = os.path.basename(filepath)
        
        # Extract model name
        model = data.get('target_model', data.get('model', 'Unknown'))
        if isinstance(model, str) and '/' in model:
            model = model.split('/')[-1]  # Get just the model name part
        
        # Determine attack type from filename
        attack_type = 'Unknown'
        if 'PAIR' in filename:
            attack_type = 'PAIR'
        elif 'Wildteam' in filename or 'WildteamAttack' in filename:
            attack_type = 'WildteamAttack'
        elif 'jailbroken' in filename:
            attack_type = 'Jailbroken'
        elif 'multi_model' in filename:
            attack_type = 'Multi-Model'
        
        # Special handling for WildteamAttack summary files
        if 'WildteamAttack_summary' in filename:
            return {
                'file': filepath,
                'filename': filename,
                'model': model,
                'attack_type': 'WildteamAttack',
                'dataset_size': data.get('dataset_size', 0),
                'successful': 0,  # Will be updated if we find actual results
                'success_rate': 0.0,  # Will be updated if we find actual results
                'timestamp': data.get('timestamp', 'Unknown'),
                'schema_type': data.get('schema_type', 'Unknown')
            }
            
        # Extract results data
        results = data.get('results', data.get('evaluation_results', []))
        if not isinstance(results, list):
            results = []
            
        dataset_size = len(results)
        
        # Calculate success rate
        successful = 0
        for result in results:
            if isinstance(result, dict):
                # Check various possible success indicators
                is_success = (
                    result.get('jailbroken', False) or
                    result.get('success', False) or
                    result.get('is_jailbroken', False) or
                    result.get('attack_success', False)
                )
                if is_success:
                    successful += 1
        
        success_rate = (successful / dataset_size * 100) if dataset_size > 0 else 0
        
        return {
            'file': filepath,
            'filename': filename,
            'model': model,
            'attack_type': attack_type,
            'dataset_size': dataset_size,
            'successful': successful,
            'success_rate': success_rate,
            'timestamp': data.get('timestamp', 'Unknown')
        }
    except Exception as e:
        print(f'Warning: Could not analyze JSON file {filepath}: {e}', file=sys.stderr)
        return None


def analyze_markdown_file(filepath):
    """Analyze a markdown report file and extract key metrics"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        filename = os.path.basename(filepath)
        
        # Extract model information from markdown content
        model = 'Unknown'
        attack_type = 'Multi-Model'
        success_rate = 0.0
        dataset_size = 0
        successful = 0
        total_runtime = 0
        
        lines = content.split('\n')
        
        # Parse the markdown content
        for i, line in enumerate(lines):
            # Look for executive summary metrics
            if '**Models Tested:**' in line:
                try:
                    dataset_size = int(line.split('**Models Tested:**')[1].strip())
                except:
                    pass
            elif '**Successfully Completed:**' in line:
                try:
                    successful = int(line.split('**Successfully Completed:**')[1].strip())
                except:
                    pass
            elif '**Total Runtime:**' in line:
                try:
                    runtime_str = line.split('**Total Runtime:**')[1].strip()
                    # Extract seconds from format like "98.3 seconds (1.6 minutes)"
                    if 'seconds' in runtime_str:
                        total_runtime = float(runtime_str.split('seconds')[0].strip())
                except:
                    pass
            
            # Look for model performance table
            elif '| Model |' in line and 'ASR' in line and i + 2 < len(lines):
                # Find the data row (skip header and separator)
                data_line = lines[i + 2].strip()
                if data_line.startswith('|') and data_line.endswith('|'):
                    parts = [p.strip() for p in data_line.split('|')[1:-1]]
                    if len(parts) >= 3:
                        model = parts[0]
                        # Parse ASR (Attack Success Rate) - look for percentage values
                        for j, part in enumerate(parts):
                            if '%' in part and j >= 2:  # ASR columns typically start from index 2
                                asr_str = part.replace('%', '').strip()
                                try:
                                    success_rate = float(asr_str)
                                    break
                                except:
                                    pass
            
            # Look for structure analysis table
            elif '| Structure |' in line and 'ASR' in line and i + 2 < len(lines):
                data_line = lines[i + 2].strip()
                if data_line.startswith('|') and data_line.endswith('|'):
                    parts = [p.strip() for p in data_line.split('|')[1:-1]]
                    if len(parts) >= 3:
                        # Structure analysis: Structure | Models Tested | Average ASR | Best ASR
                        try:
                            # Use structure table data if we don't have model performance data
                            if success_rate == 0.0:
                                asr_str = parts[2].replace('%', '').strip() if len(parts) > 2 else '0'
                                success_rate = float(asr_str)
                        except:
                            pass
        
        # If we have success rate and dataset size, calculate successful attacks
        if dataset_size > 0 and success_rate > 0:
            successful = max(successful, int((success_rate / 100.0) * dataset_size))
        elif success_rate > 0 and successful == 0:
            # If we don't have dataset size but have success rate, use successful count from summary
            if dataset_size == 0:
                dataset_size = 1
            if successful == 0:
                successful = 1
        
        # Extract timestamp from content
        timestamp = 'Unknown'
        for line in lines:
            if '**Generated:**' in line:
                timestamp = line.split('**Generated:**')[1].strip()
                break
        
        # Determine attack type from path or filename
        if 'multi_model' in filepath.lower():
            attack_type = 'Multi-Model'
        elif 'PAIR' in filepath:
            attack_type = 'PAIR'
        elif 'benchmark' in filepath.lower():
            attack_type = 'Benchmark'
        
        # Skip if no meaningful data found
        if success_rate == 0.0 and dataset_size == 0:
            return None
        
        result = {
            'file': filepath,
            'filename': filename,
            'model': model,
            'attack_type': attack_type,
            'dataset_size': dataset_size,
            'successful': successful,
            'success_rate': success_rate,
            'timestamp': timestamp
        }
        
        # Add runtime if available
        if total_runtime > 0:
            result['runtime'] = total_runtime
            
        return result
        
    except Exception as e:
        print(f'Warning: Could not analyze markdown file {filepath}: {e}', file=sys.stderr)
        return None


def analyze_log_file(filepath):
    """Analyze a log file and extract key metrics"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        filename = os.path.basename(filepath)
        
        # Skip server logs and non-attack logs
        if any(skip_pattern in filename.lower() for skip_pattern in ['server_', 'logger.log', 'pair_debug.log']):
            return None
        
        # Extract model name from filename or path
        model = 'Unknown'
        if 'llama-3.2-3b' in filepath:
            model = 'Llama-3.2-3B-Instruct'
        elif 'llama-3.1-70b' in filepath:
            model = 'Llama-3.1-70B-Instruct'
        elif 'llama' in filepath.lower():
            model = 'Llama'
            
        # Determine attack type from filename or path
        attack_type = 'Unknown'
        if 'PAIR' in filepath:
            attack_type = 'PAIR'
        elif 'Wildteam' in filepath or 'WildteamAttack' in filepath:
            attack_type = 'WildteamAttack'
        elif 'jailbroken' in filepath:
            attack_type = 'Jailbroken'
        elif 'multi_model' in filepath:
            attack_type = 'Multi-Model'
        
        # Parse log content for metrics
        lines = content.split('\n')
        
        # Look for attack report section
        total_queries = 0
        asr_query = 0.0
        efficiency = 0.0
        refusal_query = 0.0
        has_report = False
        
        # Count jailbreak attempts and successes from log lines
        jailbreak_attempts = 0
        jailbreak_successes = 0
        
        for line in lines:
            # Parse attack report if available
            if 'Total queries:' in line:
                try:
                    total_queries = int(line.split('Total queries:')[1].strip())
                    has_report = True
                except:
                    pass
            elif 'ASR Query:' in line:
                try:
                    asr_str = line.split('ASR Query:')[1].strip().replace('%', '')
                    asr_query = float(asr_str)
                    has_report = True
                except:
                    pass
            elif 'Efficiency (avg queries until success):' in line:
                try:
                    efficiency = float(line.split('Efficiency (avg queries until success):')[1].strip())
                except:
                    pass
            elif 'Refusal Query:' in line:
                try:
                    refusal_str = line.split('Refusal Query:')[1].strip().replace('%', '')
                    refusal_query = float(refusal_str)
                except:
                    pass
            
            # Count jailbreak attempts and successes from individual log entries
            if 'Jailbreak Prompt:' in line:
                jailbreak_attempts += 1
            elif 'Found a jailbreak. Exiting.' in line:
                jailbreak_successes += 1
        
        # If no attack report, use counted values
        if not has_report and jailbreak_attempts > 0:
            total_queries = jailbreak_attempts
            asr_query = (jailbreak_successes / jailbreak_attempts * 100) if jailbreak_attempts > 0 else 0
            efficiency = 1.0 if jailbreak_successes > 0 else 0.0
        
        # Skip logs with no meaningful data
        if total_queries == 0 and jailbreak_attempts == 0:
            return None
        
        # Calculate successful attacks
        successful = int((asr_query / 100.0) * total_queries) if total_queries > 0 else jailbreak_successes
        
        # Extract timestamp from filename or content
        timestamp = 'Unknown'
        if 'attack_' in filename:
            try:
                timestamp_part = filename.split('attack_')[1].split('.log')[0]
                timestamp = timestamp_part.replace('_', ' ').replace('-', '/')
            except:
                pass
        
        return {
            'file': filepath,
            'filename': filename,
            'model': model,
            'attack_type': attack_type,
            'dataset_size': total_queries or jailbreak_attempts,
            'successful': successful,
            'success_rate': asr_query,
            'timestamp': timestamp,
            'efficiency': efficiency,
            'refusal_rate': refusal_query
        }
        
    except Exception as e:
        print(f'Warning: Could not analyze log file {filepath}: {e}', file=sys.stderr)
        return None


def find_result_files(search_dir=None):
    """Find all result files in the current directory or specified directory"""
    if search_dir:
        # If a specific directory is provided, search recursively in that directory
        if not os.path.exists(search_dir):
            print(f'Error: Directory {search_dir} does not exist')
            return []
        
        if not os.path.isdir(search_dir):
            print(f'Error: {search_dir} is not a directory')
            return []
        
        print(f'🔍 Searching recursively in directory: {search_dir}')
        result_files = []
        
        # Search recursively for all supported file types
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Check if file matches our supported formats
                if (file.endswith('.json') or 
                    file.endswith('.jsonl') or 
                    file.endswith('.log') or 
                    file.endswith('.md')):
                    # Skip server logs and debug logs
                    if not any(skip_pattern in file.lower() for skip_pattern in 
                              ['server_', 'logger.log', 'pair_debug.log', 'target_server_']):
                        result_files.append(file_path)
        
        # Sort by modification time (most recent first)
        result_files = sorted(result_files, key=os.path.getmtime, reverse=True)
        return result_files
    
    else:
        # Original behavior - search in current directory
        patterns = [
            '*results*.json', 
            'PAIR_*.json', 
            'WildteamAttack_*.json', 
            'jailbroken_*.json', 
            'multi_model_*.json',
            '*.log',  # Log files
            '*.md'    # Markdown reports
        ]
        
        result_files = []
        for pattern in patterns:
            result_files.extend(glob.glob(pattern))
        
        # Also search in subdirectories for log files and reports
        search_dirs = ['logs', 'PAIR-*-results', '*-results', 'logs/*']
        for search_pattern in search_dirs:
            for directory in glob.glob(search_pattern):
                if os.path.isdir(directory):
                    result_files.extend(glob.glob(os.path.join(directory, '*.log')))
                    result_files.extend(glob.glob(os.path.join(directory, '*.json')))
                    result_files.extend(glob.glob(os.path.join(directory, '*.md')))
                    result_files.extend(glob.glob(os.path.join(directory, '*/*.log')))
                    result_files.extend(glob.glob(os.path.join(directory, '*/*.json')))
                    result_files.extend(glob.glob(os.path.join(directory, '*/*.md')))
        
        # Remove duplicates and sort by modification time
        result_files = sorted(set(result_files), key=os.path.getmtime, reverse=True)
        
        return result_files[:30]  # Limit to 30 most recent files


def display_summary_table(results):
    """Display a formatted summary table"""
    print()
    print('📊 Benchmark Results Summary')
    print('=' * 90)
    print(f"{'Attack Type':<18} {'Model':<20} {'Success Rate':<13} {'Successful':<11} {'Total':<8} {'Efficiency':<10} {'File':<15}")
    print('-' * 90)
    
    for r in results:
        model_display = r['model'][:19] if len(r['model']) > 19 else r['model']
        filename_display = r['filename'][:14] if len(r['filename']) > 14 else r['filename']
        
        # Format efficiency
        efficiency_display = ''
        if 'efficiency' in r and r['efficiency'] > 0:
            efficiency_display = f"{r['efficiency']:.2f}"
        
        print(f"{r['attack_type']:<18} {model_display:<20} "
              f"{r['success_rate']:>6.1f}%      {r['successful']:>6}      "
              f"{r['dataset_size']:>5}   {efficiency_display:<10} {filename_display:<15}")
    
    print('-' * 90)
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def save_markdown_report(results, output_file='benchmark_summary.md'):
    """Save detailed markdown report"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('# Benchmark Results Summary\n\n')
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary statistics
            total_attacks = len(results)
            avg_success_rate = sum(r['success_rate'] for r in results) / total_attacks if total_attacks > 0 else 0
            
            f.write('## Summary Statistics\n\n')
            f.write(f'- **Total Attack Results**: {total_attacks}\n')
            f.write(f'- **Average Success Rate**: {avg_success_rate:.1f}%\n')
            f.write(f'- **Attack Types**: {len(set(r["attack_type"] for r in results))}\n')
            f.write(f'- **Models Tested**: {len(set(r["model"] for r in results))}\n\n')
            
            # Summary table
            f.write('## Summary Table\n\n')
            f.write('| Attack Type | Model | Success Rate | Successful | Total | Efficiency | Filename |\n')
            f.write('|-------------|-------|--------------|------------|-------|------------|----------|\n')
            
            for r in results:
                efficiency_str = f"{r.get('efficiency', 0):.2f}" if r.get('efficiency', 0) > 0 else 'N/A'
                f.write(f"| {r['attack_type']} | {r['model']} | {r['success_rate']:.1f}% | "
                       f"{r['successful']} | {r['dataset_size']} | {efficiency_str} | {r['filename']} |\n")
            
            # Detailed results
            f.write('\n## Detailed Results\n\n')
            for r in results:
                f.write(f"### {r['attack_type']} - {r['model']}\n\n")
                f.write(f"- **File**: `{r['file']}`\n")
                f.write(f"- **Success Rate**: {r['success_rate']:.1f}% ({r['successful']}/{r['dataset_size']})\n")
                if r.get('efficiency', 0) > 0:
                    f.write(f"- **Efficiency**: {r['efficiency']:.2f} avg queries until success\n")
                if r.get('refusal_rate', 0) > 0:
                    f.write(f"- **Refusal Rate**: {r['refusal_rate']:.1f}%\n")
                if r['timestamp'] != 'Unknown':
                    f.write(f"- **Timestamp**: {r['timestamp']}\n")
                f.write('\n')
            
            # Result files list
            f.write('\n## Result Files\n\n')
            for r in results:
                f.write(f"- `{r['file']}`\n")
        
        print()
        print(f'✅ Detailed summary saved to: {output_file}')
        
    except Exception as e:
        print(f'Warning: Could not save markdown summary: {e}', file=sys.stderr)


def main():
    """Main analysis function"""
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Analyze benchmark result files (JSON, log, markdown)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_benchmark_results.py                    # Auto-discover files in current directory
  python analyze_benchmark_results.py file1.json file2.log  # Analyze specific files
  python analyze_benchmark_results.py --dir logs/        # Analyze all files in logs/ directory
  python analyze_benchmark_results.py -d /path/to/results   # Analyze all files in specified directory
        """
    )
    
    parser.add_argument('files', nargs='*', 
                       help='Specific files to analyze (JSON, log, or markdown)')
    parser.add_argument('-d', '--dir', '--directory', 
                       help='Directory to search recursively for result files')
    parser.add_argument('--output', '-o', default='benchmark_summary.md',
                       help='Output markdown file (default: benchmark_summary.md)')
    parser.add_argument('--limit', type=int, default=50,
                       help='Maximum number of files to analyze (default: 50)')
    
    args = parser.parse_args()
    
    # Determine which files to analyze
    result_files = []
    
    if args.files:
        # Specific files provided
        result_files = args.files
        print(f'📁 Analyzing {len(result_files)} specified files')
    elif args.dir:
        # Directory specified
        result_files = find_result_files(args.dir)
        if result_files:
            # Apply limit
            result_files = result_files[:args.limit]
            print(f'📁 Found {len(result_files)} result files in {args.dir}')
        else:
            print(f'No result files found in directory: {args.dir}')
            return 1
    else:
        # Auto-discovery in current directory
        result_files = find_result_files()
        if result_files:
            result_files = result_files[:args.limit]
            print(f'📁 Found {len(result_files)} result files in current directory')
        else:
            print('No result files found in current directory')
            return 1
    
    if not result_files:
        print('No result files found to analyze')
        print()
        print('Usage examples:')
        print('  python analyze_benchmark_results.py                    # Auto-discover files')
        print('  python analyze_benchmark_results.py file1.json file2.log  # Specific files')
        print('  python analyze_benchmark_results.py --dir logs/        # Analyze directory')
        return 1
    
    # Show files being analyzed if more than a few
    if len(result_files) > 5:
        print('📋 Sample files being analyzed:')
        for i, filepath in enumerate(result_files[:5]):
            print(f'  {i+1}. {os.path.relpath(filepath)}')
        if len(result_files) > 5:
            print(f'  ... and {len(result_files) - 5} more files')
    else:
        print('📋 Files being analyzed:')
        for i, filepath in enumerate(result_files):
            print(f'  {i+1}. {os.path.relpath(filepath)}')
    print()
    
    # Analyze all files
    results = []
    analyzed_count = 0
    skipped_count = 0
    
    for filepath in result_files:
        result = analyze_file(filepath)
        if result:
            results.append(result)
            analyzed_count += 1
        else:
            skipped_count += 1
    
    print(f'✅ Successfully analyzed: {analyzed_count} files')
    if skipped_count > 0:
        print(f'⚠️  Skipped: {skipped_count} files (invalid format or no data)')
    
    if not results:
        print('No valid results found to analyze')
        return 1
    
    # Sort by attack type, then by success rate (descending)
    results.sort(key=lambda x: (x['attack_type'], -x['success_rate']))
    
    # Display summary
    display_summary_table(results)
    
    # Save markdown report
    save_markdown_report(results, args.output)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
