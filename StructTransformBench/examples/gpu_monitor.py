#!/usr/bin/env python3
"""
GPU Usage Monitor
=================

Simple script to monitor GPU usage during benchmark runs.
"""

import time
import subprocess
import sys
from datetime import datetime

def get_gpu_stats():
    """Get GPU utilization stats using nvidia-smi"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        stats = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split(', ')
            if len(parts) == 5:
                stats.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'gpu_util': float(parts[2]),
                    'mem_used': float(parts[3]),
                    'mem_total': float(parts[4])
                })
        return stats
    except:
        return None

def main():
    """Monitor GPU usage"""
    print("GPU Usage Monitor")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            stats = get_gpu_stats()
            if stats:
                # Clear screen
                print("\033[H\033[J", end='')
                print(f"GPU Usage Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 80)
                
                for gpu in stats:
                    mem_percent = (gpu['mem_used'] / gpu['mem_total']) * 100
                    
                    # Color coding
                    gpu_color = '\033[91m' if gpu['gpu_util'] < 10 else '\033[93m' if gpu['gpu_util'] < 50 else '\033[92m'
                    reset_color = '\033[0m'
                    
                    print(f"GPU {gpu['index']}: {gpu['name']}")
                    print(f"  Utilization: {gpu_color}{gpu['gpu_util']:5.1f}%{reset_color}")
                    print(f"  Memory: {gpu['mem_used']:6.0f} MB / {gpu['mem_total']:6.0f} MB ({mem_percent:5.1f}%)")
                    print()
                
                # Show optimization tips if low usage
                low_usage_gpus = [g for g in stats if g['gpu_util'] < 10]
                if low_usage_gpus:
                    print("\n⚠️  Low GPU usage detected! Tips:")
                    print("  • Increase batch size in config")
                    print("  • Use --models to test multiple models")
                    print("  • Check if models are actually on GPU")
                    print("  • Enable torch.compile if supported")
            else:
                print("Failed to get GPU stats. Is nvidia-smi available?")
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    main()
