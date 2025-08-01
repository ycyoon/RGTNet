#!/usr/bin/env python3
"""
Clean incomplete benchmark logs that don't contain proper performance values
"""

import os
import shutil
import json
import re
from pathlib import Path

def check_multi_model_benchmark_log(log_dir):
    """Check if multi_model_benchmark log contains valid performance results"""
    try:
        # Required files for a complete benchmark
        required_files = [
            "analysis_report.md",
            "multi_model_summary_*.json",
            "model_comparison_summary.csv"
        ]
        
        found_files = []
        for pattern in required_files:
            if "*" in pattern:
                # Handle wildcard patterns
                matches = list(log_dir.glob(pattern))
                if matches:
                    found_files.append(pattern)
            else:
                if (log_dir / pattern).exists():
                    found_files.append(pattern)
        
        # Check if we have the essential files
        has_analysis = any("analysis_report" in f for f in found_files)
        has_summary = any("multi_model_summary" in f for f in found_files)
        
        if not (has_analysis and has_summary):
            return False, f"Missing essential files: has_analysis={has_analysis}, has_summary={has_summary}"
        
        # Check if analysis_report contains valid ASR values (not 0%)
        analysis_file = log_dir / "analysis_report.md"
        if analysis_file.exists():
            content = analysis_file.read_text()
            if "ASR" in content and ("%" in content):
                # Look for actual percentage values
                asr_matches = re.findall(r'(\d+\.?\d*)%', content)
                if asr_matches:
                    # Check if all ASR values are 0 (likely error)
                    non_zero_asr = [float(asr) for asr in asr_matches if float(asr) > 0]
                    if non_zero_asr:
                        return True, f"Found valid ASR values: {asr_matches[:3]}..."
                    else:
                        return False, f"All ASR values are 0% (likely error): {asr_matches[:3]}..."
        
        return False, "No ASR values found in analysis report"
        
    except Exception as e:
        return False, f"Error checking log: {e}"

def check_attack_log(log_dir, attack_type):
    """Check if attack log (PAIR, WildteamAttack, etc.) contains valid results"""
    try:
        # Look for log files and result files
        log_files = []
        result_files = []
        
        # Find all .log files
        log_files.extend(log_dir.rglob("*.log"))
        
        # Find result files
        result_files.extend(log_dir.rglob("*.jsonl"))
        result_files.extend(log_dir.rglob("*summary*.json"))
        result_files.extend(log_dir.rglob("*result*.json"))
        
        if not log_files:
            return False, "No log files found"
        
        # Check log content for completion indicators
        completion_indicators = [
            "Attack Success Rate",
            "ASR:",
            "completed successfully",
            "attack completed",
            "final results",
            "success rate"
        ]
        
        has_performance_data = False
        log_content_summary = []
        
        for log_file in log_files:
            try:
                content = log_file.read_text().lower()
                log_content_summary.append(f"{log_file.name}: {len(content)} chars")
                
                for indicator in completion_indicators:
                    if indicator.lower() in content:
                        has_performance_data = True
                        break
                
                # Look for percentage values (but check if they're not all 0%)
                percentage_matches = re.findall(r'(\d+\.?\d*)%', content)
                if percentage_matches:
                    # Check if there are non-zero percentages
                    non_zero_percentages = [float(p) for p in percentage_matches if float(p) > 0]
                    if non_zero_percentages:
                        has_performance_data = True
                
            except Exception as e:
                log_content_summary.append(f"{log_file.name}: error reading - {e}")
        
        if has_performance_data:
            return True, f"Found performance data in logs: {'; '.join(log_content_summary[:2])}"
        
        # Check if there are result files with actual data
        for result_file in result_files:
            if result_file.stat().st_size > 100:  # Not empty
                return True, f"Found result file: {result_file.name} ({result_file.stat().st_size} bytes)"
        
        return False, f"No performance data found. Logs: {'; '.join(log_content_summary)}"
        
    except Exception as e:
        return False, f"Error checking attack log: {e}"

def clean_incomplete_logs(logs_dir):
    """Clean logs that don't contain proper performance values"""
    
    print(f"🔍 Scanning logs directory: {logs_dir}")
    
    if not logs_dir.exists():
        print(f"❌ Logs directory not found: {logs_dir}")
        return
    
    # Get all log directories
    log_dirs = [d for d in logs_dir.iterdir() if d.is_dir()]
    
    incomplete_logs = []
    complete_logs = []
    
    for log_dir in log_dirs:
        dir_name = log_dir.name
        
        print(f"\n📂 Checking: {dir_name}")
        
        # Determine log type and check accordingly
        if dir_name.startswith("multi_model_benchmark"):
            is_complete, reason = check_multi_model_benchmark_log(log_dir)
        elif any(attack in dir_name.lower() for attack in ["wildteam", "pair", "jailbroken"]):
            attack_type = "unknown"
            if "wildteam" in dir_name.lower():
                attack_type = "WildteamAttack"
            elif "pair" in dir_name.lower():
                attack_type = "PAIR"
            elif "jailbroken" in dir_name.lower():
                attack_type = "Jailbroken"
            
            is_complete, reason = check_attack_log(log_dir, attack_type)
        else:
            # Unknown log type, check for any performance indicators
            is_complete, reason = check_attack_log(log_dir, "unknown")
        
        if is_complete:
            complete_logs.append((log_dir, reason))
            print(f"   ✅ Complete: {reason}")
        else:
            incomplete_logs.append((log_dir, reason))
            print(f"   ❌ Incomplete: {reason}")
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"   • Total directories: {len(log_dirs)}")
    print(f"   • Complete logs: {len(complete_logs)}")
    print(f"   • Incomplete logs: {len(incomplete_logs)}")
    
    if incomplete_logs:
        print(f"\n🗑️  Incomplete logs to be deleted:")
        total_size = 0
        for log_dir, reason in incomplete_logs:
            # Calculate directory size
            dir_size = sum(f.stat().st_size for f in log_dir.rglob('*') if f.is_file())
            total_size += dir_size
            print(f"   • {log_dir.name} ({dir_size/1024/1024:.1f} MB) - {reason}")
        
        print(f"\n💾 Total size to be freed: {total_size/1024/1024:.1f} MB")
        
        # Ask for confirmation
        response = input(f"\n❓ Delete {len(incomplete_logs)} incomplete log directories? (y/N): ")
        
        if response.lower() in ['y', 'yes']:
            deleted_count = 0
            for log_dir, reason in incomplete_logs:
                try:
                    shutil.rmtree(log_dir)
                    print(f"   🗑️  Deleted: {log_dir.name}")
                    deleted_count += 1
                except Exception as e:
                    print(f"   ❌ Failed to delete {log_dir.name}: {e}")
            
            print(f"\n✅ Successfully deleted {deleted_count}/{len(incomplete_logs)} incomplete logs")
            print(f"💾 Freed approximately {total_size/1024/1024:.1f} MB of disk space")
        else:
            print("❌ Deletion cancelled")
    else:
        print("✅ All logs appear to be complete!")

if __name__ == "__main__":
    logs_dir = Path("/home/ycyoon/work/RGTNet/StructTransformBench/examples/logs")
    clean_incomplete_logs(logs_dir)