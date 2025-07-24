#!/usr/bin/env python3
"""
VS Code GitHub Pro+ Setup Verification Script

This script helps verify that your Visual Studio Code is properly configured
to use GitHub Pro+ features with the RGTNet project.
"""

import json
import os
import sys
from pathlib import Path


def check_vscode_config():
    """Check if VS Code configuration files exist and are valid."""
    print("🔍 Checking VS Code configuration...")
    
    vscode_dir = Path(".vscode")
    if not vscode_dir.exists():
        print("❌ .vscode directory not found!")
        return False
    
    required_files = [
        "settings.json",
        "launch.json", 
        "tasks.json",
        "extensions.json"
    ]
    
    for file in required_files:
        file_path = vscode_dir / file
        if not file_path.exists():
            print(f"❌ {file} not found!")
            return False
        
        try:
            with open(file_path) as f:
                json.load(f)
            print(f"✅ {file} is valid")
        except json.JSONDecodeError as e:
            print(f"❌ {file} has invalid JSON: {e}")
            return False
    
    return True


def check_workspace_file():
    """Check if workspace file exists and is valid."""
    print("\n🔍 Checking workspace configuration...")
    
    workspace_file = Path("RGTNet.code-workspace")
    if not workspace_file.exists():
        print("❌ RGTNet.code-workspace not found!")
        return False
    
    try:
        with open(workspace_file) as f:
            config = json.load(f)
        
        if "folders" not in config:
            print("❌ Workspace file missing folders configuration!")
            return False
        
        if "settings" not in config:
            print("❌ Workspace file missing settings configuration!")
            return False
            
        print("✅ RGTNet.code-workspace is valid")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ RGTNet.code-workspace has invalid JSON: {e}")
        return False


def check_github_copilot_config():
    """Check if GitHub Copilot is properly configured."""
    print("\n🔍 Checking GitHub Copilot configuration...")
    
    settings_file = Path(".vscode/settings.json")
    try:
        with open(settings_file) as f:
            settings = json.load(f)
        
        copilot_enable = settings.get("github.copilot.enable", {})
        if not copilot_enable:
            print("❌ GitHub Copilot not enabled in settings!")
            return False
        
        if copilot_enable.get("python") is True:
            print("✅ GitHub Copilot enabled for Python")
        else:
            print("⚠️  GitHub Copilot not specifically enabled for Python")
            
        if "github.copilot.editor.enableAutoCompletions" in settings:
            print("✅ Copilot auto-completions configured")
        else:
            print("⚠️  Copilot auto-completions not configured")
            
        return True
        
    except Exception as e:
        print(f"❌ Error checking Copilot configuration: {e}")
        return False


def check_python_config():
    """Check Python interpreter configuration."""
    print("\n🔍 Checking Python configuration...")
    
    settings_file = Path(".vscode/settings.json")
    try:
        with open(settings_file) as f:
            settings = json.load(f)
        
        python_path = settings.get("python.defaultInterpreterPath")
        if python_path:
            print(f"✅ Python interpreter path configured: {python_path}")
        else:
            print("⚠️  No default Python interpreter path configured")
        
        if settings.get("python.linting.enabled") is True:
            print("✅ Python linting enabled")
        else:
            print("⚠️  Python linting not enabled")
            
        return True
        
    except Exception as e:
        print(f"❌ Error checking Python configuration: {e}")
        return False


def print_recommendations():
    """Print setup recommendations."""
    print("\n💡 Setup Recommendations:")
    print("1. Open this project in VS Code using: code RGTNet.code-workspace")
    print("2. Install recommended extensions when prompted")
    print("3. Sign in to GitHub: Ctrl+Shift+P → 'GitHub: Sign in'")
    print("4. Verify GitHub Pro+ subscription in GitHub settings")
    print("5. Check Copilot status in VS Code status bar")
    print("6. Select Python interpreter: F1 → 'Python: Select Interpreter'")


def main():
    """Main verification function."""
    print("🚀 RGTNet VS Code GitHub Pro+ Setup Verification")
    print("=" * 50)
    
    checks = [
        check_vscode_config(),
        check_workspace_file(),
        check_github_copilot_config(),
        check_python_config()
    ]
    
    if all(checks):
        print("\n🎉 All checks passed! Your VS Code is ready to use GitHub Pro+ features.")
        print("\nNext steps:")
        print("- Open the project: code RGTNet.code-workspace")
        print("- Start coding with Copilot assistance!")
    else:
        print("\n⚠️  Some issues found. Please check the configuration.")
        print_recommendations()
    
    print("\n📚 For more help, see the README.md file.")


if __name__ == "__main__":
    main()