#!/usr/bin/env python3
"""Simple test to identify KeyError issues"""

import argparse
import json
import os
import sys

# Test basic functionality
def test_basic_functionality():
    print("Testing basic Python functionality...")
    
    # Test dictionary access
    test_dict = {'input_ids': [1, 2, 3], 'role_mask': [True, False, True]}
    
    try:
        print(f"Accessing 'input_ids': {test_dict['input_ids']}")
        print(f"Accessing 'role_mask': {test_dict['role_mask']}")
        
        # Test accessing non-existent key
        try:
            print(f"Accessing key '0': {test_dict[0]}")
        except KeyError as e:
            print(f"KeyError caught as expected: {e}")
        
        # Test accessing with .get()
        print(f"Using .get() for key '0': {test_dict.get(0, 'default')}")
        
    except Exception as e:
        print(f"Error in basic dictionary test: {e}")
        return False
    
    return True

def test_imports():
    print("\nTesting imports...")
    
    try:
        import torch
        print("✓ PyTorch imported successfully")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import sklearn
        print("✓ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ Scikit-learn import failed: {e}")
        return False
    
    try:
        import transformers
        print("✓ Transformers imported successfully")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    return True

def test_file_operations():
    print("\nTesting file operations...")
    
    # Test creating a simple JSONL file
    test_data = [
        {'input_ids': [1, 2, 3], 'role_mask': [True, False, True], 'label': 0},
        {'input_ids': [4, 5, 6], 'role_mask': [False, True, False], 'label': 1}
    ]
    
    test_file = 'test_data.jsonl'
    
    try:
        # Write test data
        with open(test_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        print(f"✓ Successfully wrote test data to {test_file}")
        
        # Read test data
        loaded_data = []
        with open(test_file, 'r') as f:
            for line in f:
                loaded_data.append(json.loads(line))
        
        print(f"✓ Successfully loaded {len(loaded_data)} items from {test_file}")
        
        # Test accessing loaded data
        for i, item in enumerate(loaded_data):
            print(f"  Item {i}: input_ids={item['input_ids']}, label={item['label']}")
        
        # Clean up
        os.remove(test_file)
        print(f"✓ Cleaned up {test_file}")
        
    except Exception as e:
        print(f"✗ File operations failed: {e}")
        return False
    
    return True

if __name__ == '__main__':
    print("Running diagnostic tests...")
    
    success = True
    
    success &= test_basic_functionality()
    success &= test_imports()
    success &= test_file_operations()
    
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)
