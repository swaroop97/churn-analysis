#!/usr/bin/env python3
"""
Test script to identify errors in the churn analysis system.
"""

import sys
import traceback

def test_imports():
    """Test all imports."""
    print("Testing imports...")
    
    try:
        import pandas as pd
        print("✓ pandas")
    except Exception as e:
        print(f"✗ pandas: {e}")
    
    try:
        import numpy as np
        print("✓ numpy")
    except Exception as e:
        print(f"✗ numpy: {e}")
    
    try:
        import sklearn
        print("✓ sklearn")
    except Exception as e:
        print(f"✗ sklearn: {e}")
    
    try:
        import matplotlib
        print("✓ matplotlib")
    except Exception as e:
        print(f"✗ matplotlib: {e}")
    
    try:
        import seaborn
        print("✓ seaborn")
    except Exception as e:
        print(f"✗ seaborn: {e}")

def test_data_generator():
    """Test data generator."""
    print("\nTesting data generator...")
    try:
        sys.path.append('src')
        from data.data_generator import CustomerDataGenerator
        generator = CustomerDataGenerator()
        df = generator.generate_complete_dataset(n_customers=100)
        print(f"✓ Data generator works - Shape: {df.shape}, Churn rate: {df['churned'].mean():.2%}")
        return True
    except Exception as e:
        print(f"✗ Data generator error: {e}")
        traceback.print_exc()
        return False

def test_preprocessing():
    """Test preprocessing."""
    print("\nTesting preprocessing...")
    try:
        from data.preprocessing import ChurnDataPreprocessor
        preprocessor = ChurnDataPreprocessor()
        print("✓ Preprocessor imported")
        return True
    except Exception as e:
        print(f"✗ Preprocessing error: {e}")
        traceback.print_exc()
        return False

def test_model_trainer():
    """Test model trainer."""
    print("\nTesting model trainer...")
    try:
        from models.model_trainer import ChurnModelTrainer
        trainer = ChurnModelTrainer()
        models = trainer.get_models()
        print(f"✓ Model trainer works - Available models: {list(models.keys())}")
        return True
    except Exception as e:
        print(f"✗ Model trainer error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("CHURN ANALYSIS ERROR DIAGNOSTICS")
    print("=" * 50)
    
    test_imports()
    test_data_generator()
    test_preprocessing()
    test_model_trainer()
    
    print("\n" + "=" * 50)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()


