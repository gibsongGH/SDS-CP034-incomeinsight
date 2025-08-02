#!/usr/bin/env python3
"""
Quick Misclassification Check
"""

import os
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set working directory
os.chdir(r"C:\Users\arttu\SDS-CP034-incomeinsight\submissions\team-members\art-turner")

def quick_analysis():
    """Quick analysis to check if models and data are accessible."""
    print("QUICK MISCLASSIFICATION CHECK")
    print("="*40)
    
    try:
        # Load test data
        X_num_test = np.load('deep_learning_data/X_numerical_test.npy')
        X_cat_test = np.load('deep_learning_data/X_categorical_test.npy')
        y_test = np.load('deep_learning_data/y_test.npy')
        
        print(f"✓ Test data loaded: {X_num_test.shape[0]} samples")
        
        # Check if we can load comparison results
        comparison_df = pd.read_csv('models/neural_network_comparison.csv', index_col=0)
        print(f"✓ Model comparison results loaded")
        print(comparison_df)
        
        # Basic misclassification simulation
        # Using the known performance metrics to estimate disagreement
        nn_accuracy = comparison_df.loc['Neural Network', 'accuracy']
        lgb_accuracy = comparison_df.loc['LightGBM', 'accuracy']
        
        total_samples = len(y_test)
        nn_errors = int(total_samples * (1 - nn_accuracy))
        lgb_errors = int(total_samples * (1 - lgb_accuracy))
        
        print(f"\nEstimated Error Analysis:")
        print(f"  Total test samples: {total_samples:,}")
        print(f"  Neural Network errors: ~{nn_errors:,}")
        print(f"  LightGBM errors: ~{lgb_errors:,}")
        print(f"  LightGBM advantage: ~{nn_errors - lgb_errors:,} fewer errors")
        
        # Key insight: Where does LightGBM excel?
        print(f"\nKey Findings (Estimated):")
        print(f"  - LightGBM makes ~{nn_errors - lgb_errors:,} fewer errors than Neural Network")
        print(f"  - This represents a {((nn_errors - lgb_errors)/total_samples)*100:.1f}% advantage")
        print(f"  - Primary advantage areas likely: structured feature interactions")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def analyze_model_strengths():
    """Analyze what each model is good at based on architecture."""
    print(f"\nMODEL STRENGTH ANALYSIS")
    print("="*40)
    
    print("LightGBM Advantages:")
    print("  ✓ Native categorical feature handling")
    print("  ✓ Tree-based feature interactions")
    print("  ✓ Robust to feature scaling")
    print("  ✓ Built-in feature importance")
    print("  ✓ Less prone to overfitting on tabular data")
    
    print("\nNeural Network Advantages:")
    print("  ✓ Learned feature representations (embeddings)")
    print("  ✓ Non-linear activation functions")
    print("  ✓ Flexible architecture")
    print("  ✓ Can capture complex patterns with sufficient data")
    
    print("\nLikely Misclassification Patterns:")
    print("  • LightGBM excels: Cases with clear decision boundaries")
    print("  • Neural Network struggles: Complex categorical interactions")
    print("  • Borderline cases: Both models show uncertainty")

def main():
    if quick_analysis():
        analyze_model_strengths()
        print(f"\n✓ Quick analysis completed!")
        print(f"For detailed analysis, the full misclassification_analysis.py")
        print(f"script is ready to run when computational resources allow.")

if __name__ == "__main__":
    main()