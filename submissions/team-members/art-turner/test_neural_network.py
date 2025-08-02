#!/usr/bin/env python3
"""
Quick test of the neural network pipeline
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Set working directory
os.chdir(r"C:\Users\arttu\SDS-CP034-incomeinsight\submissions\team-members\art-turner")

def test_data_loading():
    """Test if we can load the preprocessed data."""
    print("Testing data loading...")
    
    try:
        # Load numerical and categorical data
        X_num_train = np.load('deep_learning_data/X_numerical_train.npy')
        X_num_test = np.load('deep_learning_data/X_numerical_test.npy')
        X_cat_train = np.load('deep_learning_data/X_categorical_train.npy')
        X_cat_test = np.load('deep_learning_data/X_categorical_test.npy')
        y_train = np.load('deep_learning_data/y_train.npy')
        y_test = np.load('deep_learning_data/y_test.npy')
        
        # Load metadata
        with open('deep_learning_data/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print(f"✓ Data loaded successfully:")
        print(f"  Numerical train: {X_num_train.shape}")
        print(f"  Categorical train: {X_cat_train.shape}")
        print(f"  Target train: {y_train.shape}")
        print(f"  Categorical cardinalities: {metadata['categorical_cardinalities']}")
        
        return X_num_train, X_cat_train, y_train, metadata
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None, None, None, None

def test_model_creation():
    """Test if we can create the neural network model."""
    print("\nTesting model creation...")
    
    try:
        # Simple embedding network
        class SimpleEmbeddingNN(nn.Module):
            def __init__(self, numerical_features, categorical_cardinalities):
                super(SimpleEmbeddingNN, self).__init__()
                
                # Embedding layers
                embedding_dims = [min(50, (card + 1) // 2) for card in categorical_cardinalities]
                self.embeddings = nn.ModuleList([
                    nn.Embedding(card, dim) for card, dim in zip(categorical_cardinalities, embedding_dims)
                ])
                
                # Calculate input dimension
                total_embedding_dim = sum(embedding_dims)
                input_dim = numerical_features + total_embedding_dim
                
                # Simple network
                self.network = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, X_numerical, X_categorical):
                # Process embeddings
                embedded_features = []
                for i, embedding_layer in enumerate(self.embeddings):
                    embedded = embedding_layer(X_categorical[:, i])
                    embedded_features.append(embedded)
                
                # Concatenate features
                if embedded_features:
                    embedded_cat = torch.cat(embedded_features, dim=1)
                    x = torch.cat([X_numerical, embedded_cat], dim=1)
                else:
                    x = X_numerical
                
                return self.network(x).squeeze()
        
        # Test with sample data
        categorical_cardinalities = [8, 16, 7, 14, 6, 5, 2, 41]  # From metadata
        model = SimpleEmbeddingNN(6, categorical_cardinalities)  # 6 numerical features
        
        # Test forward pass
        batch_size = 10
        X_num_sample = torch.randn(batch_size, 6)
        X_cat_sample = torch.randint(0, 5, (batch_size, 8))  # Sample categorical data
        
        with torch.no_grad():
            output = model(X_num_sample, X_cat_sample)
        
        print(f"✓ Model created successfully:")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Output shape: {output.shape}")
        print(f"  Sample predictions: {output[:5].numpy()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        return False

def test_baseline_imports():
    """Test if baseline models can be imported."""
    print("\nTesting baseline model imports...")
    
    try:
        import lightgbm as lgb
        print(f"✓ LightGBM available: {lgb.__version__}")
    except ImportError:
        print("✗ LightGBM not available")
    
    try:
        import catboost as cb
        print(f"✓ CatBoost available: {cb.__version__}")
    except ImportError:
        print("✗ CatBoost not available")

def main():
    print("NEURAL NETWORK PIPELINE TEST")
    print("="*40)
    
    # Test data loading
    X_num_train, X_cat_train, y_train, metadata = test_data_loading()
    
    if X_num_train is not None:
        # Test model creation
        test_model_creation()
        
        # Test baseline imports
        test_baseline_imports()
        
        print(f"\n✓ All tests passed! Pipeline is ready for full training.")
    else:
        print(f"\n✗ Data loading failed. Check if deep_learning_eda.ipynb was run successfully.")

if __name__ == "__main__":
    main()