#!/usr/bin/env python3
"""
Final ML Model Training and Comparison
====================================

Complete training and evaluation of three classification models:
- Logistic Regression
- Random Forest  
- XGBoost

Includes comprehensive metrics, visualizations, and model comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import json
import time

# Machine Learning imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve, precision_recall_curve
)
import xgboost as xgb
import joblib

def main():
    """Main execution function"""
    print("ADULT INCOME CLASSIFICATION - FINAL MODEL TRAINING")
    print("="*60)
    
    # Create directories
    Path('models').mkdir(exist_ok=True)
    
    # Load data
    print("\nLoading preprocessed data...")
    X_train = np.load('data/X_train.npy')
    X_test = np.load('data/X_test.npy') 
    y_train = np.load('data/y_train.npy')
    y_test = np.load('data/y_test.npy')
    
    with open('data/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    print(f"SUCCESS: Data loaded")
    print(f"  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Class distribution: {np.bincount(y_train)}")
    
    # Initialize results storage
    models = {}
    results = {}
    
    # 1. LOGISTIC REGRESSION
    print("\n" + "="*50)
    print("1. TRAINING LOGISTIC REGRESSION")
    print("="*50)
    
    start_time = time.time()
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver='liblinear')
    lr.fit(X_train, y_train)
    lr_time = time.time() - start_time
    
    # Evaluate
    lr_pred = lr.predict(X_test)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    
    lr_metrics = {
        'accuracy': accuracy_score(y_test, lr_pred),
        'precision': precision_score(y_test, lr_pred),
        'recall': recall_score(y_test, lr_pred),
        'f1_score': f1_score(y_test, lr_pred),
        'roc_auc': roc_auc_score(y_test, lr_proba)
    }
    
    print(f"Training time: {lr_time:.2f} seconds")
    print("Performance metrics:")
    for metric, value in lr_metrics.items():
        print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
    
    models['Logistic Regression'] = lr
    results['Logistic Regression'] = {
        'metrics': lr_metrics,
        'predictions': lr_pred,
        'probabilities': lr_proba,
        'training_time': lr_time
    }
    
    # Save model
    joblib.dump(lr, 'models/logistic_regression_final.pkl')
    print("SUCCESS: Model saved")
    
    # 2. RANDOM FOREST
    print("\n" + "="*50)
    print("2. TRAINING RANDOM FOREST")
    print("="*50)
    
    start_time = time.time()
    rf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=20, 
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42, 
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_time = time.time() - start_time
    
    # Evaluate
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    
    rf_metrics = {
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred),
        'recall': recall_score(y_test, rf_pred),
        'f1_score': f1_score(y_test, rf_pred),
        'roc_auc': roc_auc_score(y_test, rf_proba)
    }
    
    print(f"Training time: {rf_time:.2f} seconds")
    print("Performance metrics:")
    for metric, value in rf_metrics.items():
        print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
    
    models['Random Forest'] = rf
    results['Random Forest'] = {
        'metrics': rf_metrics,
        'predictions': rf_pred,
        'probabilities': rf_proba,
        'training_time': rf_time
    }
    
    # Save model
    joblib.dump(rf, 'models/random_forest_final.pkl')
    print("SUCCESS: Model saved")
    
    # 3. XGBOOST
    print("\n" + "="*50)
    print("3. TRAINING XGBOOST")
    print("="*50)
    
    start_time = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    xgb_time = time.time() - start_time
    
    # Evaluate
    xgb_pred = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    xgb_metrics = {
        'accuracy': accuracy_score(y_test, xgb_pred),
        'precision': precision_score(y_test, xgb_pred),
        'recall': recall_score(y_test, xgb_pred),
        'f1_score': f1_score(y_test, xgb_pred),
        'roc_auc': roc_auc_score(y_test, xgb_proba)
    }
    
    print(f"Training time: {xgb_time:.2f} seconds")
    print("Performance metrics:")
    for metric, value in xgb_metrics.items():
        print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
    
    models['XGBoost'] = xgb_model
    results['XGBoost'] = {
        'metrics': xgb_metrics,
        'predictions': xgb_pred,
        'probabilities': xgb_proba,
        'training_time': xgb_time
    }
    
    # Save model
    joblib.dump(xgb_model, 'models/xgboost_final.pkl')
    print("SUCCESS: Model saved")
    
    # CREATE COMPREHENSIVE COMPARISON
    print("\n" + "="*60)
    print("MODEL COMPARISON AND ANALYSIS")
    print("="*60)
    
    # Create comparison DataFrame
    comparison_data = []
    for name, data in results.items():
        row = [name] + list(data['metrics'].values()) + [data['training_time']]
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(
        comparison_data,
        columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Training Time (s)']
    )
    
    # Sort by ROC-AUC (best for imbalanced classification)
    comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)
    
    print("\nFINAL RESULTS (Ranked by ROC-AUC):")
    print(comparison_df.round(4).to_string(index=False))
    
    # Identify best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_roc_auc = comparison_df.iloc[0]['ROC-AUC']
    
    print(f"\nBEST MODEL: {best_model_name}")
    print(f"ROC-AUC Score: {best_roc_auc:.4f}")
    
    # CREATE VISUALIZATIONS
    print("\nCreating comparison visualizations...")
    
    # Set up the plot
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Adult Income Classification - Model Comparison Results', fontsize=16, fontweight='bold')
    
    # 1. Metrics comparison bar chart
    metrics_df = comparison_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]
    ax1 = axes[0, 0]
    metrics_df.plot(kind='bar', ax=ax1, rot=45)
    ax1.set_title('Performance Metrics Comparison')
    ax1.set_ylabel('Score')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # 2. ROC curves
    ax2 = axes[0, 1]
    for name, data in results.items():
        fpr, tpr, _ = roc_curve(y_test, data['probabilities'])
        auc_score = data['metrics']['roc_auc']
        ax2.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)
    
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Confusion matrix for best model
    ax3 = axes[1, 0]
    cm = confusion_matrix(y_test, results[best_model_name]['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, 
                xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
    ax3.set_title(f'{best_model_name} Confusion Matrix')
    ax3.set_ylabel('True Label')
    ax3.set_xlabel('Predicted Label')
    
    # 4. Training time comparison
    ax4 = axes[1, 1]
    times = [data['training_time'] for data in results.values()]
    names = list(results.keys())
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    bars = ax4.bar(names, times, color=colors)
    ax4.set_title('Training Time Comparison')
    ax4.set_ylabel('Time (seconds)')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add time labels on bars
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{time_val:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('models/final_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # SAVE DETAILED RESULTS
    print("\nSaving detailed results...")
    
    # Save comparison table
    comparison_df.to_csv('models/final_model_comparison.csv', index=False)
    
    # Save detailed results as JSON
    final_results = {}
    for name, data in results.items():
        final_results[name] = {
            'metrics': {k: float(v) for k, v in data['metrics'].items()},
            'training_time': float(data['training_time'])
        }
    
    with open('models/final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Create detailed classification reports
    print("\nDetailed Classification Reports:")
    for name, data in results.items():
        print(f"\n{name}:")
        print(classification_report(y_test, data['predictions'], 
                                  target_names=['<=50K', '>50K']))
    
    # FINAL SUMMARY
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print(f"SUCCESS: Trained and evaluated 3 models")
    print(f"SUCCESS: Best performing model: {best_model_name}")
    print(f"SUCCESS: Best ROC-AUC score: {best_roc_auc:.4f}")
    print(f"SUCCESS: All models saved to models/ directory")
    print(f"SUCCESS: Comparison visualizations created")
    print(f"SUCCESS: Detailed results exported")
    
    total_time = sum(data['training_time'] for data in results.values())
    print(f"\nTotal training time: {total_time:.2f} seconds")
    
    print(f"\nFiles created:")
    print(f"  - models/logistic_regression_final.pkl")
    print(f"  - models/random_forest_final.pkl") 
    print(f"  - models/xgboost_final.pkl")
    print(f"  - models/final_model_comparison.csv")
    print(f"  - models/final_model_comparison.png")
    print(f"  - models/final_results.json")

if __name__ == "__main__":
    main()