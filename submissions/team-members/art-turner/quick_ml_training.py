#!/usr/bin/env python3
"""
Quick ML Model Training and Comparison
====================================

Fast training of three models with basic hyperparameter tuning.
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
    classification_report, confusion_matrix, roc_curve
)
import xgboost as xgb
import joblib

def load_data():
    """Load preprocessed data"""
    print("Loading data...")
    X_train = np.load('data/X_train.npy')
    X_test = np.load('data/X_test.npy') 
    y_train = np.load('data/y_train.npy')
    y_test = np.load('data/y_test.npy')
    
    with open('data/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Features: {len(feature_names)}")
    return X_train, X_test, y_train, y_test, feature_names

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"\n{model_name} Results:")
    for metric, value in metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    
    return metrics, y_pred, y_pred_proba

def train_models(X_train, y_train, X_test, y_test):
    """Train all three models"""
    models = {}
    results = {}
    
    print("\n" + "="*50)
    print("TRAINING MODELS")
    print("="*50)
    
    # 1. Logistic Regression
    print("\n1. Training Logistic Regression...")
    start_time = time.time()
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_time = time.time() - start_time
    
    lr_metrics, lr_pred, lr_proba = evaluate_model(lr, X_test, y_test, "Logistic Regression")
    models['Logistic Regression'] = lr
    results['Logistic Regression'] = {
        'metrics': lr_metrics,
        'y_pred': lr_pred,
        'y_pred_proba': lr_proba,
        'training_time': lr_time
    }
    
    # 2. Random Forest
    print("\n2. Training Random Forest...")
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_time = time.time() - start_time
    
    rf_metrics, rf_pred, rf_proba = evaluate_model(rf, X_test, y_test, "Random Forest")
    models['Random Forest'] = rf
    results['Random Forest'] = {
        'metrics': rf_metrics,
        'y_pred': rf_pred,
        'y_pred_proba': rf_proba,
        'training_time': rf_time
    }
    
    # 3. XGBoost
    print("\n3. Training XGBoost...")
    start_time = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    xgb_model.fit(X_train, y_train)
    xgb_time = time.time() - start_time
    
    xgb_metrics, xgb_pred, xgb_proba = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    models['XGBoost'] = xgb_model
    results['XGBoost'] = {
        'metrics': xgb_metrics,
        'y_pred': xgb_pred,
        'y_pred_proba': xgb_proba,
        'training_time': xgb_time
    }
    
    return models, results

def create_visualizations(results, y_test):
    """Create comparison visualizations"""
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Comparison Results', fontsize=16, fontweight='bold')
    
    # 1. Metrics comparison
    metrics_df = pd.DataFrame({name: data['metrics'] for name, data in results.items()}).T
    ax1 = axes[0, 0]
    metrics_df.plot(kind='bar', ax=ax1, rot=45)
    ax1.set_title('Performance Metrics Comparison')
    ax1.set_ylabel('Score')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. ROC curves
    ax2 = axes[0, 1]
    for name, data in results.items():
        fpr, tpr, _ = roc_curve(y_test, data['y_pred_proba'])
        auc = data['metrics']['roc_auc']
        ax2.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
    
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Confusion matrix for best model
    best_model = max(results.keys(), key=lambda x: results[x]['metrics']['roc_auc'])
    ax3 = axes[1, 0]
    cm = confusion_matrix(y_test, results[best_model]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_title(f'{best_model} Confusion Matrix (Best)')
    ax3.set_ylabel('True Label')
    ax3.set_xlabel('Predicted Label')
    
    # 4. Training time comparison
    ax4 = axes[1, 1]
    times = [data['training_time'] for data in results.values()]
    names = list(results.keys())
    bars = ax4.bar(names, times, color=['lightcoral', 'lightblue', 'lightgreen'])
    ax4.set_title('Training Time Comparison')
    ax4.set_ylabel('Time (seconds)')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add time labels on bars
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{time_val:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('models/quick_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary(results):
    """Create summary report"""
    print("\n" + "="*50)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*50)
    
    # Create summary DataFrame
    summary_data = []
    for name, data in results.items():
        row = [name]
        row.extend([data['metrics'][metric] for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']])
        row.append(data['training_time'])
        summary_data.append(row)
    
    summary_df = pd.DataFrame(
        summary_data,
        columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Training Time (s)']
    )
    
    # Sort by ROC-AUC
    summary_df = summary_df.sort_values('ROC-AUC', ascending=False)
    
    print("\nRanked by ROC-AUC Score:")
    print(summary_df.round(4).to_string(index=False))
    
    # Save summary
    summary_df.to_csv('models/quick_model_summary.csv', index=False)
    
    # Best model
    best_model = summary_df.iloc[0]['Model']
    best_auc = summary_df.iloc[0]['ROC-AUC']
    
    print(f"\nBEST MODEL: {best_model} (ROC-AUC: {best_auc:.4f})")
    
    return summary_df

def main():
    """Main function"""
    print("QUICK ML MODEL TRAINING AND COMPARISON")
    print("="*60)
    
    # Create directories
    Path('models').mkdir(exist_ok=True)
    
    # Load data
    X_train, X_test, y_train, y_test, feature_names = load_data()
    
    # Train models
    models, results = train_models(X_train, y_train, X_test, y_test)
    
    # Save models
    print("\nSaving models...")
    for name, model in models.items():
        filename = name.lower().replace(' ', '_') + '_model.pkl'
        joblib.dump(model, f'models/{filename}')
        print(f"  Saved: {filename}")
    
    # Create visualizations
    create_visualizations(results, y_test)
    
    # Create summary
    summary_df = create_summary(results)
    
    # Save detailed results
    results_json = {}
    for name, data in results.items():
        results_json[name] = {
            'metrics': {k: float(v) for k, v in data['metrics'].items()},
            'training_time': float(data['training_time'])
        }
    
    with open('models/quick_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("All models trained and evaluated")
    print("Results saved to models/ directory")
    print("Visualizations created")
    
    total_time = sum(data['training_time'] for data in results.values())
    print(f"\nTotal training time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()