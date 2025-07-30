#!/usr/bin/env python3
"""
Machine Learning Model Training and Comparison (Simplified Version)
================================================================

This script trains and compares multiple classification models for income prediction:
- Logistic Regression
- Random Forest  
- XGBoost

Provides comprehensive evaluation metrics and visualizations.
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
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, cross_val_score
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
        classification_report, confusion_matrix, roc_curve, precision_recall_curve
    )
    print("SUCCESS: Scikit-learn imported successfully")
except ImportError as e:
    print(f"ERROR: Error importing scikit-learn: {e}")
    exit(1)

try:
    import xgboost as xgb
    print("SUCCESS: XGBoost imported successfully")
except ImportError as e:
    print(f"ERROR: Error importing XGBoost: {e}")
    exit(1)

try:
    import joblib
    print("SUCCESS: Joblib imported successfully")
except ImportError as e:
    print(f"ERROR: Error importing joblib: {e}")
    exit(1)

def load_preprocessed_data():
    """Load the preprocessed data from the EDA notebook"""
    print("\n" + "="*60)
    print("LOADING PREPROCESSED DATA")
    print("="*60)
    
    try:
        # Load the data arrays
        X_train = np.load('data/X_train.npy')
        X_test = np.load('data/X_test.npy') 
        y_train = np.load('data/y_train.npy')
        y_test = np.load('data/y_test.npy')
        
        # Load feature names
        with open('data/feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        print(f"SUCCESS: Data loaded successfully:")
        print(f"  Training set: {X_train.shape}")
        print(f"  Test set: {X_test.shape}")
        print(f"  Features: {len(feature_names)}")
        print(f"  Class distribution in training:")
        print(f"    <=50K: {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%)")
        print(f"    >50K: {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")
        
        return X_train, X_test, y_train, y_test, feature_names
        
    except FileNotFoundError as e:
        print(f"ERROR: Error loading data: {e}")
        print("Please run the EDA notebook first to generate preprocessed data.")
        raise

def evaluate_model(model, X_test, y_test, model_name):
    """Comprehensive model evaluation"""
    print(f"\nEvaluating {model_name}...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"\n{model_name} Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    # Detailed classification report
    print(f"\nDetailed Classification Report for {model_name}:")
    print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))
    
    return metrics, y_pred, y_pred_proba

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train Logistic Regression with hyperparameter tuning"""
    print("\n" + "="*60)
    print("TRAINING LOGISTIC REGRESSION")
    print("="*60)
    
    start_time = time.time()
    
    # Simplified hyperparameter grid for faster execution
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['liblinear'],
        'max_iter': [1000]
    }
    
    # Grid search with cross-validation
    lr = LogisticRegression(random_state=42)
    grid_search = GridSearchCV(
        lr, param_grid, cv=3, scoring='roc_auc', 
        n_jobs=-1, verbose=1
    )
    
    print("Performing hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    training_time = time.time() - start_time
    
    print(f"SUCCESS: Training completed in {training_time:.2f} seconds")
    print(f"Best parameters: {best_params}")
    print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    metrics, y_pred, y_pred_proba = evaluate_model(best_model, X_test, y_test, "Logistic Regression")
    
    # Save model
    joblib.dump(best_model, 'models/logistic_regression_model.pkl')
    
    # Save results
    results = {
        'model_name': 'Logistic Regression',
        'best_params': best_params,
        'cv_score': grid_search.best_score_,
        'metrics': metrics,
        'training_time': training_time
    }
    
    return best_model, metrics, y_pred, y_pred_proba, results

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest with hyperparameter tuning"""
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST")
    print("="*60)
    
    start_time = time.time()
    
    # Simplified hyperparameter grid for faster execution
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt']
    }
    
    # Grid search with cross-validation
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        rf, param_grid, cv=3, scoring='roc_auc',
        n_jobs=-1, verbose=1
    )
    
    print("Performing hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    training_time = time.time() - start_time
    
    print(f"SUCCESS: Training completed in {training_time:.2f} seconds")
    print(f"Best parameters: {best_params}")
    print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    metrics, y_pred, y_pred_proba = evaluate_model(best_model, X_test, y_test, "Random Forest")
    
    # Save model
    joblib.dump(best_model, 'models/random_forest_model.pkl')
    
    # Save results
    results = {
        'model_name': 'Random Forest',
        'best_params': best_params,
        'cv_score': grid_search.best_score_,
        'metrics': metrics,
        'training_time': training_time
    }
    
    return best_model, metrics, y_pred, y_pred_proba, results

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost with hyperparameter tuning"""
    print("\n" + "="*60)
    print("TRAINING XGBOOST")
    print("="*60)
    
    start_time = time.time()
    
    # Simplified hyperparameter grid for faster execution
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # Grid search with cross-validation
    xgb_model = xgb.XGBClassifier(
        random_state=42, 
        eval_metric='logloss',
        use_label_encoder=False
    )
    grid_search = GridSearchCV(
        xgb_model, param_grid, cv=3, scoring='roc_auc',
        n_jobs=-1, verbose=1
    )
    
    print("Performing hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    training_time = time.time() - start_time
    
    print(f"SUCCESS: Training completed in {training_time:.2f} seconds")
    print(f"Best parameters: {best_params}")
    print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    metrics, y_pred, y_pred_proba = evaluate_model(best_model, X_test, y_test, "XGBoost")
    
    # Save model
    joblib.dump(best_model, 'models/xgboost_model.pkl')
    
    # Save results
    results = {
        'model_name': 'XGBoost',
        'best_params': best_params,
        'cv_score': grid_search.best_score_,
        'metrics': metrics,
        'training_time': training_time
    }
    
    return best_model, metrics, y_pred, y_pred_proba, results

def create_comparison_visualizations(models_data, X_test, y_test):
    """Create comprehensive model comparison visualizations"""
    print("\n" + "="*60)
    print("CREATING MODEL COMPARISON VISUALIZATIONS")
    print("="*60)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a large figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Comparison Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Metrics Comparison (Bar Chart)
    metrics_df = pd.DataFrame({
        name: data['metrics'] for name, data in models_data.items()
    }).T
    
    ax1 = axes[0, 0]
    metrics_df.plot(kind='bar', ax=ax1, rot=45)
    ax1.set_title('Performance Metrics Comparison')
    ax1.set_ylabel('Score')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. ROC Curves
    ax2 = axes[0, 1]
    for name, data in models_data.items():
        fpr, tpr, _ = roc_curve(y_test, data['y_pred_proba'])
        auc = data['metrics']['roc_auc']
        ax2.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
    
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Precision-Recall Curves
    ax3 = axes[0, 2]
    for name, data in models_data.items():
        precision, recall, _ = precision_recall_curve(y_test, data['y_pred_proba'])
        ax3.plot(recall, precision, label=name, linewidth=2)
    
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision-Recall Curves')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Confusion Matrices
    for i, (name, data) in enumerate(models_data.items()):
        ax = axes[1, i]
        cm = confusion_matrix(y_test, data['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{name} Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('models/model_comparison_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create metrics summary table
    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*60)
    
    summary_df = pd.DataFrame(models_data).T
    summary_df = pd.DataFrame([data['metrics'] for data in models_data.values()], 
                             index=models_data.keys())
    
    # Sort by ROC-AUC (best metric for imbalanced dataset)
    summary_df = summary_df.sort_values('roc_auc', ascending=False)
    
    print("\nRanked by ROC-AUC Score:")
    print(summary_df.round(4))
    
    # Save summary table
    summary_df.to_csv('models/model_performance_summary.csv')
    
    return summary_df

def save_experiment_results(all_results):
    """Save all experiment results to JSON"""
    print("\nSaving experiment results...")
    
    # Convert to JSON-serializable format
    results_for_json = {}
    for model_name, result in all_results.items():
        results_for_json[model_name] = {
            'model_name': result['model_name'],
            'best_params': result['best_params'],
            'cv_score': float(result['cv_score']),
            'training_time': float(result['training_time']),
            'test_metrics': {k: float(v) for k, v in result['metrics'].items()}
        }
    
    with open('models/experiment_results.json', 'w') as f:
        json.dump(results_for_json, f, indent=2)
    
    print("SUCCESS: Results saved to models/experiment_results.json")

def main():
    """Main execution function"""
    print("ADULT INCOME CLASSIFICATION - MODEL TRAINING AND COMPARISON")
    print("="*70)
    
    # Create directories
    Path('models').mkdir(exist_ok=True)
    
    # Load data
    X_train, X_test, y_train, y_test, feature_names = load_preprocessed_data()
    
    # Store model results
    models_data = {}
    all_results = {}
    
    # Train Logistic Regression
    print("\nStarting Starting Logistic Regression training...")
    lr_model, lr_metrics, lr_pred, lr_proba, lr_results = train_logistic_regression(
        X_train, y_train, X_test, y_test
    )
    models_data['Logistic Regression'] = {
        'model': lr_model,
        'metrics': lr_metrics,
        'y_pred': lr_pred,
        'y_pred_proba': lr_proba
    }
    all_results['Logistic Regression'] = lr_results
    
    # Train Random Forest
    print("\nStarting Starting Random Forest training...")
    rf_model, rf_metrics, rf_pred, rf_proba, rf_results = train_random_forest(
        X_train, y_train, X_test, y_test
    )
    models_data['Random Forest'] = {
        'model': rf_model,
        'metrics': rf_metrics,
        'y_pred': rf_pred,
        'y_pred_proba': rf_proba
    }
    all_results['Random Forest'] = rf_results
    
    # Train XGBoost
    print("\nStarting Starting XGBoost training...")
    xgb_model, xgb_metrics, xgb_pred, xgb_proba, xgb_results = train_xgboost(
        X_train, y_train, X_test, y_test
    )
    models_data['XGBoost'] = {
        'model': xgb_model,
        'metrics': xgb_metrics,
        'y_pred': xgb_pred,
        'y_pred_proba': xgb_proba
    }
    all_results['XGBoost'] = xgb_results
    
    # Create visualizations and comparisons
    summary_df = create_comparison_visualizations(models_data, X_test, y_test)
    
    # Save experiment results
    save_experiment_results(all_results)
    
    # Print final summary
    print("\n" + "="*70)
    print("SUCCESS: TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"SUCCESS: Trained and evaluated 3 models")
    print(f"SUCCESS: Models saved to 'models/' directory")
    print(f"SUCCESS: Performance visualizations created")
    print(f"SUCCESS: Summary report saved")
    print(f"SUCCESS: Experiment results logged")
    
    # Best model recommendation
    best_model = summary_df.index[0]
    best_auc = summary_df.loc[best_model, 'roc_auc']
    print(f"\nBEST MODEL: BEST MODEL: {best_model} (ROC-AUC: {best_auc:.4f})")
    
    # Training time summary
    print(f"\nTraining Time Summary:")
    for model_name, result in all_results.items():
        print(f"  {model_name}: {result['training_time']:.2f} seconds")
    
    total_time = sum(result['training_time'] for result in all_results.values())
    print(f"  Total: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()