#!/usr/bin/env python3
"""
Machine Learning Model Training and Comparison
==============================================

This script trains and compares multiple classification models for income prediction:
- Logistic Regression
- Random Forest  
- XGBoost

Uses MLflow for experiment tracking and comprehensive evaluation metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve, precision_recall_curve
)
import xgboost as xgb
import joblib

# MLflow for experiment tracking
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient

def setup_mlflow():
    """Set up MLflow experiment tracking"""
    print("Setting up MLflow experiment tracking...")
    
    # Set experiment name
    experiment_name = "Adult_Income_Classification"
    mlflow.set_experiment(experiment_name)
    
    # Get experiment info
    experiment = mlflow.get_experiment_by_name(experiment_name)
    print(f"MLflow experiment: {experiment_name}")
    print(f"Experiment ID: {experiment.experiment_id}")
    print(f"Artifact location: {experiment.artifact_uri}")
    
    return experiment

def load_preprocessed_data():
    """Load the preprocessed data from the EDA notebook"""
    print("Loading preprocessed data...")
    
    try:
        # Load the data arrays
        X_train = np.load('data/X_train.npy')
        X_test = np.load('data/X_test.npy') 
        y_train = np.load('data/y_train.npy')
        y_test = np.load('data/y_test.npy')
        
        # Load feature names
        with open('data/feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        print(f"Data loaded successfully:")
        print(f"  Training set: {X_train.shape}")
        print(f"  Test set: {X_test.shape}")
        print(f"  Features: {len(feature_names)}")
        print(f"  Class distribution in training:")
        print(f"    <=50K: {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%)")
        print(f"    >50K: {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")
        
        return X_train, X_test, y_train, y_test, feature_names
        
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please run the EDA notebook first to generate preprocessed data.")
        raise

def evaluate_model(model, X_test, y_test, model_name):
    """Comprehensive model evaluation"""
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
    
    return metrics, y_pred, y_pred_proba

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train Logistic Regression with hyperparameter tuning"""
    print("\n" + "="*60)
    print("TRAINING LOGISTIC REGRESSION")
    print("="*60)
    
    with mlflow.start_run(run_name="logistic_regression") as run:
        # Hyperparameter grid
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000]
        }
        
        # Grid search with cross-validation
        lr = LogisticRegression(random_state=42)
        grid_search = GridSearchCV(
            lr, param_grid, cv=5, scoring='roc_auc', 
            n_jobs=-1, verbose=1
        )
        
        print("Performing hyperparameter tuning...")
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"Best parameters: {best_params}")
        print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        
        # Evaluate on test set
        metrics, y_pred, y_pred_proba = evaluate_model(best_model, X_test, y_test, "Logistic Regression")
        
        # Log to MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_roc_auc", grid_search.best_score_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, "model")
        
        # Save model locally
        joblib.dump(best_model, 'models/logistic_regression_model.pkl')
        
        return best_model, metrics, y_pred, y_pred_proba, run.info.run_id

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest with hyperparameter tuning"""
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST")
    print("="*60)
    
    with mlflow.start_run(run_name="random_forest") as run:
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        # Grid search with cross-validation
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='roc_auc',  # Reduced CV for speed
            n_jobs=-1, verbose=1
        )
        
        print("Performing hyperparameter tuning...")
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"Best parameters: {best_params}")
        print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        
        # Evaluate on test set
        metrics, y_pred, y_pred_proba = evaluate_model(best_model, X_test, y_test, "Random Forest")
        
        # Log to MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_roc_auc", grid_search.best_score_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, "model")
        
        # Save model locally
        joblib.dump(best_model, 'models/random_forest_model.pkl')
        
        return best_model, metrics, y_pred, y_pred_proba, run.info.run_id

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost with hyperparameter tuning"""
    print("\n" + "="*60)
    print("TRAINING XGBOOST")
    print("="*60)
    
    with mlflow.start_run(run_name="xgboost") as run:
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
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
            xgb_model, param_grid, cv=3, scoring='roc_auc',  # Reduced CV for speed
            n_jobs=-1, verbose=1
        )
        
        print("Performing hyperparameter tuning...")
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"Best parameters: {best_params}")
        print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        
        # Evaluate on test set
        metrics, y_pred, y_pred_proba = evaluate_model(best_model, X_test, y_test, "XGBoost")
        
        # Log to MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_roc_auc", grid_search.best_score_)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(best_model, "model")
        
        # Save model locally
        joblib.dump(best_model, 'models/xgboost_model.pkl')
        
        return best_model, metrics, y_pred, y_pred_proba, run.info.run_id

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

def main():
    """Main execution function"""
    print("ADULT INCOME CLASSIFICATION - MODEL TRAINING AND COMPARISON")
    print("="*70)
    
    # Create directories
    Path('models').mkdir(exist_ok=True)
    Path('mlruns').mkdir(exist_ok=True)
    
    # Set up MLflow
    experiment = setup_mlflow()
    
    # Load data
    X_train, X_test, y_train, y_test, feature_names = load_preprocessed_data()
    
    # Store model results
    models_data = {}
    
    # Train Logistic Regression
    lr_model, lr_metrics, lr_pred, lr_proba, lr_run_id = train_logistic_regression(
        X_train, y_train, X_test, y_test
    )
    models_data['Logistic Regression'] = {
        'model': lr_model,
        'metrics': lr_metrics,
        'y_pred': lr_pred,
        'y_pred_proba': lr_proba,
        'run_id': lr_run_id
    }
    
    # Train Random Forest
    rf_model, rf_metrics, rf_pred, rf_proba, rf_run_id = train_random_forest(
        X_train, y_train, X_test, y_test
    )
    models_data['Random Forest'] = {
        'model': rf_model,
        'metrics': rf_metrics,
        'y_pred': rf_pred,
        'y_pred_proba': rf_proba,
        'run_id': rf_run_id
    }
    
    # Train XGBoost
    xgb_model, xgb_metrics, xgb_pred, xgb_proba, xgb_run_id = train_xgboost(
        X_train, y_train, X_test, y_test
    )
    models_data['XGBoost'] = {
        'model': xgb_model,
        'metrics': xgb_metrics,
        'y_pred': xgb_pred,
        'y_pred_proba': xgb_proba,
        'run_id': xgb_run_id
    }
    
    # Create visualizations and comparisons
    summary_df = create_comparison_visualizations(models_data, X_test, y_test)
    
    # Print final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"✓ Trained and evaluated 3 models")
    print(f"✓ All experiments logged to MLflow")
    print(f"✓ Models saved to 'models/' directory")
    print(f"✓ Performance visualizations created")
    print(f"✓ Summary report saved")
    
    # Best model recommendation
    best_model = summary_df.index[0]
    best_auc = summary_df.loc[best_model, 'roc_auc']
    print(f"\n🏆 BEST MODEL: {best_model} (ROC-AUC: {best_auc:.4f})")
    
    print(f"\nTo view MLflow UI, run: mlflow ui")
    print(f"Then navigate to: http://localhost:5000")

if __name__ == "__main__":
    main()