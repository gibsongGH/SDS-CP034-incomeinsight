from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import pandas as pd

def train_models(df, target_col='income'):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'LogisticRegression': LogisticRegression(max_iter=500),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    results = {}

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred)
            }

            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, name="model")

            results[name] = metrics
            print(f"Model: {name} | Metrics: {metrics}")

    return results

import pickle
import os

if __name__ == "__main__":
    from data_processing import load_data, encode_features, scale_features, save_preprocessors

    # Load and preprocess data
    df = load_data('data/adult.csv')
    df, le_dict = encode_features(df)
    numeric_cols = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    df, scaler = scale_features(df, numeric_cols)

    # Train models
    results = train_models(df, target_col='income')
    print("All model results:", results)

    # Save preprocessors too
    save_preprocessors(le_dict, scaler, df.drop(columns=['income']).columns.tolist())

    # Pick the best model (example: based on F1-score)
    best_model_name = max(results, key=lambda k: results[k]['f1_score'])
    print(f"Best model selected: {best_model_name}")

    # Train again with best model to save
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=['income'])
    y = df['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if best_model_name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        best_model = LogisticRegression(max_iter=500).fit(X_train, y_train)
    elif best_model_name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        best_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    else:  # XGBoost
        from xgboost import XGBClassifier
        best_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42).fit(X_train, y_train)

    # Save model to src/model.pkl
    model_path = os.path.join("src", "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    print(f"✅ Saved best model ({best_model_name}) to {model_path}")