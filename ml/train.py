# This file trains ML models and logs results to MLflow

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import pickle
from dotenv import load_dotenv
import os
from preprocess import load_transformed_data, preprocess
from mlflow_tracker import log_all_runs

# load env
load_dotenv(dotenv_path=".env")

def train():
    df = load_transformed_data()
    X, y = preprocess(df)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_model.predict(X_test)):.4f}")

    # XGBoost
    xgb_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    print(f"XGBoost Accuracy: {accuracy_score(y_test, xgb_model.predict(X_test)):.4f}")

    # metrics
    rf_metrics = {
        'acc': accuracy_score(y_test, rf_model.predict(X_test)),
        'pre': precision_score(y_test, rf_model.predict(X_test)),
        'rec': recall_score(y_test, rf_model.predict(X_test)),
        'f1':  f1_score(y_test, rf_model.predict(X_test)),
        'auc': roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1])
    }

    xgb_metrics = {
        'acc': accuracy_score(y_test, xgb_model.predict(X_test)),
        'pre': precision_score(y_test, xgb_model.predict(X_test)),
        'rec': recall_score(y_test, xgb_model.predict(X_test)),
        'f1':  f1_score(y_test, xgb_model.predict(X_test)),
        'auc': roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:,1])
    }

    # best model
    if xgb_metrics['acc'] >= rf_metrics['acc']:
        best_model = xgb_model
        best_name = "XGBoost"
    else:
        best_model = rf_model
        best_name = "RandomForest"

    print(f"Best Model: {best_name}")

    # MLflow logging
    log_all_runs(rf_model, rf_metrics, xgb_model, xgb_metrics, best_name)

    # ✅ FIX: safe model path (NO None error)
    model_path = os.getenv('MODEL_PATH') or "model.pkl"

    print("Saving model to:", model_path)

    # save model
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)

    print(f"Model saved successfully to {model_path}")

    return best_model, X_test, y_test, best_name


if __name__ == "__main__":
    train()