# This file evaluates the saved model on test data

import pickle
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)
from dotenv import load_dotenv
import os
from preprocess import load_transformed_data, preprocess
from sklearn.model_selection import train_test_split

# load .env properly
load_dotenv(dotenv_path=".env")

def evaluate():
    # ✅ FIX: safe path (no None error)
    model_path = os.getenv('MODEL_PATH') or "model.pkl"

    print("Loading model from:", model_path)  # debug print

    # load saved model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print("Model loaded successfully")

    # load and preprocess data
    df = load_transformed_data()
    X, y = preprocess(df)

    # split same as training
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # metrics
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm  = confusion_matrix(y_test, y_pred)

    # print results
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {pre:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC AUC   : {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    return acc, pre, rec, f1, auc


if __name__ == "__main__":
    evaluate()