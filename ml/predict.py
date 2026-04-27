import pickle
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()


# =========================
# LOAD ARTIFACTS
# =========================
def load_artifacts():
    model_path = os.getenv('MODEL_PATH') or "model.pkl"

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open('./ml/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    selector = None
    if os.path.exists('./ml/selector.pkl'):
        with open('./ml/selector.pkl', 'rb') as f:
            selector = pickle.load(f)

    with open('./ml/selected_features.pkl', 'rb') as f:
        selected_features = pickle.load(f)

    return model, scaler, selector, selected_features


# 🔥 LOAD ONCE (FAST)
model, scaler, selector, selected_features = load_artifacts()


# =========================
# PREDICT FUNCTION (FIXED)
# =========================
def predict(input_data: dict):

    try:
        df = pd.DataFrame([input_data])

        # 🔥 MOST IMPORTANT FIX (17 FEATURES GUARANTEE)
        df = df.reindex(columns=scaler.feature_names_in_, fill_value=0)

        # scaling
        X_scaled = scaler.transform(df.values)

        # feature selection (safe)
        if selector is not None:
            X_processed = selector.transform(X_scaled)
        else:
            X_processed = X_scaled

        # prediction
        prediction = model.predict(X_processed)[0]

        # probability safe
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(X_processed)[0][1]
        else:
            probability = 0.0

        return {
            "prediction": int(prediction),
            "probability": round(float(probability), 4),
            "message": "Will Purchase" if prediction == 1 else "Will Not Purchase"
        }

    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")