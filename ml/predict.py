# This file loads saved model and makes predictions on new input data

import pickle
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os

# load environment variables
load_dotenv(dotenv_path=".env")


def load_artifacts():
    # ✅ FIX: safe model path (no None error)
    model_path = os.getenv('MODEL_PATH') or "model.pkl"
    print("Loading model from:", model_path)

    # load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # load scaler
    with open('./ml/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # load selector
    with open('./ml/selector.pkl', 'rb') as f:
        selector = pickle.load(f)

    # load selected features
    with open('./ml/selected_features.pkl', 'rb') as f:
        selected_features = pickle.load(f)

    return model, scaler, selector, selected_features


def predict(input_data: dict):
    model, scaler, selector, selected_features = load_artifacts()

    # convert input to dataframe
    df = pd.DataFrame([input_data])

    # scale features
    df_scaled = scaler.transform(df)

    # feature selection
    df_selected = selector.transform(df_scaled)

    # prediction
    prediction = model.predict(df_selected)[0]
    probability = model.predict_proba(df_selected)[0][1]

    result = {
        "prediction": int(prediction),
        "probability": round(float(probability), 4),
        "message": "Will Purchase" if prediction == 1 else "Will Not Purchase"
    }

    return result


if __name__ == "__main__":
    # ✅ FIXED input (PagesValues correct name)
    sample_input = {
        "Administrative": 0,
        "Administrative_Duration": 0.0,
        "Informational": 0,
        "Informational_Duration": 0.0,
        "ProductRelated": 1,
        "ProductRelated_Duration": 0.0,
        "BounceRates": 0.2,
        "ExitRates": 0.2,
        "PagesValues": 0.0,   # ✅ correct name
        "SpecialDay": 0.0,
        "Month": 2,
        "OperatingSystems": 1,
        "Browser": 1,
        "Region": 1,
        "TrafficType": 1,
        "VisitorType": 2,
        "Weekend": 0
    }

    result = predict(sample_input)
    print(result)