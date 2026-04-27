# This file contains pytest tests for FastAPI endpoints

import pytest
from fastapi.testclient import TestClient
import sys
import os

# add api and ml directory to path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../ml')))

from api.main import app

client = TestClient(app)

# sample valid input for prediction
sample_input = {
    "Administrative": 0,
    "Administrative_Duration": 0.0,
    "Informational": 0,
    "Informational_Duration": 0.0,
    "ProductRelated": 1,
    "ProductRelated_Duration": 0.0,
    "BounceRates": 0.2,
    "ExitRates": 0.2,
    "PageValues": 0.0,
    "SpecialDay": 0.0,
    "Month": 2,
    "OperatingSystems": 1,
    "Browser": 1,
    "Region": 1,
    "TrafficType": 1,
    "VisitorType": 2,
    "Weekend": 0
}


# ---------------- ROOT TEST ----------------
def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")


# ---------------- HEALTH CHECK ----------------
def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json().get("status") == "healthy"


# ---------------- VALID PREDICTION ----------------
def test_predict_valid_input():
    response = client.post("/predict", json=sample_input)

    # 🔥 IMPORTANT: if API fails, show error details
    if response.status_code != 200:
        print("ERROR RESPONSE:", response.json())

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "probability" in data
    assert "message" in data

    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["probability"] <= 1.0


# ---------------- INVALID INPUT ----------------
def test_predict_invalid_input():
    response = client.post("/predict", json={"Administrative": 0})

    assert response.status_code == 422