"""
FastAPI inference service for ECG multi-label classification.

Accepts a WFDB record filename, extracts features from the raw signal,
and returns predicted diagnostic superclass probabilities.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from features import extract_features

# Paths to the saved model and scaler
MODEL_PATH  = Path("data/model_xgb.pkl")
SCALER_PATH = Path("data/scaler.pkl")

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
THRESHOLD    = 0.5  # Probability threshold for positive prediction

app = FastAPI(
    title="ECG Classification API",
    description="Multi-label ECG diagnostic classification using PTB-XL features and XGBoost.",
    version="1.0.0",
)

# Load model and scaler once at startup — not on every request
model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


class PredictionRequest(BaseModel):
    """Request body: relative path to a WFDB record (without file extension)."""
    filename_lr: str


class PredictionResponse(BaseModel):
    """Response body: predicted labels and their probabilities."""
    filename_lr:   str
    probabilities: dict[str, float]
    predictions:   dict[str, bool]


@app.get("/health")
def health_check():
    """Check that the API is running and the model is loaded."""
    return {"status": "ok", "model": "xgboost", "classes": SUPERCLASSES}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Run inference on a single ECG record.

    Extracts 76 features from the raw signal, scales them,
    and returns a probability and binary prediction per superclass.
    """
    # Extract features from the raw signal
    features = extract_features(request.filename_lr)
    if features is None:
        raise HTTPException(
            status_code=422,
            detail=f"Could not load or process record: {request.filename_lr}"
        )

    # Build a 1-row feature matrix in the correct column order
    feature_vector = np.array([list(features.values())]).reshape(1, -1)

    # Scale using the training scaler
    feature_vector = scaler.transform(feature_vector)

    # Replace any NaN values that survived (e.g. single-peak HRV)
    feature_vector = np.nan_to_num(feature_vector, nan=0.0)

    # Get probabilities from the model
    probabilities = model.predict_proba(feature_vector)[0]

    prob_dict = {cls: round(float(prob), 4) for cls, prob in zip(SUPERCLASSES, probabilities)}
    pred_dict = {cls: prob >= THRESHOLD for cls, prob in prob_dict.items()}

    return PredictionResponse(
        filename_lr=request.filename_lr,
        probabilities=prob_dict,
        predictions=pred_dict,
    )