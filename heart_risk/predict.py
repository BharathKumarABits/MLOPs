"""Load the saved model and score a single patient record."""
import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from heart_risk.data import FEATURES

DEFAULT_MODEL_PATH = Path("models/heart_risk_model.joblib")
THRESHOLD = 0.5
LABELS = {0: "no heart disease", 1: "heart disease"}


def model_path() -> Path:
    return Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))


@lru_cache(maxsize=4)
def load_model(path: str):
    return joblib.load(path)


def _to_float(value) -> float:
    return float("nan") if value is None else float(value)


def predict_one(record: dict, path=None) -> dict:
    """Score one patient; ca and thal may be None (imputed by the pipeline)."""
    missing = [f for f in FEATURES if f not in record]
    if missing:
        raise ValueError(f"missing features: {missing}")
    model = load_model(str(path or model_path()))
    X = pd.DataFrame([{f: _to_float(record[f]) for f in FEATURES}], columns=FEATURES)
    proba = float(model.predict_proba(X)[0, 1])
    pred = int(proba >= THRESHOLD)
    return {
        "prediction": pred,
        "label": LABELS[pred],
        "confidence": round(proba if pred == 1 else 1 - proba, 4),
        "probability_disease": round(proba, 4),
    }
