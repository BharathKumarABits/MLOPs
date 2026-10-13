import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from api.main import app
from heart_risk.data import FEATURES, TARGET, clean
from heart_risk.features import build_preprocessor

SAMPLE = {"age": 55, "sex": 1, "cp": 4, "trestbps": 140, "chol": 250, "fbs": 0,
          "restecg": 1, "thalach": 130, "exang": 1, "oldpeak": 2.0, "slope": 2,
          "ca": 1, "thal": 7}


@pytest.fixture
def client(raw_df, tmp_path, monkeypatch):
    df = clean(raw_df)
    pipe = Pipeline([("prep", build_preprocessor()),
                     ("clf", LogisticRegression(max_iter=1000))])
    pipe.fit(df[FEATURES].astype("float64"), df[TARGET])
    path = tmp_path / "model.joblib"
    joblib.dump(pipe, path)
    monkeypatch.setenv("MODEL_PATH", str(path))
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_returns_prediction_and_confidence(client):
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0


def test_predict_accepts_missing_ca_and_thal(client):
    payload = {**SAMPLE, "ca": None, "thal": None}
    assert client.post("/predict", json=payload).status_code == 200


def test_predict_rejects_invalid_input(client):
    assert client.post("/predict", json={**SAMPLE, "age": -5}).status_code == 422
    assert client.post("/predict", json={**SAMPLE, "thal": 5}).status_code == 422
    incomplete = {k: v for k, v in SAMPLE.items() if k != "chol"}
    assert client.post("/predict", json=incomplete).status_code == 422


def test_metrics_endpoint_exposes_counters(client):
    client.post("/predict", json=SAMPLE)
    text = client.get("/metrics").text
    assert "api_requests_total" in text
    assert "model_predictions_total" in text
