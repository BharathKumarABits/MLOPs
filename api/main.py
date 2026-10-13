"""FastAPI service exposing the heart-disease model."""
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field, field_validator

from heart_risk.predict import load_model, model_path, predict_one

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("heart_api")

REQUESTS = Counter("api_requests_total", "Total API requests",
                   ["endpoint", "method", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"])
PREDICTIONS = Counter("model_predictions_total", "Predictions served", ["label"])


class Patient(BaseModel):
    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=50, le=300, description="Resting BP (mm Hg)")
    chol: float = Field(..., ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(..., ge=50, le=250, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression")
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment")
    ca: float | None = Field(None, ge=0, le=4, description="Major vessels (0-4)")
    thal: float | None = Field(None, description="Thalassemia: 3, 6 or 7")

    @field_validator("thal")
    @classmethod
    def check_thal(cls, value):
        if value is not None and value not in (3, 6, 7):
            raise ValueError("thal must be 3, 6 or 7")
        return value


class Prediction(BaseModel):
    prediction: int
    label: str
    confidence: float
    probability_disease: float


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        load_model(str(model_path()))
        logger.info("model loaded from %s", model_path())
    except FileNotFoundError:
        logger.error("model file missing: %s", model_path())
    yield


app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def observe(request: Request, call_next):
    start = time.perf_counter()
    request_id = uuid.uuid4().hex[:8]
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request_id=%s unhandled error on %s",
                         request_id, request.url.path)
        raise
    elapsed = time.perf_counter() - start
    path = request.url.path
    REQUESTS.labels(path, request.method, response.status_code).inc()
    LATENCY.labels(path).observe(elapsed)
    logger.info("request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
                request_id, request.method, path, response.status_code, elapsed * 1000)
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_path().exists()}


@app.post("/predict", response_model=Prediction)
def predict(patient: Patient):
    try:
        result = predict_one(patient.model_dump())
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="model file not found")
    PREDICTIONS.labels(result["label"]).inc()
    logger.info("prediction=%s confidence=%.3f", result["prediction"], result["confidence"])
    return result


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
