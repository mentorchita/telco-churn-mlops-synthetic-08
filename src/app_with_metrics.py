"""
src/app_with_metrics.py
========================
Drop-in replacement / patch for the existing FastAPI app.py in
telco-churn-mlops-synthetic-07.

HOW TO USE:
  Option A — Replace app.py entirely with this file.
  Option B — Copy only the relevant sections into your existing app.py.

Sections to add:
  1. Imports (top of file)
  2. JSONFormatter class (structured logging for Loki)
  3. MetricsMiddleware class (auto HTTP metrics)
  4. lifespan() function (record model load time)
  5. /predict endpoint changes (add metric calls)
  6. /metrics endpoint (new)

Requirements:
  pip install prometheus-client==0.19.0 structlog==23.2.0
  # Add both to requirements-api.txt
"""

# ─────────────────────────────────────────────────────────────
# SECTION 1: IMPORTS — add these to the top of app.py
# ─────────────────────────────────────────────────────────────
import time
import json
import logging
import structlog
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse

# Import ALL metrics from our centralised module
from src.metrics import (
    PREDICTIONS_TOTAL,
    PREDICTION_LATENCY,
    MODEL_ACCURACY,
    MODEL_LOAD_TIME,
    ACTIVE_MODEL_VERSION,
    PREDICTION_CONFIDENCE,
    API_REQUESTS_TOTAL,
    REQUEST_DURATION,
    ACTIVE_CONNECTIONS,
    NULL_FEATURES_TOTAL,
    validate_feature,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# ─────────────────────────────────────────────────────────────
# SECTION 2: STRUCTURED JSON LOGGING (for Loki parsing)
# ─────────────────────────────────────────────────────────────

class JSONFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON.
    Each field becomes filterable in Loki via LogQL:
      {job="churn-api"} | json | latency_ms > 500
    """
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp":     self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S.%fZ"),
            "level":         record.levelname,
            "logger":        record.name,
            "message":       record.getMessage(),
            # Optional contextual fields — set via extra={} in logging calls
            "endpoint":      getattr(record, "endpoint",      ""),
            "method":        getattr(record, "method",        ""),
            "status_code":   getattr(record, "status_code",   ""),
            "latency_ms":    getattr(record, "latency_ms",    0),
            "model_version": getattr(record, "model_version", ""),
            "outcome":       getattr(record, "outcome",       ""),
            "request_id":    getattr(record, "request_id",    ""),
        }
        # Remove empty fields to reduce log size
        return json.dumps({k: v for k, v in log_data.items() if v != ""})


# Set up root logger with JSON formatter
def setup_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(logging.INFO)

    # Configure structlog to output JSON
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )

setup_logging()
logger = logging.getLogger("churn-api")


# ─────────────────────────────────────────────────────────────
# SECTION 3: METRICS MIDDLEWARE
# Automatically records HTTP metrics for EVERY request
# ─────────────────────────────────────────────────────────────

class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware that instruments all HTTP requests with Prometheus metrics.
    Add to app BEFORE other middleware:
        app.add_middleware(MetricsMiddleware)
    """

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        ACTIVE_CONNECTIONS.inc()

        # Get clean endpoint path (strip query params)
        endpoint = request.url.path

        try:
            response = await call_next(request)
            duration = time.time() - start_time
            status_code = str(response.status_code)

            # Record metrics
            API_REQUESTS_TOTAL.labels(
                method=request.method,
                endpoint=endpoint,
                status_code=status_code,
            ).inc()

            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=endpoint,
            ).observe(duration)

            # Structured log for every request (visible in Loki)
            logger.info(
                "request.completed",
                extra={
                    "endpoint":    endpoint,
                    "method":      request.method,
                    "status_code": status_code,
                    "latency_ms":  round(duration * 1000, 2),
                }
            )
            return response

        except Exception as exc:
            duration = time.time() - start_time
            API_REQUESTS_TOTAL.labels(
                method=request.method,
                endpoint=endpoint,
                status_code="500",
            ).inc()
            logger.error(
                "request.error",
                extra={
                    "endpoint":   endpoint,
                    "method":     request.method,
                    "latency_ms": round(duration * 1000, 2),
                    "error":      str(exc),
                }
            )
            raise
        finally:
            ACTIVE_CONNECTIONS.dec()


# ─────────────────────────────────────────────────────────────
# SECTION 4: LIFESPAN (model loading with metrics)
# Replace the existing @app.on_event("startup") if present
# ─────────────────────────────────────────────────────────────

# Global model state
model = None
model_version = "unknown"
mlflow_run_id = "unknown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, record load time and version metrics."""
    global model, model_version, mlflow_run_id

    logger.info("startup.begin", extra={"message": "Loading model from MLflow"})
    load_start = time.time()

    try:
        # ── Replace this block with your actual model loading code ──
        import mlflow
        import os

        MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        MODEL_NAME = os.getenv("MODEL_NAME", "churn-model")
        MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()

        # Get latest version in stage
        versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        if versions:
            latest = versions[0]
            model_version = f"v{latest.version}"
            mlflow_run_id = latest.run_id
            model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
            model = mlflow.sklearn.load_model(model_uri)
        else:
            # Fallback: load from local file if MLflow not available
            import joblib
            model = joblib.load("models/churn_model.pkl")
            model_version = os.getenv("MODEL_VERSION", "v1.0")
        # ── End model loading block ──────────────────────────────────

        load_time = time.time() - load_start
        MODEL_LOAD_TIME.labels(model_version=model_version).set(load_time)
        ACTIVE_MODEL_VERSION.labels(
            model_version=model_version,
            stage=MODEL_STAGE if 'MODEL_STAGE' in dir() else "production",
            run_id=mlflow_run_id,
        ).set(1)

        logger.info(
            "startup.model_loaded",
            extra={
                "model_version": model_version,
                "load_time_s":   round(load_time, 3),
                "message":       f"Model {model_version} loaded successfully",
            }
        )

    except Exception as e:
        logger.error(
            "startup.model_load_failed",
            extra={"error": str(e), "message": "Failed to load model"}
        )
        # Don't raise — API starts but /predict will return 503

    yield  # app runs here

    # Cleanup on shutdown
    logger.info("shutdown.begin", extra={"message": "API shutting down"})
    if model_version != "unknown":
        ACTIVE_MODEL_VERSION.labels(
            model_version=model_version,
            stage="production",
            run_id=mlflow_run_id,
        ).set(0)


# ─────────────────────────────────────────────────────────────
# SECTION 5: FASTAPI APP
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Telco Churn Prediction API",
    description="MLOps prediction service for telco customer churn",
    version="1.0.0",
    lifespan=lifespan,
)

# Add metrics middleware FIRST (outermost layer)
app.add_middleware(MetricsMiddleware)


# ─────────────────────────────────────────────────────────────
# SECTION 6: /metrics ENDPOINT
# Prometheus scrapes this endpoint every 10-15 seconds
# ─────────────────────────────────────────────────────────────

@app.get("/metrics", include_in_schema=False)
async def metrics():
    """
    Prometheus metrics endpoint.
    Scraped by Prometheus every 10s (configured in prometheus.yml).
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ─────────────────────────────────────────────────────────────
# SECTION 7: /health ENDPOINT (enhanced)
# ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check — returns model status."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail={"status": "unhealthy", "reason": "model not loaded"}
        )
    return {
        "status":        "healthy",
        "model_version": model_version,
        "run_id":        mlflow_run_id,
    }


# ─────────────────────────────────────────────────────────────
# SECTION 8: /predict ENDPOINT (instrumented)
# Replace your existing /predict handler with this
# ─────────────────────────────────────────────────────────────

# Feature validation ranges for telco-churn dataset
FEATURE_RANGES = {
    "tenure":         (0, 100),
    "MonthlyCharges": (0, 500),
    "TotalCharges":   (0, 100000),
    "SeniorCitizen":  (0, 1),
}


@app.post("/predict")
async def predict(request: Request):
    """
    Churn prediction endpoint.
    Accepts customer feature JSON, returns churn probability.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    import uuid
    req_id = str(uuid.uuid4())[:8]

    try:
        data = await request.json()
    except Exception:
        PREDICTIONS_TOTAL.labels(
            model_version=model_version,
            outcome="error",
            contract_type="unknown",
        ).inc()
        raise HTTPException(status_code=422, detail="Invalid JSON body")

    contract_type = data.get("Contract", "Unknown")

    # ── Validate input features ──────────────────────────────
    for feature, range_ in FEATURE_RANGES.items():
        if feature in data:
            validate_feature(feature, data.get(feature), range_)

    # ── Run inference with latency measurement ──────────────
    try:
        with PREDICTION_LATENCY.labels(model_version=model_version).time():
            import pandas as pd

            # Build feature dataframe (match training column order)
            feature_cols = [
                "tenure", "MonthlyCharges", "TotalCharges",
                "gender", "SeniorCitizen", "Partner", "Dependents",
                "PhoneService", "MultipleLines", "InternetService",
                "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies",
                "Contract", "PaperlessBilling", "PaymentMethod",
            ]
            features_df = pd.DataFrame([{col: data.get(col) for col in feature_cols}])
            proba = model.predict_proba(features_df)[0]

        churn_prob = float(proba[1])
        outcome = "churn" if churn_prob >= 0.5 else "no_churn"

        # ── Record metrics ──────────────────────────────────
        PREDICTIONS_TOTAL.labels(
            model_version=model_version,
            outcome=outcome,
            contract_type=contract_type,
        ).inc()

        PREDICTION_CONFIDENCE.labels(
            model_version=model_version,
            outcome=outcome,
        ).observe(churn_prob)

        # ── Structured log for Loki ─────────────────────────
        logger.info(
            "prediction.made",
            extra={
                "request_id":    req_id,
                "model_version": model_version,
                "outcome":       outcome,
                "churn_prob":    round(churn_prob, 4),
                "contract_type": contract_type,
                "tenure":        data.get("tenure"),
            }
        )

        return {
            "churn_probability": round(churn_prob, 4),
            "churn_prediction":  outcome == "churn",
            "outcome":           outcome,
            "model_version":     model_version,
            "request_id":        req_id,
        }

    except Exception as exc:
        PREDICTIONS_TOTAL.labels(
            model_version=model_version,
            outcome="error",
            contract_type=contract_type,
        ).inc()
        logger.error(
            "prediction.error",
            extra={
                "request_id":  req_id,
                "error":       str(exc),
                "contract":    contract_type,
            }
        )
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)}")


# ─────────────────────────────────────────────────────────────
# SECTION 9: DEMO ENDPOINT — inject latency (DEMO 4)
# Remove after demo!
# ─────────────────────────────────────────────────────────────

import random
import os

INJECT_LATENCY = os.getenv("INJECT_LATENCY", "false").lower() == "true"


@app.post("/predict-slow")
async def predict_slow(request: Request):
    """
    DEMO 4: Same as /predict but with artificial latency injection.
    Enable via env var: INJECT_LATENCY=true
    Or use this endpoint directly in load tests.
    """
    if INJECT_LATENCY and random.random() < 0.30:   # 30% slow
        await __import__("asyncio").sleep(random.uniform(0.8, 2.5))
    return await predict(request)


# ─────────────────────────────────────────────────────────────
# Entry point for local dev
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app_with_metrics:app", host="0.0.0.0", port=8000, reload=True)
