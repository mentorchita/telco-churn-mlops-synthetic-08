import warnings
import time
warnings.filterwarnings("ignore", message=".*protected namespace.*", category=UserWarning)

from fastapi import FastAPI, HTTPException
from fastapi import Response
from datetime import datetime
import logging

# HTTP metrics (автоматичні)
from prometheus_fastapi_instrumentator import Instrumentator

# ML метрики (наші власні)
from src.metrics import (
    PREDICTIONS_TOTAL,
    PREDICTION_LATENCY,
    MODEL_LOAD_TIME,
    ACTIVE_MODEL_VERSION,
    NULL_FEATURES_TOTAL,
    PREDICTION_CONFIDENCE,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from src.api.models import CustomerFeatures, PredictionResponse
from src.api import predict as predict_module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Версія моделі — буде оновлена при старті
MODEL_VERSION = "v1.0"

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="API для прогнозування відтоку клієнтів (churn prediction)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Автоматичні HTTP метрики (http_requests_total, http_request_duration і т.д.)
Instrumentator().instrument(app).expose(app)


@app.on_event("startup")
async def startup_event():
    global MODEL_VERSION

    if predict_module.model is None:
        logger.error("Модель не завантажилася при старті API!")
        return

    logger.info("Модель успішно завантажена при старті API")

    # Визначити версію моделі
    model_source = getattr(predict_module, "model_source", None)
    if model_source and "MLflow" in str(model_source):
        MODEL_VERSION = "mlflow-production"
    else:
        MODEL_VERSION = "v1.0-local"

    # Записати час завантаження (старт — немає таймера, ставимо символічне значення)
    MODEL_LOAD_TIME.labels(model_version=MODEL_VERSION).set(0)

    # Позначити активну версію моделі
    ACTIVE_MODEL_VERSION.labels(
        model_version=MODEL_VERSION,
        stage="production",
        run_id=str(getattr(predict_module, "model_source", "local")),
    ).set(1)

    logger.info(f"Model version label: {MODEL_VERSION}")


@app.get("/health")
def health():
    model_status = "завантажена" if predict_module.model is not None else "НЕ завантажена"
    return {
        "status": "healthy" if predict_module.model is not None else "degraded",
        "service": "churn-prediction-api",
        "timestamp": datetime.utcnow().isoformat(),
        "model_status": model_status,
        "model_path": getattr(predict_module, "MODEL_PATH", "невідомо"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    input_data = features.dict()

    # Перевірити null значення у ключових полях
    key_fields = ["tenure", "MonthlyCharges", "TotalCharges", "Contract"]
    for field in key_fields:
        if input_data.get(field) is None:
            NULL_FEATURES_TOTAL.labels(feature_name=field).inc()

    contract_type = str(input_data.get("Contract", "Unknown"))

    try:
        # Вимірювати час inference
        with PREDICTION_LATENCY.labels(model_version=MODEL_VERSION).time():
            result = predict_module.predict_churn(input_data)

        if "error" in result:
            # Помилка моделі — зафіксувати
            PREDICTIONS_TOTAL.labels(
                model_version=MODEL_VERSION,
                outcome="error",
                contract_type=contract_type,
            ).inc()
            raise ValueError(result["error"])

        churn_prob = result["churn_probability"]
        outcome    = "churn" if result["churn_prediction"] == 1 else "no_churn"

        # Основний лічильник
        PREDICTIONS_TOTAL.labels(
            model_version=MODEL_VERSION,
            outcome=outcome,
            contract_type=contract_type,
        ).inc()

        # Розподіл впевненості моделі
        PREDICTION_CONFIDENCE.labels(
            model_version=MODEL_VERSION,
            outcome=outcome,
        ).observe(churn_prob)

        return PredictionResponse(
            churn_probability=churn_prob,
            churn_prediction=result["churn_prediction"],
            features_used=result["features_used"],
        )

    except ValueError:
        raise HTTPException(status_code=500, detail=str(result.get("error", "Unknown error")))
    except Exception as e:
        logger.error(f"Помилка під час прогнозу: {str(e)}", exc_info=True)
        PREDICTIONS_TOTAL.labels(
            model_version=MODEL_VERSION,
            outcome="error",
            contract_type=contract_type,
        ).inc()
        raise HTTPException(
            status_code=500,
            detail=f"Помилка обробки запиту: {str(e)}",
        )
