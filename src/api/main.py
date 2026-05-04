import warnings
# Suppress Pydantic v2 warnings from transitive dependencies (e.g., LangChain)
warnings.filterwarnings("ignore", message=".*protected namespace.*", category=UserWarning)

from fastapi import FastAPI, HTTPException
from datetime import datetime
import logging

# Імпорти з власного модуля
from src.api.models import CustomerFeatures, PredictionResponse
from src.api import predict as predict_module

# Налаштування логування (корисно в контейнері)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="API для прогнозування відтоку клієнтів (churn prediction)",
    version="1.0.0",
    docs_url="/docs",          # Swagger UI
    redoc_url="/redoc",        # ReDoc (додатково)
)

# Перевірка моделі при старті (логування)
@app.on_event("startup")
async def startup_event():
    if predict_module.model is None:
        logger.error("Модель не завантажилася при старті API!")
        # Можна навіть підняти виняток, якщо критичний запуск без моделі:
        # raise RuntimeError("Не вдалося завантажити модель churn")
    else:
        logger.info("Модель успішно завантажена при старті API")

@app.get("/health")
def health():
    """
    Перевірка стану сервісу та наявності моделі
    """
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
    """
    Прогноз ймовірності відтоку клієнта.
    Надішліть JSON з ознаками клієнта.
    """
    try:
        # Перетворюємо Pydantic-модель у dict
        input_data = features.dict()

        # Виклик прогнозу
        result = predict_module.predict_churn(input_data)

        if "error" in result:
            raise ValueError(result["error"])

        return PredictionResponse(
            churn_probability=result["churn_probability"],
            churn_prediction=result["churn_prediction"],
            features_used=result["features_used"]
        )

    except Exception as e:
        logger.error(f"Помилка під час прогнозу: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Помилка обробки запиту: {str(e)}"
        )
