import os
import logging
import mlflow
import mlflow.pyfunc
import mlflow.sklearn

logger = logging.getLogger(__name__)

MODEL_NAME  = os.getenv("MLFLOW_MODEL_NAME", "diabetes_model")
MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "champion")

_model = None
_model_info = {}

def load_model():
    global _model, _model_info

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    _model = mlflow.sklearn.load_model(uri)

    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)

    _model_info = {
        "model_name": MODEL_NAME,
        "model_version": mv.version,
        "model_alias": MODEL_ALIAS,
    }

def try_load_model():
    """Intenta cargar el modelo sin lanzar excepción — se llama en startup."""
    try:
        load_model()
        logger.info("Modelo cargado correctamente en startup.")
    except Exception as e:
        logger.warning(f"Modelo no disponible en startup (se cargará en la primera predicción): {e}")

def reload_model():
    """Fuerza la recarga del modelo desde MLflow. Útil cuando cambia el champion."""
    global _model, _model_info
    _model = None
    _model_info = {}
    load_model()

def get_model():
    if _model is None:
        load_model()  # lanza excepción si sigue sin estar disponible → HTTP 500
    return _model

def get_model_info() -> dict:
    return _model_info