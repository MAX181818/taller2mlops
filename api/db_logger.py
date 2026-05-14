import os
import json
from sqlalchemy import create_engine, text

# Conexión usando las variables globales de la sección 3.1
DB_HOST = os.getenv("POSTGRES_HOST", "postgres-service")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_USER = os.getenv("POSTGRES_USER", "mlops_user")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "mlops_pass")
DB_DB   = os.getenv("POSTGRES_DB", "mlops_db")

engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_DB}")

def log_inference(request_id: str, input_data: dict, prediction: int, probability: float, response_time_ms: float, model_info: dict):
    try:
        query = text("""
            INSERT INTO inference.inference_logs 
            (request_id, input_data, prediction, probability, model_name, model_version, model_alias, response_time_ms, status_code) 
            VALUES 
            (:req_id, :input_data, :pred, :prob, :m_name, :m_ver, :m_alias, :resp_time, :status)
        """)
        with engine.begin() as conn:
            conn.execute(query, {
                "req_id": request_id,
                "input_data": json.dumps(input_data),
                "pred": prediction,
                "prob": probability,
                "m_name": model_info.get("model_name"),
                "m_ver": str(model_info.get("model_version")),
                "m_alias": model_info.get("model_alias"),
                "resp_time": response_time_ms,
                "status": 200
            })
    except Exception as e:
        print(f"Error al registrar la inferencia en la BD: {e}")