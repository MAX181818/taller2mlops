import time, uuid, os
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Importaciones relativas dentro del paquete api/
from .schemas import PredictRequest, PredictResponse
from .model_loader import get_model, get_model_info, reload_model
from .db_logger import log_inference

app = FastAPI(title="Diabetes MLOps API", version="1.0")

# Métricas de Prometheus
REQUEST_COUNT = Counter("api_requests_total", "Total requests", ["endpoint","status"])
LATENCY = Histogram("api_latency_seconds", "Latency", ["endpoint"])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/model-info")
def model_info():
    return get_model_info()

@app.post("/reload-model")
def reload():
    reload_model()
    info = get_model_info()
    return {"status": "ok", "model_version": info["model_version"], "model_alias": info["model_alias"]}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.time()
    model = get_model()
    
    # 1. Convertir request a DataFrame
    import pandas as pd
    df = pd.DataFrame([req.dict()])
    
    # 2. Hacer predicción
    pred = int(model.predict(df)[0])
    proba = float(model.predict_proba(df)[0][1])
    
    # 3. Calcular tiempos y recolectar metadatos
    elapsed = (time.time() - start) * 1000
    rid = str(uuid.uuid4())
    info = get_model_info()
    
    # 4. Registrar métricas de Prometheus
    REQUEST_COUNT.labels("/predict", "200").inc()
    LATENCY.labels("/predict").observe(elapsed / 1000)
    
    # 5. Guardar asincrónicamente en la BD PostgreSQL (inference_logs)
    log_inference(rid, req.dict(), pred, proba, elapsed, info)
    
    # 6. Retornar el contrato acordado
    return PredictResponse(
        prediction=pred, 
        probability=proba,
        model_name=info["model_name"],
        model_version=info["model_version"],
        model_alias=info["model_alias"],
        response_time_ms=elapsed, 
        request_id=rid
    )