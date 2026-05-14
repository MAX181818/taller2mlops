# =========================================================
# dags/diabetes_pipeline.py
# DAG principal — MLOps Proyecto 2
# Persona 2 | feat/p2-airflow
#
# Flujo:
#   t1_validate_source
#   └─► t2_load_raw_batch
#           └─► t3_validate_quality
#                   └─► t4_preprocess
#                           └─► t5_store_clean
#                                   └─► t6_split_data
#                                           └─► t7_train_model
#                                                   └─► t8_register_mlflow
#                                                           └─► t9_compare_models
#                                                                   └─► t10_promote_champion
# =========================================================

import os
import hashlib
import json
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

from airflow import DAG
from airflow.operators.python import PythonOperator

# ── Constantes (desde variables de entorno — .env.example) ───────────────────

DAG_ID        = os.getenv("AIRFLOW_DAG_ID",        "diabetes_mlops_pipeline")
BATCH_SIZE    = int(os.getenv("BATCH_SIZE",         15000))
SOURCE_FILE   = os.getenv("SOURCE_FILE",            "/opt/airflow/data/diabetic_data.csv")

POSTGRES_HOST = os.getenv("POSTGRES_HOST",          "postgres-service")
POSTGRES_PORT = os.getenv("POSTGRES_PORT",          "5432")
POSTGRES_USER = os.getenv("POSTGRES_USER",          "mlops_user")
POSTGRES_PASS = os.getenv("POSTGRES_PASSWORD",      "mlops_pass")
POSTGRES_DB   = os.getenv("POSTGRES_DB",            "mlops_db")

MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI",    "http://mlflow-service:5000")
EXPERIMENT    = os.getenv("MLFLOW_EXPERIMENT_NAME", "diabetes_experiment")
MODEL_NAME    = os.getenv("MLFLOW_MODEL_NAME",      "diabetes_model")
MODEL_ALIAS   = os.getenv("MLFLOW_MODEL_ALIAS",     "champion")

DB_CONN_STR = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASS}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# ── Argumentos por defecto del DAG ───────────────────────────────────────────

default_args = {
    "owner":            "persona2",
    "depends_on_past":  False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}

# ─────────────────────────────────────────────────────────────────────────────
# TAREA 1 — Validar archivo fuente
# ─────────────────────────────────────────────────────────────────────────────

def t1_validate_source(**context):
    """
    Verifica que el archivo CSV existe y no está vacío.
    Calcula cuántos lotes quedan pendientes por cargar.
    Publica en XCom: total_rows, loaded_rows, pending_batches, next_offset.
    """
    if not os.path.isfile(SOURCE_FILE):
        raise FileNotFoundError(f"Archivo fuente no encontrado: {SOURCE_FILE}")

    file_size = os.path.getsize(SOURCE_FILE)
    if file_size == 0:
        raise ValueError(f"Archivo fuente vacío: {SOURCE_FILE}")

    # Contar filas del CSV sin cargar todo en memoria
    total_rows = sum(1 for _ in open(SOURCE_FILE)) - 1  # -1 por header
    logging.info(f"Archivo validado: {SOURCE_FILE} | {total_rows} filas | {file_size} bytes")

    # Consultar cuántas filas ya fueron cargadas en raw
    engine = create_engine(DB_CONN_STR)
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) FROM raw.diabetes_raw")
        )
        loaded_rows = result.scalar() or 0

    pending = total_rows - loaded_rows
    next_offset = loaded_rows
    pending_batches = max(0, -(-pending // BATCH_SIZE))  # ceil division

    logging.info(
        f"Filas cargadas: {loaded_rows} | Pendientes: {pending} | "
        f"Próximo offset: {next_offset} | Lotes pendientes: {pending_batches}"
    )

    if pending <= 0:
        logging.warning("No hay filas nuevas que cargar. DAG completará sin insertar datos.")

    context["ti"].xcom_push(key="total_rows",      value=total_rows)
    context["ti"].xcom_push(key="loaded_rows",     value=loaded_rows)
    context["ti"].xcom_push(key="next_offset",     value=next_offset)
    context["ti"].xcom_push(key="pending_batches", value=pending_batches)
    context["ti"].xcom_push(key="source_file",     value=SOURCE_FILE)


# ─────────────────────────────────────────────────────────────────────────────
# TAREA 2 — Carga incremental en raw.diabetes_raw
# ─────────────────────────────────────────────────────────────────────────────

def _compute_row_hash(row: pd.Series) -> str:
    """MD5 de la fila completa serializada. Garantiza deduplicación."""
    row_str = "|".join(str(v) for v in row.values)
    return hashlib.md5(row_str.encode("utf-8")).hexdigest()


def t2_load_raw_batch(**context):
    """
    Lee exactamente BATCH_SIZE filas nuevas del CSV (usando offset).
    Calcula row_hash para deduplicación.
    Inserta solo filas cuyo row_hash no exista en raw.diabetes_raw.
    """
    ti          = context["ti"]
    next_offset = ti.xcom_pull(key="next_offset", task_ids="t1_validate_source")
    source_file = ti.xcom_pull(key="source_file",  task_ids="t1_validate_source")

    # Leer el batch desde el offset
    df_batch = pd.read_csv(
        source_file,
        skiprows=range(1, next_offset + 1),  # salta filas ya procesadas
        nrows=BATCH_SIZE,
    )

    if df_batch.empty:
        logging.warning("Batch vacío — no hay filas nuevas que insertar.")
        context["ti"].xcom_push(key="batch_id",    value=None)
        context["ti"].xcom_push(key="batch_rows",  value=0)
        return

    # Calcular row_hash por fila
    df_batch["row_hash"] = df_batch.apply(_compute_row_hash, axis=1)

    # Generar batch_id único para esta ejecución
    run_ts   = context["execution_date"].strftime("%Y%m%d_%H%M%S")
    batch_id = f"batch_{run_ts}"

    engine = create_engine(DB_CONN_STR)

    # Obtener hashes ya existentes para evitar duplicados
    with engine.connect() as conn:
        existing = pd.read_sql(
            "SELECT row_hash FROM raw.diabetes_raw", conn
        )["row_hash"].tolist()

    df_new = df_batch[~df_batch["row_hash"].isin(existing)].copy()
    duplicates = len(df_batch) - len(df_new)
    logging.info(f"Filas en batch: {len(df_batch)} | Duplicados omitidos: {duplicates} | Nuevas: {len(df_new)}")

    if df_new.empty:
        logging.warning("Todas las filas del batch ya existen. No se insertó nada.")
        context["ti"].xcom_push(key="batch_id",   value=batch_id)
        context["ti"].xcom_push(key="batch_rows", value=0)
        return

    # Construir registros para raw.diabetes_raw
    records = []
    for _, row in df_new.iterrows():
        raw_data = {
            col: (None if pd.isna(val) else val)
            for col, val in row.items()
            if col != "row_hash"
        }
        records.append({
            "batch_id":        batch_id,
            "load_timestamp":  datetime.utcnow(),
            "source_file":     source_file,
            "row_hash":        row["row_hash"],
            "status":          "loaded",
            "raw_data":        json.dumps(raw_data),
        })

    df_insert = pd.DataFrame(records)

    with engine.begin() as conn:
        df_insert.to_sql(
            "diabetes_raw",
            conn,
            schema="raw",
            if_exists="append",
            index=False,
            method="multi",
        )

    logging.info(f"Batch '{batch_id}' insertado: {len(df_insert)} filas en raw.diabetes_raw")

    context["ti"].xcom_push(key="batch_id",   value=batch_id)
    context["ti"].xcom_push(key="batch_rows", value=len(df_insert))


# ─────────────────────────────────────────────────────────────────────────────
# TAREA 3 — Validación de calidad
# ─────────────────────────────────────────────────────────────────────────────

def t3_validate_quality(**context):
    """
    Lee el batch recién insertado desde raw.diabetes_raw.
    Verifica que ninguna columna supere el 50% de nulos.
    Actualiza status='validated' en las filas del batch.
    """
    ti       = context["ti"]
    batch_id = ti.xcom_pull(key="batch_id",   task_ids="t2_load_raw_batch")
    batch_rows = ti.xcom_pull(key="batch_rows", task_ids="t2_load_raw_batch")

    if not batch_id or batch_rows == 0:
        logging.warning("Batch vacío — saltando validación de calidad.")
        return

    engine = create_engine(DB_CONN_STR)

    # Leer raw_data del batch para validar
    with engine.connect() as conn:
        df_raw = pd.read_sql(
            text("SELECT raw_data FROM raw.diabetes_raw WHERE batch_id = :bid"),
            conn,
            params={"bid": batch_id},
        )

    # Deserializar JSONB → DataFrame (psycopg2 ya lo convierte a dict)
    records = [r if isinstance(r, dict) else json.loads(r) for r in df_raw["raw_data"]]
    df = pd.DataFrame(records).replace("?", np.nan)

    # Validar solo columnas numéricas del contrato (las columnas con alta
    # tasa de nulos por diseño — weight, payer_code, A1Cresult, max_glu_serum —
    # se eliminan o se imputan en preprocessing, no se validan aquí)
    COLS_TO_VALIDATE = [
        "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications", "number_outpatient", "number_emergency",
        "number_inpatient", "number_diagnoses", "age",
        "gender", "race", "admission_type_id",
        "discharge_disposition_id", "admission_source_id",
        "metformin", "insulin", "readmitted",
    ]
    df_check = df[[c for c in COLS_TO_VALIDATE if c in df.columns]]
    null_ratios = df_check.isnull().mean()
    bad_cols    = null_ratios[null_ratios > 0.5].to_dict()

    if bad_cols:
        raise ValueError(
            f"Validación fallida — columnas con >50% nulos: {bad_cols}"
        )

    logging.info(f"Validación de calidad OK para batch '{batch_id}'")

    # Actualizar status → 'validated'
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE raw.diabetes_raw SET status = 'validated' "
                "WHERE batch_id = :bid"
            ),
            {"bid": batch_id},
        )

    context["ti"].xcom_push(key="validated_batch_id", value=batch_id)


# ─────────────────────────────────────────────────────────────────────────────
# TAREA 4 — Preprocesamiento y feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def t4_preprocess(**context):
    """
    Lee las filas validadas de raw.diabetes_raw, aplica preprocessing y
    escribe directamente en clean.diabetes_clean. Solo pasa batch_id por XCom.
    """
    ti       = context["ti"]
    batch_id = ti.xcom_pull(key="validated_batch_id", task_ids="t3_validate_quality")

    if not batch_id:
        logging.warning("Sin batch validado — saltando preprocesamiento.")
        return

    engine = create_engine(DB_CONN_STR)

    with engine.connect() as conn:
        df_raw = pd.read_sql(
            text("SELECT raw_data FROM raw.diabetes_raw WHERE batch_id = :bid"),
            conn,
            params={"bid": batch_id},
        )

    records = [r if isinstance(r, dict) else json.loads(r) for r in df_raw["raw_data"]]
    df = pd.DataFrame(records)

    from training.preprocessing import run_preprocessing
    df_clean = run_preprocessing(df)
    df_clean["batch_id"]            = batch_id
    df_clean["processed_timestamp"] = datetime.utcnow()

    logging.info(f"Preprocessing completado: {df_clean.shape[0]} filas, {df_clean.shape[1]} columnas")

    # Escritura directa a BD — no se pasan datos por XCom
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM clean.diabetes_clean WHERE batch_id = :bid"),
            {"bid": batch_id},
        )
        df_clean.to_sql(
            "diabetes_clean",
            conn,
            schema="clean",
            if_exists="append",
            index=False,
            method="multi",
        )

    logging.info(f"Batch '{batch_id}' escrito en clean.diabetes_clean: {len(df_clean)} filas")
    context["ti"].xcom_push(key="clean_batch_id", value=batch_id)


# ─────────────────────────────────────────────────────────────────────────────
# TAREA 5 — Verificar almacenamiento en clean.diabetes_clean
# ─────────────────────────────────────────────────────────────────────────────

def t5_store_clean(**context):
    """
    Verifica que el batch fue escrito correctamente en clean.diabetes_clean.
    Los datos ya fueron insertados por t4 — aquí solo se valida el conteo.
    """
    ti       = context["ti"]
    batch_id = ti.xcom_pull(key="clean_batch_id", task_ids="t4_preprocess")

    if not batch_id:
        logging.warning("Sin batch_id de t4 — saltando verificación.")
        return

    engine = create_engine(DB_CONN_STR)

    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM clean.diabetes_clean WHERE batch_id = :bid"),
            {"bid": batch_id},
        ).scalar()

    if count == 0:
        raise ValueError(f"Batch '{batch_id}' no encontrado en clean.diabetes_clean")

    logging.info(f"Verificación OK: batch '{batch_id}' tiene {count} filas en clean.diabetes_clean")
    context["ti"].xcom_push(key="stored_batch_id", value=batch_id)


# ─────────────────────────────────────────────────────────────────────────────
# TAREA 6 — Split train / val / test
# ─────────────────────────────────────────────────────────────────────────────

def t6_split_data(**context):
    """
    Lee TODOS los datos acumulados en clean.diabetes_clean.
    Hace split 70/15/15 estratificado por target.
    Guarda los índices en /tmp para que t7 los consuma.
    """
    from sklearn.model_selection import train_test_split

    engine = create_engine(DB_CONN_STR)

    with engine.connect() as conn:
        df = pd.read_sql("SELECT * FROM clean.diabetes_clean", conn)

    if df.empty:
        raise ValueError("clean.diabetes_clean está vacía — no se puede hacer split.")

    FEATURE_COLS = [
        "age", "time_in_hospital", "num_lab_procedures",
        "num_procedures", "num_medications", "number_outpatient",
        "number_emergency", "number_inpatient", "number_diagnoses",
        "race_encoded", "gender_encoded", "admission_type_encoded",
        "discharge_encoded", "admission_source_encoded",
        "a1c_result_encoded", "metformin_encoded", "insulin_encoded",
    ]
    TARGET_COL = "readmitted"

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # Split 70/30 → luego 30 en 50/50 → 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    logging.info(
        f"Split → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}"
    )

    # Persistir en /tmp para t7
    split_path = "/tmp/diabetes_split"
    os.makedirs(split_path, exist_ok=True)

    X_train.to_csv(f"{split_path}/X_train.csv", index=False)
    X_val.to_csv(f"{split_path}/X_val.csv",     index=False)
    X_test.to_csv(f"{split_path}/X_test.csv",   index=False)
    y_train.to_csv(f"{split_path}/y_train.csv", index=False)
    y_val.to_csv(f"{split_path}/y_val.csv",     index=False)
    y_test.to_csv(f"{split_path}/y_test.csv",   index=False)

    context["ti"].xcom_push(key="split_path",   value=split_path)
    context["ti"].xcom_push(key="train_size",   value=len(X_train))
    context["ti"].xcom_push(key="val_size",     value=len(X_val))
    context["ti"].xcom_push(key="test_size",    value=len(X_test))


# ─────────────────────────────────────────────────────────────────────────────
# TAREA 7 — Entrenar modelo
# ─────────────────────────────────────────────────────────────────────────────

def t7_train_model(**context):
    """
    Llama a training.train.run_training() — implementado por Persona 3.
    Recibe run_id, model_version y métricas.
    Consume split_path de t6 para usar los archivos CSV pre-divididos (70/15/15).
    """
    ti         = context["ti"]
    batch_id   = ti.xcom_pull(key="stored_batch_id", task_ids="t5_store_clean")
    split_path = ti.xcom_pull(key="split_path",      task_ids="t6_split_data")

    # Interfaz acordada con Persona 3
    from training.train import run_training

    result = run_training(
        db_conn=POSTGRES_HOST,
        mlflow_uri=MLFLOW_URI,
        experiment_name=EXPERIMENT,
        model_name=MODEL_NAME,
        batch_id=batch_id,
        split_path=split_path,
    )

    logging.info(
        f"Entrenamiento completado | run_id: {result['run_id']} | "
        f"version: {result['model_version']} | métricas: {result['metrics']}"
    )

    context["ti"].xcom_push(key="run_id",        value=result["run_id"])
    context["ti"].xcom_push(key="model_version", value=result["model_version"])
    context["ti"].xcom_push(key="metrics",       value=json.dumps(result["metrics"]))


# ─────────────────────────────────────────────────────────────────────────────
# TAREA 8 — Validar registro en MLflow
# ─────────────────────────────────────────────────────────────────────────────

def t8_register_mlflow(**context):
    """
    Verifica que el run_id fue registrado correctamente en MLflow.
    Loguea parámetros y métricas en los logs de Airflow.
    """
    import mlflow

    ti         = context["ti"]
    run_id     = ti.xcom_pull(key="run_id",        task_ids="t7_train_model")
    metrics_js = ti.xcom_pull(key="metrics",       task_ids="t7_train_model")
    version    = ti.xcom_pull(key="model_version", task_ids="t7_train_model")

    if not run_id:
        raise ValueError("run_id no encontrado — t7 no completó correctamente.")

    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.MlflowClient()

    # Verificar que el run existe
    run = client.get_run(run_id)
    if run.info.status != "FINISHED":
        raise ValueError(f"Run {run_id} no está en estado FINISHED: {run.info.status}")

    metrics = json.loads(metrics_js)
    logging.info(
        f"MLflow run verificado | run_id: {run_id} | "
        f"model_version: {version} | "
        f"val_roc_auc: {metrics.get('val_roc_auc')} | "
        f"val_f1: {metrics.get('val_f1')} | "
        f"val_recall: {metrics.get('val_recall')}"
    )

    context["ti"].xcom_push(key="verified_run_id", value=run_id)


# ─────────────────────────────────────────────────────────────────────────────
# TAREA 9 — Comparar contra el modelo champion
# ─────────────────────────────────────────────────────────────────────────────

def t9_compare_models(**context):
    """
    Llama a training.promote.compare_with_champion() — implementado por Persona 3.
    Publica si el nuevo modelo supera al champion actual.
    """
    ti     = context["ti"]
    run_id = ti.xcom_pull(key="verified_run_id", task_ids="t8_register_mlflow")

    from training.promote import compare_with_champion

    should_promote = compare_with_champion(
        run_id=run_id,
        model_name=MODEL_NAME,
        alias=MODEL_ALIAS,
        metric="val_roc_auc",
    )

    logging.info(
        f"Comparación con champion | run_id: {run_id} | "
        f"¿Promover?: {should_promote}"
    )

    context["ti"].xcom_push(key="should_promote", value=should_promote)
    context["ti"].xcom_push(key="run_id_to_promote", value=run_id)


# ─────────────────────────────────────────────────────────────────────────────
# TAREA 10 — Promover champion
# ─────────────────────────────────────────────────────────────────────────────

def t10_promote_champion(**context):
    """
    Si t9 indica que el nuevo modelo es mejor, llama a promote_champion().
    Registra en logs qué modelo fue promovido y con qué métrica.
    """
    ti             = context["ti"]
    should_promote = ti.xcom_pull(key="should_promote",    task_ids="t9_compare_models")
    run_id         = ti.xcom_pull(key="run_id_to_promote", task_ids="t9_compare_models")
    metrics_js     = ti.xcom_pull(key="metrics",           task_ids="t7_train_model")

    if not should_promote:
        logging.info(
            f"Modelo run_id={run_id} NO supera al champion actual. "
            "No se realiza promoción."
        )
        return

    from training.promote import promote_champion

    promote_champion(
        run_id=run_id,
        model_name=MODEL_NAME,
        alias=MODEL_ALIAS,
    )

    metrics = json.loads(metrics_js) if metrics_js else {}
    logging.info(
        f"✅ Nuevo champion promovido | run_id: {run_id} | "
        f"model: {MODEL_NAME} | alias: {MODEL_ALIAS} | "
        f"val_roc_auc: {metrics.get('val_roc_auc')}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# DEFINICIÓN DEL DAG
# ─────────────────────────────────────────────────────────────────────────────

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Pipeline MLOps: ingesta incremental → limpieza → entrenamiento → promoción",
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,          # No ejecutar fechas pasadas al activar el DAG
    max_active_runs=1,      # Evita ejecuciones paralelas que dupliquen datos
    tags=["mlops", "diabetes", "p2"],
) as dag:

    task_t1 = PythonOperator(
        task_id="t1_validate_source",
        python_callable=t1_validate_source,
        provide_context=True,
    )

    task_t2 = PythonOperator(
        task_id="t2_load_raw_batch",
        python_callable=t2_load_raw_batch,
        provide_context=True,
    )

    task_t3 = PythonOperator(
        task_id="t3_validate_quality",
        python_callable=t3_validate_quality,
        provide_context=True,
    )

    task_t4 = PythonOperator(
        task_id="t4_preprocess",
        python_callable=t4_preprocess,
        provide_context=True,
    )

    task_t5 = PythonOperator(
        task_id="t5_store_clean",
        python_callable=t5_store_clean,
        provide_context=True,
    )

    task_t6 = PythonOperator(
        task_id="t6_split_data",
        python_callable=t6_split_data,
        provide_context=True,
    )

    task_t7 = PythonOperator(
        task_id="t7_train_model",
        python_callable=t7_train_model,
        provide_context=True,
    )

    task_t8 = PythonOperator(
        task_id="t8_register_mlflow",
        python_callable=t8_register_mlflow,
        provide_context=True,
    )

    task_t9 = PythonOperator(
        task_id="t9_compare_models",
        python_callable=t9_compare_models,
        provide_context=True,
    )

    task_t10 = PythonOperator(
        task_id="t10_promote_champion",
        python_callable=t10_promote_champion,
        provide_context=True,
    )

    # Dependencias lineales
    (
        task_t1
        >> task_t2
        >> task_t3
        >> task_t4
        >> task_t5
        >> task_t6
        >> task_t7
        >> task_t8
        >> task_t9
        >> task_t10
    )