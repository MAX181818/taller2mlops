# =========================================================
# docker/airflow/Dockerfile
# Imagen personalizada de Airflow para MLOps Proyecto 2
# Persona 2 | feat/p2-airflow
# =========================================================

FROM apache/airflow:2.8.1-python3.10

# Cambiar a root para instalar dependencias del sistema
USER root

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Volver al usuario airflow para instalar paquetes Python
USER airflow

# Copiar requirements e instalar
COPY docker/airflow/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copiar el módulo de training para que el DAG pueda importarlo
COPY training/ /opt/airflow/training/
