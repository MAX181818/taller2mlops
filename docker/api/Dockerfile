FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema requeridas para PostgreSQL
RUN apt-get update && apt-get install -y libpq-dev gcc && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias de Python
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la API
COPY api/ ./api/

EXPOSE 8000

# Comando para arrancar FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]