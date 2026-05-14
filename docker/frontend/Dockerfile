FROM python:3.9-slim

WORKDIR /app

# Copiar e instalar dependencias de Python
COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de Streamlit
COPY frontend/ ./frontend/

EXPOSE 8501

# Comando para arrancar la interfaz
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]