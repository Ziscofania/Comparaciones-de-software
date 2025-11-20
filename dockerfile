FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# Dependencias necesarias para numpy, scipy y streamlit
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no-root para evitar problemas de permisos con volúmenes
RUN useradd -m appuser

WORKDIR /app

# Copiamos solo requirements para usar la cache correctamente
COPY requirements.txt /app/

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Cambiamos a usuario normal
USER appuser

# Puerto de streamlit
EXPOSE 8501

# Variables de configuración de Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

CMD ["streamlit", "run", "app.py"]
