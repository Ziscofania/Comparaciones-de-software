FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

CMD ["streamlit", "run", "app.py"]
