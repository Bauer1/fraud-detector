FROM python:3.12-slim

WORKDIR /app

# System dependencies for LightGBM / OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config/ config/
COPY src/ src/
COPY dashboard/ dashboard/
COPY pyproject.toml README.md ./

RUN mkdir -p data/raw data/processed models/artifacts reports

EXPOSE 8000 8501

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
