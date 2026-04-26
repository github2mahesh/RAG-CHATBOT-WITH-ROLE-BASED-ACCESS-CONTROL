FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libmagic1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/main.py .
COPY backend/guardrails.py .
COPY backend/cost_tracker.py .
COPY backend/build_db.py .
COPY data/ ./data/

EXPOSE 8000

CMD ["sh", "-c", "python build_db.py && uvicorn main:app --host 0.0.0.0 --port 8000"]