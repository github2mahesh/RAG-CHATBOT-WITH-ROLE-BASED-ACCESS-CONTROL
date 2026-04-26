FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libmagic1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the backend needs
COPY backend/main.py .
COPY backend/guardrails.py .
COPY data/ ./data/
COPY chroma_db/ ./chroma_db/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]