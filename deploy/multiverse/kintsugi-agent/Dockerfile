# Kintsugi Mutual Aid Agent
# ARM64-compatible base for Hetzner cax41
FROM python:3.12-slim

WORKDIR /app

# System deps for psycopg2 and numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
