FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pharos library
COPY pack_builder.py router.py encoder.py injector.py evaluator.py ./
COPY packs/ ./packs/

# API server
COPY deploy/server.py .

EXPOSE 8080

CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8080"]
