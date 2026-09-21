FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    INSPECTAI_MODEL_PATH=/app/artifacts/best_model.pt

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY artifacts ./artifacts
COPY data ./data
EXPOSE 8000
CMD ["uvicorn", "inspectai.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

