FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GIT_PYTHON_REFRESH=quiet \
    MODEL_DIR=/models

WORKDIR /app

# Dependencies first: this layer is cached until requirements change.
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# Shared feature code (same code as training) then the API itself.
COPY pyproject.toml .
COPY src ./src
RUN pip install --no-deps .
COPY api ./api

# Non-root user; /models is owned by it so the named volume inherits the ownership.
RUN useradd --create-home --uid 10001 appuser && mkdir -p /models && chown appuser /models
USER appuser

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
