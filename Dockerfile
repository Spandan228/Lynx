# ===========================================================================
# Stage 1: Build & Dependency Caching
# ===========================================================================
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build dependencies for compiling binary extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment for isolated layer caching
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python requirements
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Pre-download FastEmbed embedding model during build to eliminate runtime download delay
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"

# ===========================================================================
# Stage 2: Production Minimal Runtime
# ===========================================================================
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    HF_HOME="/app/.cache/huggingface" \
    FASTEMBED_CACHE_PATH="/app/.cache/fastembed"

# Install lightweight runtime utilities for container healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Security: Create non-root system user and group (UID/GID 10001)
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

WORKDIR /app

# Copy virtual environment and cached FastEmbed model from builder stage
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /root/.cache /app/.cache

# Copy application source files (src/lynx package + static frontend + data)
COPY src/ ./src/
COPY static/ ./static/
COPY requirements.txt pyproject.toml ./

# Create necessary persistent storage directories and assign permissions to non-root appuser
RUN mkdir -p /app/data /app/qdrant_storage /app/.cache && \
    chown -R appuser:appgroup /app /app/.cache

# Switch to non-root execution
USER appuser

# Expose FastAPI backend (8000) and Streamlit frontend (8501) ports
EXPOSE 8000 8501

# Healthcheck targeting FastAPI /health endpoint
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default Command: Start FastAPI Backend Server
CMD ["uvicorn", "lynx.app:app", "--host", "0.0.0.0", "--port", "8000"]
