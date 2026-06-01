# Multi-stage production Dockerfile
# Compatible with: local Docker, Hugging Face Spaces, Docker Hub
#
# Build: docker build --provenance=false -t adaptive-ids .
# Run locally: docker run -p 7860:7860 adaptive-ids
# HF Spaces:   automatically builds and serves on port 7860

# ── Stage 1: Build ──────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install only runtime serving dependencies (no jupyter, pyarrow, pytest, etc.)
COPY requirements-serve.txt .
RUN pip install --no-cache-dir --user -r requirements-serve.txt

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.10-slim AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PATH=/home/appuser/.local/bin:/root/.local/bin:$PATH

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Create non-root user required by Hugging Face Spaces
RUN useradd -m -u 1000 appuser && \
    cp -r /root/.local /home/appuser/.local && \
    chown -R appuser:appuser /home/appuser/.local && \
    chown -R appuser:appuser /app

# Copy application source
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY pyproject.toml /app/

# Copy ONLY the exact ML assets needed for inference
# Preprocessor pipeline: ~7 KB
COPY data/processed/preprocessing_pipeline.joblib /app/data/processed/preprocessing_pipeline.joblib
# Serving model: ~1.6 MB (multiclass hybrid pipeline only)
COPY models/multiclass/hybrid_pipeline/model.joblib /app/models/multiclass/hybrid_pipeline/model.joblib

# Fix ownership for non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# HF Spaces requires port 7860; override locally with -e PORT=8000
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["python", "src/api/main.py"]
