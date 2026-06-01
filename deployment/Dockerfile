# Multi-stage production Dockerfile for Network ML Platform
# Stage 1: Build dependencies
FROM python:3.10-slim AS builder

WORKDIR /app

# Prevent python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system utilities needed for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Final minimal runtime environment
FROM python:3.10-slim AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app /app

# Ensure local user path is added to environment
ENV PATH=/root/.local/bin:$PATH

# Copy modular workspace directories
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY tests/ /app/tests/
COPY pyproject.toml /app/

# Port configurations (API standard)
EXPOSE 8000

# Set entry point to run training as default, but allows overriding for API serving or testing
CMD ["python", "src/models/train.py"]
