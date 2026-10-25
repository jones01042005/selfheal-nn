
# ─────────────────────────────────────────────────────────────────────────────
# SelfHealingNN — Production Dockerfile
# Base: Python 3.10 slim
# PyTorch: CPU-only for portable deployment
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Install Python dependencies
COPY requirements-prod.txt ./requirements.txt

# Install CPU-only PyTorch and torchvision
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Prepare CIFAR-10 before copying application files so this layer can be reused
# when only the API or frontend changes.
COPY prepare_cifar10.py ./
RUN mkdir -p datasets experiments \
    && python prepare_cifar10.py

# Copy application source
COPY ae_model.py ./
COPY train.py ./
COPY fault.py ./
COPY heal.py ./
COPY train_ae.py ./
COPY main.py ./
COPY api.py ./

# Copy trained model checkpoints
COPY models/baseline_cnn.pth models/
COPY models/autoencoders/ae_fc1.pth models/autoencoders/
COPY models/autoencoders/ae_fc2.pth models/autoencoders/

# Copy frontend assets
COPY static/ static/

# Set permissions
RUN chown -R appuser:appuser /app

USER appuser

# Render web service port
EXPOSE 10000

ENV PORT=10000
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Start FastAPI with one worker to reduce memory consumption
CMD ["sh", "-c", "exec python -m uvicorn api:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1 --timeout-keep-alive 120"]
