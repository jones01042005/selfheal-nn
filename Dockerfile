# ─────────────────────────────────────────────────────────────────────────────
# SelfHealingNN — Production Dockerfile
# Base: python:3.10-slim  (Debian Bookworm)
# PyTorch: CPU-only wheel (~200 MB) for portable edge-AI deployment
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim AS base

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# ── Python dependencies (CPU-only torch first to keep image small) ─────────────
COPY requirements-prod.txt ./requirements.txt

# Install CPU-only PyTorch from the official wheel index, then the rest
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY ae_model.py        ./
COPY train.py           ./
COPY fault.py           ./
COPY heal.py            ./
COPY train_ae.py        ./
COPY main.py            ./
COPY api.py             ./

# ── Model checkpoints (runtime data — not training artifacts) ─────────────────
COPY models/baseline_cnn.pth              models/
COPY models/autoencoders/ae_fc1.pth       models/autoencoders/
COPY models/autoencoders/ae_fc2.pth       models/autoencoders/

# ── Static frontend files ─────────────────────────────────────────────────────
COPY static/    static/

# ── CIFAR-10 dataset directory (will auto-download if empty) ──────────────────
RUN mkdir -p datasets experiments

# ── Permissions ───────────────────────────────────────────────────────────────
RUN chown -R appuser:appuser /app
USER appuser

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 8000
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Use uvicorn with a single worker (avoids duplicating the large model in RAM)
CMD ["sh", "-c", "exec python -m uvicorn api:app --host 0.0.0.0 --port \"${PORT:-8000}\" --workers 1 --timeout-keep-alive 120"]
the 