# ── Stage 1: Build ──────────────────────────────────────────────────────
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ── Stage 2: Runtime ────────────────────────────────────────────────────
FROM python:3.11-slim
WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY app/        ./app/
COPY data/       ./data/
COPY train.py    ./

# Pre-train models during image build (models baked into image)
# Comment this out if you prefer volume-mounting pre-trained models
RUN python train.py

ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    AUTH_REQUIRED=false \
    JWT_SECRET_KEY=change_me_in_production

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

WORKDIR /app/app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
