# PaymentPipeline — Production Dockerfile
# Multi-stage build: builder → runtime

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Non-root user for security
RUN groupadd -r paymentpipeline && useradd -r -g paymentpipeline ppuser

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy application code
COPY app/      ./app/
COPY ai_agents/ ./ai_agents/
COPY frontend/  ./frontend/
COPY run.py    ./run.py

# Data directory for SQLite
RUN mkdir -p /data && chown ppuser:paymentpipeline /data

USER ppuser

EXPOSE 5000 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health/live')"

CMD ["python", "run.py"]
