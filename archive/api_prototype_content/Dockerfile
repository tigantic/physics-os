# ═══════════════════════════════════════════════════════════════════
# HyperTensor Physics API — GPU-accelerated inference container
#
# Build:
#   docker build -f api/Dockerfile -t hypertensor-api:1.0.0 .
#
# Run:
#   docker run --gpus all -p 8000:8000 \
#     -e HYPERTENSOR_API_KEYS=your-secret-key \
#     hypertensor-api:1.0.0
#
# Run without GPU:
#   docker run -p 8000:8000 \
#     -e HYPERTENSOR_API_KEYS=your-secret-key \
#     -e HYPERTENSOR_DEVICE=cpu \
#     hypertensor-api:1.0.0
# ═══════════════════════════════════════════════════════════════════

FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04 AS base

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3-pip \
        && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r hypertensor && useradd -r -g hypertensor -m hypertensor

WORKDIR /app

# ── Dependencies ────────────────────────────────────────────────────
COPY requirements-lock.txt ./
RUN python3.12 -m pip install --no-cache-dir --break-system-packages \
        -r requirements-lock.txt \
        pydantic-settings uvicorn[standard]

# ── Application code ────────────────────────────────────────────────
# Copy ONLY the packages needed for serving — not scripts, notebooks,
# data, or research artifacts.
COPY tensornet/ ./tensornet/
COPY api/ ./api/

# ── Security ────────────────────────────────────────────────────────
# Drop to non-root
USER hypertensor

# ── Runtime ─────────────────────────────────────────────────────────
EXPOSE 8000

ENV HYPERTENSOR_HOST=0.0.0.0 \
    HYPERTENSOR_PORT=8000 \
    HYPERTENSOR_WORKERS=1 \
    HYPERTENSOR_LOG_LEVEL=info \
    HYPERTENSOR_REQUIRE_AUTH=true

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python3.12 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/health')" || exit 1

ENTRYPOINT ["python3.12", "-m", "uvicorn", "api.main:app"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
