# Ontic Engine Discovery Engine
# Production-ready container image

FROM python:3.11-slim-bookworm AS base

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 physos \
    && useradd --uid 1000 --gid physos --shell /bin/bash --create-home physos

WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------------------------------------------
# Development stage
FROM base AS development

# Additional dev tools
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    isort \
    mypy \
    ipython

# Copy source
COPY --chown=physos:physos . .

USER physos

CMD ["python", "-m", "ontic.discovery", "serve", "--port", "8000"]

# -------------------------------------------------------------------
# Production stage
FROM base AS production

# Copy only necessary files
COPY --chown=physos:physos ontic/ ./ontic/
COPY --chown=physos:physos proofs/ ./proofs/
COPY --chown=physos:physos pyproject.toml ./

# Set ownership
RUN chown -R physos:physos /app

USER physos

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "ontic.discovery", "serve", "--port", "8000"]

# -------------------------------------------------------------------
# Minimal stage for edge deployment
FROM python:3.11-alpine AS minimal

ENV PYTHONUNBUFFERED=1

RUN apk add --no-cache \
    libffi \
    openssl

WORKDIR /app

COPY requirements-minimal.txt ./
RUN pip install --no-cache-dir -r requirements-minimal.txt

COPY --chown=1000:1000 ontic/ ./ontic/

RUN adduser -D -u 1000 physos
USER physos

EXPOSE 8000

CMD ["python", "-m", "ontic.discovery", "serve", "--port", "8000"]
