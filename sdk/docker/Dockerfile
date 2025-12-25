# HyperTensor Docker Images
#
# Multi-stage build for optimized production images
#
# Build targets:
#   - base: Core Python package
#   - cuda: GPU-enabled version
#   - jupyter: Interactive notebook environment
#   - server: Production API server
#
# Usage:
#   docker build -t hypertensor .
#   docker build --target cuda -t hypertensor:cuda .
#   docker build --target jupyter -t hypertensor:jupyter .

# =============================================================================
# BASE IMAGE
# =============================================================================
FROM python:3.11-slim as base

# Labels
LABEL org.opencontainers.image.title="HyperTensor"
LABEL org.opencontainers.image.description="QTT-compressed tensor network library"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.vendor="Tigantic Labs"
LABEL org.opencontainers.image.source="https://github.com/tigantic/HyperTensor"
LABEL org.opencontainers.image.licenses="Apache-2.0"

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash hypertensor
USER hypertensor
WORKDIR /home/hypertensor

# Install Python dependencies
COPY --chown=hypertensor:hypertensor requirements-lock.txt .
RUN pip install --user -r requirements-lock.txt

# Install HyperTensor
COPY --chown=hypertensor:hypertensor . /home/hypertensor/hypertensor
RUN pip install --user -e /home/hypertensor/hypertensor

# Add user bin to PATH
ENV PATH="/home/hypertensor/.local/bin:${PATH}"

# Default command
CMD ["python", "-c", "import tensornet; print(f'HyperTensor v{tensornet.__version__}')"]

# =============================================================================
# CUDA IMAGE
# =============================================================================
FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04 as cuda

# Labels
LABEL org.opencontainers.image.title="HyperTensor CUDA"
LABEL org.opencontainers.image.description="GPU-enabled HyperTensor"

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-venv \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Create non-root user
RUN useradd --create-home --shell /bin/bash hypertensor
USER hypertensor
WORKDIR /home/hypertensor

# Install PyTorch with CUDA
RUN pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
COPY --chown=hypertensor:hypertensor requirements-lock.txt .
RUN pip install --user -r requirements-lock.txt

# Install HyperTensor
COPY --chown=hypertensor:hypertensor . /home/hypertensor/hypertensor
RUN pip install --user -e /home/hypertensor/hypertensor

# Add user bin to PATH
ENV PATH="/home/hypertensor/.local/bin:${PATH}"

# Verify CUDA
RUN python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

CMD ["python", "-c", "import tensornet; import torch; print(f'HyperTensor (CUDA: {torch.cuda.is_available()})')"]

# =============================================================================
# JUPYTER IMAGE
# =============================================================================
FROM base as jupyter

USER hypertensor

# Install Jupyter
RUN pip install --user \
    jupyterlab \
    ipywidgets \
    matplotlib \
    plotly

# Copy notebooks
COPY --chown=hypertensor:hypertensor notebooks/ /home/hypertensor/notebooks/

# Expose port
EXPOSE 8888

# Start Jupyter
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''"]

# =============================================================================
# SERVER IMAGE
# =============================================================================
FROM base as server

USER hypertensor

# Install server dependencies
RUN pip install --user \
    fastapi \
    uvicorn \
    pyzmq

# Copy server code
COPY --chown=hypertensor:hypertensor sdk/server/ /home/hypertensor/server/

# Expose ports
EXPOSE 8000  # REST API
EXPOSE 5555  # ZMQ Bridge

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start server
CMD ["python", "-m", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
