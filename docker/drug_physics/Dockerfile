# HyperTensor Drug Physics Engine
# KRAS G12D Multi-Mechanism Binding Simulation
#
# Build: docker build -t hypertensor/drug-physics .
# Run:   docker run --gpus all hypertensor/drug-physics

FROM nvidia/cuda:12.1-runtime-ubuntu22.04

LABEL maintainer="HyperTensor Team"
LABEL description="Drug Physics Engine for KRAS G12D inhibitor design"
LABEL version="1.0.0"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Create virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    torch>=2.0.0 \
    requests>=2.28.0

# Copy drug physics engine files
COPY tig011a_multimechanism.py /app/
COPY tig011a_dielectric_gauntlet.py /app/
COPY tig011a_docking_qmmm.py /app/
COPY tig011a_attestation.py /app/

# Copy physics working directory if exists
COPY physics_working/ /app/physics_working/

# Create output directory
RUN mkdir -p /app/output

# Default: run the multi-mechanism simulation
ENTRYPOINT ["python3"]
CMD ["tig011a_multimechanism.py"]
