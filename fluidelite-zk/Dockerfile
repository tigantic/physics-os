# FluidElite ZK Prover - Production Dockerfile
#
# Build:
#   docker build -t fluidelite-zk:latest .
#
# Run:
#   docker run -p 8080:8080 fluidelite-zk:latest

# ============================================
# Stage 1: Build
# ============================================
FROM rust:1.75-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Cargo files first for dependency caching
COPY Cargo.toml Cargo.lock* ./
COPY src ./src
COPY benches ./benches
COPY tests ./tests

# Build release binary with production features
RUN cargo build --release --features production

# ============================================
# Stage 2: Runtime
# ============================================
FROM debian:bookworm-slim AS runtime

# Install runtime dependencies
# curl is needed for container health checks and debugging
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 fluidelite

# Copy binaries from builder
COPY --from=builder /app/target/release/prover-node /usr/local/bin/
COPY --from=builder /app/target/release/fluidelite-cli /usr/local/bin/

# Create data directories
RUN mkdir -p /data/weights /data/proofs /data/logs && \
    chown -R fluidelite:fluidelite /data

# Switch to non-root user
USER fluidelite
WORKDIR /data

# Environment variables
ENV RUST_LOG=info
ENV FLUIDELITE_DATA_DIR=/data

# Expose API port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
ENTRYPOINT ["prover-node"]
CMD ["--network", "mock", "--weights", "/data/weights/model.bin"]
