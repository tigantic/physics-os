#!/bin/bash
# =============================================================================
# FluidElite ZK Production Server Startup Script
# =============================================================================

set -e

# Default configuration
HOST="${FLUIDELITE_HOST:-0.0.0.0}"
PORT="${FLUIDELITE_PORT:-8080}"
METRICS_PORT="${FLUIDELITE_METRICS_PORT:-9090}"
TIMEOUT="${FLUIDELITE_TIMEOUT:-120}"
RATE_LIMIT="${FLUIDELITE_RATE_LIMIT:-60}"

# Icicle backend path (for GPU acceleration)
export ICICLE_BACKEND_INSTALL_DIR="${ICICLE_BACKEND_INSTALL_DIR:-/opt/icicle/lib/backend}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         FluidElite ZK - Production Launcher              ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"

# Check for API key
if [ -z "$FLUIDELITE_API_KEY" ]; then
    echo -e "${YELLOW}WARNING: FLUIDELITE_API_KEY not set - authentication disabled${NC}"
    echo -e "${YELLOW}         Set environment variable for production use${NC}"
fi

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ GPU detected:${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
else
    echo -e "${YELLOW}⚠ No GPU detected - running in CPU mode${NC}"
fi

# Check Icicle backend
if [ -d "$ICICLE_BACKEND_INSTALL_DIR" ]; then
    echo -e "${GREEN}✓ Icicle backend: $ICICLE_BACKEND_INSTALL_DIR${NC}"
else
    echo -e "${YELLOW}⚠ Icicle backend not found at $ICICLE_BACKEND_INSTALL_DIR${NC}"
fi

echo ""
echo "Configuration:"
echo "  Host:         $HOST"
echo "  Port:         $PORT"
echo "  Metrics Port: $METRICS_PORT"
echo "  Timeout:      ${TIMEOUT}s"
echo "  Rate Limit:   ${RATE_LIMIT}/min"
echo ""

# Build arguments
ARGS="--host $HOST --port $PORT --metrics-port $METRICS_PORT --timeout $TIMEOUT --rate-limit $RATE_LIMIT"

# Add API key if set
if [ -n "$FLUIDELITE_API_KEY" ]; then
    ARGS="$ARGS --api-key $FLUIDELITE_API_KEY"
fi

# JSON logs for production
if [ "${FLUIDELITE_JSON_LOGS:-false}" = "true" ]; then
    ARGS="$ARGS --json-logs"
fi

# Test mode
if [ "${FLUIDELITE_TEST_MODE:-false}" = "true" ]; then
    ARGS="$ARGS --test"
    echo -e "${YELLOW}Running in TEST mode${NC}"
fi

# Custom circuit k
if [ -n "$FLUIDELITE_CIRCUIT_K" ]; then
    ARGS="$ARGS --circuit-k $FLUIDELITE_CIRCUIT_K"
fi

echo -e "${GREEN}Starting FluidElite ZK Server...${NC}"
echo ""

# Find the binary
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY=""

# Check common locations
for path in \
    "$SCRIPT_DIR/target/release/fluidelite-server" \
    "$SCRIPT_DIR/../target/release/fluidelite-server" \
    "/usr/local/bin/fluidelite-server" \
    "$(command -v fluidelite-server 2>/dev/null)"; do
    if [ -x "$path" ]; then
        BINARY="$path"
        break
    fi
done

if [ -z "$BINARY" ]; then
    echo -e "${RED}ERROR: fluidelite-server binary not found${NC}"
    echo "Build it with: cargo build --release --features production-gpu"
    exit 1
fi

echo "Using binary: $BINARY"
echo ""

# Run the server
exec $BINARY $ARGS
