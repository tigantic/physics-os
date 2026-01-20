#!/bin/bash
# FluidElite GPU-Accelerated Prover Launch Script
# 
# This script sets up the environment for GPU-accelerated ZK proof generation
# using ICICLE CUDA backend.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE_ROOT="$(dirname "$PROJECT_ROOT")"
TARGET_DIR="$WORKSPACE_ROOT/target/release"

# Find ICICLE libraries in workspace target - need all three: bn254, runtime, hash
ICICLE_LIB_PATHS=""
for lib_dir in $(find "$WORKSPACE_ROOT/target/release/build" -name "libicicle*.so" -printf "%h\n" 2>/dev/null | sort -u); do
    if [ -d "$lib_dir" ]; then
        ICICLE_LIB_PATHS="$lib_dir:$ICICLE_LIB_PATHS"
    fi
done

if [ -z "$ICICLE_LIB_PATHS" ]; then
    echo "❌ ICICLE libraries not found. Build with GPU feature first:"
    echo "   cargo build --release --features gpu"
    exit 1
fi

# Set up environment
export LD_LIBRARY_PATH="$ICICLE_LIB_PATHS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ICICLE_BACKEND_INSTALL_DIR="${ICICLE_BACKEND_INSTALL_DIR:-/opt/icicle/lib/backend}"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║          FluidElite GPU-Accelerated Prover               ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║ ICICLE_BACKEND: $ICICLE_BACKEND_INSTALL_DIR"
echo "║ LD_LIBRARY_PATH: $ICICLE_LIB_DIR"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check if CUDA backend exists
if [ ! -d "$ICICLE_BACKEND_INSTALL_DIR" ]; then
    echo "⚠️  ICICLE CUDA backend not found at $ICICLE_BACKEND_INSTALL_DIR"
    echo "   Download from: https://github.com/ingonyama-zk/icicle/releases"
    echo "   Install to: /opt/icicle/lib/backend"
    echo ""
    echo "Proceeding with CPU fallback..."
fi

# Parse command
CMD="${1:-help}"
shift || true

case "$CMD" in
    server)
        echo "🚀 Starting GPU-accelerated prover server..."
        exec "$TARGET_DIR/fluidelite-server" "$@"
        ;;
    cli)
        exec "$TARGET_DIR/fluidelite-cli" "$@"
        ;;
    gpu-test)
        exec "$TARGET_DIR/gpu-test"
        ;;
    bench)
        echo "📊 Running proof benchmarks..."
        cd "$PROJECT_ROOT"
        cargo bench --features halo2 --bench proof_bench
        ;;
    help|*)
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  server [--test] [--port PORT]  Start prover API server"
        echo "  cli <subcommand>               Run CLI tool"
        echo "  gpu-test                       Test GPU acceleration"
        echo "  bench                          Run benchmarks"
        echo ""
        echo "Examples:"
        echo "  $0 server --test              Start server with test config"
        echo "  $0 server --port 9000         Start server on port 9000"
        echo "  $0 cli prove --token 42       Generate a proof"
        echo "  $0 gpu-test                   Verify GPU is working"
        ;;
esac
