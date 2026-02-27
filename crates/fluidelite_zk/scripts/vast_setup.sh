#!/bin/bash
# =============================================================================
# FluidElite ZK Vast.ai Instance Setup Script
# 
# Run this script on a fresh vast.ai GPU instance to set up the environment
# for running FluidElite ZK benchmarks.
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/TiganticLabz/HyperTensor-VM/main/fluidelite-zk/scripts/vast_setup.sh | bash
#   
# Or download and run:
#   wget https://raw.githubusercontent.com/TiganticLabz/HyperTensor-VM/main/fluidelite-zk/scripts/vast_setup.sh
#   chmod +x vast_setup.sh
#   ./vast_setup.sh
# =============================================================================

set -e

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║           FluidElite ZK - Vast.ai GPU Instance Setup                     ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if running on a GPU instance
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: nvidia-smi not found. This script requires an NVIDIA GPU instance."
    echo "   Please select a GPU instance on vast.ai (RTX 4090, A100, H100, etc.)"
    exit 1
fi

# Display GPU info
echo "🖥️  GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
echo ""

# Check CUDA version
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | cut -d',' -f1 || echo "not found")
echo "📦 CUDA Version: $CUDA_VERSION"
echo ""

# Update package list
echo "📥 Updating package list..."
sudo apt-get update -qq

# Install required packages
echo "📦 Installing dependencies..."
sudo apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    cmake \
    git \
    ca-certificates \
    wget \
    jq \
    bc \
    htop \
    nvtop 2>/dev/null || true

# Install Rust if not present
if ! command -v rustc &> /dev/null; then
    echo "🦀 Installing Rust (nightly)..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly-2024-01-01
    source "$HOME/.cargo/env"
else
    echo "🦀 Rust already installed: $(rustc --version)"
fi

# Ensure cargo is in PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Create ICICLE directory
ICICLE_DIR="/opt/icicle/lib/backend"
echo "📁 Setting up ICICLE backend directory..."
sudo mkdir -p "$ICICLE_DIR"
sudo chown -R $(whoami):$(id -gn) /opt/icicle

# Download and install ICICLE CUDA backend
if [ ! -f "$ICICLE_DIR/backend.toml" ]; then
    echo "📥 Downloading ICICLE v4.0.0 CUDA backend..."
    cd /tmp
    wget -q --show-progress https://github.com/ingonyama-zk/icicle/releases/download/v4.0.0/icicle_ubuntu22_cuda122_with_bn254_with_bls12-377_with_bls12-381_with_bw6-761_with_grumpkin.tar.gz
    
    echo "📦 Extracting ICICLE backend..."
    tar -xzf icicle*.tar.gz -C "$ICICLE_DIR"
    rm -f icicle*.tar.gz
    
    echo "✅ ICICLE backend installed to $ICICLE_DIR"
else
    echo "✅ ICICLE backend already installed"
fi

# Set environment variables
export ICICLE_BACKEND_INSTALL_DIR="$ICICLE_DIR"

# Add to .bashrc for persistence
if ! grep -q "ICICLE_BACKEND_INSTALL_DIR" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# FluidElite ZK GPU Environment" >> ~/.bashrc
    echo "export ICICLE_BACKEND_INSTALL_DIR=$ICICLE_DIR" >> ~/.bashrc
    echo "export PATH=\"\$HOME/.cargo/bin:\$PATH\"" >> ~/.bashrc
fi

# Clone the repository if not present
REPO_DIR="$HOME/HyperTensor-VM"
if [ ! -d "$REPO_DIR" ]; then
    echo "📥 Cloning HyperTensor-VM repository..."
    git clone --depth 1 https://github.com/TiganticLabz/HyperTensor-VM.git "$REPO_DIR"
else
    echo "📁 Repository already exists at $REPO_DIR"
    echo "   Pulling latest changes..."
    cd "$REPO_DIR" && git pull --rebase || true
fi

cd "$REPO_DIR/fluidelite-zk"

# Build benchmarks with GPU support
echo ""
echo "🔨 Building FluidElite ZK benchmarks with GPU support..."
echo "   This may take 5-10 minutes on first build..."
echo ""

cargo build --release --features gpu

# Verify build
echo ""
echo "✅ Build complete! Verifying binaries..."
for bin in gpu-test gpu-sustained-bench gpu-realworld-tps gpu-halo2-benchmark gpu-pipelined-tps k-ladder-stress; do
    if [ -f "target/release/$bin" ]; then
        echo "   ✓ $bin"
    else
        echo "   ✗ $bin (not built - may require additional features)"
    fi
done

# Create results directory
mkdir -p "$HOME/benchmark_results"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                        Setup Complete!                                   ║"
echo "╠══════════════════════════════════════════════════════════════════════════╣"
echo "║                                                                          ║"
echo "║  To run benchmarks:                                                      ║"
echo "║    cd ~/HyperTensor-VM/fluidelite-zk                                     ║"
echo "║    ./scripts/run_vastai_benchmarks.sh                                    ║"
echo "║                                                                          ║"
echo "║  Individual benchmarks:                                                  ║"
echo "║    ./scripts/run-gpu.sh gpu-test           # Quick GPU test             ║"
echo "║    ./scripts/run-gpu.sh gpu-sustained-bench # Sustained TPS test        ║"
echo "║    ./scripts/run-gpu.sh gpu-realworld-tps   # Real-world TPS test       ║"
echo "║                                                                          ║"
echo "║  Results saved to: ~/benchmark_results/                                  ║"
echo "║                                                                          ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

# Quick GPU test to verify setup
echo "🧪 Running quick GPU verification test..."
./scripts/run-gpu.sh gpu-test 2>&1 | head -50 || echo "⚠️  GPU test had issues - check CUDA installation"
