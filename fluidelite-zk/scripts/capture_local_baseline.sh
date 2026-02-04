#!/bin/bash
# =============================================================================
# FluidElite ZK Local Baseline Capture
#
# Run this script locally to create a baseline JSON that can be compared
# against vast.ai cloud GPU results.
#
# Usage:
#   ./scripts/capture_local_baseline.sh
#
# Output:
#   output/local_benchmark_baseline.json
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE_ROOT="$(dirname "$PROJECT_ROOT")"
TARGET_DIR="$WORKSPACE_ROOT/target/release"
OUTPUT_DIR="$PROJECT_ROOT/output"
BASELINE_FILE="$OUTPUT_DIR/local_benchmark_baseline.json"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Set up environment
export ICICLE_BACKEND_INSTALL_DIR="${ICICLE_BACKEND_INSTALL_DIR:-/opt/icicle/lib/backend}"

# Find ICICLE libraries
ICICLE_LIB_PATHS=""
for lib_dir in $(find "$WORKSPACE_ROOT/target/release/build" -name "libicicle*.so" -printf "%h\n" 2>/dev/null | sort -u); do
    if [ -d "$lib_dir" ]; then
        ICICLE_LIB_PATHS="$lib_dir:$ICICLE_LIB_PATHS"
    fi
done
export LD_LIBRARY_PATH="$ICICLE_LIB_PATHS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║          FluidElite ZK Local Baseline Capture                                ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check for GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. GPU required for baseline."
    exit 1
fi

# Collect system info
echo "📊 Collecting system information..."

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
GPU_CUDA_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | cut -d',' -f1 || echo "unknown")
CPU_MODEL=$(lscpu 2>/dev/null | grep "Model name" | cut -d':' -f2 | xargs || cat /proc/cpuinfo 2>/dev/null | grep "model name" | head -1 | cut -d':' -f2 | xargs || echo "unknown")
CPU_CORES=$(nproc 2>/dev/null || echo "1")
RAM_GB=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || echo "0")
HOSTNAME=$(hostname)
OS=$(uname -s)

echo "   GPU: $GPU_NAME ($GPU_VRAM MB VRAM)"
echo "   CPU: $CPU_MODEL ($CPU_CORES cores)"
echo ""

# Run quick benchmark and capture output
echo "🔬 Running baseline benchmark..."

BENCH_OUTPUT=""
if [ -f "$TARGET_DIR/gpu-sustained-bench" ]; then
    BENCH_OUTPUT=$("$TARGET_DIR/gpu-sustained-bench" 2>&1 || true)
    echo "$BENCH_OUTPUT" | tail -30
elif [ -f "$TARGET_DIR/gpu-test" ]; then
    BENCH_OUTPUT=$("$TARGET_DIR/gpu-test" 2>&1 || true)
    echo "$BENCH_OUTPUT" | tail -20
else
    echo "⚠️  Benchmark binaries not found. Building..."
    cd "$PROJECT_ROOT"
    cargo build --release --features gpu
    
    if [ -f "$TARGET_DIR/gpu-test" ]; then
        BENCH_OUTPUT=$("$TARGET_DIR/gpu-test" 2>&1 || true)
        echo "$BENCH_OUTPUT" | tail -20
    fi
fi

# Extract TPS from output (if present)
TPS_2_16=$(echo "$BENCH_OUTPUT" | grep -oP '2\^16.*?(\d+\.?\d*)\s*TPS' | grep -oP '\d+\.?\d*(?=\s*TPS)' | tail -1 || echo "0")
TPS_2_18=$(echo "$BENCH_OUTPUT" | grep -oP '2\^18.*?(\d+\.?\d*)\s*TPS' | grep -oP '\d+\.?\d*(?=\s*TPS)' | tail -1 || echo "0")
TPS_2_20=$(echo "$BENCH_OUTPUT" | grep -oP '2\^20.*?(\d+\.?\d*)\s*TPS' | grep -oP '\d+\.?\d*(?=\s*TPS)' | tail -1 || echo "0")

# Create baseline JSON
cat > "$BASELINE_FILE" << EOF
{
  "platform": "local",
  "hostname": "$HOSTNAME",
  "os": "$OS",
  "timestamp": "$(date -Iseconds)",
  "system": {
    "gpu_name": "$GPU_NAME",
    "gpu_vram_mb": $GPU_VRAM,
    "gpu_driver": "$GPU_DRIVER",
    "cuda_version": "$CUDA_VERSION",
    "compute_capability": "$GPU_CUDA_CAP",
    "cpu_model": "$CPU_MODEL",
    "cpu_cores": $CPU_CORES,
    "ram_gb": $RAM_GB
  },
  "baseline_metrics": {
    "tps_2_16": $TPS_2_16,
    "tps_2_18": $TPS_2_18,
    "tps_2_20": $TPS_2_20,
    "note": "Compare against vast.ai runs for consistency"
  },
  "environment": {
    "icicle_backend": "$ICICLE_BACKEND_INSTALL_DIR",
    "rust_version": "$(rustc --version 2>/dev/null | awk '{print $2}' || echo 'unknown')"
  }
}
EOF

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                      LOCAL BASELINE CAPTURED                                 ║"
echo "╠══════════════════════════════════════════════════════════════════════════════╣"
echo "║  GPU: $GPU_NAME"
echo "║  TPS @ 2^16: $TPS_2_16"
echo "║  TPS @ 2^18: $TPS_2_18"
echo "║  TPS @ 2^20: $TPS_2_20"
echo "╠══════════════════════════════════════════════════════════════════════════════╣"
echo "║  Baseline saved to:"
echo "║    $BASELINE_FILE"
echo "║"
echo "║  Use this to compare against vast.ai results:"
echo "║    jq -s '.[0].system.gpu_name, .[1].system.gpu_name' \\"
echo "║       $BASELINE_FILE \\"
echo "║       ~/benchmark_results/vastai_*.json"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
