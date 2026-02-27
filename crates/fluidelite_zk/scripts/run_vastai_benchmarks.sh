#!/bin/bash
# =============================================================================
# FluidElite ZK Full Benchmark Suite for Vast.ai Cloud GPU
#
# This script runs the complete benchmark suite and saves results in JSON
# format for comparison with local runs.
#
# Usage:
#   ./run_vastai_benchmarks.sh [OPTIONS]
#
# Options:
#   --quick      Run quick sanity benchmarks only (2-3 minutes)
#   --standard   Run standard benchmark suite (10-15 minutes)
#   --full       Run exhaustive benchmarks (30+ minutes)
#   --help       Show this help message
#
# Results are saved to: ~/benchmark_results/vastai_TIMESTAMP.json
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE_ROOT="$(dirname "$PROJECT_ROOT")"
TARGET_DIR="$WORKSPACE_ROOT/target/release"
RESULTS_DIR="${HOME}/benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$RESULTS_DIR/vastai_${TIMESTAMP}.json"

# Set up environment
export ICICLE_BACKEND_INSTALL_DIR="${ICICLE_BACKEND_INSTALL_DIR:-/opt/icicle/lib/backend}"

# Find and set LD_LIBRARY_PATH for ICICLE libraries
ICICLE_LIB_PATHS=""
for lib_dir in $(find "$WORKSPACE_ROOT/target/release/build" -name "libicicle*.so" -printf "%h\n" 2>/dev/null | sort -u); do
    if [ -d "$lib_dir" ]; then
        ICICLE_LIB_PATHS="$lib_dir:$ICICLE_LIB_PATHS"
    fi
done
export LD_LIBRARY_PATH="$ICICLE_LIB_PATHS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Benchmark mode
MODE="${1:-standard}"

# Parse arguments
case "$1" in
    --quick|-q)
        MODE="quick"
        shift
        ;;
    --standard|-s)
        MODE="standard"
        shift
        ;;
    --full|-f)
        MODE="full"
        shift
        ;;
    --help|-h)
        echo "FluidElite ZK Benchmark Suite for Vast.ai"
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --quick, -q      Quick sanity check (2-3 minutes)"
        echo "  --standard, -s   Standard benchmarks (10-15 minutes)"
        echo "  --full, -f       Full exhaustive suite (30+ minutes)"
        echo "  --help, -h       Show this help"
        echo ""
        echo "Results saved to: ~/benchmark_results/"
        exit 0
        ;;
esac

# Create results directory
mkdir -p "$RESULTS_DIR"

# Header
echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║          FluidElite ZK Benchmark Suite - Vast.ai Cloud GPU                   ║"
echo "╠══════════════════════════════════════════════════════════════════════════════╣"
echo "║  Mode: $MODE"
echo "║  Results: $RESULTS_FILE"
echo "║  Started: $(date)"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Collect system info
echo "📊 Collecting system information..."

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
GPU_CUDA_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | cut -d',' -f1 || echo "unknown")
CPU_MODEL=$(lscpu 2>/dev/null | grep "Model name" | cut -d':' -f2 | xargs || echo "unknown")
CPU_CORES=$(nproc)
RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
HOSTNAME=$(hostname)
PLATFORM="vast.ai"

echo "   GPU: $GPU_NAME ($GPU_VRAM MB VRAM)"
echo "   Driver: $GPU_DRIVER | CUDA: $CUDA_VERSION"
echo "   CPU: $CPU_MODEL ($CPU_CORES cores)"
echo "   RAM: ${RAM_GB} GB"
echo ""

# Initialize results JSON
cat > "$RESULTS_FILE" << EOF
{
  "platform": "$PLATFORM",
  "hostname": "$HOSTNAME",
  "timestamp": "$(date -Iseconds)",
  "mode": "$MODE",
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
  "benchmarks": {
EOF

# Function to run a benchmark and capture output
run_benchmark() {
    local name="$1"
    local binary="$2"
    local description="$3"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔬 Running: $description"
    echo "   Binary: $binary"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    local output_file="$RESULTS_DIR/${name}_${TIMESTAMP}.txt"
    local start_time=$(date +%s.%N)
    
    if [ -f "$TARGET_DIR/$binary" ]; then
        # Run benchmark and capture output
        "$TARGET_DIR/$binary" 2>&1 | tee "$output_file"
        local exit_code=$?
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc)
        
        echo ""
        echo "   ✅ Completed in ${duration}s (exit code: $exit_code)"
        echo "   📄 Output saved to: $output_file"
        echo ""
        
        return $exit_code
    else
        echo "   ⚠️  Binary not found: $TARGET_DIR/$binary"
        echo "   Skipping this benchmark..."
        echo ""
        return 1
    fi
}

# Track benchmark success
BENCHMARKS_RUN=0
BENCHMARKS_PASSED=0

# ==============================================================================
# QUICK MODE: Basic GPU functionality test
# ==============================================================================
if [ "$MODE" = "quick" ] || [ "$MODE" = "standard" ] || [ "$MODE" = "full" ]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "  PHASE 1: GPU Functionality Test"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    
    if run_benchmark "gpu_test" "gpu-test" "GPU ICICLE Backend Test"; then
        BENCHMARKS_PASSED=$((BENCHMARKS_PASSED + 1))
    fi
    BENCHMARKS_RUN=$((BENCHMARKS_RUN + 1))
fi

# ==============================================================================
# STANDARD MODE: Core performance benchmarks
# ==============================================================================
if [ "$MODE" = "standard" ] || [ "$MODE" = "full" ]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "  PHASE 2: Sustained TPS Benchmark"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    
    if run_benchmark "gpu_sustained" "gpu-sustained-bench" "GPU Sustained TPS (Back-to-back MSM)"; then
        BENCHMARKS_PASSED=$((BENCHMARKS_PASSED + 1))
    fi
    BENCHMARKS_RUN=$((BENCHMARKS_RUN + 1))
    
    echo ""
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "  PHASE 3: Real-World TPS Benchmark"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    
    if run_benchmark "gpu_realworld" "gpu-realworld-tps" "GPU Real-World TPS (Production Simulation)"; then
        BENCHMARKS_PASSED=$((BENCHMARKS_PASSED + 1))
    fi
    BENCHMARKS_RUN=$((BENCHMARKS_RUN + 1))
fi

# ==============================================================================
# FULL MODE: Exhaustive benchmarks
# ==============================================================================
if [ "$MODE" = "full" ]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "  PHASE 4: Halo2 GPU Benchmark"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    
    if run_benchmark "gpu_halo2" "gpu-halo2-benchmark" "GPU Halo2 Prover Benchmark"; then
        BENCHMARKS_PASSED=$((BENCHMARKS_PASSED + 1))
    fi
    BENCHMARKS_RUN=$((BENCHMARKS_RUN + 1))
    
    echo ""
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "  PHASE 5: Pipelined TPS Benchmark"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    
    if run_benchmark "gpu_pipelined" "gpu-pipelined-tps" "GPU Pipelined TPS (CUDA Streams)"; then
        BENCHMARKS_PASSED=$((BENCHMARKS_PASSED + 1))
    fi
    BENCHMARKS_RUN=$((BENCHMARKS_RUN + 1))
    
    echo ""
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "  PHASE 6: K-Ladder Stress Test"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    
    if run_benchmark "k_ladder" "k-ladder-stress" "K-Ladder Stress Test (MSM Scaling)"; then
        BENCHMARKS_PASSED=$((BENCHMARKS_PASSED + 1))
    fi
    BENCHMARKS_RUN=$((BENCHMARKS_RUN + 1))
fi

# ==============================================================================
# Finalize results
# ==============================================================================

# Close the JSON
cat >> "$RESULTS_FILE" << EOF
  },
  "summary": {
    "benchmarks_run": $BENCHMARKS_RUN,
    "benchmarks_passed": $BENCHMARKS_PASSED,
    "completed_at": "$(date -Iseconds)"
  }
}
EOF

# Print summary
echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                         BENCHMARK SUITE COMPLETE                             ║"
echo "╠══════════════════════════════════════════════════════════════════════════════╣"
echo "║  Platform:        $PLATFORM"
echo "║  GPU:             $GPU_NAME"
echo "║  Mode:            $MODE"
echo "║  Benchmarks:      $BENCHMARKS_PASSED / $BENCHMARKS_RUN passed"
echo "║  Completed:       $(date)"
echo "╠══════════════════════════════════════════════════════════════════════════════╣"
echo "║  Results saved to:"
echo "║    $RESULTS_FILE"
echo "║"
echo "║  Individual outputs:"
ls -la "$RESULTS_DIR"/*_${TIMESTAMP}.txt 2>/dev/null | while read line; do
    echo "║    $(echo $line | awk '{print $NF}')"
done
echo "║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"

# Compare with local baseline (if exists)
LOCAL_BASELINE="$PROJECT_ROOT/output/local_benchmark_baseline.json"
if [ -f "$LOCAL_BASELINE" ]; then
    echo ""
    echo "📊 Comparison with local baseline available:"
    echo "   Local baseline: $LOCAL_BASELINE"
    echo "   Cloud results:  $RESULTS_FILE"
    echo ""
    echo "   Use 'jq' to compare: jq -s '.[0].system.gpu_name, .[1].system.gpu_name' $LOCAL_BASELINE $RESULTS_FILE"
fi

echo ""
echo "🎉 Done! To download results:"
echo "   scp user@host:$RESULTS_FILE ."
echo ""
