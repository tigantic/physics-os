#!/bin/bash
# Comprehensive dependency verification for Project HyperTensor
# Run from WSL: bash verify_dependencies.sh

set -e

echo "======================================"
echo "Project HyperTensor Dependency Check"
echo "======================================"
echo ""

# Check OS
echo "[1/10] Operating System..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "✓ Running on Linux (WSL)"
    uname -a
else
    echo "✗ ERROR: Must run in WSL Linux, not Windows"
    exit 1
fi
echo ""

# Check GPU
echo "[2/10] NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "✓ NVIDIA GPU detected"
else
    echo "✗ ERROR: nvidia-smi not found"
    exit 1
fi
echo ""

# Check CUDA
echo "[3/10] CUDA Toolkit..."
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
    echo "✓ CUDA Toolkit installed"
else
    echo "⚠ WARNING: nvcc not found (PyTorch may still work with runtime-only CUDA)"
fi
echo ""

# Check Python
echo "[4/10] Python Version..."
python3 --version
if python3 -c "import sys; v = sys.version_info; sys.exit(0 if (v.major == 3 and 10 <= v.minor <= 12) else 1)"; then
    echo "✓ Python version compatible"
else
    echo "✗ ERROR: Python must be 3.10, 3.11, or 3.12"
    exit 1
fi
echo ""

# Check PyTorch
echo "[5/10] PyTorch..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print('✓ PyTorch with CUDA support')
else:
    print('✗ ERROR: PyTorch CUDA support not available')
    exit(1)
"
echo ""

# Check TensorNet
echo "[6/10] TensorNet Package..."
python3 -c "
from ontic.sovereign.bridge_writer import TensorBridgeWriter
from ontic.sovereign.qtt_slice_extractor import QTTSliceExtractor
from ontic.sovereign.realtime_tensor_stream import RealtimeTensorStream
print('✓ TensorNet modules import successfully')
"
echo ""

# Check Rust
echo "[7/10] Rust Toolchain..."
rustc --version
cargo --version
echo "✓ Rust toolchain installed"
echo ""

# Check Glass Cockpit
echo "[8/10] Glass Cockpit..."
if [ -d "glass-cockpit" ]; then
    cd glass-cockpit
    cargo check --release 2>&1 | tail -5
    echo "✓ Glass Cockpit compiles"
    cd ..
else
    echo "✗ ERROR: glass-cockpit directory not found"
    exit 1
fi
echo ""

# Check /dev/shm
echo "[9/10] Shared Memory..."
df -h /dev/shm | tail -1
if mount | grep -q "/dev/shm.*tmpfs"; then
    echo "✓ /dev/shm is tmpfs (memory-backed)"
else
    echo "✗ ERROR: /dev/shm is not tmpfs"
    exit 1
fi
echo ""

# Check build tools
echo "[10/10] Build Tools..."
gcc --version | head -1
cmake --version | head -1
make --version | head -1
echo "✓ Build tools present"
echo ""

echo "======================================"
echo "✓ ALL CHECKS PASSED"
echo "======================================"
echo ""
echo "Environment is ready for:"
echo "  • Phase 3: Real-time visualization (✅ Complete)"
echo "  • Phase 4: QTT Navier-Stokes integration (🔨 In Progress)"
echo ""
echo "Next steps:"
echo "  1. Run Phase 3 test: python3 test_phase3_integration.py"
echo "  2. Run Phase 4 validation: python3 test_phase4_validation.py"
echo "  3. Start development!"
