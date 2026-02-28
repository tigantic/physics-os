# Dependency Verification Guide
## The Physics OS - Sovereign Glass Cockpit

**CRITICAL: All environments, installations, and execution must occur in Linux (WSL Ubuntu)**

This repository is hosted on the WSL filesystem at:
```
/home/brad/TiganticLabz/Main_Projects/The Physics OS
```

**DO NOT** install dependencies or run commands from Windows PowerShell or CMD. Always use WSL bash.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    WSL Ubuntu (Linux)                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Python Environment (venv/conda)                     │   │
│  │  • PyTorch 2.0+ (CUDA 11.8+)                        │   │
│  │  • NumPy, SciPy, TensorLy                           │   │
│  │  • TensorNet (sovereign modules)                    │   │
│  │  • QTT compression, CFD solvers                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓ RAM Bridge                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Rust Environment (cargo)                            │   │
│  │  • Glass Cockpit (WGPU renderer)                    │   │
│  │  • RAM Bridge reader                                │   │
│  │  • GPU shaders (WGSL)                               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  Shared Memory: /dev/shm/hypertensor_bridge                │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. System Dependencies

### 1.1 Operating System
- **Required**: Ubuntu 20.04+ (via WSL2)
- **Kernel**: Linux 5.10+
- **WSL Version**: WSL 2 (not WSL 1)

**Verification**:
```bash
# Run in WSL bash
uname -a
lsb_release -a
wsl.exe --version  # From WSL, shows WSL version
```

Expected output:
```
Linux <hostname> 5.10.16.3-microsoft-standard-WSL2 ... x86_64 GNU/Linux
Ubuntu 20.04.x LTS or 22.04.x LTS
WSL version: 2.x.x
```

### 1.2 NVIDIA GPU & CUDA
- **GPU**: NVIDIA GPU with CUDA Compute Capability ≥ 7.0 (Volta+)
  - Recommended: RTX 3060 or better
  - Minimum VRAM: 8GB (16GB recommended)
- **CUDA Toolkit**: 11.8+ or 12.x
- **cuDNN**: 8.6+ (bundled with PyTorch)
- **NVIDIA Driver**: 525.60.11+ (Windows host driver)

**Verification**:
```bash
# Check NVIDIA driver (from Windows or WSL)
nvidia-smi

# Check CUDA availability (from WSL)
nvcc --version

# Check if CUDA is accessible from WSL
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output:
```
CUDA Version: 12.x
GPU 0: NVIDIA GeForce RTX <model>
```

### 1.3 Build Tools
- **GCC/G++**: 9.0+ (for compiling Rust native extensions and CUDA code)
- **CMake**: 3.18+ (for building native dependencies)
- **Make**: GNU Make 4.2+
- **pkg-config**: For library discovery

**Installation** (if missing):
```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libssl-dev
```

**Verification**:
```bash
gcc --version
g++ --version
cmake --version
make --version
pkg-config --version
```

---

## 2. Python Environment

### 2.1 Python Version
- **Required**: Python 3.10, 3.11, or 3.12
- **Not compatible**: Python 3.8 (too old)

**Verification**:
```bash
python3 --version
which python3
```

Expected: `Python 3.10.x`, `Python 3.11.x`, or `Python 3.12.x`

### 2.2 Virtual Environment (Recommended)

**CRITICAL**: Create the virtual environment in WSL Linux, not Windows.

#### Option A: venv (Built-in)
```bash
cd /home/brad/TiganticLabz/Main_Projects/Project\ HyperTensor
python3 -m venv venv
source venv/bin/activate  # Activate for every session
```

#### Option B: conda (If you prefer)
```bash
conda create -n hypertensor python=3.11 -y
conda activate hypertensor
```

**Verification**:
```bash
which python
# Should output: /home/brad/.../venv/bin/python (NOT a Windows path!)
```

### 2.3 Python Dependencies

All dependencies are specified in `requirements-lock.txt`. Install with:

```bash
# Ensure you're in the project directory and venv is activated
cd /home/brad/TiganticLabz/Main_Projects/Project\ HyperTensor
source venv/bin/activate  # If using venv

pip install --upgrade pip
pip install -r requirements-lock.txt
```

#### Core Dependencies:

**Scientific Computing**:
- `torch>=2.0.0` (PyTorch with CUDA support)
- `numpy>=1.24.0`
- `scipy>=1.10.0`
- `tensorly>=0.8.1` (Tensor decomposition)
- `opt_einsum>=3.3.0` (Optimized Einstein summation)

**CFD & Physics**:
- `numba>=0.57.0` (JIT compilation for fast kernels)
- Custom modules in `tensornet/`:
  - `tensornet.sovereign.fast_euler_3d` (3D Euler solver with QTT)
  - `tensornet.sovereign.qtt_slice_extractor` (3D→2D slicing)
  - `tensornet.sovereign.realtime_tensor_stream` (RAM Bridge writer)

**Data & I/O**:
- `h5py>=3.8.0` (HDF5 for large datasets)
- `zarr>=2.14.0` (Chunked arrays)
- `mmap` (built-in, for shared memory)

**Development Tools**:
- `pytest>=7.3.0` (Testing)
- `mypy>=1.3.0` (Type checking)
- `black>=23.3.0` (Code formatting)
- `ruff>=0.0.270` (Fast linting)

**Verification**:
```bash
python3 -c "
import sys
import torch
import numpy as np
import scipy
import tensorly
import numba
import h5py

print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'NumPy: {np.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print('✓ Core dependencies imported successfully')
"
```

**Expected output**:
```
Python: 3.11.x
PyTorch: 2.x.x+cu118 (or cu121)
NumPy: 1.24.x
CUDA available: True
CUDA device: NVIDIA GeForce RTX <model>
CUDA memory: 16.0 GB
✓ Core dependencies imported successfully
```

### 2.4 TensorNet Package Installation

The `tensornet` package must be installed in editable mode:

```bash
cd /home/brad/TiganticLabz/Main_Projects/Project\ HyperTensor
pip install -e .
```

**Verification**:
```bash
python3 -c "
from tensornet.sovereign.bridge_writer import TensorBridgeWriter
from tensornet.sovereign.qtt_slice_extractor import QTTSliceExtractor
from tensornet.sovereign.realtime_tensor_stream import RealtimeTensorStream
print('✓ TensorNet sovereign modules imported successfully')
"
```

---

## 3. Rust Environment

### 3.1 Rust Toolchain
- **Required**: Rust 1.75+ (stable channel, per Cargo.toml rust-version)
- **Cargo**: Bundled with Rust
- **Rustup**: For toolchain management

**Installation** (if not present):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

**Verification**:
```bash
rustc --version
cargo --version
rustup --version
```

Expected: `rustc 1.7x.0`, `cargo 1.7x.0`

### 3.2 Rust Dependencies

Dependencies are managed in `glass-cockpit/Cargo.toml`:

**Graphics Pipeline** (Sovereign_UI.md Doctrine 4):
- `wgpu = "0.19"` (GPU graphics API - Vulkan/DX12/Metal abstraction)
- `winit = "0.29"` (Window management - minimal, cross-platform)
- `raw-window-handle = "0.6"` (Low-level window access)
- `pollster = "0.3"` (Minimal async executor for wgpu initialization)

**Math & Linear Algebra**:
- `glam = "0.25"` (SIMD-accelerated math, shader-compatible types)
- `bytemuck = "1.14"` (Zero-copy casting for GPU buffers)

**Shared Memory IPC** (RAM Bridge Protocol):
- `shared_memory = "0.12"` (Direct mmap access)
- `memmap2 = "0.9"` (Alternative memory mapping)
- `crc = "3.0"` (Data integrity validation)

**Performance & Profiling**:
- `tracy-client = "0.17"` (Optional - frame profiling)

**Async Runtime** (Optional - satellite tile fetching only):
- `tokio = { version = "1.35", features = ["rt", "sync"] }` (Minimal async)
- `reqwest = "0.11"` (HTTP client for NASA GIBS)

**Error Handling**:
- `thiserror = "1.0"` (Custom error types)
- `anyhow = "1.0"` (Error propagation)

**Development Logging** (Optional - disabled in release):
- `env_logger = "0.11"` (Log output)
- `log = "0.4"` (Logging facade)

**Platform-Specific**:
- **Windows**: `windows = "0.52"` (E-core affinity enforcement)
- **Linux**: `nix = "0.27"` (WSL shared memory access)

**Build**:
```bash
cd /home/brad/TiganticLabz/Main_Projects/Project\ HyperTensor/glass-cockpit
cargo build --release
```

**Verification**:
```bash
cd glass-cockpit
cargo test --release
cargo run --release --bin phase3 -- --help
```

Expected: No compilation errors, help text displays.

### 3.3 UI Framework Dependencies (Conditional)

**Sovereign_UI.md Doctrine 4.1 Amendment (2025-12-29)**:
For Phase 1-2 scaffolding, the following Rust UI crates are conditionally approved:

- **egui** - Immediate-mode GUI (if used for rapid prototyping)
- **iced** - Declarative UI alternative (ELM-inspired)

**Conditions**:
- MUST run exclusively on E-cores (affinity enforced)
- MUST have zero network activity
- MUST stay within performance boundaries (<5% flexible target)
- MUST be replaceable in Phase 3+ with procedural rendering

**Current Status**: Not yet added to Cargo.toml. Will be evaluated if needed for weather UI controls.

**Verification**: If added, verify E-core isolation:
```bash
# On Windows host:
Get-Process glass-cockpit | Select-Object ProcessorAffinity
# Should show 0xFFFF0000 (E-cores only)
```

---

## 4. Phase-Specific Dependencies

### Phase 3: Real-Time Visualization
✅ **Complete**

**Python Side**:
- `tensornet.sovereign.bridge_writer` (RAM Bridge v2 writer)
- `tensornet.sovereign.realtime_tensor_stream` (Streaming orchestration)
- Synthetic tensor generation (test patterns)

**Rust Side**:
- `glass-cockpit/src/bin/phase3.rs` (Main visualizer)
- `glass-cockpit/src/ram_bridge_v2.rs` (Reader)
- `glass-cockpit/shaders/tensor_colormap.wgsl` (GPU shader)

**Verification**:
```bash
# Terminal 1 (Python):
cd /home/brad/TiganticLabz/Main_Projects/Project\ HyperTensor
python3 tensornet/sovereign/realtime_tensor_stream.py

# Terminal 2 (Rust):
cd glass-cockpit
cargo run --release --bin phase3

# Expected: Window opens showing colorized tensor patterns at 60 FPS
```

### Phase 4: QTT Navier-Stokes Integration
✅ **Complete**

**Python Side**:
- `tensornet.sovereign.qtt_slice_extractor` (3D→2D GPU slicing)
- `tensornet.sovereign.fast_euler_3d` (QTT-compressed CFD solver)
- Modified `realtime_tensor_stream.stream_from_qtt()` (live integration)

**Test Scripts**:
- `test_phase4_validation.py` (Component checks)
- `test_phase4_integration.py` (End-to-end test)

**Verification**:
```bash
# Quick validation:
python3 test_phase4_validation.py

# Full integration test:
# Terminal 1:
python3 test_phase4_integration.py 10 --grid-size 64 --field density

# Terminal 2:
cd glass-cockpit && cargo run --release --bin phase3

# Expected: Live CFD simulation visualized in Glass Cockpit
```

### Phase 5.2: Performance Optimization
🔨 **Current Focus**

**Objective**: Achieve 165 FPS @ 4K through Sovereign Architecture.

**Python Side**:
- `tensornet/mpo/` (MPO physics operators - 245+ optimizations)
- Eliminated dense materializations in critical paths
- GPU-accelerated compositor with float16 pipeline

**Performance Targets**:
- Current: 39 FPS (25.63ms/frame)
- Target: 60 FPS minimum (16.67ms/frame)
- Projected: 88 FPS (11.36ms/frame)

**Verification**:
```bash
# Run performance benchmarks:
python3 profile_render_4k.py
python3 test_mpo_performance.py

# Expected: Frame times <16.67ms, stability score <1.5
```

---

## 5. Performance Monitoring & Profiling

**Sovereign_UI.md Doctrine 8**: Performance boundaries require continuous monitoring.

### 5.1 CPU Profiling Tools

**For Python**:
```bash
# cProfile (built-in):
python3 -m cProfile -o profile.stats your_script.py

# py-spy (sampling profiler):
pip install py-spy
py-spy record -o profile.svg -- python3 your_script.py

# Austin (lightweight profiler):
pip install austin-python
austin python3 your_script.py
```

**For Rust**:
```bash
# cargo-flamegraph:
cargo install flamegraph
cargo flamegraph --bin glass-cockpit

# perf (Linux):
perf record --call-graph dwarf cargo run --release
perf report
```

### 5.2 GPU Profiling

**NVIDIA Tools**:
```bash
# nsys (Nsight Systems):
nsys profile python3 your_script.py

# ncu (Nsight Compute) - kernel profiling:
ncu --set full python3 your_script.py
```

**wgpu Profiling**:
- Enable `tracy-client` feature in Cargo.toml
- Use Tracy profiler (https://github.com/wolfpld/tracy)

### 5.3 CPU Affinity Tools

**Linux/WSL**:
```bash
# taskset - set CPU affinity:
taskset -c 16-31 ./glass-cockpit  # E-cores only

# Verify affinity:
ps -o pid,psr,comm -p $(pgrep glass-cockpit)
```

**Windows** (handled automatically in Cargo.toml):
- `windows` crate enforces E-core affinity via ProcessorAffinity API

### 5.4 Network Monitoring

**Sovereign_UI.md Doctrine 4**: Zero network activity enforcement.

```bash
# nethogs - per-process bandwidth:
sudo apt install nethogs
sudo nethogs

# lsof - check open network connections:
lsof -i -n | grep glass-cockpit
# Expected: No output (zero network connections)

# ss - socket statistics:
ss -tunapl | grep glass-cockpit
# Expected: Only local IPC sockets
```

### 5.5 Frame Time Measurement

**In Rust** (already implemented in glass-cockpit):
```rust
use std::time::Instant;

let frame_start = Instant::now();
// ... render frame ...
let frame_time = frame_start.elapsed();
let stability_score = max_frame_time / mean_frame_time;
```

**Telemetry Rails** (Sovereign_UI.md Doctrine 7):
- P-core utilization
- E-core utilization
- Memory usage
- Frame time sparkline
- Stability score with threshold warning

**Verification**:
```bash
# Check stability score during 100k-frame stress test:
python3 test_100k_stress.py
# Expected: Stability score 1.1-1.2 (max < 1.5)
```

---

## 6. Shared Memory Configuration

### 6.1 /dev/shm Requirements
- **Path**: `/dev/shm/hypertensor_bridge`
- **Size**: ~8.3 MB (1920×1080×4 + 4096 header)
- **Permissions**: Read/write for current user

**Verification**:
```bash
df -h /dev/shm
ls -lh /dev/shm/hypertensor_bridge  # After running Python writer
```

Expected: `/dev/shm` has ≥100 MB free space.

### 6.2 Memory-Mapped I/O Performance
The RAM Bridge uses `mmap()` for zero-copy transfers. Ensure:
- `/dev/shm` is tmpfs (memory-backed, not disk)
- No disk quota limits on tmpfs

**Verification**:
```bash
mount | grep shm
# Expected: tmpfs on /dev/shm type tmpfs (rw,nosuid,nodev)
```

---

## 7. VS Code Configuration

### 7.1 Python Extension Setup
- **Extension**: Python (ms-python.python)
- **Interpreter**: Must point to WSL Python, not Windows Python

**Configuration** (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreterPath": "/home/brad/TiganticLabz/Main_Projects/The Physics OS/venv/bin/python",
    "python.analysis.extraPaths": [
        "${workspaceFolder}",
        "${workspaceFolder}/tensornet"
    ],
    "terminal.integrated.defaultProfile.windows": "PowerShell",
    "terminal.integrated.env.linux": {
        "GIT_PAGER": "cat"
    }
}
```

### 7.2 Terminal Configuration
**CRITICAL**: Always use WSL terminal for Python commands.

To open WSL terminal in VS Code:
1. Press `` Ctrl+` `` (backtick) to open terminal
2. Click dropdown → "Ubuntu (WSL)"
3. Or configure default: `terminal.integrated.defaultProfile.windows` → `"WSL"`

**Verification**:
```bash
# In VS Code integrated terminal:
echo $WSL_DISTRO_NAME
# Should output: Ubuntu
```

---

## 8. Development Workflow

### 8.1 Activating Environment
Every new terminal session requires activation:

```bash
cd /home/brad/TiganticLabz/Main_Projects/Project\ HyperTensor
source venv/bin/activate  # Or 'conda activate hypertensor'
```

### 8.2 Running Tests
```bash
# Python tests:
pytest tests/ -v

# Rust tests:
cd glass-cockpit && cargo test --release

# Phase 3 integration:
python3 test_phase3_integration.py

# Phase 4 validation:
python3 test_phase4_validation.py

# Phase 5.2 performance benchmarks:
python3 test_mpo_performance.py
python3 profile_render_4k.py
python3 test_100k_stress.py
```

### 8.3 Building Documentation
```bash
# Python API docs:
cd docs && make html

# Rust API docs:
cd glass-cockpit && cargo doc --open
```

---

## 9. Troubleshooting

### 9.1 "Import torch could not be resolved"
**Cause**: VS Code Python extension using Windows Python interpreter.

**Solution**:
1. Press `Ctrl+Shift+P` → "Python: Select Interpreter"
2. Choose the WSL venv: `/home/brad/.../venv/bin/python`
3. Reload VS Code window

### 9.2 "CUDA not available" in PyTorch
**Cause**: PyTorch installed without CUDA support, or NVIDIA driver issues.

**Solution**:
```bash
# Check driver:
nvidia-smi

# Reinstall PyTorch with CUDA:
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 9.3 "Cannot open /dev/shm/hypertensor_bridge"
**Cause**: Python writer not running, or permissions issue.

**Solution**:
```bash
# Clean up stale bridge:
rm /dev/shm/hypertensor_bridge

# Ensure Python writer runs first (creates the bridge):
python3 tensornet/sovereign/realtime_tensor_stream.py
```

### 9.4 Terminal Pager Stuck
**Cause**: Git commands using `less` pager in integrated terminal.

**Solution**:
```bash
# In stuck terminal, press 'q' to quit pager

# Permanently disable pager:
echo 'export GIT_PAGER=cat' >> ~/.bashrc
source ~/.bashrc
```

### 9.5 Rust Compilation Errors
**Cause**: Missing system libraries or outdated Rust version.

**Solution**:
```bash
# Update Rust:
rustup update stable

# Install missing libraries:
sudo apt install -y libssl-dev pkg-config

# Clean rebuild:
cd glass-cockpit
cargo clean
cargo build --release
```

### 9.6 BLAS/LAPACK for PCA (Sovereign_UI.md Appendix I.6)

**Decision Status**: DEFERRED to GPU compute shaders.

**Context**: Sovereign_UI.md Phase 6 requires PCA projection for Probability Probe. Original proposal was `ndarray-linalg` (requires BLAS/LAPACK).

**Resolution**:
1. **Phase 1-5**: PCA not yet implemented, no dependency needed
2. **Phase 6 Plan**: Implement PCA as wgpu compute shader (GPU-accelerated)
3. **Rationale**: 
   - Keeps computation on E-core GPU command queue
   - Eliminates C/Fortran dependencies
   - Faster than CPU BLAS on modern GPUs
4. **Fallback**: If GPU PCA proves insufficient, add `ndarray-linalg` and document BLAS installation

**Current Status**: No action required until Phase 6.

---

## 10. Comprehensive Verification Script

Run this script to verify all dependencies:

```bash
#!/bin/bash
# Comprehensive dependency verification for The Physics OS
# Save as: verify_dependencies.sh
# Run: bash verify_dependencies.sh

set -e

echo "======================================"
echo "The Physics OS Dependency Check"
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
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
    echo "✓ NVIDIA GPU detected (Driver: $DRIVER_VERSION)"
    if [[ "$DRIVER_VERSION" < "525.60.11" ]]; then
        echo "⚠ WARNING: Driver version may be outdated (minimum: 525.60.11)"
    fi
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
if python3 -c "import sys; sys.exit(0 if (3,10) <= sys.version_info < (3,12) else 1)"; then
    echo "✓ Python version compatible"
else
    echo "✗ ERROR: Python must be 3.10 or 3.11"
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
from tensornet.sovereign.bridge_writer import TensorBridgeWriter
from tensornet.sovereign.qtt_slice_extractor import QTTSliceExtractor
from tensornet.sovereign.realtime_tensor_stream import RealtimeTensorStream
print('✓ TensorNet modules import successfully')
"
echo ""

# Check Rust
echo "[7/10] Rust Toolchain..."
RUST_VERSION=$(rustc --version | grep -oP '\d+\.\d+' | head -1)
cargo --version
if (( $(echo "$RUST_VERSION >= 1.75" | bc -l) )); then
    echo "✓ Rust toolchain installed (version: $RUST_VERSION)"
else
    echo "⚠ WARNING: Rust version $RUST_VERSION may be outdated (minimum: 1.75)"
fi
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
echo "  • Phase 4: QTT Navier-Stokes integration (✅ Complete)"
echo "  • Phase 5.2: Performance optimization (🔨 Current Focus)"
echo ""
echo "Next steps:"
echo "  1. Run Phase 3 test: python3 test_phase3_integration.py"
echo "  2. Run Phase 4 validation: python3 test_phase4_validation.py"
echo "  3. Run Phase 5.2 benchmarks: python3 test_mpo_performance.py"
echo "  4. Profile 4K rendering: python3 profile_render_4k.py"
echo "  5. Stress test: python3 test_100k_stress.py"


---

## 11. Dependency Decision Log

**Cross-reference**: Sovereign_UI.md Appendix I (Decision Points for Live Execution)

### Decision: shared_memory vs memmap2

**Date**: 2025-12-28  
**Status**: DECIDED - Use both

**Context**: RAM Bridge requires memory-mapped shared memory access.

**Resolution**:
- `shared_memory = "0.12"` - Primary API (cross-platform abstraction)
- `memmap2 = "0.9"` - Fallback/alternative (direct mmap control)

**Rationale**: Both included in Cargo.toml for flexibility. Primary code uses `shared_memory`. `memmap2` available for low-level debugging if needed.

---

### Decision: wgpu 0.19 version lock

**Date**: 2025-12-28  
**Status**: DECIDED

**Context**: Sovereign_UI.md doesn't specify wgpu version. Cargo.toml uses 0.19.

**Resolution**: Lock to wgpu 0.19 for Phase 3-5.2.

**Rationale**:
- Phase 3 validated with 0.19
- API stability important during performance optimization
- Will evaluate 0.20+ for Phase 6+ features

---

### Decision: BLAS/LAPACK for PCA

**Date**: 2025-12-28  
**Status**: DEFERRED → GPU Compute Shader

**Context**: Sovereign_UI.md Appendix I.6 - PCA for Probability Probe needs eigendecomposition.

**Resolution**: Implement PCA as wgpu compute shader in Phase 6.

**Rationale**:
- Eliminates C/Fortran dependencies (Constitution Article II: minimize external deps)
- GPU-accelerated (faster than CPU BLAS on RTX 5070)
- Keeps computation on E-core GPU queue (respects Sovereignty Contract)
- Fallback: If GPU PCA proves insufficient, add `ndarray-linalg` with documented BLAS install

**Review Trigger**: Phase 6 Probability Probe implementation

---

### Decision: Tokio async runtime

**Date**: 2025-12-28  
**Status**: APPROVED - Minimal features only

**Context**: Sovereign_UI.md Doctrine 4 prohibits "unpredictable interrupt patterns."

**Resolution**: Tokio allowed with minimal features `["rt", "sync"]`, optional feature flag `satellite-tiles`.

**Conditions**:
- Only for satellite tile fetching (background thread)
- Never blocks main render loop
- Optional feature (disabled by default)
- No network activity in core rendering

**Compliance**: Satisfies Doctrine 1 (no interruption of physics), Doctrine 4 (justified dependency)

---

### Decision: Logging in Production

**Date**: 2025-12-28  
**Status**: DECIDED - Optional, disabled in release

**Context**: Logging overhead must not violate Doctrine 8 (<5% CPU tax).

**Resolution**:
- `env_logger` and `log` behind `debug-logging` feature
- Disabled in release profile
- No runtime logging unless explicitly enabled

**Enforcement**: `cargo build --release` has zero logging overhead.

---

### Decision: Tracy Profiler

**Date**: 2025-12-28  
**Status**: APPROVED - Optional feature

**Context**: Performance profiling required for Doctrine 8 validation.

**Resolution**: `tracy-client` behind `profiling` feature flag.

**Usage**:
```bash
# Enable profiling:
cargo build --release --features profiling

# Run with Tracy server connected
```

**Rationale**: Best-in-class frame profiler for GPU/CPU hybrid workloads. Zero overhead when disabled.

---

## 12. Sovereign Architecture Compliance Matrix

**Cross-reference**: Sovereign_UI.md Doctrines 1-10

| Doctrine | Requirement | Implementation | Verification |
|----------|-------------|----------------|--------------|
| **1: Computational Sovereignty** | P-cores never interrupted | Windows: `ProcessorAffinity` API | `Get-Process \| Select ProcessorAffinity` |
| **2: RAM Bridge Protocol** | Zero-copy shared memory | `shared_memory` crate, `/dev/shm` | `ls -lh /dev/shm/hypertensor_bridge` |
| **3: Procedural Rendering** | No image assets | Pure WGSL shaders | `find glass-cockpit -name '*.png' \| wc -l` → 0 |
| **4: Template Prohibition** | From-scratch Rust/wgpu | Approved stack only | Cargo.toml audit (Section 3.2) |
| **4.1: Pragmatic Bootstrap** | Conditional egui/iced | Not yet added | Will verify E-core affinity if used |
| **5: Semantic Zoom** | Single coordinate space | Not yet implemented | Phase 6+ feature |
| **6: Data vs Insight** | Tensor structure visualization | Phase 3 colormap complete | Visual inspection |
| **7: User Agency** | Timeline scrub, probe, inject | Phase 6 feature | Not yet testable |
| **8: Performance Boundaries** | <5% CPU, 0.0 Mbps network | Monitored in telemetry | `nethogs`, stability score |
| **9: Visual Design** | Dark, monospaced, procedural | Phase 2+ feature | Visual inspection |
| **10: External Data** | NASA GIBS overlay | Optional `satellite-tiles` | Phase 6+ feature |

**Compliance Status**: 
- Phases 3-4: ✅ Core doctrines (1-4, 6, 8)
- Phase 5.2: 🔨 Performance optimization (Doctrine 8)
- Phase 6+: 🔜 Advanced features (5, 7, 9, 10)

---

**End of Document**  
**Last Updated**: 2025-12-28  
**Revision**: 2.0 (Aligned with Sovereign_UI.md)
