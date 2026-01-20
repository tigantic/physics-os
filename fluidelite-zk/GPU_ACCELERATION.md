# FluidElite ZK GPU Acceleration

## Overview

FluidElite ZK prover now supports GPU-accelerated cryptographic operations using Ingonyama's ICICLE library. This provides significant speedups for Multi-Scalar Multiplication (MSM) operations, which are the primary bottleneck in Halo2/KZG proof generation.

## Hardware Requirements

- **NVIDIA GPU**: Compute Capability 7.0+ (RTX 20xx or newer)
- **CUDA**: Version 12.0+
- **VRAM**: Minimum 4GB, 8GB+ recommended for production

### Tested Configuration
- NVIDIA GeForce RTX 5070 Laptop GPU (8GB VRAM)
- CUDA 13.1 / Driver 591.74
- ICICLE v4.0.0 with BN254 curve support

## Performance Results

| MSM Size | Points | Time | Throughput |
|----------|--------|------|------------|
| 2^10 | 1,024 | ~7 ms | 140K pts/sec |
| 2^14 | 16,384 | ~8 ms | 2M pts/sec |
| 2^16 | 65,536 | ~10 ms | 6M pts/sec |
| 2^18 | 262,144 | ~16 ms | 16M pts/sec |
| **2^20** | **1,048,576** | **~37 ms** | **28M pts/sec** |

## Installation

### 1. Download ICICLE CUDA Backend

```bash
# Download ICICLE v4.0.0 CUDA backend
wget https://github.com/ingonyama-zk/icicle/releases/download/v4.0.0/icicle_ubuntu22_cuda122_with_bn254_with_bls12-377_with_bls12-381_with_bw6-761_with_grumpkin.tar.gz

# Extract to /opt/icicle
sudo mkdir -p /opt/icicle/lib/backend
sudo tar -xzf icicle*.tar.gz -C /opt/icicle/lib/backend

# Verify installation
ls /opt/icicle/lib/backend/
# Should show: backend.toml, cuda12/
```

### 2. Build with GPU Support

```bash
cd fluidelite-zk

# Build with GPU feature
cargo build --release --features gpu

# Or build production server with GPU
cargo build --release --features production-gpu
```

### 3. Run GPU Test

```bash
# Using the launcher script (recommended)
./scripts/run-gpu.sh gpu-test

# Or manually
export ICICLE_BACKEND_INSTALL_DIR=/opt/icicle/lib/backend
export LD_LIBRARY_PATH="$(find target/release/build -name 'libicicle*.so' -printf '%h\n' | sort -u | tr '\n' ':')$LD_LIBRARY_PATH"
./target/release/gpu-test
```

## Usage

### GPU Launcher Script

The `scripts/run-gpu.sh` script handles all environment setup:

```bash
# Run GPU benchmark
./scripts/run-gpu.sh gpu-test

# Start GPU-accelerated server
./scripts/run-gpu.sh server --test

# Use CLI with GPU
./scripts/run-gpu.sh cli prove 42
```

### Programmatic API

```rust
use fluidelite_zk::gpu::GpuAccelerator;

// Initialize GPU (loads CUDA backend)
let gpu = GpuAccelerator::new()?;

// Check device info
println!("Device: {}", gpu.device_name());
println!("Using GPU: {}", gpu.is_gpu());

// Run MSM on GPU
let result = gpu.msm_bn254(&points, &scalars)?;
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ICICLE_BACKEND_INSTALL_DIR` | `/opt/icicle/lib/backend` | Path to ICICLE CUDA backend |
| `LD_LIBRARY_PATH` | - | Must include ICICLE build libraries |

### Cargo Features

| Feature | Description |
|---------|-------------|
| `gpu` | Enable GPU acceleration (ICICLE) |
| `production` | Production server features |
| `production-gpu` | Both production + GPU |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FluidElite ZK Prover                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │   Halo2 Axiom   │  │   KZG Commits   │  │  PLONK IOP  │  │
│  └────────┬────────┘  └────────┬────────┘  └──────┬──────┘  │
│           │                    │                   │         │
│           ▼                    ▼                   ▼         │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              GPU Accelerator (gpu.rs)                   ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  ││
│  │  │  msm_bn254  │  │ ntt_forward │  │  ntt_inverse    │  ││
│  │  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘  ││
│  └─────────┼────────────────┼──────────────────┼───────────┘│
│            │                │                  │             │
├────────────┼────────────────┼──────────────────┼─────────────┤
│            ▼                ▼                  ▼             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │           ICICLE v4.0.0 (BN254 Curve)                   ││
│  │  ┌───────────────────────────────────────────────────┐  ││
│  │  │   CUDA Backend (/opt/icicle/lib/backend/cuda12)   │  ││
│  │  │   • GPU MSM Kernel (Pippenger)                    │  ││
│  │  │   • GPU NTT Kernel (Cooley-Tukey)                 │  ││
│  │  └───────────────────────────────────────────────────┘  ││
│  └─────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│                    NVIDIA RTX 5070 GPU                      │
│                  8GB VRAM | CUDA 13.1                       │
└─────────────────────────────────────────────────────────────┘
```

## Troubleshooting

### GPU Not Detected

```
⚠️  Falling back to CPU (CUDA backend not loaded)
```

**Solution**: Verify ICICLE backend is installed:
```bash
ls -la /opt/icicle/lib/backend/
# Should contain: backend.toml, cuda12/
```

### Missing Shared Libraries

```
error while loading shared libraries: libicicle_field_bn254.so: cannot open shared object file
```

**Solution**: Set LD_LIBRARY_PATH or use the launcher script:
```bash
./scripts/run-gpu.sh <command>
```

### License Warning

```
[WARNING] Defaulting to Ingonyama icicle-cuda-license-server
```

This is informational - the community license server is used automatically.

## License

ICICLE is licensed under Ingonyama's terms. For commercial use, contact support@ingonyama.com.

## References

- [ICICLE GitHub](https://github.com/ingonyama-zk/icicle)
- [ICICLE v4 Documentation](https://dev.ingonyama.com/)
- [Halo2 Axiom](https://github.com/axiom-crypto/halo2)
