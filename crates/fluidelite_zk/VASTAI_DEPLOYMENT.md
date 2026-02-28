# FluidElite ZK Benchmark Deployment on Vast.ai

This guide covers deploying and running the Fluid-ZK benchmark suite on vast.ai cloud GPU instances to validate performance consistency across different hardware.

## 🎯 Quick Start

### Option A: Pre-built Docker (Recommended)

```bash
# On vast.ai instance with Docker template
docker pull your-registry/fluidelite-zk-bench:latest
docker run --gpus all -it fluidelite-zk-bench:latest
./run_vastai_benchmarks.sh --standard
```

### Option B: Fresh Instance Setup

```bash
# SSH into your vast.ai instance, then:
curl -sSL https://raw.githubusercontent.com/tigantic/physics-os/main/crates/fluidelite_zk/scripts/vast_setup.sh | bash

# After setup completes:
cd ~/physics-os/fluidelite-zk
./scripts/run_vastai_benchmarks.sh --standard
```

---

## 📋 Prerequisites

### Vast.ai Instance Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| GPU | RTX 3080 (10GB VRAM) | RTX 4090 / A100 (24GB+) |
| CUDA | 12.0+ | 12.2+ |
| CPU | 4 cores | 8+ cores |
| RAM | 16 GB | 32+ GB |
| Disk | 30 GB | 50+ GB |
| Image | `nvidia/cuda:12.2.2-devel-ubuntu22.04` | Same |

### Recommended GPU Instances for Benchmarking

| GPU | VRAM | Expected TPS (2^18) | Price/hr (approx) |
|-----|------|---------------------|-------------------|
| RTX 4090 | 24 GB | ~150-200 TPS | $0.40-0.60 |
| RTX 3090 | 24 GB | ~100-140 TPS | $0.25-0.40 |
| A100 40GB | 40 GB | ~180-250 TPS | $1.00-1.50 |
| A100 80GB | 80 GB | ~200-280 TPS | $1.50-2.50 |
| H100 | 80 GB | ~300-400 TPS | $2.50-4.00 |

---

## 📖 Step-by-Step Deployment

### Step 1: Create Vast.ai Instance

1. Go to [vast.ai/console/create/](https://vast.ai/console/create/)
2. Select GPU type (RTX 4090 or A100 recommended)
3. Choose Docker image: `nvidia/cuda:12.2.2-devel-ubuntu22.04`
4. Set disk space: 50 GB minimum
5. Enable SSH access
6. Click "RENT"

### Step 2: Connect to Instance

```bash
# Get SSH command from vast.ai console, looks like:
ssh -p 12345 root@ssh1.vast.ai

# Or use the vast.ai CLI:
vastai ssh-url <instance_id>
```

### Step 3: Run Setup Script

```bash
# Download and run the setup script
wget https://raw.githubusercontent.com/tigantic/physics-os/main/crates/fluidelite_zk/scripts/vast_setup.sh
chmod +x vast_setup.sh
./vast_setup.sh
```

This script will:
- Verify GPU availability and CUDA installation
- Install Rust (nightly) and build dependencies
- Download ICICLE v4.0.0 CUDA backend
- Clone the physics-os repository
- Build all benchmark binaries with GPU support

### Step 4: Run Benchmarks

```bash
cd ~/physics-os/fluidelite-zk

# Quick sanity check (2-3 minutes)
./scripts/run_vastai_benchmarks.sh --quick

# Standard benchmarks (10-15 minutes) - RECOMMENDED
./scripts/run_vastai_benchmarks.sh --standard

# Full exhaustive suite (30+ minutes)
./scripts/run_vastai_benchmarks.sh --full
```

### Step 5: Download Results

```bash
# From your local machine:
scp -P 12345 root@ssh1.vast.ai:~/benchmark_results/*.json ./

# Or use rsync for all results:
rsync -avz -e "ssh -p 12345" root@ssh1.vast.ai:~/benchmark_results/ ./vastai_results/
```

---

## 📊 Benchmark Modes

### Quick Mode (`--quick`)
- **Duration**: 2-3 minutes
- **Purpose**: Verify GPU functionality
- **Tests**: `gpu-test`

### Standard Mode (`--standard`)
- **Duration**: 10-15 minutes  
- **Purpose**: Performance validation
- **Tests**: 
  - `gpu-test` - ICICLE backend verification
  - `gpu-sustained-bench` - Sustained MSM throughput
  - `gpu-realworld-tps` - Production workload simulation

### Full Mode (`--full`)
- **Duration**: 30+ minutes
- **Purpose**: Exhaustive characterization
- **Tests**:
  - All standard tests plus:
  - `gpu-halo2-benchmark` - Full Halo2 prover benchmark
  - `gpu-pipelined-tps` - Multi-stream CUDA pipelining
  - `k-ladder-stress` - MSM scaling across circuit sizes

---

## 🔍 Understanding Results

### Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| TPS | Transactions (proofs) per second | ≥88 @ 2^18 |
| P50 Latency | Median proof time | <15ms |
| P99 Latency | Tail latency | <50ms |
| VRAM Usage | Peak GPU memory | <80% available |

### Comparing with Local Results

Your local benchmark baseline (RTX 5070 Laptop):
- GPU: NVIDIA GeForce RTX 5070 Laptop GPU (8GB VRAM)
- TPS @ 2^18: ~88-120 TPS
- MSM 2^20: ~37ms

Cloud instances should show:
- **RTX 4090**: 1.5-2x local performance
- **A100**: 2-3x local performance
- **H100**: 3-4x local performance

### Sample Output

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         BENCHMARK SUMMARY                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ┌──────────┬──────────────┬────────────┬────────────┬────────────┐          ║
║  │   Size   │  Total Proofs│    TPS     │  P50 (ms)  │  P99 (ms)  │          ║
║  ├──────────┼──────────────┼────────────┼────────────┼────────────┤          ║
║  │  2^16    │         1000 │     234.56 │       4.12 │       6.78 │          ║
║  │  2^18    │          500 │     156.78 │      12.34 │      18.90 │          ║
║  │  2^20    │          100 │      45.67 │      38.45 │      52.10 │          ║
║  └──────────┴──────────────┴────────────┴────────────┴────────────┘          ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 🐳 Docker Deployment (Alternative)

### Build Docker Image Locally

```bash
cd fluidelite-zk

# Build the vast.ai optimized image
docker build -f Dockerfile.vastai -t fluidelite-zk-bench:latest .

# Test locally (if you have NVIDIA GPU)
docker run --gpus all -it fluidelite-zk-bench:latest
./run_vastai_benchmarks.sh --quick
```

### Push to Registry

```bash
# Tag for Docker Hub
docker tag fluidelite-zk-bench:latest your-dockerhub/fluidelite-zk-bench:latest
docker push your-dockerhub/fluidelite-zk-bench:latest

# Or for GitHub Container Registry
docker tag fluidelite-zk-bench:latest ghcr.io/tiganticlabz/fluidelite-zk-bench:latest
docker push ghcr.io/tiganticlabz/fluidelite-zk-bench:latest
```

### Use on Vast.ai

1. Create instance with "Custom Docker Image"
2. Enter: `your-registry/fluidelite-zk-bench:latest`
3. Enable "Run interactive shell"
4. Launch and run benchmarks

---

## 🔧 Troubleshooting

### ICICLE Backend Not Found

```bash
# Verify ICICLE installation
ls -la /opt/icicle/lib/backend/

# Should show:
# backend.toml
# cuda12/

# If missing, reinstall:
wget https://github.com/ingonyama-zk/icicle/releases/download/v4.0.0/icicle_ubuntu22_cuda122_with_bn254_with_bls12-377_with_bls12-381_with_bw6-761_with_grumpkin.tar.gz
sudo tar -xzf icicle*.tar.gz -C /opt/icicle/lib/backend
```

### CUDA Version Mismatch

```bash
# Check CUDA version
nvcc --version
nvidia-smi

# ICICLE v4.0.0 requires CUDA 12.x
# If you have CUDA 11.x, use a different vast.ai image:
# nvidia/cuda:12.2.2-devel-ubuntu22.04
```

### Build Failures

```bash
# Clean and rebuild
cd ~/physics-os/fluidelite-zk
cargo clean
cargo build --release --features gpu

# Check for missing dependencies
apt-get install -y build-essential pkg-config libssl-dev cmake
```

### GPU Not Detected

```bash
# Verify GPU access
nvidia-smi

# Check Docker GPU access (if using Docker)
docker run --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi
```

---

## 📈 Performance Expectations by GPU

Based on MSM 2^18 (262,144 points) benchmark:

| GPU | Architecture | Memory BW | Expected TPS | Notes |
|-----|--------------|-----------|--------------|-------|
| RTX 3080 | Ampere | 760 GB/s | 80-100 | Entry-level cloud |
| RTX 3090 | Ampere | 936 GB/s | 100-130 | Good value |
| RTX 4090 | Ada | 1,008 GB/s | 150-200 | Best consumer |
| A100 40GB | Ampere | 1,555 GB/s | 180-250 | Enterprise |
| A100 80GB | Ampere | 2,039 GB/s | 200-280 | High memory |
| H100 | Hopper | 3,350 GB/s | 300-400 | Top performance |

---

## 📝 JSON Results Format

Results are saved in JSON format for easy parsing:

```json
{
  "platform": "vast.ai",
  "hostname": "C.12345",
  "timestamp": "2026-02-01T15:30:00Z",
  "mode": "standard",
  "system": {
    "gpu_name": "NVIDIA A100-SXM4-40GB",
    "gpu_vram_mb": 40960,
    "gpu_driver": "535.104.05",
    "cuda_version": "12.2",
    "compute_capability": "8.0",
    "cpu_model": "AMD EPYC 7742",
    "cpu_cores": 16,
    "ram_gb": 128
  },
  "benchmarks": {
    // Individual benchmark results
  },
  "summary": {
    "benchmarks_run": 3,
    "benchmarks_passed": 3,
    "completed_at": "2026-02-01T15:45:00Z"
  }
}
```

---

## 🎯 Success Criteria

Your vast.ai benchmark run is successful if:

1. ✅ All benchmark binaries execute without error
2. ✅ TPS @ 2^18 meets or exceeds local baseline
3. ✅ P99 latency < 50ms for 2^18 proofs
4. ✅ No GPU memory errors or OOM
5. ✅ Results are reproducible (±10% variance)

---

## 📞 Support

- **Issues**: Open GitHub issue with benchmark logs
- **Performance questions**: Include full JSON results
- **GPU recommendations**: Check latest vast.ai pricing
