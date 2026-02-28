# The Physics OS — QTT Engine — Executive Summary

**Quantized Tensor Train Navier-Stokes Solver vs Dense CFD**

| | |
|---|---|
| **Date** | February 22, 2026 |
| **Author** | Brad Adams, Tigantic Holdings LLC |
| **Classification** | Commercial — Pre-Release Benchmark |
| **Engine** | Ontic Engine QTT v2.0 |
| **Hardware** | Single NVIDIA RTX 5070 Laptop GPU (8 GB VRAM) |

---

## The Problem

Industrial CFD datasets are unsustainably large. NVIDIA's PhysicsNeMo Ahmed Body
dataset — 4,064 parametric RANS simulations of a single automotive geometry — consumes
**46.7 GB** of storage for surface-only data. Gridded volumetric equivalents exceed
**95 GB**. These datasets must be transferred, stored, and served at scale across
data center infrastructure. For parametric design sweeps at production resolutions
(512³+), dense storage requirements grow cubically and become prohibitive.

## The Solution

The Physics OS's Quantized Tensor Train (QTT) Navier-Stokes engine replaces dense CFD
entirely. Instead of solving on a mesh and storing N³ arrays, the solver operates
natively in compressed tensor-train format with **O(r² log N)** storage and compute —
logarithmic in grid size. A complete 3D velocity field at 512³ resolution (134 million
cells) is stored in **68.6 KB**. Not megabytes. Kilobytes.

## Benchmark Results

### Multi-Resolution Gauntlet — Ahmed Body Flow (Re = 2.75M)

| Resolution | Dense Size | QTT Size | Compression | Wall Time | Hardware |
|:----------:|:----------:|:--------:|:-----------:|:---------:|:--------:|
| 128³ | 25.2 MB | 272 KB | **92×** | 289 s | 1× RTX 5070 |
| 256³ | 201.3 MB | 182 KB | **1,109×** | 400 s | 1× RTX 5070 |
| 512³ | 1,610.6 MB | 68.6 KB | **23,465×** | 546 s | 1× RTX 5070 |

### Critical Observation

**QTT storage decreases as resolution increases.** At 512³, the velocity field
requires less storage (68.6 KB) than at 128³ (272 KB). This is the defining property
of tensor-train compression applied to smooth physical fields: finer grids resolve
more structure, but adaptive rank truncation captures that structure with fewer
degrees of freedom per scale.

### Direct Comparison with NVIDIA

| Metric | NVIDIA PhysicsNeMo | Ontic Engine QTT | Advantage |
|:-------|:------------------:|:----------------:|:---------:|
| Storage per sample | 11.5 MB (VTP) | 68.6 KB | **168×** |
| Data content | Surface fields only | **Full 3D volume** | Volumetric |
| 4,064-sample equivalent | 46.7 GB | 279 MB | **167×** |
| Generation method | HPC cluster RANS | Single laptop GPU | **Orders cheaper** |
| Grid resolution | 128 × 64 × 64 | **512 × 512 × 512** | **246× more cells** |

One QTT sample is **168× smaller** than one NVIDIA VTP file, while containing the
**complete volumetric flow field** — not just surface quantities.

### Scaling Projection

| Resolution | Dense | QTT (projected) | Compression |
|:----------:|:-----:|:----------------:|:-----------:|
| 512³ | 1.6 GB | 0.07 MB | 23,465× |
| 1024³ | 12.9 GB | 0.08 MB | 152,051× |
| 2048³ | 103.1 GB | 0.10 MB | 1,005,295× |
| 4096³ | 824.6 GB | 0.12 MB | **6,757,816×** |

At 4096³ — a resolution relevant to production LES — a single velocity field would
require 825 GB dense. QTT stores it in approximately **120 KB**.

### Physics Validation

The solver produces physically correct turbulent energy spectra. At 128³, the fitted
spectral exponent is **α = −1.664**, within 0.003 of the Kolmogorov −5/3 law
(R² = 0.934). Energy dissipation across all resolutions is 2.9–3.2%, consistent
with physical viscous decay.

## Strategic Implications

1. **Data Center Storage**: A parametric sweep of 10,000 simulations at 512³ requires
   ~670 MB in QTT vs ~15 TB dense. This eliminates storage as a cost driver.

2. **Edge Deployment**: Full volumetric CFD results can be transmitted over cellular
   networks in milliseconds. A 512³ velocity field (68.6 KB) transfers in <100 ms
   on 4G LTE.

3. **Real-Time Design**: Wall times of 5–10 minutes per simulation on a laptop GPU
   enable interactive parametric exploration without HPC infrastructure.

4. **AI/ML Training**: QTT fields can be evaluated at arbitrary query points without
   decompression. Training physics-informed models against QTT representations
   eliminates the I/O bottleneck of loading dense arrays.

## Engineering Compliance

All six QTT engineering rules are verified:

| Rule | Status | Detail |
|:-----|:------:|:-------|
| QTT is Native | ✓ | Morton-ordered TT cores, zero external format conversion |
| SVD = rSVD | ✓ | Randomized SVD threshold=48, fires on every truncation |
| Python → Triton | ✓ | Hadamard and inner product dispatch to L2-optimized Triton kernels |
| Adaptive Rank | ✓ | Bell-curve rank profile: higher scale → lower rank |
| No Decompression | ✓ | Zero dense reconstruction in solver hot path |
| No Dense | ✓ | Separable fields (sponge, corrections) built zero-dense |

---

*Total gauntlet execution: 20.6 minutes on a single NVIDIA RTX 5070 Laptop GPU.*

*Ontic Engine QTT Engine — Tigantic Holdings LLC — February 2026*
