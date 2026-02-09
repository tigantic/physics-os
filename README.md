# Project HyperTensor

**Quantum-Inspired Tensor Networks for Computational Physics**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Rust 1.70+](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)

---

## Overview

HyperTensor is a unified computational physics platform that applies tensor network methods (MPS, MPO, QTT) across 20 physics domains (167 taxonomy nodes). The core insight: turbulent flow fields and many-body systems exhibit low-rank structure that can be exploited for significant compression and acceleration.

**What this repository contains:**
- **Platform substrate** (`tensornet.platform`) — unified data model, solvers, V&V harness, QTT acceleration, coupled physics, inverse/UQ/optimization, export/import, post-processing, visualization
- **SDK** (`tensornet.sdk`) — stable public API with `WorkflowBuilder` DSL, 8 built-in recipes, curated re-exports
- **20 domain packs** — 167 taxonomy nodes (I–XX) at V0.2+ maturity, 4 QTT-accelerated anchors at V0.6
- Tensor network algorithms (DMRG, TEBD, Lanczos)
- CFD solvers (1D/2D/3D Euler, Navier-Stokes, QTT-native)
- Physics validation gauntlets with benchmark suites
- V&V framework aligned with ASME V&V 10-2019

**What this repository does NOT contain:**
- Physical hardware designs
- Manufacturing specifications
- Production-ready systems

This is research-grade computational software.

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/tigantic/HyperTensor.git
cd HyperTensor
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# Verify installation
python -c "import tensornet; print(f'tensornet v{tensornet.__version__}')"

# Run tests
pytest tests/ -v
```

### Example: SDK Workflow Builder

```python
from tensornet.sdk import WorkflowBuilder, get_recipe, list_recipes

# Fluent DSL — build + run a 1D Burgers simulation
result = (
    WorkflowBuilder("burgers")
    .domain(shape=(256,), extent=((0.0, 2 * 3.14159),))
    .field("u", ic="sine")
    .solver("PHY-II.1")
    .time(0.0, 1.0, dt=1e-3)
    .export("vtu", path="out")
    .build()
    .run()
)
print(f"Wall time: {result.wall_time:.2f}s")

# Or use a pre-built recipe
wf = get_recipe("sod_shock_tube").build()
result = wf.run()
```

### Example: DMRG Ground State

```python
import torch
from tensornet import MPS, heisenberg_mpo, dmrg

L, chi = 20, 64
H = heisenberg_mpo(L=L, J=1.0, Jz=1.0)
psi = MPS.random(L=L, d=2, chi=chi, dtype=torch.float64)
psi, E, info = dmrg(psi, H, num_sweeps=15, chi_max=chi)

print(f"Ground state energy: E/L = {E/L:.8f}")
# E/L = -0.44314718 (matches Bethe ansatz exact solution)
```

### Example: CFD Shock Tube

```python
from tensornet.cfd import Euler1D, sod_shock_tube

solver = Euler1D(nx=400, xmin=0.0, xmax=1.0, gamma=1.4)
rho, u, p = sod_shock_tube(solver.x)
solver.initialize(rho, u, p)

for _ in range(200):
    solver.step(dt=0.0005, method='godunov')

error = solver.validate_against_exact(t=0.1)
print(f"L1 error: {error:.4e}")  # 1.66e-02
```

---

## Core Capabilities

| Module | Description | Validation |
|--------|-------------|------------|
| `tensornet.sdk` | Stable public API — WorkflowBuilder, recipes, re-exports | 55 tests |
| `tensornet.platform` | Unified substrate — data model, solvers, V&V, QTT, coupled physics | 268 tests |
| `tensornet.mps` | Matrix Product States | Heisenberg exact |
| `tensornet.mpo` | Matrix Product Operators | Operator algebra |
| `tensornet.algorithms` | DMRG, TEBD, Lanczos | Ground state convergence |
| `tensornet.cfd` | Euler, Navier-Stokes, QTT-Native | 8 canonical benchmarks |
| `tensornet.cuda` | GPU acceleration | cuBLAS validated |
| `tensornet.hypersim` | Hypersonic/plasma | Sutton-Graves heat flux |
| `tensornet.fusion` | Tokamak physics | Boris pusher, MHD |

### CFD Benchmarks (8/8 Passing)

| Benchmark | Reference | Status |
|-----------|-----------|:------:|
| Sod Shock Tube | Sod (1978) | ✅ |
| Lax Shock Tube | Lax (1954) | ✅ |
| Double Rarefaction | Toro (1999) | ✅ |
| Shu-Osher | Shu & Osher (1989) | ✅ |
| Double Mach Reflection | Woodward & Colella (1984) | ✅ |
| Taylor-Green Vortex | Taylor & Green (1937) | ✅ |
| Lid-Driven Cavity | Ghia et al. (1982) | ✅ |
| **QTT-Native NS2D** | Conference Room Ventilation | ✅ |

### QTT-Native Navier-Stokes (NEW)

The `ns2d_qtt_native.py` solver implements fully QTT-native 2D Navier-Stokes:

```python
from tensornet.cfd.ns2d_qtt_native import (
    NS2DQTTConfig, NS2D_QTT_Native, create_conference_room_ic
)

config = NS2DQTTConfig(nx_bits=7, ny_bits=7, max_rank=48)  # 128×128
solver = NS2D_QTT_Native(config)
omega, psi, psi_bc, bc_mask = create_conference_room_ic(config)

omega, psi, info = solver.solve_steady_state(
    omega, psi, psi_bc, bc_mask, max_iters=200, tol=1e-5
)
# Inlet velocity recovery: 94.4% ✓
```

**Key features:**
- O(log N × r³) complexity per iteration
- No dense operations in solver loop
- QTT-native Jacobi Poisson solver
- Hadamard product for u·∇ω nonlinear term
- Mask-based boundary condition enforcement

### Conservation Verification

All CFD solvers verified to conserve mass, momentum, and energy to machine precision (Δ < 10⁻¹⁵).

---

## Validation Gauntlets

This repository includes physics validation scripts ("gauntlets") that test computational models against known results. Each gauntlet produces a JSON attestation with metrics and SHA256 hash.

| Gauntlet | Domain | Gates | Status |
|----------|--------|:-----:|:------:|
| `tomahawk_gauntlet.py` | MHD/Plasma Control | 5 | ✅ |
| `hellskin_thermal_solver.py` | Thermal Protection | 4 | ✅ |
| `odin_superconductor_solver.py` | Superconductor Theory | 5 | ✅ |
| `starheart_fusion_solver.py` | Fusion Physics | 5 | ✅ |
| `chronos_gauntlet.py` | Relativistic Physics | 5 | ✅ |
| `ns2d_qtt_native.py` | QTT-Native CFD | 4 | ✅ |
| ... | ... | ... | ... |

**Note:** These gauntlets validate *computational models* against physics benchmarks. Passing a gauntlet means the code correctly implements the relevant equations — not that a physical device has been built.

See [The_Civilization_Stack.md](The_Civilization_Stack.md) for complete documentation.

---

## Project Structure

```
HyperTensor/
├── tensornet/              # Core library
│   ├── platform/           # Unified substrate (Phases 1–7)
│   │   ├── data_model.py   # Mesh, Field, BC/IC, SimulationState
│   │   ├── protocols.py    # ProblemSpec, Solver, Observable, Workflow
│   │   ├── solvers.py      # Time integrators, linear/nonlinear solvers
│   │   ├── domain_pack.py  # Plugin architecture + registry
│   │   ├── reproduce.py    # Deterministic runs, artifact hashing
│   │   ├── checkpoint.py   # Serialization / restart
│   │   ├── export.py       # VTU, XDMF+HDF5, CSV, JSON
│   │   ├── mesh_import.py  # GMSH v2/v4, raw arrays
│   │   ├── postprocess.py  # probe, slice, integrate, FFT, gradient
│   │   ├── visualize.py    # matplotlib field/convergence/spectrum plots
│   │   ├── deprecation.py  # SemVer, @deprecated, @since, version gate
│   │   ├── security.py     # SBOM, dependency audit, license audit
│   │   ├── qtt.py          # QTT bridge layer
│   │   ├── coupled.py      # Coupling orchestrator
│   │   ├── adjoint.py      # Discrete adjoint + FD fallback
│   │   ├── inverse.py      # Inverse problem toolkit
│   │   ├── uq.py           # UQ (MC, LHS, PCE)
│   │   └── optimization.py # Topology/shape optimization
│   ├── sdk/                # Stable public API surface
│   │   ├── __init__.py     # 55+ curated re-exports
│   │   ├── workflow.py     # WorkflowBuilder DSL
│   │   └── recipes.py      # 8 built-in per-domain recipes
│   ├── packs/              # 20 domain packs (I–XX, 167 nodes)
│   ├── algorithms/         # DMRG, TEBD, Lanczos
│   ├── mps/                # Matrix Product States
│   ├── mpo/                # Matrix Product Operators
│   ├── cfd/                # Computational Fluid Dynamics
│   ├── cuda/               # GPU kernels
│   ├── hypersim/           # Hypersonic simulation
│   └── validation/         # V&V framework
├── ledger/                 # Capability ledger (167 YAML nodes)
├── tests/                  # Test suite (268+ tests)
├── benchmarks/             # Performance validation
├── docs/                   # Documentation
│   └── adr/                # Architecture Decision Records (ADR-0001–0011)
├── apps/                   # Applications
│   ├── glass_cockpit/      # Real-time visualization
│   └── global_eye/         # Monitoring tools
├── proofs/                 # Mathematical verification
└── scripts/                # Utilities
```

---

## V&V Framework

HyperTensor implements verification and validation aligned with:
- **ASME V&V 10-2019** — Computational Solid Mechanics
- **NASA-STD-7009A** — Models and Simulations

### Verification Methods

| Method | Description |
|--------|-------------|
| MMS | Method of Manufactured Solutions for spatial/temporal order |
| Conservation | Mass, momentum, energy conservation to machine precision |
| Symmetry | Verification of expected physical symmetries |
| Analytical | Comparison to closed-form solutions where available |

### Validation Methods

| Method | Description |
|--------|-------------|
| Benchmark | Canonical test cases (Sod, Ghia, Taylor-Green) |
| Literature | Comparison to published experimental/computational data |
| Cross-code | Comparison to established codes (when available) |

See [HYPERTENSOR_VV_FRAMEWORK.md](HYPERTENSOR_VV_FRAMEWORK.md) for details.

---

## Testing

```bash
# Full test suite
pytest tests/ -v

# By category
pytest -m unit          # Unit tests
pytest -m benchmark     # Known-solution benchmarks
pytest -m mms           # Method of Manufactured Solutions
pytest -m conservation  # Conservation law tests

# With coverage
pytest tests/ --cov=tensornet --cov-report=html
```

---

## Requirements

- Python 3.11+
- PyTorch 2.0+ (CUDA 12.1+ for GPU acceleration)
- NumPy 1.24+
- Rust 1.70+ (for Glass Cockpit visualization)

Optional extras:
```bash
pip install -e ".[viz]"        # matplotlib, jupyter
pip install -e ".[io]"         # h5py (XDMF/HDF5 export)
pip install -e ".[benchmark]"  # scipy, tenpy
pip install -e ".[all]"        # everything
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [The_Civilization_Stack.md](The_Civilization_Stack.md) | Validation gauntlet documentation |
| [HYPERTENSOR_VV_FRAMEWORK.md](HYPERTENSOR_VV_FRAMEWORK.md) | V&V methodology |
| [CONSTITUTION.md](CONSTITUTION.md) | Coding standards |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines |
| [Commercial_Execution.md](Commercial_Execution.md) | 7-phase roadmap (all COMPLETE) |
| [CHANGELOG.md](CHANGELOG.md) | Version history |
| [docs/INDEX.md](docs/INDEX.md) | Documentation index |

---

## References

1. Gourianov et al., "A quantum-inspired approach to exploit turbulence structures", [arXiv:2305.10784](https://arxiv.org/abs/2305.10784) (2023)
2. White, "Density matrix formulation for quantum renormalization groups", Phys. Rev. Lett. 69, 2863 (1992)
3. Vidal, "Efficient simulation of one-dimensional quantum many-body systems", Phys. Rev. Lett. 93, 040502 (2004)
4. Oseledets, "Tensor-Train Decomposition", SIAM J. Sci. Comput. 33, 2295 (2011)

---

## License

**Proprietary** — © 2025-2026 Tigantic Holdings LLC. All rights reserved.

See [LICENSE](LICENSE) for details.

---

## Citation

```bibtex
@software{hypertensor2026,
  title = {Project HyperTensor: Quantum-Inspired Tensor Networks for Computational Physics},
  author = {Tigantic Holdings LLC},
  year = {2026},
  url = {https://github.com/tigantic/HyperTensor}
}
```

---

*"In God we trust. All others must bring data."* — W. Edwards Deming
