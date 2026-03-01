---
hide:
  - navigation
---

# The Physics OS

**The Physics-First Tensor Network Engine**

Quantum-inspired tensor network computation for computational physics, CFD,
quantum mechanics, and 168 physics taxonomy nodes — all in pure PyTorch with
a Rust performance substrate.

## Quick Install

```bash
# Core
pip install ontic-engine

# With CFD domain pack
pip install ontic-engine[cfd]

# With all domain packs + dev tools
pip install ontic-engine[all]
```

## Quick Start

```python
from ontic import MPS, heisenberg_mpo, dmrg

H = heisenberg_mpo(L=10, J=1.0)
psi = MPS.random(L=10, d=2, chi=32)
psi, E, info = dmrg(psi, H, num_sweeps=10)
print(f"Ground state energy: {E:.8f}")
```

## Architecture

The Physics OS is a **monorepo** with three tiers:

| Tier | Package | Language | Purpose |
|------|---------|----------|---------|
| 1 | `ontic` | Python | Physics engine — MPS, MPO, QTT, DMRG, TEBD, CFD |
| 2 | `ontic` | Python | Licensed execution fabric — API, SDK, MCP, billing |
| 3 | `crates/*` | Rust | Performance substrate — GPU kernels, ZK proofs, IPC |

## Domain Packs

Install only the physics you need:

```bash
pip install ontic-engine[cfd]        # Euler, Navier-Stokes, LES
pip install ontic-engine[quantum]    # QM, QFT, condensed matter
pip install ontic-engine[fluids]     # Multiphase, free-surface, FSI
pip install ontic-engine[aerospace]  # Flight dynamics, guidance
pip install ontic-engine[physics-all]  # Everything
```
