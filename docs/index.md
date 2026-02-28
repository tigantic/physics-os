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
pip install tensornet

# With CFD domain pack
pip install tensornet[cfd]

# With all domain packs + dev tools
pip install tensornet[all]
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
| 1 | `tensornet` | Python | Physics engine — MPS, MPO, QTT, DMRG, TEBD, CFD |
| 2 | `ontic` | Python | Licensed execution fabric — API, SDK, MCP, billing |
| 3 | `crates/*` | Rust | Performance substrate — GPU kernels, ZK proofs, IPC |

## Domain Packs

Install only the physics you need:

```bash
pip install tensornet[cfd]        # Euler, Navier-Stokes, LES
pip install tensornet[quantum]    # QM, QFT, condensed matter
pip install tensornet[fluids]     # Multiphase, free-surface, FSI
pip install tensornet[aerospace]  # Flight dynamics, guidance
pip install tensornet[physics-all]  # Everything
```
