# Vision

## One-line
A **physics-first tensor network operating system** powered by **QTT** and built on **PyTenNet** foundations.

## What QTT unlocks
- Operating on *effective* grids of size 2^k per dimension (k=10..40+) without dense materialization.
- Logarithmic scaling of storage (under rank control assumptions).
- A unification surface for PDE solvers, operators, ML features, and compression-as-a-service.

## Design doctrine
- **Never Go Dense** (dense materialization is a guarded escape hatch).
- **Rank control is the product** (everything else is a consequence).
- **Operator-first** (MPOs as first-class, not ad-hoc functions).
