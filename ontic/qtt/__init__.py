"""
ontic.qtt — QTT-specific algorithms
=========================================

Modules in this sub-package operate directly on Tensor-Train (TT) cores
without materialising full dense arrays, achieving :math:`O(n r^3)` or
:math:`O(n r^2 d)` complexity where *n* is the number of TT-cores,
*r* the bond dimension, and *d* the local mode size.

Sub-modules
-----------
* ``sparse_direct``   — LU / Cholesky in TT format
* ``rank_adaptive``   — Information-theoretic rank selection (AIC/BIC/MDL)
* ``unstructured``    — QTT on FEM / FVM meshes via RCM + quantics
* ``eigensolvers``    — Lanczos / Davidson in TT format
* ``krylov``          — CG / GMRES entirely in TT
* ``dynamic_rank``    — Rank adaptation during time integration
* ``differentiable``  — Autograd-compatible TT operations
* ``pde_solvers``     — Implicit time-steppers (backward Euler, CN, BDF-2)
* ``qtci_v2``         — Enhanced TCI with rook pivoting & error certification
* ``time_series``     — Temporal signal compression via quantics mapping
"""
