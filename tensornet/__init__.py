"""
TensorNet: Quantum-Inspired Tensor Networks for Computational Physics
======================================================================

A pure PyTorch implementation of Matrix Product States (MPS) and
Matrix Product Operators (MPO) for many-body physics and CFD.

Core Components:
    MPS     - Matrix Product State representation
    MPO     - Matrix Product Operator (Hamiltonians)
    dmrg    - Density Matrix Renormalization Group
    tebd    - Time-Evolving Block Decimation
    tdvp    - Time-Dependent Variational Principle

Hamiltonians:
    heisenberg_mpo  - Heisenberg XXZ chain
    tfim_mpo        - Transverse-field Ising model
    bose_hubbard_mpo - Bose-Hubbard model

Example:
    >>> from tensornet import MPS, heisenberg_mpo, dmrg
    >>> H = heisenberg_mpo(L=10, J=1.0)
    >>> psi = MPS.random(L=10, d=2, chi=32)
    >>> psi, E, info = dmrg(psi, H, num_sweeps=10)
    >>> print(f"Ground state energy: {E:.8f}")
"""

__version__ = "0.1.0"
__author__ = "TiganticLabz"

# Core classes
from tensornet.core.mps import MPS
from tensornet.core.mpo import MPO

# Algorithms
from tensornet.algorithms.dmrg import dmrg
from tensornet.algorithms.tebd import tebd, tebd_step
from tensornet.algorithms.lanczos import lanczos_ground_state
from tensornet.algorithms.tdvp import tdvp, tdvp_step, imaginary_time_tdvp, TDVPResult

# Hamiltonians
from tensornet.mps.hamiltonians import (
    heisenberg_mpo,
    tfim_mpo,
    xx_mpo,
    xyz_mpo,
    bose_hubbard_mpo,
    pauli_matrices,
    spin_operators,
)

# Utility functions
from tensornet.core.decompositions import svd_truncated, qr_positive
from tensornet.core.states import ghz_mps, product_mps, random_mps

# CFD Module (Phase 2)
from tensornet.cfd.euler_1d import (
    Euler1D,
    EulerState,
    euler_to_mps,
    mps_to_euler,
    sod_shock_tube_ic,
    lax_shock_tube_ic,
    shu_osher_ic,
)
from tensornet.cfd.godunov import roe_flux, hll_flux, hllc_flux, exact_riemann
from tensornet.cfd.limiters import minmod, superbee, van_leer, mc_limiter

# CFD Module (Phase 3 - 2D)
from tensornet.cfd.euler_2d import (
    Euler2D,
    Euler2DState,
    supersonic_wedge_ic,
    oblique_shock_exact,
)
from tensornet.cfd.geometry import WedgeGeometry, ImmersedBoundary
from tensornet.cfd.boundaries import BCType, FlowState, BoundaryManager
from tensornet.cfd.qtt import (
    field_to_qtt,
    qtt_to_field,
    euler_to_qtt,
    qtt_to_euler,
    QTTCompressionResult,
)

__all__ = [
    # Core
    "MPS",
    "MPO",
    # Algorithms
    "dmrg",
    "tebd",
    "tebd_step",
    "lanczos_ground_state",
    "tdvp",
    "tdvp_step",
    "imaginary_time_tdvp",
    "TDVPResult",
    # Hamiltonians
    "heisenberg_mpo",
    "tfim_mpo",
    "xx_mpo",
    "xyz_mpo",
    "bose_hubbard_mpo",
    "pauli_matrices",
    "spin_operators",
    # Utilities
    "svd_truncated",
    "qr_positive",
    "ghz_mps",
    "product_mps",
    "random_mps",
    # CFD 1D
    "Euler1D",
    "EulerState",
    "euler_to_mps",
    "mps_to_euler",
    "sod_shock_tube_ic",
    "lax_shock_tube_ic",
    "shu_osher_ic",
    "roe_flux",
    "hll_flux",
    "hllc_flux",
    "exact_riemann",
    "minmod",
    "superbee",
    "van_leer",
    "mc_limiter",
    # CFD 2D
    "Euler2D",
    "Euler2DState",
    "supersonic_wedge_ic",
    "oblique_shock_exact",
    "WedgeGeometry",
    "ImmersedBoundary",
    "BCType",
    "FlowState",
    "BoundaryManager",
    # QTT Compression (Phase 5)
    "field_to_qtt",
    "qtt_to_field",
    "euler_to_qtt",
    "qtt_to_euler",
    "QTTCompressionResult",
]
