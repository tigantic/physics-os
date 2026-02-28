"""QTT Topology Optimization for 1D EM Design.

Phase 6 of the QTT Frequency-Domain Maxwell program.

Implements density-based topology optimization of the permittivity
distribution to achieve target S-parameter specifications.

Architecture
------------
The optimization solves:

.. math::

    \\min_{\\rho} \\; \\mathcal{J}(\\rho) + \\alpha\\,R(\\rho)

    \\text{subject to:}\\quad H(\\rho)\\,E = -J, \\quad 0 \\le \\rho \\le 1

where:
  - ρ(x) ∈ [0,1] is the density field (design variable)
  - ε(x) = ε_min + ρ(x)·(ε_max − ε_min) interpolates permittivity
  - J is the objective functional (target S-parameter)
  - R is a regulariser (total variation, Tikhonov)
  - α is the regularisation weight

Method
------
Uses the **adjoint method** for gradient computation:

1. **Forward solve**: H(ρ)·E = -J
2. **Objective evaluation**: J = f(S₁₁(E), target)
3. **Adjoint solve**: H(ρ)ᵀ·λ = -∂J/∂E
4. **Design sensitivity**: dJ/dρ = Re[λᵀ · ∂H/∂ρ · E]

The Helmholtz operator H = L_s + k²·diag(ε(ρ)) has the simple
derivative ∂H/∂ρᵢ = k²·(ε_max − ε_min)·δᵢⱼ (diagonal), making
the adjoint gradient computation very efficient.

Projection & Filtering
----------------------
To push designs toward binary (0/1), a Heaviside projection is
applied:

.. math::

    \\tilde{\\rho} = \\frac{\\tanh(\\beta\\eta) + \\tanh(\\beta(\\rho - \\eta))}
                         {\\tanh(\\beta\\eta) + \\tanh(\\beta(1 - \\eta))}

with β increasing over iterations (continuation) and η = 0.5.

Dependencies
------------
- ``ontic.em.boundaries``: ``Geometry1D``, ``helmholtz_mpo_with_bc``
- ``ontic.em.s_parameters``: ``Port``, ``compute_s11``,
  ``port_source_tt``, ``s_to_db``
- ``ontic.em.qtt_helmholtz``: ``tt_amen_solve``, ``reconstruct_1d``,
  ``array_to_tt``, ``diag_mpo_from_tt``, ``mpo_add_c``, ``mpo_scale_c``
"""

from __future__ import annotations

import math
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from ontic.em.qtt_helmholtz import (
    array_to_tt,
    reconstruct_1d,
    tt_amen_solve,
    diag_mpo_from_tt,
    mpo_add_c,
    mpo_scale_c,
)
from ontic.em.boundaries import (
    Geometry1D,
    PMLConfig,
    MaterialRegion,
    helmholtz_mpo_with_bc,
    stretched_laplacian_mpo_1d,
    build_pml_profile_1d,
)
from ontic.em.s_parameters import (
    Port,
    port_source_tt,
    compute_s11,
    extract_mode_coefficients_lsq,
    s_to_db,
)


# =====================================================================
# Section 1: Design Parameterisation
# =====================================================================

@dataclass
class DesignRegion:
    """Defines the spatial region where material is optimised.

    Parameters
    ----------
    x_start : float
        Start of design region in normalised coordinates [0, 1].
    x_end : float
        End of design region.
    eps_min : complex
        Minimum permittivity (ρ = 0 → vacuum / air).
    eps_max : complex
        Maximum permittivity (ρ = 1 → solid dielectric).
    """

    x_start: float = 0.3
    x_end: float = 0.7
    eps_min: complex = 1.0 + 0j
    eps_max: complex = 4.0 + 0j

    @property
    def eps_contrast(self) -> complex:
        """Permittivity contrast Δε = ε_max − ε_min."""
        return self.eps_max - self.eps_min

    def contains(self, x: NDArray) -> NDArray:
        """Boolean mask: True where x is inside design region."""
        return (x >= self.x_start) & (x < self.x_end)

    def n_design_cells(self, N: int) -> int:
        """Number of grid cells in the design region."""
        h = 1.0 / N
        x = np.linspace(h / 2, 1.0 - h / 2, N)
        return int(np.sum(self.contains(x)))


@dataclass
class OptimizationConfig:
    """Configuration for topology optimization.

    Parameters
    ----------
    max_iterations : int
        Maximum number of optimization iterations.
    learning_rate : float
        Gradient descent step size.
    beta_init : float
        Initial Heaviside projection sharpness.
    beta_max : float
        Maximum projection sharpness.
    beta_increase_every : int
        Increase β every this many iterations.
    beta_factor : float
        Multiplicative factor for β increase.
    eta : float
        Heaviside projection threshold (typically 0.5).
    regularisation_weight : float
        Weight α for the regulariser term.
    regularisation_type : str
        "tv" (total variation) or "tikhonov" (L² gradient penalty).
    volume_fraction : float
        Target volume fraction constraint (0 = no constraint).
        If > 0, penalises deviation from target fill.
    volume_penalty : float
        Penalty weight for volume fraction constraint.
    convergence_tol : float
        Stop when relative objective change < this.
    min_feature_size : int
        Minimum feature size in grid cells (density filter radius).
    """

    max_iterations: int = 100
    learning_rate: float = 0.3
    beta_init: float = 1.0
    beta_max: float = 32.0
    beta_increase_every: int = 20
    beta_factor: float = 2.0
    eta: float = 0.5
    regularisation_weight: float = 0.01
    regularisation_type: str = "tv"
    volume_fraction: float = 0.0
    volume_penalty: float = 0.0
    convergence_tol: float = 1e-4
    min_feature_size: int = 0


# =====================================================================
# Section 2: Heaviside Projection & Density Filter
# =====================================================================

def heaviside_projection(
    rho: NDArray,
    beta: float,
    eta: float = 0.5,
) -> NDArray:
    """Smooth Heaviside projection toward binary design.

    Maps continuous ρ ∈ [0,1] toward {0,1} with controllable
    sharpness β.  At β → ∞, this becomes a step function at η.

    .. math::

        \\tilde{\\rho} = \\frac{\\tanh(\\beta\\eta) + \\tanh(\\beta(\\rho - \\eta))}
                             {\\tanh(\\beta\\eta) + \\tanh(\\beta(1 - \\eta))}

    Parameters
    ----------
    rho : NDArray
        Raw density field ∈ [0, 1].
    beta : float
        Projection sharpness (≥ 0). Higher = more binary.
    eta : float
        Threshold (default 0.5).

    Returns
    -------
    NDArray
        Projected density field ∈ [0, 1].
    """
    if beta < 1e-10:
        return rho.copy()

    # Clip for numerical safety
    beta = min(beta, 500.0)

    num = np.tanh(beta * eta) + np.tanh(beta * (rho - eta))
    den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return np.clip(num / den, 0.0, 1.0)


def heaviside_gradient(
    rho: NDArray,
    beta: float,
    eta: float = 0.5,
) -> NDArray:
    """Derivative of Heaviside projection: d(proj)/d(rho).

    Parameters
    ----------
    rho : NDArray
        Raw density.
    beta : float
        Projection sharpness.
    eta : float
        Threshold.

    Returns
    -------
    NDArray
        d(projected_rho) / d(rho).
    """
    if beta < 1e-10:
        return np.ones_like(rho)

    beta = min(beta, 500.0)
    den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    sech2 = 1.0 / np.cosh(beta * (rho - eta)) ** 2
    return beta * sech2 / den


def density_filter(
    rho: NDArray,
    radius: int,
) -> NDArray:
    """Conic density filter for minimum feature size control.

    Convolves ρ with a normalised cone kernel of given radius.
    This smooths the design and enforces minimum feature sizes.

    Parameters
    ----------
    rho : NDArray
        Raw density field.
    radius : int
        Filter radius in grid cells.

    Returns
    -------
    NDArray
        Filtered density.
    """
    if radius <= 0:
        return rho.copy()

    N = len(rho)
    kernel = np.zeros(2 * radius + 1)
    for i in range(-radius, radius + 1):
        kernel[i + radius] = max(1.0 - abs(i) / radius, 0.0)
    kernel /= kernel.sum()

    # Convolve with zero-padding (boundary = extend)
    padded = np.pad(rho, radius, mode="edge")
    filtered = np.convolve(padded, kernel, mode="valid")
    return filtered[:N]


def density_filter_gradient(
    grad: NDArray,
    radius: int,
) -> NDArray:
    """Transpose of density filter (adjoint convolution).

    Parameters
    ----------
    grad : NDArray
        Gradient w.r.t. filtered density.
    radius : int
        Filter radius (same as forward filter).

    Returns
    -------
    NDArray
        Gradient w.r.t. raw density.
    """
    if radius <= 0:
        return grad.copy()

    N = len(grad)
    kernel = np.zeros(2 * radius + 1)
    for i in range(-radius, radius + 1):
        kernel[i + radius] = max(1.0 - abs(i) / radius, 0.0)
    kernel /= kernel.sum()

    # Adjoint of convolve = correlation
    padded = np.pad(grad, radius, mode="edge")
    corr = np.correlate(padded, kernel, mode="valid")
    return corr[:N]


# =====================================================================
# Section 3: Objective Functions
# =====================================================================

@dataclass
class ObjectiveSpec:
    """S-parameter objective specification.

    Parameters
    ----------
    target_s11_db : float
        Target |S₁₁| in dB (e.g. -20 for good matching).
    target_type : str
        "minimize" — minimise |S₁₁| (matching objective).
        "target" — drive |S₁₁| toward target_s11_db.
        "bandpass" — minimise |S₁₁| in passband, keep high outside.
    weight : float
        Objective weight for multi-objective optimization.
    """

    target_s11_db: float = -20.0
    target_type: str = "minimize"
    weight: float = 1.0


def objective_minimize_s11(
    s11: complex,
) -> tuple[float, complex]:
    """Objective: minimise |S₁₁|².

    J = |S₁₁|² = S₁₁ · S₁₁*

    Gradient: dJ/dS₁₁ = S₁₁* (conjugate).

    Parameters
    ----------
    s11 : complex
        Reflection coefficient.

    Returns
    -------
    tuple[float, complex]
        (objective_value, dJ/dS₁₁).
    """
    J = abs(s11) ** 2
    dJ = np.conj(s11)
    return float(J), complex(dJ)


def objective_target_s11_db(
    s11: complex,
    target_db: float,
) -> tuple[float, complex]:
    """Objective: drive |S₁₁| toward target dB level.

    J = (|S₁₁|_dB − target_dB)²
      = (20·log₁₀|S₁₁| − target_dB)²

    Parameters
    ----------
    s11 : complex
        Reflection coefficient.
    target_db : float
        Target |S₁₁| in dB.

    Returns
    -------
    tuple[float, complex]
        (objective_value, dJ/dS₁₁).
    """
    mag = abs(s11)
    if mag < 1e-30:
        return (target_db ** 2, 0.0 + 0j)

    s11_db = 20.0 * math.log10(mag)
    diff = s11_db - target_db

    J = diff ** 2

    # Chain rule: dJ/dS₁₁ = 2·diff · d(dB)/d|S₁₁| · d|S₁₁|/dS₁₁
    # d(dB)/d|S₁₁| = 20/(|S₁₁|·ln10)
    # d|S₁₁|/dS₁₁ = S₁₁*/(2|S₁₁|)  [Wirtinger derivative]
    d_db_d_mag = 20.0 / (mag * math.log(10.0))
    d_mag_d_s11 = np.conj(s11) / (2.0 * mag)
    dJ = 2.0 * diff * d_db_d_mag * d_mag_d_s11

    return float(J), complex(dJ)


# =====================================================================
# Section 4: Regularisation
# =====================================================================

def total_variation_1d(
    rho: NDArray,
    epsilon: float = 1e-6,
) -> tuple[float, NDArray]:
    """Total variation regulariser for 1D density field.

    R(ρ) = Σ √((ρ_{i+1} − ρ_i)² + ε²) ≈ Σ |ρ_{i+1} − ρ_i|

    Smoothed with ε for differentiability.

    Parameters
    ----------
    rho : NDArray
        Density field.
    epsilon : float
        Smoothing parameter.

    Returns
    -------
    tuple[float, NDArray]
        (R, dR/dρ).
    """
    diff = np.diff(rho)
    smooth_abs = np.sqrt(diff ** 2 + epsilon ** 2)
    R = float(np.sum(smooth_abs))

    # Gradient
    grad = np.zeros_like(rho)
    d_smooth = diff / smooth_abs
    grad[:-1] -= d_smooth
    grad[1:] += d_smooth

    return R, grad


def tikhonov_regulariser(
    rho: NDArray,
) -> tuple[float, NDArray]:
    """Tikhonov (L² gradient) regulariser.

    R(ρ) = ½ Σ (ρ_{i+1} − ρ_i)²

    Parameters
    ----------
    rho : NDArray
        Density field.

    Returns
    -------
    tuple[float, NDArray]
        (R, dR/dρ).
    """
    diff = np.diff(rho)
    R = 0.5 * float(np.sum(diff ** 2))

    grad = np.zeros_like(rho)
    grad[:-1] -= diff
    grad[1:] += diff

    return R, grad


# =====================================================================
# Section 5: Forward Model (Density → S₁₁)
# =====================================================================

def build_eps_from_density(
    rho: NDArray,
    design: DesignRegion,
    geometry: Geometry1D,
    beta: float = 1.0,
    eta: float = 0.5,
    filter_radius: int = 0,
) -> NDArray:
    """Build permittivity profile from density field.

    Applies density filter, Heaviside projection, and interpolates
    to the full grid including non-design regions.

    Parameters
    ----------
    rho : NDArray
        Raw density field (length = n_design_cells).
    design : DesignRegion
        Design region specification.
    geometry : Geometry1D
        Full geometry (provides grid, background ε, PML, etc.).
    beta : float
        Heaviside projection sharpness.
    eta : float
        Heaviside threshold.
    filter_radius : int
        Density filter radius.

    Returns
    -------
    NDArray
        Full ε(x) array of length N = 2^n_bits.
    """
    N = 2 ** geometry.n_bits
    h = 1.0 / N
    x = np.linspace(h / 2, 1.0 - h / 2, N)

    # Start with base geometry ε
    eps = geometry.build_eps_profile()

    # Design region mask
    mask = design.contains(x)

    # Pipeline: filter → project → interpolate
    rho_filtered = density_filter(rho, filter_radius)
    rho_projected = heaviside_projection(rho_filtered, beta, eta)

    # Interpolate ε in design region
    eps[mask] = (
        design.eps_min + rho_projected * design.eps_contrast
    )

    return eps


def _build_helmholtz_from_eps(
    eps_arr: NDArray,
    geometry: Geometry1D,
    k0: float,
    max_rank: int = 64,
    damping: float = 0.01,
) -> list[NDArray]:
    """Build Helmholtz MPO from explicit ε array.

    Uses the geometry's PML for the stretched Laplacian, but
    overrides the permittivity with the provided array.

    Parameters
    ----------
    eps_arr : NDArray
        Full permittivity array of length N.
    geometry : Geometry1D
        Geometry (used for n_bits, PML, conductors).
    k0 : float
        Wavenumber.
    max_rank : int
        Maximum QTT rank.
    damping : float
        Damping parameter.

    Returns
    -------
    list[NDArray]
        Helmholtz MPO cores.
    """
    n_bits = geometry.n_bits
    N = 2 ** n_bits
    h = 1.0 / N

    # Stretched Laplacian (PML)
    L_s = stretched_laplacian_mpo_1d(
        n_bits, k0, h, geometry.pml, max_rank=max_rank,
    )

    # Apply damping to ε
    eps_damped = eps_arr * (1.0 + 1j * damping)

    # k² · diag(ε)
    eps_tt = array_to_tt(eps_damped.astype(np.complex128),
                         max_rank=max_rank, cutoff=1e-12)
    eps_mpo = diag_mpo_from_tt(eps_tt)
    k2_eps = mpo_scale_c(eps_mpo, k0 * k0)

    H = mpo_add_c(L_s, k2_eps)

    # PEC penalty
    if geometry.has_conductors():
        P = geometry.build_penalty_mpo(penalty=1e8, max_rank=max_rank)
        P_complex = [c.astype(np.complex128) for c in P]
        H = mpo_add_c(H, P_complex)

    return H


# =====================================================================
# Section 6: Adjoint Gradient Computation
# =====================================================================

def compute_adjoint_gradient(
    rho: NDArray,
    design: DesignRegion,
    geometry: Geometry1D,
    k0: float,
    port: Port,
    objective_fn: Callable[[complex], tuple[float, complex]],
    beta: float = 1.0,
    eta: float = 0.5,
    filter_radius: int = 0,
    max_rank: int = 128,
    solver_tol: float = 1e-4,
    n_sweeps: int = 40,
    damping: float = 0.01,
    n_probes: int = 8,
) -> tuple[float, NDArray, complex, float]:
    """Compute objective value and adjoint gradient w.r.t. density.

    Full adjoint pipeline:
    1. Build ε(ρ) → Helmholtz MPO H
    2. Forward solve: H·E = -J
    3. Evaluate objective J and dJ/dS₁₁
    4. Compute dS₁₁/dE (from mode extraction)
    5. Adjoint solve: Hᵀ·λ = -(dJ/dE)ᵀ
    6. Gradient: dJ/dρᵢ = Re[λᵀ · ∂H/∂ρᵢ · E]

    For efficiency, uses the **direct differentiation** approach
    based on the dense field, avoiding a second QTT solve.

    Parameters
    ----------
    rho : NDArray
        Raw density field (length = n_design_cells).
    design : DesignRegion
        Design region specification.
    geometry : Geometry1D
        Full geometry.
    k0 : float
        Wavenumber.
    port : Port
        Port definition.
    objective_fn : callable
        Maps S₁₁ → (objective_value, dJ/dS₁₁).
    beta : float
        Heaviside projection sharpness.
    eta : float
        Heaviside threshold.
    filter_radius : int
        Density filter radius.
    max_rank : int
        DMRG max rank.
    solver_tol : float
        DMRG tolerance.
    n_sweeps : int
        DMRG sweeps.
    damping : float
        Helmholtz damping.
    n_probes : int
        Mode extraction probes.

    Returns
    -------
    tuple[float, NDArray, complex, float]
        (objective_value, dJ/d_rho, S₁₁, solver_residual).
    """
    N = 2 ** geometry.n_bits
    h = 1.0 / N
    x = np.linspace(h / 2, 1.0 - h / 2, N)
    n_bits = geometry.n_bits

    # --- Build ε from density ---
    eps_full = build_eps_from_density(
        rho, design, geometry, beta, eta, filter_radius,
    )

    # --- Forward solve ---
    H = _build_helmholtz_from_eps(
        eps_full, geometry, k0, max_rank=max_rank, damping=damping,
    )
    rhs = port_source_tt(n_bits, k0, port, max_rank=max_rank)
    result = tt_amen_solve(
        H, rhs, max_rank=max_rank, n_sweeps=n_sweeps,
        tol=solver_tol, verbose=False,
    )

    E_dense = reconstruct_1d(result.x)

    # --- Extract S₁₁ ---
    eps_local = complex(port.eps_r) * (1.0 + 1j * damping)
    k_ref = k0 * np.sqrt(eps_local)
    lam_local = 2.0 * math.pi / max(abs(k_ref.real), 1e-30)
    span = min(lam_local / 2.0, 0.1)

    x_ref = port.ref_position
    if port.direction > 0:
        x_start = max(x_ref, 0.01)
        x_end = min(x_ref + span, 0.99)
    else:
        x_start = max(x_ref - span, 0.01)
        x_end = min(x_ref, 0.99)

    A_fwd, A_bwd, _ = extract_mode_coefficients_lsq(
        E_dense, k_ref, x_start, x_end, n_probes=n_probes,
    )

    if port.direction > 0:
        incident, reflected = A_fwd, A_bwd
    else:
        incident, reflected = A_bwd, A_fwd

    s11 = reflected / incident if abs(incident) > 1e-30 else 0.0 + 0j

    # --- Evaluate objective ---
    J_val, dJ_dS11 = objective_fn(s11)

    # --- Finite-difference gradient w.r.t. design density ---
    # Using direct perturbation approach (more robust than adjoint
    # for moderate design variable counts).  For production-scale
    # problems, replace with full adjoint solve.
    design_mask = design.contains(x)
    n_design = int(np.sum(design_mask))

    # dS₁₁/dε at each design cell → dJ/dε → chain through pipeline
    # Use semi-analytical gradient: dH/dε = k²·I at each cell
    # Combined with adjoint: dJ/dε = Re[E* · k² · λ]
    # For efficiency, compute 2·Re[dJ/dS₁₁ · dS₁₁/dE · E] via
    # finite-difference of S₁₁ w.r.t. a global ε perturbation,
    # then distribute proportionally.

    # Approximate gradient via field-based sensitivity
    # dS₁₁/dε_i ≈ -k² · E_i² / (2·incident·h) (Born approximation)
    delta_eps = design.eps_contrast
    grad_design = np.zeros(n_design, dtype=np.float64)

    # Sensitivity: dJ/dρ_i = Re[dJ/dS₁₁ · dS₁₁/dρ_i]
    # dS₁₁/dρ_i ≈ -k₀² · E_i · λ_i · Δε / incident
    # where λ is the adjoint field.
    # Using Born approximation for λ ≈ E (weak scattering limit):
    E_design = E_dense[design_mask]

    # dJ/dρ via Born: each design cell contributes proportional to |E|²
    # sensitivity = k₀² · |E_i|² · |Δε| · |dJ/dS₁₁| / |incident|
    # Sign determined by dJ/dS₁₁ direction.

    # Build adjoint RHS: dJ/dE
    # dJ/dE_i = dJ/dS₁₁ · dS₁₁/dE_i
    # For S₁₁ = reflected/incident, and reflected comes from mode
    # extraction (linear in E), dS₁₁/dE_i is a sum of basis functions.

    # Pragmatic approach: use semi-analytical + FD validation
    # dJ/dρ_j = Re[ dJ/dS₁₁ * (-k₀²·Δε·h/(incident)) * E_j² ] ← Born
    # This is exact in the limit of small perturbations.

    if abs(incident) > 1e-30:
        # Mode extraction derivative w.r.t. E at design cells
        # Under Born approximation:
        #   δS₁₁ ≈ (-1/(2·incident)) · k₀² · Δε · h · Σ_j δρ_j · G(x_ref, x_j) · E_j
        # Where G is the Green's function. Approximating G·E ≈ E·E/2:
        # This simplifies to a pointwise sensitivity proportional to E².

        # Direct computation using the adjoint approach:
        # Build adjoint source from dJ/dE decomposition
        dS_dE = np.zeros(N, dtype=np.complex128)

        # dS₁₁/dE comes from mode extraction LSQ:
        # [A_fwd, A_bwd] = M^{-1} · E(x_probes)
        # S₁₁ = A_bwd/A_fwd (dir=+1)
        # dS₁₁/dA_fwd = -A_bwd/A_fwd² = -s11/A_fwd
        # dS₁₁/dA_bwd = 1/A_fwd
        # dA/dE = M^{-1} · interp_matrix

        x_probes = np.linspace(x_start, x_end, n_probes)
        M_mode = np.column_stack([
            np.exp(+1j * k_ref * x_probes),
            np.exp(-1j * k_ref * x_probes),
        ])
        M_inv = np.linalg.pinv(M_mode)

        if port.direction > 0:
            # incident=A_fwd, reflected=A_bwd
            # dS₁₁/d[A_fwd,A_bwd] = [-s11/incident, 1/incident]
            dS_dcoeff = np.array([-s11 / incident, 1.0 / incident])
        else:
            # incident=A_bwd, reflected=A_fwd
            dS_dcoeff = np.array([1.0 / incident, -s11 / incident])

        # dS₁₁/dE(x_probe) = dS₁₁/dcoeff · M_inv
        dS_dE_probes = dS_dcoeff @ M_inv  # shape (n_probes,)

        # Map probe sensitivity to full grid via interpolation adjoint
        x_grid = np.linspace(h / 2, 1.0 - h / 2, N)
        for p_idx, xp in enumerate(x_probes):
            # Find two neighboring grid points
            frac_idx = (xp - h / 2) / h
            lo = int(np.floor(frac_idx))
            hi = lo + 1
            lo = max(0, min(lo, N - 1))
            hi = max(0, min(hi, N - 1))
            w_hi = frac_idx - lo
            w_lo = 1.0 - w_hi
            dS_dE[lo] += dS_dE_probes[p_idx] * w_lo
            dS_dE[hi] += dS_dE_probes[p_idx] * w_hi

        # dJ/dE = dJ/dS₁₁ · dS₁₁/dE
        dJ_dE = dJ_dS11 * dS_dE

        # Adjoint gradient: dJ/dε_i = Re[ dJ/dE^T · dE/dε_i ]
        # From H·E = -J:  dE/dε_i = -H⁻¹ · (dH/dε_i) · E
        # dH/dε_i = k₀² · (1+j·damping) · δ_ii
        # → dJ/dε_i = -Re[ dJ/dE^T · H⁻¹ · k₀²·(1+j·damp)·E_i·e_i ]
        # Let adjoint field λ = H⁻ᵀ · dJ_dE:  Hᵀλ = dJ_dE
        # Then: dJ/dε_i = -Re[ λ_i^* · k₀²·(1+j·damp)·E_i ]

        # For the adjoint solve, H is symmetric so Hᵀ = H.
        # But complex-symmetric Hᵀ ≠ H* in general.
        # With our convention, H is complex-symmetric → Hᵀ = H.
        adj_rhs_arr = np.conj(dJ_dE)  # conjugate for the Wirtinger convention
        adj_rhs_tt = array_to_tt(
            adj_rhs_arr, max_rank=max_rank, cutoff=1e-12,
        )

        adj_result = tt_amen_solve(
            H, adj_rhs_tt,
            max_rank=max_rank,
            n_sweeps=n_sweeps,
            tol=solver_tol,
            verbose=False,
        )
        lambda_dense = reconstruct_1d(adj_result.x)

        # Design sensitivity
        k2_damp = k0 * k0 * (1.0 + 1j * damping)
        sensitivity = -k2_damp * lambda_dense * E_dense

        # dJ/dε at design cells
        dJ_deps = np.real(sensitivity[design_mask])

        # Chain rule: dJ/dρ_projected = dJ/dε · Δε (design contrast)
        dJ_d_rho_proj = dJ_deps * delta_eps.real

        # Chain through Heaviside
        rho_filtered = density_filter(rho, filter_radius)
        h_grad = heaviside_gradient(rho_filtered, beta, eta)
        dJ_d_rho_filt = dJ_d_rho_proj * h_grad

        # Chain through density filter
        grad_design = density_filter_gradient(dJ_d_rho_filt, filter_radius)
    else:
        grad_design = np.zeros(n_design, dtype=np.float64)

    return J_val, grad_design, s11, result.final_residual


# =====================================================================
# Section 7: Optimization Loop
# =====================================================================

@dataclass
class OptimizationResult:
    """Result of topology optimization.

    Attributes
    ----------
    rho_final : NDArray
        Optimised density field.
    rho_projected : NDArray
        Final projected (near-binary) density.
    eps_final : NDArray
        Final permittivity profile.
    s11_final : complex
        Final S₁₁ at operating frequency.
    objective_history : list[float]
        Objective values per iteration.
    s11_history : list[complex]
        S₁₁ values per iteration.
    gradient_norm_history : list[float]
        Gradient norms per iteration.
    beta_history : list[float]
        Heaviside β values per iteration.
    n_iterations : int
        Total iterations performed.
    converged : bool
        True if convergence criterion was met.
    total_time_s : float
        Wall-clock time.
    design : DesignRegion
        Design region used.
    config : OptimizationConfig
        Configuration used.
    """

    rho_final: NDArray
    rho_projected: NDArray
    eps_final: NDArray
    s11_final: complex
    objective_history: list[float]
    s11_history: list[complex]
    gradient_norm_history: list[float]
    beta_history: list[float]
    n_iterations: int
    converged: bool
    total_time_s: float
    design: DesignRegion
    config: OptimizationConfig


def optimize_topology(
    geometry: Geometry1D,
    design: DesignRegion,
    k0: float,
    port: Port,
    config: OptimizationConfig = OptimizationConfig(),
    objective_fn: Optional[Callable[[complex], tuple[float, complex]]] = None,
    initial_rho: Optional[NDArray] = None,
    max_rank: int = 128,
    solver_tol: float = 1e-4,
    n_sweeps: int = 40,
    damping: float = 0.01,
    n_probes: int = 8,
    verbose: bool = True,
    callback: Optional[Callable[[int, float, complex, NDArray], None]] = None,
) -> OptimizationResult:
    """Run density-based topology optimization.

    Optimises permittivity distribution within the design region
    to minimise (or target) the S₁₁ at the given port and frequency.

    Uses projected gradient descent with Heaviside continuation.

    Parameters
    ----------
    geometry : Geometry1D
        Base geometry (PML, conductors, non-design regions).
    design : DesignRegion
        Design region where ε can vary.
    k0 : float
        Operating wavenumber.
    port : Port
        Port for S₁₁ evaluation.
    config : OptimizationConfig
        Optimization hyperparameters.
    objective_fn : callable, optional
        Maps S₁₁ → (obj_value, gradient).  Default: minimise |S₁₁|².
    initial_rho : NDArray, optional
        Initial density. Default: uniform 0.5.
    max_rank : int
        DMRG max rank.
    solver_tol : float
        DMRG tolerance.
    n_sweeps : int
        DMRG sweeps.
    damping : float
        Helmholtz damping.
    n_probes : int
        Mode extraction probes.
    verbose : bool
        Print progress.
    callback : callable, optional
        Called as ``callback(iter, obj, s11, rho)`` each iteration.

    Returns
    -------
    OptimizationResult
        Optimization results.
    """
    N = 2 ** geometry.n_bits
    h = 1.0 / N
    x = np.linspace(h / 2, 1.0 - h / 2, N)

    # Number of design cells
    n_design = design.n_design_cells(N)
    if n_design == 0:
        raise ValueError("Design region contains no grid cells")

    # Default objective
    if objective_fn is None:
        objective_fn = objective_minimize_s11

    # Initialise density
    if initial_rho is not None:
        if len(initial_rho) != n_design:
            raise ValueError(
                f"initial_rho length {len(initial_rho)} != n_design {n_design}"
            )
        rho = initial_rho.copy()
    else:
        rho = np.full(n_design, 0.5, dtype=np.float64)

    # Histories
    obj_history: list[float] = []
    s11_history: list[complex] = []
    grad_norm_history: list[float] = []
    beta_history: list[float] = []

    beta = config.beta_init
    converged = False
    t_start = time.perf_counter()

    if verbose:
        print(f"Topology optimization: {n_design} design cells, "
              f"k₀={k0:.4f}, max_iter={config.max_iterations}")
        print(f"  Design: ε ∈ [{design.eps_min}, {design.eps_max}], "
              f"x ∈ [{design.x_start}, {design.x_end}]")
        print(f"  β: {config.beta_init} → {config.beta_max}, "
              f"lr={config.learning_rate}, "
              f"reg={config.regularisation_weight}")

    for iteration in range(config.max_iterations):
        # --- β continuation ---
        if (iteration > 0 and
                config.beta_increase_every > 0 and
                iteration % config.beta_increase_every == 0):
            beta = min(beta * config.beta_factor, config.beta_max)
            if verbose:
                print(f"  [β → {beta:.1f}]")

        # --- Compute gradient ---
        J_val, grad, s11, residual = compute_adjoint_gradient(
            rho=rho,
            design=design,
            geometry=geometry,
            k0=k0,
            port=port,
            objective_fn=objective_fn,
            beta=beta,
            eta=config.eta,
            filter_radius=config.min_feature_size,
            max_rank=max_rank,
            solver_tol=solver_tol,
            n_sweeps=n_sweeps,
            damping=damping,
            n_probes=n_probes,
        )

        # --- Regularisation ---
        if config.regularisation_weight > 0:
            rho_projected = heaviside_projection(
                density_filter(rho, config.min_feature_size),
                beta, config.eta,
            )
            if config.regularisation_type == "tv":
                R_val, R_grad = total_variation_1d(rho_projected)
            else:
                R_val, R_grad = tikhonov_regulariser(rho_projected)

            J_val += config.regularisation_weight * R_val

            # Chain through projection and filter for reg gradient
            h_grad = heaviside_gradient(
                density_filter(rho, config.min_feature_size),
                beta, config.eta,
            )
            reg_grad_proj = config.regularisation_weight * R_grad * h_grad
            reg_grad = density_filter_gradient(
                reg_grad_proj, config.min_feature_size,
            )
            grad = grad + reg_grad

        # --- Volume constraint ---
        if config.volume_fraction > 0 and config.volume_penalty > 0:
            rho_proj = heaviside_projection(
                density_filter(rho, config.min_feature_size),
                beta, config.eta,
            )
            vf = np.mean(rho_proj)
            vf_diff = vf - config.volume_fraction
            J_val += config.volume_penalty * vf_diff ** 2
            vol_grad = (2.0 * config.volume_penalty * vf_diff / n_design *
                        np.ones(n_design))
            grad = grad + vol_grad

        grad_norm = float(np.linalg.norm(grad))

        # Record
        obj_history.append(J_val)
        s11_history.append(s11)
        grad_norm_history.append(grad_norm)
        beta_history.append(beta)

        if verbose:
            s11_db = s_to_db(s11) if abs(s11) > 1e-30 else -300.0
            print(f"  [{iteration + 1}/{config.max_iterations}] "
                  f"J={J_val:.6f}, |S₁₁|={abs(s11):.4f} ({s11_db:.1f} dB), "
                  f"|∇|={grad_norm:.2e}, β={beta:.1f}, "
                  f"res={residual:.2e}")

        if callback is not None:
            callback(iteration, J_val, s11, rho.copy())

        # --- Convergence check ---
        if iteration > 0:
            rel_change = abs(obj_history[-1] - obj_history[-2]) / (
                abs(obj_history[-2]) + 1e-30
            )
            if rel_change < config.convergence_tol and grad_norm < 1e-6:
                converged = True
                if verbose:
                    print(f"  Converged: Δobj/obj = {rel_change:.2e}")
                break

        # --- Gradient descent update ---
        if grad_norm > 1e-30:
            rho = rho - config.learning_rate * grad / grad_norm
        # Project to [0, 1]
        rho = np.clip(rho, 0.0, 1.0)

    # --- Final design ---
    total_time = time.perf_counter() - t_start

    rho_filtered_final = density_filter(rho, config.min_feature_size)
    rho_projected_final = heaviside_projection(
        rho_filtered_final, beta, config.eta,
    )
    eps_final = build_eps_from_density(
        rho, design, geometry, beta, config.eta, config.min_feature_size,
    )

    if verbose:
        print(f"Optimization complete: {len(obj_history)} iterations, "
              f"{total_time:.1f}s")
        s11_final_db = s_to_db(s11_history[-1]) if s11_history else float("inf")
        print(f"  Final |S₁₁| = {abs(s11_history[-1]):.4f} "
              f"({s11_final_db:.1f} dB)")

    return OptimizationResult(
        rho_final=rho,
        rho_projected=rho_projected_final,
        eps_final=eps_final,
        s11_final=s11_history[-1] if s11_history else 0.0 + 0j,
        objective_history=obj_history,
        s11_history=s11_history,
        gradient_norm_history=grad_norm_history,
        beta_history=beta_history,
        n_iterations=len(obj_history),
        converged=converged,
        total_time_s=total_time,
        design=design,
        config=config,
    )


# =====================================================================
# Section 8: Convenience — Anti-Reflection Coating Design
# =====================================================================

def design_antireflection_coating(
    n_bits: int,
    k0: float,
    eps_substrate: complex,
    coating_start: float = 0.35,
    coating_end: float = 0.5,
    port_position: float = 0.2,
    ref_position: float = 0.25,
    max_iterations: int = 30,
    max_rank: int = 128,
    verbose: bool = True,
) -> OptimizationResult:
    """Design an anti-reflection coating via topology optimization.

    Optimises the permittivity in a coating layer between air and
    a dielectric substrate to minimise reflection at the given
    frequency.

    The ideal quarter-wave AR coating has ε_coat = √(ε_substrate)
    and thickness = λ/(4·n_coat).

    Parameters
    ----------
    n_bits : int
        QTT resolution.
    k0 : float
        Operating wavenumber.
    eps_substrate : complex
        Substrate permittivity.
    coating_start : float
        Start of coating region.
    coating_end : float
        End of coating region.
    port_position : float
        Port source location.
    ref_position : float
        Port reference plane.
    max_iterations : int
        Maximum optimization iterations.
    max_rank : int
        DMRG max rank.
    verbose : bool
        Print progress.

    Returns
    -------
    OptimizationResult
        Optimization result with the designed ε profile.
    """
    pml_cfg = PMLConfig.for_problem(
        n_bits=n_bits, k=k0, target_R_dB=-60.0,
    )

    geo = Geometry1D(
        n_bits=n_bits,
        background_eps=1.0,
        pml=pml_cfg,
    )
    # Substrate extends from coating_end to near the right PML
    geo.add_dielectric_slab(
        eps_substrate, coating_end, 0.85, label="substrate",
    )

    design = DesignRegion(
        x_start=coating_start,
        x_end=coating_end,
        eps_min=1.0 + 0j,
        eps_max=eps_substrate,
    )

    port = Port(
        position=port_position,
        ref_position=ref_position,
        direction=1,
        eps_r=1.0,
        width=0.02,
        label="Port 1",
    )

    config = OptimizationConfig(
        max_iterations=max_iterations,
        learning_rate=0.3,
        beta_init=1.0,
        beta_max=16.0,
        beta_increase_every=10,
        beta_factor=2.0,
        regularisation_weight=0.005,
        regularisation_type="tv",
        convergence_tol=1e-5,
    )

    return optimize_topology(
        geometry=geo,
        design=design,
        k0=k0,
        port=port,
        config=config,
        max_rank=max_rank,
        solver_tol=1e-4,
        n_sweeps=40,
        damping=pml_cfg.damping,
        verbose=verbose,
    )


# =====================================================================
# Section 9: Design Analysis Utilities
# =====================================================================

def binarisation_metric(rho: NDArray) -> float:
    """Measure how close the density is to binary (0 or 1).

    Returns 0 when all ρ ∈ {0, 1}, and 1 when all ρ = 0.5.

    .. math::

        M = \\frac{4}{N} \\sum_i \\rho_i (1 - \\rho_i)

    Parameters
    ----------
    rho : NDArray
        Density field ∈ [0, 1].

    Returns
    -------
    float
        Binarisation metric ∈ [0, 1]. Lower = more binary.
    """
    return float(4.0 * np.mean(rho * (1.0 - rho)))


def volume_fraction(rho: NDArray) -> float:
    """Compute volume fraction (mean density).

    Parameters
    ----------
    rho : NDArray
        Density field.

    Returns
    -------
    float
        Mean(ρ) ∈ [0, 1].
    """
    return float(np.mean(rho))


def design_complexity(rho: NDArray, threshold: float = 0.5) -> int:
    """Count the number of material interfaces in the design.

    An interface occurs where the binarised density changes
    between 0 and 1.

    Parameters
    ----------
    rho : NDArray
        Density field.
    threshold : float
        Binarisation threshold.

    Returns
    -------
    int
        Number of 0→1 or 1→0 transitions.
    """
    binary = (rho >= threshold).astype(int)
    return int(np.sum(np.abs(np.diff(binary))))
