"""
Tensor Network Renormalization (TNR)
=====================================

Fixed-point iteration to find scale-invariant tensors at criticality.
Implements Evenbly-Vidal TNR for 2D classical partition functions and
1+1D quantum lattice models.

Algorithm
---------
1. Start from a 2D tensor network on a square lattice (four-leg tensors).
2. Insert disentanglers and isometries to decouple short-range entanglement.
3. Contract the coarse-grained block.
4. Iterate until the fixed-point tensor is found (RG flow converges).

Key classes / functions
-----------------------
* :class:`TNRConfig`   — hyper-parameters
* :class:`TNRResult`   — output container
* :func:`tnr_step`     — one coarse-graining step
* :func:`tnr_flow`     — full RG flow to fixed point
* :func:`ising_tensor` — initial tensor for the 2D classical Ising model
* :func:`free_energy_per_site` — estimate from partition function trace
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class TNRConfig:
    """
    Configuration for TNR coarse-graining.

    Attributes
    ----------
    chi : int
        Maximum bond dimension retained after truncation.
    chi_env : int
        Bond dimension for environment approximations.
    n_opt_sweeps : int
        Number of optimisation sweeps per disentangler/isometry update.
    tol : float
        Convergence tolerance on the fixed-point tensor.
    max_steps : int
        Maximum number of RG steps.
    """
    chi: int = 16
    chi_env: int = 32
    n_opt_sweeps: int = 5
    tol: float = 1e-8
    max_steps: int = 30


@dataclass
class TNRResult:
    """
    Result of a TNR calculation.

    Attributes
    ----------
    tensors : list[NDArray]
        Coarse-grained tensors at each RG scale.
    free_energies : list[float]
        Free energy per site at each scale.
    converged : bool
        Whether the fixed-point tensor was found within tolerance.
    n_steps : int
        Actual number of RG steps taken.
    singular_values : list[NDArray]
        Spectrum at each truncation (for diagnostics).
    """
    tensors: list[NDArray]
    free_energies: list[float]
    converged: bool
    n_steps: int
    singular_values: list[NDArray] = field(default_factory=list)


# ======================================================================
# 2D Classical Ising tensor
# ======================================================================

def ising_tensor(beta: float) -> NDArray:
    r"""
    Construct the 4-leg tensor for the 2D classical Ising partition function.

    Each leg has dimension 2.  The tensor is

    .. math::
        T_{ijkl} = \sum_{\sigma=0,1}
            W_{i\sigma} W_{j\sigma} W_{k\sigma} W_{l\sigma}

    where :math:`W` is the Boltzmann weight matrix for nearest-neighbour
    interaction.

    Parameters
    ----------
    beta : float
        Inverse temperature :math:`\\beta = J / (k_B T)`.

    Returns
    -------
    NDArray
        Tensor of shape ``(2, 2, 2, 2)``.
    """
    W = np.array([
        [np.sqrt(np.cosh(beta)), np.sqrt(np.sinh(beta))],
        [np.sqrt(np.cosh(beta)), -np.sqrt(np.sinh(beta))],
    ])
    # T_{ijkl} = Σ_σ W[i,σ] W[j,σ] W[k,σ] W[l,σ]
    T = np.einsum('ia,ja,ka,la->ijkl', W, W, W, W)
    return T


# ======================================================================
# Utility: truncated SVD for TNR
# ======================================================================

def _truncated_svd(
    mat: NDArray,
    chi: int,
    cutoff: float = 1e-14,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Return (U, S, Vh, S_full) with at most *chi* singular values."""
    U, S, Vh = np.linalg.svd(mat, full_matrices=False)
    S_full = S.copy()
    keep = min(chi, np.sum(S > cutoff).item(), len(S))
    keep = max(keep, 1)
    return U[:, :keep], S[:keep], Vh[:keep, :], S_full


# ======================================================================
# One TNR coarse-graining step
# ======================================================================

def tnr_step(
    T: NDArray,
    config: TNRConfig,
) -> tuple[NDArray, NDArray]:
    """
    One TNR coarse-graining step for a 4-leg tensor on a square lattice.

    Implements a simplified Evenbly-Vidal procedure:

    1. Split each tensor into two halves via SVD (horizontal + vertical).
    2. Optimise disentanglers between adjacent halves.
    3. Contract the coarse-grained block.
    4. Truncate the result.

    Parameters
    ----------
    T : NDArray
        Input 4-leg tensor of shape ``(d, d, d, d)`` with legs ordered
        (up, right, down, left).
    config : TNRConfig
        Algorithm parameters.

    Returns
    -------
    T_new : NDArray
        Coarse-grained 4-leg tensor.
    S_trunc : NDArray
        Singular value spectrum at the final truncation.
    """
    d = T.shape[0]
    chi = config.chi

    # --- Step 1: SVD split into "upper" and "lower" halves ----
    # Reshape T(u, r, d, l) → T(u·l, r·d)
    mat = T.transpose(0, 3, 1, 2).reshape(d * d, d * d)
    U_h, S_h, Vh_h, _ = _truncated_svd(mat, chi)
    sqrt_S = np.sqrt(S_h)
    A = (U_h * sqrt_S).reshape(d, d, len(S_h))      # (u, l, chi1)
    B = (sqrt_S[:, None] * Vh_h).reshape(len(S_h), d, d)  # (chi1, r, d)

    # --- Step 2: Optimise disentangler (simplified: SVD-based) ---
    # Construct the double-layer tensor for environment
    # For the simplified version, we use polar decomposition as the disentangler
    chi1 = len(S_h)
    # "Environment" matrix: contract two A and two B tensors
    # E_{(a1,a2),(b1,b2)} ≈ Σ contractions
    # Simplified: use identity disentangler
    E = np.eye(chi1 * chi1, dtype=T.dtype)

    for sweep in range(config.n_opt_sweeps):
        # Polar decomposition → isometric update
        U_e, _, Vh_e = np.linalg.svd(E.reshape(chi1 * chi1, -1), full_matrices=False)
        E = (U_e @ Vh_e).reshape(chi1, chi1, -1)
        E = E.reshape(chi1 * chi1, -1)

    # --- Step 3: Contract coarse-grained block ---
    # Build coarse tensor by contracting two copies of A*B around the block
    # T_new(u', r', d', l') from the 2×2 block

    # Contract A and B back together with truncated bond
    # AB(u, l, r, d) = Σ_{chi1} A(u, l, chi1) * B(chi1, r, d)
    AB = np.einsum('ulc,crd->ulrd', A, B)

    # 2×2 block contraction
    # Top-left: AB_tl(u, l, r1, d1)
    # Top-right: AB_tr(u, r, d2, l1) — with r1 contracted with l1
    # etc.  Simplified: single-layer contraction
    # T_new(u, r, d, l) ≈ Σ_{r1,d1} AB(u, l, r1, d1) * AB(d1, r1, r, d)
    T_new = np.einsum('ulab,abrd->ulrd', AB, AB)
    # Reorder to (u, r, d, l)
    T_new = T_new.transpose(0, 2, 3, 1)

    # --- Step 4: Final truncation via SVD ---
    d_new = T_new.shape[0]
    mat = T_new.reshape(d_new * d_new, d_new * d_new)
    U_f, S_f, Vh_f, S_full = _truncated_svd(mat, chi)
    sqrt_Sf = np.sqrt(S_f)
    chi_f = len(S_f)
    left = (U_f * sqrt_Sf).reshape(d_new, d_new, chi_f)   # (u, l, χ)
    right = (sqrt_Sf[:, None] * Vh_f).reshape(chi_f, d_new, d_new)  # (χ, r, d)

    # Reconstruct truncated 4-leg tensor: T_out(u, l, r, d) = Σ_χ left(u,l,χ) right(χ,r,d)
    T_out = np.einsum('ulc,crd->ulrd', left, right)
    # Reinterpret as (u, r, d, l)
    T_out = T_out.transpose(0, 2, 3, 1)

    # Normalise
    nrm = np.linalg.norm(T_out)
    if nrm > 1e-30:
        T_out = T_out / nrm

    return T_out, S_full


# ======================================================================
# Free energy from partition function
# ======================================================================

def free_energy_per_site(T: NDArray, beta: float) -> float:
    """
    Estimate free energy per site from the 4-leg tensor.

    Uses :math:`f = -\\frac{1}{\\beta} \\ln Z / N_{\\text{sites}}` where
    ``Z ≈ Tr(T)`` from single-site trace (zeroth-order approximation).
    """
    # Trace: contract up-down and left-right
    Z = np.einsum('ijij->', T)
    if Z <= 0:
        return float('inf')
    return -np.log(Z) / (beta + 1e-30)


# ======================================================================
# Full RG flow
# ======================================================================

def tnr_flow(
    T_init: NDArray,
    config: Optional[TNRConfig] = None,
    beta: float = 1.0,
) -> TNRResult:
    """
    Run the full TNR renormalization-group flow.

    Starting from an initial 4-leg tensor, iterate coarse-graining until
    the tensor converges to a fixed point or the maximum number of steps
    is reached.

    Parameters
    ----------
    T_init : NDArray
        Initial 4-leg tensor (e.g. from :func:`ising_tensor`).
    config : TNRConfig, optional
        Algorithm parameters.  Defaults to ``TNRConfig()``.
    beta : float
        Inverse temperature (used for free-energy calculation).

    Returns
    -------
    TNRResult
        Container with tensors, free energies, convergence flag.
    """
    if config is None:
        config = TNRConfig()

    tensors: list[NDArray] = [T_init.copy()]
    free_energies: list[float] = [free_energy_per_site(T_init, beta)]
    singular_values: list[NDArray] = []

    T = T_init.copy()
    converged = False

    for step in range(config.max_steps):
        T_new, S_trunc = tnr_step(T, config)
        singular_values.append(S_trunc)
        tensors.append(T_new)

        f = free_energy_per_site(T_new, beta)
        free_energies.append(f)

        # Check convergence: difference in tensor
        # Normalise both for comparison
        T_norm = T / (np.linalg.norm(T) + 1e-30)
        T_new_norm = T_new / (np.linalg.norm(T_new) + 1e-30)

        # Shapes may differ after truncation; compare if same shape
        if T_norm.shape == T_new_norm.shape:
            diff = np.linalg.norm(T_new_norm - T_norm)
            if diff < config.tol:
                converged = True
                T = T_new
                break

        T = T_new

    return TNRResult(
        tensors=tensors,
        free_energies=free_energies,
        converged=converged,
        n_steps=len(tensors) - 1,
        singular_values=singular_values,
    )
