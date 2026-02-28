"""QTT Frequency-Domain Helmholtz Solver.

Solves the Helmholtz equation

    ∇²E + k²ε_r(x) E(x) = -J(x)

entirely in QTT format using TT-GMRES with Hermitian inner products.

Phase 1 of the frequency-domain QTT Maxwell program.

Validated by Phase 0: Helmholtz solutions have bounded QTT rank
(χ ≤ 40 across 3 media × 3 wavenumbers × 7 grid sizes up to 2^20).

Architecture
------------
1. Build Helmholtz operator H = ∇² + k²·diag(ε_pml) as MPO
   - ∇² from existing ``laplacian_mpo_1d`` (bond dim ≤ 5)
   - diag(ε_pml) from QTT representation of PML permittivity
   - Combined via MPO addition (bond dim ≤ 5 + rank(ε_pml))

2. Source J in QTT format (Gaussian, plane wave, or port mode)

3. Solve H·E = -J via complex TT-GMRES (Arnoldi + Hermitian inner products)

Dependencies (all existing in ontic)
-----------------------------------------
- ``ontic.vm.operators``: ``laplacian_mpo_1d``, ``identity_mpo``
- ``ontic.qtt.sparse_direct``: ``tt_matvec``, ``tt_round``
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ontic.engine.vm.operators import (
    laplacian_mpo_1d,
    identity_mpo,
    _embed_1d_mpo,
)
from ontic.qtt.sparse_direct import tt_matvec, tt_round


# ======================================================================
# Section 1: Complex TT-SVD
# ======================================================================

def array_to_tt(
    arr: NDArray,
    max_rank: int = 64,
    cutoff: float = 1e-12,
) -> list[NDArray]:
    """Decompose a 1-D array of length 2^n into QTT (TT) cores.

    Handles real and complex arrays via standard left-to-right SVD
    decomposition.  Each core has shape ``(r_left, 2, r_right)``.

    Parameters
    ----------
    arr : NDArray
        1-D array of length 2^n.
    max_rank : int
        Maximum bond dimension.
    cutoff : float
        Relative SVD cutoff (Frobenius norm fraction).

    Returns
    -------
    list[NDArray]
        TT cores, each of shape ``(r_left, 2, r_right)``.
    """
    N = len(arr)
    n_bits = int(math.log2(N))
    if 2 ** n_bits != N:
        raise ValueError(f"N={N} must be a power of 2")

    tensor = arr.reshape([2] * n_bits)
    cores: list[NDArray] = []
    C = tensor.reshape(2, -1)
    r = 1

    for k in range(n_bits - 1):
        C = C.reshape(r * 2, -1)
        U, S, Vh = np.linalg.svd(C, full_matrices=False)

        new_r = len(S)
        if cutoff > 0:
            total_sq = np.sum(S ** 2)
            if total_sq > 0:
                cum = np.cumsum(S ** 2)
                keep = int(np.searchsorted(cum / total_sq, 1.0 - cutoff ** 2)) + 1
                new_r = min(new_r, keep)
        new_r = min(new_r, max_rank)
        new_r = max(new_r, 1)

        cores.append(U[:, :new_r].reshape(r, 2, new_r))
        C = np.diag(S[:new_r]) @ Vh[:new_r, :]
        r = new_r

    cores.append(C.reshape(r, 2, 1))
    return cores


def reconstruct_1d(tt_cores: list[NDArray]) -> NDArray:
    """Reconstruct full dense 1-D array from QTT cores.

    Only suitable for small grids (validation purposes).

    Parameters
    ----------
    tt_cores : list[NDArray]
        QTT cores, each ``(r_l, 2, r_r)``.

    Returns
    -------
    NDArray
        Dense 1-D array of length 2^n.
    """
    result = tt_cores[0]  # (1, 2, r1)
    for k in range(1, len(tt_cores)):
        result = np.einsum('...i,ijk->...jk', result, tt_cores[k])
        shape = result.shape
        result = result.reshape(shape[0], -1, shape[-1])
    return result.reshape(-1)


# ======================================================================
# Section 2: Complex TT Algebra
# ======================================================================

def tt_inner_hermitian(a: list[NDArray], b: list[NDArray]) -> complex:
    r"""Hermitian inner product :math:`\langle a, b \rangle = \bar{a}^T b`.

    For real inputs this is identical to the standard dot product.
    For complex inputs the first argument is conjugated.

    Parameters
    ----------
    a, b : list[NDArray]
        TT cores, each ``(r_left, d, r_right)``.

    Returns
    -------
    complex
        The inner product.
    """
    N = len(a)
    if len(b) != N:
        raise ValueError(f"TT lengths differ: {N} vs {len(b)}")
    dtype = np.result_type(a[0].dtype, b[0].dtype)
    T = np.ones((1, 1), dtype=dtype)
    for k in range(N):
        T = np.einsum('ij,idk,jdl->kl', T, np.conj(a[k]), b[k])
    return complex(T.item())


def tt_norm_c(cores: list[NDArray]) -> float:
    """Frobenius norm of a (possibly complex) TT vector."""
    val = tt_inner_hermitian(cores, cores)
    return math.sqrt(max(val.real, 0.0))


def tt_scale_c(cores: list[NDArray], alpha: complex) -> list[NDArray]:
    """Scale TT vector by a (possibly complex) scalar."""
    result = [c.copy() for c in cores]
    result[0] = result[0] * alpha
    return result


def tt_add_c(a: list[NDArray], b: list[NDArray]) -> list[NDArray]:
    """Add two TT vectors with proper dtype promotion."""
    N = len(a)
    if len(b) != N:
        raise ValueError(f"TT lengths differ: {N} vs {len(b)}")
    dtype = np.result_type(a[0].dtype, b[0].dtype)
    cores: list[NDArray] = []
    for k in range(N):
        ra_l, d, ra_r = a[k].shape
        rb_l, db, rb_r = b[k].shape
        if d != db:
            raise ValueError(
                f"Physical dimension mismatch at site {k}: {d} vs {db}"
            )
        if k == 0:
            C = np.zeros((1, d, ra_r + rb_r), dtype=dtype)
            C[:, :, :ra_r] = a[k]
            C[:, :, ra_r:] = b[k]
        elif k == N - 1:
            C = np.zeros((ra_l + rb_l, d, 1), dtype=dtype)
            C[:ra_l, :, :] = a[k]
            C[ra_l:, :, :] = b[k]
        else:
            C = np.zeros((ra_l + rb_l, d, ra_r + rb_r), dtype=dtype)
            C[:ra_l, :, :ra_r] = a[k]
            C[ra_l:, :, ra_r:] = b[k]
        cores.append(C)
    return cores


def tt_axpy_c(
    alpha: complex,
    x: list[NDArray],
    y: list[NDArray],
    max_rank: int = 128,
    cutoff: float = 1e-14,
) -> list[NDArray]:
    """Compute ``y + alpha * x`` with TT rounding (complex-safe)."""
    sx = tt_scale_c(x, alpha)
    return tt_round(tt_add_c(sx, y), max_rank=max_rank, cutoff=cutoff)


# ======================================================================
# Section 3: Complex MPO Helpers
# ======================================================================

def mpo_add_c(a: list[NDArray], b: list[NDArray]) -> list[NDArray]:
    """Add two MPOs with dtype promotion (handles mixed real/complex)."""
    n = len(a)
    if len(b) != n:
        raise ValueError(f"MPO lengths differ: {n} vs {len(b)}")
    dtype = np.result_type(a[0].dtype, b[0].dtype)
    cores: list[NDArray] = []
    for k in range(n):
        ac, bc = a[k], b[k]
        ra_l, d_out, d_in, ra_r = ac.shape
        rb_l, _, _, rb_r = bc.shape

        if k == 0:
            core = np.zeros((1, d_out, d_in, ra_r + rb_r), dtype=dtype)
            core[0, :, :, :ra_r] = ac[0]
            core[0, :, :, ra_r:] = bc[0]
        elif k == n - 1:
            core = np.zeros((ra_l + rb_l, d_out, d_in, 1), dtype=dtype)
            core[:ra_l, :, :, 0] = ac[:, :, :, 0]
            core[ra_l:, :, :, 0] = bc[:, :, :, 0]
        else:
            core = np.zeros(
                (ra_l + rb_l, d_out, d_in, ra_r + rb_r), dtype=dtype
            )
            core[:ra_l, :, :, :ra_r] = ac
            core[ra_l:, :, :, ra_r:] = bc
        cores.append(core)
    return cores


def mpo_scale_c(cores: list[NDArray], alpha: complex) -> list[NDArray]:
    """Scale MPO by a (possibly complex) scalar."""
    out = [c.copy() for c in cores]
    out[0] = out[0] * alpha
    return out


def diag_mpo_from_tt(tt_cores: list[NDArray]) -> list[NDArray]:
    r"""Build a diagonal MPO from TT vector cores.

    Given TT vector *v* with cores ``V_k`` of shape ``(r_l, d, r_r)``,
    produces diagonal MPO with cores of shape ``(r_l, d, d, r_r)``
    where ``M_k[a, j, i, b] = V_k[a, i, b] · δ(i, j)``.

    This represents the operator :math:`\mathrm{diag}(v)`:
    :math:`x \mapsto v \odot x` (Hadamard product).

    Parameters
    ----------
    tt_cores : list[NDArray]
        TT vector cores, each ``(r_l, d, r_r)``.

    Returns
    -------
    list[NDArray]
        MPO cores, each ``(r_l, d, d, r_r)``.
    """
    mpo_cores: list[NDArray] = []
    for core in tt_cores:
        r_l, d, r_r = core.shape
        mpo_core = np.zeros((r_l, d, d, r_r), dtype=core.dtype)
        for i in range(d):
            mpo_core[:, i, i, :] = core[:, i, :]
        mpo_cores.append(mpo_core)
    return mpo_cores


# ======================================================================
# Section 4: PML Profile Construction
# ======================================================================

def build_pml_eps_profile(
    n_bits: int,
    eps_r: NDArray | float = 1.0,
    pml_cells: int = 20,
    sigma_max: float = 10.0,
    damping: float = 0.0,
    max_rank: int = 64,
    cutoff: float = 1e-12,
) -> list[NDArray]:
    r"""Build complex PML permittivity profile as QTT cores.

    PML stretching:
    :math:`\varepsilon_{\mathrm{pml}}(x) = \varepsilon_r(x)\,(1 + j\,\sigma(x))`
    where σ(x) is a quadratic ramp in the absorbing layers.

    An optional global *damping* term adds a uniform imaginary part
    :math:`\varepsilon \to \varepsilon\,(1 + j\,\delta)` everywhere.
    This breaks exact periodic-boundary resonances that arise because
    the QTT Laplacian operator uses periodic boundary conditions.
    A value of ``damping ≈ 0.01`` is sufficient to regularise all
    practical configurations without measurably affecting the physics.

    Parameters
    ----------
    n_bits : int
        Number of QTT bits (grid has ``N = 2^n_bits`` points).
    eps_r : NDArray or float
        Relative permittivity.  Scalar for uniform, array for varying.
    pml_cells : int
        Number of PML cells on each end.
    sigma_max : float
        Maximum PML conductivity.
    damping : float
        Global imaginary shift applied to ε everywhere (0 = no shift).
    max_rank : int
        Maximum QTT rank for the profile.
    cutoff : float
        SVD cutoff.

    Returns
    -------
    list[NDArray]
        QTT cores for ε_pml(x), dtype ``complex128``.
    """
    N = 2 ** n_bits

    if isinstance(eps_r, (int, float)):
        eps_arr = np.full(N, eps_r, dtype=np.complex128)
    else:
        eps_arr = np.asarray(eps_r).astype(np.complex128).copy()

    # Global damping: regularises the indefinite Helmholtz operator
    # by ensuring no eigenvalue lies exactly on the real axis.
    if damping > 0:
        eps_arr *= (1.0 + 1j * damping)

    # Quadratic PML ramp on each boundary
    for i in range(N):
        if i < pml_cells:
            depth = (pml_cells - i) / pml_cells
            sigma = sigma_max * depth ** 2
            eps_arr[i] *= (1.0 + 1j * sigma)
        elif i >= N - pml_cells:
            depth = (i - (N - pml_cells - 1)) / pml_cells
            sigma = sigma_max * depth ** 2
            eps_arr[i] *= (1.0 + 1j * sigma)

    return array_to_tt(eps_arr, max_rank=max_rank, cutoff=cutoff)


# ======================================================================
# Section 5: Helmholtz MPO Assembly
# ======================================================================

def helmholtz_mpo_1d(
    n_bits: int,
    k: float,
    h: float,
    eps_pml_cores: Optional[list[NDArray]] = None,
    eps_r: float = 1.0,
    pml_cells: int = 20,
    sigma_max: float = 10.0,
    damping: float = 0.01,
    max_rank: int = 64,
) -> list[NDArray]:
    r"""Build 1-D Helmholtz operator :math:`H = \nabla^2 + k^2 \varepsilon` as MPO.

    The Helmholtz equation: :math:`(\nabla^2 + k^2\varepsilon)E = -J`

    Combined bond dimension ≤ 5 + rank(ε_pml):
    - ∇² contributes bond dim ≤ 5 (shift + identity sum)
    - k²·diag(ε_pml) contributes bond dim = rank(ε_pml)

    Parameters
    ----------
    n_bits : int
        Number of QTT sites.
    k : float
        Wavenumber.
    h : float
        Grid spacing.
    eps_pml_cores : list[NDArray], optional
        Pre-built QTT cores for ε_pml profile.  If ``None``, builds
        uniform ε_r with PML from the remaining parameters.
    eps_r : float
        Uniform relative permittivity (used when *eps_pml_cores* is ``None``).
    pml_cells : int
        PML cells (used when *eps_pml_cores* is ``None``).
    sigma_max : float
        PML conductivity (used when *eps_pml_cores* is ``None``).
    max_rank : int
        Maximum QTT rank for the ε profile.

    Returns
    -------
    list[NDArray]
        Complex MPO cores for the Helmholtz operator.
    """
    # Laplacian MPO (real, bond dim ≤ 5)
    L_cores = laplacian_mpo_1d(n_bits, h)

    # PML permittivity as QTT
    if eps_pml_cores is None:
        eps_pml_cores = build_pml_eps_profile(
            n_bits, eps_r=eps_r, pml_cells=pml_cells,
            sigma_max=sigma_max, damping=damping, max_rank=max_rank,
        )

    # k² · diag(ε_pml) MPO
    eps_mpo = diag_mpo_from_tt(eps_pml_cores)
    k2_eps_mpo = mpo_scale_c(eps_mpo, k * k)

    # H = L + k²·diag(ε_pml) — cast Laplacian to complex for addition
    L_complex = [c.astype(np.complex128) for c in L_cores]
    return mpo_add_c(L_complex, k2_eps_mpo)


def helmholtz_mpo_3d(
    n_bits_per_dim: tuple[int, int, int],
    k: float,
    domain: tuple[tuple[float, float], ...],
    eps_pml_cores: Optional[list[NDArray]] = None,
) -> list[NDArray]:
    r"""Build 3-D Helmholtz operator :math:`H = \nabla^2 + k^2\varepsilon` as MPO.

    Uses Kronecker structure for the Laplacian:

    .. math::

        \nabla^2 = \nabla^2_x \otimes I_y \otimes I_z
                 + I_x \otimes \nabla^2_y \otimes I_z
                 + I_x \otimes I_y \otimes \nabla^2_z

    Parameters
    ----------
    n_bits_per_dim : tuple[int, int, int]
        QTT bits per dimension.
    k : float
        Wavenumber.
    domain : tuple
        Physical domain bounds per dimension, e.g.
        ``((0, 1), (0, 1), (0, 1))``.
    eps_pml_cores : list[NDArray], optional
        QTT cores for 3-D ε_pml field.  If ``None``, uses uniform ε=1.

    Returns
    -------
    list[NDArray]
        Complex MPO cores for the 3-D Helmholtz operator.
    """
    from ontic.engine.vm.operators import laplacian_mpo

    # Full Laplacian (sums over all 3 dims internally)
    L_cores = laplacian_mpo(n_bits_per_dim, domain)
    L_complex = [c.astype(np.complex128) for c in L_cores]

    # k²·ε term
    total_sites = sum(n_bits_per_dim)
    if eps_pml_cores is not None:
        eps_mpo = diag_mpo_from_tt(eps_pml_cores)
        k2_eps_mpo = mpo_scale_c(eps_mpo, k * k)
    else:
        # Uniform ε = 1 → k²·I
        I_cores = identity_mpo(total_sites)
        I_complex = [c.astype(np.complex128) for c in I_cores]
        k2_eps_mpo = mpo_scale_c(I_complex, k * k)

    return mpo_add_c(L_complex, k2_eps_mpo)


# ======================================================================
# Section 6: Complex TT-GMRES
# ======================================================================

@dataclass
class TTGMRESResult:
    """Result of complex TT-GMRES solve."""

    x: list[NDArray]
    residual_norms: list[float]
    converged: bool
    n_iter: int
    final_residual: float


def tt_gmres_complex(
    mpo_cores: list[NDArray],
    b_cores: list[NDArray],
    x0: Optional[list[NDArray]] = None,
    max_iter: int = 200,
    restart: int = 30,
    tol: float = 1e-8,
    max_rank: int = 128,
    verbose: bool = False,
) -> TTGMRESResult:
    r"""Complex GMRES in TT format for Helmholtz-type systems.

    Arnoldi iteration with Hermitian inner products and complex
    Givens rotations.  All intermediate vectors stay in TT format
    with rank bounded by *max_rank*.

    The algorithm follows Saad's "Iterative Methods for Sparse Linear
    Systems" §6.5, adapted so that all vectors are stored and
    manipulated in TT format.

    Parameters
    ----------
    mpo_cores : list[NDArray]
        Complex MPO cores for the operator.
    b_cores : list[NDArray]
        Complex TT-vector cores for the right-hand side.
    x0 : list[NDArray], optional
        Initial guess (zero if ``None``).
    max_iter : int
        Maximum total GMRES iterations.
    restart : int
        Arnoldi restart period.
    tol : float
        Relative residual tolerance.
    max_rank : int
        Maximum TT bond dimension.
    verbose : bool
        Print convergence diagnostics.

    Returns
    -------
    TTGMRESResult
    """
    N_sites = len(b_cores)
    d = b_cores[0].shape[1]

    if x0 is None:
        x = [np.zeros((1, d, 1), dtype=np.complex128) for _ in range(N_sites)]
    else:
        x = [c.copy().astype(np.complex128) for c in x0]

    b_norm = tt_norm_c(b_cores)
    if b_norm < 1e-30:
        b_norm = 1.0

    all_residuals: list[float] = []
    total_iter = 0

    for outer in range(max_iter // max(restart, 1) + 1):
        # r = b - A·x
        Ax = tt_matvec(mpo_cores, x, max_rank=max_rank)
        r = tt_axpy_c(-1.0, Ax, [c.copy() for c in b_cores], max_rank=max_rank)
        beta = tt_norm_c(r)
        all_residuals.append(beta)

        if verbose:
            print(f"  GMRES outer={outer} residual={beta:.4e} "
                  f"(rel={beta / b_norm:.4e})")

        if beta / b_norm < tol:
            return TTGMRESResult(
                x=x, residual_norms=all_residuals,
                converged=True, n_iter=total_iter,
                final_residual=beta / b_norm,
            )

        # Arnoldi basis: V[0] = r / ||r||
        V: list[list[NDArray]] = [
            tt_scale_c(r, 1.0 / max(beta, 1e-30))
        ]

        m = min(restart, max_iter - total_iter)
        H = np.zeros((m + 1, m), dtype=np.complex128)
        g = np.zeros(m + 1, dtype=np.complex128)
        g[0] = beta

        # Storage for Givens rotations
        cs = np.zeros(m, dtype=np.complex128)
        sn = np.zeros(m, dtype=np.complex128)

        for j in range(m):
            total_iter += 1

            # w = A · v_j
            w = tt_matvec(mpo_cores, V[j], max_rank=max_rank)

            # Modified Gram-Schmidt with Hermitian inner product
            for i in range(j + 1):
                H[i, j] = tt_inner_hermitian(V[i], w)
                w = tt_axpy_c(-H[i, j], V[i], w, max_rank=max_rank)

            H[j + 1, j] = tt_norm_c(w)

            if abs(H[j + 1, j]) > 1e-14:
                v_new = tt_scale_c(w, 1.0 / H[j + 1, j])
                v_new = tt_round(v_new, max_rank=max_rank)
                V.append(v_new)
            else:
                # Lucky breakdown
                V.append(w)

            # Apply previous Givens rotations to column j
            for i in range(j):
                h_i = H[i, j]
                h_i1 = H[i + 1, j]
                H[i, j] = np.conj(cs[i]) * h_i + np.conj(sn[i]) * h_i1
                H[i + 1, j] = -sn[i] * h_i + cs[i] * h_i1

            # Compute new Givens rotation for (H[j,j], H[j+1,j])
            a_val = H[j, j]
            b_val = H[j + 1, j]
            rho = math.sqrt(abs(a_val) ** 2 + abs(b_val) ** 2)

            if rho > 1e-30:
                cs[j] = a_val / rho
                sn[j] = b_val / rho
            else:
                cs[j] = 1.0
                sn[j] = 0.0

            # Apply current rotation
            H[j, j] = rho
            H[j + 1, j] = 0.0

            g_j = g[j]
            g_j1 = g[j + 1]
            g[j] = np.conj(cs[j]) * g_j + np.conj(sn[j]) * g_j1
            g[j + 1] = -sn[j] * g_j + cs[j] * g_j1

            res_est = abs(g[j + 1])
            all_residuals.append(res_est)

            if verbose and (j + 1) % 10 == 0:
                print(f"    iter={total_iter:3d} residual={res_est:.4e} "
                      f"(rel={res_est / b_norm:.4e})")

            if res_est / b_norm < tol:
                # Solve upper triangular system
                y = np.linalg.solve(H[:j + 1, :j + 1], g[:j + 1])
                for i in range(j + 1):
                    x = tt_axpy_c(y[i], V[i], x, max_rank=max_rank)
                return TTGMRESResult(
                    x=x, residual_norms=all_residuals,
                    converged=True, n_iter=total_iter,
                    final_residual=res_est / b_norm,
                )

        # End of restart: update x
        y = np.linalg.solve(H[:m, :m], g[:m])
        for i in range(m):
            x = tt_axpy_c(y[i], V[i], x, max_rank=max_rank)

    final_res = all_residuals[-1] / b_norm if all_residuals else float('inf')
    return TTGMRESResult(
        x=x, residual_norms=all_residuals,
        converged=False, n_iter=total_iter,
        final_residual=final_res,
    )


# ======================================================================
# Section 6b: AMEN (Alternating Minimal Energy) Solver
# ======================================================================
#
# DMRG-style alternating sweep solver for Ax = b in TT format.
# Converges in O(n_sweeps * n_sites) local solves, each of size
# O(r^2 * d^2), independent of the full system dimension N = d^n.
# This makes it scalable where GMRES (needing O(N/2) Krylov vectors)
# is not.
#
# Reference: Dolgov & Savostyanov, "Alternating minimal energy
# methods for linear systems in higher dimensions", SIAM J. Sci.
# Comput. 36(5), 2014.
# ======================================================================


def _right_orth(cores: list[NDArray]) -> list[NDArray]:
    """Return right-orthogonalized copy of TT cores.

    After this, cores[i] satisfies::

        sum_{s, r'} conj(G[i][:, s, r']) * G[i][:, s, r'] == I_{r_i}

    for all i > 0 (the left-most core absorbs the norms).
    """
    x = [c.copy() for c in cores]
    n = len(x)
    for i in range(n - 1, 0, -1):
        r_l, d_i, r_r = x[i].shape
        mat = x[i].reshape(r_l, d_i * r_r)
        # Thin SVD: mat = U S Vh  →  keep Vh as right-orthogonal core, push U*S left
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)
        r_new = len(S)
        x[i] = Vh.reshape(r_new, d_i, r_r)
        # Absorb U * diag(S) into the left core
        US = U * S[np.newaxis, :]  # (r_l_prev, r_new)
        x[i - 1] = np.einsum("idk, kj -> idj", x[i - 1], US)
    return x


def _update_phi_left(
    phi_l: NDArray,
    x_core: NDArray,
    A_core: NDArray,
) -> NDArray:
    """Update left operator projection (L→R).

    Parameters
    ----------
    phi_l : (rx, rA, rx)
        Left-accumulated <x|A|x> up to (but not including) current site.
    x_core : (rx_l, d, rx_r)
        Current TT core of x.
    A_core : (rA_l, d, d, rA_r)
        Current MPO core of A.

    Returns
    -------
    NDArray, shape (rx_r, rA_r, rx_r)
    """
    # phi_l[p, a, q] * x_ket[q, s, q'] → tmp[p, a, s, q']
    tmp = np.einsum("paq, qsr -> pasr", phi_l, x_core)
    # tmp[p, a, s, q'] * A[a, t, s, b] → tmp2[p, t, q', b]
    tmp2 = np.einsum("pasr, atsb -> ptrb", tmp, A_core)
    # conj(x_bra)[p, t, m] * tmp2[p, t, r, b] → out[m, b, r]
    out = np.einsum("ptm, ptrb -> mbr", np.conj(x_core), tmp2)
    return out


def _update_phi_right(
    phi_r: NDArray,
    x_core: NDArray,
    A_core: NDArray,
) -> NDArray:
    """Update right operator projection (R→L).

    Parameters
    ----------
    phi_r : (rx, rA, rx)
        Right-accumulated <x|A|x> from sites to the right.
    x_core : (rx_l, d, rx_r)
        Current TT core of x.
    A_core : (rA_l, d, d, rA_r)
        Current MPO core of A.

    Returns
    -------
    NDArray, shape (rx_l, rA_l, rx_l)
    """
    # phi_r[m, b, n] * x_ket[q, s, n] → tmp[m, b, q, s]
    tmp = np.einsum("mbn, qsn -> mbqs", phi_r, x_core)
    # tmp[m, b, q, s] * A[a, t, s, b] → tmp2[m, q, a, t]
    tmp2 = np.einsum("mbqs, atsb -> mqat", tmp, A_core)
    # conj(x_bra)[p, t, m] * tmp2[m, q, a, t] → out[p, a, q]
    out = np.einsum("ptm, mqat -> paq", np.conj(x_core), tmp2)
    return out


def _update_psi_left(
    psi_l: NDArray,
    x_core: NDArray,
    b_core: NDArray,
) -> NDArray:
    """Update left RHS projection <x|b> (L→R).

    Parameters
    ----------
    psi_l : (rx, rb)
    x_core : (rx_l, d, rx_r)
    b_core : (rb_l, d, rb_r)

    Returns
    -------
    NDArray, shape (rx_r, rb_r)
    """
    # psi_l[p, c] * b[c, s, d] → tmp[p, s, d]
    tmp = np.einsum("pc, csd -> psd", psi_l, b_core)
    # conj(x)[p, s, m] * tmp[p, s, d] → out[m, d]
    out = np.einsum("psm, psd -> md", np.conj(x_core), tmp)
    return out


def _update_psi_right(
    psi_r: NDArray,
    x_core: NDArray,
    b_core: NDArray,
) -> NDArray:
    """Update right RHS projection <x|b> (R→L).

    Parameters
    ----------
    psi_r : (rx, rb)
    x_core : (rx_l, d, rx_r)
    b_core : (rb_l, d, rb_r)

    Returns
    -------
    NDArray, shape (rx_l, rb_l)
    """
    # psi_r[m, d] * b[c, s, d] → tmp[m, c, s]
    tmp = np.einsum("md, csd -> mcs", psi_r, b_core)
    # conj(x)[p, s, m] * tmp[m, c, s] → out[p, c]
    out = np.einsum("psm, mcs -> pc", np.conj(x_core), tmp)
    return out


def _form_local_op(
    phi_l: NDArray,
    A_core: NDArray,
    phi_r: NDArray,
) -> NDArray:
    """Form the local Galerkin operator matrix at one site.

    H_local[(α,σ',β), (α',σ,β')]
        = PhiL[α, a, α'] · A[a, σ', σ, b] · PhiR[β, b, β']

    Parameters
    ----------
    phi_l : (rx_l, rA_l, rx_l)
    A_core : (rA_l, d, d, rA_r)
    phi_r : (rx_r, rA_r, rx_r)

    Returns
    -------
    NDArray, shape (rx_l*d*rx_r, rx_l*d*rx_r)
    """
    rx_l = phi_l.shape[0]
    _, d, _, _ = A_core.shape
    rx_r = phi_r.shape[0]

    # PhiL[p, a, q] * A[a, t, s, b] → tmp[p, q, t, s, b]
    tmp = np.einsum("paq, atsb -> pqtsb", phi_l, A_core)
    # tmp[p, q, t, s, b] * PhiR[r, b, w] → H6[p, q, t, s, r, w]
    H6 = np.einsum("pqtsb, rbw -> pqtsrw", tmp, phi_r)

    # Reshape: row=(p, t, r) = bra, col=(q, s, w) = ket
    H6 = H6.transpose(0, 2, 4, 1, 3, 5)  # → (p, t, r, q, s, w)
    H = H6.reshape(rx_l * d * rx_r, rx_l * d * rx_r)
    return H


def _form_local_rhs(
    psi_l: NDArray,
    b_core: NDArray,
    psi_r: NDArray,
) -> NDArray:
    """Form the local projected RHS vector at one site.

    f[(α, σ, β)] = psiL[α, c] · b[c, σ, e] · psiR[β, e]

    Parameters
    ----------
    psi_l : (rx_l, rb_l)
    b_core : (rb_l, d, rb_r)
    psi_r : (rx_r, rb_r)

    Returns
    -------
    NDArray, shape (rx_l * d * rx_r,)
    """
    # psi_l[α, c] * b[c, σ, e] → tmp[α, σ, e]
    tmp = np.einsum("pc, cse -> pse", psi_l, b_core)
    # tmp * psi_r[β, e] → f[α, σ, β]
    f = np.einsum("pse, re -> psr", tmp, psi_r)
    return f.reshape(-1)


def _merge_mpo_cores(
    A_i: NDArray, A_j: NDArray,
) -> NDArray:
    """Merge two adjacent MPO cores into a single supercore.

    A_i: (rA_l, d, d, rA_m)
    A_j: (rA_m, d, d, rA_r)

    Returns: (rA_l, d*d, d*d, rA_r) — merged MPO core with
    physical dimension d² (output: (σ_i, σ_j), input: (s_i, s_j)).
    """
    # Contract over the shared bond index
    # A_i[a, t, s, b] * A_j[b, u, v, c] → tmp[a, t, s, u, v, c]
    tmp = np.einsum("atsb, buvc -> atsuvc", A_i, A_j)
    rA_l, d1, d2, _, d3, rA_r = tmp.shape
    assert d1 == d2 == d3, "Physical dimensions must match"
    d = d1
    # Reorder: output = (t, u), input = (s, v) → (a, t, u, s, v, c)
    tmp = tmp.transpose(0, 1, 3, 2, 4, 5)
    return tmp.reshape(rA_l, d * d, d * d, rA_r)


def _merge_tt_cores(
    b_i: NDArray, b_j: NDArray,
) -> NDArray:
    """Merge two adjacent TT vector cores into a single supercore.

    b_i: (rb_l, d, rb_m)
    b_j: (rb_m, d, rb_r)

    Returns: (rb_l, d*d, rb_r)
    """
    # b_i[e, s, f] * b_j[f, t, g] → tmp[e, s, t, g]
    tmp = np.einsum("esf, ftg -> estg", b_i, b_j)
    rb_l, d1, d2, rb_r = tmp.shape
    return tmp.reshape(rb_l, d1 * d2, rb_r)


def tt_amen_solve(
    mpo_cores: list[NDArray],
    b_cores: list[NDArray],
    x0: Optional[list[NDArray]] = None,
    max_rank: int = 64,
    n_sweeps: int = 40,
    tol: float = 1e-8,
    verbose: bool = False,
) -> TTGMRESResult:
    r"""Two-site DMRG solver for :math:`Ax = b` in TT format.

    Sweeps through pairs of adjacent TT cores, solving local
    Galerkin-projected systems of size :math:`O(r^2 d^4)` and
    splitting via truncated SVD.  This automatically discovers the
    optimal bond dimension at each sweep — no manual enrichment
    needed.

    Convergence: :math:`O(n_{\text{sweeps}})` full passes of length
    ``n−1``, each costing :math:`O(n \, r^3 d^4)`.  Independent of
    the full system dimension :math:`N = d^n`, making this
    dramatically more scalable than Krylov methods for large *N*.

    Parameters
    ----------
    mpo_cores : list[NDArray]
        MPO cores, each shape ``(rA_l, d, d, rA_r)``.
    b_cores : list[NDArray]
        RHS TT cores, each shape ``(rb_l, d, rb_r)``.
    x0 : list[NDArray], optional
        Initial guess.  If ``None``, uses *b_cores* as warm start.
    max_rank : int
        Maximum TT bond dimension for the solution.
    n_sweeps : int
        Maximum number of full (L→R + R→L) sweeps.
    tol : float
        Relative residual tolerance.
    verbose : bool
        Print per-sweep convergence diagnostics.

    Returns
    -------
    TTGMRESResult
        Solution, convergence status, and diagnostics.
    """
    n = len(mpo_cores)
    if n < 2:
        raise ValueError("Need at least 2 sites for two-site DMRG")
    dtype = np.complex128
    d = mpo_cores[0].shape[1]  # physical dimension per site

    # ── Initialise x ──────────────────────────────────────────────
    if x0 is not None:
        x = [c.copy().astype(dtype) for c in x0]
    else:
        x = [c.copy().astype(dtype) for c in b_cores]

    b = [c.astype(dtype) for c in b_cores]
    A = [c.astype(dtype) for c in mpo_cores]

    # Right-orthogonalise x (all cores except x[0])
    x = _right_orth(x)

    b_norm = tt_norm_c(b)
    if b_norm < 1e-30:
        b_norm = 1.0

    # ── Build right projections for sites 1..n ────────────────────
    PhiR: list[Optional[NDArray]] = [None] * (n + 1)
    PsiR: list[Optional[NDArray]] = [None] * (n + 1)
    PhiR[n] = np.ones((1, 1, 1), dtype=dtype)
    PsiR[n] = np.ones((1, 1), dtype=dtype)

    for i in range(n - 1, 0, -1):
        PhiR[i] = _update_phi_right(PhiR[i + 1], x[i], A[i])
        PsiR[i] = _update_psi_right(PsiR[i + 1], x[i], b[i])

    # Left projections: built on the fly
    PhiL: list[Optional[NDArray]] = [None] * (n + 1)
    PsiL: list[Optional[NDArray]] = [None] * (n + 1)
    PhiL[0] = np.ones((1, 1, 1), dtype=dtype)
    PsiL[0] = np.ones((1, 1), dtype=dtype)

    residual_norms: list[float] = []

    for sweep in range(n_sweeps):
        # ── Left-to-right half-sweep (bonds 0..n-2) ──────────────
        for bond in range(n - 1):
            i, j = bond, bond + 1
            rx_l = x[i].shape[0]
            rx_r = x[j].shape[2]

            # Merged operator and RHS for the supersite (i, j)
            A_merged = _merge_mpo_cores(A[i], A[j])
            b_merged = _merge_tt_cores(b[i], b[j])

            # Local Galerkin system
            H_loc = _form_local_op(PhiL[i], A_merged, PhiR[j + 1])
            f_loc = _form_local_rhs(PsiL[i], b_merged, PsiR[j + 1])

            # Solve dense local system
            sol = np.linalg.solve(H_loc, f_loc)

            # Reshape to (rx_l * d, d * rx_r) and SVD-split
            sol_2d = sol.reshape(rx_l * d, d * rx_r)
            U, S, Vh = np.linalg.svd(sol_2d, full_matrices=False)

            # Truncate to max_rank with SVD-based error control
            r_new = min(len(S), max_rank)
            # Additional truncation: drop negligible singular values
            if r_new > 1:
                s_thresh = S[0] * 1e-14
                while r_new > 1 and S[r_new - 1] < s_thresh:
                    r_new -= 1
            U = U[:, :r_new]
            S = S[:r_new]
            Vh = Vh[:r_new, :]

            # x[i] = U (left-orthogonal), x[j] = diag(S) @ Vh
            x[i] = U.reshape(rx_l, d, r_new)
            x[j] = (S[:, np.newaxis] * Vh).reshape(r_new, d, rx_r)

            # Update left projections (needed for next bond)
            PhiL[i + 1] = _update_phi_left(PhiL[i], x[i], A[i])
            PsiL[i + 1] = _update_psi_left(PsiL[i], x[i], b[i])

        # ── Right-to-left half-sweep (bonds n-2..0) ──────────────
        for bond in range(n - 2, -1, -1):
            i, j = bond, bond + 1
            rx_l = x[i].shape[0]
            rx_r = x[j].shape[2]

            # Merged operator and RHS
            A_merged = _merge_mpo_cores(A[i], A[j])
            b_merged = _merge_tt_cores(b[i], b[j])

            H_loc = _form_local_op(PhiL[i], A_merged, PhiR[j + 1])
            f_loc = _form_local_rhs(PsiL[i], b_merged, PsiR[j + 1])

            sol = np.linalg.solve(H_loc, f_loc)

            # Reshape and SVD-split (absorb into LEFT this time)
            sol_2d = sol.reshape(rx_l * d, d * rx_r)
            U, S, Vh = np.linalg.svd(sol_2d, full_matrices=False)

            r_new = min(len(S), max_rank)
            if r_new > 1:
                s_thresh = S[0] * 1e-14
                while r_new > 1 and S[r_new - 1] < s_thresh:
                    r_new -= 1
            U = U[:, :r_new]
            S = S[:r_new]
            Vh = Vh[:r_new, :]

            # x[i] = U @ diag(S), x[j] = Vh (right-orthogonal)
            x[i] = (U * S[np.newaxis, :]).reshape(rx_l, d, r_new)
            x[j] = Vh.reshape(r_new, d, rx_r)

            # Update right projections (needed for next bond)
            PhiR[j] = _update_phi_right(PhiR[j + 1], x[j], A[j])
            PsiR[j] = _update_psi_right(PsiR[j + 1], x[j], b[j])

        # ── Check convergence ─────────────────────────────────────
        Ax = tt_matvec(A, x, max_rank=max_rank)
        res_cores = tt_axpy_c(-1.0, Ax, [c.copy() for c in b],
                              max_rank=max_rank)
        res_norm = tt_norm_c(res_cores)
        rel_res = res_norm / b_norm
        residual_norms.append(rel_res)

        if verbose:
            ranks = [c.shape[2] for c in x[:-1]]
            chi = max(ranks) if ranks else 1
            print(f"  DMRG sweep {sweep:3d}  rel_residual={rel_res:.4e}  "
                  f"chi_max={chi}")

        if rel_res < tol:
            return TTGMRESResult(
                x=x, residual_norms=residual_norms,
                converged=True, n_iter=sweep + 1,
                final_residual=rel_res,
            )

    return TTGMRESResult(
        x=x, residual_norms=residual_norms,
        converged=False, n_iter=n_sweeps,
        final_residual=residual_norms[-1] if residual_norms else float("inf"),
    )


# ======================================================================
# Section 7: Source Construction
# ======================================================================

def gaussian_source_tt(
    n_bits: int,
    h: float,
    position: float = 0.5,
    width: float = 0.02,
    max_rank: int = 64,
    cutoff: float = 1e-12,
) -> list[NDArray]:
    r"""Build Gaussian current source as QTT cores.

    .. math::

        J(x) = \frac{1}{\sqrt{2\pi}\,\sigma}
               \exp\!\left(-\frac{(x - x_0)^2}{2\sigma^2}\right)

    Returns ``-J`` (the Helmholtz RHS) in complex QTT format.

    Parameters
    ----------
    n_bits : int
        Number of QTT bits.
    h : float
        Grid spacing.
    position : float
        Source centre in [0, 1].
    width : float
        Gaussian width σ.
    max_rank : int
        Maximum QTT rank.
    cutoff : float
        SVD cutoff.

    Returns
    -------
    list[NDArray]
        Complex QTT cores for ``-J``.
    """
    N = 2 ** n_bits
    x = np.linspace(h / 2, 1.0 - h / 2, N)
    J = np.exp(-((x - position) ** 2) / (2 * width ** 2))
    J = J / (np.sqrt(2 * np.pi) * width)
    J[0] = 0.0
    J[N - 1] = 0.0
    rhs = -J.astype(np.complex128)
    return array_to_tt(rhs, max_rank=max_rank, cutoff=cutoff)


# ======================================================================
# Section 8: High-Level Solver
# ======================================================================

@dataclass
class HelmholtzConfig:
    """Configuration for QTT Helmholtz solve."""

    n_bits: int = 14
    k: float = 2.0 * math.pi
    eps_r: float | NDArray = 1.0
    pml_cells: int = 20
    pml_sigma_max: float = 10.0
    damping: float = 0.01
    source_position: float = 0.5
    source_width: float = 0.02
    max_rank: int = 128
    solver: str = "dmrg"  # "dmrg" (two-site DMRG) or "gmres"
    dmrg_sweeps: int = 40
    gmres_restart: int = 30
    gmres_max_iter: int = 200
    tol: float = 1e-4
    verbose: bool = True


@dataclass
class HelmholtzResult:
    """Result of QTT Helmholtz solve."""

    E_cores: list[NDArray]
    converged: bool
    n_iter: int
    final_residual: float
    residual_norms: list[float]
    chi_max: int
    n_bits: int
    N: int
    solve_time_s: float
    setup_time_s: float
    mpo_bond_dim: int


def helmholtz_solve_1d(config: HelmholtzConfig) -> HelmholtzResult:
    r"""Solve 1-D Helmholtz equation entirely in QTT format.

    Solves :math:`(\nabla^2 + k^2\varepsilon_{\mathrm{pml}})\,E = -J`
    via two-site DMRG (default) or complex TT-GMRES.

    Parameters
    ----------
    config : HelmholtzConfig
        Solver configuration.

    Returns
    -------
    HelmholtzResult
    """
    n = config.n_bits
    N = 2 ** n
    h = 1.0 / N

    if config.verbose:
        print(f"  QTT Helmholtz 1D: N={N:,} (2^{n}), k={config.k:.4f}, "
              f"solver={config.solver}")

    t0 = time.perf_counter()

    # PML permittivity profile (damping regularises periodic-BC resonances)
    eps_pml_cores = build_pml_eps_profile(
        n,
        eps_r=config.eps_r,
        pml_cells=config.pml_cells,
        sigma_max=config.pml_sigma_max,
        damping=config.damping,
        max_rank=config.max_rank,
    )
    eps_ranks = [c.shape[2] for c in eps_pml_cores[:-1]]
    eps_chi = max(eps_ranks) if eps_ranks else 1

    # Helmholtz MPO
    H_cores = helmholtz_mpo_1d(
        n, config.k, h,
        eps_pml_cores=eps_pml_cores,
    )
    mpo_bond = max(
        max(c.shape[0] for c in H_cores[1:]),
        max(c.shape[3] for c in H_cores[:-1]),
    )

    # Source
    rhs_cores = gaussian_source_tt(
        n, h,
        position=config.source_position,
        width=config.source_width,
        max_rank=config.max_rank,
    )

    t_setup = time.perf_counter() - t0

    if config.verbose:
        print(f"  MPO bond dim: {mpo_bond} (Laplacian ≤5, ε_pml χ={eps_chi})")
        print(f"  Setup: {t_setup:.3f}s")

    # ── Solve ─────────────────────────────────────────────────────
    t1 = time.perf_counter()

    if config.solver == "dmrg":
        result = tt_amen_solve(
            H_cores, rhs_cores,
            max_rank=config.max_rank,
            n_sweeps=config.dmrg_sweeps,
            tol=config.tol,
            verbose=config.verbose,
        )
    elif config.solver == "gmres":
        result = tt_gmres_complex(
            H_cores, rhs_cores,
            max_iter=config.gmres_max_iter,
            restart=config.gmres_restart,
            tol=config.tol,
            max_rank=config.max_rank,
            verbose=config.verbose,
        )
    else:
        raise ValueError(f"Unknown solver: {config.solver!r} "
                         f"(expected 'dmrg' or 'gmres')")

    t_solve = time.perf_counter() - t1

    # Solution rank profile
    ranks = [1] + [c.shape[2] for c in result.x]
    chi_max = max(ranks)

    if config.verbose:
        status = "CONVERGED" if result.converged else "NOT CONVERGED"
        label = "sweeps" if config.solver == "dmrg" else "iterations"
        print(f"  {status} in {result.n_iter} {label}")
        print(f"  Final residual: {result.final_residual:.4e}")
        print(f"  Solution χ_max: {chi_max}")
        print(f"  Solve time: {t_solve:.3f}s")

    return HelmholtzResult(
        E_cores=result.x,
        converged=result.converged,
        n_iter=result.n_iter,
        final_residual=result.final_residual,
        residual_norms=result.residual_norms,
        chi_max=chi_max,
        n_bits=n,
        N=N,
        solve_time_s=t_solve,
        setup_time_s=t_setup,
        mpo_bond_dim=mpo_bond,
    )
