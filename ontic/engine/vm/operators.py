"""QTT Physics VM — Operator library.

Constructs differential operators (gradient, Laplacian) as MPOs and
applies them to QTT tensors.  Uses analytic MPO constructions for
the shift operator (binary carry chain) and derives gradient /
Laplacian from it.

For multi-dimensional fields, operators act on a specified dimension
by tensoring with identity MPOs on the other dimensions.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .qtt_tensor import QTTTensor


# ======================================================================
# MPO primitives
# ======================================================================

def identity_mpo(n_sites: int) -> list[NDArray]:
    """Identity MPO (bond dim 1, d_in = d_out = 2)."""
    cores: list[NDArray] = []
    for _ in range(n_sites):
        core = np.zeros((1, 2, 2, 1), dtype=np.float64)
        core[0, 0, 0, 0] = 1.0
        core[0, 1, 1, 0] = 1.0
        cores.append(core)
    return cores


def _shift_right_mpo(n_sites: int) -> list[NDArray]:
    r"""Right-shift operator S₊: S₊[i, i+1] = 1 (open BC).

    Implemented via binary carry chain (bond dim 2).
    Carry propagates from LSB (core n-1) to MSB (core 0).

    MPO core convention: ``(r_left, d_out, d_in, r_right)``.
    Bond index carries the carry bit: 0 = no carry, 1 = carry.

    At the LSB (rightmost core), we always add 1 → carry_in = 1.
    """
    cores: list[NDArray] = []

    for k in range(n_sites):
        if k == n_sites - 1:
            # LSB: carry_in is always 1 (adding 1), no bond to right
            # Shape: (2, 2, 2, 1) — carry_out ∈ {0,1}
            core = np.zeros((2, 2, 2, 1), dtype=np.float64)
            # out=0: in=0 XOR 1=1, carry_out = 0 AND 1 = 0
            core[0, 0, 1, 0] = 1.0
            # out=1: in=1 XOR 1=0, carry_out = 1 AND 1 = 1
            core[1, 1, 0, 0] = 1.0
        elif k == 0:
            # MSB: no bond to the left (boundary r_left=1)
            # Carry arriving from the right
            # Shape: (1, 2, 2, 2) — carry_in from right
            core = np.zeros((1, 2, 2, 2), dtype=np.float64)
            # carry_in=0: identity (no carry, output = input)
            core[0, 0, 0, 0] = 1.0
            core[0, 1, 1, 0] = 1.0
            # carry_in=1: flip bit, carry is discarded (open BC)
            core[0, 1, 0, 1] = 1.0
            core[0, 0, 1, 1] = 1.0
        else:
            # Middle core: bond dim 2 on both sides
            # Shape: (2, 2, 2, 2) — carry_out on left, carry_in on right
            core = np.zeros((2, 2, 2, 2), dtype=np.float64)
            # carry_in=0 (from right): identity, carry_out=0
            core[0, 0, 0, 0] = 1.0
            core[0, 1, 1, 0] = 1.0
            # carry_in=1 (from right):
            #   out=0: in=1 (flip), carry_out=0 (0 AND 1=0)
            core[0, 0, 1, 1] = 1.0
            #   out=1: in=0 (flip), carry_out=1 (1 AND 1=1)
            core[1, 1, 0, 1] = 1.0

        cores.append(core)

    return cores


def _shift_left_mpo(n_sites: int) -> list[NDArray]:
    r"""Left-shift operator S₋: S₋[i+1, i] = 1 → transpose of S₊.

    Swaps d_in ↔ d_out of the right-shift MPO.
    """
    right = _shift_right_mpo(n_sites)
    cores: list[NDArray] = []
    for c in right:
        # c: (r_l, d_out, d_in, r_r) → transpose to (r_l, d_in, d_out, r_r)
        cores.append(c.transpose(0, 2, 1, 3).copy())
    return cores


def mpo_add(a: list[NDArray], b: list[NDArray]) -> list[NDArray]:
    """Sum of two MPOs via block-diagonal stacking on bond indices.

    Result bond dim = bond_a + bond_b.
    """
    n = len(a)
    assert len(b) == n, "MPOs must have the same number of sites"
    cores: list[NDArray] = []

    for k in range(n):
        ac, bc = a[k], b[k]
        ra_l, d_out, d_in, ra_r = ac.shape
        rb_l, _, _, rb_r = bc.shape

        if k == 0:
            # First core: concatenate along right bond
            core = np.zeros((1, d_out, d_in, ra_r + rb_r), dtype=np.float64)
            core[0, :, :, :ra_r] = ac[0]
            core[0, :, :, ra_r:] = bc[0]
        elif k == n - 1:
            # Last core: concatenate along left bond
            core = np.zeros((ra_l + rb_l, d_out, d_in, 1), dtype=np.float64)
            core[:ra_l, :, :, 0] = ac[:, :, :, 0]
            core[ra_l:, :, :, 0] = bc[:, :, :, 0]
        else:
            # Middle: block-diagonal
            core = np.zeros((ra_l + rb_l, d_out, d_in, ra_r + rb_r),
                            dtype=np.float64)
            core[:ra_l, :, :, :ra_r] = ac
            core[ra_l:, :, :, ra_r:] = bc
        cores.append(core)

    return cores


def mpo_scale(mpo_cores: list[NDArray], alpha: float) -> list[NDArray]:
    """Scale an MPO by a scalar (multiplied into the first core)."""
    out = [c.copy() for c in mpo_cores]
    out[0] = out[0] * alpha
    return out


def mpo_apply(
    mpo_cores: list[NDArray],
    tt: QTTTensor,
    max_rank: int = 64,
    cutoff: float = 1e-12,
) -> QTTTensor:
    """Apply an MPO to a QTT tensor and round the result.

    Uses ``ontic.qtt.sparse_direct.tt_matvec`` for the contraction
    and ``tt_round`` for compression.
    """
    from ontic.qtt.sparse_direct import tt_matvec
    new_cores = tt_matvec(mpo_cores, tt.cores, max_rank=max_rank, cutoff=cutoff)
    return QTTTensor(cores=new_cores, bits_per_dim=tt.bits_per_dim,
                     domain=tt.domain)


# ======================================================================
# Differential operator MPOs
# ======================================================================

def gradient_mpo_1d(n_bits: int, h: float) -> list[NDArray]:
    """Central-difference gradient MPO: D₁ = (S₊ − S₋) / (2h).

    Bond dimension ≤ 4 (sum of two rank-2 MPOs).
    """
    sp = _shift_right_mpo(n_bits)
    sm = _shift_left_mpo(n_bits)
    alpha = 1.0 / (2.0 * h)
    return mpo_add(mpo_scale(sp, alpha), mpo_scale(sm, -alpha))


def laplacian_mpo_1d(n_bits: int, h: float) -> list[NDArray]:
    """Second-order central-difference Laplacian: D₂ = (S₊ - 2I + S₋) / h².

    Bond dimension ≤ 5 (sum of three MPOs: rank 2 + 1 + 2).
    """
    sp = _shift_right_mpo(n_bits)
    sm = _shift_left_mpo(n_bits)
    I = identity_mpo(n_bits)
    alpha = 1.0 / (h * h)
    terms = mpo_add(mpo_scale(sp, alpha), mpo_scale(sm, alpha))
    return mpo_add(terms, mpo_scale(I, -2.0 * alpha))


# ======================================================================
# Multi-dimensional operators
# ======================================================================

def _embed_1d_mpo(
    mpo_1d: list[NDArray],
    dim: int,
    bits_per_dim: tuple[int, ...],
) -> list[NDArray]:
    """Embed a 1-D MPO into a specific dimension of a multi-dim QTT field.

    Other dimensions get identity MPO cores.

    Parameters
    ----------
    mpo_1d : list[NDArray]
        1-D MPO cores of length ``bits_per_dim[dim]``.
    dim : int
        Target dimension.
    bits_per_dim : tuple[int, ...]
        Bits per dimension of the full field.
    """
    full_cores: list[NDArray] = []
    for d in range(len(bits_per_dim)):
        if d == dim:
            full_cores.extend(mpo_1d)
        else:
            full_cores.extend(identity_mpo(bits_per_dim[d]))
    return full_cores


def gradient_mpo(
    dim: int,
    bits_per_dim: tuple[int, ...],
    domain: tuple[tuple[float, float], ...],
) -> list[NDArray]:
    """Gradient MPO ∂/∂x_dim for a multi-dimensional QTT field."""
    lo, hi = domain[dim]
    N = 2 ** bits_per_dim[dim]
    h = (hi - lo) / N
    mpo_1d = gradient_mpo_1d(bits_per_dim[dim], h)
    if len(bits_per_dim) == 1:
        return mpo_1d
    return _embed_1d_mpo(mpo_1d, dim, bits_per_dim)


def laplacian_mpo(
    bits_per_dim: tuple[int, ...],
    domain: tuple[tuple[float, float], ...],
    dim: int | None = None,
) -> list[NDArray]:
    """Laplacian MPO ∇² for a multi-dimensional QTT field.

    If *dim* is ``None`` (full Laplacian), sums over all dimensions.
    If *dim* is specified, returns the 1-D Laplacian along that dimension.
    """
    if dim is not None:
        lo, hi = domain[dim]
        N = 2 ** bits_per_dim[dim]
        h = (hi - lo) / N
        mpo_1d = laplacian_mpo_1d(bits_per_dim[dim], h)
        if len(bits_per_dim) == 1:
            return mpo_1d
        return _embed_1d_mpo(mpo_1d, dim, bits_per_dim)

    # Full Laplacian: sum of per-dimension Laplacians
    full: list[NDArray] | None = None
    for d in range(len(bits_per_dim)):
        lo, hi = domain[d]
        N = 2 ** bits_per_dim[d]
        h = (hi - lo) / N
        mpo_1d = laplacian_mpo_1d(bits_per_dim[d], h)
        embedded = _embed_1d_mpo(mpo_1d, d, bits_per_dim) if len(bits_per_dim) > 1 else mpo_1d
        if full is None:
            full = embedded
        else:
            full = mpo_add(full, embedded)
    assert full is not None
    return full


# ======================================================================
# Operator cache (avoids rebuilding MPOs every time step)
# ======================================================================

class OperatorCache:
    """Caches compiled MPO operators for a given grid configuration.

    Keyed by ``(operator_name, dim, bits_per_dim, domain)``.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[Any, ...], list[NDArray]] = {}

    def get_gradient(
        self,
        dim: int,
        bits_per_dim: tuple[int, ...],
        domain: tuple[tuple[float, float], ...],
    ) -> list[NDArray]:
        key = ("grad", dim, bits_per_dim, domain)
        if key not in self._cache:
            self._cache[key] = gradient_mpo(dim, bits_per_dim, domain)
        return self._cache[key]

    def get_laplacian(
        self,
        bits_per_dim: tuple[int, ...],
        domain: tuple[tuple[float, float], ...],
        dim: int | None = None,
    ) -> list[NDArray]:
        key = ("laplace", dim, bits_per_dim, domain)
        if key not in self._cache:
            self._cache[key] = laplacian_mpo(bits_per_dim, domain, dim)
        return self._cache[key]

    def clear(self) -> None:
        self._cache.clear()


# ======================================================================
# Poisson solver (CG in QTT format)
# ======================================================================

def poisson_solve(
    rhs: QTTTensor,
    dim: int | None = None,
    max_rank: int = 64,
    cutoff: float = 1e-10,
    max_iter: int = 80,
    tol: float = 1e-8,
) -> QTTTensor:
    """Solve ∇²φ = rhs via CG entirely in QTT format.

    Parameters
    ----------
    rhs : QTTTensor
        Right-hand side.
    dim : int | None
        If specified, solve along this dimension only.
    max_rank : int
        Maximum rank during CG iterations.
    cutoff : float
        SVD cutoff.
    max_iter : int
        Maximum CG iterations.
    tol : float
        Convergence tolerance on the residual norm.

    Returns
    -------
    QTTTensor
        Approximate solution φ.
    """
    from ontic.qtt.eigensolvers import tt_inner, tt_axpy
    from ontic.qtt.sparse_direct import tt_matvec, tt_round

    lap = laplacian_mpo(rhs.bits_per_dim, rhs.domain, dim=dim)

    # Initial guess: zero
    x = QTTTensor.zeros(rhs.bits_per_dim, rhs.domain)

    # r = rhs - A*x = rhs (since x=0)
    r = rhs.clone()
    p = r.clone()
    rs_old = tt_inner(r.cores, r.cores)

    if rs_old < tol * tol:
        return x

    for _ in range(max_iter):
        # Ap = L * p
        Ap_cores = tt_matvec(lap, p.cores, max_rank=max_rank, cutoff=cutoff)
        pAp = tt_inner(p.cores, Ap_cores)
        if abs(pAp) < 1e-30:
            break

        alpha = rs_old / pAp

        # x = x + alpha * p
        x_cores = tt_axpy(alpha, p.cores, x.cores, max_rank=max_rank)
        x = QTTTensor(cores=x_cores, bits_per_dim=rhs.bits_per_dim,
                       domain=rhs.domain)

        # r = r - alpha * Ap
        Ap_tensor = QTTTensor(cores=Ap_cores, bits_per_dim=rhs.bits_per_dim,
                              domain=rhs.domain)
        r_cores = tt_axpy(-alpha, Ap_tensor.cores, r.cores, max_rank=max_rank)
        r = QTTTensor(cores=r_cores, bits_per_dim=rhs.bits_per_dim,
                       domain=rhs.domain)

        rs_new = tt_inner(r.cores, r.cores)
        if rs_new < tol * tol:
            break

        beta = rs_new / rs_old
        p_cores = tt_axpy(beta, p.cores, r.cores, max_rank=max_rank)
        p = QTTTensor(cores=p_cores, bits_per_dim=rhs.bits_per_dim,
                       domain=rhs.domain)
        rs_old = rs_new

    return x
