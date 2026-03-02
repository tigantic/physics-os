"""QTT Physics VM — Operator library.

Constructs differential operators (gradient, Laplacian) as MPOs and
applies them to QTT tensors.  Uses analytic MPO constructions for
the shift operator (binary carry chain) and derives gradient /
Laplacian from it.

For multi-dimensional fields, operators act on a specified dimension
by tensoring with identity MPOs on the other dimensions.

Operator variants
-----------------
Every MPO family carries a version tag (``OperatorVariant``) that
identifies the stencil order and construction method.  This is exposed
in the benchmark registry as ``operator_variant`` so convergence
studies can sweep over discretization quality.

Supported families and variants:

+----------+------------------+-------------------------------------------+
| Family   | Variant          | Description                               |
+==========+==================+===========================================+
| grad     | grad_v1          | 2nd-order central difference (S₊−S₋)/2h  |
|          | grad_v2_high     | 4th-order central difference              |
+----------+------------------+-------------------------------------------+
| lap      | lap_v1           | 2nd-order (S₊−2I+S₋)/h²                  |
|          | lap_v2_high      | 4th-order (−S₊₊+16S₊−30I+16S₋−S₋₋)/12h² |
+----------+------------------+-------------------------------------------+
| elliptic | elliptic_var_v1  | Variable-coefficient ∇·(a∇u) composite   |
+----------+------------------+-------------------------------------------+
"""

from __future__ import annotations

import enum
import functools
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .qtt_tensor import QTTTensor


# ======================================================================
# Operator family / variant registry
# ======================================================================

class OperatorFamily(enum.Enum):
    """Enumeration of MPO operator families."""
    GRADIENT = "grad"
    LAPLACIAN = "lap"
    ELLIPTIC = "elliptic"
    IDENTITY = "identity"
    SHIFT = "shift"


@dataclass(frozen=True)
class OperatorVariant:
    """Versioned identifier for an MPO construction.

    Parameters
    ----------
    family : OperatorFamily
        Which operator family this belongs to.
    tag : str
        Machine-readable tag (e.g. ``"grad_v1"``, ``"lap_v2_high_order"``).
    order : int
        Formal accuracy order of the stencil.
    description : str
        Human-readable description.
    """
    family: OperatorFamily
    tag: str
    order: int
    description: str


# Canonical variant registry (append-only)
OPERATOR_VARIANTS: dict[str, OperatorVariant] = {
    "grad_v1": OperatorVariant(
        family=OperatorFamily.GRADIENT,
        tag="grad_v1",
        order=2,
        description="2nd-order central difference gradient: (S₊ − S₋) / (2h)",
    ),
    "grad_v2_high_order": OperatorVariant(
        family=OperatorFamily.GRADIENT,
        tag="grad_v2_high_order",
        order=4,
        description="4th-order central difference gradient: "
                    "(-S₊₊ + 8S₊ - 8S₋ + S₋₋) / (12h)",
    ),
    "lap_v1": OperatorVariant(
        family=OperatorFamily.LAPLACIAN,
        tag="lap_v1",
        order=2,
        description="2nd-order central difference Laplacian: (S₊ - 2I + S₋) / h²",
    ),
    "lap_v2_high_order": OperatorVariant(
        family=OperatorFamily.LAPLACIAN,
        tag="lap_v2_high_order",
        order=4,
        description="4th-order central difference Laplacian: "
                    "(-S₊₊ + 16S₊ - 30I + 16S₋ - S₋₋) / (12h²)",
    ),
    "elliptic_var_v1": OperatorVariant(
        family=OperatorFamily.ELLIPTIC,
        tag="elliptic_var_v1",
        order=2,
        description="Variable-coefficient elliptic: ∇·(a∇u) as MPO pipeline",
    ),
}


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


def mpo_compose(a: list[NDArray], b: list[NDArray]) -> list[NDArray]:
    """Compose two MPOs: result = A · B (apply A after B).

    Core contraction sums over the intermediate physical index m:

        C_k[α_l⊗β_l, j, i, α_r⊗β_r] = Σ_m A_k[α_l, j, m, α_r] · B_k[β_l, m, i, β_r]

    Result bond dimension = bond_A × bond_B at each site.
    """
    n = len(a)
    assert len(b) == n, "MPOs must have the same number of sites"
    cores: list[NDArray] = []
    for k in range(n):
        ac, bc = a[k], b[k]
        ra_l, d_out, d_mid, ra_r = ac.shape
        rb_l, _d_mid2, d_in, rb_r = bc.shape
        assert d_mid == _d_mid2, f"Physical dim mismatch at site {k}"
        # Contract over d_mid
        # ac: [al, j, m, ar],  bc: [bl, m, i, br]
        # result: [al, bl, j, i, ar, br]
        C = np.einsum("ajmb,cmid->acjibd", ac, bc)
        core = C.reshape(ra_l * rb_l, d_out, d_in, ra_r * rb_r)
        cores.append(core)
    return cores


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
# 4th-order MPO variants
# ======================================================================

def _shift_double_right_mpo(n_bits: int) -> list[NDArray]:
    """Double right-shift S₊² = S₊ · S₊ (shift by +2 grid points).

    Constructed via MPO composition of two single shifts.
    """
    sp = _shift_right_mpo(n_bits)
    return mpo_compose(sp, sp)


def _shift_double_left_mpo(n_bits: int) -> list[NDArray]:
    """Double left-shift S₋² = S₋ · S₋ (shift by −2 grid points).

    Constructed via MPO composition of two single shifts.
    """
    sm = _shift_left_mpo(n_bits)
    return mpo_compose(sm, sm)


def gradient_mpo_1d_4th(n_bits: int, h: float) -> list[NDArray]:
    r"""4th-order central-difference gradient MPO.

    D₁⁴ = (−S₊² + 8S₊ − 8S₋ + S₋²) / (12h)

    Formal accuracy order: 4. Bond dimension ≤ 4 + 4 + 4 + 4 = 16
    (before rounding).  The shift-double MPOs have bond dim ≤ 4.
    """
    sp = _shift_right_mpo(n_bits)
    sm = _shift_left_mpo(n_bits)
    sp2 = _shift_double_right_mpo(n_bits)
    sm2 = _shift_double_left_mpo(n_bits)

    inv12h = 1.0 / (12.0 * h)
    result = mpo_add(mpo_scale(sp2, -inv12h), mpo_scale(sp, 8.0 * inv12h))
    result = mpo_add(result, mpo_scale(sm, -8.0 * inv12h))
    result = mpo_add(result, mpo_scale(sm2, inv12h))
    return result


def laplacian_mpo_1d_4th(n_bits: int, h: float) -> list[NDArray]:
    r"""4th-order central-difference Laplacian MPO.

    D₂⁴ = (−S₊² + 16S₊ − 30I + 16S₋ − S₋²) / (12h²)

    Formal accuracy order: 4. Bond dimension ≤ 4 + 2 + 1 + 2 + 4 = 13
    (before rounding).
    """
    sp = _shift_right_mpo(n_bits)
    sm = _shift_left_mpo(n_bits)
    sp2 = _shift_double_right_mpo(n_bits)
    sm2 = _shift_double_left_mpo(n_bits)
    I = identity_mpo(n_bits)

    inv12h2 = 1.0 / (12.0 * h * h)
    result = mpo_add(mpo_scale(sp2, -inv12h2), mpo_scale(sp, 16.0 * inv12h2))
    result = mpo_add(result, mpo_scale(I, -30.0 * inv12h2))
    result = mpo_add(result, mpo_scale(sm, 16.0 * inv12h2))
    result = mpo_add(result, mpo_scale(sm2, -inv12h2))
    return result


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
    variant: str = "grad_v1",
) -> list[NDArray]:
    """Gradient MPO ∂/∂x_dim for a multi-dimensional QTT field.

    Parameters
    ----------
    dim : int
        Dimension along which to differentiate.
    bits_per_dim : tuple[int, ...]
        Bits per spatial dimension.
    domain : tuple[tuple[float, float], ...]
        Physical domain bounds per dimension.
    variant : str
        Operator variant tag: ``"grad_v1"`` (2nd order) or
        ``"grad_v2_high_order"`` (4th order).
    """
    lo, hi = domain[dim]
    N = 2 ** bits_per_dim[dim]
    h = (hi - lo) / N
    if variant == "grad_v2_high_order":
        mpo_1d = gradient_mpo_1d_4th(bits_per_dim[dim], h)
    else:
        mpo_1d = gradient_mpo_1d(bits_per_dim[dim], h)
    if len(bits_per_dim) == 1:
        return mpo_1d
    return _embed_1d_mpo(mpo_1d, dim, bits_per_dim)


def laplacian_mpo(
    bits_per_dim: tuple[int, ...],
    domain: tuple[tuple[float, float], ...],
    dim: int | None = None,
    variant: str = "lap_v1",
) -> list[NDArray]:
    """Laplacian MPO ∇² for a multi-dimensional QTT field.

    If *dim* is ``None`` (full Laplacian), sums over all dimensions.
    If *dim* is specified, returns the 1-D Laplacian along that dimension.

    Parameters
    ----------
    bits_per_dim : tuple[int, ...]
        Bits per spatial dimension.
    domain : tuple[tuple[float, float], ...]
        Physical domain bounds per dimension.
    dim : int | None
        Target dimension (None = full Laplacian).
    variant : str
        Operator variant tag: ``"lap_v1"`` (2nd order) or
        ``"lap_v2_high_order"`` (4th order).
    """
    build_1d = laplacian_mpo_1d_4th if variant == "lap_v2_high_order" else laplacian_mpo_1d

    if dim is not None:
        lo, hi = domain[dim]
        N = 2 ** bits_per_dim[dim]
        h = (hi - lo) / N
        mpo_1d = build_1d(bits_per_dim[dim], h)
        if len(bits_per_dim) == 1:
            return mpo_1d
        return _embed_1d_mpo(mpo_1d, dim, bits_per_dim)

    # Full Laplacian: sum of per-dimension Laplacians
    full: list[NDArray] | None = None
    for d in range(len(bits_per_dim)):
        lo, hi = domain[d]
        N = 2 ** bits_per_dim[d]
        h = (hi - lo) / N
        mpo_1d = build_1d(bits_per_dim[d], h)
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

    Keyed by ``(operator_name, variant, dim, bits_per_dim, domain)``.
    Variant-aware: callers specify which stencil order to use.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[Any, ...], list[NDArray]] = {}

    def get_gradient(
        self,
        dim: int,
        bits_per_dim: tuple[int, ...],
        domain: tuple[tuple[float, float], ...],
        variant: str = "grad_v1",
    ) -> list[NDArray]:
        key = ("grad", variant, dim, bits_per_dim, domain)
        if key not in self._cache:
            self._cache[key] = gradient_mpo(dim, bits_per_dim, domain,
                                            variant=variant)
        return self._cache[key]

    def get_laplacian(
        self,
        bits_per_dim: tuple[int, ...],
        domain: tuple[tuple[float, float], ...],
        dim: int | None = None,
        variant: str = "lap_v1",
    ) -> list[NDArray]:
        key = ("laplace", variant, dim, bits_per_dim, domain)
        if key not in self._cache:
            self._cache[key] = laplacian_mpo(bits_per_dim, domain, dim,
                                             variant=variant)
        return self._cache[key]
        return self._cache[key]

    def clear(self) -> None:
        self._cache.clear()


def get_variant_info(tag: str) -> OperatorVariant:
    """Look up a registered operator variant by tag.

    Raises ``KeyError`` if the tag is not registered.
    """
    return OPERATOR_VARIANTS[tag]


# ======================================================================
# Variable-coefficient elliptic operator: ∇·(a∇u)
# ======================================================================

def variable_coeff_elliptic_apply(
    coeff_field: QTTTensor,
    u: QTTTensor,
    bits_per_dim: tuple[int, ...],
    domain: tuple[tuple[float, float], ...],
    max_rank: int = 64,
    cutoff: float = 1e-12,
    grad_variant: str = "grad_v1",
) -> QTTTensor:
    """Apply ∇·(a∇u) as a composable MPO pipeline.

    Computes the variable-coefficient elliptic operator by:
    1. Compute ∇u via gradient MPO (per dimension)
    2. Multiply by coefficient field a(x): a ⊙ ∇u (Hadamard)
    3. Apply divergence via gradient MPO transpose

    For 1D: d/dx(a · du/dx)
    For multi-D: Σ_d ∂/∂x_d (a · ∂u/∂x_d)

    Parameters
    ----------
    coeff_field : QTTTensor
        The coefficient field a(x) in QTT format.
    u : QTTTensor
        The field to operate on.
    bits_per_dim : tuple[int, ...]
        Bits per spatial dimension.
    domain : tuple[tuple[float, float], ...]
        Physical domain bounds.
    max_rank : int
        Maximum bond dimension for intermediate truncations.
    cutoff : float
        rSVD cutoff for truncations.
    grad_variant : str
        Which gradient variant to use (``"grad_v1"`` or ``"grad_v2_high_order"``).

    Returns
    -------
    QTTTensor
        Result of ∇·(a∇u) in QTT format.
    """
    n_dims = len(bits_per_dim)
    result: QTTTensor | None = None

    for d in range(n_dims):
        # Step 1: ∂u/∂x_d
        grad_op = gradient_mpo(d, bits_per_dim, domain, variant=grad_variant)
        du = mpo_apply(grad_op, u, max_rank=max_rank, cutoff=cutoff)

        # Step 2: a(x) · ∂u/∂x_d (Hadamard product)
        a_du = coeff_field.hadamard(du)
        a_du = a_du.truncate(max_rank=max_rank, cutoff=cutoff)

        # Step 3: ∂/∂x_d (a · ∂u/∂x_d) — apply gradient again for divergence
        div_op = gradient_mpo(d, bits_per_dim, domain, variant=grad_variant)
        term = mpo_apply(div_op, a_du, max_rank=max_rank, cutoff=cutoff)

        # Accumulate
        if result is None:
            result = term
        else:
            from ontic.qtt.sparse_direct import tt_round
            added_cores = [
                np.concatenate([rc, tc], axis=0) if i == 0 else
                np.block([[rc, np.zeros_like(tc)],
                          [np.zeros_like(rc), tc]]) if 0 < i < len(result.cores) - 1 else
                np.concatenate([rc, tc], axis=0)
                for i, (rc, tc) in enumerate(zip(result.cores, term.cores))
            ]
            # Use proper TT addition + rounding
            from ontic.qtt.sparse_direct import tt_axpy
            added_cores = tt_axpy(1.0, term.cores, result.cores,
                                  max_rank=max_rank)
            result = QTTTensor(cores=added_cores,
                               bits_per_dim=bits_per_dim,
                               domain=domain)

    assert result is not None, "variable_coeff_elliptic_apply requires ≥1 dimension"
    return result


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
    variant: str = "lap_v1",
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
    variant : str
        Laplacian variant tag for the system matrix.

    Returns
    -------
    QTTTensor
        Approximate solution φ.
    """
    from ontic.qtt.eigensolvers import tt_inner, tt_axpy
    from ontic.qtt.sparse_direct import tt_matvec, tt_round

    lap = laplacian_mpo(rhs.bits_per_dim, rhs.domain, dim=dim, variant=variant)

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
