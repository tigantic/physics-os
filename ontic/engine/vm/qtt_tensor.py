"""QTT Physics VM — Tensor representation.

Thin wrapper around ``list[NDArray]`` QTT cores providing ergonomic
operations, dimension-tracking, and conversion utilities.  Delegates
all heavy computation to ``ontic.qtt.sparse_direct`` and
``ontic.qtt.eigensolvers``.

Core shape convention: ``(r_left, 2, r_right)`` for QTT (binary mode).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass
class QTTTensor:
    """Quantized Tensor-Train tensor.

    Wraps a ``list[NDArray]`` of 3-way cores in QTT (d=2) format with
    metadata about the dimensional layout.

    Parameters
    ----------
    cores : list[NDArray]
        Cores of shape ``(r_left, 2, r_right)``.
    bits_per_dim : tuple[int, ...]
        How the cores map to physical dimensions.
        ``sum(bits_per_dim) == len(cores)``.
    domain : tuple[tuple[float, float], ...]
        Physical domain bounds per dimension, e.g. ``((0.0, 1.0),)``.
    """

    cores: list[NDArray]
    bits_per_dim: tuple[int, ...] = ()
    domain: tuple[tuple[float, float], ...] = ()

    # ── construction helpers ────────────────────────────────────────

    def __post_init__(self) -> None:
        if not self.bits_per_dim:
            object.__setattr__(self, "bits_per_dim", (len(self.cores),))
        if not self.domain:
            n_dims = len(self.bits_per_dim)
            object.__setattr__(self, "domain", tuple((0.0, 1.0) for _ in range(n_dims)))
        total = sum(self.bits_per_dim)
        if total != len(self.cores):
            raise ValueError(
                f"bits_per_dim sums to {total} but got {len(self.cores)} cores"
            )
        for i, c in enumerate(self.cores):
            if c.ndim != 3 or c.shape[1] != 2:
                raise ValueError(
                    f"Core {i} has shape {c.shape}, expected (r_left, 2, r_right)"
                )

    @classmethod
    def from_function(
        cls,
        fn: Callable[..., NDArray],
        bits_per_dim: tuple[int, ...],
        domain: tuple[tuple[float, float], ...] | None = None,
        max_rank: int = 64,
        cutoff: float = 1e-12,
    ) -> QTTTensor:
        """Sample *fn* on a product grid and compress to QTT.

        Parameters
        ----------
        fn : callable
            Scalar function accepting one array argument per dimension.
            For 1-D: ``fn(x) -> NDArray[float]`` of length ``2**L``.
            For 2-D: ``fn(x, v) -> NDArray[float]`` of shape ``(Nx, Nv)``.
        bits_per_dim : tuple[int, ...]
            Bits per spatial dimension.
        domain : tuple of (lo, hi) pairs, optional
            Physical domain bounds per dimension.  Default ``(0, 1)`` each.
        max_rank, cutoff : int, float
            TT-SVD rounding parameters.
        """
        n_dims = len(bits_per_dim)
        if domain is None:
            domain = tuple((0.0, 1.0) for _ in range(n_dims))

        # Build grid points per dimension
        grids: list[NDArray] = []
        for d in range(n_dims):
            N = 2 ** bits_per_dim[d]
            lo, hi = domain[d]
            grids.append(np.linspace(lo, hi, N, endpoint=False))

        # Evaluate on product grid
        if n_dims == 1:
            values = np.asarray(fn(grids[0]), dtype=np.float64)
        elif n_dims == 2:
            xx, yy = np.meshgrid(grids[0], grids[1], indexing="ij")
            values = np.asarray(fn(xx, yy), dtype=np.float64)
        else:
            mesh = np.meshgrid(*grids, indexing="ij")
            values = np.asarray(fn(*mesh), dtype=np.float64)

        # Reshape to (2, 2, ..., 2) and TT-SVD
        total_bits = sum(bits_per_dim)
        values = values.reshape([2] * total_bits)
        cores = _tt_svd(values, max_rank=max_rank, cutoff=cutoff)
        return cls(cores=cores, bits_per_dim=bits_per_dim, domain=domain)

    @classmethod
    def zeros(cls, bits_per_dim: tuple[int, ...],
              domain: tuple[tuple[float, float], ...] | None = None) -> QTTTensor:
        """All-zeros QTT tensor (rank 1)."""
        if domain is None:
            domain = tuple((0.0, 1.0) for _ in range(len(bits_per_dim)))
        total = sum(bits_per_dim)
        cores: list[NDArray] = []
        for k in range(total):
            core = np.zeros((1, 2, 1), dtype=np.float64)
            cores.append(core)
        return cls(cores=cores, bits_per_dim=bits_per_dim, domain=domain)

    @classmethod
    def ones(cls, bits_per_dim: tuple[int, ...],
             domain: tuple[tuple[float, float], ...] | None = None) -> QTTTensor:
        """All-ones QTT tensor (rank 1)."""
        if domain is None:
            domain = tuple((0.0, 1.0) for _ in range(len(bits_per_dim)))
        total = sum(bits_per_dim)
        cores: list[NDArray] = []
        for k in range(total):
            core = np.ones((1, 2, 1), dtype=np.float64)
            cores.append(core)
        return cls(cores=cores, bits_per_dim=bits_per_dim, domain=domain)

    @classmethod
    def constant(cls, value: float, bits_per_dim: tuple[int, ...],
                 domain: tuple[tuple[float, float], ...] | None = None) -> QTTTensor:
        """Constant-value QTT tensor (rank 1)."""
        t = cls.ones(bits_per_dim, domain)
        t.cores[0] = t.cores[0] * value
        return t

    @classmethod
    def coordinate(
        cls,
        dim: int,
        bits_per_dim: tuple[int, ...],
        domain: tuple[tuple[float, float], ...] | None = None,
    ) -> QTTTensor:
        """QTT tensor representing the coordinate function ``x_dim``.

        For 1-D with ``bits = (L,)``: produces a QTT encoding the
        grid-point values ``[0, dx, 2*dx, ...]``.

        For multi-dimensional fields, it's constant in all dims
        except ``dim``.
        """
        if domain is None:
            domain = tuple((0.0, 1.0) for _ in range(len(bits_per_dim)))
        lo, hi = domain[dim]

        def fn(*arrays: NDArray) -> NDArray:
            return arrays[dim]

        return cls.from_function(fn, bits_per_dim, domain, max_rank=64)

    # ── properties ──────────────────────────────────────────────────

    @property
    def n_cores(self) -> int:
        return len(self.cores)

    @property
    def n_dims(self) -> int:
        return len(self.bits_per_dim)

    @property
    def ranks(self) -> list[int]:
        """Bond dimensions: ``[r_0, r_1, ..., r_L]`` (L+1 values)."""
        r = [self.cores[0].shape[0]]
        for c in self.cores:
            r.append(c.shape[2])
        return r

    @property
    def max_rank(self) -> int:
        return max(self.ranks)

    @property
    def numel_compressed(self) -> int:
        return sum(c.size for c in self.cores)

    @property
    def numel_full(self) -> int:
        return 2 ** self.n_cores

    @property
    def compression_ratio(self) -> float:
        nc = self.numel_compressed
        return self.numel_full / nc if nc > 0 else float("inf")

    # ── core TT operations ──────────────────────────────────────────

    def clone(self) -> QTTTensor:
        return QTTTensor(
            cores=[c.copy() for c in self.cores],
            bits_per_dim=self.bits_per_dim,
            domain=self.domain,
        )

    def norm(self) -> float:
        """Frobenius norm via environment contraction."""
        from ontic.qtt.eigensolvers import tt_norm
        return tt_norm(self.cores)

    def inner(self, other: QTTTensor) -> float:
        """Inner product ⟨self | other⟩."""
        from ontic.qtt.eigensolvers import tt_inner
        return tt_inner(self.cores, other.cores)

    def sum(self) -> float:
        """Sum of all elements (contraction with all-ones)."""
        vec = np.ones(2, dtype=np.float64)
        result = np.ones((1, 1), dtype=np.float64)
        for c in self.cores:
            # c: (r_l, 2, r_r) -> contract physical index with vec
            contracted = np.einsum("ijk,j->ik", c, vec)  # (r_l, r_r)
            result = result @ contracted
        return float(result.item())

    def scale(self, alpha: float) -> QTTTensor:
        """Return alpha * self (scales first core)."""
        from ontic.qtt.eigensolvers import tt_scale
        new_cores = tt_scale(self.cores, alpha)
        return QTTTensor(cores=new_cores, bits_per_dim=self.bits_per_dim,
                         domain=self.domain)

    def add(self, other: QTTTensor) -> QTTTensor:
        """Return self + other (block-diagonal stacking, rank-additive)."""
        from ontic.qtt.eigensolvers import tt_add
        new_cores = tt_add(self.cores, other.cores)
        return QTTTensor(cores=new_cores, bits_per_dim=self.bits_per_dim,
                         domain=self.domain)

    def sub(self, other: QTTTensor) -> QTTTensor:
        """Return self - other."""
        neg = other.scale(-1.0)
        return self.add(neg)

    def negate(self) -> QTTTensor:
        return self.scale(-1.0)

    def hadamard(self, other: QTTTensor) -> QTTTensor:
        """Elementwise (Hadamard) product via Kronecker on bond dims.

        Result rank = rank(self) * rank(other).  Caller should truncate.
        """
        if len(self.cores) != len(other.cores):
            raise ValueError("Hadamard requires same number of cores")
        new_cores: list[NDArray] = []
        for a, b in zip(self.cores, other.cores):
            # a: (ra_l, 2, ra_r), b: (rb_l, 2, rb_r)
            # result: (ra_l*rb_l, 2, ra_r*rb_r)
            c = np.einsum("ajb,cjd->acjbd", a, b)
            ra_l, rb_l = a.shape[0], b.shape[0]
            ra_r, rb_r = a.shape[2], b.shape[2]
            c = c.reshape(ra_l * rb_l, 2, ra_r * rb_r)
            new_cores.append(c)
        return QTTTensor(cores=new_cores, bits_per_dim=self.bits_per_dim,
                         domain=self.domain)

    def truncate(self, max_rank: int = 64, cutoff: float = 1e-12) -> QTTTensor:
        """TT-SVD rounding."""
        from ontic.qtt.sparse_direct import tt_round
        new_cores = tt_round(self.cores, max_rank=max_rank, cutoff=cutoff)
        return QTTTensor(cores=new_cores, bits_per_dim=self.bits_per_dim,
                         domain=self.domain)

    def to_dense(self) -> NDArray:
        """Contract all cores to a dense array.  Only feasible for small L."""
        total = self.n_cores
        if total > 22:
            raise ValueError(f"to_dense with {total} cores would require "
                             f"2^{total} elements — too large")
        result = self.cores[0]  # (1, 2, r1)
        for c in self.cores[1:]:
            # result: (1, 2, ..., 2, r_k),  c: (r_k, 2, r_{k+1})
            result = np.einsum("...i,ijk->...jk", result, c)
        return result.reshape(-1)

    # ── dimension-aware helpers ─────────────────────────────────────

    def dim_core_range(self, dim: int) -> tuple[int, int]:
        """Return (start, end) core indices for physical dimension *dim*."""
        start = sum(self.bits_per_dim[:dim])
        end = start + self.bits_per_dim[dim]
        return start, end

    def grid_spacing(self, dim: int) -> float:
        """Grid spacing h for dimension *dim*."""
        lo, hi = self.domain[dim]
        N = 2 ** self.bits_per_dim[dim]
        return (hi - lo) / N

    def integrate_along(self, dim: int) -> QTTTensor:
        """Partial summation (numerical integration) along *dim*.

        Contracts the cores for dimension *dim* with the all-ones vector,
        producing a QTT tensor with that dimension removed.

        Returns a tensor with ``n_dims - 1`` dimensions.
        """
        start, end = self.dim_core_range(dim)
        vec = np.ones(2, dtype=np.float64)

        # Contract the dim-cores down to a chain of matrices
        transfer = np.eye(self.cores[start].shape[0], dtype=np.float64)
        for k in range(start, end):
            contracted = np.einsum("ijk,j->ik", self.cores[k], vec)
            transfer = transfer @ contracted

        # Build new core list: keep non-dim cores, inject transfer matrix
        new_cores: list[NDArray] = []
        new_bits: list[int] = []
        new_domain: list[tuple[float, float]] = []

        dim_idx = 0
        for d in range(self.n_dims):
            s, e = self.dim_core_range(d)
            if d == dim:
                # Absorb transfer into the next available core
                continue
            for k in range(s, e):
                new_cores.append(self.cores[k].copy())
            new_bits.append(self.bits_per_dim[d])
            new_domain.append(self.domain[d])

        if len(new_cores) == 0:
            # All dims integrated: return scalar as 1-core tensor
            scalar = np.trace(transfer)
            c = np.array([[[scalar]]], dtype=np.float64)
            return QTTTensor(cores=[c], bits_per_dim=(1,),
                             domain=((0.0, 1.0),))

        # Absorb transfer into the first core after the removed dimension
        # Find which core to absorb into
        if dim == 0 and len(new_cores) > 0:
            # Transfer goes before the remaining cores
            new_cores[0] = np.einsum("ij,jkl->ikl", transfer, new_cores[0])
        elif dim == self.n_dims - 1 and len(new_cores) > 0:
            # Transfer goes after the remaining cores
            new_cores[-1] = np.einsum("ijk,kl->ijl", new_cores[-1], transfer)
        else:
            # Middle dimension: absorb into next core
            insert_idx = sum(
                self.bits_per_dim[d] for d in range(self.n_dims) if d < dim and d != dim
            )
            if insert_idx < len(new_cores):
                new_cores[insert_idx] = np.einsum(
                    "ij,jkl->ikl", transfer, new_cores[insert_idx]
                )
            else:
                new_cores[-1] = np.einsum("ijk,kl->ijl", new_cores[-1], transfer)

        h = self.grid_spacing(dim)
        # Scale by h (trapezoidal quadrature approximation)
        new_cores[0] = new_cores[0] * h

        return QTTTensor(
            cores=new_cores,
            bits_per_dim=tuple(new_bits),
            domain=tuple(new_domain),
        )

    def broadcast_to(self, target_bits: tuple[int, ...],
                     target_domain: tuple[tuple[float, float], ...],
                     source_dim: int) -> QTTTensor:
        """Broadcast a single-dimension tensor to a multi-dim layout.

        Pads with all-ones cores for dimensions other than *source_dim*.

        Parameters
        ----------
        target_bits : tuple[int, ...]
            ``bits_per_dim`` of the target layout.
        target_domain : tuple of (lo, hi) pairs
            Domain bounds for the target.
        source_dim : int
            Which target dimension this tensor represents.
        """
        if self.n_dims != 1:
            raise ValueError("broadcast_to requires a 1-D source tensor")
        if self.bits_per_dim[0] != target_bits[source_dim]:
            raise ValueError(
                f"Source has {self.bits_per_dim[0]} bits but target dim "
                f"{source_dim} has {target_bits[source_dim]}"
            )

        new_cores: list[NDArray] = []
        for d in range(len(target_bits)):
            if d == source_dim:
                new_cores.extend(c.copy() for c in self.cores)
            else:
                for _ in range(target_bits[d]):
                    new_cores.append(np.ones((1, 2, 1), dtype=np.float64))

        return QTTTensor(
            cores=new_cores,
            bits_per_dim=target_bits,
            domain=target_domain,
        )


# ── TT-SVD (internal) ──────────────────────────────────────────────

def _tt_svd(
    tensor: NDArray,
    max_rank: int = 64,
    cutoff: float = 1e-12,
) -> list[NDArray]:
    """TT-SVD decomposition of a tensor shaped ``(d_0, d_1, ..., d_{L-1})``."""
    shape = tensor.shape
    ndim = len(shape)
    cores: list[NDArray] = []
    C = tensor.reshape(shape[0], -1).astype(np.float64)
    r = 1

    for k in range(ndim - 1):
        C = C.reshape(r * shape[k], -1)
        U, S, Vh = np.linalg.svd(C, full_matrices=False)

        # Truncate
        new_r = len(S)
        if cutoff > 0:
            total_sq = np.sum(S ** 2)
            if total_sq > 0:
                cum = np.cumsum(S ** 2)
                keep = int(np.searchsorted(cum / total_sq, 1.0 - cutoff ** 2)) + 1
                new_r = min(new_r, keep)
        new_r = min(new_r, max_rank)
        new_r = max(new_r, 1)

        U = U[:, :new_r]
        S = S[:new_r]
        Vh = Vh[:new_r, :]

        cores.append(U.reshape(r, shape[k], new_r))
        C = np.diag(S) @ Vh
        r = new_r

    cores.append(C.reshape(r, shape[-1], 1))
    return cores
