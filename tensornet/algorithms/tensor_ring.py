"""
Tensor Ring (TR) Decomposition
================================

Periodic boundary tensor-train: the first and last cores share a bond,
eliminating edge bias of open-boundary TT.

:math:`T(i_1,\\dots,i_N) = \\operatorname{Tr}\\bigl[G_1(i_1) \\cdots G_N(i_N)\\bigr]`

Key classes
-----------
* :class:`TensorRing` — core data structure with periodic trace contraction
* :func:`random_tensor_ring` — random initialisation
* :func:`tensor_to_tr` — dense tensor → TR via periodic SVD
* :func:`tr_add` / :func:`tr_hadamard` — ring algebra
* :func:`tr_round` — re-compression via cyclic SVD sweep
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray


# ======================================================================
# Core data structure
# ======================================================================

@dataclass
class TensorRing:
    """
    Tensor Ring decomposition.

    Stores a list of 3-index cores ``G_k`` of shape
    ``(r_{k-1}, n_k, r_k)`` where ``r_0 == r_N`` (periodic bond).

    Attributes
    ----------
    cores : list[NDArray]
        Core tensors.  ``cores[k].shape == (r_{k-1}, n_k, r_k)``.
    """
    cores: list[NDArray]

    # ------------------------------------------------------------------
    @property
    def order(self) -> int:
        """Number of modes (N)."""
        return len(self.cores)

    @property
    def mode_dims(self) -> list[int]:
        """Physical dimensions (n_1, ..., n_N)."""
        return [c.shape[1] for c in self.cores]

    @property
    def ranks(self) -> list[int]:
        """Bond dimensions (r_0, r_1, ..., r_{N-1}).  r_0 == r_{N}."""
        return [c.shape[0] for c in self.cores]

    # ------------------------------------------------------------------
    # Dense reconstruction
    # ------------------------------------------------------------------
    def to_tensor(self) -> NDArray:
        """
        Contract to full dense tensor by taking trace over the periodic bond.

        .. warning:: Exponential in order — testing only.
        """
        N = self.order
        # Build product of transfer matrices for every index combo
        dims = self.mode_dims
        size = int(np.prod(dims))
        r0 = self.cores[0].shape[0]

        result = np.zeros(dims, dtype=self.cores[0].dtype)
        for flat_idx in range(size):
            idx = np.unravel_index(flat_idx, dims)
            mat = np.eye(r0, dtype=self.cores[0].dtype)
            for k in range(N):
                mat = mat @ self.cores[k][:, idx[k], :]
            result[idx] = np.trace(mat)
        return result

    # ------------------------------------------------------------------
    # Norm (Frobenius via trace transfer matrix)
    # ------------------------------------------------------------------
    def norm(self) -> float:
        r"""
        Frobenius norm :math:`\|T\|_F` computed via transfer-matrix trace.

        Cost: :math:`O(N r^4 n)`.
        """
        N = self.order
        # Transfer matrix: T = Σ_i G_k(i) ⊗ G_k(i)*
        r0 = self.cores[0].shape[0]
        E = np.eye(r0 * r0, dtype=self.cores[0].dtype).reshape(r0, r0, r0, r0)

        for k in range(N):
            G = self.cores[k]  # (r_left, n, r_right)
            r_l, n, r_r = G.shape
            # Transfer: E'[a,b,c,d] = sum_{i} E[a,b,e,f] G[e,i,c] G*[f,i,d]
            Enew = np.zeros((r0, r0, r_r, r_r), dtype=G.dtype)
            if k == 0:
                for i in range(n):
                    Gi = G[:, i, :]      # (r_l, r_r)
                    Gic = G[:, i, :].conj()
                    Enew += np.einsum('ac,bd->abcd', Gi, Gic)
            else:
                for i in range(n):
                    Gi = G[:, i, :]
                    Gic = G[:, i, :].conj()
                    Enew += np.einsum('abef,ec,fd->abcd', E, Gi, Gic)
            E = Enew
            r0_next = r_r
            # For the next iteration, the "open" indices are (a, b) from start, (c, d) = r_r
            # But we keep tracking the first two indices (a, b) of size r0

        # Trace: sum_{a,c} E[a,a,c,c]
        val = np.einsum('aacc->', E)
        return float(np.sqrt(np.abs(val)))

    # ------------------------------------------------------------------
    # Point evaluation
    # ------------------------------------------------------------------
    def evaluate(self, index: Sequence[int]) -> complex:
        """Evaluate at a multi-index ``(i_1, ..., i_N)``."""
        r0 = self.cores[0].shape[0]
        mat = np.eye(r0, dtype=self.cores[0].dtype)
        for k, ik in enumerate(index):
            mat = mat @ self.cores[k][:, ik, :]
        return complex(np.trace(mat))

    # ------------------------------------------------------------------
    # Copy
    # ------------------------------------------------------------------
    def copy(self) -> "TensorRing":
        return TensorRing(cores=[c.copy() for c in self.cores])


# ======================================================================
# Constructors
# ======================================================================

def random_tensor_ring(
    dims: Sequence[int],
    rank: int = 4,
    seed: Optional[int] = None,
) -> TensorRing:
    """
    Random tensor ring with uniform bond dimension.

    Parameters
    ----------
    dims : sequence of int
        Mode dimensions ``(n_1, ..., n_N)``.
    rank : int
        Bond dimension (same for every bond, including periodic).
    seed : int, optional
        RNG seed.
    """
    rng = np.random.default_rng(seed)
    cores: list[NDArray] = []
    N = len(dims)
    for k in range(N):
        core = rng.standard_normal((rank, dims[k], rank))
        core /= np.linalg.norm(core) + 1e-30
        cores.append(core)
    return TensorRing(cores=cores)


def tensor_to_tr(
    tensor: NDArray,
    rank: int = 8,
    n_sweeps: int = 10,
    seed: Optional[int] = None,
) -> TensorRing:
    """
    Approximate a dense tensor by a Tensor Ring via alternating least-squares.

    Parameters
    ----------
    tensor : NDArray
        Dense tensor of shape ``(n_1, ..., n_N)``.
    rank : int
        Target bond dimension.
    n_sweeps : int
        Number of ALS sweeps.
    seed : int, optional
        RNG seed.

    Returns
    -------
    TensorRing
        Approximation with the specified rank.
    """
    dims = tensor.shape
    N = len(dims)
    rng = np.random.default_rng(seed)

    # Initialise random
    tr = random_tensor_ring(dims, rank=rank, seed=seed)

    for sweep in range(n_sweeps):
        for k in range(N):
            # Build the "environment" by contracting all cores except k
            # For ALS we fix all cores ≠ k and solve for core k
            # This is a linear least-squares problem
            # Build left partial product and right partial product

            # Left partial product: cores 0, 1, ..., k-1
            # Right partial product: cores k+1, ..., N-1
            r = rank
            n_k = dims[k]
            total_size = int(np.prod(dims))

            # For small tensors, direct ALS via Khatri-Rao style
            # Flatten tensor to vector
            b = tensor.ravel()

            # Build design matrix A such that A @ vec(G_k) ≈ b
            # Each row of A corresponds to one multi-index
            rows: list[NDArray] = []
            for flat_idx in range(total_size):
                idx = np.unravel_index(flat_idx, dims)
                # Product of all cores except k
                mat_left = np.eye(r, dtype=tensor.dtype)
                for j in range(k):
                    mat_left = mat_left @ tr.cores[j][:, idx[j], :]
                mat_right = np.eye(r, dtype=tensor.dtype)
                for j in range(N - 1, k, -1):
                    mat_right = tr.cores[j][:, idx[j], :] @ mat_right
                # The contribution to the trace from core k at index idx[k]:
                # Tr(mat_left @ G_k[:, idx[k], :] @ mat_right)
                # = vec(mat_right @ mat_left)^T @ vec(G_k[:, idx[k], :])
                # Build row for this specific idx[k]
                env = (mat_right @ mat_left).T.ravel()  # (r^2,)
                row = np.zeros(r * n_k * r, dtype=tensor.dtype)
                offset = idx[k] * r * r
                row[offset:offset + r * r] = env
                rows.append(row)

            A = np.array(rows)
            # Solve least squares
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            tr.cores[k] = x.reshape(r, n_k, r)

    return tr


# ======================================================================
# Arithmetic
# ======================================================================

def tr_add(a: TensorRing, b: TensorRing) -> TensorRing:
    """
    Add two tensor rings: result has bond dimension ``r_a + r_b``.

    The periodic structure is preserved via block-diagonal padding.
    """
    assert a.order == b.order
    cores: list[NDArray] = []
    N = a.order
    for k in range(N):
        Ga = a.cores[k]  # (ra_l, n, ra_r)
        Gb = b.cores[k]  # (rb_l, n, rb_r)
        ra_l, n, ra_r = Ga.shape
        rb_l, _, rb_r = Gb.shape
        C = np.zeros((ra_l + rb_l, n, ra_r + rb_r), dtype=Ga.dtype)
        C[:ra_l, :, :ra_r] = Ga
        C[ra_l:, :, ra_r:] = Gb
        cores.append(C)
    return TensorRing(cores=cores)


def tr_hadamard(a: TensorRing, b: TensorRing) -> TensorRing:
    """
    Element-wise (Hadamard) product.  Result rank = ``r_a * r_b``.
    """
    assert a.order == b.order
    cores: list[NDArray] = []
    for k in range(a.order):
        Ga = a.cores[k]  # (ra_l, n, ra_r)
        Gb = b.cores[k]  # (rb_l, n, rb_r)
        ra_l, n, ra_r = Ga.shape
        rb_l, _, rb_r = Gb.shape
        # Kronecker on bond indices, shared physical
        C = np.zeros((ra_l * rb_l, n, ra_r * rb_r), dtype=Ga.dtype)
        for i in range(n):
            C[:, i, :] = np.kron(Ga[:, i, :], Gb[:, i, :])
        cores.append(C)
    return TensorRing(cores=cores)


# ======================================================================
# Re-compression (cyclic SVD sweep)
# ======================================================================

def tr_round(
    tr: TensorRing,
    max_rank: int = 64,
    cutoff: float = 1e-14,
    n_sweeps: int = 2,
) -> TensorRing:
    """
    Re-compress a Tensor Ring by cyclic SVD sweeps.

    Each sweep performs left-to-right QR then right-to-left SVD truncation,
    handling the periodic bond by absorbing remainders cyclically.

    Parameters
    ----------
    tr : TensorRing
        Input ring (not modified).
    max_rank : int
        Maximum bond dimension after compression.
    cutoff : float
        SVD truncation tolerance.
    n_sweeps : int
        Number of compression sweeps.

    Returns
    -------
    TensorRing
        Compressed ring.
    """
    result = tr.copy()
    N = result.order

    for _ in range(n_sweeps):
        # Left-to-right QR
        for k in range(N - 1):
            G = result.cores[k]
            r_l, n, r_r = G.shape
            mat = G.reshape(r_l * n, r_r)
            Q, R = np.linalg.qr(mat)
            new_r = Q.shape[1]
            result.cores[k] = Q.reshape(r_l, n, new_r)
            # Absorb R into next core
            G_next = result.cores[k + 1]
            result.cores[k + 1] = np.tensordot(R, G_next, axes=([1], [0]))

        # Right-to-left SVD truncation
        for k in range(N - 1, 0, -1):
            G = result.cores[k]
            r_l, n, r_r = G.shape
            mat = G.reshape(r_l, n * r_r)
            U, S, Vh = np.linalg.svd(mat, full_matrices=False)
            keep = min(max_rank, np.sum(S > cutoff).item(), len(S))
            keep = max(keep, 1)
            result.cores[k] = (np.diag(S[:keep]) @ Vh[:keep]).reshape(keep, n, r_r)
            # Absorb U into previous core
            G_prev = result.cores[k - 1]
            r_pl, n_p, r_pr = G_prev.shape
            mat_prev = G_prev.reshape(r_pl * n_p, r_pr)
            mat_prev = mat_prev @ U[:, :keep]
            result.cores[k - 1] = mat_prev.reshape(r_pl, n_p, keep)

    return result
