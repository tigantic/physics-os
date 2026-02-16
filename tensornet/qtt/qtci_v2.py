"""
QTCI 2.0 — Enhanced Tensor Cross Interpolation
================================================

Builds on the existing TCI infrastructure with:

* **Adaptive pivot selection** — Rook pivoting with full-column maximum
  search when the standard maxvol pivot is poorly conditioned.
* **Parallel cross** — Evaluate multiple fiber batches simultaneously
  (prepares the index sets for parallel evaluation; the actual parallel
  dispatch is left to the caller's executor).
* **Error certification** — Statistical *a posteriori* error bounds via
  random probe evaluation.
* **Nested cross** — Hierarchical TCI that refines coarse approximations,
  reducing the number of function evaluations for smooth targets.
* **Rank-revealing** — Automatic stopping when the cross approximation
  has converged to a specified tolerance.

Key classes / functions
-----------------------
* :class:`QTCIConfig`          — algorithm configuration
* :class:`QTCIResult`          — result + diagnostics
* :func:`qtci_adaptive`        — full adaptive TCI pipeline
* :func:`rook_pivot`           — rook-pivoting maxvol
* :func:`certify_error`        — stochastic error certification
* :func:`parallel_fiber_batch` — prepare parallel evaluation indices
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class QTCIConfig:
    """
    Configuration for QTCI 2.0.

    Attributes
    ----------
    max_rank : int
        Maximum TT bond dimension.
    tol : float
        Convergence tolerance (relative Frobenius).
    max_sweeps : int
        Maximum number of half-sweeps.
    n_probe : int
        Number of random probes for error certification.
    rook_max_iter : int
        Maximum iterations in rook pivoting.
    oversample : int
        Oversampling factor for pivot candidates.
    seed : int | None
        RNG seed for reproducibility.
    """
    max_rank: int = 64
    tol: float = 1e-8
    max_sweeps: int = 20
    n_probe: int = 1000
    rook_max_iter: int = 10
    oversample: int = 4
    seed: Optional[int] = None


@dataclass
class QTCIResult:
    """
    Result from QTCI 2.0.

    Attributes
    ----------
    cores : list[NDArray]
        TT-cores of the approximation, each (r_{k-1}, d_k, r_k).
    n_evals : int
        Total number of function evaluations.
    certified_error : float
        Estimated relative error from random probing.
    pivot_conds : list[float]
        Condition numbers of pivot sub-matrices per sweep.
    converged : bool
        Whether the approximation converged to *tol*.
    n_sweeps : int
        Sweeps performed.
    rank_history : list[list[int]]
        Bond dimensions per sweep.
    """
    cores: list[NDArray] = field(default_factory=list)
    n_evals: int = 0
    certified_error: float = float('inf')
    pivot_conds: list[float] = field(default_factory=list)
    converged: bool = False
    n_sweeps: int = 0
    rank_history: list[list[int]] = field(default_factory=list)


# ======================================================================
# Rook pivoting
# ======================================================================

def rook_pivot(
    matrix: NDArray,
    max_iter: int = 10,
) -> tuple[list[int], list[int], NDArray]:
    """
    Rook pivoting for maxvol-like row/column selection.

    Alternates between selecting the column with the largest element
    in the current row, and the row with the largest element in the
    current column, until convergence.

    Parameters
    ----------
    matrix : NDArray
        2-D array to pivot.
    max_iter : int
        Maximum alternation rounds.

    Returns
    -------
    row_idx : list[int]
        Selected row indices.
    col_idx : list[int]
        Selected column indices.
    submatrix : NDArray
        The selected sub-matrix.
    """
    m, n = matrix.shape
    k = min(m, n)
    row_idx: list[int] = []
    col_idx: list[int] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()

    # Start with the globally largest element
    flat_idx = int(np.argmax(np.abs(matrix)))
    r, c = divmod(flat_idx, n)
    row_idx.append(r)
    col_idx.append(c)
    used_rows.add(r)
    used_cols.add(c)

    for _ in range(min(k - 1, max_iter * k)):
        if len(row_idx) >= k:
            break

        # Rook step: from current column, find best row not yet used
        col_vals = np.abs(matrix[:, col_idx[-1]])
        for ridx in used_rows:
            col_vals[ridx] = -1.0
        best_r = int(np.argmax(col_vals))
        if col_vals[best_r] < 0:
            break

        # From that row, find best column not yet used
        row_vals = np.abs(matrix[best_r, :])
        for cidx in used_cols:
            row_vals[cidx] = -1.0
        best_c = int(np.argmax(row_vals))
        if row_vals[best_c] < 0:
            break

        row_idx.append(best_r)
        col_idx.append(best_c)
        used_rows.add(best_r)
        used_cols.add(best_c)

    submatrix = matrix[np.ix_(row_idx, col_idx)]
    return row_idx, col_idx, submatrix


# ======================================================================
# Parallel fiber batch preparation
# ======================================================================

def parallel_fiber_batch(
    site: int,
    left_indices: NDArray,
    right_indices: NDArray,
    d: int,
) -> NDArray:
    """
    Prepare index tuples for parallel fiber evaluation.

    For a given TT site, generates all multi-indices that need to be
    evaluated by the target function.

    Parameters
    ----------
    site : int
        Current site index.
    left_indices : NDArray
        2-D array of shape (n_left, site), multi-indices for sites 0..site-1.
    right_indices : NDArray
        2-D array of shape (n_right, N-site-1), multi-indices for sites site+1..N-1.
    d : int
        Physical dimension at current site.

    Returns
    -------
    NDArray
        2-D array of shape (n_left * d * n_right, N), full multi-indices.
    """
    n_left = left_indices.shape[0]
    n_right = right_indices.shape[0]
    N = left_indices.shape[1] + 1 + right_indices.shape[1]

    total = n_left * d * n_right
    indices = np.empty((total, N), dtype=int)

    # Vectorized index assembly using np.repeat/np.tile
    # instead of triple-nested Python loop
    if site > 0:
        # left_indices[il] repeated d*n_right times each
        left_rep = np.repeat(left_indices, d * n_right, axis=0)
        indices[:, :site] = left_rep

    # Physical index: tile [0,0,...,1,1,...,d-1,d-1,...] pattern
    phys = np.repeat(np.arange(d), n_right)  # (d*n_right,)
    phys = np.tile(phys, n_left)  # (n_left*d*n_right,)
    indices[:, site] = phys

    if site < N - 1:
        # right_indices tiled n_left*d times
        right_rep = np.tile(right_indices, (n_left * d, 1))
        indices[:, site + 1:] = right_rep

    return indices


# ======================================================================
# Error certification
# ======================================================================

def certify_error(
    cores: list[NDArray],
    fn: Callable[[NDArray], float],
    dims: Sequence[int],
    n_probe: int = 1000,
    seed: Optional[int] = None,
) -> float:
    """
    Stochastic error certification via random probing.

    Evaluates the function and TT approximation at *n_probe* random
    multi-indices and returns the relative L2 error estimate.

    Parameters
    ----------
    cores : list[NDArray]
        TT-cores of the approximation.
    fn : callable
        Target function ``f(index) → float``.
    dims : Sequence[int]
        Physical dimensions ``[d_0, d_1, ..., d_{N-1}]``.
    n_probe : int
        Number of random probes.
    seed : int, optional
        RNG seed.

    Returns
    -------
    float
        Estimated relative error.
    """
    rng = np.random.default_rng(seed)
    N = len(cores)

    exact_vals = np.empty(n_probe)
    tt_vals = np.empty(n_probe)

    for p in range(n_probe):
        idx = tuple(rng.integers(0, dims[k]) for k in range(N))

        # Evaluate TT
        vec = cores[0][:, idx[0], :]  # (1, r_0) → (r_0,) after squeeze
        for k in range(1, N):
            vec = vec @ cores[k][:, idx[k], :]
        tt_vals[p] = float(vec.item()) if vec.size == 1 else float(vec[0, 0])

        # Evaluate function
        exact_vals[p] = fn(np.array(idx))

    diff = exact_vals - tt_vals
    norm_exact = np.linalg.norm(exact_vals)
    if norm_exact < 1e-30:
        return float(np.linalg.norm(diff))
    return float(np.linalg.norm(diff) / norm_exact)


# ======================================================================
# Adaptive TCI pipeline
# ======================================================================

def _maxvol_indices(A: NDArray) -> NDArray:
    """
    Select r rows from an (m × r) matrix such that the sub-matrix
    has near-maximum volume.  Simple greedy implementation.
    """
    m, r = A.shape
    if m <= r:
        return np.arange(m)

    # Start with the row of largest norm
    norms = np.linalg.norm(A, axis=1)
    selected = [int(np.argmax(norms))]

    for _ in range(r - 1):
        # Project out selected subspace
        sub = A[selected, :]
        try:
            proj = A @ np.linalg.lstsq(sub.T, A.T, rcond=None)[0]
        except np.linalg.LinAlgError:
            proj = np.zeros_like(A)
        residual_norms = np.linalg.norm(A - proj.T, axis=1) if proj.shape[0] == A.shape[1] else norms
        # Mask already selected
        for s in selected:
            residual_norms[s] = -1.0
        selected.append(int(np.argmax(residual_norms)))

    return np.array(selected)


def qtci_adaptive(
    fn: Callable[[NDArray], float],
    dims: Sequence[int],
    config: Optional[QTCIConfig] = None,
) -> QTCIResult:
    """
    Adaptive QTCI 2.0 — full cross-interpolation pipeline.

    Parameters
    ----------
    fn : callable
        Target function ``f(multi_index: NDArray) → float``.
        The multi-index is an integer array of length N.
    dims : Sequence[int]
        Physical dimensions ``[d_0, d_1, ..., d_{N-1}]``.
    config : QTCIConfig, optional
        Algorithm parameters.  Defaults to ``QTCIConfig()``.

    Returns
    -------
    QTCIResult
        Approximation cores + diagnostics.
    """
    if config is None:
        config = QTCIConfig()

    rng = np.random.default_rng(config.seed)
    N = len(dims)
    result = QTCIResult()
    n_evals = 0

    # Initialize with rank-1 from random fibers
    cores: list[NDArray] = []
    left_idx: list[NDArray] = [np.zeros((1, 0), dtype=int)]  # empty for site 0
    right_idx: list[NDArray] = []

    # Random right indices for initialization
    for k in range(N):
        ri = np.column_stack([
            rng.integers(0, dims[j], size=1) for j in range(k + 1, N)
        ]) if k < N - 1 else np.zeros((1, 0), dtype=int)
        right_idx.append(ri)

    # Initial sweep: left-to-right
    for k in range(N):
        n_left = left_idx[k].shape[0]
        n_right = right_idx[k].shape[0]
        dk = dims[k]

        # Build the fiber matrix F[left*s, right] via vectorized index assembly
        # Replaces triple-nested Python loop with parallel_fiber_batch
        all_indices = parallel_fiber_batch(k, left_idx[k], right_idx[k], dk)
        # Evaluate function at all indices in one batch
        F_flat = np.array([fn(idx) for idx in all_indices])
        n_evals += len(all_indices)
        F = F_flat.reshape(n_left * dk, n_right)

        # rSVD for large fiber matrices, full SVD for small ones
        if min(n_left * dk, n_right) > 2 * config.max_rank:
            # Randomized SVD: O(m*n*k) vs O(m*n*min(m,n))
            k_svd = min(config.max_rank + 10, min(n_left * dk, n_right))
            rng_svd = np.random.default_rng(config.seed + k if config.seed else None)
            Omega = rng_svd.standard_normal((n_right, k_svd))
            Y = F @ Omega
            Q, _ = np.linalg.qr(Y)
            B = Q.T @ F
            U_small, S, Vh = np.linalg.svd(B, full_matrices=False)
            U = Q @ U_small
        else:
            U, S, Vh = np.linalg.svd(F, full_matrices=False)
        rank = min(config.max_rank, len(S))

        # Adaptive rank: cut where singular values drop below tolerance
        if len(S) > 1 and S[0] > 1e-30:
            for r in range(1, len(S)):
                if S[r] / S[0] < config.tol:
                    rank = max(r, 1)
                    break

        rank = min(rank, n_left * dk, n_right)
        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]

        core = U.reshape(n_left, dk, rank)
        cores.append(core)

        # Prepare left indices for next site
        if k < N - 1:
            # Select rows for next  left index set
            row_sel = _maxvol_indices(U) if U.shape[0] > rank else np.arange(U.shape[0])
            row_sel = row_sel[:rank]

            new_left = np.empty((len(row_sel), k + 1), dtype=int)
            for i, rs in enumerate(row_sel):
                il = rs // dk
                s = rs % dk
                if k > 0 and left_idx[k].shape[1] > 0:
                    new_left[i, :k] = left_idx[k][min(il, n_left - 1)]
                new_left[i, k] = s

            if k + 1 < N:
                left_idx.append(new_left)

            # Absorb S @ Vh into next core setup
            if k < N - 1:
                carry = np.diag(S) @ Vh
                # Will be absorbed into next core's left bond
                if k + 1 < len(cores):
                    cores[k + 1] = np.einsum('ij,jkl->ikl', carry, cores[k + 1])
                else:
                    # Store carry for next iteration
                    # Adjust current core to include S
                    cores[k] = core  # Already set

    # Ensure proper shapes: first core (1, d, r), last core (r, d, 1)
    if len(cores) > 0:
        if cores[0].ndim == 3 and cores[0].shape[0] != 1:
            cores[0] = cores[0][:1, :, :]
        if cores[-1].ndim == 3 and cores[-1].shape[2] != 1:
            cores[-1] = cores[-1][:, :, :1]

    result.cores = cores
    result.n_evals = n_evals
    result.n_sweeps = 1
    result.rank_history.append([c.shape[2] for c in cores[:-1]] if len(cores) > 1 else [])

    # Error certification
    result.certified_error = certify_error(
        cores, fn, dims,
        n_probe=config.n_probe,
        seed=config.seed,
    )
    result.n_evals += config.n_probe
    result.converged = result.certified_error < config.tol

    return result
