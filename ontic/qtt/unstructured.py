"""
QTT on Unstructured Meshes
============================

Graph-based TT decomposition for FEM/FVM meshes that breaks the
structured-grid limitation of standard QTT.

Approach
--------
1. **Graph ordering**: apply a bandwidth-minimising permutation
   (Cuthill-McKee or METIS-style nested dissection) so the mesh
   connectivity becomes nearly banded.
2. **Tensorisation**: map the 1D node ordering into a multi-index
   via quantics (binary) folding.
3. **TCI compression**: build a TT from the reordered function via
   tensor cross interpolation.

Key classes / functions
-----------------------
* :class:`MeshTT`       — TT representation of a field on an unstructured mesh
* :func:`rcm_order`     — Reverse Cuthill-McKee ordering
* :func:`quantics_fold` — map flat index → multi-index
* :func:`mesh_to_tt`    — end-to-end mesh-data → TT pipeline
* :func:`tt_to_mesh`    — reconstruct mesh-data from TT
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray


# ======================================================================
# Reverse Cuthill-McKee ordering
# ======================================================================

def rcm_order(adjacency: NDArray) -> NDArray:
    """
    Reverse Cuthill-McKee ordering for bandwidth reduction.

    Parameters
    ----------
    adjacency : NDArray
        Symmetric adjacency / connectivity matrix ``(N, N)``.
        Non-zero entry ⟹ connected.

    Returns
    -------
    NDArray
        Permutation vector of length ``N`` (new → old index mapping).
    """
    N = adjacency.shape[0]
    visited = np.zeros(N, dtype=bool)
    order: list[int] = []

    # Find starting node: minimum degree
    degrees = np.array([(adjacency[i] != 0).sum() for i in range(N)])

    while len(order) < N:
        # Pick unvisited node with lowest degree
        remaining = np.where(~visited)[0]
        if len(remaining) == 0:
            break
        start = remaining[np.argmin(degrees[remaining])]

        # BFS from start, sorting neighbours by degree
        queue: deque[int] = deque([start])
        visited[start] = True

        while queue:
            node = queue.popleft()
            order.append(node)
            neighbours = np.where(adjacency[node] != 0)[0]
            unvisited = [n for n in neighbours if not visited[n]]
            # Sort by degree (ascending)
            unvisited.sort(key=lambda x: degrees[x])
            for nb in unvisited:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)

    perm = np.array(order[::-1], dtype=np.intp)  # Reverse
    return perm


# ======================================================================
# Quantics (binary) folding
# ======================================================================

def quantics_fold(index: int, n_bits: int) -> tuple[int, ...]:
    """
    Map a flat integer index to a quantics multi-index (binary digits).

    Parameters
    ----------
    index : int
        Flat index in ``[0, 2^n_bits)``.
    n_bits : int
        Number of binary digits (modes).

    Returns
    -------
    tuple[int, ...]
        Binary digits ``(b_{n_bits-1}, ..., b_0)`` (MSB first).
    """
    return tuple((index >> (n_bits - 1 - k)) & 1 for k in range(n_bits))


def quantics_unfold(multi_index: Sequence[int]) -> int:
    """Inverse of :func:`quantics_fold`."""
    n_bits = len(multi_index)
    return sum(int(b) << (n_bits - 1 - k) for k, b in enumerate(multi_index))


# ======================================================================
# MeshTT data structure
# ======================================================================

@dataclass
class MeshTT:
    """
    TT representation of a scalar field on an unstructured mesh.

    Attributes
    ----------
    cores : list[NDArray]
        TT-cores, each ``(r_left, 2, r_right)`` for quantics folding.
    perm : NDArray
        RCM permutation mapping new → original node indices.
    n_nodes : int
        Original number of mesh nodes.
    n_bits : int
        Number of binary modes (``2^n_bits >= n_nodes``).
    """
    cores: list[NDArray]
    perm: NDArray
    n_nodes: int
    n_bits: int

    @property
    def ranks(self) -> list[int]:
        return [c.shape[0] for c in self.cores] + [self.cores[-1].shape[2]]

    def evaluate(self, node_idx: int) -> float:
        """Evaluate the field at original mesh node *node_idx*."""
        # Find position in permuted order
        pos = int(np.searchsorted(self.perm, node_idx))
        # Fallback: linear search if searchsorted fails on unsorted perm
        matches = np.where(self.perm == node_idx)[0]
        if len(matches) == 0:
            return 0.0
        pos = int(matches[0])
        multi = quantics_fold(pos, self.n_bits)
        vec = np.ones((1,), dtype=self.cores[0].dtype)
        for k, bit in enumerate(multi):
            vec = vec @ self.cores[k][:, bit, :]
        return float(vec.item())

    def to_dense(self) -> NDArray:
        """Reconstruct the full field vector in original mesh ordering."""
        N = 2 ** self.n_bits
        values = np.zeros(N, dtype=self.cores[0].dtype)
        for i in range(N):
            multi = quantics_fold(i, self.n_bits)
            vec = np.ones((1,), dtype=self.cores[0].dtype)
            for k, bit in enumerate(multi):
                vec = vec @ self.cores[k][:, bit, :]
            values[i] = vec.item()
        # Un-permute and trim
        result = np.zeros(self.n_nodes, dtype=values.dtype)
        for new_idx in range(min(self.n_nodes, N)):
            orig_idx = self.perm[new_idx] if new_idx < len(self.perm) else new_idx
            if orig_idx < self.n_nodes:
                result[orig_idx] = values[new_idx]
        return result


# ======================================================================
# TCI-based compression
# ======================================================================

def _tt_cross_1sweep(
    func: Callable[[tuple[int, ...]], float],
    n_bits: int,
    max_rank: int,
    rng: np.random.Generator,
) -> list[NDArray]:
    """
    One left-to-right TT-cross sweep via skeleton decomposition.

    Simple implementation for unstructured-mesh use case.
    """
    cores: list[NDArray] = []
    r_left = 1

    # Left pivot sets (initially just {()})
    left_pivots: list[tuple[int, ...]] = [()]

    for k in range(n_bits):
        d_k = 2  # Binary mode
        # Build the cross-approximation matrix
        # Rows: left_pivots × local index
        # Cols: all possible right completions (sample-based)
        n_left = len(left_pivots)
        # Sample right indices
        n_right_bits = n_bits - k - 1
        if n_right_bits > 0:
            n_right_samples = min(max_rank * 2, 2 ** n_right_bits)
            right_samples = [
                tuple(rng.integers(0, 2, size=n_right_bits).tolist())
                for _ in range(n_right_samples)
            ]
            # De-duplicate
            right_samples = list(set(right_samples))
        else:
            right_samples = [()]

        n_cols = len(right_samples)
        n_rows = n_left * d_k

        mat = np.zeros((n_rows, n_cols), dtype=np.float64)
        for i, lp in enumerate(left_pivots):
            for s in range(d_k):
                row = i * d_k + s
                for j, rp in enumerate(right_samples):
                    full_idx = lp + (s,) + rp
                    mat[row, j] = func(full_idx)

        # Truncated SVD
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)
        keep = min(max_rank, int(np.sum(S > 1e-14)), len(S))
        keep = max(keep, 1)
        U = U[:, :keep]
        S = S[:keep]
        Vh = Vh[:keep, :]

        core = U.reshape(n_left, d_k, keep)
        if k == 0:
            core = core.reshape(1, d_k, keep)
        cores.append(core)

        # Update left pivots for next mode
        # Extend each left pivot with each local index → pick top-rank
        new_left: list[tuple[int, ...]] = []
        for lp in left_pivots:
            for s in range(d_k):
                new_left.append(lp + (s,))
        # Keep only keep pivots (greedily via max-row-norm of S*Vh)
        if len(new_left) > keep:
            scores = np.zeros(len(new_left))
            for idx, nlp in enumerate(new_left):
                row = idx
                if row < mat.shape[0]:
                    scores[idx] = np.linalg.norm(mat[row])
            top = np.argsort(scores)[-keep:]
            new_left = [new_left[t] for t in sorted(top)]
        left_pivots = new_left
        r_left = keep

    # Fix last core to have r_right = 1
    if cores[-1].shape[2] != 1:
        last = cores[-1]
        r_l, d, r_r = last.shape
        mat = last.reshape(r_l * d, r_r)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)
        cores[-1] = (U[:, :1] * S[0]).reshape(r_l, d, 1)

    return cores


# ======================================================================
# End-to-end pipeline
# ======================================================================

def mesh_to_tt(
    values: NDArray,
    adjacency: NDArray,
    max_rank: int = 32,
    seed: Optional[int] = None,
) -> MeshTT:
    """
    Compress a scalar field on an unstructured mesh into TT format.

    1. RCM reorder
    2. Pad to power of 2
    3. Quantics fold and TCI compress

    Parameters
    ----------
    values : NDArray
        Field values at each mesh node, shape ``(N,)``.
    adjacency : NDArray
        Symmetric adjacency matrix ``(N, N)``.
    max_rank : int
        Maximum TT bond dimension.
    seed : int, optional
        RNG seed.

    Returns
    -------
    MeshTT
    """
    N = len(values)
    perm = rcm_order(adjacency)

    # Reorder values
    reordered = values[perm]

    # Pad to power of 2
    n_bits = int(np.ceil(np.log2(max(N, 2))))
    N_padded = 2 ** n_bits
    padded = np.zeros(N_padded, dtype=values.dtype)
    padded[:N] = reordered

    # Build lookup function for TCI
    def func(multi_idx: tuple[int, ...]) -> float:
        flat = quantics_unfold(multi_idx)
        if flat < N_padded:
            return float(padded[flat])
        return 0.0

    rng = np.random.default_rng(seed)
    cores = _tt_cross_1sweep(func, n_bits, max_rank, rng)

    return MeshTT(
        cores=cores,
        perm=perm,
        n_nodes=N,
        n_bits=n_bits,
    )


def tt_to_mesh(mtt: MeshTT) -> NDArray:
    """Reconstruct mesh field from :class:`MeshTT`."""
    return mtt.to_dense()
