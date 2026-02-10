"""
Hierarchical Matrices (H-Matrices)
====================================

Data-sparse approximation of dense matrices arising from integral
operators and Green's functions.  The key insight: off-diagonal blocks
representing well-separated point clusters admit low-rank
approximation (rank k ≪ block size).

Implements:

1. **ClusterTree** — recursive bisection of index sets.
2. **AdmissibilityCondition** — standard criterion
   :math:`\\min(\\text{diam}(\\sigma), \\text{diam}(\\tau)) \\le \\eta \\cdot \\text{dist}(\\sigma, \\tau)`.
3. **ACA** (Adaptive Cross Approximation) — partially-pivoted rank-revealing
   column/row selection for low-rank blocks.
4. **HMatrix** — block cluster tree with full/low-rank leaf storage,
   matrix-vector product, and optional LU.

References:
    [1] Hackbusch, *Hierarchical Matrices: Algorithms and Analysis*,
        Springer 2015.
    [2] Bebendorf, *Hierarchical Matrices*, Springer 2008.
    [3] Börm, Grasedyck & Hackbusch, "Hierarchical matrices",
        Numer. Math. 2003.

Domain I.3.6 — Numerics / Solvers.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Cluster tree
# ---------------------------------------------------------------------------

@dataclass
class Cluster:
    """Axis-aligned bounding box cluster of DOF indices."""
    indices: NDArray          # integer array of DOF indices
    bbox_lo: NDArray          # lower-left corner of bounding box
    bbox_hi: NDArray          # upper-right corner
    left: Optional["Cluster"] = None
    right: Optional["Cluster"] = None

    @property
    def size(self) -> int:
        return len(self.indices)

    @property
    def diameter(self) -> float:
        return float(np.linalg.norm(self.bbox_hi - self.bbox_lo))


def build_cluster_tree(
    points: NDArray, indices: Optional[NDArray] = None, leaf_size: int = 32,
) -> Cluster:
    """
    Recursively bisect point cloud to build a balanced cluster tree.

    Parameters:
        points: (N, d) coordinates.
        indices: Subset of indices to cluster.
        leaf_size: Maximum cluster size for a leaf.

    Returns:
        Root Cluster node.
    """
    if indices is None:
        indices = np.arange(len(points))

    pts = points[indices]
    bbox_lo = pts.min(axis=0)
    bbox_hi = pts.max(axis=0)

    if len(indices) <= leaf_size:
        return Cluster(indices=indices, bbox_lo=bbox_lo, bbox_hi=bbox_hi)

    # Split along longest axis
    widths = bbox_hi - bbox_lo
    axis = int(np.argmax(widths))
    coords = pts[:, axis]
    median = np.median(coords)

    mask_left = coords <= median
    mask_right = ~mask_left

    # Avoid empty splits
    if not np.any(mask_left) or not np.any(mask_right):
        return Cluster(indices=indices, bbox_lo=bbox_lo, bbox_hi=bbox_hi)

    left = build_cluster_tree(points, indices[mask_left], leaf_size)
    right = build_cluster_tree(points, indices[mask_right], leaf_size)

    return Cluster(indices=indices, bbox_lo=bbox_lo, bbox_hi=bbox_hi,
                   left=left, right=right)


def cluster_distance(c1: Cluster, c2: Cluster) -> float:
    """
    L2 distance between bounding boxes of two clusters.
    """
    gap = np.maximum(0, np.maximum(c1.bbox_lo - c2.bbox_hi, c2.bbox_lo - c1.bbox_hi))
    return float(np.linalg.norm(gap))


# ---------------------------------------------------------------------------
# ACA (Adaptive Cross Approximation)
# ---------------------------------------------------------------------------

def aca(
    kernel: Callable[[NDArray, NDArray], float],
    rows: NDArray,
    cols: NDArray,
    points: NDArray,
    tol: float = 1e-6,
    max_rank: int = 50,
) -> Tuple[NDArray, NDArray]:
    """
    Adaptive Cross Approximation with partial pivoting.

    Approximates the sub-matrix :math:`K[rows, cols]` as :math:`U V^T`
    where U is (m, k) and V is (n, k).

    Parameters:
        kernel: Function ``kernel(p_i, p_j) -> float``.
        rows: Row indices into points array.
        cols: Column indices into points array.
        points: (N, d) point coordinates.
        tol: Relative Frobenius tolerance.
        max_rank: Maximum rank.

    Returns:
        (U, V) such that K ≈ U @ V.T.
    """
    m, n = len(rows), len(cols)
    U_cols: List[NDArray] = []
    V_cols: List[NDArray] = []
    used_rows: set = set()

    # Initial pivot row
    pivot_row = 0
    frob_sq = 0.0

    for k in range(min(max_rank, min(m, n))):
        # Compute residual row
        row_vec = np.array([
            kernel(points[rows[pivot_row]], points[cols[j]]) for j in range(n)
        ])
        for prev_k in range(k):
            row_vec -= U_cols[prev_k][pivot_row] * V_cols[prev_k]

        # Pivot: find column with largest absolute residual entry
        pivot_col = int(np.argmax(np.abs(row_vec)))
        delta = row_vec[pivot_col]

        if abs(delta) < 1e-30:
            break

        # Compute residual column
        col_vec = np.array([
            kernel(points[rows[i]], points[cols[pivot_col]]) for i in range(m)
        ])
        for prev_k in range(k):
            col_vec -= V_cols[prev_k][pivot_col] * U_cols[prev_k]

        u = col_vec / delta
        v = row_vec.copy()

        U_cols.append(u)
        V_cols.append(v)

        # Update Frobenius estimate
        uv_frob_sq = np.dot(u, u) * np.dot(v, v)
        frob_sq += uv_frob_sq
        for prev_k in range(k):
            frob_sq += 2 * np.dot(U_cols[prev_k], u) * np.dot(V_cols[prev_k], v)

        if uv_frob_sq < tol**2 * frob_sq:
            break

        # Next pivot row: row with max absolute u entry among unused
        used_rows.add(pivot_row)
        abs_u = np.abs(u)
        for r in used_rows:
            abs_u[r] = -1.0
        pivot_row = int(np.argmax(abs_u))

    if len(U_cols) == 0:
        return np.zeros((m, 1)), np.zeros((n, 1))

    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    return U, V


# ---------------------------------------------------------------------------
# H-Matrix block
# ---------------------------------------------------------------------------

class BlockType(enum.Enum):
    FULL = "full"
    LOW_RANK = "low_rank"
    HIERARCHICAL = "hierarchical"


@dataclass
class HBlock:
    """One block in the H-matrix tree."""
    row_cluster: Cluster
    col_cluster: Cluster
    block_type: BlockType = BlockType.FULL

    # Full storage
    full_matrix: Optional[NDArray] = None

    # Low-rank storage: K ≈ U V^T
    U: Optional[NDArray] = None
    V: Optional[NDArray] = None

    # Children (for hierarchical blocks)
    children: List["HBlock"] = field(default_factory=list)


class HMatrix:
    """
    Hierarchical matrix with ACA-compressed off-diagonal blocks.

    Parameters:
        points: (N, d) point coordinates.
        kernel: Callable ``(p_i, p_j) -> float``.
        eta: Admissibility parameter (default 1.0).
        leaf_size: Cluster tree leaf size.
        aca_tol: ACA tolerance.
        aca_max_rank: ACA max rank.

    Example::

        # Laplace kernel in 2D
        def kernel(pi, pj):
            r = np.linalg.norm(pi - pj)
            return -np.log(r + 1e-15) / (2 * np.pi)

        pts = np.random.rand(1000, 2)
        H = HMatrix(pts, kernel)
        y = H.matvec(x)
    """

    def __init__(
        self,
        points: NDArray,
        kernel: Callable[[NDArray, NDArray], float],
        eta: float = 1.0,
        leaf_size: int = 32,
        aca_tol: float = 1e-6,
        aca_max_rank: int = 50,
    ) -> None:
        self.points = points
        self.kernel = kernel
        self.eta = eta
        self.aca_tol = aca_tol
        self.aca_max_rank = aca_max_rank
        self.n = len(points)

        self._row_tree = build_cluster_tree(points, leaf_size=leaf_size)
        self._col_tree = build_cluster_tree(points, leaf_size=leaf_size)
        self.root = self._build_block(self._row_tree, self._col_tree)

    def _is_admissible(self, rc: Cluster, cc: Cluster) -> bool:
        """Standard admissibility: min(diam) ≤ η · dist."""
        d = cluster_distance(rc, cc)
        return min(rc.diameter, cc.diameter) <= self.eta * d

    def _build_block(self, rc: Cluster, cc: Cluster) -> HBlock:
        """Recursively build block cluster tree."""
        # Leaf: both clusters small or admissible
        if rc.left is None and cc.left is None:
            # Full block
            blk = HBlock(row_cluster=rc, col_cluster=cc, block_type=BlockType.FULL)
            m, n = len(rc.indices), len(cc.indices)
            mat = np.zeros((m, n))
            for i in range(m):
                for j in range(n):
                    mat[i, j] = self.kernel(
                        self.points[rc.indices[i]],
                        self.points[cc.indices[j]],
                    )
            blk.full_matrix = mat
            return blk

        if self._is_admissible(rc, cc):
            # Low-rank via ACA
            blk = HBlock(row_cluster=rc, col_cluster=cc, block_type=BlockType.LOW_RANK)
            blk.U, blk.V = aca(
                self.kernel, rc.indices, cc.indices, self.points,
                tol=self.aca_tol, max_rank=self.aca_max_rank,
            )
            return blk

        # Hierarchical: subdivide
        blk = HBlock(row_cluster=rc, col_cluster=cc, block_type=BlockType.HIERARCHICAL)

        row_children = [rc.left, rc.right] if rc.left is not None else [rc]
        col_children = [cc.left, cc.right] if cc.left is not None else [cc]

        for r_child in row_children:
            if r_child is None:
                continue
            for c_child in col_children:
                if c_child is None:
                    continue
                blk.children.append(self._build_block(r_child, c_child))

        return blk

    def _matvec_block(self, blk: HBlock, x: NDArray, y: NDArray) -> None:
        """Recursive matrix-vector product: y[row_idx] += block @ x[col_idx]."""
        ri = blk.row_cluster.indices
        ci = blk.col_cluster.indices

        if blk.block_type == BlockType.FULL:
            assert blk.full_matrix is not None
            y[ri] += blk.full_matrix @ x[ci]

        elif blk.block_type == BlockType.LOW_RANK:
            assert blk.U is not None and blk.V is not None
            # y[ri] += U (V^T x[ci])
            y[ri] += blk.U @ (blk.V.T @ x[ci])

        elif blk.block_type == BlockType.HIERARCHICAL:
            for child in blk.children:
                self._matvec_block(child, x, y)

    def matvec(self, x: NDArray) -> NDArray:
        """
        Compute y = H x.

        Parameters:
            x: Input vector of length n.

        Returns:
            y: Result vector of length n.
        """
        if len(x) != self.n:
            raise ValueError(f"Expected vector of length {self.n}, got {len(x)}")
        y = np.zeros(self.n)
        self._matvec_block(self.root, x, y)
        return y

    def compression_ratio(self) -> float:
        """
        Compute storage ratio vs. dense n×n.

        Returns:
            Ratio (stored entries) / n².
        """
        stored = self._count_entries(self.root)
        return stored / (self.n * self.n)

    def _count_entries(self, blk: HBlock) -> int:
        if blk.block_type == BlockType.FULL:
            assert blk.full_matrix is not None
            return blk.full_matrix.size
        elif blk.block_type == BlockType.LOW_RANK:
            assert blk.U is not None and blk.V is not None
            return blk.U.size + blk.V.size
        else:
            return sum(self._count_entries(c) for c in blk.children)
