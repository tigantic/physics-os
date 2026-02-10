"""
Algebraic Multigrid (AMG)
==========================

Classical (Ruge-Stüben) algebraic multigrid for sparse linear systems
:math:`Ax = b` arising from elliptic PDEs on unstructured meshes.

Hierarchy construction:
    1. **Strength-of-connection**: identify strong couplings.
    2. **C/F splitting**: partition DOFs into Coarse/Fine.
    3. **Interpolation**: build prolongation :math:`P`.
    4. **Galerkin coarse operator**: :math:`A_c = R A P` with :math:`R = P^T`.

Smoothing: weighted Jacobi or Gauss-Seidel.

V-cycle:
    .. math::
        \\text{pre-smooth} \\to \\text{restrict residual}
        \\to \\text{solve coarse} \\to \\text{prolongate correction}
        \\to \\text{post-smooth}

References:
    [1] Ruge & Stüben, "Algebraic multigrid", in *Multigrid Methods*,
        SIAM 1987.
    [2] Briggs, Henson & McCormick, *A Multigrid Tutorial*, SIAM 2000.
    [3] Trottenberg, Oosterlee & Schüller, *Multigrid*, Academic Press 2001.

Domain I.3.3 — Numerics / Solvers.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse import csr_matrix, linalg as splinalg


class SmoothingType(enum.Enum):
    JACOBI = "jacobi"
    GAUSS_SEIDEL = "gauss_seidel"


@dataclass
class AMGLevel:
    """One level in the AMG hierarchy."""
    A: csr_matrix
    P: Optional[csr_matrix] = None   # prolongation to next finer
    R: Optional[csr_matrix] = None   # restriction to next coarser


class AMGSolver:
    """
    Classical (Ruge-Stüben) Algebraic Multigrid.

    Example::

        from scipy.sparse import diags
        n = 1000
        A = diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), format='csr')
        b = np.ones(n)
        amg = AMGSolver(A, max_levels=5)
        x = amg.solve(b)
    """

    def __init__(
        self,
        A: csr_matrix,
        max_levels: int = 10,
        max_coarse: int = 50,
        strength_threshold: float = 0.25,
        smoother: SmoothingType = SmoothingType.GAUSS_SEIDEL,
        omega: float = 2.0 / 3.0,
    ) -> None:
        self.smoother = smoother
        self.omega = omega
        self.levels: List[AMGLevel] = []
        self._build_hierarchy(A, max_levels, max_coarse, strength_threshold)

    def _strength_of_connection(
        self, A: csr_matrix, theta: float,
    ) -> csr_matrix:
        """
        Identify strong connections: |a_{ij}| ≥ θ max_{k≠i} |a_{ik}|.

        Returns:
            Boolean strength matrix (CSR).
        """
        n = A.shape[0]
        rows, cols, data = [], [], []
        A_csr = A.tocsr()

        for i in range(n):
            start, end = A_csr.indptr[i], A_csr.indptr[i + 1]
            col_idx = A_csr.indices[start:end]
            vals = A_csr.data[start:end]

            # Max off-diagonal magnitude
            off_diag = np.abs(vals[col_idx != i])
            max_off = np.max(off_diag) if len(off_diag) > 0 else 0.0

            threshold = theta * max_off
            for k in range(start, end):
                j = A_csr.indices[k]
                if j != i and abs(A_csr.data[k]) >= threshold:
                    rows.append(i)
                    cols.append(j)
                    data.append(1.0)

        return csr_matrix((data, (rows, cols)), shape=(n, n))

    def _cf_splitting(self, S: csr_matrix) -> NDArray:
        """
        First-pass C/F splitting based on strength graph.

        Returns:
            Labels array: 1 = Coarse, 0 = Fine.
        """
        n = S.shape[0]
        # Heuristic: nodes with most strong connections become coarse
        weights = np.array(S.sum(axis=1)).ravel() + np.random.uniform(0, 0.01, n)
        labels = np.zeros(n, dtype=np.int32)
        undecided = set(range(n))

        while undecided:
            i = max(undecided, key=lambda x: weights[x])
            labels[i] = 1  # Coarse
            undecided.discard(i)

            # Neighbours of i become fine
            start, end = S.indptr[i], S.indptr[i + 1]
            for k in range(start, end):
                j = S.indices[k]
                if j in undecided:
                    labels[j] = 0  # Fine
                    undecided.discard(j)
                    # Boost remaining undecided neighbours of j
                    s2, e2 = S.indptr[j], S.indptr[j + 1]
                    for k2 in range(s2, e2):
                        m = S.indices[k2]
                        if m in undecided:
                            weights[m] += 1

        return labels

    def _interpolation(
        self, A: csr_matrix, S: csr_matrix, labels: NDArray,
    ) -> csr_matrix:
        """
        Build classical AMG interpolation (direct interpolation).

        For fine point i with strong coarse neighbours C_i:
        w_{ij} = -a_{ij} / a_{ii}  for j ∈ C_i
        """
        n = A.shape[0]
        coarse_indices = np.where(labels == 1)[0]
        coarse_map = {c: k for k, c in enumerate(coarse_indices)}
        nc = len(coarse_indices)

        rows_p, cols_p, data_p = [], [], []

        for i in range(n):
            if labels[i] == 1:
                # Coarse point: identity interpolation
                rows_p.append(i)
                cols_p.append(coarse_map[i])
                data_p.append(1.0)
            else:
                # Fine point: interpolate from strong coarse neighbours
                start, end = A.indptr[i], A.indptr[i + 1]
                a_ii = A[i, i]
                if abs(a_ii) < 1e-30:
                    continue

                w_sum = 0.0
                weights = []
                for k in range(start, end):
                    j = A.indices[k]
                    if j != i and labels[j] == 1 and S[i, j] != 0:
                        w = -A.data[k] / a_ii
                        weights.append((j, w))
                        w_sum += w

                # Normalise
                if abs(w_sum) > 1e-30:
                    for j, w in weights:
                        rows_p.append(i)
                        cols_p.append(coarse_map[j])
                        data_p.append(w / w_sum)
                else:
                    # Fallback: equal weights
                    for j, w in weights:
                        rows_p.append(i)
                        cols_p.append(coarse_map[j])
                        data_p.append(1.0 / max(len(weights), 1))

        return csr_matrix((data_p, (rows_p, cols_p)), shape=(n, nc))

    def _build_hierarchy(
        self,
        A: csr_matrix,
        max_levels: int,
        max_coarse: int,
        theta: float,
    ) -> None:
        """Build the full AMG hierarchy."""
        self.levels = [AMGLevel(A=A)]

        for _ in range(max_levels - 1):
            A_cur = self.levels[-1].A
            n = A_cur.shape[0]
            if n <= max_coarse:
                break

            S = self._strength_of_connection(A_cur, theta)
            labels = self._cf_splitting(S)

            if np.sum(labels) == 0 or np.sum(labels) == n:
                break

            P = self._interpolation(A_cur, S, labels)
            R = P.T.tocsr()
            A_coarse = R @ A_cur @ P

            self.levels[-1].P = P
            self.levels[-1].R = R
            self.levels.append(AMGLevel(A=A_coarse))

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    def _smooth(self, A: csr_matrix, x: NDArray, b: NDArray, n_iter: int) -> NDArray:
        """Apply smoother."""
        if self.smoother == SmoothingType.JACOBI:
            D_inv = 1.0 / (A.diagonal() + 1e-30)
            for _ in range(n_iter):
                x = x + self.omega * D_inv * (b - A @ x)
        else:
            # Gauss-Seidel (forward sweep)
            n = len(x)
            for _ in range(n_iter):
                for i in range(n):
                    start, end = A.indptr[i], A.indptr[i + 1]
                    sigma = 0.0
                    a_ii = 0.0
                    for k in range(start, end):
                        j = A.indices[k]
                        if j == i:
                            a_ii = A.data[k]
                        else:
                            sigma += A.data[k] * x[j]
                    if abs(a_ii) > 1e-30:
                        x[i] = (b[i] - sigma) / a_ii
        return x

    def v_cycle(
        self,
        b: NDArray,
        x: Optional[NDArray] = None,
        level: int = 0,
        pre_smooth: int = 2,
        post_smooth: int = 2,
    ) -> NDArray:
        """
        One V-cycle.

        Parameters:
            b: Right-hand side.
            x: Initial guess (default: zero).
            level: Current level (0 = finest).
            pre_smooth: Number of pre-smoothing iterations.
            post_smooth: Number of post-smoothing iterations.

        Returns:
            Updated solution.
        """
        lev = self.levels[level]
        A = lev.A
        n = A.shape[0]

        if x is None:
            x = np.zeros(n)

        # Coarsest level: direct solve
        if level == self.n_levels - 1 or lev.P is None:
            return splinalg.spsolve(A, b)

        # Pre-smooth
        x = self._smooth(A, x, b, pre_smooth)

        # Restrict residual
        r = b - A @ x
        r_c = lev.R @ r

        # Recurse
        e_c = self.v_cycle(r_c, level=level + 1,
                           pre_smooth=pre_smooth, post_smooth=post_smooth)

        # Prolongate and correct
        x = x + lev.P @ e_c

        # Post-smooth
        x = self._smooth(A, x, b, post_smooth)

        return x

    def solve(
        self,
        b: NDArray,
        x0: Optional[NDArray] = None,
        max_iter: int = 100,
        tol: float = 1e-10,
    ) -> NDArray:
        """
        Solve Ax = b using AMG V-cycles.

        Returns:
            Solution vector x.
        """
        A = self.levels[0].A
        x = x0 if x0 is not None else np.zeros(A.shape[0])

        r_norm_0 = np.linalg.norm(b - A @ x)
        if r_norm_0 < 1e-30:
            return x

        for _ in range(max_iter):
            x = self.v_cycle(b, x)
            r_norm = np.linalg.norm(b - A @ x)
            if r_norm / r_norm_0 < tol:
                break

        return x
