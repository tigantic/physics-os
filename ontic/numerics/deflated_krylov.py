"""
Deflated Krylov Solvers
========================

Krylov methods augmented with deflation of near-null-space components
to accelerate convergence for ill-conditioned and multi-scale problems.

Implements:

1. **Deflated CG** (INIT-CG / DEF-CG) — project out known near-kernel
   vectors from the Krylov iteration.
2. **Deflated GMRES** — augmented GMRES with recycled Krylov subspaces
   (GCRO-DR style).

Deflation subspace :math:`W \\in \\mathbb{R}^{n \\times k}` satisfies:

.. math::
    P_{\\text{def}} = I - A W (W^T A W)^{-1} W^T

so that :math:`P_{\\text{def}} A` has :math:`k` eigenvalues exactly at 1.

References:
    [1] Nicolaides, "Deflation of conjugate gradients with applications
        to boundary value problems", SIAM J Numer Anal 1987.
    [2] Parks et al., "Recycling Krylov subspaces for sequences of
        linear systems", SIAM J Sci Comp 2006.
    [3] Nabben & Vuik, "A comparison of deflation and the balancing
        preconditioner", SIAM J Sci Comp 2006.

Domain I.3.5 — Numerics / Solvers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator


@dataclass
class KrylovResult:
    """Result container for Krylov solvers."""
    x: NDArray
    residuals: List[float]
    converged: bool
    iterations: int


class DeflatedCG:
    """
    Deflated Conjugate Gradient (DEF-CG).

    Given :math:`Ax = b` with SPD operator A and deflation vectors W,
    constructs the deflation projector and solves within the complementary
    subspace, then corrects with the deflation component.

    Parameters:
        W: Deflation subspace ``(n, k)`` — columns are near-null vectors.

    Example::

        W = compute_near_null_vectors(A, k=5)
        solver = DeflatedCG(W)
        result = solver.solve(A, b)
    """

    def __init__(self, W: NDArray) -> None:
        if W.ndim != 2:
            raise ValueError("W must be 2-D with shape (n, k)")
        self.W = W.copy()
        self._AW: Optional[NDArray] = None
        self._E_inv: Optional[NDArray] = None

    def _build_projector(self, A_matvec: Callable[[NDArray], NDArray]) -> None:
        """Compute E = W^T A W and its inverse."""
        n, k = self.W.shape
        AW = np.zeros_like(self.W)
        for j in range(k):
            AW[:, j] = A_matvec(self.W[:, j])
        self._AW = AW
        E = self.W.T @ AW
        self._E_inv = np.linalg.inv(E)

    def _project(self, r: NDArray) -> NDArray:
        """Apply P_def = I - A W E^{-1} W^T."""
        assert self._AW is not None and self._E_inv is not None
        return r - self._AW @ (self._E_inv @ (self.W.T @ r))

    def _deflation_correction(self, b: NDArray) -> NDArray:
        """Compute x_def = W E^{-1} W^T b."""
        assert self._E_inv is not None
        return self.W @ (self._E_inv @ (self.W.T @ b))

    def solve(
        self,
        A: NDArray | csr_matrix,
        b: NDArray,
        x0: Optional[NDArray] = None,
        max_iter: int = 1000,
        tol: float = 1e-10,
        M_inv: Optional[Callable[[NDArray], NDArray]] = None,
    ) -> KrylovResult:
        """
        Solve Ax = b with deflated CG.

        Parameters:
            A: SPD matrix (dense or sparse).
            b: Right-hand side.
            x0: Initial guess.
            max_iter: Maximum iterations.
            tol: Relative residual tolerance.
            M_inv: Optional preconditioner (callable).

        Returns:
            KrylovResult with solution and convergence info.
        """
        n = len(b)

        if sparse.issparse(A):
            A_matvec = lambda v: A @ v
        else:
            A_matvec = lambda v: A @ v

        self._build_projector(A_matvec)

        x = x0.copy() if x0 is not None else np.zeros(n)
        # Initial deflation correction
        x = x + self._deflation_correction(b - A_matvec(x))

        r = b - A_matvec(x)
        r = self._project(r)

        if M_inv is not None:
            z = M_inv(r)
        else:
            z = r.copy()
        z = self._project(z)

        p = z.copy()
        rz = np.dot(r, z)

        r0_norm = np.linalg.norm(b)
        residuals = [np.linalg.norm(r)]

        for k in range(max_iter):
            Ap = A_matvec(p)
            Ap = self._project(Ap)
            pAp = np.dot(p, Ap)

            if abs(pAp) < 1e-30:
                break

            alpha = rz / pAp
            x = x + alpha * p
            r = r - alpha * Ap

            r_norm = np.linalg.norm(r)
            residuals.append(r_norm)

            if r_norm / (r0_norm + 1e-30) < tol:
                return KrylovResult(x=x, residuals=residuals, converged=True, iterations=k + 1)

            if M_inv is not None:
                z_new = M_inv(r)
            else:
                z_new = r.copy()
            z_new = self._project(z_new)

            rz_new = np.dot(r, z_new)
            beta = rz_new / (rz + 1e-30)
            p = z_new + beta * p
            rz = rz_new

        return KrylovResult(x=x, residuals=residuals, converged=False, iterations=max_iter)


class DeflatedGMRES:
    """
    Deflated GMRES with recycled Krylov subspace (GCRO-DR flavour).

    Maintains a small deflation subspace U, C = A U that captures
    harmonic Ritz vectors from the previous restart cycle.

    Parameters:
        k: Deflation subspace size.
        m: Restart parameter (Arnoldi subspace size per cycle).

    Example::

        solver = DeflatedGMRES(k=10, m=40)
        result = solver.solve(A, b)
    """

    def __init__(self, k: int = 10, m: int = 40) -> None:
        if k >= m:
            raise ValueError(f"k ({k}) must be less than m ({m})")
        self.k = k
        self.m = m
        self.U: Optional[NDArray] = None
        self.C: Optional[NDArray] = None

    def _arnoldi(
        self, A_matvec: Callable[[NDArray], NDArray], v: NDArray, m: int,
    ) -> Tuple[NDArray, NDArray]:
        """
        Arnoldi iteration to build Krylov basis.

        Returns:
            V: (n, m+1) orthonormal basis
            H: (m+1, m) upper Hessenberg
        """
        n = len(v)
        V = np.zeros((n, m + 1))
        H = np.zeros((m + 1, m))

        V[:, 0] = v / (np.linalg.norm(v) + 1e-30)

        for j in range(m):
            w = A_matvec(V[:, j])
            for i in range(j + 1):
                H[i, j] = np.dot(V[:, i], w)
                w = w - H[i, j] * V[:, i]
            H[j + 1, j] = np.linalg.norm(w)
            if H[j + 1, j] < 1e-14:
                return V[:, : j + 2], H[: j + 2, : j + 1]
            V[:, j + 1] = w / H[j + 1, j]

        return V, H

    def _harmonic_ritz(
        self, H: NDArray, k: int,
    ) -> NDArray:
        """
        Compute k harmonic Ritz vectors from Hessenberg matrix.

        Returns indices of the k smallest harmonic Ritz values.
        """
        m = H.shape[1]
        Hm = H[:m, :m]
        # Harmonic Rayleigh quotient: H_m^{-H} H_m^T H_m
        try:
            HtH = Hm.T @ Hm
            vals, vecs = np.linalg.eig(HtH)
            idx = np.argsort(np.abs(vals))[:k]
            return vecs[:, idx].real
        except np.linalg.LinAlgError:
            return np.eye(m, k)

    def solve(
        self,
        A: NDArray | csr_matrix,
        b: NDArray,
        x0: Optional[NDArray] = None,
        max_iter: int = 200,
        tol: float = 1e-10,
    ) -> KrylovResult:
        """
        Solve Ax = b with deflated GMRES (recycled Krylov).

        Parameters:
            A: System matrix (dense or sparse).
            b: Right-hand side vector.
            x0: Initial guess.
            max_iter: Maximum outer (restart) iterations.
            tol: Relative residual tolerance.

        Returns:
            KrylovResult with solution and convergence info.
        """
        n = len(b)
        if sparse.issparse(A):
            A_matvec = lambda v: A @ v
        else:
            A_matvec = lambda v: A @ v

        x = x0.copy() if x0 is not None else np.zeros(n)
        r = b - A_matvec(x)
        r0_norm = np.linalg.norm(b)
        residuals = [np.linalg.norm(r)]

        for cycle in range(max_iter):
            r = b - A_matvec(x)
            r_norm = np.linalg.norm(r)
            residuals.append(r_norm)

            if r_norm / (r0_norm + 1e-30) < tol:
                return KrylovResult(x=x, residuals=residuals, converged=True, iterations=cycle + 1)

            # If we have a recycled subspace, deflate
            if self.U is not None and self.C is not None:
                k_cur = self.U.shape[1]
                CtC = self.C.T @ self.C
                try:
                    alpha_def = np.linalg.solve(CtC, self.C.T @ r)
                    x = x + self.U @ alpha_def
                    r = b - A_matvec(x)
                except np.linalg.LinAlgError:
                    pass

            # Arnoldi
            V, H = self._arnoldi(A_matvec, r, self.m)
            m_actual = H.shape[1]

            # Solve least-squares in Arnoldi coordinates
            beta = np.linalg.norm(r)
            e1 = np.zeros(m_actual + 1)
            e1[0] = beta

            # QR via Givens rotations for least-squares
            y, res_ls, _, _ = np.linalg.lstsq(H, e1, rcond=None)
            x = x + V[:, :m_actual] @ y

            # Extract harmonic Ritz vectors for recycling
            if m_actual >= self.k:
                P_hr = self._harmonic_ritz(H, min(self.k, m_actual))
                k_hr = P_hr.shape[1]
                self.U = V[:, :m_actual] @ P_hr
                self.C = np.zeros((n, k_hr))
                for j in range(k_hr):
                    self.C[:, j] = A_matvec(self.U[:, j])

                # Orthonormalize C, U
                Q, R_qr = np.linalg.qr(self.C)
                try:
                    self.U = self.U @ np.linalg.inv(R_qr)
                except np.linalg.LinAlgError:
                    self.U = None
                    self.C = None
                else:
                    self.C = Q

        return KrylovResult(x=x, residuals=residuals, converged=False, iterations=max_iter)
