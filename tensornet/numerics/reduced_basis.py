"""
Reduced Basis Method (RBM)
===========================

Model order reduction for parametric PDEs. Builds a low-dimensional
subspace from "snapshots" (high-fidelity solves at selected parameter
values) and performs rapid online evaluation via Galerkin projection.

Two-phase workflow:

1. **Offline** — greedy or POD snapshot selection + affine decomposition.
2. **Online** — assemble and solve the reduced :math:`N \\times N`
   system (:math:`N \\ll \\mathcal{N}_h`).

The a-posteriori error estimator drives the greedy algorithm:

.. math::
    \\Delta_N(\\mu) = \\frac{\\| r(\\mu) \\|_{X'}}{\\alpha_{\\text{LB}}(\\mu)}

where :math:`r` is the residual and :math:`\\alpha_{\\text{LB}}` a
lower bound on the coercivity constant (SCM).

Implements:

1. **GreedyRBM** — offline greedy enrichment with error estimator.
2. **POD_RBM** — proper-orthogonal-decomposition basis.
3. **ReducedBasis** — online solver (Galerkin projection).

References:
    [1] Hesthaven, Rozza & Stamm, *Certified Reduced Basis Methods
        for Parametrized PDEs*, Springer 2016.
    [2] Quarteroni, Manzoni & Negri, *Reduced Basis Methods for PDEs*,
        Springer 2016.

Domain I.3.10 — Numerics / Solvers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class RBSnapshot:
    """One high-fidelity snapshot."""
    mu: NDArray       # parameter value
    u_h: NDArray      # solution vector


@dataclass
class ReducedBasisData:
    """Offline data for the reduced basis."""
    V: NDArray                       # (N_h, N) basis matrix
    A_reduced: Optional[NDArray] = None  # (N, N) reduced stiffness
    f_reduced: Optional[NDArray] = None  # (N,) reduced RHS
    mu_train: List[NDArray] = field(default_factory=list)


class ReducedBasis:
    """
    Online reduced-basis solver.

    Given a basis V, assemble and solve the N×N system:
    :math:`V^T A(\\mu) V \\, u_N = V^T f(\\mu)`

    Parameters:
        V: (N_h, N) orthonormal basis.
        assemble_A: Function ``μ → A(μ)`` returning (N_h, N_h) matrix.
        assemble_f: Function ``μ → f(μ)`` returning (N_h,) vector.

    Example::

        rb = ReducedBasis(V, assemble_A, assemble_f)
        u_N = rb.solve(mu_new)
        u_h = rb.reconstruct(u_N)
    """

    def __init__(
        self,
        V: NDArray,
        assemble_A: Callable[[NDArray], NDArray],
        assemble_f: Callable[[NDArray], NDArray],
    ) -> None:
        self.V = V
        self.assemble_A = assemble_A
        self.assemble_f = assemble_f

    @property
    def n_basis(self) -> int:
        return self.V.shape[1]

    @property
    def n_full(self) -> int:
        return self.V.shape[0]

    def solve(self, mu: NDArray) -> NDArray:
        """
        Solve the reduced system at parameter μ.

        Returns:
            u_N: (N,) reduced coefficient vector.
        """
        A = self.assemble_A(mu)
        f = self.assemble_f(mu)

        A_N = self.V.T @ A @ self.V
        f_N = self.V.T @ f

        return np.linalg.solve(A_N, f_N)

    def reconstruct(self, u_N: NDArray) -> NDArray:
        """Map reduced solution back to full space: u_h ≈ V u_N."""
        return self.V @ u_N

    def residual_norm(self, mu: NDArray, u_N: NDArray) -> float:
        """
        Compute the residual norm ||f - A V u_N||.
        """
        A = self.assemble_A(mu)
        f = self.assemble_f(mu)
        u_h = self.reconstruct(u_N)
        return float(np.linalg.norm(f - A @ u_h))


class GreedyRBM:
    """
    Greedy reduced-basis construction with a-posteriori error estimator.

    Parameters:
        assemble_A: High-fidelity system assembly.
        assemble_f: High-fidelity RHS assembly.
        solve_hf: High-fidelity solver ``(A, f) -> u_h``.
        mu_train: (M, p) training parameter samples.
        max_basis: Maximum basis size.
        tol: Greedy tolerance on error estimator.

    Example::

        greedy = GreedyRBM(assemble_A, assemble_f, solve_hf, mu_train)
        rb_data = greedy.run()
        rb = ReducedBasis(rb_data.V, assemble_A, assemble_f)
    """

    def __init__(
        self,
        assemble_A: Callable[[NDArray], NDArray],
        assemble_f: Callable[[NDArray], NDArray],
        solve_hf: Callable[[NDArray, NDArray], NDArray],
        mu_train: NDArray,
        max_basis: int = 50,
        tol: float = 1e-6,
    ) -> None:
        self.assemble_A = assemble_A
        self.assemble_f = assemble_f
        self.solve_hf = solve_hf
        self.mu_train = mu_train
        self.max_basis = max_basis
        self.tol = tol

    def run(self) -> ReducedBasisData:
        """
        Execute the greedy algorithm.

        Returns:
            ReducedBasisData with orthonormal basis V.
        """
        snapshots: List[RBSnapshot] = []
        V: Optional[NDArray] = None

        for k in range(self.max_basis):
            if V is None:
                # First snapshot: pick random or first
                idx = 0
            else:
                # Evaluate error estimator at all training points
                errors = np.zeros(len(self.mu_train))
                rb = ReducedBasis(V, self.assemble_A, self.assemble_f)
                for i, mu in enumerate(self.mu_train):
                    u_N = rb.solve(mu)
                    errors[i] = rb.residual_norm(mu, u_N)
                idx = int(np.argmax(errors))

                if errors[idx] < self.tol:
                    break

            mu_star = self.mu_train[idx]
            A = self.assemble_A(mu_star)
            f = self.assemble_f(mu_star)
            u_h = self.solve_hf(A, f)

            snapshots.append(RBSnapshot(mu=mu_star, u_h=u_h))

            # Orthonormalize
            U = np.column_stack([s.u_h for s in snapshots])
            V, _ = np.linalg.qr(U)

        data = ReducedBasisData(
            V=V if V is not None else np.zeros((0, 0)),
            mu_train=[s.mu for s in snapshots],
        )
        return data


class POD_RBM:
    """
    POD-based reduced basis from a large snapshot ensemble.

    Computes the SVD of the snapshot matrix and retains modes
    capturing a prescribed energy fraction.

    Parameters:
        assemble_A: High-fidelity system assembly.
        assemble_f: High-fidelity RHS assembly.
        solve_hf: High-fidelity solver.
        mu_samples: (M, p) parameter samples for snapshots.
        energy_fraction: Fraction of energy to retain (default 0.9999).

    Example::

        pod = POD_RBM(assemble_A, assemble_f, solve_hf, mu_samples)
        rb_data = pod.run()
    """

    def __init__(
        self,
        assemble_A: Callable[[NDArray], NDArray],
        assemble_f: Callable[[NDArray], NDArray],
        solve_hf: Callable[[NDArray, NDArray], NDArray],
        mu_samples: NDArray,
        energy_fraction: float = 0.9999,
    ) -> None:
        self.assemble_A = assemble_A
        self.assemble_f = assemble_f
        self.solve_hf = solve_hf
        self.mu_samples = mu_samples
        self.energy_fraction = energy_fraction

    def run(self) -> ReducedBasisData:
        """
        Compute all snapshots, perform SVD, truncate.

        Returns:
            ReducedBasisData with POD basis.
        """
        snapshots: List[NDArray] = []
        for mu in self.mu_samples:
            A = self.assemble_A(mu)
            f = self.assemble_f(mu)
            snapshots.append(self.solve_hf(A, f))

        S = np.column_stack(snapshots)
        U, sigma, _ = np.linalg.svd(S, full_matrices=False)

        total_energy = np.sum(sigma**2)
        cumulative = np.cumsum(sigma**2) / total_energy
        n_basis = int(np.searchsorted(cumulative, self.energy_fraction) + 1)
        n_basis = min(n_basis, len(sigma))

        V = U[:, :n_basis]

        return ReducedBasisData(
            V=V,
            mu_train=[mu.copy() for mu in self.mu_samples],
        )
