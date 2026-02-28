"""
Proper Generalized Decomposition (PGD)
========================================

Separated-representation solver for high-dimensional parametric PDEs.

PGD seeks the solution as a finite sum of rank-one products:

.. math::
    u(x_1, x_2, \\ldots, x_d) \\approx \\sum_{n=1}^{N}
        F_1^n(x_1) \\otimes F_2^n(x_2) \\otimes \\cdots \\otimes F_d^n(x_d)

Each enrichment step computes one new term via alternating-direction
fixed-point iteration (greedy rank-one update).

Implements:

1. **PGDSolver** — generic d-dimensional separated solver.
2. **PGDThermalSteady** — convenience wrapper for steady heat with
   parameterised conductivity :math:`k(\\mu)`.
3. **SeparatedFunction** — container for the decomposition.

References:
    [1] Chinesta, Ladeveze & Cueto, "A short review on model order
        reduction based on PGD", Archives CAMES 2011.
    [2] Nouy, "A priori model reduction through PGD", Comput. Methods
        Appl. Mech. Engrg. 2010.

Domain I.3.8 — Numerics / Solvers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class SeparatedFunction:
    """
    Rank-N separated representation in d dimensions.

    Attributes:
        modes: List of d arrays per enrichment term.
                modes[n][k] is the 1-D function for dimension k
                at enrichment n.
        n_terms: Number of enrichment terms.
    """
    modes: List[List[NDArray]] = field(default_factory=list)

    @property
    def n_terms(self) -> int:
        return len(self.modes)

    @property
    def n_dims(self) -> int:
        return len(self.modes[0]) if self.modes else 0

    def evaluate_full(self) -> NDArray:
        """
        Reconstruct full tensor from separated representation.

        Returns:
            d-dimensional numpy array.
        """
        if self.n_terms == 0:
            raise ValueError("Empty decomposition")

        result = np.zeros(1)
        for n in range(self.n_terms):
            term = self.modes[n][0].copy()
            for k in range(1, self.n_dims):
                term = np.outer(term, self.modes[n][k]).ravel()
            if n == 0:
                result = term
            else:
                result = result + term

        shape = tuple(len(self.modes[0][k]) for k in range(self.n_dims))
        return result.reshape(shape)


class PGDSolver:
    """
    Generic PGD solver for d-dimensional separated problems.

    The user provides:
    - ``operators[k]``: list of d stiffness-like matrices (n_k × n_k)
      for each dimension.
    - ``rhs_factors[k]``: list of d vectors defining the separated RHS.

    The problem solved is:

    .. math::
        \\sum_{\\text{terms}} \\bigotimes_k A_k \\, u = \\sum_{\\text{terms}} \\bigotimes_k f_k

    Parameters:
        operators: List of (d,) lists of (n_k, n_k) matrices.
        rhs_modes: List of (d,) lists of (n_k,) vectors.
        max_terms: Maximum enrichment terms.
        max_fp_iter: Max fixed-point iterations per enrichment.
        tol: Convergence tolerance for enrichment.
        fp_tol: Tolerance for fixed-point iteration.

    Example::

        # 2-D Laplacian: A = Kx ⊗ My + Mx ⊗ Ky
        ops = [[Kx, My], [Mx, Ky]]
        rhs = [[fx, fy]]  # f = fx ⊗ fy
        pgd = PGDSolver(ops, rhs, max_terms=20)
        sol = pgd.solve()
    """

    def __init__(
        self,
        operators: List[List[NDArray]],
        rhs_modes: List[List[NDArray]],
        max_terms: int = 30,
        max_fp_iter: int = 50,
        tol: float = 1e-8,
        fp_tol: float = 1e-10,
    ) -> None:
        self.operators = operators  # each entry is [A_1^(t), A_2^(t), ..., A_d^(t)]
        self.rhs_modes = rhs_modes
        self.n_ops = len(operators)
        self.d = len(operators[0])
        self.max_terms = max_terms
        self.max_fp_iter = max_fp_iter
        self.tol = tol
        self.fp_tol = fp_tol

        self._validate()

    def _validate(self) -> None:
        for op in self.operators:
            if len(op) != self.d:
                raise ValueError("All operator terms must have same dimensionality d")
        for rhs in self.rhs_modes:
            if len(rhs) != self.d:
                raise ValueError("RHS terms must match dimensionality d")

    def solve(self) -> SeparatedFunction:
        """
        Compute PGD approximation via greedy enrichment.

        Returns:
            SeparatedFunction with the decomposition modes.
        """
        result = SeparatedFunction()

        for _n in range(self.max_terms):
            new_modes = self._enrich(result)
            if new_modes is None:
                break
            result.modes.append(new_modes)

            # Check enrichment amplitude
            norms = [np.linalg.norm(new_modes[k]) for k in range(self.d)]
            amp = 1.0
            for nm in norms:
                amp *= nm
            if amp < self.tol:
                break

        return result

    def _enrich(self, current: SeparatedFunction) -> Optional[List[NDArray]]:
        """
        Compute one new enrichment term via alternating direction
        fixed-point iteration.
        """
        # Initialise modes randomly
        sizes = [self.operators[0][k].shape[0] for k in range(self.d)]
        modes = [np.random.randn(sizes[k]) for k in range(self.d)]
        # Normalise
        for k in range(self.d):
            nm = np.linalg.norm(modes[k])
            if nm > 1e-30:
                modes[k] /= nm

        for _fp in range(self.max_fp_iter):
            modes_old = [m.copy() for m in modes]

            for dim in range(self.d):
                # Build reduced system for dimension 'dim' by contracting others
                A_red = np.zeros((sizes[dim], sizes[dim]))
                b_red = np.zeros(sizes[dim])

                # Operator contributions
                for t in range(self.n_ops):
                    coeff = 1.0
                    for k in range(self.d):
                        if k != dim:
                            coeff *= float(modes[k] @ self.operators[t][k] @ modes[k])
                    A_red += coeff * self.operators[t][dim]

                # RHS: original RHS projected
                for r in range(len(self.rhs_modes)):
                    coeff = 1.0
                    for k in range(self.d):
                        if k != dim:
                            coeff *= float(modes[k] @ self.rhs_modes[r][k])
                    b_red += coeff * self.rhs_modes[r][dim]

                # Previous enrichment residual contributions
                for prev_n in range(current.n_terms):
                    for t in range(self.n_ops):
                        coeff = 1.0
                        for k in range(self.d):
                            if k != dim:
                                coeff *= float(
                                    modes[k] @ self.operators[t][k] @ current.modes[prev_n][k]
                                )
                        b_red -= coeff * (self.operators[t][dim] @ current.modes[prev_n][dim])

                # Solve
                try:
                    modes[dim] = np.linalg.solve(A_red, b_red)
                except np.linalg.LinAlgError:
                    modes[dim] = np.linalg.lstsq(A_red, b_red, rcond=None)[0]

            # Check convergence
            delta = sum(
                np.linalg.norm(modes[k] - modes_old[k])
                / (np.linalg.norm(modes[k]) + 1e-30)
                for k in range(self.d)
            )
            if delta < self.fp_tol:
                break

        return modes


class PGDThermalSteady:
    """
    PGD for steady-state heat equation with parameterised conductivity.

    Solves :math:`-\\nabla \\cdot [k(\\mu) \\nabla T] = f` on a 1-D mesh
    for a range of parameter values μ simultaneously.

    The separated form is T(x, μ) = ∑ R_n(x) · S_n(μ).

    Parameters:
        n_x: Spatial DOFs.
        n_mu: Parameter DOFs.
        k_of_mu: Function mapping parameter index to conductivity.
        f_x: Source vector (n_x,).
        max_terms: Maximum enrichment terms.

    Example::

        solver = PGDThermalSteady(
            n_x=50, n_mu=20,
            k_of_mu=lambda j: 1.0 + 0.5 * j / 20,
            f_x=np.ones(50),
        )
        sol = solver.solve()
    """

    def __init__(
        self,
        n_x: int,
        n_mu: int,
        k_of_mu: Callable[[int], float],
        f_x: NDArray,
        max_terms: int = 20,
    ) -> None:
        self.n_x = n_x
        self.n_mu = n_mu

        # 1-D Laplacian (spatial stiffness)
        h = 1.0 / (n_x + 1)
        diag = np.ones(n_x) * 2.0 / h
        off = np.ones(n_x - 1) * (-1.0 / h)
        Kx = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)

        # Mass matrix (spatial)
        Mx = np.eye(n_x) * h

        # Parameter "stiffness" = diag(k(μ_j))
        k_vals = np.array([k_of_mu(j) for j in range(n_mu)])
        Kmu = np.diag(k_vals)
        Mmu = np.eye(n_mu)

        # A = Kx ⊗ Kmu
        operators = [[Kx, Kmu]]

        # RHS: f(x) ⊗ 1(μ)
        rhs_modes = [[f_x[:n_x], np.ones(n_mu)]]

        self._solver = PGDSolver(operators, rhs_modes, max_terms=max_terms)

    def solve(self) -> SeparatedFunction:
        """
        Solve the parameterised thermal problem.

        Returns:
            SeparatedFunction with spatial and parameter modes.
        """
        return self._solver.solve()
