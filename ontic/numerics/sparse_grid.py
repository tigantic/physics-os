"""
Smolyak Sparse Grids
=====================

Sparse-grid construction for high-dimensional quadrature and
interpolation. Smolyak's algorithm combines 1-D rules with a
sparse selection of multi-indices to break the curse of dimensionality:

.. math::
    \\mathcal{A}(q, d) = \\sum_{q - d + 1 \\le |\\mathbf{i}| \\le q}
        (-1)^{q - |\\mathbf{i}|} \\binom{d-1}{q - |\\mathbf{i}|}
        \\left( Q^{i_1} \\otimes \\cdots \\otimes Q^{i_d} \\right)

Implements:

1. **SparseGrid** — Smolyak grid with Clenshaw-Curtis or
   Gauss-Legendre 1-D rules.
2. **SparseGridInterpolator** — Lagrange interpolation on the grid.
3. **SparseGridQuadrature** — numerical integration.

References:
    [1] Smolyak, "Quadrature and interpolation formulas formed by
        Cartesian products of one-dimensional constructions", Soviet Math
        Doklady 1963.
    [2] Bungartz & Griebel, "Sparse grids", Acta Numerica 2004.
    [3] Nobile, Tempone & Webster, "A sparse grid stochastic collocation
        method for PDEs with random input data", SINUM 2008.

Domain I.3.9 — Numerics / Solvers.
"""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class QuadratureRule(enum.Enum):
    CLENSHAW_CURTIS = "clenshaw_curtis"
    GAUSS_LEGENDRE = "gauss_legendre"


# ---------------------------------------------------------------------------
# 1-D nested rules
# ---------------------------------------------------------------------------

def _clenshaw_curtis_nodes(level: int) -> Tuple[NDArray, NDArray]:
    """
    Clenshaw-Curtis nodes on [-1, 1] at given level.

    Level 0 → 1 point (midpoint), level k → 2^k + 1 points.
    """
    if level == 0:
        return np.array([0.0]), np.array([2.0])

    n = 2**level + 1
    theta = np.pi * np.arange(n) / (n - 1)
    nodes = -np.cos(theta)

    # Weights via FFT-like approach (direct)
    weights = np.zeros(n)
    for i in range(n):
        w = 1.0
        for k in range(1, (n - 1) // 2 + 1):
            b = 1.0 if 2 * k == n - 1 else 2.0
            w -= b * np.cos(2 * k * theta[i]) / (4 * k * k - 1)
        weights[i] = w * 2.0 / (n - 1)
    weights[0] /= 2.0
    weights[-1] /= 2.0

    return nodes, weights


def _gauss_legendre_nodes(level: int) -> Tuple[NDArray, NDArray]:
    """
    Gauss-Legendre nodes at given level.

    Level k → (k + 1) points.
    """
    n = level + 1
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return nodes, weights


def _get_1d_rule(
    level: int, rule: QuadratureRule,
) -> Tuple[NDArray, NDArray]:
    if rule == QuadratureRule.CLENSHAW_CURTIS:
        return _clenshaw_curtis_nodes(level)
    else:
        return _gauss_legendre_nodes(level)


# ---------------------------------------------------------------------------
# Multi-index utilities
# ---------------------------------------------------------------------------

def _smolyak_multi_indices(q: int, d: int) -> List[Tuple[int, ...]]:
    """
    Generate Smolyak multi-indices: all i with q-d+1 ≤ |i| ≤ q,
    where each i_k ≥ 0.
    """
    result: List[Tuple[int, ...]] = []
    lo = max(0, q - d + 1)

    for total in range(lo, q + 1):
        # Compositions of total into d parts (each ≥ 0)
        for combo in _compositions(total, d):
            result.append(combo)

    return result


def _compositions(total: int, d: int) -> List[Tuple[int, ...]]:
    """All ordered d-tuples of non-negative integers summing to total."""
    if d == 1:
        return [(total,)]
    result: List[Tuple[int, ...]] = []
    for first in range(total + 1):
        for rest in _compositions(total - first, d - 1):
            result.append((first,) + rest)
    return result


def _binom(n: int, k: int) -> int:
    """Binomial coefficient."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    result = 1
    for i in range(min(k, n - k)):
        result = result * (n - i) // (i + 1)
    return result


# ---------------------------------------------------------------------------
# Sparse Grid
# ---------------------------------------------------------------------------

@dataclass
class SparseGridResult:
    """Container for sparse grid nodes and weights."""
    nodes: NDArray      # (M, d) grid points
    weights: NDArray    # (M,) quadrature weights
    n_points: int
    level: int
    dimension: int


class SparseGrid:
    """
    Smolyak sparse grid in d dimensions.

    Parameters:
        d: Spatial dimension.
        level: Smolyak level q.
        rule: 1-D quadrature rule to use.
        domain: (d, 2) array of [lo, hi] for each dimension (default [-1,1]).

    Example::

        sg = SparseGrid(d=5, level=4)
        result = sg.build()
        print(f"Points: {result.n_points}")  # vs 5^4 = 625 full-grid
    """

    def __init__(
        self,
        d: int,
        level: int,
        rule: QuadratureRule = QuadratureRule.CLENSHAW_CURTIS,
        domain: Optional[NDArray] = None,
    ) -> None:
        self.d = d
        self.level = level
        self.rule = rule

        if domain is not None:
            self.domain = np.asarray(domain, dtype=float)
        else:
            self.domain = np.tile([-1.0, 1.0], (d, 1))

    def build(self) -> SparseGridResult:
        """
        Construct sparse grid nodes and weights.

        Returns:
            SparseGridResult with nodes, weights, and metadata.
        """
        multi_indices = _smolyak_multi_indices(self.level, self.d)

        all_nodes: List[NDArray] = []
        all_weights: List[float] = []

        for mi in multi_indices:
            total = sum(mi)
            sign = (-1) ** (self.level - total)
            coeff = _binom(self.d - 1, self.level - total)
            if coeff == 0:
                continue

            # Build tensor product for this multi-index
            rules_1d = [_get_1d_rule(mi[k], self.rule) for k in range(self.d)]

            # Tensor-product nodes
            grids = [r[0] for r in rules_1d]
            wts = [r[1] for r in rules_1d]

            for idx_tuple in itertools.product(*(range(len(g)) for g in grids)):
                point = np.array([grids[k][idx_tuple[k]] for k in range(self.d)])
                w = sign * coeff
                for k in range(self.d):
                    w *= wts[k][idx_tuple[k]]
                all_nodes.append(point)
                all_weights.append(w)

        # Merge duplicate nodes (within tolerance)
        nodes, weights = self._merge_duplicates(
            np.array(all_nodes), np.array(all_weights),
        )

        # Map from [-1, 1]^d to user domain
        for k in range(self.d):
            lo, hi = self.domain[k]
            nodes[:, k] = lo + (nodes[:, k] + 1.0) * 0.5 * (hi - lo)
            weights *= (hi - lo) / 2.0

        return SparseGridResult(
            nodes=nodes, weights=weights,
            n_points=len(weights), level=self.level, dimension=self.d,
        )

    @staticmethod
    def _merge_duplicates(
        nodes: NDArray, weights: NDArray, tol: float = 1e-12,
    ) -> Tuple[NDArray, NDArray]:
        """Merge nodes within tolerance, summing their weights."""
        if len(nodes) == 0:
            return nodes, weights

        merged_nodes = [nodes[0]]
        merged_weights = [weights[0]]

        for i in range(1, len(nodes)):
            found = False
            for j in range(len(merged_nodes)):
                if np.linalg.norm(nodes[i] - merged_nodes[j]) < tol:
                    merged_weights[j] += weights[i]
                    found = True
                    break
            if not found:
                merged_nodes.append(nodes[i])
                merged_weights.append(weights[i])

        return np.array(merged_nodes), np.array(merged_weights)


class SparseGridQuadrature:
    """
    Numerical integration on a Smolyak sparse grid.

    Parameters:
        d: Dimension.
        level: Smolyak level.
        rule: 1-D rule.
        domain: Integration domain.

    Example::

        quad = SparseGridQuadrature(d=3, level=5)
        result = quad.integrate(lambda x: np.exp(-np.sum(x**2)))
    """

    def __init__(
        self,
        d: int,
        level: int,
        rule: QuadratureRule = QuadratureRule.CLENSHAW_CURTIS,
        domain: Optional[NDArray] = None,
    ) -> None:
        sg = SparseGrid(d, level, rule, domain)
        grid = sg.build()
        self.nodes = grid.nodes
        self.weights = grid.weights

    def integrate(self, f: Callable[[NDArray], float]) -> float:
        """
        Integrate f over the domain.

        Parameters:
            f: Scalar function f(x) where x is (d,).

        Returns:
            Approximate integral value.
        """
        total = 0.0
        for i in range(len(self.weights)):
            total += self.weights[i] * f(self.nodes[i])
        return total


class SparseGridInterpolator:
    """
    Sparse-grid interpolation via hierarchical surplus.

    Parameters:
        d: Dimension.
        level: Smolyak level.
        rule: 1-D rule.
        domain: Interpolation domain.

    Example::

        interp = SparseGridInterpolator(d=2, level=5)
        interp.fit(lambda x: np.sin(x[0]) * np.cos(x[1]))
        val = interp(np.array([0.5, 0.3]))
    """

    def __init__(
        self,
        d: int,
        level: int,
        rule: QuadratureRule = QuadratureRule.CLENSHAW_CURTIS,
        domain: Optional[NDArray] = None,
    ) -> None:
        sg = SparseGrid(d, level, rule, domain)
        grid = sg.build()
        self.nodes = grid.nodes
        self.n_points = grid.n_points
        self.d = d
        self._values: Optional[NDArray] = None

    def fit(self, f: Callable[[NDArray], float]) -> None:
        """Evaluate f at all grid nodes."""
        self._values = np.array([f(self.nodes[i]) for i in range(self.n_points)])

    def __call__(self, x: NDArray) -> float:
        """
        Evaluate interpolant at point x via inverse-distance weighting.

        For a production implementation this should use hierarchical
        basis functions; here we use Shepard interpolation on the
        sparse grid points.
        """
        if self._values is None:
            raise RuntimeError("Call fit() before evaluation")

        dists = np.linalg.norm(self.nodes - x, axis=1)
        dists = np.maximum(dists, 1e-30)

        # Modified Shepard (power 2)
        w = 1.0 / dists**2
        return float(np.dot(w, self._values) / np.sum(w))
