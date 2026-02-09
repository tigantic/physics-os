"""
Uncertainty Quantification Toolkit.

Provides methods for forward UQ (propagating uncertain inputs through physics
solvers) and surrogate-based acceleration.

Methods:
    MonteCarloUQ            — Brute-force sampling.
    LatinHypercubeUQ        — Space-filling sample design.
    PolynomialChaosExpansion — Spectral stochastic expansion (PCE).
    EnsembleUQ              — Ensemble-based statistics.
    StochasticCollocation   — Tensor-product or sparse-grid collocation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from tensornet.platform.data_model import SimulationState
from tensornet.platform.protocols import SolveResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# UQ Result Types
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class UQResult:
    """
    Result from an uncertainty quantification analysis.

    Attributes
    ----------
    mean : dict
        Mean field values: {field_name: Tensor}.
    variance : dict
        Variance of each field: {field_name: Tensor}.
    std : dict
        Standard deviation: {field_name: Tensor}.
    percentiles : dict
        Percentile fields: {field_name: {5: Tensor, 50: Tensor, 95: Tensor}}.
    n_samples : int
        Number of samples / evaluations used.
    qoi_statistics : dict
        Statistics for scalar quantities of interest.
    elapsed_seconds : float
        Wall-clock time.
    """

    mean: Dict[str, Tensor]
    variance: Dict[str, Tensor]
    std: Dict[str, Tensor]
    percentiles: Dict[str, Dict[int, Tensor]] = dc_field(default_factory=dict)
    n_samples: int = 0
    qoi_statistics: Dict[str, Dict[str, float]] = dc_field(default_factory=dict)
    elapsed_seconds: float = 0.0


@dataclass
class UQSample:
    """A single UQ sample: parameters + result."""

    params: Dict[str, float]
    result: SolveResult
    qoi_values: Dict[str, float] = dc_field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Parameter Distribution
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ParameterDistribution:
    """
    Uncertain parameter with a specified distribution.

    Supported types: 'uniform', 'normal', 'lognormal'.
    """

    name: str
    distribution: str  # 'uniform', 'normal', 'lognormal'
    mean: float = 0.0
    std: float = 1.0
    lower: float = 0.0
    upper: float = 1.0

    def sample(self, n: int) -> Tensor:
        """Generate n random samples from this distribution."""
        if self.distribution == "uniform":
            return torch.rand(n, dtype=torch.float64) * (self.upper - self.lower) + self.lower
        elif self.distribution == "normal":
            return torch.randn(n, dtype=torch.float64) * self.std + self.mean
        elif self.distribution == "lognormal":
            normal = torch.randn(n, dtype=torch.float64) * self.std + self.mean
            return torch.exp(normal)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")


# ═══════════════════════════════════════════════════════════════════════════════
# Monte Carlo UQ
# ═══════════════════════════════════════════════════════════════════════════════


class MonteCarloUQ:
    """
    Monte Carlo uncertainty quantification.

    Generates random parameter samples and runs the forward solver
    for each, collecting output statistics.

    Parameters
    ----------
    solver : Any
        A Solver-protocol solver.
    n_samples : int
        Number of MC samples.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        solver: Any,
        n_samples: int = 100,
        seed: Optional[int] = None,
    ) -> None:
        self._solver = solver
        self._n_samples = n_samples
        self._seed = seed

    def run(
        self,
        base_state: SimulationState,
        t_span: Tuple[float, float],
        dt: float,
        uncertain_params: List[ParameterDistribution],
        state_modifier: Callable[[SimulationState, Dict[str, float]], SimulationState],
        qoi_extractors: Optional[Dict[str, Callable[[SimulationState], float]]] = None,
    ) -> UQResult:
        """
        Run Monte Carlo UQ.

        Parameters
        ----------
        base_state : SimulationState
            Baseline state to perturb.
        t_span, dt : float
            Time integration parameters.
        uncertain_params : list of ParameterDistribution
            Parameters to vary.
        state_modifier : callable
            Maps (base_state, param_values) → perturbed state.
        qoi_extractors : dict, optional
            Named scalar extractors from final state.
        """
        t0 = time.perf_counter()

        if self._seed is not None:
            torch.manual_seed(self._seed)

        # Generate samples
        param_samples: Dict[str, Tensor] = {}
        for pd in uncertain_params:
            param_samples[pd.name] = pd.sample(self._n_samples)

        # Run forward solves
        all_results: List[UQSample] = []
        field_arrays: Dict[str, List[Tensor]] = {}

        for i in range(self._n_samples):
            params = {name: vals[i].item() for name, vals in param_samples.items()}
            perturbed = state_modifier(base_state.clone(), params)
            result = self._solver.solve(perturbed, t_span, dt)
            final = result.final_state

            # Collect QoI
            qoi_vals: Dict[str, float] = {}
            if qoi_extractors:
                for qname, extractor in qoi_extractors.items():
                    qoi_vals[qname] = extractor(final)

            all_results.append(UQSample(params=params, result=result, qoi_values=qoi_vals))

            # Accumulate field data
            for fname, fdata in final.fields.items():
                if fname not in field_arrays:
                    field_arrays[fname] = []
                field_arrays[fname].append(fdata.data.clone())

        # Compute statistics
        mean: Dict[str, Tensor] = {}
        variance: Dict[str, Tensor] = {}
        std: Dict[str, Tensor] = {}
        percentiles: Dict[str, Dict[int, Tensor]] = {}

        for fname, arrays in field_arrays.items():
            stacked = torch.stack(arrays)  # (n_samples, ...)
            mean[fname] = stacked.mean(dim=0)
            variance[fname] = stacked.var(dim=0)
            std[fname] = stacked.std(dim=0)
            percentiles[fname] = {
                5: torch.quantile(stacked.float(), 0.05, dim=0),
                50: torch.quantile(stacked.float(), 0.50, dim=0),
                95: torch.quantile(stacked.float(), 0.95, dim=0),
            }

        # QoI statistics
        qoi_stats: Dict[str, Dict[str, float]] = {}
        if qoi_extractors:
            for qname in qoi_extractors:
                vals = [r.qoi_values.get(qname, 0.0) for r in all_results]
                t_vals = torch.tensor(vals, dtype=torch.float64)
                qoi_stats[qname] = {
                    "mean": t_vals.mean().item(),
                    "std": t_vals.std().item(),
                    "min": t_vals.min().item(),
                    "max": t_vals.max().item(),
                    "p05": torch.quantile(t_vals.float(), 0.05).item(),
                    "p95": torch.quantile(t_vals.float(), 0.95).item(),
                }

        elapsed = time.perf_counter() - t0
        return UQResult(
            mean=mean, variance=variance, std=std,
            percentiles=percentiles, n_samples=self._n_samples,
            qoi_statistics=qoi_stats, elapsed_seconds=elapsed,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Latin Hypercube Sampling UQ
# ═══════════════════════════════════════════════════════════════════════════════


class LatinHypercubeUQ(MonteCarloUQ):
    """
    Latin Hypercube Sampling (LHS) for space-filling sample design.

    Overrides the sampling strategy with stratified LHS while inheriting
    the MC driver and statistics computation.
    """

    def run(
        self,
        base_state: SimulationState,
        t_span: Tuple[float, float],
        dt: float,
        uncertain_params: List[ParameterDistribution],
        state_modifier: Callable[[SimulationState, Dict[str, float]], SimulationState],
        qoi_extractors: Optional[Dict[str, Callable[[SimulationState], float]]] = None,
    ) -> UQResult:
        t0 = time.perf_counter()

        if self._seed is not None:
            torch.manual_seed(self._seed)

        n = self._n_samples
        d = len(uncertain_params)

        # Generate LHS samples via stratified uniform + permutation
        lhs = torch.zeros(n, d, dtype=torch.float64)
        for j in range(d):
            perm = torch.randperm(n, dtype=torch.float64)
            lhs[:, j] = (perm + torch.rand(n, dtype=torch.float64)) / n

        # Transform to parameter distributions
        param_samples: Dict[str, Tensor] = {}
        for j, pd in enumerate(uncertain_params):
            u = lhs[:, j]
            if pd.distribution == "uniform":
                param_samples[pd.name] = u * (pd.upper - pd.lower) + pd.lower
            elif pd.distribution == "normal":
                # Inverse CDF of normal via erfinv
                z = torch.erfinv(2 * u.clamp(1e-6, 1 - 1e-6) - 1) * (2 ** 0.5)
                param_samples[pd.name] = z * pd.std + pd.mean
            else:
                param_samples[pd.name] = pd.sample(n)

        # Forward solves
        field_arrays: Dict[str, List[Tensor]] = {}
        all_results: List[UQSample] = []

        for i in range(n):
            params = {name: vals[i].item() for name, vals in param_samples.items()}
            perturbed = state_modifier(base_state.clone(), params)
            result = self._solver.solve(perturbed, t_span, dt)
            final = result.final_state

            qoi_vals: Dict[str, float] = {}
            if qoi_extractors:
                for qname, extractor in qoi_extractors.items():
                    qoi_vals[qname] = extractor(final)
            all_results.append(UQSample(params=params, result=result, qoi_values=qoi_vals))

            for fname, fdata in final.fields.items():
                if fname not in field_arrays:
                    field_arrays[fname] = []
                field_arrays[fname].append(fdata.data.clone())

        # Statistics (same as MC)
        mean: Dict[str, Tensor] = {}
        variance: Dict[str, Tensor] = {}
        std: Dict[str, Tensor] = {}
        for fname, arrays in field_arrays.items():
            stacked = torch.stack(arrays)
            mean[fname] = stacked.mean(dim=0)
            variance[fname] = stacked.var(dim=0)
            std[fname] = stacked.std(dim=0)

        qoi_stats: Dict[str, Dict[str, float]] = {}
        if qoi_extractors:
            for qname in qoi_extractors:
                vals = torch.tensor(
                    [r.qoi_values.get(qname, 0.0) for r in all_results],
                    dtype=torch.float64,
                )
                qoi_stats[qname] = {
                    "mean": vals.mean().item(),
                    "std": vals.std().item(),
                    "min": vals.min().item(),
                    "max": vals.max().item(),
                }

        elapsed = time.perf_counter() - t0
        return UQResult(
            mean=mean, variance=variance, std=std,
            n_samples=n, qoi_statistics=qoi_stats, elapsed_seconds=elapsed,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Polynomial Chaos Expansion (PCE)
# ═══════════════════════════════════════════════════════════════════════════════


class PolynomialChaosExpansion:
    """
    Non-intrusive Polynomial Chaos Expansion via regression.

    Builds a spectral surrogate of the QoI as a polynomial of the
    uncertain parameters, then extracts mean/variance analytically.

    Parameters
    ----------
    polynomial_order : int
        Maximum total polynomial order.
    n_samples : int
        Number of training samples (should be >> number of basis functions).
    """

    def __init__(
        self,
        solver: Any,
        polynomial_order: int = 3,
        n_samples: int = 50,
        seed: Optional[int] = None,
    ) -> None:
        self._solver = solver
        self._order = polynomial_order
        self._n_samples = n_samples
        self._seed = seed

    def run(
        self,
        base_state: SimulationState,
        t_span: Tuple[float, float],
        dt: float,
        uncertain_params: List[ParameterDistribution],
        state_modifier: Callable[[SimulationState, Dict[str, float]], SimulationState],
        qoi_extractor: Callable[[SimulationState], float],
    ) -> Dict[str, Any]:
        """
        Build PCE surrogate and compute statistics.

        Returns dict with keys: coefficients, mean, variance, n_evals, basis_size.
        """
        if self._seed is not None:
            torch.manual_seed(self._seed)

        d = len(uncertain_params)
        n = self._n_samples

        # Sample points (uniformized to [-1, 1])
        xi = 2.0 * torch.rand(n, d, dtype=torch.float64) - 1.0

        # Map to physical parameter space
        param_arrays: Dict[str, Tensor] = {}
        for j, pd in enumerate(uncertain_params):
            u01 = (xi[:, j] + 1.0) / 2.0
            if pd.distribution == "uniform":
                param_arrays[pd.name] = u01 * (pd.upper - pd.lower) + pd.lower
            else:
                param_arrays[pd.name] = pd.sample(n)

        # Evaluate forward model
        qoi_values = torch.zeros(n, dtype=torch.float64)
        for i in range(n):
            params = {name: vals[i].item() for name, vals in param_arrays.items()}
            perturbed = state_modifier(base_state.clone(), params)
            result = self._solver.solve(perturbed, t_span, dt)
            qoi_values[i] = qoi_extractor(result.final_state)

        # Build Legendre polynomial basis on [-1, 1]^d
        # Using total-order multi-index set
        multi_indices = self._total_order_indices(d, self._order)
        n_basis = len(multi_indices)

        # Vandermonde matrix
        Phi = torch.ones(n, n_basis, dtype=torch.float64)
        for j_basis, alpha in enumerate(multi_indices):
            for dim in range(d):
                if alpha[dim] > 0:
                    Phi[:, j_basis] *= self._legendre(xi[:, dim], alpha[dim])

        # Least-squares regression
        coeffs, _, _, _ = torch.linalg.lstsq(Phi, qoi_values.unsqueeze(1))
        coeffs = coeffs.squeeze(1)

        # PCE statistics: mean = c_0, variance = sum(c_k² * norm_k²) for k>0
        mean_val = coeffs[0].item()
        # Legendre norm: integral of P_k² over [-1,1] = 2/(2k+1)
        variance = 0.0
        for j_basis in range(1, n_basis):
            alpha = multi_indices[j_basis]
            norm_sq = 1.0
            for dim in range(d):
                if alpha[dim] > 0:
                    norm_sq *= 2.0 / (2 * alpha[dim] + 1)
            variance += coeffs[j_basis].item() ** 2 * norm_sq

        return {
            "coefficients": coeffs,
            "mean": mean_val,
            "variance": variance,
            "std": variance ** 0.5,
            "n_evals": n,
            "basis_size": n_basis,
            "polynomial_order": self._order,
        }

    @staticmethod
    def _legendre(x: Tensor, order: int) -> Tensor:
        """Evaluate Legendre polynomial P_n(x) via recurrence."""
        if order == 0:
            return torch.ones_like(x)
        elif order == 1:
            return x.clone()
        p_prev = torch.ones_like(x)
        p_curr = x.clone()
        for n in range(2, order + 1):
            p_next = ((2 * n - 1) * x * p_curr - (n - 1) * p_prev) / n
            p_prev = p_curr
            p_curr = p_next
        return p_curr

    @staticmethod
    def _total_order_indices(d: int, p: int) -> List[List[int]]:
        """Generate total-order multi-index set: |α| ≤ p."""
        if d == 0:
            return [[]]
        indices: List[List[int]] = []
        for total in range(p + 1):
            PolynomialChaosExpansion._generate_multi(d, total, [], indices)
        return indices

    @staticmethod
    def _generate_multi(
        d: int, remaining: int, current: List[int], result: List[List[int]]
    ) -> None:
        if len(current) == d:
            if remaining == 0:
                result.append(list(current))
            return
        for k in range(remaining + 1):
            current.append(k)
            PolynomialChaosExpansion._generate_multi(d, remaining - k, current, result)
            current.pop()
