"""
5.11 — Multi-Fidelity Surrogate Framework
===========================================

Combines models at different fidelity levels (coarse mesh / fine mesh,
reduced-order / full-order) into a single surrogate with principled
UQ propagation across levels.

Components:
    * FidelityLevel — descriptor for a simulation fidelity
    * MultiFidelityGP — multi-fidelity Gaussian Process (AR1 model)
    * CoKrigingSurrogate — co-Kriging (Kennedy-O'Hagan) surrogate
    * MultiFidelityEnsemble — ensemble across fidelity levels
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── Fidelity level descriptor ────────────────────────────────────

@dataclass
class FidelityLevel:
    """Description of a simulation fidelity."""
    name: str
    cost: float               # relative wallclock cost
    accuracy: float           # relative accuracy (1.0 = truth)
    n_dofs: int = 0           # degrees of freedom
    turnaround_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not 0 < self.accuracy <= 1.0:
            raise ValueError(f"accuracy must be in (0, 1]: got {self.accuracy}")


# ── RBF kernel ────────────────────────────────────────────────────

def rbf_kernel(
    X1: np.ndarray, X2: np.ndarray,
    length_scale: float = 1.0, variance: float = 1.0,
) -> np.ndarray:
    """Squared-exponential (RBF) kernel. Returns (N1, N2)."""
    sq_dist = (
        np.sum(X1 ** 2, axis=1, keepdims=True)
        + np.sum(X2 ** 2, axis=1, keepdims=True).T
        - 2 * X1 @ X2.T
    )
    return variance * np.exp(-0.5 * sq_dist / (length_scale ** 2))


def matern32_kernel(
    X1: np.ndarray, X2: np.ndarray,
    length_scale: float = 1.0, variance: float = 1.0,
) -> np.ndarray:
    """Matérn 3/2 kernel."""
    dist = np.sqrt(np.maximum(
        np.sum(X1 ** 2, axis=1, keepdims=True)
        + np.sum(X2 ** 2, axis=1, keepdims=True).T
        - 2 * X1 @ X2.T,
        0,
    ))
    r = math.sqrt(3) * dist / length_scale
    return variance * (1 + r) * np.exp(-r)


# ── Single-fidelity GP ───────────────────────────────────────────

class SingleFidelityGP:
    """Gaussian process regression (exact, small-data)."""

    def __init__(
        self,
        length_scale: float = 1.0,
        variance: float = 1.0,
        noise: float = 1e-6,
        kernel: str = "rbf",
    ) -> None:
        self.length_scale = length_scale
        self.variance = variance
        self.noise = noise
        self._kernel_fn = rbf_kernel if kernel == "rbf" else matern32_kernel
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._K_inv: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit GP to data.  X: (N, d), y: (N,) or (N, 1)."""
        self._X = X.astype(np.float64)
        self._y = y.ravel().astype(np.float64)
        K = self._kernel_fn(self._X, self._X, self.length_scale, self.variance)
        K += self.noise * np.eye(len(X))
        self._K_inv = np.linalg.inv(K)
        self._alpha = self._K_inv @ self._y

    def predict(self, X_star: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance at test points."""
        if self._X is None:
            raise RuntimeError("GP not fitted")
        X_star = X_star.astype(np.float64)
        Ks = self._kernel_fn(self._X, X_star, self.length_scale, self.variance)
        Kss = self._kernel_fn(X_star, X_star, self.length_scale, self.variance)

        mean = Ks.T @ self._alpha
        var_full = Kss - Ks.T @ self._K_inv @ Ks
        var = np.diag(var_full).clip(0)
        return mean.astype(np.float32), var.astype(np.float32)

    def log_marginal_likelihood(self) -> float:
        """Compute log marginal likelihood (for hyper-param tuning)."""
        if self._X is None:
            raise RuntimeError("GP not fitted")
        K = self._kernel_fn(self._X, self._X, self.length_scale, self.variance)
        K += self.noise * np.eye(len(self._X))
        sign, logdet = np.linalg.slogdet(K)
        n = len(self._y)
        return float(
            -0.5 * self._y @ np.linalg.solve(K, self._y)
            - 0.5 * logdet
            - 0.5 * n * math.log(2 * math.pi)
        )


# ── Multi-fidelity GP (AR1) ──────────────────────────────────────

class MultiFidelityGP:
    """Autoregressive multi-fidelity GP (Kennedy & O'Hagan, 2000).

    Model: f_high(x) = ρ · f_low(x) + δ(x)

    Where ρ is a scaling factor learned from data, and δ is a
    GP capturing the discrepancy.
    """

    def __init__(
        self,
        levels: Optional[List[FidelityLevel]] = None,
        kernel: str = "rbf",
    ) -> None:
        self.levels = levels or [
            FidelityLevel("coarse", cost=1.0, accuracy=0.7),
            FidelityLevel("fine", cost=10.0, accuracy=1.0),
        ]
        self._low_gp = SingleFidelityGP(kernel=kernel)
        self._delta_gp = SingleFidelityGP(kernel=kernel)
        self._rho: float = 1.0

    def fit(
        self,
        X_low: np.ndarray, y_low: np.ndarray,
        X_high: np.ndarray, y_high: np.ndarray,
    ) -> Dict[str, float]:
        """Fit multi-fidelity model.

        Parameters
        ----------
        X_low : (N_low, d)  low-fidelity inputs
        y_low : (N_low,)    low-fidelity outputs
        X_high: (N_high, d) high-fidelity inputs
        y_high: (N_high,)   high-fidelity outputs
        """
        # 1) Fit low-fidelity GP
        self._low_gp.fit(X_low, y_low)

        # 2) Estimate ρ via least-squares on shared points
        f_low_at_high, _ = self._low_gp.predict(X_high)
        # ρ = (f_low^T y_high) / (f_low^T f_low)
        self._rho = float(
            f_low_at_high @ y_high.ravel()
            / (f_low_at_high @ f_low_at_high + 1e-12)
        )

        # 3) Fit discrepancy GP
        delta = y_high.ravel() - self._rho * f_low_at_high
        self._delta_gp.fit(X_high, delta)

        return {
            "rho": self._rho,
            "lml_low": self._low_gp.log_marginal_likelihood(),
            "lml_delta": self._delta_gp.log_marginal_likelihood(),
        }

    def predict(self, X_star: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Multi-fidelity prediction."""
        f_low, var_low = self._low_gp.predict(X_star)
        delta_mean, delta_var = self._delta_gp.predict(X_star)

        mean = self._rho * f_low + delta_mean
        var = self._rho ** 2 * var_low + delta_var
        return mean, var

    @property
    def rho(self) -> float:
        return self._rho


# ── Co-Kriging Surrogate ─────────────────────────────────────────

class CoKrigingSurrogate:
    """Multi-fidelity co-Kriging surrogate (generalised AR model).

    Supports N fidelity levels with chained AR corrections:
    f_k(x) = ρ_k · f_{k-1}(x) + δ_k(x)
    """

    def __init__(self, levels: List[FidelityLevel], kernel: str = "rbf") -> None:
        if len(levels) < 2:
            raise ValueError("Need at least 2 fidelity levels")
        self.levels = sorted(levels, key=lambda l: l.cost)
        self._gps: List[SingleFidelityGP] = [
            SingleFidelityGP(kernel=kernel)
            for _ in levels
        ]
        self._rhos: List[float] = [1.0] * (len(levels) - 1)

    def fit(
        self,
        data: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, Any]:
        """Fit all levels.

        Parameters
        ----------
        data : [(X_0, y_0), (X_1, y_1), ...] ordered by fidelity
        """
        if len(data) != len(self.levels):
            raise ValueError("data must match number of fidelity levels")

        # Fit lowest fidelity
        self._gps[0].fit(data[0][0], data[0][1])
        info: Dict[str, Any] = {"rhos": [], "lmls": [self._gps[0].log_marginal_likelihood()]}

        # Chain higher fidelities
        for k in range(1, len(self.levels)):
            X_k, y_k = data[k]
            prev_mean, _ = self._gps[k - 1].predict(X_k)
            rho = float(
                prev_mean @ y_k.ravel()
                / (prev_mean @ prev_mean + 1e-12)
            )
            self._rhos[k - 1] = rho

            delta = y_k.ravel() - rho * prev_mean
            self._gps[k].fit(X_k, delta)
            info["rhos"].append(rho)
            info["lmls"].append(self._gps[k].log_marginal_likelihood())

        return info

    def predict(self, X_star: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict at highest fidelity."""
        mean, var = self._gps[0].predict(X_star)
        for k in range(1, len(self.levels)):
            rho = self._rhos[k - 1]
            d_mean, d_var = self._gps[k].predict(X_star)
            mean = rho * mean + d_mean
            var = rho ** 2 * var + d_var
        return mean, var


# ── Multi-fidelity ensemble ──────────────────────────────────────

class MultiFidelityEnsemble:
    """Ensemble wrapper that picks optimal fidelity for a query budget."""

    def __init__(self, model: CoKrigingSurrogate) -> None:
        self.model = model

    def predict_within_budget(
        self,
        X_star: np.ndarray,
        budget: float,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """Predict using highest affordable fidelity.

        Returns (mean, var, level_name).
        """
        # Determine affordable level
        chosen = self.model.levels[0]
        for level in self.model.levels:
            if level.cost * len(X_star) <= budget:
                chosen = level

        # Always use the full model (it's already trained)
        mean, var = self.model.predict(X_star)
        return mean, var, chosen.name

    def uncertainty_weighted_prediction(
        self,
        X_star: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty from all levels combined."""
        return self.model.predict(X_star)


__all__ = [
    "FidelityLevel",
    "SingleFidelityGP",
    "MultiFidelityGP",
    "CoKrigingSurrogate",
    "MultiFidelityEnsemble",
    "rbf_kernel",
    "matern32_kernel",
]
