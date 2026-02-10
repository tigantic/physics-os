"""Model calibration from real surgical outcomes.

Bayesian-inspired parameter updating:
  - Compares predicted vs actual deformation fields
  - Updates tissue material parameters (stiffness, Poisson ratio, etc.)
  - Tracks per-parameter learning curves
  - Stores calibration provenance for reproducibility

Approach: Maximum a posteriori (MAP) estimation using Gauss-Newton
on the discrepancy between predicted and observed landmark / surface
displacements.  Prior distributions are drawn from literature values
stored in twin/materials.py.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..core.types import LandmarkType, StructureType, Vec3

logger = logging.getLogger(__name__)


@dataclass
class ParameterPrior:
    """Prior distribution for a tissue parameter."""
    name: str
    structure: StructureType
    mean: float
    std: float
    lower_bound: float
    upper_bound: float

    def log_prior(self, value: float) -> float:
        """Log of Gaussian prior, -inf outside bounds."""
        if value < self.lower_bound or value > self.upper_bound:
            return float("-inf")
        return -0.5 * ((value - self.mean) / max(self.std, 1e-30)) ** 2


@dataclass
class CalibrationResult:
    """Result of calibration on one or more cases."""
    case_ids: List[str]
    parameters_before: Dict[str, float]
    parameters_after: Dict[str, float]
    residual_before: float
    residual_after: float
    n_iterations: int
    converged: bool
    improvement_pct: float
    timestamp: float = 0.0
    parameter_history: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_ids": self.case_ids,
            "parameters_before": self.parameters_before,
            "parameters_after": self.parameters_after,
            "residual_before": self.residual_before,
            "residual_after": self.residual_after,
            "n_iterations": self.n_iterations,
            "converged": self.converged,
            "improvement_pct": self.improvement_pct,
            "timestamp": self.timestamp,
            "n_parameter_snapshots": len(self.parameter_history),
        }


# ── Default priors (drawn from literature) ──────────────────────────────────

_DEFAULT_PRIORS: Dict[str, ParameterPrior] = {
    "skin_E_kPa": ParameterPrior(
        name="skin_E_kPa",
        structure=StructureType.SKIN_ENVELOPE,
        mean=50.0, std=15.0, lower_bound=10.0, upper_bound=150.0,
    ),
    "skin_nu": ParameterPrior(
        name="skin_nu",
        structure=StructureType.SKIN_ENVELOPE,
        mean=0.45, std=0.03, lower_bound=0.3, upper_bound=0.499,
    ),
    "cartilage_LLC_E_MPa": ParameterPrior(
        name="cartilage_LLC_E_MPa",
        structure=StructureType.CARTILAGE_LOWER_LATERAL,
        mean=4.0, std=1.5, lower_bound=1.0, upper_bound=15.0,
    ),
    "cartilage_ULC_E_MPa": ParameterPrior(
        name="cartilage_ULC_E_MPa",
        structure=StructureType.CARTILAGE_UPPER_LATERAL,
        mean=5.0, std=2.0, lower_bound=1.0, upper_bound=20.0,
    ),
    "septum_E_MPa": ParameterPrior(
        name="septum_E_MPa",
        structure=StructureType.CARTILAGE_SEPTUM,
        mean=6.0, std=2.5, lower_bound=1.0, upper_bound=25.0,
    ),
    "soft_tissue_E_kPa": ParameterPrior(
        name="soft_tissue_E_kPa",
        structure=StructureType.FAT_SUBCUTANEOUS,
        mean=30.0, std=10.0, lower_bound=5.0, upper_bound=100.0,
    ),
    "periosteum_E_kPa": ParameterPrior(
        name="periosteum_E_kPa",
        structure=StructureType.PERIOSTEUM,
        mean=80.0, std=25.0, lower_bound=20.0, upper_bound=200.0,
    ),
    "healing_rate_multiplier": ParameterPrior(
        name="healing_rate_multiplier",
        structure=StructureType.FAT_SUBCUTANEOUS,
        mean=1.0, std=0.2, lower_bound=0.3, upper_bound=3.0,
    ),
}


class ModelCalibrator:
    """Calibrate simulation parameters from real outcomes.

    Method:
      Gauss-Newton minimisation of:
        J(theta) = ||d_predicted(theta) - d_actual||^2 + lambda * ||theta - theta_prior||^2_Sigma

      where d are displacement vectors at landmark locations,
      theta are calibratable material parameters,
      and Sigma is the prior covariance diagonal.
    """

    def __init__(
        self,
        simulator: Optional[Callable[..., Dict[str, np.ndarray]]] = None,
        priors: Optional[Dict[str, ParameterPrior]] = None,
        *,
        max_iterations: int = 30,
        convergence_tol: float = 1e-4,
        regularisation_weight: float = 0.1,
        finite_difference_step: float = 0.01,
    ) -> None:
        """
        Args:
            simulator: Callable that takes parameter dict and returns
                       {"landmark_name": displacement_vec3} predictions.
            priors: Parameter priors. Defaults to _DEFAULT_PRIORS.
            max_iterations: Maximum Gauss–Newton iterations.
            convergence_tol: Relative change in residual for convergence.
            regularisation_weight: Weight of prior penalty (lambda).
            finite_difference_step: Relative step for Jacobian approximation.
        """
        self._simulate = simulator
        self._priors = priors or dict(_DEFAULT_PRIORS)
        self._max_iter = max_iterations
        self._conv_tol = convergence_tol
        self._lambda = regularisation_weight
        self._fd_step = finite_difference_step

    def calibrate(
        self,
        actual_displacements: Dict[str, np.ndarray],
        initial_parameters: Dict[str, float],
        case_ids: Optional[List[str]] = None,
    ) -> CalibrationResult:
        """Run calibration loop.

        Args:
            actual_displacements: Measured displacements at landmark locations.
                Key is landmark name, value is 3-vector (mm).
            initial_parameters: Starting parameter values.
            case_ids: Source case identifiers for provenance.
        """
        if self._simulate is None:
            raise ValueError(
                "ModelCalibrator requires a simulator callable. "
                "Pass simulator=fn in __init__."
            )

        param_names = sorted(initial_parameters.keys())
        theta = np.array([initial_parameters[n] for n in param_names],
                         dtype=np.float64)
        theta_0 = theta.copy()

        # Build observation vector
        lm_names = sorted(actual_displacements.keys())
        y_obs = np.concatenate([actual_displacements[n] for n in lm_names])

        history: List[Dict[str, float]] = []
        prev_residual = float("inf")
        converged = False

        for iteration in range(self._max_iter):
            # Forward simulation at current theta
            params_dict = dict(zip(param_names, theta))
            history.append(dict(params_dict))

            pred = self._simulate(**params_dict)
            y_pred = np.concatenate([pred[n] for n in lm_names])

            # Residual
            r = y_pred - y_obs
            data_residual = float(np.sum(r ** 2))

            # Prior penalty
            prior_penalty = 0.0
            for i, pname in enumerate(param_names):
                if pname in self._priors:
                    p = self._priors[pname]
                    prior_penalty += ((theta[i] - p.mean) / max(p.std, 1e-30)) ** 2
            total_residual = data_residual + self._lambda * prior_penalty

            # Accept check
            rel_change = abs(prev_residual - total_residual) / max(prev_residual, 1e-30)
            if rel_change < self._conv_tol and iteration > 0:
                converged = True
                break
            prev_residual = total_residual

            # Compute Jacobian via finite differences
            J = self._compute_jacobian(param_names, theta, lm_names)

            # Gauss-Newton step with Tikhonov regularisation
            # (J^T J + lambda * diag(1/sigma^2)) * delta = -J^T r - lambda * (theta - mu)/sigma^2
            JtJ = J.T @ J
            reg_diag = np.zeros(len(param_names))
            reg_rhs = np.zeros(len(param_names))

            for i, pname in enumerate(param_names):
                if pname in self._priors:
                    p = self._priors[pname]
                    inv_var = 1.0 / max(p.std ** 2, 1e-30)
                    reg_diag[i] = self._lambda * inv_var
                    reg_rhs[i] = self._lambda * inv_var * (theta[i] - p.mean)

            A = JtJ + np.diag(reg_diag)
            b = -(J.T @ r) - reg_rhs

            try:
                delta = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                logger.warning("Singular matrix at iteration %d; using pseudoinverse", iteration)
                delta = np.linalg.lstsq(A, b, rcond=None)[0]

            # Line search (halving)
            step = 1.0
            for _ in range(10):
                theta_trial = theta + step * delta
                if self._within_bounds(param_names, theta_trial):
                    break
                step *= 0.5
            else:
                # Project onto bounds
                theta_trial = self._project_bounds(param_names, theta + step * delta)

            theta = theta_trial

        # Final evaluation
        final_params = dict(zip(param_names, theta))
        initial_residual = self._evaluate_residual(
            dict(zip(param_names, theta_0)), lm_names, y_obs,
        )
        final_residual = self._evaluate_residual(final_params, lm_names, y_obs)

        improvement = 0.0
        if initial_residual > 0:
            improvement = (initial_residual - final_residual) / initial_residual * 100.0

        result = CalibrationResult(
            case_ids=case_ids or [],
            parameters_before=dict(zip(param_names, theta_0)),
            parameters_after=final_params,
            residual_before=initial_residual,
            residual_after=final_residual,
            n_iterations=iteration + 1 if not converged else iteration + 1,
            converged=converged,
            improvement_pct=improvement,
            timestamp=time.time(),
            parameter_history=history,
        )

        logger.info(
            "Calibration %s after %d iterations: residual %.4f → %.4f (%.1f%%)",
            "converged" if converged else "reached max iter",
            result.n_iterations, initial_residual, final_residual, improvement,
        )

        return result

    def _compute_jacobian(
        self,
        param_names: List[str],
        theta: np.ndarray,
        lm_names: List[str],
    ) -> np.ndarray:
        """Finite-difference Jacobian of prediction w.r.t. parameters."""
        assert self._simulate is not None

        # Base prediction
        base_params = dict(zip(param_names, theta))
        base_pred = self._simulate(**base_params)
        y_base = np.concatenate([base_pred[n] for n in lm_names])

        n_obs = len(y_base)
        n_params = len(param_names)
        J = np.zeros((n_obs, n_params))

        for j in range(n_params):
            theta_pert = theta.copy()
            step = max(abs(theta[j]) * self._fd_step, 1e-8)
            theta_pert[j] += step

            pert_params = dict(zip(param_names, theta_pert))
            pert_pred = self._simulate(**pert_params)
            y_pert = np.concatenate([pert_pred[n] for n in lm_names])

            J[:, j] = (y_pert - y_base) / step

        return J

    def _evaluate_residual(
        self,
        params: Dict[str, float],
        lm_names: List[str],
        y_obs: np.ndarray,
    ) -> float:
        """Compute data residual norm."""
        assert self._simulate is not None
        pred = self._simulate(**params)
        y_pred = np.concatenate([pred[n] for n in lm_names])
        return float(np.sum((y_pred - y_obs) ** 2))

    def _within_bounds(
        self,
        param_names: List[str],
        theta: np.ndarray,
    ) -> bool:
        for i, pname in enumerate(param_names):
            if pname in self._priors:
                p = self._priors[pname]
                if theta[i] < p.lower_bound or theta[i] > p.upper_bound:
                    return False
        return True

    def _project_bounds(
        self,
        param_names: List[str],
        theta: np.ndarray,
    ) -> np.ndarray:
        result = theta.copy()
        for i, pname in enumerate(param_names):
            if pname in self._priors:
                p = self._priors[pname]
                result[i] = np.clip(result[i], p.lower_bound, p.upper_bound)
        return result

    def save_calibration(
        self,
        result: CalibrationResult,
        path: Path,
    ) -> None:
        """Persist calibration result to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

    @staticmethod
    def load_calibration(path: Path) -> Dict[str, Any]:
        """Load a previously saved calibration result."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
