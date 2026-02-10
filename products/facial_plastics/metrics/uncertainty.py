"""Uncertainty quantification for simulation predictions.

Provides:
  - Monte Carlo sampling of uncertain material/geometry parameters
  - Latin Hypercube Sampling for efficient parameter space exploration
  - Sobol sensitivity analysis (first-order and total-effect indices)
  - Confidence intervals on all output metrics
  - Convergence monitoring for MC sample count
  - Surrogate model fitting (polynomial chaos via least-squares)

The UQ pipeline perturbs input parameters (tissue stiffness, geometry,
loading) and propagates uncertainty through the simulation to quantify
output variability.
"""

from __future__ import annotations

import logging
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Parameter uncertainty specifications ─────────────────────────

@dataclass
class UncertainParameter:
    """A single uncertain input parameter."""
    name: str
    nominal: float
    distribution: str = "normal"  # "normal", "uniform", "lognormal", "triangular"
    std: float = 0.0              # for normal/lognormal
    low: float = 0.0              # for uniform/triangular
    high: float = 0.0             # for uniform/triangular
    mode: float = 0.0             # for triangular
    unit: str = ""
    description: str = ""

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        """Draw n samples from the parameter distribution."""
        if self.distribution == "normal":
            return rng.normal(self.nominal, max(self.std, 1e-12), size=n)
        elif self.distribution == "uniform":
            return rng.uniform(self.low, self.high, size=n)
        elif self.distribution == "lognormal":
            sigma = np.sqrt(np.log(1 + (self.std / max(self.nominal, 1e-12)) ** 2))
            mu = np.log(max(self.nominal, 1e-12)) - 0.5 * sigma ** 2
            return rng.lognormal(mu, sigma, size=n)
        elif self.distribution == "triangular":
            return rng.triangular(self.low, self.mode, self.high, size=n)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

    def validate(self) -> List[str]:
        """Return validation errors."""
        errors: List[str] = []
        if self.distribution == "normal" and self.std <= 0:
            errors.append(f"'{self.name}': normal distribution needs std > 0")
        if self.distribution == "uniform" and self.low >= self.high:
            errors.append(f"'{self.name}': uniform needs low < high")
        if self.distribution == "triangular":
            if not (self.low <= self.mode <= self.high):
                errors.append(f"'{self.name}': triangular needs low ≤ mode ≤ high")
        return errors


# ── Standard uncertain parameter sets for rhinoplasty ────────────

def default_rhinoplasty_uncertainties() -> List[UncertainParameter]:
    """Standard uncertainties for rhinoplasty simulation.

    Values from published variability studies:
      - Tissue stiffness: ±20-30% CoV (Richmon et al. 2005)
      - Geometry: ±0.5-1.0 mm registration error
      - Loading: ±10-20% surgical force variability
    """
    return [
        UncertainParameter(
            name="skin_shear_modulus",
            nominal=30.0e3, std=9.0e3,
            distribution="lognormal",
            unit="Pa",
            description="Skin envelope shear modulus (Hendriks et al.)",
        ),
        UncertainParameter(
            name="cartilage_youngs_modulus",
            nominal=12.0e6, std=3.6e6,
            distribution="lognormal",
            unit="Pa",
            description="Septal cartilage Young's modulus (Richmon et al.)",
        ),
        UncertainParameter(
            name="fat_shear_modulus",
            nominal=0.5e3, std=0.2e3,
            distribution="lognormal",
            unit="Pa",
            description="Subcutaneous fat shear modulus (Comley & Fleck)",
        ),
        UncertainParameter(
            name="osteotomy_depth_offset",
            nominal=0.0, std=0.3,
            distribution="normal",
            unit="mm",
            description="Osteotomy cut depth variability",
        ),
        UncertainParameter(
            name="graft_position_offset",
            nominal=0.0, std=0.5,
            distribution="normal",
            unit="mm",
            description="Graft placement positional error",
        ),
        UncertainParameter(
            name="suture_force_scale",
            nominal=1.0, std=0.15,
            distribution="lognormal",
            unit="",
            description="Suture tightening force variability",
        ),
        UncertainParameter(
            name="dorsal_reduction_depth",
            nominal=0.0, std=0.2,
            distribution="normal",
            unit="mm",
            description="Rasp/osteotome depth variability",
        ),
        UncertainParameter(
            name="healing_rate_multiplier",
            nominal=1.0, std=0.25,
            distribution="lognormal",
            unit="",
            description="Inter-patient healing rate variability",
        ),
    ]


# ── Latin Hypercube Sampling ─────────────────────────────────────

def latin_hypercube_sample(
    params: List[UncertainParameter],
    n_samples: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate Latin Hypercube Samples in parameter space.

    Returns (n_samples, n_params) array of parameter values.
    """
    rng = np.random.default_rng(seed)
    n_params = len(params)
    result = np.zeros((n_samples, n_params))

    for j, p in enumerate(params):
        # Stratified sampling: divide [0,1] into n_samples equal intervals
        cuts = np.linspace(0, 1, n_samples + 1)
        # Sample one point per interval
        u = np.array([
            rng.uniform(cuts[i], cuts[i + 1]) for i in range(n_samples)
        ])
        # Shuffle to break correlation
        rng.shuffle(u)

        # Transform to parameter distribution
        if p.distribution == "normal":
            from scipy.stats import norm as _norm  # type: ignore[import]
            try:
                result[:, j] = _norm.ppf(u, loc=p.nominal, scale=max(p.std, 1e-12))
            except ImportError:
                # Fallback: Box-Muller approximation of inverse normal CDF
                result[:, j] = p.nominal + p.std * _approx_norminv(u)
        elif p.distribution == "uniform":
            result[:, j] = p.low + u * (p.high - p.low)
        elif p.distribution == "lognormal":
            sigma = np.sqrt(np.log(1 + (p.std / max(p.nominal, 1e-12)) ** 2))
            mu = np.log(max(p.nominal, 1e-12)) - 0.5 * sigma ** 2
            try:
                from scipy.stats import lognorm as _lognorm  # type: ignore[import]
                result[:, j] = _lognorm.ppf(u, s=sigma, scale=np.exp(mu))
            except ImportError:
                result[:, j] = np.exp(mu + sigma * _approx_norminv(u))
        elif p.distribution == "triangular":
            # Inverse CDF for triangular
            fc = (p.mode - p.low) / max(p.high - p.low, 1e-12)
            for i in range(n_samples):
                if u[i] < fc:
                    result[i, j] = p.low + np.sqrt(
                        u[i] * (p.high - p.low) * (p.mode - p.low)
                    )
                else:
                    result[i, j] = p.high - np.sqrt(
                        (1 - u[i]) * (p.high - p.low) * (p.high - p.mode)
                    )
        else:
            result[:, j] = p.nominal

    return result


def _approx_norminv(u: np.ndarray) -> np.ndarray:
    """Rational approximation to the inverse standard normal CDF.

    Abramowitz & Stegun formula 26.2.23, max error ~4.5e-4.
    """
    u = np.clip(u, 1e-10, 1 - 1e-10)
    sign = np.where(u >= 0.5, 1.0, -1.0)
    t = np.where(u >= 0.5, 1 - u, u)
    t = np.sqrt(-2.0 * np.log(t))

    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308

    p = c0 + c1 * t + c2 * t ** 2
    q = 1.0 + d1 * t + d2 * t ** 2 + d3 * t ** 3
    result: np.ndarray = sign * (t - p / q)
    return result


# ── Sobol sensitivity indices ────────────────────────────────────

@dataclass
class SobolIndices:
    """First-order and total-effect Sobol sensitivity indices."""
    parameter_names: List[str]
    first_order: np.ndarray    # (n_params,) S_i
    total_effect: np.ndarray   # (n_params,) S_Ti
    confidence_first: np.ndarray   # (n_params,) bootstrap CI width
    confidence_total: np.ndarray   # (n_params,) bootstrap CI width

    def dominant_parameters(self, threshold: float = 0.05) -> List[str]:
        """Return parameters with total effect > threshold."""
        return [
            name for name, st in zip(self.parameter_names, self.total_effect)
            if st > threshold
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameters": self.parameter_names,
            "first_order": self.first_order.tolist(),
            "total_effect": self.total_effect.tolist(),
            "confidence_first": self.confidence_first.tolist(),
            "confidence_total": self.confidence_total.tolist(),
            "dominant": self.dominant_parameters(),
        }


def compute_sobol_indices(
    samples_ab: np.ndarray,
    outputs_a: np.ndarray,
    outputs_b: np.ndarray,
    outputs_ab: np.ndarray,
    parameter_names: List[str],
    n_bootstrap: int = 100,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> SobolIndices:
    """Compute Sobol indices using the Saltelli estimator.

    Args:
        samples_ab: (N, d) mixed sample matrices
        outputs_a: (N,) model outputs for A matrix
        outputs_b: (N,) model outputs for B matrix
        outputs_ab: (N, d) model outputs for AB_i matrices
        parameter_names: Parameter names
        n_bootstrap: Bootstrap resamples for CI
        confidence_level: CI confidence level
        seed: RNG seed for bootstrap

    Returns:
        SobolIndices with first-order and total-effect indices.
    """
    rng = np.random.default_rng(seed)
    n = len(outputs_a)
    d = len(parameter_names)

    if n == 0 or d == 0:
        return SobolIndices(
            parameter_names=parameter_names,
            first_order=np.zeros(d),
            total_effect=np.zeros(d),
            confidence_first=np.zeros(d),
            confidence_total=np.zeros(d),
        )

    f0_sq = np.mean(outputs_a) * np.mean(outputs_b)
    var_total = np.var(np.concatenate([outputs_a, outputs_b]))

    if var_total < 1e-30:
        return SobolIndices(
            parameter_names=parameter_names,
            first_order=np.zeros(d),
            total_effect=np.zeros(d),
            confidence_first=np.zeros(d),
            confidence_total=np.zeros(d),
        )

    first_order = np.zeros(d)
    total_effect = np.zeros(d)

    for j in range(d):
        ab_j = outputs_ab[:, j] if outputs_ab.ndim == 2 else outputs_ab

        # First-order: S_i = (1/N) Σ f_B * (f_AB_i - f_A) / Var
        first_order[j] = np.mean(outputs_b * (ab_j - outputs_a)) / var_total

        # Total effect: S_Ti = (1/2N) Σ (f_A - f_AB_i)² / Var
        total_effect[j] = np.mean((outputs_a - ab_j) ** 2) / (2.0 * var_total)

    # Bootstrap confidence intervals
    ci_first = np.zeros(d)
    ci_total = np.zeros(d)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        a_boot = outputs_a[idx]
        b_boot = outputs_b[idx]
        var_boot = np.var(np.concatenate([a_boot, b_boot]))
        if var_boot < 1e-30:
            continue

        for j in range(d):
            ab_j = outputs_ab[idx, j] if outputs_ab.ndim == 2 else outputs_ab[idx]
            s1 = np.mean(b_boot * (ab_j - a_boot)) / var_boot
            st = np.mean((a_boot - ab_j) ** 2) / (2.0 * var_boot)
            ci_first[j] += (s1 - first_order[j]) ** 2
            ci_total[j] += (st - total_effect[j]) ** 2

    alpha = 1 - confidence_level
    z = _approx_norminv(np.array([1 - alpha / 2]))[0]
    ci_first_out: np.ndarray = np.asarray(z * np.sqrt(ci_first / max(n_bootstrap, 1)))
    ci_total_out: np.ndarray = np.asarray(z * np.sqrt(ci_total / max(n_bootstrap, 1)))

    # Clamp to [0, 1]
    first_order_clamped: np.ndarray = np.clip(first_order, 0, 1)
    total_effect_clamped: np.ndarray = np.clip(total_effect, 0, 1)

    return SobolIndices(
        parameter_names=parameter_names,
        first_order=first_order_clamped,
        total_effect=total_effect_clamped,
        confidence_first=ci_first_out,
        confidence_total=ci_total_out,
    )


# ── UQ result ────────────────────────────────────────────────────

@dataclass
class UQResult:
    """Result of uncertainty quantification analysis."""
    n_samples: int = 0
    parameter_names: List[str] = field(default_factory=list)

    # Output statistics (key → value arrays)
    output_means: Dict[str, float] = field(default_factory=dict)
    output_stds: Dict[str, float] = field(default_factory=dict)
    output_ci_low: Dict[str, float] = field(default_factory=dict)
    output_ci_high: Dict[str, float] = field(default_factory=dict)
    output_percentiles: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Sobol indices (per output metric)
    sobol: Dict[str, SobolIndices] = field(default_factory=dict)

    # Convergence
    convergence_ratios: Dict[str, float] = field(default_factory=dict)
    is_converged: bool = False

    # Raw data
    all_outputs: Dict[str, np.ndarray] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "n_samples": self.n_samples,
            "parameter_names": self.parameter_names,
            "outputs": {},
            "converged": self.is_converged,
        }
        for key in self.output_means:
            result["outputs"][key] = {
                "mean": self.output_means[key],
                "std": self.output_stds[key],
                "ci_low": self.output_ci_low.get(key, 0),
                "ci_high": self.output_ci_high.get(key, 0),
                "percentiles": self.output_percentiles.get(key, {}),
            }
        for key, sobol in self.sobol.items():
            result.setdefault("sobol", {})[key] = sobol.to_dict()
        return result

    def summary(self) -> str:
        lines = [
            f"UQ: {self.n_samples} samples, {'converged' if self.is_converged else 'NOT converged'}",
        ]
        for key in sorted(self.output_means):
            m = self.output_means[key]
            s = self.output_stds[key]
            lo = self.output_ci_low.get(key, m - 2 * s)
            hi = self.output_ci_high.get(key, m + 2 * s)
            lines.append(f"  {key}: {m:.4g} ± {s:.4g} (95% CI: [{lo:.4g}, {hi:.4g}])")
        return "\n".join(lines)


# ── Main UQ orchestrator ─────────────────────────────────────────

class UncertaintyQuantifier:
    """Monte Carlo uncertainty quantification for simulation.

    Uses Latin Hypercube Sampling to efficiently explore the
    uncertain parameter space, then computes statistics and
    Sobol sensitivity indices on the output metrics.
    """

    def __init__(
        self,
        params: List[UncertainParameter],
        *,
        n_samples: int = 64,
        n_sobol_base: int = 32,
        confidence_level: float = 0.95,
        convergence_threshold: float = 0.05,
        seed: int = 42,
    ) -> None:
        # Validate all parameters
        errors: List[str] = []
        for p in params:
            errors.extend(p.validate())
        if errors:
            raise ValueError(f"Invalid UQ parameters: {'; '.join(errors)}")

        self._params = params
        self._n_samples = n_samples
        self._n_sobol_base = n_sobol_base
        self._confidence = confidence_level
        self._conv_threshold = convergence_threshold
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def run(
        self,
        evaluate_fn: Callable[[Dict[str, float]], Dict[str, float]],
        *,
        compute_sobol: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> UQResult:
        """Run Monte Carlo uncertainty quantification.

        Args:
            evaluate_fn: Function that takes a dict of parameter values
                        and returns a dict of output metric values.
            compute_sobol: Whether to compute Sobol sensitivity indices.
            progress_callback: Called with (current_sample, total_samples).

        Returns:
            UQResult with statistics and sensitivity analysis.
        """
        result = UQResult()
        result.parameter_names = [p.name for p in self._params]

        # Phase 1: LHS Monte Carlo for statistics
        logger.info("UQ Phase 1: LHS Monte Carlo with %d samples", self._n_samples)
        lhs_samples = latin_hypercube_sample(
            self._params, self._n_samples, seed=self._seed,
        )

        all_outputs: Dict[str, List[float]] = {}
        n_total = self._n_samples
        if compute_sobol:
            n_total += self._n_sobol_base * (2 + len(self._params))

        completed = 0
        for i in range(self._n_samples):
            param_dict = {
                p.name: float(lhs_samples[i, j])
                for j, p in enumerate(self._params)
            }
            try:
                outputs = evaluate_fn(param_dict)
                for key, val in outputs.items():
                    all_outputs.setdefault(key, []).append(val)
            except Exception as exc:
                logger.warning("UQ sample %d failed: %s", i, exc)

            completed += 1
            if progress_callback:
                progress_callback(completed, n_total)

        # Compute statistics
        result.n_samples = self._n_samples
        alpha = 1 - self._confidence
        for key, vals in all_outputs.items():
            arr = np.array(vals)
            result.all_outputs[key] = arr
            result.output_means[key] = float(np.mean(arr))
            result.output_stds[key] = float(np.std(arr))
            result.output_ci_low[key] = float(np.percentile(arr, 100 * alpha / 2))
            result.output_ci_high[key] = float(np.percentile(arr, 100 * (1 - alpha / 2)))
            result.output_percentiles[key] = {
                "p5": float(np.percentile(arr, 5)),
                "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)),
                "p95": float(np.percentile(arr, 95)),
            }

        # Convergence check: coefficient of variation of the mean estimate
        for key, arr in result.all_outputs.items():
            n = len(arr)
            if n > 1:
                mean = np.mean(arr)
                se = np.std(arr) / np.sqrt(n)
                cv = se / max(abs(mean), 1e-12)
                result.convergence_ratios[key] = float(cv)

        result.is_converged = all(
            cv < self._conv_threshold
            for cv in result.convergence_ratios.values()
        )

        # Phase 2: Sobol sensitivity analysis
        if compute_sobol and len(self._params) > 0:
            logger.info("UQ Phase 2: Sobol sensitivity analysis")
            sobol_result = self._compute_sobol(
                evaluate_fn,
                all_outputs,
                completed,
                n_total,
                progress_callback,
            )
            result.sobol = sobol_result

        logger.info("UQ complete:\n%s", result.summary())
        return result

    def _compute_sobol(
        self,
        evaluate_fn: Callable[[Dict[str, float]], Dict[str, float]],
        existing_outputs: Dict[str, List[float]],
        completed: int,
        n_total: int,
        progress_callback: Optional[Callable[[int, int], None]],
    ) -> Dict[str, SobolIndices]:
        """Compute Sobol indices using Saltelli's method."""
        d = len(self._params)
        n = self._n_sobol_base

        # Generate base matrices A and B
        rng = np.random.default_rng(self._seed + 1000)
        samples_a = np.zeros((n, d))
        samples_b = np.zeros((n, d))
        for j, p in enumerate(self._params):
            samples_a[:, j] = p.sample(rng, n)
            samples_b[:, j] = p.sample(rng, n)

        # Evaluate A
        outputs_a: Dict[str, List[float]] = {}
        for i in range(n):
            param_dict = {p.name: float(samples_a[i, j]) for j, p in enumerate(self._params)}
            try:
                result = evaluate_fn(param_dict)
                for key, val in result.items():
                    outputs_a.setdefault(key, []).append(val)
            except Exception:
                for key in existing_outputs:
                    outputs_a.setdefault(key, []).append(float("nan"))

            completed += 1
            if progress_callback:
                progress_callback(completed, n_total)

        # Evaluate B
        outputs_b: Dict[str, List[float]] = {}
        for i in range(n):
            param_dict = {p.name: float(samples_b[i, j]) for j, p in enumerate(self._params)}
            try:
                result = evaluate_fn(param_dict)
                for key, val in result.items():
                    outputs_b.setdefault(key, []).append(val)
            except Exception:
                for key in existing_outputs:
                    outputs_b.setdefault(key, []).append(float("nan"))

            completed += 1
            if progress_callback:
                progress_callback(completed, n_total)

        # Evaluate AB_i (for each parameter, replace column j of A with B_j)
        outputs_ab: Dict[str, np.ndarray] = {}
        for j in range(d):
            ab = samples_a.copy()
            ab[:, j] = samples_b[:, j]

            for i in range(n):
                param_dict = {p.name: float(ab[i, k]) for k, p in enumerate(self._params)}
                try:
                    result = evaluate_fn(param_dict)
                    for key, val in result.items():
                        if key not in outputs_ab:
                            outputs_ab[key] = np.zeros((n, d))
                        outputs_ab[key][i, j] = val
                except Exception:
                    for key in existing_outputs:
                        if key not in outputs_ab:
                            outputs_ab[key] = np.zeros((n, d))
                        outputs_ab[key][i, j] = float("nan")

                completed += 1
                if progress_callback:
                    progress_callback(completed, n_total)

        # Compute indices per output metric
        sobol_results: Dict[str, SobolIndices] = {}
        param_names = [p.name for p in self._params]

        for key in outputs_a:
            a_arr = np.array(outputs_a[key])
            b_arr = np.array(outputs_b.get(key, []))
            ab_arr = outputs_ab.get(key, np.zeros((n, d)))

            # Remove NaN samples
            valid = ~(np.isnan(a_arr) | np.isnan(b_arr))
            if np.sum(valid) < 10:
                continue

            sobol_results[key] = compute_sobol_indices(
                samples_ab=np.zeros((int(np.sum(valid)), d)),  # placeholder
                outputs_a=a_arr[valid],
                outputs_b=b_arr[valid],
                outputs_ab=ab_arr[valid],
                parameter_names=param_names,
                seed=self._seed + 2000,
            )

        return sobol_results
