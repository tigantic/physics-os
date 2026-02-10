"""Synthetic data augmentation for the case library.

Generates controlled variations of existing cases for:
  - Material parameter sensitivity studies
  - Anatomy perturbation (statistical shape models)
  - Surgical plan parameter sweeps
  - Training ML surrogates
"""

from __future__ import annotations

import copy
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.case_bundle import CaseBundle, PatientDemographics
from ..core.types import (
    MaterialModel,
    ProcedureType,
    SurfaceMesh,
    TissueProperties,
    Vec3,
)

logger = logging.getLogger(__name__)


@dataclass
class PerturbationSpec:
    """Specification for a single perturbation axis."""
    name: str
    param_path: str  # dot-separated path, e.g. "skin_nasal_tip.mu"
    distribution: str = "normal"  # normal, uniform, log_normal
    cv: float = 0.15  # coefficient of variation for normal
    low: float = 0.0  # for uniform
    high: float = 1.0  # for uniform
    n_samples: int = 10


@dataclass
class AnatomyPerturbation:
    """Perturbation of mesh geometry via displacement fields."""
    mode_weights: np.ndarray  # (n_modes,) — one weight per PCA mode
    displacement_field: Optional[np.ndarray] = None  # (N, 3) per-vertex displacement


@dataclass
class SyntheticVariant:
    """A generated variant with its parameter settings."""
    variant_id: str
    source_case_id: str
    material_overrides: Dict[str, Dict[str, float]] = field(default_factory=dict)
    anatomy_perturbation: Optional[AnatomyPerturbation] = None
    plan_overrides: Dict[str, Any] = field(default_factory=dict)
    seed: int = 0


class SyntheticAugmenter:
    """Generate synthetic variants of existing cases.

    Materials
    ---------
    Sample tissue material parameters from distributions around
    literature values.  Generates Latin-Hypercube or Sobol
    samples across the parameter space.

    Anatomy
    -------
    Apply PCA-based statistical shape model perturbations to
    the surface mesh.  Modes are derived from a registered
    population or specified directly.

    Plans
    -----
    Sweep surgical plan parameters (e.g. dorsal reduction 1–4mm)
    to evaluate sensitivity.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)
        self._seed = seed

    # ── Material perturbation ─────────────────────────────────

    def sample_material_params(
        self,
        base_params: Dict[str, float],
        specs: List[PerturbationSpec],
    ) -> List[Dict[str, float]]:
        """Generate material parameter samples.

        Parameters
        ----------
        base_params : dict
            Nominal material parameter values.
        specs : list of PerturbationSpec
            Perturbation specifications for each parameter.

        Returns
        -------
        List of dicts, one per sample, with perturbed parameter values.
        """
        n = max(s.n_samples for s in specs)
        samples: List[Dict[str, float]] = []

        for i in range(n):
            params = dict(base_params)
            for spec in specs:
                key = spec.param_path.split(".")[-1]
                if key not in base_params:
                    continue
                nominal = base_params[key]
                if spec.distribution == "normal":
                    sigma = nominal * spec.cv
                    val = self._rng.normal(nominal, sigma)
                    val = max(val, nominal * 0.1)  # floor at 10% of nominal
                elif spec.distribution == "uniform":
                    val = self._rng.uniform(spec.low, spec.high)
                elif spec.distribution == "log_normal":
                    sigma = np.sqrt(np.log(1 + spec.cv ** 2))
                    mu = np.log(nominal) - 0.5 * sigma ** 2
                    val = self._rng.lognormal(mu, sigma)
                else:
                    raise ValueError(f"Unknown distribution: {spec.distribution}")
                params[key] = float(val)
            samples.append(params)

        return samples

    def latin_hypercube(
        self,
        base_params: Dict[str, float],
        param_names: List[str],
        n_samples: int,
        cv: float = 0.15,
    ) -> List[Dict[str, float]]:
        """Generate Latin Hypercube Sampling (LHS) material samples.

        Parameters
        ----------
        base_params : dict
            Nominal parameter values.
        param_names : list of str
            Which parameters to vary.
        n_samples : int
            Number of samples.
        cv : float
            Coefficient of variation for each parameter.

        Returns
        -------
        List of parameter dicts.
        """
        d = len(param_names)
        if d == 0 or n_samples == 0:
            return [dict(base_params)]

        # Generate LHS grid
        intervals = np.zeros((n_samples, d))
        for j in range(d):
            perm = self._rng.permutation(n_samples)
            for i in range(n_samples):
                intervals[i, j] = (perm[i] + self._rng.uniform()) / n_samples

        # Map to parameter space (normal CDF inverse via erfinv approximation)
        samples: List[Dict[str, float]] = []
        for i in range(n_samples):
            params = dict(base_params)
            for j, name in enumerate(param_names):
                if name not in base_params:
                    continue
                nominal = base_params[name]
                sigma = nominal * cv
                # Inverse CDF of normal (using numpy)
                z = float(np.sqrt(2) * _erfinv(2 * intervals[i, j] - 1))
                val = nominal + sigma * z
                val = max(val, nominal * 0.01)
                params[name] = float(val)
            samples.append(params)

        return samples

    # ── Anatomy perturbation ──────────────────────────────────

    def perturb_mesh(
        self,
        mesh: SurfaceMesh,
        modes: np.ndarray,
        mean_shape: np.ndarray,
        n_variants: int = 5,
        sigma_scale: float = 1.0,
        eigenvalues: Optional[np.ndarray] = None,
    ) -> List[Tuple[SurfaceMesh, AnatomyPerturbation]]:
        """Generate anatomy variants using PCA shape model.

        Parameters
        ----------
        mesh : SurfaceMesh
            Base mesh (N vertices × 3).
        modes : ndarray, shape (n_modes, N*3)
            PCA mode vectors.
        mean_shape : ndarray, shape (N*3,)
            Mean shape vector.
        n_variants : int
            Number of variants to generate.
        sigma_scale : float
            Scale factor for mode weight standard deviations.
        eigenvalues : ndarray, optional
            Eigenvalues for each mode (determines variance).

        Returns
        -------
        List of (SurfaceMesh, AnatomyPerturbation) tuples.
        """
        n_modes = modes.shape[0]
        n_verts = len(mesh.vertices)

        if eigenvalues is None:
            eigenvalues = np.ones(n_modes)

        sigmas = np.sqrt(eigenvalues) * sigma_scale

        variants = []
        for _ in range(n_variants):
            weights = self._rng.normal(0, 1, size=n_modes) * sigmas
            displacement_flat = modes.T @ weights  # (N*3,)
            displacement = displacement_flat.reshape(n_verts, 3)

            new_verts = mesh.vertices + displacement.astype(np.float32)
            new_mesh = SurfaceMesh(
                vertices=new_verts,
                faces=mesh.faces.copy(),
            )
            new_mesh.compute_normals()

            perturbation = AnatomyPerturbation(
                mode_weights=weights,
                displacement_field=displacement.astype(np.float32),
            )
            variants.append((new_mesh, perturbation))

        return variants

    # ── Plan parameter sweeps ─────────────────────────────────

    def plan_parameter_sweep(
        self,
        param_ranges: Dict[str, Tuple[float, float, int]],
    ) -> List[Dict[str, float]]:
        """Generate plan parameter sweep grid.

        Parameters
        ----------
        param_ranges : dict
            {param_name: (low, high, n_steps)} for each parameter.

        Returns
        -------
        List of parameter dicts (full combinatorial grid).
        """
        import itertools

        names = list(param_ranges.keys())
        grids = []
        for name in names:
            low, high, n = param_ranges[name]
            grids.append(np.linspace(low, high, n))

        combos = list(itertools.product(*grids))
        return [dict(zip(names, combo)) for combo in combos]

    # ── Full variant generation ────────────────────────────────

    def generate_variants(
        self,
        source_bundle: CaseBundle,
        library_root: str | Path,
        *,
        material_specs: Optional[List[PerturbationSpec]] = None,
        n_material_variants: int = 0,
        n_anatomy_variants: int = 0,
        plan_sweep: Optional[Dict[str, Tuple[float, float, int]]] = None,
    ) -> List[SyntheticVariant]:
        """Generate synthetic variant descriptors from a source case.

        Does NOT create new CaseBundles — returns variant specs that
        can be applied during simulation.
        """
        variants: List[SyntheticVariant] = []
        source_id = source_bundle.case_id

        if material_specs and n_material_variants > 0:
            # Get base material params from bundle
            try:
                base_materials = source_bundle.load_json("materials", subdir="models")
            except FileNotFoundError:
                base_materials = {}

            for spec in material_specs:
                tissue = spec.param_path.split(".")[0]
                base = base_materials.get(tissue, {})
                samples = self.sample_material_params(base, [spec])
                for s in samples[:n_material_variants]:
                    vid = uuid.uuid4().hex[:8]
                    variants.append(SyntheticVariant(
                        variant_id=vid,
                        source_case_id=source_id,
                        material_overrides={tissue: s},
                        seed=int(self._rng.integers(0, 2**31)),
                    ))

        if plan_sweep:
            combos = self.plan_parameter_sweep(plan_sweep)
            for combo in combos:
                vid = uuid.uuid4().hex[:8]
                variants.append(SyntheticVariant(
                    variant_id=vid,
                    source_case_id=source_id,
                    plan_overrides=combo,
                    seed=int(self._rng.integers(0, 2**31)),
                ))

        logger.info(
            "Generated %d synthetic variants from case %s",
            len(variants), source_id,
        )
        return variants


# ── Utility functions ─────────────────────────────────────────────

def _erfinv(x: float) -> float:
    """Approximate inverse error function (Winitzki 2008).

    Accurate to ~0.01% for |x| < 0.99.
    """
    if abs(x) >= 1.0:
        return float("inf") if x > 0 else float("-inf")
    a = 0.147
    ln_term = np.log(1 - x * x)
    term1 = 2.0 / (np.pi * a) + ln_term / 2.0
    sign = 1.0 if x >= 0 else -1.0
    return sign * np.sqrt(np.sqrt(term1 * term1 - ln_term / a) - term1)
