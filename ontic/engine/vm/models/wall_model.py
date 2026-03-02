"""QTT Physics VM — Domain-Agnostic Wall Model (Lane A).

Implements the **Penalization + Calibrated Wall Model** strategy for
enforcing no-slip / no-penetration boundary conditions on immersed
boundaries without body-fitted meshes.

Strategy Overview
-----------------
Lane A: Brinkman volume penalization + algebraic wall-function closure.

1. **Volume penalization** (Brinkman):
       F_penal = -(1/η) · χ_solid · u

   where η is the penalization permeability (small → strong enforcement)
   and χ_solid is the solid mask from the geometry compiler.

2. **Near-wall shear stress closure** (Reichardt-like wall function):
       τ_wall(x) = μ · |u_tangential| / d_eff(x)

   where d_eff = max(d(x), d_min) clamps the distance proxy to avoid
   division by zero, and the result is QTT-native (Hadamard quotient
   via reciprocal approximation).

3. **Thermal wall closure** (optional):
       q_wall(x) = k · (T - T_wall) / d_eff(x)

   Same reciprocal-distance approach.

All operations are QTT-native:
- Penalization: SCALE + HADAMARD (rank-multiplicative, then truncate)
- Wall stress: HADAMARD · reciprocal_distance (pre-computed QTT field)
- Thermal: same pattern

IP Boundary Compliance
----------------------
Wall-model internals (distance proxy, reciprocal fields, stress profiles)
are **internal state** and MUST NOT leak through the sanitizer.
Only whitelisted scalar aggregates are permitted:

  ✓ integrated_wall_shear (scalar)
  ✓ max_wall_shear_proxy (scalar)
  ✓ integrated_heat_flux (scalar)
  ✗ τ_wall(x) field → FORBIDDEN
  ✗ d(x) field → FORBIDDEN
  ✗ reciprocal distance → FORBIDDEN

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from ..qtt_tensor import QTTTensor
from ..compilers.geometry_coeffs import CompiledGeometry


# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class WallModelConfig:
    """Configuration for the domain-agnostic wall model.

    Parameters
    ----------
    eta_permeability : float
        Brinkman penalization permeability η.  Smaller → stronger
        enforcement.  Typical: 1e-3 to 1e-8.  The penalization force
        is F = -(1/η) · χ_solid · u.  Note: the geometry compiler
        already produces β(x) = β₀ · χ_solid; we use β₀ = 1/η.
    d_min : float
        Minimum clamped distance for wall-function evaluations.
        Prevents division by zero in shear/thermal closures.
        Should be O(h) where h is grid spacing.
    viscosity : float
        Dynamic viscosity μ for shear stress computation.
    thermal_conductivity : float
        Thermal conductivity k for heat flux computation.
        Set to 0.0 to disable thermal closure.
    t_wall : float
        Wall temperature for thermal boundary condition.
    max_rank : int
        Maximum TT rank for intermediate wall-model fields.
    cutoff : float
        SVD cutoff for truncation of wall-model intermediates.
    """
    eta_permeability: float = 1e-4
    d_min: float = 1e-3
    viscosity: float = 1e-3
    thermal_conductivity: float = 0.0
    t_wall: float = 0.0
    max_rank: int = 64
    cutoff: float = 1e-12


# ══════════════════════════════════════════════════════════════════════
# Precomputed wall fields (QTT-native)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class WallFields:
    """Precomputed QTT fields for the wall model.

    These fields are computed once from the compiled geometry and reused
    every timestep.  All are QTT tensors — internal state that never
    leaks through the sanitizer.

    Attributes
    ----------
    near_wall_mask : QTTTensor
        Narrow-band indicator: 1 in the near-wall region, 0 elsewhere.
        Width controlled by ``WallModelConfig.d_min * band_factor``.
    reciprocal_distance : QTTTensor
        1 / max(|φ(x)|, d_min) — clamped reciprocal distance.
        Pre-computed to avoid per-step division.
    penalization_coeff : QTTTensor
        (1/η) · χ_solid(x) — penalty coefficient field.
    shear_coeff : QTTTensor | None
        μ / max(|φ(x)|, d_min) — wall shear coefficient.
        None if viscosity == 0.
    thermal_coeff : QTTTensor | None
        k / max(|φ(x)|, d_min) — thermal wall coefficient.
        None if thermal_conductivity == 0.
    rank_stats : dict[str, int]
        Internal rank statistics (PRIVATE — § 20.4 forbidden).
    """
    near_wall_mask: QTTTensor
    reciprocal_distance: QTTTensor
    penalization_coeff: QTTTensor
    shear_coeff: QTTTensor | None = None
    thermal_coeff: QTTTensor | None = None
    rank_stats: dict[str, int] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════
# Wall Model — primary class
# ══════════════════════════════════════════════════════════════════════

class WallModel:
    """Domain-agnostic wall model for QTT-based immersed boundary methods.

    The wall model operates entirely in TT-core format.  It takes the
    compiled geometry (from ``GeometryCompiler``) and produces:

    1. **Penalization force** applied during each timestep to enforce
       no-slip conditions in solid regions.
    2. **Wall shear stress proxy** for diagnostics (integrated, not
       field-level).
    3. **Thermal wall closure** (optional) for conjugate heat transfer.

    Usage
    -----
    >>> from ontic.engine.vm.models.wall_model import WallModel, WallModelConfig
    >>> wm = WallModel(config=WallModelConfig(viscosity=0.01))
    >>> wall_fields = wm.precompute(compiled_geometry)
    >>> # Each timestep:
    >>> penalized_u = wm.apply_penalization(u_field, wall_fields, dt=0.001)
    >>> diagnostics = wm.compute_diagnostics(u_field, wall_fields)
    """

    def __init__(self, config: WallModelConfig | None = None) -> None:
        self._config = config or WallModelConfig()

    @property
    def config(self) -> WallModelConfig:
        return self._config

    def precompute(self, geometry: CompiledGeometry) -> WallFields:
        """Precompute QTT wall-model fields from compiled geometry.

        This is called once before the time-stepping loop.  All resulting
        fields are QTT tensors stored in TT-core format.

        Parameters
        ----------
        geometry : CompiledGeometry
            Output of ``GeometryCompiler.compile()``.

        Returns
        -------
        WallFields
            Precomputed QTT fields for per-step wall-model application.
        """
        cfg = self._config
        bpd = geometry.solid_mask.bits_per_dim
        dom = geometry.solid_mask.domain

        # ── 1. Penalization coefficient: (1/η) · χ_solid ────────────
        inv_eta = 1.0 / max(cfg.eta_permeability, 1e-30)
        penalization_coeff = geometry.solid_mask.scale(inv_eta)
        penalization_coeff = penalization_coeff.truncate(
            max_rank=cfg.max_rank, cutoff=cfg.cutoff,
        )

        # ── 2. Near-wall mask: narrow band around interface ──────────
        # Use the distance proxy φ(x) to identify near-wall region.
        # near_wall = 1 where |φ| < band_width, 0 elsewhere.
        band_width = max(cfg.d_min * 10.0, 0.02)

        def near_wall_fn(*coords: NDArray) -> NDArray:
            # Evaluate distance proxy on the grid
            phi = geometry.distance_proxy
            phi_dense = _evaluate_qtt_on_grid(phi, coords)
            return (np.abs(phi_dense) < band_width).astype(np.float64)

        near_wall_mask = QTTTensor.from_function(
            near_wall_fn,
            bits_per_dim=bpd,
            domain=dom,
            max_rank=cfg.max_rank,
            cutoff=cfg.cutoff,
        )

        # ── 3. Reciprocal distance: 1 / max(|φ|, d_min) ────────────
        d_min = cfg.d_min

        def recip_dist_fn(*coords: NDArray) -> NDArray:
            phi = geometry.distance_proxy
            phi_dense = _evaluate_qtt_on_grid(phi, coords)
            d_eff = np.maximum(np.abs(phi_dense), d_min)
            return 1.0 / d_eff

        reciprocal_distance = QTTTensor.from_function(
            recip_dist_fn,
            bits_per_dim=bpd,
            domain=dom,
            max_rank=cfg.max_rank,
            cutoff=cfg.cutoff,
        )

        # ── 4. Shear coefficient: μ / d_eff ─────────────────────────
        shear_coeff: QTTTensor | None = None
        if cfg.viscosity > 0.0:
            shear_coeff = reciprocal_distance.scale(cfg.viscosity)
            shear_coeff = shear_coeff.truncate(
                max_rank=cfg.max_rank, cutoff=cfg.cutoff,
            )

        # ── 5. Thermal coefficient: k / d_eff ───────────────────────
        thermal_coeff: QTTTensor | None = None
        if cfg.thermal_conductivity > 0.0:
            thermal_coeff = reciprocal_distance.scale(
                cfg.thermal_conductivity,
            )
            thermal_coeff = thermal_coeff.truncate(
                max_rank=cfg.max_rank, cutoff=cfg.cutoff,
            )

        # ── 6. Rank stats (PRIVATE — never exposed externally) ──────
        rank_stats = {
            "penalization_coeff_max_rank": penalization_coeff.max_rank,
            "near_wall_mask_max_rank": near_wall_mask.max_rank,
            "reciprocal_distance_max_rank": reciprocal_distance.max_rank,
        }
        if shear_coeff is not None:
            rank_stats["shear_coeff_max_rank"] = shear_coeff.max_rank
        if thermal_coeff is not None:
            rank_stats["thermal_coeff_max_rank"] = thermal_coeff.max_rank

        return WallFields(
            near_wall_mask=near_wall_mask,
            reciprocal_distance=reciprocal_distance,
            penalization_coeff=penalization_coeff,
            shear_coeff=shear_coeff,
            thermal_coeff=thermal_coeff,
            rank_stats=rank_stats,
        )

    def apply_penalization(
        self,
        field: QTTTensor,
        wall_fields: WallFields,
        dt: float,
    ) -> QTTTensor:
        """Apply Brinkman volume penalization to a velocity/vorticity field.

        Implements:  u_new = u - dt · (1/η) · χ_solid · u

        This is the primary enforcement mechanism for no-slip conditions.
        The operation is QTT-native: Hadamard product (rank-multiplicative)
        followed by truncation.

        Parameters
        ----------
        field : QTTTensor
            Current velocity or vorticity field (QTT format).
        wall_fields : WallFields
            Precomputed wall-model fields.
        dt : float
            Current timestep size.

        Returns
        -------
        QTTTensor
            Updated field with penalization applied.
        """
        # F_penal = (1/η) · χ_solid · u
        penalty_term = wall_fields.penalization_coeff.hadamard(field)
        penalty_term = penalty_term.scale(dt)
        penalty_term = penalty_term.truncate(
            max_rank=self._config.max_rank,
            cutoff=self._config.cutoff,
        )

        # u_new = u - dt · penalty
        result = field.sub(penalty_term)
        result = result.truncate(
            max_rank=self._config.max_rank,
            cutoff=self._config.cutoff,
        )
        return result

    def compute_wall_shear_proxy(
        self,
        velocity_field: QTTTensor,
        wall_fields: WallFields,
    ) -> QTTTensor:
        """Compute wall shear stress proxy field (QTT-native).

        τ_wall(x) = μ · |u| / d_eff(x) · mask_near_wall(x)

        The result is a QTT tensor.  For IP compliance, this field
        must NOT be exposed externally — only integrated aggregates
        may leave the VM.

        Parameters
        ----------
        velocity_field : QTTTensor
            Velocity magnitude or vorticity field.
        wall_fields : WallFields
            Precomputed wall-model fields.

        Returns
        -------
        QTTTensor
            Wall shear stress proxy (internal only).
        """
        if wall_fields.shear_coeff is None:
            raise ValueError(
                "Shear coefficient not precomputed. "
                "Set viscosity > 0 in WallModelConfig."
            )

        # τ = (μ/d_eff) ⊙ |u| ⊙ mask_near_wall
        shear = wall_fields.shear_coeff.hadamard(velocity_field)
        shear = shear.hadamard(wall_fields.near_wall_mask)
        shear = shear.truncate(
            max_rank=self._config.max_rank,
            cutoff=self._config.cutoff,
        )
        return shear

    def compute_thermal_flux_proxy(
        self,
        temperature_field: QTTTensor,
        wall_fields: WallFields,
    ) -> QTTTensor:
        """Compute thermal wall heat flux proxy (QTT-native).

        q_wall(x) = k · (T - T_wall) / d_eff(x) · mask_near_wall(x)

        Parameters
        ----------
        temperature_field : QTTTensor
            Temperature field (QTT format).
        wall_fields : WallFields
            Precomputed wall-model fields.

        Returns
        -------
        QTTTensor
            Wall heat flux proxy (internal only).
        """
        if wall_fields.thermal_coeff is None:
            raise ValueError(
                "Thermal coefficient not precomputed. "
                "Set thermal_conductivity > 0 in WallModelConfig."
            )

        # ΔT = T - T_wall
        t_wall_field = QTTTensor.constant(
            self._config.t_wall,
            bits_per_dim=temperature_field.bits_per_dim,
            domain=temperature_field.domain,
        )
        delta_t = temperature_field.sub(t_wall_field)

        # q = (k/d_eff) ⊙ ΔT ⊙ mask_near_wall
        flux = wall_fields.thermal_coeff.hadamard(delta_t)
        flux = flux.hadamard(wall_fields.near_wall_mask)
        flux = flux.truncate(
            max_rank=self._config.max_rank,
            cutoff=self._config.cutoff,
        )
        return flux

    def compute_diagnostics(
        self,
        velocity_field: QTTTensor,
        wall_fields: WallFields,
        temperature_field: QTTTensor | None = None,
    ) -> dict[str, float]:
        """Compute wall-model diagnostic aggregates (sanitizer-safe).

        Returns ONLY scalar aggregates that are whitelisted for external
        reporting.  Field-level data is never exposed.

        Parameters
        ----------
        velocity_field : QTTTensor
            Current velocity or vorticity field.
        wall_fields : WallFields
            Precomputed wall-model fields.
        temperature_field : QTTTensor | None
            Current temperature field (if CHT is active).

        Returns
        -------
        dict[str, float]
            Sanitizer-safe scalar diagnostics:
            - ``integrated_wall_shear``: ∫ τ_wall dA
            - ``max_wall_shear_proxy``: approximate max τ_wall
            - ``integrated_heat_flux``: ∫ q_wall dA (if thermal active)
            - ``penalization_energy``: ∫ (1/η) χ u² dA (dissipation proxy)
        """
        diagnostics: dict[str, float] = {}

        # Grid spacing for integration
        n_dims = velocity_field.n_dims
        cell_volume = 1.0
        for d in range(n_dims):
            cell_volume *= velocity_field.grid_spacing(d)

        # ── Penalization energy: ∫ (1/η) χ u² dA ────────────────────
        u_sq = velocity_field.hadamard(velocity_field)
        u_sq = u_sq.truncate(
            max_rank=self._config.max_rank,
            cutoff=self._config.cutoff,
        )
        penal_energy = wall_fields.penalization_coeff.hadamard(u_sq)
        penal_energy = penal_energy.truncate(
            max_rank=self._config.max_rank,
            cutoff=self._config.cutoff,
        )
        diagnostics["penalization_energy"] = abs(
            cell_volume * penal_energy.sum()
        )

        # ── Integrated wall shear: ∫ τ_wall dA ──────────────────────
        if wall_fields.shear_coeff is not None:
            shear_proxy = self.compute_wall_shear_proxy(
                velocity_field, wall_fields,
            )
            diagnostics["integrated_wall_shear"] = abs(
                cell_volume * shear_proxy.sum()
            )

            # Max shear proxy: use L∞ ≈ (∫ τ^p dA)^(1/p) for large p
            # Quick approximation via norm ratio
            shear_norm = shear_proxy.norm()
            near_wall_norm = wall_fields.near_wall_mask.norm()
            if near_wall_norm > 1e-15:
                diagnostics["max_wall_shear_proxy"] = (
                    shear_norm / max(near_wall_norm, 1e-15)
                    * math.sqrt(2.0 ** sum(velocity_field.bits_per_dim))
                )
            else:
                diagnostics["max_wall_shear_proxy"] = 0.0
        else:
            diagnostics["integrated_wall_shear"] = 0.0
            diagnostics["max_wall_shear_proxy"] = 0.0

        # ── Integrated heat flux: ∫ q_wall dA ───────────────────────
        if (
            temperature_field is not None
            and wall_fields.thermal_coeff is not None
        ):
            flux_proxy = self.compute_thermal_flux_proxy(
                temperature_field, wall_fields,
            )
            diagnostics["integrated_heat_flux"] = abs(
                cell_volume * flux_proxy.sum()
            )
        else:
            diagnostics["integrated_heat_flux"] = 0.0

        return diagnostics

    def generate_ir_penalization(
        self,
        field_reg: int,
        penal_reg: int,
        tmp_reg: int,
        dt: float,
    ) -> list:
        """Generate IR instructions for penalization within a timestep.

        These instructions are injected into the compiler's instruction
        stream by domain compilers that use wall models.

        Parameters
        ----------
        field_reg : int
            Register holding the field to penalize (e.g., vorticity).
        penal_reg : int
            Register holding the preloaded penalization coefficient.
        tmp_reg : int
            Scratch register for intermediate computation.
        dt : float
            Timestep size.

        Returns
        -------
        list[Instruction]
            IR instructions implementing: field -= dt · penal_coeff ⊙ field
        """
        from ..ir import (
            Instruction, OpCode, hadamard, scale, sub, truncate,
        )

        return [
            # tmp = penal_coeff ⊙ field
            hadamard(tmp_reg, penal_reg, field_reg),
            # tmp = dt · tmp
            scale(tmp_reg, tmp_reg, dt),
            # tmp = truncate(tmp) — control rank growth
            truncate(tmp_reg),
            # field = field - tmp
            sub(field_reg, field_reg, tmp_reg),
            # field = truncate(field)
            truncate(field_reg),
        ]

    @staticmethod
    def sanitize_diagnostics(
        diagnostics: dict[str, float],
    ) -> dict[str, float]:
        """Filter diagnostics to only whitelisted aggregates.

        Ensures no wall-model internals leak through the sanitizer.
        This is the ONLY path for wall-model data to reach the public
        output.

        Parameters
        ----------
        diagnostics : dict[str, float]
            Raw diagnostics from ``compute_diagnostics()``.

        Returns
        -------
        dict[str, float]
            Only the whitelisted keys:
            - ``integrated_wall_shear``
            - ``max_wall_shear_proxy``
            - ``integrated_heat_flux``
            - ``penalization_energy``
        """
        _WHITELIST = frozenset({
            "integrated_wall_shear",
            "max_wall_shear_proxy",
            "integrated_heat_flux",
            "penalization_energy",
        })
        return {k: v for k, v in diagnostics.items() if k in _WHITELIST}


# ══════════════════════════════════════════════════════════════════════
# Internal helper
# ══════════════════════════════════════════════════════════════════════

def _evaluate_qtt_on_grid(
    qtt: QTTTensor,
    coords: tuple[NDArray, ...],
) -> NDArray:
    """Evaluate a QTT tensor on a meshgrid — internal helper.

    Uses ``QTTTensor.to_dense()`` for small tensors or the generating
    function approach for reconstruction.  This is called during
    **precomputation** (before the dispatch loop), not during execution.

    For the wall model, this is acceptable because ``precompute()`` runs
    BEFORE the VM dispatch context is entered.
    """
    # For fields that were built from_function, the internal values
    # are already encoded in the QTT cores.  We reconstruct by evaluating
    # the TT decomposition at the grid points.
    n_cores = qtt.n_cores
    if n_cores <= 22:
        # Safe to reconstruct: total elements ≤ 4M
        dense = qtt.to_dense()
        return dense
    else:
        # Large tensor — would require >4M elements.
        # Use per-point evaluation (slow but safe).
        raise NotImplementedError(
            f"QTT evaluation with {n_cores} cores ({2**n_cores} elements) "
            "exceeds safe dense reconstruction limit. "
            "Implement streaming or blocked evaluation."
        )
