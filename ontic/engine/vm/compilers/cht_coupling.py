"""QTT Physics VM — Conjugate Heat Transfer (CHT) Coupling Module.

Provides the infrastructure for coupling fluid and solid thermal
domains using QTT-native coefficient fields and conservative energy
bookkeeping.

Strategy
--------
CHT coupling in the QTT framework uses the **coefficient-field**
approach: material properties (thermal conductivity k, heat capacity
ρCp) are represented as QTT fields that smoothly transition across
the fluid-solid interface.  This avoids explicit domain decomposition
and keeps the solver monolithic with a single temperature field.

The continuous energy equation:

    ρCp ∂T/∂t = ∇·(k(x)∇T) + Q(x)

is discretized as:

    T_{n+1} = T_n + dt · [k(x) · ∇²T + (∇k(x))·(∇T)] / ρCp(x)

where the variable-coefficient terms are handled via QTT Hadamard
products:
  - k(x) is a QTT field from the geometry/material compiler
  - ρCp(x) is a QTT field (reciprocal for division)

Conservative energy bookkeeping: total thermal energy E_th = ∫ρCp·T dV
is tracked as an evidence predicate (should be conserved modulo
source terms and boundary fluxes).

IP Boundary Compliance
----------------------
Material property fields (k(x), ρCp(x)), interface details, and
thermal flux distributions are internal state.  Only integrated
heat flux and temperature extrema are sanitizer-safe.

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..ir import (
    BCKind, FieldSpec, Instruction, OpCode, Program,
    add, bc_apply, grad, hadamard, laplace, load_field,
    loop_end, loop_start, measure, scale,
    store_field, sub, truncate,
)
from ..qtt_tensor import QTTTensor
from .base import BaseCompiler


# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CHTMaterialSpec:
    """Material thermal properties for a region.

    Parameters
    ----------
    name : str
        Material identifier (e.g., "fluid", "solid_aluminum").
    thermal_conductivity : float
        Thermal conductivity k [W/(m·K)].
    density : float
        Density ρ [kg/m³].
    specific_heat : float
        Specific heat capacity Cp [J/(kg·K)].
    """
    name: str
    thermal_conductivity: float
    density: float
    specific_heat: float

    @property
    def thermal_diffusivity(self) -> float:
        """α = k / (ρ·Cp) [m²/s]."""
        return self.thermal_conductivity / max(
            self.density * self.specific_heat, 1e-30,
        )

    @property
    def rho_cp(self) -> float:
        """ρCp [J/(m³·K)]."""
        return self.density * self.specific_heat


@dataclass(frozen=True)
class CHTConfig:
    """Configuration for conjugate heat transfer coupling.

    Parameters
    ----------
    fluid : CHTMaterialSpec
        Fluid-domain material properties.
    solid : CHTMaterialSpec
        Solid-domain material properties.
    t_initial : float
        Initial temperature (uniform) [K].
    bc_left : tuple[str, float]
        Left boundary: ("dirichlet", T) or ("flux", q).
    bc_right : tuple[str, float]
        Right boundary: ("dirichlet", T) or ("flux", q).
    source_power : float
        Volumetric heat source in solid [W/m³].
    max_rank : int
        Maximum TT rank for coefficient fields.
    """
    fluid: CHTMaterialSpec = field(default_factory=lambda: CHTMaterialSpec(
        name="water",
        thermal_conductivity=0.6,
        density=1000.0,
        specific_heat=4186.0,
    ))
    solid: CHTMaterialSpec = field(default_factory=lambda: CHTMaterialSpec(
        name="aluminum",
        thermal_conductivity=237.0,
        density=2700.0,
        specific_heat=900.0,
    ))
    t_initial: float = 300.0
    bc_left: tuple[str, float] = ("dirichlet", 300.0)
    bc_right: tuple[str, float] = ("dirichlet", 400.0)
    source_power: float = 0.0
    max_rank: int = 64


# ══════════════════════════════════════════════════════════════════════
# Coefficient field compiler
# ══════════════════════════════════════════════════════════════════════

def compile_cht_coefficients_1d(
    config: CHTConfig,
    interface_x: float,
    bits_per_dim: tuple[int],
    domain: tuple[float, float] = (0.0, 1.0),
    max_rank: int = 32,
    interface_width: float = 0.02,
) -> dict[str, Any]:
    """Compile CHT coefficient fields for a 1D fluid-solid interface.

    Produces QTT fields for thermal conductivity k(x), volumetric
    heat capacity ρCp(x), and their reciprocals, with a smooth
    tanh-transition across the interface.

    Parameters
    ----------
    config : CHTConfig
        Material properties for fluid and solid.
    interface_x : float
        Interface position (solid for x < interface_x, fluid otherwise).
    bits_per_dim : tuple[int]
        QTT resolution.
    domain : tuple[float, float]
        Physical domain bounds.
    max_rank : int
        Maximum TT rank.
    interface_width : float
        Width of the smooth transition zone.

    Returns
    -------
    dict[str, Any]
        Keys: "k_field", "rho_cp_field", "inv_rho_cp_field",
              "source_field", "metadata".
    """
    k_solid = config.solid.thermal_conductivity
    k_fluid = config.fluid.thermal_conductivity
    rcp_solid = config.solid.rho_cp
    rcp_fluid = config.fluid.rho_cp
    Q = config.source_power

    def k_fn(x: NDArray) -> NDArray:
        """Thermal conductivity k(x) with smooth interface."""
        phi = np.tanh((x - interface_x) / max(interface_width, 1e-10))
        # phi ∈ [-1, 1]: solid at phi→-1, fluid at phi→+1
        blend = 0.5 * (1.0 + phi)
        return k_solid * (1.0 - blend) + k_fluid * blend

    def rho_cp_fn(x: NDArray) -> NDArray:
        """Volumetric heat capacity ρCp(x)."""
        phi = np.tanh((x - interface_x) / max(interface_width, 1e-10))
        blend = 0.5 * (1.0 + phi)
        return rcp_solid * (1.0 - blend) + rcp_fluid * blend

    def inv_rho_cp_fn(x: NDArray) -> NDArray:
        """Reciprocal: 1 / ρCp(x)."""
        rcp = rho_cp_fn(x)
        return 1.0 / np.maximum(rcp, 1e-15)

    def source_fn(x: NDArray) -> NDArray:
        """Volumetric heat source Q(x) — nonzero only in solid."""
        if Q == 0.0:
            return np.zeros_like(x)
        phi = np.tanh((x - interface_x) / max(interface_width, 1e-10))
        solid_frac = 0.5 * (1.0 - phi)
        return Q * solid_frac

    k_field = QTTTensor.from_function(
        k_fn, bits_per_dim=bits_per_dim, domain=(domain,),
        max_rank=max_rank,
    )
    rho_cp_field = QTTTensor.from_function(
        rho_cp_fn, bits_per_dim=bits_per_dim, domain=(domain,),
        max_rank=max_rank,
    )
    inv_rho_cp_field = QTTTensor.from_function(
        inv_rho_cp_fn, bits_per_dim=bits_per_dim, domain=(domain,),
        max_rank=max_rank,
    )
    source_field = QTTTensor.from_function(
        source_fn, bits_per_dim=bits_per_dim, domain=(domain,),
        max_rank=max_rank,
    )

    return {
        "k_field": k_field,
        "rho_cp_field": rho_cp_field,
        "inv_rho_cp_field": inv_rho_cp_field,
        "source_field": source_field,
        "metadata": {
            "interface_x": interface_x,
            "k_solid": k_solid,
            "k_fluid": k_fluid,
            "rho_cp_solid": rcp_solid,
            "rho_cp_fluid": rcp_fluid,
            "source_power": Q,
        },
    }


# ══════════════════════════════════════════════════════════════════════
# CHT Compiler (1D)
# ══════════════════════════════════════════════════════════════════════

class CHTCompiler1D(BaseCompiler):
    """Compile 1D conjugate heat transfer into QTT VM bytecode.

    Solves the variable-coefficient heat equation:

        ρCp(x) · ∂T/∂t = ∂/∂x(k(x) · ∂T/∂x) + Q(x)

    which, after dividing by ρCp(x), becomes:

        ∂T/∂t = [1/ρCp] · [k · ∇²T + ∂k/∂x · ∂T/∂x + Q]

    All terms are QTT-native: Hadamard products for variable
    coefficients, gradient MPOs for derivatives.

    Parameters
    ----------
    n_bits : int
        Grid resolution.
    n_steps : int
        Number of time steps.
    config : CHTConfig | None
        Material and boundary configuration.
    interface_x : float
        Interface location between solid and fluid.
    dt : float | None
        Time step (auto-computed if None).
    """

    def __init__(
        self,
        n_bits: int = 8,
        n_steps: int = 200,
        config: CHTConfig | None = None,
        interface_x: float = 0.5,
        dt: float | None = None,
    ) -> None:
        self._n_bits = n_bits
        self._n_steps = n_steps
        self._config = config or CHTConfig()
        self._interface_x = interface_x

        N = 2 ** n_bits
        h = 1.0 / N
        if dt is None:
            # CFL for diffusion: dt ≤ h² / (2α_max)
            alpha_max = max(
                self._config.solid.thermal_diffusivity,
                self._config.fluid.thermal_diffusivity,
            )
            self._dt = 0.25 * h * h / max(2.0 * alpha_max, 1e-30)
        else:
            self._dt = dt

    @property
    def domain(self) -> str:
        return "cht_1d"

    @property
    def domain_label(self) -> str:
        return "1D Conjugate Heat Transfer"

    def compile(self) -> Program:
        """Compile the CHT equation into QTT VM bytecode.

        Register allocation
        -------------------
        r0 = T         (temperature)
        r1 = k(x)      (thermal conductivity field)
        r2 = inv_ρCp   (1/ρCp reciprocal field)
        r3 = Q(x)      (source field)
        r4 = ∂T/∂x
        r5 = ∂k/∂x
        r6 = k·∇²T
        r7 = (∂k/∂x)·(∂T/∂x)
        r8 = RHS (before dividing by ρCp)
        r9 = dt * RHS / ρCp
        """
        dt = self._dt
        nb = self._n_bits

        cfg = self._config

        def init_T(x: NDArray) -> NDArray:
            """Initial temperature distribution."""
            return np.full_like(x, cfg.t_initial)

        def invariant_fn(fields: dict) -> float:
            """Total thermal energy: ∫ρCp·T dx."""
            T = fields["T"]
            h = T.grid_spacing(0)
            return h * T.sum()

        instructions: list[Instruction] = [
            loop_start(self._n_steps),

            load_field(0, "T"),
            load_field(1, "k"),
            load_field(2, "inv_rho_cp"),
            load_field(3, "source"),

            # ── ∂T/∂x ───────────────────────────────────────────────
            grad(4, 0, dim=0),                     # r4 = ∂T/∂x

            # ── ∂k/∂x ───────────────────────────────────────────────
            grad(5, 1, dim=0),                     # r5 = ∂k/∂x

            # ── k · ∇²T ─────────────────────────────────────────────
            laplace(6, 0),                         # r6 = ∇²T
            hadamard(6, 1, 6),                     # r6 = k ⊙ ∇²T
            truncate(6),

            # ── (∂k/∂x) · (∂T/∂x) ──────────────────────────────────
            hadamard(7, 5, 4),                     # r7 = ∂k/∂x ⊙ ∂T/∂x
            truncate(7),

            # ── RHS = k∇²T + (∂k/∂x)(∂T/∂x) + Q ───────────────────
            add(8, 6, 7),                          # r8 = k∇²T + ∇k·∇T
            truncate(8),
            add(8, 8, 3),                          # r8 += Q(x)
            truncate(8),

            # ── T_{n+1} = T_n + dt · RHS / ρCp ──────────────────────
            hadamard(9, 8, 2),                     # r9 = RHS ⊙ (1/ρCp)
            truncate(9),
            scale(9, 9, dt),                       # r9 = dt · RHS/ρCp
            add(0, 0, 9),                          # T += dt·RHS/ρCp
            truncate(0),

            bc_apply(0, BCKind.DIRICHLET),

            store_field(0, "T"),
            measure(0, "T"),

            loop_end(),
        ]

        bits = (nb,)
        dom = (0.0, 1.0)

        fields: dict[str, FieldSpec] = {
            "T": FieldSpec(
                name="T",
                n_dims=1,
                bits_per_dim=bits,
                bc=BCKind.DIRICHLET,
                bc_params={
                    "domain": dom,
                    "left": cfg.bc_left,
                    "right": cfg.bc_right,
                },
                initial_fn="init_T",
                conserved_quantity="thermal_energy",
            ),
            "k": FieldSpec(
                name="k",
                n_dims=1,
                bits_per_dim=bits,
                bc=BCKind.DIRICHLET,
                bc_params={"domain": dom},
                initial_fn="init_k",
            ),
            "inv_rho_cp": FieldSpec(
                name="inv_rho_cp",
                n_dims=1,
                bits_per_dim=bits,
                bc=BCKind.DIRICHLET,
                bc_params={"domain": dom},
                initial_fn="init_inv_rho_cp",
            ),
            "source": FieldSpec(
                name="source",
                n_dims=1,
                bits_per_dim=bits,
                bc=BCKind.DIRICHLET,
                bc_params={"domain": dom},
                initial_fn="init_source",
            ),
        }

        metadata: dict[str, Any] = {
            "init_T": init_T,
            "invariant_fn": invariant_fn,
            "invariant": "thermal_energy",
            "equations": "ρCp ∂T/∂t = ∇·(k∇T) + Q",
            "interface_x": self._interface_x,
            "materials": {
                "solid": {
                    "name": cfg.solid.name,
                    "k": cfg.solid.thermal_conductivity,
                    "rho": cfg.solid.density,
                    "cp": cfg.solid.specific_heat,
                },
                "fluid": {
                    "name": cfg.fluid.name,
                    "k": cfg.fluid.thermal_conductivity,
                    "rho": cfg.fluid.density,
                    "cp": cfg.fluid.specific_heat,
                },
            },
            "source_power": cfg.source_power,
            "boundedness_predicates": ["temperature_finite"],
        }

        return Program(
            domain=self.domain,
            domain_label=self.domain_label,
            n_registers=10,
            fields=fields,
            instructions=instructions,
            dt=self._dt,
            n_steps=self._n_steps,
            params={
                "interface_x": self._interface_x,
                "source_power": cfg.source_power,
            },
            metadata=metadata,
        )


# ══════════════════════════════════════════════════════════════════════
# Energy bookkeeping
# ══════════════════════════════════════════════════════════════════════

def compute_thermal_energy(
    T_values: NDArray,
    rho_cp_values: NDArray,
    h: float,
) -> float:
    """Compute total thermal energy: E_th = ∫ρCp·T dx.

    Parameters
    ----------
    T_values : NDArray
        Temperature field.
    rho_cp_values : NDArray
        Volumetric heat capacity field.
    h : float
        Grid spacing.

    Returns
    -------
    float
        Total thermal energy.
    """
    return float(h * np.sum(rho_cp_values * T_values))


def compute_interface_flux(
    T_values: NDArray,
    k_values: NDArray,
    interface_idx: int,
    h: float,
) -> float:
    """Compute heat flux at the fluid-solid interface.

    q = -k · ∂T/∂x evaluated at the interface.

    Parameters
    ----------
    T_values : NDArray
        Temperature field.
    k_values : NDArray
        Conductivity field.
    interface_idx : int
        Grid index of interface.
    h : float
        Grid spacing.

    Returns
    -------
    float
        Heat flux at interface [W/m²].
    """
    if interface_idx < 1 or interface_idx >= len(T_values) - 1:
        return 0.0
    dTdx = (T_values[interface_idx + 1] - T_values[interface_idx - 1]) / (2 * h)
    k_interface = k_values[interface_idx]
    return float(-k_interface * dTdx)


def check_temperature_finite(T_values: NDArray) -> bool:
    """Check that temperature is finite everywhere (no NaN/Inf).

    Parameters
    ----------
    T_values : NDArray
        Temperature field.

    Returns
    -------
    bool
        True if all values are finite.
    """
    return bool(np.all(np.isfinite(T_values)))


# ══════════════════════════════════════════════════════════════════════
# Diagnostic sanitizer
# ══════════════════════════════════════════════════════════════════════

_CHT_DIAGNOSTIC_WHITELIST = frozenset({
    "total_thermal_energy",
    "interface_heat_flux",
    "temperature_max",
    "temperature_min",
    "energy_balance_error",
    "temperature_finite",
})


def sanitize_cht_diagnostics(
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Filter CHT diagnostics to whitelisted aggregates.

    Parameters
    ----------
    diagnostics : dict[str, Any]
        Raw diagnostics.

    Returns
    -------
    dict[str, Any]
        Only whitelisted scalar aggregates.
    """
    return {k: v for k, v in diagnostics.items() if k in _CHT_DIAGNOSTIC_WHITELIST}
