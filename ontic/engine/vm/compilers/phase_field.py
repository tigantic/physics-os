"""QTT Physics VM — Phase-Field Multiphase Compiler (Lane A).

Implements the **diffuse-interface phase-field** approach for
immiscible two-phase flows.  This is Lane A (QTT-friendly) from
the execution plan: the interface is captured as a smooth field φ
with no sharp discontinuities, making it less rank-hostile than
VOF (Lane B).

Governing Equations
-------------------
1. **Cahn-Hilliard** (phase-field evolution):

       ∂φ/∂t + u·∇φ = M · ∇²μ

   where μ is the chemical potential:

       μ = -ε²∇²φ + F'(φ)

   and F(φ) = ¼(φ² - 1)² is the double-well potential,
   so F'(φ) = φ³ - φ.

   M = mobility, ε = interface thickness parameter.

2. **Navier-Stokes with surface tension** (coupled):

       ρ(φ) ∂u/∂t + ρ(φ)(u·∇)u = -∇p + ∇·(μ_visc(φ)∇u) + σκnδ

   The surface tension is implemented via the diffuse-interface
   capillary stress tensor, which reduces to:

       F_st = μ ∇φ

   (chemical potential times phase-field gradient).

QTT-Native Properties
---------------------
- φ is smooth → low QTT rank (unlike sharp VOF interfaces)
- IMEX splitting: implicit diffusion + explicit advection
- All operations via Hadamard + gradient/Laplacian MPOs
- Interface width ε scales with grid: ε = O(h) for convergence

IP Boundary Compliance
----------------------
Phase-field internals (φ distribution, chemical potential, mobility
coefficients) are internal state.  Only integrated diagnostics are
sanitizer-safe.

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
    add, bc_apply, grad, hadamard, laplace, laplace_solve,
    load_field, loop_end, loop_start, measure, negate,
    scale, store_field, sub, truncate,
)
from .base import BaseCompiler


# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PhaseFieldConfig:
    """Configuration for phase-field multiphase simulation.

    Parameters
    ----------
    epsilon : float
        Interface thickness parameter ε.
        Controls the width of the diffuse interface.
        Should scale as ε ~ O(h) for mesh convergence.
    mobility : float
        Mobility parameter M in the Cahn-Hilliard equation.
        Controls the relaxation rate toward equilibrium.
    sigma : float
        Surface tension coefficient σ.
        The capillary force is F_st = σμ∇φ / ε.
    rho_1 : float
        Density of phase 1 (φ = -1).
    rho_2 : float
        Density of phase 2 (φ = +1).
    mu_1 : float
        Dynamic viscosity of phase 1.
    mu_2 : float
        Dynamic viscosity of phase 2.
    """
    epsilon: float = 0.02
    mobility: float = 1e-3
    sigma: float = 0.1
    rho_1: float = 1.0
    rho_2: float = 100.0
    mu_1: float = 0.01
    mu_2: float = 0.1


# ══════════════════════════════════════════════════════════════════════
# Initial conditions
# ══════════════════════════════════════════════════════════════════════

def circle_droplet_ic(
    x: NDArray,
    y: NDArray,
    cx: float = 0.5,
    cy: float = 0.5,
    radius: float = 0.2,
    epsilon: float = 0.02,
) -> NDArray:
    """Circular droplet initial condition for phase field.

    φ = tanh((r - R) / (√2 ε))

    φ = -1 inside droplet, +1 outside.

    Parameters
    ----------
    x, y : NDArray
        Spatial coordinates.
    cx, cy : float
        Droplet center.
    radius : float
        Droplet radius.
    epsilon : float
        Interface thickness.

    Returns
    -------
    NDArray
        Phase-field values.
    """
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return np.tanh((r - radius) / (math.sqrt(2.0) * max(epsilon, 1e-10)))


def rayleigh_taylor_ic(
    x: NDArray,
    y: NDArray,
    y_interface: float = 0.5,
    amplitude: float = 0.05,
    epsilon: float = 0.02,
) -> NDArray:
    """Rayleigh-Taylor instability initial condition.

    Heavy fluid (φ = +1) on top, light fluid (φ = -1) on bottom,
    with a cosine perturbation at the interface.

    Parameters
    ----------
    x, y : NDArray
        Spatial coordinates.
    y_interface : float
        Mean interface height.
    amplitude : float
        Perturbation amplitude.
    epsilon : float
        Interface thickness.

    Returns
    -------
    NDArray
        Phase-field values.
    """
    # Perturbed interface: y_i(x) = y_interface + A*cos(2πx)
    y_perturbed = y_interface + amplitude * np.cos(2.0 * np.pi * x)
    return np.tanh((y - y_perturbed) / (math.sqrt(2.0) * max(epsilon, 1e-10)))


# ══════════════════════════════════════════════════════════════════════
# Phase-Field Compiler (2D)
# ══════════════════════════════════════════════════════════════════════

class PhaseField2DCompiler(BaseCompiler):
    """Compile 2D phase-field Cahn-Hilliard equation into QTT VM bytecode.

    This compiler handles the **decoupled** Cahn-Hilliard equation
    (no velocity coupling).  For full two-phase NS, the NS2D compiler
    can be composed with this one.

    The Cahn-Hilliard system is split as:

        μ = -ε²∇²φ + φ³ - φ     (chemical potential)
        ∂φ/∂t = M∇²μ            (evolution, no advection)

    Time integration: explicit Euler.
    Conserved quantity: ∫φ dA (total phase fraction).

    Parameters
    ----------
    n_bits : int
        Bits per dimension (grid is 2^n_bits × 2^n_bits).
    n_steps : int
        Number of time steps.
    config : PhaseFieldConfig | None
        Phase-field parameters.
    dt : float | None
        Time step (auto if None).
    ic_type : str
        Initial condition: "droplet" or "rayleigh_taylor".
    """

    def __init__(
        self,
        n_bits: int = 6,
        n_steps: int = 100,
        config: PhaseFieldConfig | None = None,
        dt: float | None = None,
        ic_type: str = "droplet",
    ) -> None:
        self._n_bits = n_bits
        self._n_steps = n_steps
        self._config = config or PhaseFieldConfig()
        self._ic_type = ic_type

        N = 2 ** n_bits
        h = 1.0 / N
        eps = self._config.epsilon
        M = self._config.mobility
        if dt is None:
            # Stability: dt ≤ h⁴ / (M * ε²) (4th-order diffusion CFL)
            # Use conservative factor
            self._dt = 0.1 * h * h / max(M * (eps ** 2 + 1.0), 1e-30)
        else:
            self._dt = dt

    @property
    def domain(self) -> str:
        return "phase_field_2d"

    @property
    def domain_label(self) -> str:
        return "2D Phase-Field (Cahn-Hilliard)"

    def compile(self) -> Program:
        """Compile the Cahn-Hilliard system into QTT VM bytecode.

        Register allocation
        -------------------
        r0  = φ           (phase field)
        r1  = μ           (chemical potential)
        r2  = ∇²φ         (Laplacian of φ)
        r3  = φ³          (cubic term)
        r4  = ε²∇²φ       (scaled Laplacian)
        r5  = ∇²μ         (Laplacian of chemical potential)
        r6  = M·∇²μ       (evolution rate)
        r7  = dt·M·∇²μ    (time increment)
        """
        cfg = self._config
        eps = cfg.epsilon
        M = cfg.mobility
        dt = self._dt
        nb = self._n_bits
        bits = (nb, nb)
        dom = ((0.0, 1.0), (0.0, 1.0))

        if self._ic_type == "droplet":
            def init_phi(x: NDArray, y: NDArray) -> NDArray:
                return circle_droplet_ic(x, y, epsilon=eps)
        elif self._ic_type == "rayleigh_taylor":
            def init_phi(x: NDArray, y: NDArray) -> NDArray:
                return rayleigh_taylor_ic(x, y, epsilon=eps)
        else:
            raise ValueError(f"Unknown IC type: {self._ic_type}")

        def invariant_fn(fields: dict) -> float:
            """Total phase: ∫φ dA (conserved by Cahn-Hilliard)."""
            phi = fields["phi"]
            hx = phi.grid_spacing(0)
            hy = phi.grid_spacing(1)
            return hx * hy * phi.sum()

        instructions: list[Instruction] = [
            loop_start(self._n_steps),

            load_field(0, "phi"),

            # ── Chemical potential: μ = -ε²∇²φ + φ³ - φ ─────────────
            # Step 1: ∇²φ
            laplace(2, 0),                         # r2 = ∇²φ

            # Step 2: ε²∇²φ
            scale(4, 2, eps * eps),                # r4 = ε²∇²φ
            negate(4, 4),                          # r4 = -ε²∇²φ

            # Step 3: φ³ (via two Hadamard products)
            hadamard(3, 0, 0),                     # r3 = φ²
            truncate(3),
            hadamard(3, 3, 0),                     # r3 = φ³
            truncate(3),

            # Step 4: μ = -ε²∇²φ + φ³ - φ
            add(1, 4, 3),                          # r1 = -ε²∇²φ + φ³
            truncate(1),
            sub(1, 1, 0),                          # r1 = μ = -ε²∇²φ + φ³ - φ
            truncate(1),

            # ── Evolution: ∂φ/∂t = M·∇²μ ────────────────────────────
            laplace(5, 1),                         # r5 = ∇²μ
            scale(6, 5, M),                        # r6 = M·∇²μ
            scale(7, 6, dt),                       # r7 = dt·M·∇²μ
            add(0, 0, 7),                          # φ += dt·M·∇²μ
            truncate(0),

            bc_apply(0, BCKind.PERIODIC),

            store_field(0, "phi"),
            measure(0, "phi"),

            loop_end(),
        ]

        fields: dict[str, FieldSpec] = {
            "phi": FieldSpec(
                name="phi",
                n_dims=2,
                bits_per_dim=bits,
                bc=BCKind.PERIODIC,
                bc_params={"domain": dom},
                initial_fn="init_phi",
                conserved_quantity="total_phase",
            ),
        }

        metadata: dict[str, Any] = {
            "init_phi": init_phi,
            "invariant_fn": invariant_fn,
            "invariant": "total_phase",
            "equations": "∂φ/∂t = M∇²(−ε²∇²φ + φ³ − φ)",
            "epsilon": eps,
            "mobility": M,
            "sigma": cfg.sigma,
            "rho_1": cfg.rho_1,
            "rho_2": cfg.rho_2,
            "ic_type": self._ic_type,
        }

        return Program(
            domain=self.domain,
            domain_label=self.domain_label,
            n_registers=8,
            fields=fields,
            instructions=instructions,
            dt=self._dt,
            n_steps=self._n_steps,
            params={
                "epsilon": eps,
                "mobility": M,
                "sigma": cfg.sigma,
            },
            metadata=metadata,
        )


# ══════════════════════════════════════════════════════════════════════
# Phase-field utilities
# ══════════════════════════════════════════════════════════════════════

def compute_interface_energy(
    phi_values: NDArray,
    epsilon: float,
    h: float,
) -> float:
    """Compute Ginzburg-Landau free energy.

    F = ∫[ε²/2 |∇φ|² + ¼(φ²-1)²] dΩ

    Parameters
    ----------
    phi_values : NDArray
        Phase-field values (2D).
    epsilon : float
        Interface thickness.
    h : float
        Grid spacing.

    Returns
    -------
    float
        Total interface energy.
    """
    # Gradient energy (central differences)
    grad_x = np.zeros_like(phi_values)
    grad_y = np.zeros_like(phi_values)

    grad_x[1:-1, :] = (phi_values[2:, :] - phi_values[:-2, :]) / (2 * h)
    grad_y[:, 1:-1] = (phi_values[:, 2:] - phi_values[:, :-2]) / (2 * h)

    grad_energy = 0.5 * epsilon ** 2 * (grad_x ** 2 + grad_y ** 2)

    # Double-well potential energy
    potential_energy = 0.25 * (phi_values ** 2 - 1.0) ** 2

    return float(h * h * np.sum(grad_energy + potential_energy))


def compute_phase_fraction(phi_values: NDArray) -> tuple[float, float]:
    """Compute volume fraction of each phase.

    Phase 1: φ < 0,  Phase 2: φ > 0.

    Parameters
    ----------
    phi_values : NDArray
        Phase-field values.

    Returns
    -------
    (f1, f2) : tuple[float, float]
        Volume fractions of phase 1 and phase 2.
    """
    n_total = phi_values.size
    # Smooth volume fraction via 0.5*(1 - φ) and 0.5*(1 + φ)
    f1 = float(np.mean(0.5 * (1.0 - phi_values)))
    f2 = float(np.mean(0.5 * (1.0 + phi_values)))
    return f1, f2


def compute_density_field(
    phi_values: NDArray,
    rho_1: float,
    rho_2: float,
) -> NDArray:
    """Compute mixture density from phase field.

    ρ(φ) = ρ₁ · (1-φ)/2 + ρ₂ · (1+φ)/2

    Parameters
    ----------
    phi_values : NDArray
        Phase-field values.
    rho_1 : float
        Phase 1 density.
    rho_2 : float
        Phase 2 density.

    Returns
    -------
    NDArray
        Mixture density field.
    """
    return rho_1 * 0.5 * (1.0 - phi_values) + rho_2 * 0.5 * (1.0 + phi_values)


def compute_viscosity_field(
    phi_values: NDArray,
    mu_1: float,
    mu_2: float,
) -> NDArray:
    """Compute mixture viscosity from phase field.

    μ(φ) = μ₁ · (1-φ)/2 + μ₂ · (1+φ)/2

    Parameters
    ----------
    phi_values : NDArray
        Phase-field values.
    mu_1 : float
        Phase 1 viscosity.
    mu_2 : float
        Phase 2 viscosity.

    Returns
    -------
    NDArray
        Mixture viscosity field.
    """
    return mu_1 * 0.5 * (1.0 - phi_values) + mu_2 * 0.5 * (1.0 + phi_values)


# ══════════════════════════════════════════════════════════════════════
# Diagnostic sanitizer
# ══════════════════════════════════════════════════════════════════════

_PHASE_FIELD_DIAGNOSTIC_WHITELIST = frozenset({
    "total_phase",
    "phase_1_fraction",
    "phase_2_fraction",
    "interface_energy",
    "mass_conservation_error",
})


def sanitize_phase_field_diagnostics(
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Filter phase-field diagnostics to whitelisted aggregates.

    Parameters
    ----------
    diagnostics : dict[str, Any]
        Raw diagnostics.

    Returns
    -------
    dict[str, Any]
        Only whitelisted scalar aggregates.
    """
    return {
        k: v for k, v in diagnostics.items()
        if k in _PHASE_FIELD_DIAGNOSTIC_WHITELIST
    }
