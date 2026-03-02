"""QTT Physics VM — 1D Compressible Euler Equations Compiler.

Solves the 1D Euler system of conservation laws:

    ∂U/∂t + ∂F(U)/∂x = 0

where U = [ρ, ρu, E] and the flux function is:

    F(U) = [ρu, ρu² + p, (E + p)u]

with the ideal gas equation of state:

    p = (γ - 1)(E - ½ρu²)

Time integration: explicit Euler (forward Euler) per instruction
stream.  Spatial differencing: QTT-native gradient MPO applied to
flux terms (central difference + truncation-governed dissipation).

Boundedness predicates: ρ > 0, p > 0 checked after each step for
evidence generation.  Violation produces immediate claim failure.

Benchmark alignment
-------------------
- Sod shock tube (C110):  ρ_L = 1.0, p_L = 1.0 | ρ_R = 0.125, p_R = 0.1
- Shu-Osher (C120):       ρ = 1 + 0.2 sin(5x) behind shock at x = -4

IP Boundary Compliance
----------------------
Internal solver state (flux fields, intermediate registers, density
reconstruction) is never exposed.  Only sanitizer-safe scalar QoIs
(L² error, conservation balance, boundedness pass/fail) can leave.

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..ir import (
    BCKind, FieldSpec, Instruction, OpCode, Program,
    add, bc_apply, grad, hadamard, laplace, load_field,
    loop_end, loop_start, measure, negate, scale,
    store_field, sub, truncate,
)
from .base import BaseCompiler


# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class EulerConfig:
    """Configuration for the 1D compressible Euler compiler.

    Parameters
    ----------
    gamma : float
        Ratio of specific heats (default: 1.4 for diatomic ideal gas).
    cfl : float
        CFL number for time step selection.
    artificial_viscosity : float
        Artificial viscosity coefficient for shock stabilization.
        Applied as a Laplacian diffusion term: ε∇²U.
    """
    gamma: float = 1.4
    cfl: float = 0.3
    artificial_viscosity: float = 0.01


# ══════════════════════════════════════════════════════════════════════
# Initial conditions
# ══════════════════════════════════════════════════════════════════════

def sod_shock_tube_ic(
    x: NDArray,
    gamma: float = 1.4,
) -> tuple[NDArray, NDArray, NDArray]:
    """Sod shock tube initial condition.

    Left state:  ρ = 1.0,   u = 0.0,  p = 1.0
    Right state: ρ = 0.125, u = 0.0,  p = 0.1
    Discontinuity at x = 0.5.

    Returns (ρ, ρu, E) fields.
    """
    rho = np.where(x < 0.5, 1.0, 0.125)
    u = np.zeros_like(x)
    p = np.where(x < 0.5, 1.0, 0.1)
    rho_u = rho * u
    E = p / (gamma - 1.0) + 0.5 * rho * u ** 2
    return rho, rho_u, E


def shu_osher_ic(
    x: NDArray,
    gamma: float = 1.4,
) -> tuple[NDArray, NDArray, NDArray]:
    """Shu-Osher problem initial condition.

    Left state (x < -4):  ρ = 3.857143, u = 2.629369, p = 10.33333
    Right state (x >= -4): ρ = 1 + 0.2*sin(5x), u = 0, p = 1

    Domain: [-5, 5].

    Returns (ρ, ρu, E) fields.
    """
    rho = np.where(
        x < -4.0,
        3.857143,
        1.0 + 0.2 * np.sin(5.0 * x),
    )
    u = np.where(x < -4.0, 2.629369, 0.0)
    p = np.where(x < -4.0, 10.33333, 1.0)
    rho_u = rho * u
    E = p / (gamma - 1.0) + 0.5 * rho * u ** 2
    return rho, rho_u, E


def smooth_sine_ic(
    x: NDArray,
    gamma: float = 1.4,
) -> tuple[NDArray, NDArray, NDArray]:
    """Smooth sinusoidal density wave (MMS-friendly).

    ρ = 1 + 0.2 * sin(2πx),  u = 1.0,  p = 1.0

    Domain: [0, 1], periodic BC.
    Exact solution: ρ(x,t) = 1 + 0.2 * sin(2π(x - t))

    Returns (ρ, ρu, E) fields.
    """
    rho = 1.0 + 0.2 * np.sin(2.0 * np.pi * x)
    u = np.ones_like(x)
    p = np.ones_like(x)
    rho_u = rho * u
    E = p / (gamma - 1.0) + 0.5 * rho * u ** 2
    return rho, rho_u, E


# ══════════════════════════════════════════════════════════════════════
# Compiler
# ══════════════════════════════════════════════════════════════════════

class CompressibleEuler1DCompiler(BaseCompiler):
    """Compile 1D compressible Euler equations into QTT VM bytecode.

    The Euler system is split into 3 conservation equations for
    density ρ, momentum ρu, and total energy E.  Each is advanced
    using explicit Euler with QTT-native gradient operators for
    spatial derivatives of the flux vector.

    Parameters
    ----------
    n_bits : int
        Grid resolution (N = 2^n_bits points).
    n_steps : int
        Number of time steps.
    domain_bounds : tuple[float, float]
        Physical domain [x_min, x_max].
    config : EulerConfig | None
        Solver configuration (gamma, CFL, artificial viscosity).
    dt : float | None
        Time step size.  Auto-computed from CFL if None.
    ic_type : str
        Initial condition type: "sod", "shu_osher", "smooth_sine".
    bc_kind : BCKind
        Boundary condition type (DIRICHLET for Sod, PERIODIC for smooth).
    """

    def __init__(
        self,
        n_bits: int = 10,
        n_steps: int = 200,
        domain_bounds: tuple[float, float] = (0.0, 1.0),
        config: EulerConfig | None = None,
        dt: float | None = None,
        ic_type: str = "sod",
        bc_kind: BCKind = BCKind.DIRICHLET,
    ) -> None:
        self._n_bits = n_bits
        self._n_steps = n_steps
        self._domain = domain_bounds
        self._config = config or EulerConfig()
        self._ic_type = ic_type
        self._bc_kind = bc_kind

        N = 2 ** n_bits
        h = (domain_bounds[1] - domain_bounds[0]) / N
        if dt is None:
            # Estimate max wave speed for CFL
            # Sod: max(|u| + c) ~ sqrt(gamma * p_max / rho_min) ~ 3.0
            max_wave = 3.0
            self._dt = self._config.cfl * h / max_wave
        else:
            self._dt = dt

    @property
    def domain(self) -> str:
        return "compressible_euler_1d"

    @property
    def domain_label(self) -> str:
        return "1D Compressible Euler Equations"

    def compile(self) -> Program:
        """Compile the Euler system into QTT VM bytecode.

        Register allocation
        -------------------
        r0  = ρ   (density)
        r1  = ρu  (momentum)
        r2  = E   (total energy)
        r3  = u   (velocity = ρu / ρ)
        r4  = p   (pressure)
        r5  = flux_rho = ρu
        r6  = flux_mom = ρu² + p
        r7  = flux_E   = (E + p) · u
        r8  = ∂(flux_rho)/∂x
        r9  = ∂(flux_mom)/∂x
        r10 = ∂(flux_E)/∂x
        r11 = scratch / dt scaling
        r12 = diffusion scratch (artificial viscosity)
        """
        cfg = self._config
        gamma = cfg.gamma
        dt = self._dt
        eps = cfg.artificial_viscosity
        nb = self._n_bits
        dom = self._domain

        # Select initial condition
        if self._ic_type == "sod":
            ic_fn = lambda x: sod_shock_tube_ic(x, gamma)
        elif self._ic_type == "shu_osher":
            ic_fn = lambda x: shu_osher_ic(x, gamma)
        elif self._ic_type == "smooth_sine":
            ic_fn = lambda x: smooth_sine_ic(x, gamma)
        else:
            raise ValueError(f"Unknown IC type: {self._ic_type}")

        def init_rho(x: NDArray) -> NDArray:
            rho, _, _ = ic_fn(x)
            return rho

        def init_rho_u(x: NDArray) -> NDArray:
            _, rho_u, _ = ic_fn(x)
            return rho_u

        def init_E(x: NDArray) -> NDArray:
            _, _, E = ic_fn(x)
            return E

        def invariant_fn(fields: dict) -> float:
            """Total mass: ∫ρ dx (conserved for Euler)."""
            rho = fields["rho"]
            h = rho.grid_spacing(0)
            return h * rho.sum()

        # ── Instruction stream ───────────────────────────────────────
        # Euler equations with artificial viscosity for stabilization:
        #   ∂ρ/∂t  = -∂(ρu)/∂x               + ε∇²ρ
        #   ∂(ρu)/∂t = -∂(ρu² + p)/∂x        + ε∇²(ρu)
        #   ∂E/∂t = -∂((E + p)u)/∂x           + ε∇²E
        #
        # Pressure: p = (γ-1)(E - ½ρu²)
        # Velocity: u = ρu / ρ (via reciprocal approximation)

        n_regs = 13
        instructions: list[Instruction] = [
            loop_start(self._n_steps),

            load_field(0, "rho"),
            load_field(1, "rho_u"),
            load_field(2, "E"),

            # ── Primitive variables ──────────────────────────────────
            # u = ρu / ρ ≈ ρu ⊙ (1/ρ)
            # We approximate by: u = ρu ⊙ recip_rho (precomputed)
            # For now, load inverse density as a field
            load_field(3, "inv_rho"),

            # r3 = u = ρu ⊙ (1/ρ)
            hadamard(3, 1, 3),
            truncate(3),

            # ── Pressure: p = (γ-1)(E - ½ρu²) ──────────────────────
            # r4 = ρu² = ρu ⊙ u
            hadamard(4, 1, 3),
            truncate(4),
            # r4 = ½ρu²
            scale(4, 4, 0.5),
            # r4 = E - ½ρu²
            sub(4, 2, 4),
            truncate(4),
            # r4 = p = (γ-1)(E - ½ρu²)
            scale(4, 4, gamma - 1.0),

            # ── Fluxes ───────────────────────────────────────────────
            # flux_rho = ρu (already r1)
            # r5 = flux_rho = ρu (copy via scale(1))
            scale(5, 1, 1.0),

            # flux_mom = ρu² + p = ρu ⊙ u + p
            hadamard(6, 1, 3),          # r6 = ρu ⊙ u = ρu²
            truncate(6),
            add(6, 6, 4),               # r6 = ρu² + p
            truncate(6),

            # flux_E = (E + p) ⊙ u
            add(7, 2, 4),               # r7 = E + p
            truncate(7),
            hadamard(7, 7, 3),          # r7 = (E+p) ⊙ u
            truncate(7),

            # ── Flux gradients ───────────────────────────────────────
            grad(8, 5, dim=0),          # r8 = ∂(flux_rho)/∂x
            grad(9, 6, dim=0),          # r9 = ∂(flux_mom)/∂x
            grad(10, 7, dim=0),         # r10 = ∂(flux_E)/∂x

            # ── Time update: U_{n+1} = U_n - dt * ∂F/∂x ────────────
            # ρ update
            scale(11, 8, dt),
            sub(0, 0, 11),              # ρ -= dt * ∂(flux_rho)/∂x
            truncate(0),

            # ρu update
            scale(11, 9, dt),
            sub(1, 1, 11),              # ρu -= dt * ∂(flux_mom)/∂x
            truncate(1),

            # E update
            scale(11, 10, dt),
            sub(2, 2, 11),              # E -= dt * ∂(flux_E)/∂x
            truncate(2),
        ]

        # ── Artificial viscosity (stabilization) ─────────────────────
        if eps > 0:
            instructions.extend([
                # ρ += dt * ε * ∇²ρ
                laplace(12, 0),
                scale(12, 12, dt * eps),
                add(0, 0, 12),
                truncate(0),

                # ρu += dt * ε * ∇²(ρu)
                laplace(12, 1),
                scale(12, 12, dt * eps),
                add(1, 1, 12),
                truncate(1),

                # E += dt * ε * ∇²E
                laplace(12, 2),
                scale(12, 12, dt * eps),
                add(2, 2, 12),
                truncate(2),
            ])

        # ── Boundary conditions ──────────────────────────────────────
        instructions.extend([
            bc_apply(0, self._bc_kind),
            bc_apply(1, self._bc_kind),
            bc_apply(2, self._bc_kind),

            # Store updated fields
            store_field(0, "rho"),
            store_field(1, "rho_u"),
            store_field(2, "E"),

            # Telemetry
            measure(0, "rho"),

            loop_end(),
        ])

        # ── Field specifications ─────────────────────────────────────
        bits = (nb,)
        fields: dict[str, FieldSpec] = {
            "rho": FieldSpec(
                name="rho",
                n_dims=1,
                bits_per_dim=bits,
                bc=self._bc_kind,
                bc_params={"domain": dom},
                initial_fn="init_rho",
                conserved_quantity="total_mass",
            ),
            "rho_u": FieldSpec(
                name="rho_u",
                n_dims=1,
                bits_per_dim=bits,
                bc=self._bc_kind,
                bc_params={"domain": dom},
                initial_fn="init_rho_u",
                conserved_quantity="total_momentum",
            ),
            "E": FieldSpec(
                name="E",
                n_dims=1,
                bits_per_dim=bits,
                bc=self._bc_kind,
                bc_params={"domain": dom},
                initial_fn="init_E",
                conserved_quantity="total_energy",
            ),
            "inv_rho": FieldSpec(
                name="inv_rho",
                n_dims=1,
                bits_per_dim=bits,
                bc=self._bc_kind,
                bc_params={"domain": dom},
                initial_fn="init_inv_rho",
            ),
        }

        metadata: dict[str, Any] = {
            "init_rho": init_rho,
            "init_rho_u": init_rho_u,
            "init_E": init_E,
            "init_inv_rho": lambda x: 1.0 / np.maximum(init_rho(x), 1e-15),
            "invariant_fn": invariant_fn,
            "invariant": "total_mass",
            "equations": (
                "∂ρ/∂t + ∂(ρu)/∂x = 0, "
                "∂(ρu)/∂t + ∂(ρu²+p)/∂x = 0, "
                "∂E/∂t + ∂((E+p)u)/∂x = 0"
            ),
            "gamma": gamma,
            "ic_type": self._ic_type,
            "artificial_viscosity": eps,
            "boundedness_predicates": ["rho_positive", "pressure_positive"],
        }

        return Program(
            domain=self.domain,
            domain_label=self.domain_label,
            n_registers=n_regs,
            fields=fields,
            instructions=instructions,
            dt=self._dt,
            n_steps=self._n_steps,
            params={
                "gamma": gamma,
                "artificial_viscosity": eps,
            },
            metadata=metadata,
        )


# ══════════════════════════════════════════════════════════════════════
# Boundedness evidence predicates
# ══════════════════════════════════════════════════════════════════════

def check_density_positive(rho_values: NDArray) -> bool:
    """Check that density is positive everywhere (ρ > 0).

    Parameters
    ----------
    rho_values : NDArray
        Dense density field for post-step verification.

    Returns
    -------
    bool
        True if ρ > 0 everywhere.
    """
    return bool(np.all(rho_values > 0))


def check_pressure_positive(
    E_values: NDArray,
    rho_values: NDArray,
    rho_u_values: NDArray,
    gamma: float = 1.4,
) -> bool:
    """Check that pressure is positive everywhere (p > 0).

    p = (γ-1)(E - ½ρu²)

    Parameters
    ----------
    E_values : NDArray
        Dense total energy field.
    rho_values : NDArray
        Dense density field.
    rho_u_values : NDArray
        Dense momentum field.
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    bool
        True if p > 0 everywhere.
    """
    u = rho_u_values / np.maximum(rho_values, 1e-30)
    p = (gamma - 1.0) * (E_values - 0.5 * rho_values * u ** 2)
    return bool(np.all(p > 0))


def compute_conservation_balance(
    rho_initial: NDArray,
    rho_final: NDArray,
    h: float,
) -> float:
    """Compute conservation balance: |∫ρ_final - ∫ρ_initial| dx.

    Parameters
    ----------
    rho_initial : NDArray
        Initial density field.
    rho_final : NDArray
        Final density field.
    h : float
        Grid spacing.

    Returns
    -------
    float
        Absolute mass balance error.
    """
    mass_initial = float(h * np.sum(rho_initial))
    mass_final = float(h * np.sum(rho_final))
    return abs(mass_final - mass_initial)


# ══════════════════════════════════════════════════════════════════════
# Exact Sod Riemann solution (for V&V reference)
# ══════════════════════════════════════════════════════════════════════

def sod_exact_solution(
    x: NDArray,
    t: float,
    gamma: float = 1.4,
    x0: float = 0.5,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute exact Riemann solution for Sod shock tube.

    Uses iterative Newton-Raphson to find the pressure in the
    star region, then constructs the full piecewise solution.

    Parameters
    ----------
    x : NDArray
        Spatial coordinates.
    t : float
        Time (> 0).
    gamma : float
        Ratio of specific heats.
    x0 : float
        Initial discontinuity position.

    Returns
    -------
    (rho, u, p) : tuple[NDArray, NDArray, NDArray]
        Exact density, velocity, and pressure profiles.
    """
    # Left and right states
    rho_L, u_L, p_L = 1.0, 0.0, 1.0
    rho_R, u_R, p_R = 0.125, 0.0, 0.1

    # Sound speeds
    c_L = math.sqrt(gamma * p_L / rho_L)
    c_R = math.sqrt(gamma * p_R / rho_R)

    gm1 = gamma - 1.0
    gp1 = gamma + 1.0

    # Newton-Raphson for star region pressure
    p_star = 0.5 * (p_L + p_R)
    for _ in range(50):
        # Left wave (rarefaction)
        if p_star <= p_L:
            f_L = (
                (2.0 * c_L / gm1)
                * ((p_star / p_L) ** (gm1 / (2.0 * gamma)) - 1.0)
            )
            df_L = (
                (1.0 / (rho_L * c_L))
                * (p_star / p_L) ** (-(gp1) / (2.0 * gamma))
            )
        else:
            A_L = 2.0 / (gp1 * rho_L)
            B_L = gm1 / gp1 * p_L
            f_L = (p_star - p_L) * math.sqrt(A_L / (p_star + B_L))
            df_L = math.sqrt(A_L / (p_star + B_L)) * (
                1.0 - (p_star - p_L) / (2.0 * (p_star + B_L))
            )

        # Right wave (shock)
        if p_star <= p_R:
            f_R = (
                (2.0 * c_R / gm1)
                * ((p_star / p_R) ** (gm1 / (2.0 * gamma)) - 1.0)
            )
            df_R = (
                (1.0 / (rho_R * c_R))
                * (p_star / p_R) ** (-(gp1) / (2.0 * gamma))
            )
        else:
            A_R = 2.0 / (gp1 * rho_R)
            B_R = gm1 / gp1 * p_R
            f_R = (p_star - p_R) * math.sqrt(A_R / (p_star + B_R))
            df_R = math.sqrt(A_R / (p_star + B_R)) * (
                1.0 - (p_star - p_R) / (2.0 * (p_star + B_R))
            )

        f = f_L + f_R + (u_R - u_L)
        df = df_L + df_R

        if abs(df) < 1e-30:
            break
        dp = -f / df
        p_star = max(p_star + dp, 1e-15)
        if abs(dp) < 1e-12 * p_star:
            break

    u_star = 0.5 * (u_L + u_R) + 0.5 * (f_R - f_L)

    # Construct solution
    rho = np.zeros_like(x)
    u_out = np.zeros_like(x)
    p_out = np.zeros_like(x)

    xi = (x - x0) / max(t, 1e-30)

    for i in range(len(x)):
        s = xi[i]

        if s < u_star:
            # Left of contact
            if p_star <= p_L:
                # Left rarefaction
                c_star_L = c_L * (p_star / p_L) ** (gm1 / (2.0 * gamma))
                s_head = u_L - c_L
                s_tail = u_star - c_star_L
                if s <= s_head:
                    rho[i] = rho_L
                    u_out[i] = u_L
                    p_out[i] = p_L
                elif s >= s_tail:
                    rho[i] = rho_L * (p_star / p_L) ** (1.0 / gamma)
                    u_out[i] = u_star
                    p_out[i] = p_star
                else:
                    # Inside fan
                    rho[i] = rho_L * (
                        (2.0 / gp1 + gm1 / (gp1 * c_L) * (u_L - s))
                        ** (2.0 / gm1)
                    )
                    u_out[i] = 2.0 / gp1 * (c_L + gm1 / 2.0 * u_L + s)
                    p_out[i] = p_L * (
                        (2.0 / gp1 + gm1 / (gp1 * c_L) * (u_L - s))
                        ** (2.0 * gamma / gm1)
                    )
            else:
                # Left shock
                s_shock = u_L - c_L * math.sqrt(
                    (gp1 / (2.0 * gamma)) * (p_star / p_L) + gm1 / (2.0 * gamma)
                )
                if s < s_shock:
                    rho[i] = rho_L
                    u_out[i] = u_L
                    p_out[i] = p_L
                else:
                    rho[i] = rho_L * (
                        (p_star / p_L + gm1 / gp1)
                        / (gm1 / gp1 * p_star / p_L + 1.0)
                    )
                    u_out[i] = u_star
                    p_out[i] = p_star
        else:
            # Right of contact
            if p_star <= p_R:
                # Right rarefaction
                c_star_R = c_R * (p_star / p_R) ** (gm1 / (2.0 * gamma))
                s_head = u_R + c_R
                s_tail = u_star + c_star_R
                if s >= s_head:
                    rho[i] = rho_R
                    u_out[i] = u_R
                    p_out[i] = p_R
                elif s <= s_tail:
                    rho[i] = rho_R * (p_star / p_R) ** (1.0 / gamma)
                    u_out[i] = u_star
                    p_out[i] = p_star
                else:
                    rho[i] = rho_R * (
                        (2.0 / gp1 - gm1 / (gp1 * c_R) * (u_R - s))
                        ** (2.0 / gm1)
                    )
                    u_out[i] = 2.0 / gp1 * (-c_R + gm1 / 2.0 * u_R + s)
                    p_out[i] = p_R * (
                        (2.0 / gp1 - gm1 / (gp1 * c_R) * (u_R - s))
                        ** (2.0 * gamma / gm1)
                    )
            else:
                # Right shock
                s_shock = u_R + c_R * math.sqrt(
                    (gp1 / (2.0 * gamma)) * (p_star / p_R) + gm1 / (2.0 * gamma)
                )
                if s > s_shock:
                    rho[i] = rho_R
                    u_out[i] = u_R
                    p_out[i] = p_R
                else:
                    rho[i] = rho_R * (
                        (p_star / p_R + gm1 / gp1)
                        / (gm1 / gp1 * p_star / p_R + 1.0)
                    )
                    u_out[i] = u_star
                    p_out[i] = p_star

    return rho, u_out, p_out
