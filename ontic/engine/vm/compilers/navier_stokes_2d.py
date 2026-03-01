"""QTT Physics VM — 2D Navier–Stokes compiler.

2-D incompressible Navier–Stokes with lid-driven cavity flow:

    ∂ω/∂t + u·∂ω/∂x + v·∂ω/∂y = ν ∇²ω
    ∇²ψ = -ω
    u = ∂ψ/∂y,  v = -∂ψ/∂x

Vorticity-stream function formulation avoids the pressure Poisson
equation and enforces incompressibility exactly.

Explicit Euler time integration on a 2D QTT grid with
``bits_per_dim = (n_bits, n_bits)`` cores.

Initial condition: Taylor–Green vortex.
Conserved quantity: total circulation Γ = ∫ω dA (Kelvin's theorem,
periodic BC; also conserved under viscous diffusion since
∫∇²ω dA = 0 for periodic domains).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..ir import (
    BCKind, FieldSpec, Instruction, Program,
    add, bc_apply, grad, hadamard, laplace, laplace_solve,
    load_field, loop_end, loop_start, measure, negate,
    scale, store_field, truncate,
)
from .base import BaseCompiler


class NavierStokes2DCompiler(BaseCompiler):
    """Compile 2-D vorticity-stream NS into QTT VM bytecode.

    Parameters
    ----------
    n_bits : int
        Bits per spatial dimension.  Grid is 2^n_bits × 2^n_bits.
    n_steps : int
        Number of time steps.
    viscosity : float
        Kinematic viscosity ν.
    dt : float | None
        Time step.  Auto from CFL if None.
    """

    def __init__(
        self,
        n_bits: int = 6,
        n_steps: int = 50,
        viscosity: float = 0.01,
        dt: float | None = None,
    ) -> None:
        self._n_bits = n_bits
        self._n_steps = n_steps
        self._viscosity = viscosity
        N = 2 ** n_bits
        h = 1.0 / N
        if dt is None:
            self._dt = 0.25 * h * h / (2.0 * viscosity + 1e-30)
        else:
            self._dt = dt

    @property
    def domain(self) -> str:
        return "navier_stokes_2d"

    @property
    def domain_label(self) -> str:
        return "2D Navier–Stokes (vorticity-stream)"

    def compile(self) -> Program:
        nu = self._viscosity
        dt = self._dt
        nb = self._n_bits
        bits = (nb, nb)
        dom = ((0.0, 1.0), (0.0, 1.0))

        def init_omega(x: NDArray, y: NDArray) -> NDArray:
            """Taylor–Green-like initial vortex."""
            return (
                2.0 * np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)
            )

        def init_psi(x: NDArray, y: NDArray) -> NDArray:
            return np.zeros_like(x)

        def invariant_fn(fields: dict) -> float:
            omega = fields["omega"]
            hx = omega.grid_spacing(0)
            hy = omega.grid_spacing(1)
            return hx * hy * omega.sum()

        # Register allocation:
        # r0 = ω,       r1 = ψ
        # r2 = u = ∂ψ/∂y
        # r3 = v_neg = ∂ψ/∂x  (v = -∂ψ/∂x, handled via negate)
        # r4 = ∂ω/∂x,  r5 = ∂ω/∂y
        # r6 = u·∂ω/∂x, r7 = v·∂ω/∂y  (advection terms)
        # r8 = advection total = -(u·∂ω/∂x + v·∂ω/∂y)
        # r9 = ∇²ω (diffusion)
        # r10 = RHS, r11 = dt*RHS

        instructions: list[Instruction] = [
            loop_start(self._n_steps),

            load_field(0, "omega"),
            load_field(1, "psi"),

            # Solve Poisson: ∇²ψ = -ω  → ψ = Lap⁻¹(-ω)
            negate(11, 0),                         # r11 = -ω
            laplace_solve(1, 11),                  # r1 = ψ (Poisson solve)

            # Velocity from stream function
            grad(2, 1, dim=1),                     # r2 = u = ∂ψ/∂y
            grad(3, 1, dim=0),                     # r3 = ∂ψ/∂x
            negate(3, 3),                          # r3 = v = -∂ψ/∂x

            # Vorticity gradient
            grad(4, 0, dim=0),                     # r4 = ∂ω/∂x
            grad(5, 0, dim=1),                     # r5 = ∂ω/∂y

            # Advection: u·∂ω/∂x + v·∂ω/∂y
            hadamard(6, 2, 4),                     # r6 = u·∂ω/∂x
            hadamard(7, 3, 5),                     # r7 = v·∂ω/∂y
            add(8, 6, 7),                          # r8 = u·∂ω/∂x + v·∂ω/∂y
            negate(8, 8),                          # r8 = -(advection)

            # Diffusion: ν∇²ω
            laplace(9, 0),                         # r9 = ∇²ω
            scale(9, 9, nu),                       # r9 = ν∇²ω

            # RHS and time update
            add(10, 8, 9),                         # r10 = -adv + ν∇²ω
            scale(11, 10, dt),                     # r11 = dt * RHS
            add(0, 0, 11),                         # ω += dt * RHS
            truncate(0),
            bc_apply(0, BCKind.PERIODIC),

            store_field(0, "omega"),
            store_field(1, "psi"),
            measure(0, "omega"),

            loop_end(),
        ]

        return Program(
            domain=self.domain,
            domain_label=self.domain_label,
            n_registers=12,
            fields={
                "omega": FieldSpec(
                    name="omega",
                    n_dims=2,
                    bits_per_dim=bits,
                    bc=BCKind.PERIODIC,
                    bc_params={"domain": dom},
                    initial_fn="init_omega",
                    conserved_quantity="total_circulation",
                ),
                "psi": FieldSpec(
                    name="psi",
                    n_dims=2,
                    bits_per_dim=bits,
                    bc=BCKind.PERIODIC,
                    bc_params={"domain": dom},
                    initial_fn="init_psi",
                ),
            },
            instructions=instructions,
            dt=self._dt,
            n_steps=self._n_steps,
            params={"viscosity": self._viscosity},
            metadata={
                "init_omega": init_omega,
                "init_psi": init_psi,
                # Separable: 2 sin(2πx) sin(2πy) = f(x) × g(y)
                "init_omega_separable": (
                    [
                        lambda x: np.sin(2.0 * np.pi * x),
                        lambda y: np.sin(2.0 * np.pi * y),
                    ],
                    2.0,  # scale factor
                ),
                # init_psi: zero — handled automatically by GPUQTTTensor.zeros()
                "invariant_fn": invariant_fn,
                "invariant": "total_circulation",
                "equations": "∂ω/∂t + (u·∇)ω = ν∇²ω, ∇²ψ = −ω",
            },
        )
