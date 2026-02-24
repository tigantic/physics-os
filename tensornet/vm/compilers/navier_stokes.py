"""QTT Physics VM — Burgers equation compiler.

Viscous Burgers equation (1-D Navier–Stokes):

    ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²

Explicit Euler time integration.
Initial condition: u(x, 0) = sin(2πx) on [0, 1] with periodic BC.
Conserved quantity: ∫u dx (for periodic BC, ∂/∂t ∫u dx = 0 since
the advection term integrates to zero and the diffusion term vanishes).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..ir import (
    BCKind, FieldSpec, Instruction, Program,
    add, bc_apply, grad, hadamard, laplace, load_field,
    loop_end, loop_start, measure, negate, scale, store_field, truncate,
)
from .base import BaseCompiler


class BurgersCompiler(BaseCompiler):
    """Compile the 1-D Burgers equation into QTT VM bytecode.

    Parameters
    ----------
    n_bits : int
        Grid resolution: N = 2^n_bits points.
    n_steps : int
        Number of time steps.
    viscosity : float
        Kinematic viscosity ν.
    dt : float | None
        Time step.  Auto-computed from CFL if None.
    """

    def __init__(
        self,
        n_bits: int = 8,
        n_steps: int = 100,
        viscosity: float = 0.01,
        dt: float | None = None,
    ) -> None:
        self._n_bits = n_bits
        self._n_steps = n_steps
        self._viscosity = viscosity
        N = 2 ** n_bits
        h = 1.0 / N
        if dt is None:
            # CFL: dt ≤ h² / (2ν) for stability
            self._dt = 0.4 * h * h / (2.0 * viscosity + 1e-30)
        else:
            self._dt = dt

    @property
    def domain(self) -> str:
        return "burgers"

    @property
    def domain_label(self) -> str:
        return "Viscous Burgers (1D Navier–Stokes)"

    def compile(self) -> Program:
        nu = self._viscosity
        dt = self._dt

        def init_u(x: NDArray) -> NDArray:
            return np.sin(2.0 * np.pi * x)

        def invariant_fn(fields: dict) -> float:
            u = fields["u"]
            h = u.grid_spacing(0)
            return h * u.sum()

        # Register allocation:
        # r0 = u
        # r1 = ∂u/∂x
        # r2 = u * ∂u/∂x  (advection)
        # r3 = ∇²u (diffusion)
        # r4 = RHS
        instructions: list[Instruction] = [
            loop_start(self._n_steps),

            load_field(0, "u"),
            grad(1, 0, dim=0),                    # r1 = ∂u/∂x
            hadamard(2, 0, 1),                     # r2 = u ⊙ ∂u/∂x
            negate(2, 2),                          # r2 = -u ∂u/∂x
            laplace(3, 0),                         # r3 = ∇²u
            scale(3, 3, nu),                       # r3 = ν∇²u
            add(4, 2, 3),                          # r4 = -u∂u/∂x + ν∇²u
            scale(4, 4, dt),                       # r4 = dt * RHS
            add(0, 0, 4),                          # u += dt * RHS
            truncate(0),
            bc_apply(0, BCKind.PERIODIC),
            store_field(0, "u"),
            measure(0, "u"),

            loop_end(),
        ]

        return Program(
            domain=self.domain,
            domain_label=self.domain_label,
            n_registers=5,
            fields={
                "u": FieldSpec(
                    name="u",
                    n_dims=1,
                    bits_per_dim=(self._n_bits,),
                    bc=BCKind.PERIODIC,
                    bc_params={"domain": (0.0, 1.0)},
                    initial_fn="init_u",
                    conserved_quantity="total_mass",
                ),
            },
            instructions=instructions,
            dt=self._dt,
            n_steps=self._n_steps,
            params={"viscosity": self._viscosity},
            metadata={
                "init_u": init_u,
                "invariant_fn": invariant_fn,
                "invariant": "total_mass",
                "equations": "∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²",
            },
        )
