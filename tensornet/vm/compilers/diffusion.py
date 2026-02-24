"""QTT Physics VM — Advection-diffusion equation compiler.

1-D advection-diffusion (scalar transport):

    ∂u/∂t + v ∂u/∂x = κ ∂²u/∂x²

where v is a constant advection velocity and κ is diffusivity.

Explicit Euler.
Initial condition: Gaussian pulse.
Conserved quantity: ∫u dx (total mass, periodic BC, advection-only).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..ir import (
    BCKind, FieldSpec, Instruction, Program,
    add, bc_apply, grad, laplace, load_field, loop_end,
    loop_start, measure, scale, store_field, truncate,
)
from .base import BaseCompiler


class DiffusionCompiler(BaseCompiler):
    """Compile 1-D advection-diffusion into QTT VM bytecode.

    Parameters
    ----------
    n_bits : int
        Grid resolution.
    n_steps : int
        Number of time steps.
    velocity : float
        Constant advection velocity v.
    diffusivity : float
        Diffusion coefficient κ.
    dt : float | None
        Time step.
    """

    def __init__(
        self,
        n_bits: int = 8,
        n_steps: int = 100,
        velocity: float = 1.0,
        diffusivity: float = 0.01,
        dt: float | None = None,
    ) -> None:
        self._n_bits = n_bits
        self._n_steps = n_steps
        self._velocity = velocity
        self._diffusivity = diffusivity
        N = 2 ** n_bits
        h = 1.0 / N
        if dt is None:
            # CFL: min(h/v, h²/(2κ))
            dt_adv = 0.4 * h / (abs(velocity) + 1e-30)
            dt_diff = 0.4 * h * h / (2.0 * diffusivity + 1e-30)
            self._dt = min(dt_adv, dt_diff)
        else:
            self._dt = dt

    @property
    def domain(self) -> str:
        return "advection_diffusion"

    @property
    def domain_label(self) -> str:
        return "Advection-Diffusion (scalar transport)"

    def compile(self) -> Program:
        v = self._velocity
        kappa = self._diffusivity
        dt = self._dt
        sigma = 0.05

        def init_u(x: NDArray) -> NDArray:
            return np.exp(-((x - 0.3) ** 2) / (2.0 * sigma ** 2))

        def invariant_fn(fields: dict) -> float:
            u = fields["u"]
            h = u.grid_spacing(0)
            return h * u.sum()

        # Register allocation:
        # r0 = u
        # r1 = ∂u/∂x
        # r2 = -v · ∂u/∂x  (advection)
        # r3 = ∇²u          (diffusion)
        # r4 = κ ∇²u
        # r5 = RHS

        instructions: list[Instruction] = [
            loop_start(self._n_steps),

            load_field(0, "u"),
            grad(1, 0, dim=0),                  # r1 = ∂u/∂x
            scale(2, 1, -v),                    # r2 = -v ∂u/∂x
            laplace(3, 0),                      # r3 = ∇²u
            scale(4, 3, kappa),                 # r4 = κ∇²u
            add(5, 2, 4),                       # r5 = RHS
            scale(5, 5, dt),                    # r5 = dt · RHS
            add(0, 0, 5),                       # u += dt · RHS
            truncate(0),
            bc_apply(0, BCKind.PERIODIC),
            store_field(0, "u"),
            measure(0, "u"),

            loop_end(),
        ]

        domain_bounds = (0.0, 1.0)
        return Program(
            domain=self.domain,
            domain_label=self.domain_label,
            n_registers=6,
            fields={
                "u": FieldSpec(
                    name="u",
                    n_dims=1,
                    bits_per_dim=(self._n_bits,),
                    bc=BCKind.PERIODIC,
                    bc_params={"domain": domain_bounds},
                    initial_fn="init_u",
                    conserved_quantity="total_mass",
                ),
            },
            instructions=instructions,
            dt=self._dt,
            n_steps=self._n_steps,
            params={"velocity": self._velocity, "diffusivity": self._diffusivity},
            metadata={
                "init_u": init_u,
                "invariant_fn": invariant_fn,
                "invariant": "total_mass",
                "equations": "∂u/∂t + v·∂u/∂x = κ·∂²u/∂x²",
            },
        )
