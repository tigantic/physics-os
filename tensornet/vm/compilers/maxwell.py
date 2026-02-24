"""QTT Physics VM — Maxwell equations compiler.

1-D Maxwell equations (TE mode):

    ∂E/∂t = c · ∂B/∂x
    ∂B/∂t = c · ∂E/∂x

Leap-frog (symplectic) time integration.
Initial condition: Gaussian E-field pulse, B = 0.
Conserved quantity: electromagnetic energy ½∫(E² + B²)dx.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..ir import (
    BCKind, FieldSpec, Instruction, Program,
    add, bc_apply, grad, load_field, loop_end, loop_start,
    measure, scale, store_field, truncate,
)
from .base import BaseCompiler


class MaxwellCompiler(BaseCompiler):
    """Compile 1-D Maxwell equations into QTT VM bytecode.

    Parameters
    ----------
    n_bits : int
        Grid resolution: N = 2^n_bits.
    n_steps : int
        Number of time steps.
    c : float
        Wave speed (speed of light in normalized units).
    dt : float | None
        Time step.  Auto from CFL if None.
    """

    def __init__(
        self,
        n_bits: int = 8,
        n_steps: int = 100,
        c: float = 1.0,
        dt: float | None = None,
    ) -> None:
        self._n_bits = n_bits
        self._n_steps = n_steps
        self._c = c
        N = 2 ** n_bits
        h = 1.0 / N
        if dt is None:
            self._dt = 0.4 * h / c  # CFL for wave equation
        else:
            self._dt = dt

    @property
    def domain(self) -> str:
        return "maxwell"

    @property
    def domain_label(self) -> str:
        return "Maxwell Equations (1D TE mode)"

    def compile(self) -> Program:
        c = self._c
        dt = self._dt

        sigma = 0.05

        def init_E(x: NDArray) -> NDArray:
            return np.exp(-((x - 0.5) ** 2) / (2.0 * sigma ** 2))

        def init_B(x: NDArray) -> NDArray:
            return np.zeros_like(x)

        def invariant_fn(fields: dict) -> float:
            E, B = fields["E"], fields["B"]
            h = E.grid_spacing(0)
            return 0.5 * h * (E.inner(E) + B.inner(B))

        # Register allocation:
        # r0 = E
        # r1 = B
        # r2 = ∂B/∂x
        # r3 = c·dt · ∂B/∂x
        # r4 = ∂E/∂x
        # r5 = c·dt · ∂E/∂x
        cdt = c * dt

        instructions: list[Instruction] = [
            loop_start(self._n_steps),

            # Half-step E
            load_field(0, "E"),
            load_field(1, "B"),
            grad(2, 1, dim=0),                     # r2 = ∂B/∂x
            scale(3, 2, cdt),                      # r3 = c·dt · ∂B/∂x
            add(0, 0, 3),                          # E += c·dt · ∂B/∂x
            truncate(0),
            bc_apply(0, BCKind.PERIODIC),

            # Full-step B
            grad(4, 0, dim=0),                     # r4 = ∂E/∂x
            scale(5, 4, cdt),                      # r5 = c·dt · ∂E/∂x
            add(1, 1, 5),                          # B += c·dt · ∂E/∂x
            truncate(1),
            bc_apply(1, BCKind.PERIODIC),

            store_field(0, "E"),
            store_field(1, "B"),
            measure(0, "E"),
            measure(1, "B"),

            loop_end(),
        ]

        domain_bounds = (0.0, 1.0)
        return Program(
            domain=self.domain,
            domain_label=self.domain_label,
            n_registers=6,
            fields={
                "E": FieldSpec(
                    name="E",
                    n_dims=1,
                    bits_per_dim=(self._n_bits,),
                    bc=BCKind.PERIODIC,
                    bc_params={"domain": domain_bounds},
                    initial_fn="init_E",
                    conserved_quantity="em_energy",
                ),
                "B": FieldSpec(
                    name="B",
                    n_dims=1,
                    bits_per_dim=(self._n_bits,),
                    bc=BCKind.PERIODIC,
                    bc_params={"domain": domain_bounds},
                    initial_fn="init_B",
                ),
            },
            instructions=instructions,
            dt=self._dt,
            n_steps=self._n_steps,
            params={"c": self._c},
            metadata={
                "init_E": init_E,
                "init_B": init_B,
                "invariant_fn": invariant_fn,
                "invariant": "em_energy",
                "equations": "∂E/∂t = c·∂B/∂x, ∂B/∂t = c·∂E/∂x",
            },
        )
