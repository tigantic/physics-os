"""QTT Physics VM — 3D Maxwell equations compiler.

3-D Maxwell equations in vacuum:

    ∂E/∂t = c ∇×B
    ∂B/∂t = -c ∇×E

Component form (curl expansion):

    ∂Ex/∂t = c(∂Bz/∂y − ∂By/∂z)
    ∂Ey/∂t = c(∂Bx/∂z − ∂Bz/∂x)
    ∂Ez/∂t = c(∂By/∂x − ∂Bx/∂y)
    ∂Bx/∂t = -c(∂Ez/∂y − ∂Ey/∂z)
    ∂By/∂t = -c(∂Ex/∂z − ∂Ez/∂x)
    ∂Bz/∂t = -c(∂Ey/∂x − ∂Ex/∂y)

Leap-frog (symplectic) time integration.
Initial condition: 3-D Gaussian EM pulse.
Conserved quantity: electromagnetic energy ½∫(|E|² + |B|²)dV.

Each field is a 3D QTT tensor with ``bits_per_dim = (n, n, n)`` cores.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..ir import (
    BCKind, FieldSpec, Instruction, Program,
    add, bc_apply, grad, load_field, loop_end, loop_start,
    measure, negate, scale, store_field, sub, truncate,
)
from .base import BaseCompiler


class Maxwell3DCompiler(BaseCompiler):
    """Compile 3-D Maxwell equations into QTT VM bytecode.

    Parameters
    ----------
    n_bits : int
        Bits per spatial dimension.  Grid is (2^n)³.
    n_steps : int
        Number of time steps.
    c : float
        Speed of light (normalized).
    dt : float | None
        Time step.  Auto from CFL if None.
    """

    def __init__(
        self,
        n_bits: int = 4,
        n_steps: int = 50,
        c: float = 1.0,
        dt: float | None = None,
    ) -> None:
        self._n_bits = n_bits
        self._n_steps = n_steps
        self._c = c
        N = 2 ** n_bits
        h = 1.0 / N
        if dt is None:
            # 3D CFL: dt ≤ h / (c * √3)
            self._dt = 0.3 * h / (c * np.sqrt(3.0))
        else:
            self._dt = dt

    @property
    def domain(self) -> str:
        return "maxwell_3d"

    @property
    def domain_label(self) -> str:
        return "3D Maxwell Equations (full curl)"

    def compile(self) -> Program:
        c = self._c
        dt = self._dt
        nb = self._n_bits
        bits = (nb, nb, nb)
        dom = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))

        sigma = 0.08

        def init_Ex(x: NDArray, y: NDArray, z: NDArray) -> NDArray:
            r2 = (x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2
            return np.exp(-r2 / (2.0 * sigma ** 2))

        def init_zero(x: NDArray, y: NDArray, z: NDArray) -> NDArray:
            return np.zeros_like(x)

        def invariant_fn(fields: dict) -> float:
            h = fields["Ex"].grid_spacing(0)
            dV = h ** 3
            energy = 0.0
            for name in ("Ex", "Ey", "Ez", "Bx", "By", "Bz"):
                f = fields[name]
                energy += f.inner(f)
            return 0.5 * dV * energy

        cdt = c * dt
        half_cdt = 0.5 * cdt

        # Register map (24 registers):
        # r0-r5:   Ex, Ey, Ez, Bx, By, Bz  (field registers)
        # r6-r11:  scratch for gradients
        # r12-r17: scratch for curl sums
        # r18-r23: scratch for scaled updates
        #
        # Störmer-Verlet (half-kick E / full-kick B / half-kick E)
        # ensures E and B are synchronised at each measurement.

        def _half_kick_E(alpha: float) -> list[Instruction]:
            """E += alpha * curl(B), where alpha = ±c*dt/2."""
            return [
                # curl(B)_x = ∂Bz/∂y − ∂By/∂z
                grad(6, 5, dim=1), grad(7, 4, dim=2),
                sub(12, 6, 7), scale(18, 12, alpha),
                add(0, 0, 18), truncate(0),
                # curl(B)_y = ∂Bx/∂z − ∂Bz/∂x
                grad(8, 3, dim=2), grad(9, 5, dim=0),
                sub(13, 8, 9), scale(19, 13, alpha),
                add(1, 1, 19), truncate(1),
                # curl(B)_z = ∂By/∂x − ∂Bx/∂y
                grad(10, 4, dim=0), grad(11, 3, dim=1),
                sub(14, 10, 11), scale(20, 14, alpha),
                add(2, 2, 20), truncate(2),
                # periodic BC on E
                bc_apply(0, BCKind.PERIODIC),
                bc_apply(1, BCKind.PERIODIC),
                bc_apply(2, BCKind.PERIODIC),
            ]

        def _full_kick_B() -> list[Instruction]:
            """B -= c*dt * curl(E)  (full step)."""
            return [
                # curl(E)_x = ∂Ez/∂y − ∂Ey/∂z
                grad(6, 2, dim=1), grad(7, 1, dim=2),
                sub(15, 6, 7), scale(21, 15, -cdt),
                add(3, 3, 21), truncate(3),
                # curl(E)_y = ∂Ex/∂z − ∂Ez/∂x
                grad(8, 0, dim=2), grad(9, 2, dim=0),
                sub(16, 8, 9), scale(22, 16, -cdt),
                add(4, 4, 22), truncate(4),
                # curl(E)_z = ∂Ey/∂x − ∂Ex/∂y
                grad(10, 1, dim=0), grad(11, 0, dim=1),
                sub(17, 10, 11), scale(23, 17, -cdt),
                add(5, 5, 23), truncate(5),
                # periodic BC on B
                bc_apply(3, BCKind.PERIODIC),
                bc_apply(4, BCKind.PERIODIC),
                bc_apply(5, BCKind.PERIODIC),
            ]

        instructions: list[Instruction] = [
            loop_start(self._n_steps),

            # Load all fields
            load_field(0, "Ex"), load_field(1, "Ey"), load_field(2, "Ez"),
            load_field(3, "Bx"), load_field(4, "By"), load_field(5, "Bz"),

            # ── half-kick E ────────────────────────────────────────
            *_half_kick_E(half_cdt),

            # ── full-kick B ────────────────────────────────────────
            *_full_kick_B(),

            # ── half-kick E ────────────────────────────────────────
            *_half_kick_E(half_cdt),

            # Store and measure
            store_field(0, "Ex"), store_field(1, "Ey"), store_field(2, "Ez"),
            store_field(3, "Bx"), store_field(4, "By"), store_field(5, "Bz"),
            measure(0, "Ex"),

            loop_end(),
        ]

        field_specs = {}
        for i, name in enumerate(("Ex", "Ey", "Ez", "Bx", "By", "Bz")):
            init_name = f"init_{name}"
            field_specs[name] = FieldSpec(
                name=name,
                n_dims=3,
                bits_per_dim=bits,
                bc=BCKind.PERIODIC,
                bc_params={"domain": dom},
                initial_fn=init_name,
                conserved_quantity="em_energy" if name == "Ex" else "",
            )

        metadata = {
            "init_Ex": init_Ex,
            "init_Ey": init_zero,
            "init_Ez": init_zero,
            "init_Bx": init_zero,
            "init_By": init_zero,
            "init_Bz": init_zero,
            "invariant_fn": invariant_fn,
            "invariant": "em_energy",
            "equations": "∂E/∂t = c∇×B, ∂B/∂t = −c∇×E",
        }

        return Program(
            domain=self.domain,
            domain_label=self.domain_label,
            n_registers=24,
            fields=field_specs,
            instructions=instructions,
            dt=self._dt,
            n_steps=self._n_steps,
            params={"c": self._c},
            metadata=metadata,
        )
