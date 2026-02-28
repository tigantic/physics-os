"""QTT Physics VM — Schrödinger equation compiler.

1-D time-dependent Schrödinger equation (ℏ = m = 1):

    i ∂ψ/∂t = −½ ∂²ψ/∂x² + V(x)ψ

Split into real / imaginary components:

    ∂ψ_re/∂t =  Hψ_im
    ∂ψ_im/∂t = −Hψ_re

where H = −½∇² + V.

Störmer-Verlet (symplectic) integration — preserves ∫|ψ|² dx:

    1. ψ_im  ← ψ_im − (dt/2)·H·ψ_re          (half-kick)
    2. ψ_re  ← ψ_re + dt·H·ψ_im               (drift)
    3. ψ_im  ← ψ_im − (dt/2)·H·ψ_re           (half-kick)

Initial condition: Gaussian wave packet.
Potential: harmonic oscillator V(x) = ½ω²(x − x₀)².
Conserved quantity: total probability ∫|ψ|² dx = 1.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..ir import (
    BCKind, FieldSpec, Instruction, Program,
    add, bc_apply, hadamard, laplace, load_field, loop_end,
    loop_start, measure, negate, scale, store_field, truncate,
)
from .base import BaseCompiler


class SchrodingerCompiler(BaseCompiler):
    """Compile 1-D Schrödinger equation into QTT VM bytecode.

    Parameters
    ----------
    n_bits : int
        Grid resolution.
    n_steps : int
        Number of time steps.
    omega : float
        Harmonic oscillator frequency.
    x0 : float
        Potential center.
    k0 : float
        Initial wave packet momentum.
    dt : float | None
        Time step.
    """

    def __init__(
        self,
        n_bits: int = 8,
        n_steps: int = 100,
        omega: float = 4.0,
        x0: float = 0.5,
        k0: float = 10.0,
        dt: float | None = None,
    ) -> None:
        self._n_bits = n_bits
        self._n_steps = n_steps
        self._omega = omega
        self._x0 = x0
        self._k0 = k0
        N = 2 ** n_bits
        h = 1.0 / N
        if dt is None:
            self._dt = 0.1 * h * h  # small enough for explicit
        else:
            self._dt = dt

    @property
    def domain(self) -> str:
        return "schrodinger"

    @property
    def domain_label(self) -> str:
        return "Schrödinger Equation (1D harmonic oscillator)"

    def compile(self) -> Program:
        dt = self._dt
        omega = self._omega
        x0 = self._x0
        k0 = self._k0
        sigma = 0.05

        def init_psi_re(x: NDArray) -> NDArray:
            env = np.exp(-((x - x0) ** 2) / (2.0 * sigma ** 2))
            return env * np.cos(k0 * x)

        def init_psi_im(x: NDArray) -> NDArray:
            env = np.exp(-((x - x0) ** 2) / (2.0 * sigma ** 2))
            return env * np.sin(k0 * x)

        def init_V(x: NDArray) -> NDArray:
            return 0.5 * omega * omega * (x - x0) ** 2

        def invariant_fn(fields: dict) -> float:
            psi_re = fields["psi_re"]
            psi_im = fields["psi_im"]
            h = psi_re.grid_spacing(0)
            return h * (psi_re.inner(psi_re) + psi_im.inner(psi_im))

        # Register allocation:
        # r0 = ψ_re,  r1 = ψ_im,  r2 = V(x)
        # r3, r4 = Laplacians (scratch)
        # r5, r6 = kinetic terms (-½∇²ψ)
        # r7, r8 = potential terms (V·ψ)
        # r9 = H·ψ combined
        # r10 = scaled update

        # Störmer-Verlet (symplectic): preserves ∫|ψ|² to machine ε.
        # Step 1: half-kick  ψ_im -= (dt/2)·H·ψ_re
        # Step 2: drift      ψ_re += dt·H·ψ_im
        # Step 3: half-kick  ψ_im -= (dt/2)·H·ψ_re   (uses updated ψ_re)

        half_dt = 0.5 * dt

        instructions: list[Instruction] = [
            loop_start(self._n_steps),

            load_field(0, "psi_re"),
            load_field(1, "psi_im"),
            load_field(2, "V"),

            # ── Step 1: half-kick ψ_im ─────────────────────────────
            laplace(3, 0),                          # ∇²ψ_re
            scale(5, 3, -0.5),                      # -½∇²ψ_re
            hadamard(7, 2, 0),                      # V·ψ_re
            add(9, 5, 7),                           # H·ψ_re
            scale(10, 9, -half_dt),                 # -(dt/2)·H·ψ_re
            add(1, 1, 10),                          # ψ_im -= (dt/2)·H·ψ_re
            truncate(1),

            # ── Step 2: full drift ψ_re ────────────────────────────
            laplace(4, 1),                          # ∇²ψ_im (uses updated ψ_im)
            scale(6, 4, -0.5),                      # -½∇²ψ_im
            hadamard(8, 2, 1),                      # V·ψ_im
            add(9, 6, 8),                           # H·ψ_im
            scale(10, 9, dt),                       # dt·H·ψ_im
            add(0, 0, 10),                          # ψ_re += dt·H·ψ_im
            truncate(0),

            # ── Step 3: half-kick ψ_im ─────────────────────────────
            laplace(3, 0),                          # ∇²ψ_re (uses updated ψ_re)
            scale(5, 3, -0.5),                      # -½∇²ψ_re
            hadamard(7, 2, 0),                      # V·ψ_re
            add(9, 5, 7),                           # H·ψ_re
            scale(10, 9, -half_dt),                 # -(dt/2)·H·ψ_re
            add(1, 1, 10),                          # ψ_im -= (dt/2)·H·ψ_re
            truncate(1),

            bc_apply(0, BCKind.DIRICHLET, {"left": 0.0, "right": 0.0}),
            bc_apply(1, BCKind.DIRICHLET, {"left": 0.0, "right": 0.0}),

            store_field(0, "psi_re"),
            store_field(1, "psi_im"),
            measure(0, "psi_re"),
            measure(1, "psi_im"),

            loop_end(),
        ]

        domain_bounds = (0.0, 1.0)
        return Program(
            domain=self.domain,
            domain_label=self.domain_label,
            n_registers=11,
            fields={
                "psi_re": FieldSpec(
                    name="psi_re",
                    n_dims=1,
                    bits_per_dim=(self._n_bits,),
                    bc=BCKind.DIRICHLET,
                    bc_params={"domain": domain_bounds, "left": 0.0, "right": 0.0},
                    initial_fn="init_psi_re",
                    conserved_quantity="probability",
                ),
                "psi_im": FieldSpec(
                    name="psi_im",
                    n_dims=1,
                    bits_per_dim=(self._n_bits,),
                    bc=BCKind.DIRICHLET,
                    bc_params={"domain": domain_bounds, "left": 0.0, "right": 0.0},
                    initial_fn="init_psi_im",
                ),
                "V": FieldSpec(
                    name="V",
                    n_dims=1,
                    bits_per_dim=(self._n_bits,),
                    bc=BCKind.DIRICHLET,
                    bc_params={"domain": domain_bounds},
                    initial_fn="init_V",
                ),
            },
            instructions=instructions,
            dt=self._dt,
            n_steps=self._n_steps,
            params={"omega": self._omega, "k0": self._k0, "x0": self._x0},
            metadata={
                "init_psi_re": init_psi_re,
                "init_psi_im": init_psi_im,
                "init_V": init_V,
                "invariant_fn": invariant_fn,
                "invariant": "probability",
                "equations": "i∂ψ/∂t = -½∇²ψ + V(x)ψ",
            },
        )
