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

Wall Model Support (Phase E)
----------------------------
When ``wall_model=True``, the compiler injects Brinkman volume
penalization into the timestep loop:

    ω_new = ω - dt · (1/η) · χ_solid · ω

This enforces no-slip conditions on immersed boundaries via the
penalization coefficient field from the geometry compiler.  The
penalization field is loaded into a dedicated register at the start
of each timestep and applied after the advection-diffusion update.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..ir import (
    BCKind, FieldSpec, Instruction, Program,
    add, bc_apply, grad, hadamard, laplace, laplace_solve,
    load_field, loop_end, loop_start, measure, negate,
    scale, store_field, sub, truncate,
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
    wall_model : bool
        If True, inject Brinkman penalization for immersed boundaries.
        Requires ``penalization_field`` in geometry setup.
    eta_permeability : float
        Brinkman permeability η (only used when ``wall_model=True``).
        Smaller → stronger no-slip enforcement.
    bc_kind : BCKind
        Boundary condition type.  Default: PERIODIC.
    grad_variant : str
        MPO variant for gradient operators.
        ``"grad_v1"`` (2nd order) or ``"grad_v2_high_order"`` (4th order).
    lap_variant : str
        MPO variant for Laplacian operators.
        ``"lap_v1"`` (2nd order) or ``"lap_v2_high_order"`` (4th order).
    op_variant : str
        Algorithmic variant tag for the NS2D formulation.
        Stored in program metadata.
    poisson_tol : float | None
        CG convergence tolerance for the Poisson solver.
        If None, the runtime default is used.
    poisson_max_iters : int | None
        Maximum CG iterations for the Poisson solver.
        If None, the runtime default is used.
    """

    def __init__(
        self,
        n_bits: int = 6,
        n_steps: int = 50,
        viscosity: float = 0.01,
        dt: float | None = None,
        wall_model: bool = False,
        eta_permeability: float = 1e-4,
        bc_kind: BCKind = BCKind.PERIODIC,
        grad_variant: str = "grad_v1",
        lap_variant: str = "lap_v1",
        op_variant: str = "ns2d_vorticity_v1",
        poisson_tol: float | None = None,
        poisson_max_iters: int | None = None,
    ) -> None:
        self._n_bits = n_bits
        self._n_steps = n_steps
        self._viscosity = viscosity
        self._wall_model = wall_model
        self._eta_permeability = eta_permeability
        self._bc_kind = bc_kind
        self._grad_variant = grad_variant
        self._lap_variant = lap_variant
        self._op_variant = op_variant
        self._poisson_tol = poisson_tol
        self._poisson_max_iters = poisson_max_iters
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
        gv = self._grad_variant
        lv = self._lap_variant

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
        # r12 = penalization coefficient (if wall_model)
        # r13 = penalization scratch (if wall_model)

        n_regs = 14 if self._wall_model else 12

        instructions: list[Instruction] = [
            loop_start(self._n_steps),

            load_field(0, "omega"),
            load_field(1, "psi"),

            # Solve Poisson: ∇²ψ = -ω  → ψ = Lap⁻¹(-ω)
            negate(11, 0),                         # r11 = -ω
            laplace_solve(1, 11,                   # r1 = ψ (Poisson solve)
                          tol=self._poisson_tol,
                          max_iter=self._poisson_max_iters),

            # Velocity from stream function
            grad(2, 1, dim=1, operator_variant=gv), # r2 = u = ∂ψ/∂y
            grad(3, 1, dim=0, operator_variant=gv), # r3 = ∂ψ/∂x
            negate(3, 3),                           # r3 = v = -∂ψ/∂x

            # Vorticity gradient
            grad(4, 0, dim=0, operator_variant=gv), # r4 = ∂ω/∂x
            grad(5, 0, dim=1, operator_variant=gv), # r5 = ∂ω/∂y

            # Advection: u·∂ω/∂x + v·∂ω/∂y
            hadamard(6, 2, 4),                     # r6 = u·∂ω/∂x
            hadamard(7, 3, 5),                     # r7 = v·∂ω/∂y
            add(8, 6, 7),                          # r8 = u·∂ω/∂x + v·∂ω/∂y
            negate(8, 8),                          # r8 = -(advection)

            # Diffusion: ν∇²ω
            laplace(9, 0, operator_variant=lv),    # r9 = ∇²ω
            scale(9, 9, nu),                       # r9 = ν∇²ω

            # RHS and time update
            add(10, 8, 9),                         # r10 = -adv + ν∇²ω
            scale(11, 10, dt),                     # r11 = dt * RHS
            add(0, 0, 11),                         # ω += dt * RHS
            truncate(0),
            bc_apply(0, self._bc_kind),
        ]

        # ── Wall model penalization (Phase E) ────────────────────────
        if self._wall_model:
            instructions.extend([
                # Load penalization coefficient: (1/η) · χ_solid
                load_field(12, "penalization_coeff"),
                # r13 = penal_coeff ⊙ ω
                hadamard(13, 12, 0),
                # r13 = dt · penal_coeff ⊙ ω
                scale(13, 13, dt),
                truncate(13),
                # ω = ω - dt · penal_coeff ⊙ ω
                sub(0, 0, 13),
                truncate(0),
            ])

        instructions.extend([
            store_field(0, "omega"),
            store_field(1, "psi"),
            measure(0, "omega"),

            loop_end(),
        ])

        # ── Fields ───────────────────────────────────────────────────
        fields: dict[str, FieldSpec] = {
            "omega": FieldSpec(
                name="omega",
                n_dims=2,
                bits_per_dim=bits,
                bc=self._bc_kind,
                bc_params={"domain": dom},
                initial_fn="init_omega",
                conserved_quantity="total_circulation",
            ),
            "psi": FieldSpec(
                name="psi",
                n_dims=2,
                bits_per_dim=bits,
                bc=self._bc_kind,
                bc_params={"domain": dom},
                initial_fn="init_psi",
            ),
        }

        # Add penalization field spec when wall model is active
        if self._wall_model:
            fields["penalization_coeff"] = FieldSpec(
                name="penalization_coeff",
                n_dims=2,
                bits_per_dim=bits,
                bc=BCKind.DIRICHLET,
                bc_params={"value": 0.0, "domain": dom},
                conserved_quantity=None,
            )

        metadata: dict[str, Any] = {
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
            "invariant_fn": invariant_fn,
            "invariant": "total_circulation",
            "equations": "∂ω/∂t + (u·∇)ω = ν∇²ω, ∇²ψ = −ω",
            "op_variant": self._op_variant,
            "grad_variant": self._grad_variant,
            "lap_variant": self._lap_variant,
        }

        if self._poisson_tol is not None:
            metadata["poisson_tol"] = self._poisson_tol
        if self._poisson_max_iters is not None:
            metadata["poisson_max_iters"] = self._poisson_max_iters

        if self._wall_model:
            metadata["wall_model"] = True
            metadata["eta_permeability"] = self._eta_permeability
            metadata["penalization_beta"] = 1.0 / max(
                self._eta_permeability, 1e-30,
            )

        return Program(
            domain=self.domain,
            domain_label=self.domain_label,
            n_registers=n_regs,
            fields=fields,
            instructions=instructions,
            dt=self._dt,
            n_steps=self._n_steps,
            params={"viscosity": self._viscosity},
            metadata=metadata,
        )
