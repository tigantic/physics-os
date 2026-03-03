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


def _build_omega_separable(
    ic_type: str,
    ic_n_modes: int,
) -> Any:
    """Build separable factor specification for init_omega.

    Returns either a single ``(factors, scale)`` tuple (original
    Taylor-Green, rank 1) or a list of ``(factors, scale)`` tuples
    (multi-mode, rank = number of terms).

    Multi-mode spectrum:
        ω₀(x, y) = Σ_{k,m=1}^{K} [2 / (k² + m²)] sin(2πkx) sin(2πmy)

    This Kolmogorov-like 1/(k²+m²) decay keeps the vorticity bounded
    while injecting broadband content across wavenumbers 1..K per dim.
    Each term is rank-1 in QTT (product of 1-D functions).  The sum
    of K² terms produces a rank-K² tensor that the runtime truncates
    to the governor's adaptive rank.
    """
    if ic_type == "multi_mode":
        terms: list[tuple[list[Any], float]] = []
        for k in range(1, ic_n_modes + 1):
            for m in range(1, ic_n_modes + 1):
                amp = 2.0 / (k * k + m * m)
                # Create closures with bound k, m values
                kk, mm = k, m  # bind loop vars
                terms.append((
                    [
                        lambda x, _k=kk: np.sin(2.0 * np.pi * _k * x),
                        lambda y, _m=mm: np.sin(2.0 * np.pi * _m * y),
                    ],
                    amp,
                ))
        return terms

    # Default: single Taylor-Green mode (1,1), rank 1
    return (
        [
            lambda x: np.sin(2.0 * np.pi * x),
            lambda y: np.sin(2.0 * np.pi * y),
        ],
        2.0,
    )


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
    poisson_precond : str | None
        Preconditioner for the Poisson solver: ``"none"`` (plain CG)
        or ``"mg"`` (QTT multigrid V-cycle preconditioner).
        If None, the runtime default (``"none"``) is used.
    ic_type : str
        Initial condition type:
        - ``"taylor_green"`` (default): single-mode Taylor–Green vortex
          ω₀ = 2 sin(2πx) sin(2πy).
        - ``"multi_mode"``: superposition of Fourier modes with
          Kolmogorov-like 1/(k²+m²) energy spectrum.  Injects
          broadband content to stress the Poisson solver.
    ic_n_modes : int
        Number of wavenumbers per dimension when ``ic_type="multi_mode"``.
        Total initial modes = ic_n_modes².  Default 4 (→ 16 modes,
        rank ≤ 16 before truncation).
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
        poisson_precond: str | None = None,
        ic_type: str = "taylor_green",
        ic_n_modes: int = 4,
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
        self._poisson_precond = poisson_precond
        self._ic_type = ic_type
        self._ic_n_modes = ic_n_modes
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
        ic_type = self._ic_type
        ic_n_modes = self._ic_n_modes

        def init_omega(x: NDArray, y: NDArray) -> NDArray:
            """Initial vorticity (callable fallback for small grids)."""
            if ic_type == "multi_mode":
                result = np.zeros_like(x)
                K = ic_n_modes
                for k in range(1, K + 1):
                    for m in range(1, K + 1):
                        amp = 2.0 / (k * k + m * m)
                        result += amp * np.sin(
                            2.0 * np.pi * k * x
                        ) * np.sin(2.0 * np.pi * m * y)
                return result
            # Default: Taylor–Green single mode (1,1)
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
                          max_iter=self._poisson_max_iters,
                          precond=self._poisson_precond,
                          operator_variant=lv,
                          nullspace="constant"),

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
            "init_omega_separable": _build_omega_separable(ic_type, ic_n_modes),
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
        if self._poisson_precond is not None:
            metadata["poisson_precond"] = self._poisson_precond

        metadata["ic_type"] = self._ic_type
        if self._ic_type == "multi_mode":
            metadata["ic_n_modes"] = self._ic_n_modes

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
