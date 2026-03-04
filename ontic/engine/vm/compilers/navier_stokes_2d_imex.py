"""QTT Physics VM — 2D Navier–Stokes IMEX compiler (CNAB2).

IMEX time integration for 2-D incompressible Navier–Stokes:

    ∂ω/∂t + J(ψ, ω) = ν ∇²ω
    ∇²ψ = -ω
    u = ∂ψ/∂y,  v = -∂ψ/∂x

Vorticity-stream function formulation.  CNAB2 (Crank–Nicolson for
diffusion + Adams–Bashforth 2 for advection) eliminates the diffusive
CFL constraint:

    Explicit Euler:  dt ∝ h²/ν  (diffusion-limited)
    CNAB2:           dt ∝ h/|u|  (advection CFL-limited)

At 4096², this is a ~1000× larger timestep.

CNAB2 scheme
─────────────
Left-hand side (implicit diffusion):
    (I − α∇²) ω^{n+1} = RHS

Right-hand side (explicit advection + CN diffusion source):
    RHS = (I + α∇²) ω^n + dt · AB2(A^n, A^{n−1})

where:
    α = dt·ν/2
    A = −J(ψ, ω)  (advection term)
    AB2(A^n, A^{n-1}) = (3/2)·A^n − (1/2)·A^{n−1}

Bootstrap: A^{−1} = A^0 (degrades to forward Euler for the first step,
second-order Adams–Bashforth thereafter).

The Helmholtz solve (I − α∇²) is SPD with eigenvalues ≥ 1, so CG
converges in O(√κ) ≈ 3–7 iterations for typical parameters.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..ir import (
    BCKind, FieldSpec, Instruction, Program,
    add, bc_apply, copy, grad, hadamard, helmholtz_solve,
    laplace, laplace_solve, load_field, loop_end, loop_start,
    measure, negate, scale, store_field, sub, truncate,
)
from .base import BaseCompiler


def _build_omega_separable_imex(
    ic_type: str,
    ic_n_modes: int,
) -> Any:
    """Build separable factor specification for init_omega.

    Reuses the same IC spec as the explicit compiler.
    """
    if ic_type == "multi_mode":
        terms: list[tuple[list[Any], float]] = []
        for k in range(1, ic_n_modes + 1):
            for m in range(1, ic_n_modes + 1):
                amp = 2.0 / (k * k + m * m)
                kk, mm = k, m
                terms.append((
                    [
                        lambda x, _k=kk: np.sin(2.0 * np.pi * _k * x),
                        lambda y, _m=mm: np.sin(2.0 * np.pi * _m * y),
                    ],
                    amp,
                ))
        return terms

    return (
        [
            lambda x: np.sin(2.0 * np.pi * x),
            lambda y: np.sin(2.0 * np.pi * y),
        ],
        2.0,
    )


class NavierStokes2DImexCompiler(BaseCompiler):
    """Compile 2-D vorticity-stream NS with IMEX (CNAB2) time integration.

    Parameters
    ----------
    n_bits : int
        Bits per spatial dimension.  Grid is 2^n_bits × 2^n_bits.
    n_steps : int
        Number of time steps.
    viscosity : float
        Kinematic viscosity ν.
    dt : float | None
        Time step.  Auto from advective CFL if None.
    cfl : float
        CFL number for automatic dt computation (advective).
    bc_kind : BCKind
        Boundary condition type.  Default: PERIODIC.
    grad_variant : str
        MPO variant for gradient operators.
    lap_variant : str
        MPO variant for Laplacian operators.
    poisson_tol : float | None
        CG tolerance for the Poisson solve (∇²ψ = −ω).
    poisson_max_iters : int | None
        Maximum CG iterations for the Poisson solve.
    helmholtz_tol : float | None
        CG tolerance for the Helmholtz solve ((I − α∇²)ω = RHS).
    helmholtz_max_iters : int | None
        Maximum CG iterations for the Helmholtz solve.
    ic_type : str
        Initial condition type: ``"taylor_green"`` or ``"multi_mode"``.
    ic_n_modes : int
        Number of wavenumbers per dimension for multi_mode IC.
    """

    def __init__(
        self,
        n_bits: int = 9,
        n_steps: int = 100,
        viscosity: float = 0.01,
        dt: float | None = None,
        cfl: float = 0.25,
        bc_kind: BCKind = BCKind.PERIODIC,
        grad_variant: str = "grad_v1",
        lap_variant: str = "lap_v1",
        poisson_tol: float | None = None,
        poisson_max_iters: int | None = None,
        poisson_precond: str | None = None,
        helmholtz_tol: float | None = None,
        helmholtz_max_iters: int | None = None,
        ic_type: str = "taylor_green",
        ic_n_modes: int = 4,
    ) -> None:
        self._n_bits = n_bits
        self._n_steps = n_steps
        self._viscosity = viscosity
        self._bc_kind = bc_kind
        self._grad_variant = grad_variant
        self._lap_variant = lap_variant
        # For IMEX, cap max iterations to avoid noise accumulation.
        # With tol=1e-4 and warm-starting, CG converges in ≤10 iters.
        self._poisson_max_iters = poisson_max_iters if poisson_max_iters is not None else 50
        self._poisson_precond = poisson_precond
        self._helmholtz_tol = helmholtz_tol
        self._helmholtz_max_iters = helmholtz_max_iters if helmholtz_max_iters is not None else 50
        self._ic_type = ic_type
        self._ic_n_modes = ic_n_modes
        self._cfl = cfl

        N = 2 ** n_bits
        h = 1.0 / N

        # ── Automatic dt: advective CFL ──────────────────────────────
        # For Taylor-Green vortex on [0,1]² with ω₀ = 2sin(2πx)sin(2πy):
        #   ψ₀ = sin(2πx)sin(2πy)/(4π²)
        #   u = ∂ψ/∂y = sin(2πx)cos(2πy)/(2π)
        #   |u|_max = 1/(2π) ≈ 0.159
        #
        # CFL: dt = CFL · h / |u|_max
        # At 4096²:  dt ≈ 0.25 · (1/4096) / 0.159 ≈ 3.84e-4
        #
        # Compared to explicit diffusion limit:
        #   dt_diffusion = 0.25 · h² / (2ν) ≈ 7.45e-7  (×516 smaller!)
        if dt is None:
            u_max_estimate = 1.0 / (2.0 * np.pi)
            self._dt = cfl * h / (u_max_estimate + 1e-30)
        else:
            self._dt = dt

        # Helmholtz coefficient: α = dt·ν/2
        self._helmholtz_alpha = self._dt * viscosity / 2.0

        # ── Poisson tolerance (IMEX-specific) ─────────────────────────
        # IMEX takes ~515× larger timesteps than explicit Euler, so
        # the warm-start residual per step is ~dt × ‖dω/dt‖/‖ω‖
        # ≈ 3.6e-4 at 4096².
        #
        # The Poisson operator has κ = O(N²/π²) ≈ 1.7M at 4096².
        # Unpreconditioned CG needs ~4600 iterations per decade of
        # residual reduction — far too many for QTT arithmetic.
        #
        # Setting tol = 1e-3 ensures the warm-start residual
        # (~3.6e-4) is ALREADY BELOW tolerance at most steps,
        # giving 0-1 CG iterations.  Physics justification:
        #
        #   ψ is only used to derive u = curl(ψ).  For ANY ψ,
        #   div(u) = 0 exactly, so total circulation is conserved
        #   regardless of Poisson accuracy.  The tolerance only
        #   affects advection quality (transport fidelity), not
        #   the conservation invariant.
        #
        #   Velocity error: δu/u ~ tol ≈ 1e-3 (0.1%).
        #   Temporal error: O(dt²) ≈ 1.5e-7.
        #   The Poisson error dominates but is acceptable for
        #   the 100-step validation target.
        if poisson_tol is not None:
            self._poisson_tol = poisson_tol
        else:
            self._poisson_tol = 1e-3

        # ── Helmholtz tolerance ───────────────────────────────────────
        # Although (I − α∇²) has lower κ (~636) than the Laplacian
        # (~1.7M at 4096²), at later steps the vorticity develops
        # multi-mode structure from nonlinear advection.  CG then
        # requires >10 iterations where QTT truncation noise
        # accumulates and prevents convergence below ~1e-4.
        #
        # With tol = 1e-3, the warm-start residual (~3.6e-4) is
        # below tolerance at most steps, giving 0-1 CG iterations.
        # Conservation is preserved regardless (structural property
        # of the vorticity-stream formulation, not solver accuracy).
        if helmholtz_tol is None:
            self._helmholtz_tol = 1e-3
        else:
            self._helmholtz_tol = helmholtz_tol

    @property
    def domain(self) -> str:
        return "navier_stokes_2d"

    @property
    def domain_label(self) -> str:
        return "2D Navier–Stokes (vorticity-stream, IMEX-CNAB2)"

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
        alpha = self._helmholtz_alpha  # dt·ν/2

        def init_omega(x: NDArray, y: NDArray) -> NDArray:
            """Initial vorticity (callable fallback)."""
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

        # ── Register allocation ──────────────────────────────────────
        # CNAB2 needs more registers than explicit Euler:
        #
        # r0  = ω (vorticity, evolving)
        # r1  = ψ (stream function)
        # r2  = u = ∂ψ/∂y
        # r3  = v = −∂ψ/∂x
        # r4  = ∂ω/∂x
        # r5  = ∂ω/∂y
        # r6  = u·∂ω/∂x
        # r7  = v·∂ω/∂y
        # r8  = A^n = −(u·∂ω/∂x + v·∂ω/∂y) (current advection)
        # r9  = ∇²ω (for CN diffusion source)
        # r10 = A^{n-1} (previous advection, for AB2)
        # r11 = scratch (Helmholtz RHS assembly)
        # r12 = scratch (AB2 term)
        # r13 = scratch
        n_regs = 14

        # ── Setup phase (before LOOP_START) ──────────────────────────
        # Bootstrap: compute initial advection A^0 from IC fields and
        # store in r10 (the "previous advection" register).
        # This makes the first AB2 step degrade to forward Euler:
        #   AB2(A^0, A^0) = 3/2·A^0 − 1/2·A^0 = A^0
        setup: list[Instruction] = [
            # Load initial fields
            load_field(0, "omega"),
            load_field(1, "psi"),

            # Solve initial Poisson: ∇²ψ = −ω → ψ
            negate(11, 0),
            laplace_solve(1, 11,
                          tol=self._poisson_tol,
                          max_iter=self._poisson_max_iters,
                          precond=self._poisson_precond,
                          operator_variant=lv,
                          nullspace="constant"),

            # Initial velocity
            grad(2, 1, dim=1, operator_variant=gv),  # u = ∂ψ/∂y
            grad(3, 1, dim=0, operator_variant=gv),   # ∂ψ/∂x
            negate(3, 3),                              # v = −∂ψ/∂x

            # Initial vorticity gradients
            grad(4, 0, dim=0, operator_variant=gv),   # ∂ω/∂x
            grad(5, 0, dim=1, operator_variant=gv),   # ∂ω/∂y

            # Initial advection A^0 = −(u·∂ω/∂x + v·∂ω/∂y)
            hadamard(6, 2, 4),
            hadamard(7, 3, 5),
            add(8, 6, 7),
            negate(8, 8),                              # A^0
            truncate(8),

            # Store A^0 → r10 as A^{-1} for the first AB2 step
            copy(10, 8),

            # Store initial fields back
            store_field(0, "omega"),
            store_field(1, "psi"),
        ]

        # ── Time-step loop body ──────────────────────────────────────
        #
        # CNAB2 per step:
        #   1. Load ω, ψ
        #   2. Poisson: ∇²ψ = −ω → ψ
        #   3. Velocity: u = ∂ψ/∂y, v = −∂ψ/∂x
        #   4. Advection: A^n = −J(ψ, ω) = −(u·∂ω/∂x + v·∂ω/∂y)
        #   5. AB2: adv_rhs = dt · (3/2·A^n − 1/2·A^{n-1})
        #   6. CN diffusion source: diff_rhs = (I + α∇²) ω^n
        #   7. Helmholtz RHS: rhs = diff_rhs + adv_rhs
        #   8. Helmholtz solve: (I − α∇²) ω^{n+1} = rhs
        #   9. Copy A^n → A^{n-1} for next step
        #  10. Store ω, ψ, measure

        loop: list[Instruction] = [
            loop_start(self._n_steps),

            load_field(0, "omega"),
            load_field(1, "psi"),

            # ── 2. Poisson: ∇²ψ = −ω ──────────────────────────────
            negate(11, 0),
            laplace_solve(1, 11,
                          tol=self._poisson_tol,
                          max_iter=self._poisson_max_iters,
                          precond=self._poisson_precond,
                          operator_variant=lv,
                          nullspace="constant"),

            # ── 3. Velocity ────────────────────────────────────────
            grad(2, 1, dim=1, operator_variant=gv),   # u = ∂ψ/∂y
            grad(3, 1, dim=0, operator_variant=gv),    # ∂ψ/∂x
            negate(3, 3),                               # v = −∂ψ/∂x

            # ── 4. Vorticity gradients & advection A^n ─────────────
            grad(4, 0, dim=0, operator_variant=gv),    # ∂ω/∂x
            grad(5, 0, dim=1, operator_variant=gv),    # ∂ω/∂y
            hadamard(6, 2, 4),                          # u·∂ω/∂x
            hadamard(7, 3, 5),                          # v·∂ω/∂y
            add(8, 6, 7),                               # u·∂ω/∂x + v·∂ω/∂y
            negate(8, 8),                               # A^n = −(advection)
            truncate(8),

            # ── 5. AB2 extrapolation ───────────────────────────────
            # adv_rhs = dt · (3/2 · A^n − 1/2 · A^{n-1})
            scale(11, 8, 1.5),                          # 3/2 · A^n
            scale(12, 10, 0.5),                         # 1/2 · A^{n-1}
            sub(11, 11, 12),                            # 3/2·A^n − 1/2·A^{n-1}
            scale(11, 11, dt),                          # dt · AB2(advection)
            truncate(11),

            # ── 6. CN diffusion source: (I + α∇²) ω^n ──────────────
            # diff_rhs = ω^n + α · ∇²ω^n
            laplace(9, 0, operator_variant=lv),         # ∇²ω
            scale(9, 9, alpha),                         # α · ∇²ω
            add(12, 0, 9),                              # ω + α · ∇²ω = (I + α∇²)ω
            truncate(12),

            # ── 7. Helmholtz RHS ────────────────────────────────────
            # rhs = (I + α∇²)ω^n + dt · AB2(advection)
            add(13, 12, 11),
            truncate(13),

            # ── 8. Helmholtz solve: (I − α∇²) ω^{n+1} = rhs ───────
            helmholtz_solve(0, 13,
                            alpha=alpha,
                            tol=self._helmholtz_tol,
                            max_iter=self._helmholtz_max_iters,
                            operator_variant=lv),
            truncate(0),
            bc_apply(0, self._bc_kind),

            # ── 9. Save current advection for next AB2 step ─────────
            copy(10, 8),

            # ── 10. Store & measure ─────────────────────────────────
            store_field(0, "omega"),
            store_field(1, "psi"),
            measure(0, "omega"),

            loop_end(),
        ]

        instructions = setup + loop

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

        metadata: dict[str, Any] = {
            "init_omega": init_omega,
            "init_psi": init_psi,
            "init_omega_separable": _build_omega_separable_imex(
                ic_type, ic_n_modes,
            ),
            "invariant_fn": invariant_fn,
            "invariant": "total_circulation",
            "equations": (
                "∂ω/∂t + (u·∇)ω = ν∇²ω, ∇²ψ = −ω  "
                "[IMEX-CNAB2: CN diffusion + AB2 advection]"
            ),
            "time_integration": "IMEX-CNAB2",
            "helmholtz_alpha": alpha,
            "op_variant": "ns2d_vorticity_imex_cnab2",
            "grad_variant": self._grad_variant,
            "lap_variant": self._lap_variant,
            "ic_type": self._ic_type,
        }

        if self._poisson_tol is not None:
            metadata["poisson_tol"] = self._poisson_tol
        if self._poisson_max_iters is not None:
            metadata["poisson_max_iters"] = self._poisson_max_iters
        if self._poisson_precond is not None:
            metadata["poisson_precond"] = self._poisson_precond
        if self._helmholtz_tol is not None:
            metadata["helmholtz_tol"] = self._helmholtz_tol
        if self._helmholtz_max_iters is not None:
            metadata["helmholtz_max_iters"] = self._helmholtz_max_iters
        if self._ic_type == "multi_mode":
            metadata["ic_n_modes"] = self._ic_n_modes

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
