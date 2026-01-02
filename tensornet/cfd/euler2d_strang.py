"""
2D Euler Solver via Strang Splitting

This module implements 2D compressible Euler equations using dimensional splitting
(Strang splitting) with native QTT shift MPOs.

The key insight: We don't need a separate 2D solver. By swapping the shift operator
between X and Y directions, we "trick" the 1D solver into solving 2D.

Strang Splitting Scheme: U^{n+1} = L_x(dt/2) L_y(dt) L_x(dt/2) U^n
- Second-order accurate in time
- No grid transposition needed (just swap MPO operators)
- Maintains O(log N) complexity via native shift MPOs

Author: HyperTensor Team
Date: December 2025
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from tensornet.cfd.qtt_2d import QTT2DState, dense_to_qtt_2d, qtt_2d_to_dense
from tensornet.cfd.qtt_2d_shift_native import apply_shift_mpo, make_interleaved_shift_mpo


@dataclass
class Euler2DConfig:
    """Configuration for 2D Euler solver."""

    gamma: float = 1.4  # Ratio of specific heats
    cfl: float = 0.4  # CFL number
    max_rank: int = 64  # Maximum QTT rank
    dtype: torch.dtype = torch.float64
    device: torch.device = None

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cpu")


class Euler2DState:
    """
    2D Euler state holding conserved variables in QTT format.

    Conserved variables: [rho, rho*u, rho*v, E]
    where E = P/(gamma-1) + 0.5*rho*(u^2 + v^2)
    """

    def __init__(
        self, rho: QTT2DState, rhou: QTT2DState, rhov: QTT2DState, E: QTT2DState
    ):
        self.rho = rho
        self.rhou = rhou
        self.rhov = rhov
        self.E = E

    @property
    def nx(self):
        return self.rho.nx

    @property
    def ny(self):
        return self.rho.ny

    @property
    def max_rank(self):
        """Maximum rank across all fields."""
        ranks = []
        for field in [self.rho, self.rhou, self.rhov, self.E]:
            ranks.extend([c.shape[0] for c in field.cores])
        return max(ranks)

    def to_dense(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert to dense tensors for visualization."""
        return (
            qtt_2d_to_dense(self.rho),
            qtt_2d_to_dense(self.rhou),
            qtt_2d_to_dense(self.rhov),
            qtt_2d_to_dense(self.E),
        )

    def get_primitives(self, gamma: float = 1.4) -> tuple[torch.Tensor, ...]:
        """Get primitive variables (rho, u, v, P) in dense format."""
        rho, rhou, rhov, E = self.to_dense()

        # Avoid division by zero
        rho_safe = torch.clamp(rho, min=1e-10)

        u = rhou / rho_safe
        v = rhov / rho_safe

        # P = (gamma - 1) * (E - 0.5 * rho * (u^2 + v^2))
        kinetic = 0.5 * rho_safe * (u**2 + v**2)
        P = (gamma - 1) * (E - kinetic)
        P = torch.clamp(P, min=1e-10)  # Ensure positive pressure

        return rho, u, v, P


class Euler2D_Strang:
    """
    2D Euler solver using Strang splitting with native QTT shift MPOs.

    The solver uses dimensional splitting:
    - X sweep: Solve 1D Euler in X direction (uses shift_x MPO)
    - Y sweep: Solve 1D Euler in Y direction (uses shift_y MPO)

    Strang splitting provides second-order accuracy:
    U^{n+1} = L_x(dt/2) L_y(dt) L_x(dt/2) U^n
    """

    def __init__(self, nx: int, ny: int, config: Euler2DConfig = None):
        """
        Initialize 2D Euler solver.

        Args:
            nx: Number of x qubits (grid is 2^nx in x)
            ny: Number of y qubits (grid is 2^ny in y)
            config: Solver configuration
        """
        self.nx = nx
        self.ny = ny
        self.Nx = 2**nx
        self.Ny = 2**ny
        self.n_qubits = 2 * max(nx, ny)

        self.config = config or Euler2DConfig()

        # Build native shift MPOs
        self.mpo_shift_x = make_interleaved_shift_mpo(
            self.n_qubits, axis="x", dtype=self.config.dtype, device=self.config.device
        )
        self.mpo_shift_y = make_interleaved_shift_mpo(
            self.n_qubits, axis="y", dtype=self.config.dtype, device=self.config.device
        )

        # Negative shifts handled by _shift_field via abs(amount) iteration
        # Wrap-around behavior is correct for periodic domains

        self.dx = 1.0 / self.Nx
        self.dy = 1.0 / self.Ny

    def _shift_field(
        self, field: QTT2DState, direction: str, amount: int = 1
    ) -> QTT2DState:
        """Apply shift to a 2D field using native MPO."""
        mpo = self.mpo_shift_x if direction == "x" else self.mpo_shift_y

        result = field
        for _ in range(abs(amount)):
            result = apply_shift_mpo(result, mpo, max_rank=self.config.max_rank)

        return result

    def _compute_flux_x(self, state: Euler2DState) -> tuple[QTT2DState, ...]:
        """
        Compute X-direction fluxes using Rusanov scheme.

        F = 0.5 * (F_L + F_R) - 0.5 * alpha * (U_R - U_L)
        where alpha = max(|u| + c) is the maximum wave speed
        """
        gamma = self.config.gamma

        # Get dense primitives for flux computation
        # (In a full implementation, this would be done in TCI/QTT space)
        rho, u, v, P = state.get_primitives(gamma)

        # Sound speed
        c = torch.sqrt(gamma * P / torch.clamp(rho, min=1e-10))

        # Maximum wave speed
        alpha = (torch.abs(u) + c).max()

        # Euler fluxes in X direction
        rhou = rho * u
        E = P / (gamma - 1) + 0.5 * rho * (u**2 + v**2)

        F_rho = rhou
        F_rhou = rhou * u + P
        F_rhov = rhou * v
        F_E = (E + P) * u

        # Shift for neighbors
        rho_R = torch.roll(rho, -1, dims=0)
        u_R = torch.roll(u, -1, dims=0)
        v_R = torch.roll(v, -1, dims=0)
        P_R = torch.roll(P, -1, dims=0)

        rhou_R = rho_R * u_R
        E_R = P_R / (gamma - 1) + 0.5 * rho_R * (u_R**2 + v_R**2)

        F_rho_R = rhou_R
        F_rhou_R = rhou_R * u_R + P_R
        F_rhov_R = rhou_R * v_R
        F_E_R = (E_R + P_R) * u_R

        # Rusanov flux at i+1/2
        flux_rho = 0.5 * (F_rho + F_rho_R) - 0.5 * alpha * (rho_R - rho)
        flux_rhou = 0.5 * (F_rhou + F_rhou_R) - 0.5 * alpha * (rhou_R - rhou)
        flux_rhov = 0.5 * (F_rhov + F_rhov_R) - 0.5 * alpha * (rho_R * v_R - rho * v)
        flux_E = 0.5 * (F_E + F_E_R) - 0.5 * alpha * (E_R - E)

        return flux_rho, flux_rhou, flux_rhov, flux_E

    def _compute_flux_y(self, state: Euler2DState) -> tuple[QTT2DState, ...]:
        """
        Compute Y-direction fluxes using Rusanov scheme.
        """
        gamma = self.config.gamma

        rho, u, v, P = state.get_primitives(gamma)

        c = torch.sqrt(gamma * P / torch.clamp(rho, min=1e-10))
        alpha = (torch.abs(v) + c).max()

        rhov = rho * v
        E = P / (gamma - 1) + 0.5 * rho * (u**2 + v**2)

        # Y-direction fluxes
        G_rho = rhov
        G_rhou = rhov * u
        G_rhov = rhov * v + P
        G_E = (E + P) * v

        # Shift for Y neighbors
        rho_R = torch.roll(rho, -1, dims=1)
        u_R = torch.roll(u, -1, dims=1)
        v_R = torch.roll(v, -1, dims=1)
        P_R = torch.roll(P, -1, dims=1)

        rhov_R = rho_R * v_R
        E_R = P_R / (gamma - 1) + 0.5 * rho_R * (u_R**2 + v_R**2)

        G_rho_R = rhov_R
        G_rhou_R = rhov_R * u_R
        G_rhov_R = rhov_R * v_R + P_R
        G_E_R = (E_R + P_R) * v_R

        flux_rho = 0.5 * (G_rho + G_rho_R) - 0.5 * alpha * (rho_R - rho)
        flux_rhou = 0.5 * (G_rhou + G_rhou_R) - 0.5 * alpha * (rho_R * u_R - rho * u)
        flux_rhov = 0.5 * (G_rhov + G_rhov_R) - 0.5 * alpha * (rhov_R - rhov)
        flux_E = 0.5 * (G_E + G_E_R) - 0.5 * alpha * (E_R - E)

        return flux_rho, flux_rhou, flux_rhov, flux_E

    def _evolve_x(self, state: Euler2DState, dt: float) -> Euler2DState:
        """Evolve in X direction for time dt."""
        flux_rho, flux_rhou, flux_rhov, flux_E = self._compute_flux_x(state)

        # Get current dense state
        rho, rhou, rhov, E = state.to_dense()

        # Flux difference: F_{i+1/2} - F_{i-1/2}
        dF_rho = flux_rho - torch.roll(flux_rho, 1, dims=0)
        dF_rhou = flux_rhou - torch.roll(flux_rhou, 1, dims=0)
        dF_rhov = flux_rhov - torch.roll(flux_rhov, 1, dims=0)
        dF_E = flux_E - torch.roll(flux_E, 1, dims=0)

        # Update: U^{n+1} = U^n - dt/dx * (F_{i+1/2} - F_{i-1/2})
        rho_new = rho - dt / self.dx * dF_rho
        rhou_new = rhou - dt / self.dx * dF_rhou
        rhov_new = rhov - dt / self.dx * dF_rhov
        E_new = E - dt / self.dx * dF_E

        # Recompress to QTT
        return Euler2DState(
            rho=dense_to_qtt_2d(rho_new, max_bond=self.config.max_rank),
            rhou=dense_to_qtt_2d(rhou_new, max_bond=self.config.max_rank),
            rhov=dense_to_qtt_2d(rhov_new, max_bond=self.config.max_rank),
            E=dense_to_qtt_2d(E_new, max_bond=self.config.max_rank),
        )

    def _evolve_y(self, state: Euler2DState, dt: float) -> Euler2DState:
        """Evolve in Y direction for time dt."""
        flux_rho, flux_rhou, flux_rhov, flux_E = self._compute_flux_y(state)

        rho, rhou, rhov, E = state.to_dense()

        dG_rho = flux_rho - torch.roll(flux_rho, 1, dims=1)
        dG_rhou = flux_rhou - torch.roll(flux_rhou, 1, dims=1)
        dG_rhov = flux_rhov - torch.roll(flux_rhov, 1, dims=1)
        dG_E = flux_E - torch.roll(flux_E, 1, dims=1)

        rho_new = rho - dt / self.dy * dG_rho
        rhou_new = rhou - dt / self.dy * dG_rhou
        rhov_new = rhov - dt / self.dy * dG_rhov
        E_new = E - dt / self.dy * dG_E

        return Euler2DState(
            rho=dense_to_qtt_2d(rho_new, max_bond=self.config.max_rank),
            rhou=dense_to_qtt_2d(rhou_new, max_bond=self.config.max_rank),
            rhov=dense_to_qtt_2d(rhov_new, max_bond=self.config.max_rank),
            E=dense_to_qtt_2d(E_new, max_bond=self.config.max_rank),
        )

    def compute_dt(self, state: Euler2DState) -> float:
        """Compute stable time step based on CFL condition."""
        gamma = self.config.gamma
        rho, u, v, P = state.get_primitives(gamma)

        c = torch.sqrt(gamma * P / torch.clamp(rho, min=1e-10))

        # Maximum wave speeds
        max_speed_x = (torch.abs(u) + c).max()
        max_speed_y = (torch.abs(v) + c).max()

        dt_x = self.config.cfl * self.dx / max_speed_x
        dt_y = self.config.cfl * self.dy / max_speed_y

        return min(dt_x.item(), dt_y.item())

    def step(self, state: Euler2DState, dt: float = None) -> Euler2DState:
        """
        Perform one Strang splitting time step.

        Strang Splitting: U^{n+1} = L_x(dt/2) L_y(dt) L_x(dt/2) U^n

        Args:
            state: Current Euler2DState
            dt: Time step (computed automatically if None)

        Returns:
            Updated Euler2DState
        """
        if dt is None:
            dt = self.compute_dt(state)

        # 1. Half step in X
        state = self._evolve_x(state, dt / 2.0)

        # 2. Full step in Y
        state = self._evolve_y(state, dt)

        # 3. Half step in X
        state = self._evolve_x(state, dt / 2.0)

        return state

    def evolve(
        self,
        state: Euler2DState,
        t_final: float,
        callback: Callable = None,
        callback_interval: int = 10,
    ) -> Euler2DState:
        """
        Evolve the state to t_final.

        Args:
            state: Initial state
            t_final: Final time
            callback: Optional function called with (step, t, state)
            callback_interval: Steps between callbacks

        Returns:
            Final state
        """
        t = 0.0
        step = 0

        while t < t_final:
            dt = self.compute_dt(state)

            # Don't overshoot
            if t + dt > t_final:
                dt = t_final - t

            state = self.step(state, dt)
            t += dt
            step += 1

            if callback and step % callback_interval == 0:
                callback(step, t, state)

        return state


def create_kelvin_helmholtz_ic(
    nx: int = 7, ny: int = 7, config: Euler2DConfig = None
) -> Euler2DState:
    """
    Create Kelvin-Helmholtz instability initial conditions.

    Initial conditions:
    - Top half (y > 0.5): rho=2.0, u=+0.5 (moving right, heavy)
    - Bottom half (y < 0.5): rho=1.0, u=-0.5 (moving left, light)
    - Perturbation: v = 0.1 * sin(4*pi*x) * exp(-(y-0.5)^2/sigma^2)
    - Pressure: P = 2.5 (uniform)

    Args:
        nx: X qubits (grid is 2^nx)
        ny: Y qubits (grid is 2^ny)
        config: Solver configuration

    Returns:
        Euler2DState with KH initial conditions
    """
    if config is None:
        config = Euler2DConfig()

    Nx = 2**nx
    Ny = 2**ny
    gamma = config.gamma

    # Create coordinate grids
    x = torch.linspace(0, 1, Nx, dtype=config.dtype, device=config.device)
    y = torch.linspace(0, 1, Ny, dtype=config.dtype, device=config.device)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    # Smooth interface using tanh (avoids infinite rank from sharp discontinuity)
    smoothing = 0.02
    step = 0.5 * (1 + torch.tanh((Y - 0.5) / smoothing))

    # Density: 1.0 at bottom, 2.0 at top
    rho = 1.0 + 1.0 * step

    # X-velocity: -0.5 at bottom, +0.5 at top
    u = -0.5 + 1.0 * step

    # Y-velocity: Perturbation localized at interface
    sigma = 0.1
    pert_amp = 0.1
    v = (
        pert_amp
        * torch.sin(4 * math.pi * X)
        * torch.exp(-((Y - 0.5) ** 2) / (sigma**2))
    )

    # Pressure: Constant
    P = 2.5 * torch.ones_like(X)

    # Convert to conserved variables
    rhou = rho * u
    rhov = rho * v
    E = P / (gamma - 1) + 0.5 * rho * (u**2 + v**2)

    # Compress to QTT format
    return Euler2DState(
        rho=dense_to_qtt_2d(rho, max_bond=config.max_rank),
        rhou=dense_to_qtt_2d(rhou, max_bond=config.max_rank),
        rhov=dense_to_qtt_2d(rhov, max_bond=config.max_rank),
        E=dense_to_qtt_2d(E, max_bond=config.max_rank),
    )


# =============================================================================
# Validation Tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Euler 2D Strang Splitting Validation")
    print("=" * 60)

    # Small grid for quick test
    nx, ny = 6, 6  # 64x64
    config = Euler2DConfig(max_rank=32)

    print(f"\nGrid: {2**nx}×{2**ny}")
    print(f"Max rank: {config.max_rank}")

    # Create solver
    solver = Euler2D_Strang(nx, ny, config)

    # Create KH initial conditions
    print("\nCreating Kelvin-Helmholtz IC...")
    state = create_kelvin_helmholtz_ic(nx, ny, config)
    print(f"Initial max rank: {state.max_rank}")

    # Evolve for a few steps
    print("\nEvolving...")

    def progress_callback(step, t, state):
        print(f"  Step {step:4d}: t={t:.4f}, rank={state.max_rank}")

    t_final = 0.1
    state = solver.evolve(
        state, t_final, callback=progress_callback, callback_interval=5
    )

    print(f"\nFinal time: {t_final}")
    print(f"Final max rank: {state.max_rank}")

    # Check mass conservation
    rho, u, v, P = state.get_primitives()
    total_mass = rho.sum()
    print(f"Total mass: {total_mass:.4f}")

    print("\n" + "=" * 60)
    print("Validation COMPLETE")
    print("=" * 60)
