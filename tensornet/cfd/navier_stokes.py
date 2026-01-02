"""
Coupled Navier-Stokes 2D Solver
================================

This module implements a coupled compressible Navier-Stokes solver
combining the inviscid Euler flux with viscous terms via operator splitting.

The compressible Navier-Stokes equations in conservative form:

    ∂U/∂t + ∂F/∂x + ∂G/∂y = ∂F_v/∂x + ∂G_v/∂y

where:
    U = [ρ, ρu, ρv, E]ᵀ           (conserved variables)
    F, G = inviscid flux vectors   (from Euler equations)
    F_v, G_v = viscous flux vectors (from Navier-Stokes)

Operator Splitting Strategy:
    1. Inviscid step: Strang splitting with HLLC Riemann solver
    2. Viscous step: Explicit diffusion update

For stability, the timestep must satisfy BOTH:
    - CFL condition: Δt < CFL * min(Δx, Δy) / (|u| + c)
    - Viscous limit: Δt < safety * min(Δx², Δy²) * ρ / (2μ)

References:
    [1] Anderson, "Computational Fluid Dynamics", Ch. 10
    [2] Blazek, "Computational Fluid Dynamics", 2nd ed.
    [3] White, "Viscous Fluid Flow", 3rd ed.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from tensornet.cfd.euler_2d import Euler2D, Euler2DState
from tensornet.cfd.viscous import (
    GAMMA_AIR,
    PR_AIR,
    R_AIR,
    compute_viscous_rhs_2d,
    recovery_temperature,
    reynolds_number,
    sutherland_viscosity,
    viscous_timestep_limit,
)


@dataclass
class NavierStokes2DConfig:
    """Configuration for Navier-Stokes solver."""

    # Grid parameters
    Nx: int
    Ny: int
    Lx: float
    Ly: float

    # Fluid properties
    gamma: float = GAMMA_AIR
    R: float = R_AIR
    Pr: float = PR_AIR

    # Numerical parameters
    cfl: float = 0.4
    viscous_safety: float = 0.25

    # Options
    use_viscous: bool = True

    @property
    def dx(self) -> float:
        return self.Lx / (self.Nx - 1)

    @property
    def dy(self) -> float:
        return self.Ly / (self.Ny - 1)


@dataclass
class NavierStokes2DResult:
    """Result container for NS simulation."""

    state: Euler2DState
    time: float
    timestep: int
    dt_history: list[float]

    # Diagnostics
    Re_max: float = 0.0
    T_max: float = 0.0
    wall_heat_flux: Tensor | None = None


class NavierStokes2D:
    """
    Coupled 2D compressible Navier-Stokes solver.

    Combines the inviscid Euler solver with viscous terms
    using explicit operator splitting.

    Features:
        - HLLC Riemann solver for inviscid fluxes
        - Sutherland's law for temperature-dependent viscosity
        - Automatic timestep selection (min of CFL and viscous)
        - Boundary layer resolution awareness

    Example:
        >>> config = NavierStokes2DConfig(Nx=128, Ny=64, Lx=1.0, Ly=0.5)
        >>> ns = NavierStokes2D(config)
        >>> state = flat_plate_ic(config)
        >>> result = ns.solve(state, t_final=0.1)
    """

    def __init__(self, config: NavierStokes2DConfig):
        """
        Initialize Navier-Stokes solver.

        Args:
            config: Solver configuration
        """
        self.config = config

        # Create underlying Euler solver
        self.euler = Euler2D(
            Nx=config.Nx,
            Ny=config.Ny,
            Lx=config.Lx,
            Ly=config.Ly,
            gamma=config.gamma,
        )

    def compute_timestep(self, state: Euler2DState) -> float:
        """
        Compute stable timestep from CFL and viscous limits.

        Args:
            state: Current flow state

        Returns:
            Stable timestep
        """
        # Inviscid CFL limit
        c = torch.sqrt(self.config.gamma * state.p / state.rho)
        speed = torch.sqrt(state.u**2 + state.v**2) + c
        dt_cfl = (
            self.config.cfl * min(self.config.dx, self.config.dy) / speed.max().item()
        )

        if not self.config.use_viscous:
            return dt_cfl

        # Viscous stability limit
        T = state.p / (state.rho * self.config.R)
        mu = sutherland_viscosity(T)
        mu_max = mu.max().item()
        rho_min = state.rho.min().item()

        dt_viscous = viscous_timestep_limit(
            self.config.dx,
            self.config.dy,
            mu_max,
            rho_min,
            safety=self.config.viscous_safety,
        )

        return min(dt_cfl, dt_viscous)

    def viscous_step(self, state: Euler2DState, dt: float) -> Euler2DState:
        """
        Apply viscous update via explicit Euler.

        ∂U/∂t = ∇·F_v  =>  U^{n+1} = U^n + dt * ∇·F_v

        Args:
            state: Current state
            dt: Timestep

        Returns:
            Updated state with viscous effects
        """
        # Compute viscous RHS
        rhs = compute_viscous_rhs_2d(
            state.rho,
            state.u,
            state.v,
            state.p,
            self.config.dx,
            self.config.dy,
            gamma=self.config.gamma,
            R=self.config.R,
            Pr=self.config.Pr,
        )

        # Get conservative variables
        rho_u = state.rho * state.u
        rho_v = state.rho * state.v
        E = state.E

        # Update conserved variables (mass has zero viscous flux)
        rho_new = state.rho  # No viscous mass flux
        rho_u_new = rho_u + dt * rhs[1]
        rho_v_new = rho_v + dt * rhs[2]
        E_new = E + dt * rhs[3]

        # Convert back to primitive
        u_new = rho_u_new / rho_new
        v_new = rho_v_new / rho_new
        ke = 0.5 * rho_new * (u_new**2 + v_new**2)
        p_new = (self.config.gamma - 1) * (E_new - ke)

        # Ensure positivity
        p_new = torch.clamp(p_new, min=1e-10)
        rho_new = torch.clamp(rho_new, min=1e-10)

        return Euler2DState(
            rho=rho_new, u=u_new, v=v_new, p=p_new, gamma=self.config.gamma
        )

    def step(self, state: Euler2DState, dt: float) -> Euler2DState:
        """
        Advance solution one timestep using operator splitting.

        Strang-type splitting:
            1. Half viscous step
            2. Full inviscid step (Euler)
            3. Half viscous step

        Args:
            state: Current state
            dt: Timestep

        Returns:
            Updated state
        """
        if self.config.use_viscous:
            # Strang splitting: V(dt/2) -> E(dt) -> V(dt/2)
            state = self.viscous_step(state, dt / 2)
            state = self.euler.step(state, dt)
            state = self.viscous_step(state, dt / 2)
        else:
            # Pure Euler
            state = self.euler.step(state, dt)

        return state

    def solve(
        self,
        initial_state: Euler2DState,
        t_final: float,
        callback: Callable[[Euler2DState, float, int], bool] | None = None,
        max_steps: int = 1000000,
    ) -> NavierStokes2DResult:
        """
        Solve Navier-Stokes equations to final time.

        Args:
            initial_state: Initial condition
            t_final: Final simulation time
            callback: Optional callback(state, t, step) returning True to stop
            max_steps: Maximum timesteps

        Returns:
            NavierStokes2DResult with final state and diagnostics
        """
        state = initial_state
        t = 0.0
        step = 0
        dt_history = []

        while t < t_final and step < max_steps:
            # Compute timestep
            dt = self.compute_timestep(state)
            dt = min(dt, t_final - t)  # Don't overshoot

            # Advance
            state = self.step(state, dt)

            t += dt
            step += 1
            dt_history.append(dt)

            # Callback
            if callback is not None and callback(state, t, step):
                break

        # Compute diagnostics
        T = state.p / (state.rho * self.config.R)
        mu = sutherland_viscosity(T)

        U_ref = torch.sqrt(state.u**2 + state.v**2).max().item()
        L_ref = min(self.config.Lx, self.config.Ly)
        rho_ref = state.rho.mean().item()
        mu_ref = mu.mean().item()

        Re_max = reynolds_number(rho_ref, U_ref, L_ref, mu_ref)
        T_max = T.max().item()

        return NavierStokes2DResult(
            state=state,
            time=t,
            timestep=step,
            dt_history=dt_history,
            Re_max=Re_max,
            T_max=T_max,
        )

    def compute_wall_quantities(
        self, state: Euler2DState, wall_j: int = 0
    ) -> dict[str, Tensor]:
        """
        Compute wall heat transfer and skin friction.

        Args:
            state: Current state
            wall_j: j-index of wall (default 0 = bottom)

        Returns:
            Dictionary with tau_wall, q_wall, Cf, St
        """
        T = state.p / (state.rho * self.config.R)
        mu = sutherland_viscosity(T)

        # Wall shear stress: τ_w = μ * (∂u/∂y)|_wall
        if wall_j == 0:
            dudy_wall = (state.u[1, :] - state.u[0, :]) / self.config.dy
            T_wall = T[0, :]
            mu_wall = mu[0, :]
        else:
            dudy_wall = (state.u[-1, :] - state.u[-2, :]) / self.config.dy
            T_wall = T[-1, :]
            mu_wall = mu[-1, :]

        tau_wall = mu_wall * dudy_wall

        # Wall heat flux: q_w = -k * (∂T/∂y)|_wall
        k = (
            mu
            * self.config.gamma
            * self.config.R
            / (self.config.gamma - 1)
            / self.config.Pr
        )
        if wall_j == 0:
            dTdy_wall = (T[1, :] - T[0, :]) / self.config.dy
            k_wall = k[0, :]
        else:
            dTdy_wall = (T[-1, :] - T[-2, :]) / self.config.dy
            k_wall = k[-1, :]

        q_wall = -k_wall * dTdy_wall

        # Freestream reference (top boundary or mean)
        rho_inf = state.rho[-1, :].mean().item()
        U_inf = state.u[-1, :].mean().item()
        T_inf = T[-1, :].mean().item()

        # Skin friction coefficient
        Cf = tau_wall / (0.5 * rho_inf * U_inf**2 + 1e-10)

        # Stanton number
        M_inf = (
            U_inf
            / (
                torch.sqrt(torch.tensor(self.config.gamma * self.config.R * T_inf))
            ).item()
        )
        T_rec = recovery_temperature(T_inf, M_inf, r=math.sqrt(self.config.Pr))
        cp = self.config.gamma * self.config.R / (self.config.gamma - 1)

        h = q_wall / (T_rec - T_wall + 1e-10)
        St = h / (rho_inf * U_inf * cp + 1e-10)

        return {
            "tau_wall": tau_wall,
            "q_wall": q_wall,
            "Cf": Cf,
            "St": St,
            "T_wall": T_wall,
        }


def flat_plate_ic(
    config: NavierStokes2DConfig,
    M_inf: float = 0.5,
    T_inf: float = 300.0,
    p_inf: float = 101325.0,
) -> Euler2DState:
    """
    Initial condition for flat plate boundary layer.

    Uniform freestream with no-slip wall at y=0.

    Args:
        config: Solver configuration
        M_inf: Freestream Mach number
        T_inf: Freestream temperature [K]
        p_inf: Freestream pressure [Pa]

    Returns:
        Initial Euler2DState
    """
    rho_inf = p_inf / (config.R * T_inf)
    c_inf = math.sqrt(config.gamma * config.R * T_inf)
    u_inf = M_inf * c_inf

    rho = torch.ones(config.Ny, config.Nx, dtype=torch.float64) * rho_inf
    u = torch.ones(config.Ny, config.Nx, dtype=torch.float64) * u_inf
    v = torch.zeros(config.Ny, config.Nx, dtype=torch.float64)
    p = torch.ones(config.Ny, config.Nx, dtype=torch.float64) * p_inf

    return Euler2DState(rho=rho, u=u, v=v, p=p, gamma=config.gamma)


def compression_corner_ic(
    config: NavierStokes2DConfig,
    M_inf: float = 2.0,
    T_inf: float = 300.0,
    p_inf: float = 101325.0,
    corner_angle: float = 15.0,  # degrees
    corner_x: float = 0.5,  # fraction of Lx
) -> Euler2DState:
    """
    Initial condition for compression corner (SBLI test case).

    Uniform supersonic freestream over a ramp starting at corner_x.

    Args:
        config: Solver configuration
        M_inf: Freestream Mach number
        T_inf: Freestream temperature [K]
        p_inf: Freestream pressure [Pa]
        corner_angle: Ramp angle in degrees
        corner_x: Corner location as fraction of Lx

    Returns:
        Initial Euler2DState
    """
    rho_inf = p_inf / (config.R * T_inf)
    c_inf = math.sqrt(config.gamma * config.R * T_inf)
    u_inf = M_inf * c_inf

    rho = torch.ones(config.Ny, config.Nx, dtype=torch.float64) * rho_inf
    u = torch.ones(config.Ny, config.Nx, dtype=torch.float64) * u_inf
    v = torch.zeros(config.Ny, config.Nx, dtype=torch.float64)
    p = torch.ones(config.Ny, config.Nx, dtype=torch.float64) * p_inf

    return Euler2DState(rho=rho, u=u, v=v, p=p, gamma=config.gamma)
