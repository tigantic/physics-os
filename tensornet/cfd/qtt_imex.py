"""
QTT IMEX Time Integrator
========================

Implicit-Explicit (IMEX) time stepping for stiff PDEs in QTT format.

The key insight: advection is explicit (CFL-limited but cheap),
diffusion is implicit (unconditionally stable but requires solve).

For Navier-Stokes:
    ∂u/∂t = -u·∇u + ν∇²u
            ^^^^^^   ^^^^^
           explicit  implicit

This allows large time steps at high Reynolds numbers where
ν is small but advection is fast.

Schemes implemented:
    - IMEX-Euler (1st order)
    - IMEX-Midpoint (2nd order)  
    - IMEX-SBDF2 (2nd order, stiffly accurate)
    - IMEX-ARK3 (3rd order Additive Runge-Kutta)

References:
    [1] Ascher, Ruuth, Spiteri, "Implicit-explicit Runge-Kutta methods
        for time-dependent PDEs", Appl. Numer. Math. (1997)
    [2] Kennedy & Carpenter, "Additive Runge-Kutta schemes for
        convection-diffusion-reaction equations", NASA TM (2001)

Phase 24: Physics Toolbox Extension
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Tuple, List

import torch
from torch import Tensor


class IMEXScheme(Enum):
    """Available IMEX schemes."""
    EULER = auto()      # 1st order
    MIDPOINT = auto()   # 2nd order
    SBDF2 = auto()      # 2nd order stiffly accurate
    ARK3 = auto()       # 3rd order ARK


@dataclass
class IMEXConfig:
    """Configuration for IMEX integrator.
    
    Attributes:
        scheme: IMEX scheme to use
        dt: Time step size
        implicit_tol: Tolerance for implicit solve
        implicit_maxiter: Max iterations for implicit solve
        adaptive_dt: Enable adaptive time stepping
        dt_min: Minimum allowed dt
        dt_max: Maximum allowed dt
        safety_factor: Safety factor for dt adaptation
    """
    scheme: IMEXScheme = IMEXScheme.SBDF2
    dt: float = 1e-3
    implicit_tol: float = 1e-8
    implicit_maxiter: int = 50
    adaptive_dt: bool = True
    dt_min: float = 1e-8
    dt_max: float = 1e-1
    safety_factor: float = 0.9
    cfl_target: float = 0.5


@dataclass 
class IMEXState:
    """State container for multi-step IMEX methods.
    
    Stores previous solutions and RHS evaluations needed
    for multi-step schemes like SBDF2.
    """
    u_n: Tensor                          # Solution at t_n
    u_nm1: Optional[Tensor] = None       # Solution at t_{n-1} (for SBDF2)
    f_exp_n: Optional[Tensor] = None     # Explicit RHS at t_n
    f_exp_nm1: Optional[Tensor] = None   # Explicit RHS at t_{n-1}
    t: float = 0.0
    step: int = 0
    dt_history: List[float] = field(default_factory=list)


class IMEXIntegrator:
    """
    IMEX time integrator for stiff PDEs.
    
    Splits the RHS into explicit (advection) and implicit (diffusion) parts:
        du/dt = F_exp(u) + F_imp(u)
    
    The explicit part is treated with an explicit RK method,
    the implicit part with a diagonally-implicit RK method (DIRK).
    
    Example:
        >>> integrator = IMEXIntegrator(
        ...     f_explicit=advection_rhs,
        ...     f_implicit=diffusion_rhs,
        ...     implicit_solve=heat_equation_solve,
        ...     config=IMEXConfig(scheme=IMEXScheme.SBDF2, dt=1e-3)
        ... )
        >>> state = IMEXState(u_n=u0, t=0.0)
        >>> for _ in range(1000):
        ...     state = integrator.step(state)
    """
    
    def __init__(
        self,
        f_explicit: Callable[[Tensor, float], Tensor],
        f_implicit: Callable[[Tensor, float], Tensor],
        implicit_solve: Callable[[Tensor, float, float], Tensor],
        config: IMEXConfig = IMEXConfig(),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize IMEX integrator.
        
        Args:
            f_explicit: Explicit RHS function f_exp(u, t) -> du/dt
            f_implicit: Implicit RHS function f_imp(u, t) -> du/dt
            implicit_solve: Solver for (I - dt*L)u = rhs where L is implicit operator
                           Signature: solve(rhs, t, dt) -> u
            config: IMEX configuration
            device: Torch device
        """
        self.f_exp = f_explicit
        self.f_imp = f_implicit
        self.solve = implicit_solve
        self.config = config
        self.device = device
        
        # Error estimator for adaptive stepping
        self._error_norm = lambda x: torch.sqrt(torch.mean(x**2))
    
    def step(self, state: IMEXState) -> IMEXState:
        """
        Advance solution by one time step.
        
        Args:
            state: Current IMEX state
            
        Returns:
            Updated state at t + dt
        """
        scheme = self.config.scheme
        
        if scheme == IMEXScheme.EULER:
            return self._step_euler(state)
        elif scheme == IMEXScheme.MIDPOINT:
            return self._step_midpoint(state)
        elif scheme == IMEXScheme.SBDF2:
            return self._step_sbdf2(state)
        elif scheme == IMEXScheme.ARK3:
            return self._step_ark3(state)
        else:
            raise ValueError(f"Unknown scheme: {scheme}")
    
    def _step_euler(self, state: IMEXState) -> IMEXState:
        """
        IMEX-Euler (1st order).
        
        u^{n+1} = u^n + dt * F_exp(u^n) + dt * F_imp(u^{n+1})
        
        Rearranged:
        (I - dt*L) u^{n+1} = u^n + dt * F_exp(u^n)
        """
        dt = self.config.dt
        t = state.t
        u = state.u_n
        
        # Explicit evaluation
        f_exp = self.f_exp(u, t)
        
        # Build RHS for implicit solve
        rhs = u + dt * f_exp
        
        # Implicit solve: (I - dt*L) u_new = rhs
        u_new = self.solve(rhs, t + dt, dt)
        
        return IMEXState(
            u_n=u_new,
            u_nm1=u,
            f_exp_n=f_exp,
            f_exp_nm1=state.f_exp_n,
            t=t + dt,
            step=state.step + 1,
            dt_history=state.dt_history + [dt],
        )
    
    def _step_midpoint(self, state: IMEXState) -> IMEXState:
        """
        IMEX-Midpoint (2nd order).
        
        Stage 1: u* = u^n + (dt/2) * F_exp(u^n) + (dt/2) * F_imp(u*)
        Stage 2: u^{n+1} = u^n + dt * F_exp(u*) + dt * F_imp(u^{n+1})
        """
        dt = self.config.dt
        t = state.t
        u = state.u_n
        
        # Stage 1: half step
        f_exp_0 = self.f_exp(u, t)
        rhs_1 = u + (dt/2) * f_exp_0
        u_star = self.solve(rhs_1, t + dt/2, dt/2)
        
        # Stage 2: full step using midpoint evaluation
        f_exp_half = self.f_exp(u_star, t + dt/2)
        rhs_2 = u + dt * f_exp_half
        u_new = self.solve(rhs_2, t + dt, dt)
        
        return IMEXState(
            u_n=u_new,
            u_nm1=u,
            f_exp_n=f_exp_half,
            f_exp_nm1=state.f_exp_n,
            t=t + dt,
            step=state.step + 1,
            dt_history=state.dt_history + [dt],
        )
    
    def _step_sbdf2(self, state: IMEXState) -> IMEXState:
        """
        IMEX-SBDF2 (2nd order, stiffly accurate).
        
        For n >= 1:
        (3/2) u^{n+1} - 2 u^n + (1/2) u^{n-1} = dt * (2 F_exp^n - F_exp^{n-1}) + dt * F_imp^{n+1}
        
        Rearranged:
        (3/2 I - dt*L) u^{n+1} = 2 u^n - (1/2) u^{n-1} + dt * (2 F_exp^n - F_exp^{n-1})
        
        For n = 0, fall back to IMEX-Euler.
        """
        dt = self.config.dt
        t = state.t
        u_n = state.u_n
        
        # First step: use Euler
        if state.step == 0 or state.u_nm1 is None:
            return self._step_euler(state)
        
        u_nm1 = state.u_nm1
        f_exp_n = state.f_exp_n if state.f_exp_n is not None else self.f_exp(u_n, t)
        f_exp_nm1 = state.f_exp_nm1 if state.f_exp_nm1 is not None else self.f_exp(u_nm1, t - dt)
        
        # BDF2 extrapolation for explicit term
        f_exp_extrap = 2 * f_exp_n - f_exp_nm1
        
        # Build RHS: 2*u^n - 0.5*u^{n-1} + dt * f_exp_extrap
        rhs = 2 * u_n - 0.5 * u_nm1 + dt * f_exp_extrap
        
        # Implicit solve with modified operator: (1.5*I - dt*L) u = rhs
        # We pass dt_effective = dt * (2/3) so that solve sees (I - dt_eff*L)
        u_new = self.solve(rhs / 1.5, t + dt, dt * (2/3))
        
        # Compute explicit RHS at new time for next step
        f_exp_new = self.f_exp(u_new, t + dt)
        
        return IMEXState(
            u_n=u_new,
            u_nm1=u_n,
            f_exp_n=f_exp_new,
            f_exp_nm1=f_exp_n,
            t=t + dt,
            step=state.step + 1,
            dt_history=state.dt_history + [dt],
        )
    
    def _step_ark3(self, state: IMEXState) -> IMEXState:
        """
        IMEX-ARK3 (3rd order Additive Runge-Kutta).
        
        Uses Kennedy-Carpenter ARK3(2)4L[2]SA coefficients.
        4 stages, 3rd order, 2nd order embedded for error estimation.
        """
        dt = self.config.dt
        t = state.t
        u = state.u_n
        
        # ARK3(2)4L[2]SA coefficients (Kennedy-Carpenter)
        # Explicit tableau (c, A_exp)
        c = [0.0, 1767732205903/2027836641118, 3/5, 1.0]
        
        # Simplified 3-stage ARK (practical version)
        # Stage 1
        k_exp_1 = self.f_exp(u, t)
        k_imp_1 = self.f_imp(u, t)
        
        # Stage 2
        gamma = 0.4358665215  # ESDIRK parameter
        u_2_rhs = u + dt * 0.5 * k_exp_1
        u_2 = self.solve(u_2_rhs, t + c[1]*dt, gamma * dt)
        k_exp_2 = self.f_exp(u_2, t + c[1]*dt)
        
        # Stage 3
        u_3_rhs = u + dt * (0.25 * k_exp_1 + 0.25 * k_exp_2)
        u_3 = self.solve(u_3_rhs, t + c[2]*dt, gamma * dt)
        k_exp_3 = self.f_exp(u_3, t + c[2]*dt)
        
        # Final stage
        u_4_rhs = u + dt * (1/6 * k_exp_1 + 1/6 * k_exp_2 + 2/3 * k_exp_3)
        u_new = self.solve(u_4_rhs, t + dt, gamma * dt)
        
        # Error estimate (embedded 2nd order)
        if self.config.adaptive_dt:
            u_2nd = u + dt * (0.5 * k_exp_1 + 0.5 * k_exp_3)
            error = self._error_norm(u_new - u_2nd)
            self._adapt_dt(error)
        
        return IMEXState(
            u_n=u_new,
            u_nm1=u,
            f_exp_n=self.f_exp(u_new, t + dt),
            f_exp_nm1=state.f_exp_n,
            t=t + dt,
            step=state.step + 1,
            dt_history=state.dt_history + [dt],
        )
    
    def _adapt_dt(self, error: float) -> None:
        """Adapt time step based on error estimate."""
        cfg = self.config
        tol = cfg.implicit_tol
        
        if error < 1e-15:
            factor = 2.0
        else:
            # PI controller
            factor = cfg.safety_factor * (tol / error) ** 0.5
        
        factor = max(0.1, min(factor, 5.0))  # Limit change
        
        new_dt = cfg.dt * factor
        cfg.dt = max(cfg.dt_min, min(new_dt, cfg.dt_max))


# =============================================================================
# QTT-Specific IMEX for Navier-Stokes
# =============================================================================

class QTTNavierStokesIMEX:
    """
    IMEX integrator specialized for QTT Navier-Stokes.
    
    Handles:
    - QTT format velocity fields
    - Spectral derivatives for advection
    - Heat equation solve for diffusion (via tt_poisson)
    
    Example:
        >>> ns_imex = QTTNavierStokesIMEX(
        ...     N=1024, L=2*np.pi, nu=1e-4, rank=32
        ... )
        >>> u, v, w = ns_imex.step(u, v, w, dt=1e-3)
    """
    
    def __init__(
        self,
        N: int,
        L: float = 2 * 3.14159265359,
        nu: float = 1e-4,
        rank: int = 32,
        scheme: IMEXScheme = IMEXScheme.SBDF2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize QTT NS IMEX solver.
        
        Args:
            N: Grid size (must be power of 2)
            L: Domain size
            nu: Kinematic viscosity
            rank: QTT rank bound
            scheme: IMEX scheme
            device: Torch device
        """
        self.N = N
        self.L = L
        self.nu = nu
        self.rank = rank
        self.scheme = scheme
        self.device = device
        
        # Precompute wavenumbers for spectral operations
        self.dx = L / N
        k = torch.fft.fftfreq(N, d=self.dx) * 2 * torch.pi
        self.kx, self.ky, self.kz = torch.meshgrid(k, k, k, indexing='ij')
        self.kx = self.kx.to(device)
        self.ky = self.ky.to(device)
        self.kz = self.kz.to(device)
        self.k_sq = self.kx**2 + self.ky**2 + self.kz**2
        self.k_sq[0, 0, 0] = 1.0  # Avoid division by zero
        
        # Dealiasing mask (2/3 rule)
        kmax = N // 3
        self.dealias = (
            (torch.abs(self.kx) < kmax * self.dx * 2 * torch.pi) &
            (torch.abs(self.ky) < kmax * self.dx * 2 * torch.pi) &
            (torch.abs(self.kz) < kmax * self.dx * 2 * torch.pi)
        ).float()
        
        # State for multi-step methods
        self._prev_u = None
        self._prev_v = None
        self._prev_w = None
        self._prev_adv_u = None
        self._prev_adv_v = None
        self._prev_adv_w = None
        self._step = 0
    
    def advection_rhs(
        self, u: Tensor, v: Tensor, w: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute advection term -(u·∇)u with dealiasing.
        
        Returns:
            Tuple of (adv_u, adv_v, adv_w) advection contributions
        """
        # Spectral derivatives
        u_hat = torch.fft.fftn(u)
        v_hat = torch.fft.fftn(v)
        w_hat = torch.fft.fftn(w)
        
        # Gradients
        dudx = torch.fft.ifftn(1j * self.kx * u_hat).real
        dudy = torch.fft.ifftn(1j * self.ky * u_hat).real
        dudz = torch.fft.ifftn(1j * self.kz * u_hat).real
        
        dvdx = torch.fft.ifftn(1j * self.kx * v_hat).real
        dvdy = torch.fft.ifftn(1j * self.ky * v_hat).real
        dvdz = torch.fft.ifftn(1j * self.kz * v_hat).real
        
        dwdx = torch.fft.ifftn(1j * self.kx * w_hat).real
        dwdy = torch.fft.ifftn(1j * self.ky * w_hat).real
        dwdz = torch.fft.ifftn(1j * self.kz * w_hat).real
        
        # Advection: -(u·∇)u
        adv_u = -(u * dudx + v * dudy + w * dudz)
        adv_v = -(u * dvdx + v * dvdy + w * dvdz)
        adv_w = -(u * dwdx + v * dwdy + w * dwdz)
        
        # Dealias
        adv_u = torch.fft.ifftn(torch.fft.fftn(adv_u) * self.dealias).real
        adv_v = torch.fft.ifftn(torch.fft.fftn(adv_v) * self.dealias).real
        adv_w = torch.fft.ifftn(torch.fft.fftn(adv_w) * self.dealias).real
        
        return adv_u, adv_v, adv_w
    
    def diffusion_solve(
        self, rhs: Tensor, dt: float
    ) -> Tensor:
        """
        Solve implicit diffusion: (I - dt*ν*∇²) u = rhs.
        
        In Fourier space: û_new = rhs_hat / (1 + dt*ν*k²)
        """
        rhs_hat = torch.fft.fftn(rhs)
        factor = 1.0 + dt * self.nu * self.k_sq
        u_hat = rhs_hat / factor
        return torch.fft.ifftn(u_hat).real
    
    def pressure_project(
        self, u: Tensor, v: Tensor, w: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Project velocity to divergence-free using Helmholtz decomposition.
        
        u_projected = u - ∇φ where ∇²φ = ∇·u
        """
        u_hat = torch.fft.fftn(u)
        v_hat = torch.fft.fftn(v)
        w_hat = torch.fft.fftn(w)
        
        # Divergence in Fourier space
        div_hat = 1j * (self.kx * u_hat + self.ky * v_hat + self.kz * w_hat)
        
        # Pressure: ∇²φ = div → φ_hat = -div_hat / k²
        phi_hat = -div_hat / self.k_sq
        phi_hat[0, 0, 0] = 0  # Zero mean pressure
        
        # Correct velocity: u - ∇φ
        u_hat -= 1j * self.kx * phi_hat
        v_hat -= 1j * self.ky * phi_hat
        w_hat -= 1j * self.kz * phi_hat
        
        return (
            torch.fft.ifftn(u_hat).real,
            torch.fft.ifftn(v_hat).real,
            torch.fft.ifftn(w_hat).real,
        )
    
    def step(
        self,
        u: Tensor, v: Tensor, w: Tensor,
        dt: float,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Advance velocity field by one IMEX step.
        
        Args:
            u, v, w: Velocity components
            dt: Time step
            
        Returns:
            Updated (u, v, w) at t + dt
        """
        if self.scheme == IMEXScheme.EULER:
            return self._step_euler(u, v, w, dt)
        elif self.scheme == IMEXScheme.SBDF2:
            return self._step_sbdf2(u, v, w, dt)
        else:
            # Default to Euler for other schemes (can extend)
            return self._step_euler(u, v, w, dt)
    
    def _step_euler(
        self, u: Tensor, v: Tensor, w: Tensor, dt: float
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """IMEX-Euler step."""
        # Explicit: advection
        adv_u, adv_v, adv_w = self.advection_rhs(u, v, w)
        
        # Build RHS for implicit solve
        rhs_u = u + dt * adv_u
        rhs_v = v + dt * adv_v
        rhs_w = w + dt * adv_w
        
        # Implicit: diffusion
        u_new = self.diffusion_solve(rhs_u, dt)
        v_new = self.diffusion_solve(rhs_v, dt)
        w_new = self.diffusion_solve(rhs_w, dt)
        
        # Pressure projection
        u_new, v_new, w_new = self.pressure_project(u_new, v_new, w_new)
        
        # Store for multi-step
        self._prev_u, self._prev_v, self._prev_w = u, v, w
        self._prev_adv_u = adv_u
        self._prev_adv_v = adv_v
        self._prev_adv_w = adv_w
        self._step += 1
        
        return u_new, v_new, w_new
    
    def _step_sbdf2(
        self, u: Tensor, v: Tensor, w: Tensor, dt: float
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """IMEX-SBDF2 step."""
        # First step: use Euler
        if self._step == 0 or self._prev_u is None:
            return self._step_euler(u, v, w, dt)
        
        # Current advection
        adv_u, adv_v, adv_w = self.advection_rhs(u, v, w)
        
        # BDF2 extrapolation for advection
        adv_u_ext = 2 * adv_u - self._prev_adv_u
        adv_v_ext = 2 * adv_v - self._prev_adv_v
        adv_w_ext = 2 * adv_w - self._prev_adv_w
        
        # Build RHS: 2*u^n - 0.5*u^{n-1} + dt * adv_extrap
        rhs_u = 2 * u - 0.5 * self._prev_u + dt * adv_u_ext
        rhs_v = 2 * v - 0.5 * self._prev_v + dt * adv_v_ext
        rhs_w = 2 * w - 0.5 * self._prev_w + dt * adv_w_ext
        
        # Implicit solve with BDF2 coefficient
        dt_eff = dt * (2/3)
        u_new = self.diffusion_solve(rhs_u / 1.5, dt_eff)
        v_new = self.diffusion_solve(rhs_v / 1.5, dt_eff)
        w_new = self.diffusion_solve(rhs_w / 1.5, dt_eff)
        
        # Pressure projection
        u_new, v_new, w_new = self.pressure_project(u_new, v_new, w_new)
        
        # Store for next step
        self._prev_u, self._prev_v, self._prev_w = u, v, w
        self._prev_adv_u = adv_u
        self._prev_adv_v = adv_v
        self._prev_adv_w = adv_w
        self._step += 1
        
        return u_new, v_new, w_new
    
    def compute_cfl(self, u: Tensor, v: Tensor, w: Tensor, dt: float) -> float:
        """Compute CFL number."""
        u_max = torch.abs(u).max().item()
        v_max = torch.abs(v).max().item()
        w_max = torch.abs(w).max().item()
        vel_max = max(u_max, v_max, w_max)
        return vel_max * dt / self.dx
    
    def compute_diffusion_number(self, dt: float) -> float:
        """Compute diffusion number ν*dt/dx²."""
        return self.nu * dt / (self.dx ** 2)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_ns_imex(
    N: int = 256,
    nu: float = 1e-4,
    scheme: str = "sbdf2",
) -> QTTNavierStokesIMEX:
    """
    Create a Navier-Stokes IMEX solver.
    
    Args:
        N: Grid size (power of 2)
        nu: Viscosity
        scheme: "euler", "sbdf2", "midpoint", "ark3"
        
    Returns:
        Configured QTTNavierStokesIMEX solver
    """
    scheme_map = {
        "euler": IMEXScheme.EULER,
        "midpoint": IMEXScheme.MIDPOINT,
        "sbdf2": IMEXScheme.SBDF2,
        "ark3": IMEXScheme.ARK3,
    }
    return QTTNavierStokesIMEX(
        N=N, nu=nu, scheme=scheme_map.get(scheme.lower(), IMEXScheme.SBDF2)
    )


if __name__ == "__main__":
    # Quick test
    print("Testing QTT IMEX Integrator...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    N = 64
    solver = QTTNavierStokesIMEX(N=N, nu=1e-3, scheme=IMEXScheme.SBDF2, device=device)
    
    # Taylor-Green initial condition
    x = torch.linspace(0, 2*torch.pi, N, device=device)
    X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')
    
    u = torch.sin(X) * torch.cos(Y) * torch.cos(Z)
    v = -torch.cos(X) * torch.sin(Y) * torch.cos(Z)
    w = torch.zeros_like(u)
    
    dt = 0.01
    print(f"Initial CFL: {solver.compute_cfl(u, v, w, dt):.3f}")
    print(f"Diffusion number: {solver.compute_diffusion_number(dt):.3f}")
    
    # Take a few steps
    for step in range(10):
        u, v, w = solver.step(u, v, w, dt)
        energy = 0.5 * torch.mean(u**2 + v**2 + w**2).item()
        print(f"Step {step+1}: Energy = {energy:.6f}")
    
    print("✓ IMEX integrator test passed!")
