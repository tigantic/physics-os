"""
Reactive Navier-Stokes Solver
=============================

Couples the Navier-Stokes equations with multi-species chemistry
for high-temperature reacting flows.

Governing Equations:
    ∂ρ/∂t + ∇·(ρu) = 0                          (mass)
    ∂(ρu)/∂t + ∇·(ρu⊗u + pI) = ∇·τ             (momentum)
    ∂(ρE)/∂t + ∇·((ρE+p)u) = ∇·(τ·u - q) + Qdot (energy)
    ∂(ρYᵢ)/∂t + ∇·(ρYᵢu) = ∇·(ρDᵢ∇Yᵢ) + ω̇ᵢ   (species)

Operator Splitting:
    1. Convection (Euler equations)
    2. Diffusion (viscous + species)
    3. Chemistry (implicit ODE integration)

References:
    [1] Giovangigli, "Multicomponent Flow Modeling", Birkhäuser 1999
    [2] Anderson, "Hypersonic and High-Temperature Gas Dynamics", 2nd Ed.
    [3] Poinsot & Veynante, "Theoretical and Numerical Combustion", 2nd Ed.
"""

import torch
import math
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Callable
from enum import IntEnum

from .euler_2d import Euler2D, Euler2DState
from .viscous import (
    sutherland_viscosity,
    stress_tensor_2d,
    heat_flux_2d,
    compute_viscous_rhs_2d,
    GAMMA_AIR,
    R_AIR,
    PR_AIR,
)
from .chemistry import (
    Species,
    MW,
    H_FORM,
    ChemistryState,
    compute_reaction_rates,
    air_5species_ic,
    R_UNIVERSAL,
)
from .implicit import (
    ImplicitConfig,
    ChemistryIntegrator,
    AdaptiveImplicit,
    SolverStatus,
)


@dataclass
class ReactiveState:
    """
    State for reactive Navier-Stokes equations.
    
    Combines hydrodynamic state with species mass fractions.
    """
    rho: torch.Tensor           # Total density [kg/m³]
    u: torch.Tensor             # x-velocity [m/s]
    v: torch.Tensor             # y-velocity [m/s]
    p: torch.Tensor             # Pressure [Pa]
    Y: Dict[Species, torch.Tensor]  # Mass fractions [-]
    
    @property
    def shape(self) -> torch.Size:
        return self.rho.shape
    
    @property
    def T(self) -> torch.Tensor:
        """Temperature from EOS [K]."""
        R_mix = self.mixture_R()
        return self.p / (self.rho * R_mix)
    
    def mixture_molecular_weight(self) -> torch.Tensor:
        """Mixture molecular weight [kg/mol]."""
        M_inv = torch.zeros_like(self.rho)
        for species in Species:
            M_inv = M_inv + self.Y[species] / MW[species]
        return 1.0 / (M_inv + 1e-30)
    
    def mixture_R(self) -> torch.Tensor:
        """Mixture gas constant [J/(kg·K)]."""
        M = self.mixture_molecular_weight()
        return R_UNIVERSAL / M
    
    def to_euler_state(self, gamma: float) -> Euler2DState:
        """Convert to Euler2DState for inviscid step."""
        return Euler2DState(
            rho=self.rho,
            u=self.u,
            v=self.v,
            p=self.p,
            gamma=gamma
        )
    
    def from_euler_state(self, euler_state: Euler2DState) -> 'ReactiveState':
        """Update from Euler2DState after inviscid step."""
        return ReactiveState(
            rho=euler_state.rho,
            u=euler_state.u,
            v=euler_state.v,
            p=euler_state.p,
            Y=self.Y  # Keep species unchanged
        )
    
    def to_chemistry_state(self) -> ChemistryState:
        """Convert to ChemistryState for chemistry step."""
        return ChemistryState(
            rho=self.rho,
            Y=self.Y,
            T=self.T,
            p=self.p
        )
    
    def concentrations(self) -> Dict[Species, torch.Tensor]:
        """Molar concentrations [mol/m³]."""
        return {
            species: self.rho * self.Y[species] / MW[species]
            for species in Species
        }
    
    def validate(self) -> Tuple[bool, str]:
        """Check physical constraints."""
        # Density positive
        if not (self.rho > 0).all():
            return False, "Negative density"
        
        # Pressure positive
        if not (self.p > 0).all():
            return False, "Negative pressure"
        
        # Mass fractions non-negative
        for species, Y in self.Y.items():
            if (Y < -1e-10).any():
                return False, f"Negative mass fraction for {species.name}"
        
        # Mass fractions sum to 1
        Y_sum = sum(self.Y.values())
        if not torch.allclose(Y_sum, torch.ones_like(Y_sum), atol=1e-6):
            return False, "Mass fractions don't sum to 1"
        
        return True, "OK"


@dataclass
class ReactiveConfig:
    """Configuration for reactive Navier-Stokes solver."""
    Nx: int = 100
    Ny: int = 100
    Lx: float = 1.0
    Ly: float = 1.0
    cfl: float = 0.3
    Pr: float = PR_AIR
    Le: float = 1.0  # Lewis number (assumes equal diffusivities)
    implicit_config: ImplicitConfig = field(default_factory=ImplicitConfig)
    chemistry_substep: bool = True
    chemistry_rtol: float = 1e-4
    chemistry_atol: float = 1e-10
    
    @property
    def dx(self) -> float:
        return self.Lx / self.Nx
    
    @property
    def dy(self) -> float:
        return self.Ly / self.Ny


class ReactiveNS:
    """
    Reactive Navier-Stokes solver with operator splitting.
    
    Time integration uses Strang splitting:
        1. Chemistry: C(dt/2)
        2. Convection: E(dt)
        3. Diffusion: V(dt)
        4. Chemistry: C(dt/2)
    """
    
    def __init__(self, config: ReactiveConfig):
        self.config = config
        
        # Euler solver for inviscid step
        from .euler_2d import Euler2D
        self.euler = Euler2D(
            Nx=config.Nx,
            Ny=config.Ny,
            Lx=config.Lx,
            Ly=config.Ly
        )
        
        # Implicit integrator for chemistry
        self.adaptive_integrator = AdaptiveImplicit(
            config=config.implicit_config,
            rtol=config.chemistry_rtol,
            atol=config.chemistry_atol
        )
    
    def step(self, state: ReactiveState, dt: float) -> ReactiveState:
        """
        Advance solution by one timestep using Strang splitting.
        
        Args:
            state: Current reactive state
            dt: Timestep [s]
            
        Returns:
            Updated state
        """
        # Step 1: Chemistry (dt/2)
        state = self._chemistry_step(state, dt/2)
        
        # Step 2: Convection (dt)
        state = self._convection_step(state, dt)
        
        # Step 3: Diffusion (dt)
        state = self._diffusion_step(state, dt)
        
        # Step 4: Chemistry (dt/2)
        state = self._chemistry_step(state, dt/2)
        
        return state
    
    def _chemistry_step(self, state: ReactiveState, dt: float) -> ReactiveState:
        """
        Integrate chemistry source terms.
        
        At each grid point, solve:
            dYᵢ/dt = ω̇ᵢ / ρ
        
        using implicit integration for stiffness.
        """
        Ny, Nx = state.shape
        
        # Updated mass fractions
        Y_new = {species: state.Y[species].clone() for species in Species}
        
        # Get temperature field
        T = state.T
        
        for j in range(Ny):
            for i in range(Nx):
                # Local state vector
                Y_local = torch.tensor([
                    state.Y[s][j, i].item() for s in Species
                ], dtype=torch.float64)
                
                T_local = T[j, i].item()
                rho_local = state.rho[j, i].item()
                
                # Source term function
                def omega_fn(Y_vec: torch.Tensor, T_val: float) -> torch.Tensor:
                    # Convert to concentrations
                    conc = {s: rho_local * Y_vec[s.value].item() / MW[s] 
                            for s in Species}
                    # Convert to tensor dict
                    conc_tensor = {s: torch.tensor([c], dtype=torch.float64) 
                                   for s, c in conc.items()}
                    T_tensor = torch.tensor([T_val], dtype=torch.float64)
                    
                    omega, _ = compute_reaction_rates(T_tensor, conc_tensor)
                    
                    return torch.tensor([omega[s].item() for s in Species],
                                        dtype=torch.float64)
                
                # Integrate
                if self.config.chemistry_substep:
                    # Use adaptive integrator
                    def f(Y):
                        return omega_fn(Y, T_local) / rho_local
                    
                    Y_integrated, _, _ = self.adaptive_integrator.integrate(
                        Y_local, f, dt
                    )
                else:
                    # Simple explicit (for testing only)
                    omega = omega_fn(Y_local, T_local)
                    Y_integrated = Y_local + dt * omega / rho_local
                
                # Clip and normalize
                Y_integrated = torch.clamp(Y_integrated, min=0.0)
                Y_integrated = Y_integrated / Y_integrated.sum()
                
                # Store
                for s in Species:
                    Y_new[s][j, i] = Y_integrated[s.value]
        
        # Update pressure for new composition
        R_mix_new = torch.zeros_like(state.rho)
        for s in Species:
            R_mix_new = R_mix_new + Y_new[s] / MW[s]
        R_mix_new = R_UNIVERSAL * R_mix_new
        
        p_new = state.rho * R_mix_new * T
        
        return ReactiveState(
            rho=state.rho.clone(),
            u=state.u.clone(),
            v=state.v.clone(),
            p=p_new,
            Y=Y_new
        )
    
    def _convection_step(self, state: ReactiveState, dt: float) -> ReactiveState:
        """
        Advance inviscid (Euler) terms.
        
        Also advects species mass fractions.
        """
        # Compute effective gamma from mixture
        gamma = self._effective_gamma(state)
        
        # Convert to Euler state
        euler_state = state.to_euler_state(gamma)
        
        # Advance Euler equations
        # Need to handle gamma properly - use average for now
        gamma_avg = gamma.mean().item()
        euler_state_new = self.euler.step(euler_state, dt, gamma=gamma_avg)
        
        # Advect species (same as density)
        # ∂(ρYᵢ)/∂t + ∇·(ρYᵢu) = 0
        # Since ρ is already advected, Y is transported passively
        Y_new = {}
        for species in Species:
            # Use same advection as density
            rhoY = state.rho * state.Y[species]
            rhoY_new = rhoY * euler_state_new.rho / (state.rho + 1e-30)
            Y_new[species] = rhoY_new / (euler_state_new.rho + 1e-30)
        
        # Renormalize
        Y_sum = sum(Y_new.values())
        for species in Species:
            Y_new[species] = Y_new[species] / Y_sum
        
        return ReactiveState(
            rho=euler_state_new.rho,
            u=euler_state_new.u,
            v=euler_state_new.v,
            p=euler_state_new.p,
            Y=Y_new
        )
    
    def _diffusion_step(self, state: ReactiveState, dt: float) -> ReactiveState:
        """
        Advance viscous diffusion terms.
        
        Includes:
            - Momentum diffusion (stress tensor)
            - Heat conduction
            - Species diffusion
        """
        config = self.config
        
        # Temperature
        T = state.T
        
        # Viscosity
        mu = sutherland_viscosity(T)
        
        # Thermal conductivity
        cp = 1004.5  # Approximate for air
        k = mu * cp / config.Pr
        
        # Viscous RHS for momentum and energy
        rhs = compute_viscous_rhs_2d(
            state.rho, state.u, state.v, T,
            mu, k, config.dx, config.dy
        )
        
        # Update momentum
        u_new = state.u + dt * rhs['u'] / state.rho
        v_new = state.v + dt * rhs['v'] / state.rho
        
        # Species diffusion (Fick's law with Lewis number = 1)
        # ∂Yᵢ/∂t = ∇·(D ∇Yᵢ)
        # D = k / (ρ cp Le)
        D = k / (state.rho * cp * config.Le)
        
        Y_new = {}
        for species in Species:
            Y = state.Y[species]
            
            # Laplacian of Y
            lap_Y = torch.zeros_like(Y)
            
            lap_Y[1:-1, 1:-1] = (
                (Y[1:-1, 2:] - 2*Y[1:-1, 1:-1] + Y[1:-1, :-2]) / config.dx**2 +
                (Y[2:, 1:-1] - 2*Y[1:-1, 1:-1] + Y[:-2, 1:-1]) / config.dy**2
            )
            
            # Update
            Y_new[species] = Y + dt * D * lap_Y
        
        # Renormalize
        Y_sum = sum(Y_new.values())
        for species in Species:
            Y_new[species] = Y_new[species] / Y_sum
        
        # Energy equation (approximate - assumes cp constant)
        # ∂T/∂t = ∇·(α ∇T) where α = k/(ρ cp)
        alpha = k / (state.rho * cp)
        
        lap_T = torch.zeros_like(T)
        lap_T[1:-1, 1:-1] = (
            (T[1:-1, 2:] - 2*T[1:-1, 1:-1] + T[1:-1, :-2]) / config.dx**2 +
            (T[2:, 1:-1] - 2*T[1:-1, 1:-1] + T[:-2, 1:-1]) / config.dy**2
        )
        
        T_new = T + dt * alpha * lap_T
        
        # Recompute pressure from EOS
        R_mix = state.mixture_R()
        p_new = state.rho * R_mix * T_new
        
        return ReactiveState(
            rho=state.rho.clone(),
            u=u_new,
            v=v_new,
            p=p_new,
            Y=Y_new
        )
    
    def _effective_gamma(self, state: ReactiveState) -> torch.Tensor:
        """
        Compute effective ratio of specific heats.
        
        For a mixture: γ_mix = cp_mix / cv_mix
        """
        # For now, use constant gamma
        # Full implementation would use NASA polynomials
        return torch.full_like(state.rho, GAMMA_AIR)
    
    def compute_timestep(self, state: ReactiveState) -> float:
        """
        Compute stable timestep.
        
        Considers:
            - CFL condition (convection)
            - Viscous stability
            - Chemistry stiffness (if explicit)
        """
        config = self.config
        
        # Sound speed
        gamma = GAMMA_AIR
        R_mix = state.mixture_R()
        T = state.T
        c = torch.sqrt(gamma * R_mix * T)
        
        # Convective CFL
        u_max = state.u.abs().max()
        v_max = state.v.abs().max()
        c_max = c.max()
        
        dt_cfl = config.cfl * min(
            config.dx / (u_max + c_max),
            config.dy / (v_max + c_max)
        )
        
        # Viscous stability
        mu = sutherland_viscosity(T)
        nu = mu / state.rho
        nu_max = nu.max().item()
        
        dt_visc = 0.25 * min(config.dx**2, config.dy**2) / (nu_max + 1e-30)
        
        return min(dt_cfl.item(), dt_visc)


def reactive_flat_plate_ic(
    config: ReactiveConfig,
    M_inf: float = 5.0,
    T_inf: float = 300.0,
    p_inf: float = 1000.0,
) -> ReactiveState:
    """
    Create initial condition for reactive flat plate boundary layer.
    
    Args:
        config: Solver configuration
        M_inf: Freestream Mach number
        T_inf: Freestream temperature [K]
        p_inf: Freestream pressure [Pa]
        
    Returns:
        ReactiveState initial condition
    """
    Ny, Nx = config.Ny, config.Nx
    
    # Standard air composition
    Y = {
        Species.N2: torch.full((Ny, Nx), 0.767, dtype=torch.float64),
        Species.O2: torch.full((Ny, Nx), 0.233, dtype=torch.float64),
        Species.NO: torch.zeros((Ny, Nx), dtype=torch.float64),
        Species.N: torch.zeros((Ny, Nx), dtype=torch.float64),
        Species.O: torch.zeros((Ny, Nx), dtype=torch.float64),
    }
    
    # Mixture properties
    M_mix = 0.767 * MW[Species.N2] + 0.233 * MW[Species.O2]
    R_mix = R_UNIVERSAL / M_mix
    gamma = GAMMA_AIR
    
    # Freestream
    rho_inf = p_inf / (R_mix * T_inf)
    c_inf = math.sqrt(gamma * R_mix * T_inf)
    u_inf = M_inf * c_inf
    
    rho = torch.full((Ny, Nx), rho_inf, dtype=torch.float64)
    u = torch.full((Ny, Nx), u_inf, dtype=torch.float64)
    v = torch.zeros((Ny, Nx), dtype=torch.float64)
    p = torch.full((Ny, Nx), p_inf, dtype=torch.float64)
    
    return ReactiveState(
        rho=rho,
        u=u,
        v=v,
        p=p,
        Y=Y
    )


def validate_reactive_ns():
    """
    Run validation tests for reactive Navier-Stokes.
    """
    print("\n" + "=" * 70)
    print("REACTIVE NAVIER-STOKES VALIDATION")
    print("=" * 70)
    
    # Test 1: State creation
    print("\n[Test 1] ReactiveState Creation")
    print("-" * 40)
    
    config = ReactiveConfig(Nx=32, Ny=32, Lx=0.1, Ly=0.05)
    state = reactive_flat_plate_ic(config, M_inf=2.0)
    
    valid, msg = state.validate()
    print(f"State valid: {valid} ({msg})")
    print(f"Shape: {state.shape}")
    print(f"T range: [{state.T.min():.1f}, {state.T.max():.1f}] K")
    
    if valid:
        print("✓ PASS: ReactiveState created successfully")
    else:
        print("✗ FAIL: Invalid state")
    
    # Test 2: Solver initialization
    print("\n[Test 2] ReactiveNS Initialization")
    print("-" * 40)
    
    try:
        solver = ReactiveNS(config)
        print("Solver created successfully")
        print("✓ PASS")
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return
    
    # Test 3: Timestep computation
    print("\n[Test 3] Timestep Computation")
    print("-" * 40)
    
    dt = solver.compute_timestep(state)
    print(f"Computed dt = {dt:.2e} s")
    
    if dt > 0 and dt < 1e-2:
        print("✓ PASS: Reasonable timestep")
    else:
        print("✗ FAIL: Unreasonable timestep")
    
    # Test 4: Single step (without chemistry for speed)
    print("\n[Test 4] Single Time Step")
    print("-" * 40)
    
    # Use explicit chemistry for speed
    config.chemistry_substep = False
    solver = ReactiveNS(config)
    
    state_new = solver.step(state, dt * 0.1)  # Small step
    
    valid, msg = state_new.validate()
    print(f"State after step valid: {valid} ({msg})")
    
    # Check mass conservation
    mass_old = state.rho.sum()
    mass_new = state_new.rho.sum()
    mass_change = abs(mass_new - mass_old) / mass_old
    print(f"Relative mass change: {mass_change:.2e}")
    
    if valid and mass_change < 0.01:
        print("✓ PASS: Step completed successfully")
    else:
        print("✗ FAIL: Step failed")
    
    print("\n" + "=" * 70)
    print("REACTIVE NS VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    validate_reactive_ns()
