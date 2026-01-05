"""
Resistive MHD Solver
====================

Magnetohydrodynamics with resistivity for plasma physics.

Applications:
    - Tokamak plasma evolution (STAR-HEART)
    - Magnetic reconnection
    - Solar flares
"""

import numpy as np
from typing import Tuple, Dict, Optional


class ResistiveMHD:
    """
    1D Resistive MHD solver for plasma dynamics.
    
    Equations (simplified 1D slab geometry):
        ∂ρ/∂t + ∇·(ρv) = 0              (continuity)
        ρ(∂v/∂t + v·∇v) = -∇p + J×B     (momentum)
        ∂B/∂t = ∇×(v×B) + η∇²B          (induction with resistivity)
    
    where:
        ρ = mass density
        v = velocity
        p = pressure
        B = magnetic field
        J = current density = ∇×B
        η = resistivity (enables reconnection)
    
    Args:
        nx: Number of grid points
        L: Domain length
        eta: Resistivity (default: 1e-4)
    """
    
    def __init__(self, nx: int = 64, L: float = 1.0, eta: float = 1e-4):
        self.nx = nx
        self.L = L
        self.dx = L / nx
        self.eta = eta
        self.x = np.linspace(0, L, nx)
        
    def initialize_harris_sheet(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Harris current sheet - classic reconnection test case.
        
        B(x) = B₀ tanh((x - L/2) / δ)
        
        This configuration has reversed field across the center,
        which drives magnetic reconnection.
        
        Returns:
            (rho, v, B): Initial density, velocity, magnetic field
        """
        delta = 0.1  # Current sheet width
        rho = np.ones(self.nx)
        v = np.zeros(self.nx)
        B = np.tanh((self.x - self.L/2) / delta)
        return rho, v, B
    
    def initialize_alfven_wave(self, amplitude: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Alfvén wave initial condition.
        
        Returns:
            (rho, v, B): Initial state with sinusoidal perturbation
        """
        rho = np.ones(self.nx)
        v = amplitude * np.sin(2 * np.pi * self.x / self.L)
        B = np.ones(self.nx) + amplitude * np.sin(2 * np.pi * self.x / self.L)
        return rho, v, B
    
    def step(self, rho: np.ndarray, v: np.ndarray, B: np.ndarray, 
             dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Single MHD timestep using finite differences.
        
        Uses:
            - Central differences for spatial derivatives
            - Forward Euler for time stepping
            - Periodic boundary conditions
        
        Args:
            rho: Density array
            v: Velocity array
            B: Magnetic field array
            dt: Time step
            
        Returns:
            (rho_new, v_new, B_new)
        """
        dx = self.dx
        
        # Periodic finite difference operators
        def ddx(f):
            """Central difference: df/dx"""
            return (np.roll(f, -1) - np.roll(f, 1)) / (2 * dx)
        
        def d2dx2(f):
            """Second derivative: d²f/dx²"""
            return (np.roll(f, -1) - 2*f + np.roll(f, 1)) / dx**2
        
        # Pressure (isothermal equation of state: p = ρ)
        p = rho
        
        # Current density: J = dB/dx (in 1D)
        J = ddx(B)
        
        # Lorentz force: J × B (scalar in 1D)
        F_lorentz = J * B
        
        # Time derivatives
        drho_dt = -ddx(rho * v)  # Continuity
        dv_dt = -v * ddx(v) - ddx(p) / rho + F_lorentz / rho  # Momentum
        dB_dt = -ddx(v * B) + self.eta * d2dx2(B)  # Induction
        
        # Forward Euler update
        rho_new = rho + dt * drho_dt
        v_new = v + dt * dv_dt
        B_new = B + dt * dB_dt
        
        # Ensure positivity of density
        rho_new = np.maximum(rho_new, 0.01)
        
        return rho_new, v_new, B_new
    
    def compute_energy(self, rho: np.ndarray, v: np.ndarray, B: np.ndarray) -> Dict:
        """
        Compute MHD energies.
        
        Returns:
            Dictionary with kinetic, thermal, magnetic, and total energy
        """
        E_kinetic = 0.5 * np.sum(rho * v**2) * self.dx
        E_thermal = np.sum(rho) * self.dx  # p = ρ, so E_th ~ ∫ρ dx
        E_magnetic = 0.5 * np.sum(B**2) * self.dx
        
        return {
            "kinetic": E_kinetic,
            "thermal": E_thermal,
            "magnetic": E_magnetic,
            "total": E_kinetic + E_thermal + E_magnetic
        }
    
    def run(self, n_steps: int, dt: float = 1e-4,
            initial_condition: str = "harris") -> Dict:
        """
        Run MHD simulation.
        
        Args:
            n_steps: Number of time steps
            dt: Time step size
            initial_condition: "harris" or "alfven"
            
        Returns:
            Dictionary with final state and diagnostics
        """
        # Initialize
        if initial_condition == "harris":
            rho, v, B = self.initialize_harris_sheet()
        elif initial_condition == "alfven":
            rho, v, B = self.initialize_alfven_wave()
        else:
            raise ValueError(f"Unknown initial condition: {initial_condition}")
        
        # Track field at center for reconnection rate
        B_history = [B.copy()]
        energy_history = [self.compute_energy(rho, v, B)]
        
        # Time evolution
        for _ in range(n_steps):
            rho, v, B = self.step(rho, v, B, dt)
            B_history.append(B.copy())
        
        energy_history.append(self.compute_energy(rho, v, B))
        
        # Reconnection rate: dB/dt at center
        center_idx = self.nx // 2
        dB_dt_center = (B_history[-1][center_idx] - B_history[0][center_idx]) / (n_steps * dt)
        
        return {
            "final_rho": rho,
            "final_v": v,
            "final_B": B,
            "x": self.x,
            "reconnection_rate": abs(dB_dt_center),
            "max_velocity": np.max(np.abs(v)),
            "energy_initial": energy_history[0],
            "energy_final": energy_history[-1],
            "stable": np.all(np.isfinite(B)) and np.all(np.isfinite(v)),
            "n_steps": n_steps,
            "dt": dt
        }


class IdealMHD(ResistiveMHD):
    """
    Ideal MHD (zero resistivity).
    
    In ideal MHD, magnetic field lines are "frozen in" to the plasma.
    No reconnection can occur.
    """
    
    def __init__(self, nx: int = 64, L: float = 1.0):
        super().__init__(nx=nx, L=L, eta=0.0)
