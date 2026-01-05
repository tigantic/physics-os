"""
Heat Equation Solver
====================

1D and 2D heat diffusion for thermal analysis.

Applications:
    - Material thermal analysis
    - Composite wall design
    - Steady-state temperature profiles
"""

import numpy as np
from typing import Tuple, Dict, Optional, Callable


class HeatEquation1D:
    """
    1D Heat equation solver.
    
    Equation:
        ∂T/∂t = α·∂²T/∂x²
    
    where α = k/(ρ·c_p) is thermal diffusivity.
    
    Args:
        nx: Number of grid points
        L: Domain length (m)
        alpha: Thermal diffusivity (m²/s)
    """
    
    def __init__(self, nx: int = 100, L: float = 1.0, alpha: float = 1e-5):
        self.nx = nx
        self.L = L
        self.dx = L / (nx - 1)
        self.alpha = alpha
        self.x = np.linspace(0, L, nx)
        
    def step(self, T: np.ndarray, dt: float,
             T_left: Optional[float] = None,
             T_right: Optional[float] = None) -> np.ndarray:
        """
        Single time step with Dirichlet BCs.
        
        Args:
            T: Temperature array
            dt: Time step
            T_left: Left boundary temperature (None = Neumann)
            T_right: Right boundary temperature (None = Neumann)
            
        Returns:
            Updated temperature array
        """
        T_new = T.copy()
        
        # Interior points: central difference
        T_new[1:-1] = T[1:-1] + self.alpha * dt / self.dx**2 * (
            T[2:] - 2*T[1:-1] + T[:-2]
        )
        
        # Boundary conditions
        if T_left is not None:
            T_new[0] = T_left
        else:
            T_new[0] = T_new[1]  # Zero flux
            
        if T_right is not None:
            T_new[-1] = T_right
        else:
            T_new[-1] = T_new[-2]  # Zero flux
            
        return T_new
    
    def steady_state(self, T_left: float, T_right: float) -> np.ndarray:
        """
        Analytical steady-state solution (linear profile).
        """
        return T_left + (T_right - T_left) * self.x / self.L
    
    def run(self, T0: np.ndarray, n_steps: int, dt: float,
            T_left: float = None, T_right: float = None) -> Dict:
        """
        Run heat equation simulation.
        """
        T = T0.copy()
        
        for _ in range(n_steps):
            T = self.step(T, dt, T_left, T_right)
        
        return {
            "final_T": T,
            "x": self.x,
            "max_T": np.max(T),
            "min_T": np.min(T),
            "n_steps": n_steps
        }


class CompositeWall:
    """
    Multi-layer composite wall thermal analysis.
    
    Computes steady-state temperature profile through layered materials.
    
    Args:
        layers: List of (thickness_m, conductivity_W_mK) tuples
        T_hot: Hot side temperature (°C)
        T_cold: Cold side coolant temperature (°C)
        h_conv: Convection coefficient to coolant (W/m²·K)
    """
    
    def __init__(self, layers: list, T_hot: float, T_cold: float, 
                 h_conv: float = 10000.0):
        self.layers = layers  # [(L1, k1), (L2, k2), ...]
        self.T_hot = T_hot
        self.T_cold = T_cold
        self.h_conv = h_conv
        
    def thermal_resistance(self) -> float:
        """
        Total thermal resistance R_total (m²·K/W)
        
        R = Σ(L_i / k_i) + 1/h_conv
        """
        R = sum(L / k for L, k in self.layers)
        R += 1 / self.h_conv  # Convection resistance
        return R
    
    def heat_flux(self) -> float:
        """
        Steady-state heat flux q (W/m²)
        
        q = ΔT / R_total
        """
        return (self.T_hot - self.T_cold) / self.thermal_resistance()
    
    def temperature_profile(self, nx_per_layer: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute temperature profile through wall.
        
        Returns:
            (x, T): Position and temperature arrays
        """
        q = self.heat_flux()
        
        x_list = []
        T_list = []
        
        x_current = 0.0
        T_current = self.T_hot
        
        for L, k in self.layers:
            x_layer = np.linspace(x_current, x_current + L, nx_per_layer)
            # Linear temperature drop: dT = -q·dx/k
            T_layer = T_current - q * (x_layer - x_current) / k
            
            x_list.extend(x_layer)
            T_list.extend(T_layer)
            
            x_current += L
            T_current = T_layer[-1]
        
        return np.array(x_list), np.array(T_list)
    
    def analyze(self) -> Dict:
        """
        Full thermal analysis of composite wall.
        """
        q = self.heat_flux()
        R_total = self.thermal_resistance()
        x, T = self.temperature_profile()
        
        # Temperature at each interface
        interface_temps = [self.T_hot]
        T_current = self.T_hot
        for L, k in self.layers:
            T_current -= q * L / k
            interface_temps.append(T_current)
        
        return {
            "heat_flux_W_m2": q,
            "R_total_m2K_W": R_total,
            "T_surface": self.T_hot,
            "T_coolant_side": interface_temps[-1],
            "interface_temps": interface_temps,
            "x": x,
            "T": T,
            "layers": self.layers
        }
