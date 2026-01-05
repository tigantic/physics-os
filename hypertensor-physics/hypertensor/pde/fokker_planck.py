"""
Fokker-Planck Solver
====================

Evolution of probability distributions under drift and diffusion.

Applications:
    - Risk modeling
    - Uncertainty quantification
    - Stochastic dynamics
    - Option pricing (Black-Scholes limit)
"""

import numpy as np
from typing import Callable, Tuple, Dict, Optional


class FokkerPlanck:
    """
    Fokker-Planck equation solver for probability evolution.
    
    The Fokker-Planck equation:
        ∂P/∂t = -∂/∂x[A(x)P] + D·∂²P/∂x²
    
    where:
        P(x,t) = probability density
        A(x) = drift coefficient (deterministic force)
        D = diffusion coefficient (noise strength)
    
    This is the "forward Kolmogorov equation" - it describes how
    probability distributions evolve under stochastic dynamics.
    
    Args:
        nx: Number of grid points
        x_range: Domain (x_min, x_max)
        drift_fn: A(x) function (default: -x for Ornstein-Uhlenbeck)
        diffusion: D coefficient (default: 0.5)
    """
    
    def __init__(self, nx: int = 128, x_range: Tuple[float, float] = (-5, 5),
                 drift_fn: Callable = None, diffusion: float = 0.5):
        self.nx = nx
        self.x_min, self.x_max = x_range
        self.x = np.linspace(self.x_min, self.x_max, nx)
        self.dx = self.x[1] - self.x[0]
        self.D = diffusion
        
        # Default drift: Ornstein-Uhlenbeck (mean-reverting)
        self.drift = drift_fn if drift_fn else lambda x: -x
        
    def initialize_gaussian(self, mean: float = 0, std: float = 1) -> np.ndarray:
        """
        Create normalized Gaussian initial distribution.
        
        Args:
            mean: Center of Gaussian
            std: Standard deviation
            
        Returns:
            Normalized probability density P(x)
        """
        P = np.exp(-0.5 * ((self.x - mean) / std)**2)
        P /= np.trapezoid(P, self.x)  # Normalize
        return P
    
    def initialize_delta(self, x0: float = 0) -> np.ndarray:
        """
        Create approximate delta function at x0.
        
        Args:
            x0: Location of delta function
            
        Returns:
            Narrow Gaussian approximating δ(x - x0)
        """
        return self.initialize_gaussian(mean=x0, std=self.dx * 2)
    
    def step(self, P: np.ndarray, dt: float) -> np.ndarray:
        """
        Single time step using upwind scheme for drift + central for diffusion.
        
        Args:
            P: Current probability distribution
            dt: Time step
            
        Returns:
            Updated probability distribution
        """
        dx = self.dx
        A = self.drift(self.x)
        
        # Upwind scheme for advection (drift term)
        # Split A into positive and negative parts
        A_pos = np.maximum(A, 0)
        A_neg = np.minimum(A, 0)
        
        # Upwind differences
        drift_term = (
            -A_pos * (P - np.roll(P, 1)) / dx
            -A_neg * (np.roll(P, -1) - P) / dx
        )
        
        # Central difference for diffusion
        diff_term = self.D * (np.roll(P, -1) - 2*P + np.roll(P, 1)) / dx**2
        
        # Time update
        P_new = P + dt * (drift_term + diff_term)
        
        # Enforce non-negativity and normalization
        P_new = np.maximum(P_new, 0)
        norm = np.trapezoid(P_new, self.x)
        if norm > 0:
            P_new /= norm
            
        return P_new
    
    def compute_moments(self, P: np.ndarray) -> Dict:
        """
        Compute statistical moments of distribution.
        
        Returns:
            Dictionary with mean, variance, skewness, kurtosis
        """
        # Mean
        mean = np.trapezoid(self.x * P, self.x)
        
        # Variance
        var = np.trapezoid((self.x - mean)**2 * P, self.x)
        std = np.sqrt(max(var, 0))
        
        # Skewness
        if std > 0:
            skew = np.trapezoid((self.x - mean)**3 * P, self.x) / std**3
        else:
            skew = 0.0
        
        # Excess kurtosis
        if std > 0:
            kurt = np.trapezoid((self.x - mean)**4 * P, self.x) / std**4 - 3
        else:
            kurt = 0.0
        
        return {
            "mean": mean,
            "std": std,
            "variance": var,
            "skewness": skew,
            "kurtosis": kurt
        }
    
    def compute_entropy(self, P: np.ndarray) -> float:
        """
        Compute Shannon entropy: S = -∫ P log(P) dx
        """
        P_safe = np.maximum(P, 1e-30)
        return -np.trapezoid(P_safe * np.log(P_safe), self.x)
    
    def run(self, P0: np.ndarray, n_steps: int, dt: float = 0.01,
            save_every: int = 10) -> Dict:
        """
        Evolve probability distribution.
        
        Args:
            P0: Initial distribution
            n_steps: Number of time steps
            dt: Time step size
            save_every: Save distribution every N steps
            
        Returns:
            Dictionary with final state and diagnostics
        """
        P = P0.copy()
        
        # Track evolution
        P_history = [P.copy()]
        entropy_history = [self.compute_entropy(P)]
        moment_history = [self.compute_moments(P)]
        
        for i in range(n_steps):
            P = self.step(P, dt)
            
            if (i + 1) % save_every == 0:
                P_history.append(P.copy())
            
            entropy_history.append(self.compute_entropy(P))
        
        # Final moments
        final_moments = self.compute_moments(P)
        moment_history.append(final_moments)
        
        return {
            "final_P": P,
            "x": self.x,
            "mean": final_moments["mean"],
            "std": final_moments["std"],
            "final_entropy": entropy_history[-1],
            "initial_entropy": entropy_history[0],
            "entropy_change": entropy_history[-1] - entropy_history[0],
            "P_history": P_history,
            "entropy_history": entropy_history,
            "n_steps": n_steps,
            "dt": dt
        }
    
    def steady_state(self) -> np.ndarray:
        """
        Compute analytical steady state for Ornstein-Uhlenbeck.
        
        For drift A(x) = -x and diffusion D:
            P_ss(x) ∝ exp(-x²/(2D))
        """
        P_ss = np.exp(-self.x**2 / (2 * self.D))
        P_ss /= np.trapezoid(P_ss, self.x)
        return P_ss
