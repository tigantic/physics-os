"""
Langevin Dynamics
=================

Stochastic dynamics for molecular systems at finite temperature.

Applications:
    - Drug binding stability (wiggle test)
    - Protein folding
    - Thermal fluctuations
"""

import numpy as np
from typing import Callable, Tuple, Dict
from hypertensor.core.constants import k_B


class LangevinDynamics:
    """
    Langevin dynamics integrator using BAOAB splitting.
    
    The Langevin equation:
        m·dv/dt = -γ·v - ∇U(x) + √(2γk_BT)·η(t)
    
    where:
        γ = friction coefficient
        U(x) = potential energy
        η(t) = white noise (Gaussian, uncorrelated)
    
    The BAOAB splitting gives excellent sampling properties:
        B: half kick (velocity from force)
        A: half drift (position from velocity)
        O: Ornstein-Uhlenbeck (friction + noise)
        A: half drift
        B: half kick
    
    Args:
        potential_fn: U(x) -> scalar potential energy
        temperature: Temperature in Kelvin (default: 310 K = body temp)
        friction: Damping coefficient γ (default: 1.0)
        mass: Particle mass (default: 1.0)
    """
    
    def __init__(self, potential_fn: Callable, temperature: float = 310.0,
                 friction: float = 1.0, mass: float = 1.0, max_rank: int = 12):
        self.potential = potential_fn
        self.T = temperature
        self.gamma = friction
        self.mass = mass
        self.k_B = k_B
        self.max_rank = max_rank
        
    def force(self, x: np.ndarray) -> np.ndarray:
        """
        Compute force as negative gradient of potential.
        
        Uses central finite differences.
        """
        eps = 1e-6
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (self.potential(x_plus) - self.potential(x_minus)) / (2 * eps)
        return -grad
    
    def step(self, x: np.ndarray, v: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single BAOAB Langevin step.
        
        Args:
            x: Position
            v: Velocity
            dt: Time step
            
        Returns:
            (x_new, v_new)
        """
        # Noise amplitude from fluctuation-dissipation theorem
        sigma = np.sqrt(2 * self.gamma * self.k_B * self.T / self.mass)
        
        # B: Half kick
        v = v + 0.5 * dt * self.force(x) / self.mass
        
        # A: Half drift
        x = x + 0.5 * dt * v
        
        # O: Ornstein-Uhlenbeck (friction + noise)
        c1 = np.exp(-self.gamma * dt)
        c2 = np.sqrt(1 - c1**2) * sigma / self.gamma if self.gamma > 0 else 0
        v = c1 * v + c2 * np.random.randn(*v.shape)
        
        # A: Half drift
        x = x + 0.5 * dt * v
        
        # B: Half kick
        v = v + 0.5 * dt * self.force(x) / self.mass
        
        return x, v
    
    def run(self, x0: np.ndarray, n_steps: int, dt: float = 1e-15,
            v0: np.ndarray = None) -> Dict:
        """
        Run Langevin dynamics simulation.
        
        Args:
            x0: Initial position
            n_steps: Number of time steps
            dt: Time step (default: 1 fs)
            v0: Initial velocity (default: zero)
            
        Returns:
            Dictionary with:
                - final_position: Final coordinates
                - displacement: Distance from start
                - rmsd: Root-mean-square deviation
                - mean_energy: Average potential energy
                - stable: True if RMSD < 2.0 (stays in binding pocket)
        """
        x = x0.copy()
        v = v0 if v0 is not None else np.zeros_like(x)
        
        trajectory = [x.copy()]
        energies = [self.potential(x)]
        
        for _ in range(n_steps):
            x, v = self.step(x, v, dt)
            trajectory.append(x.copy())
            energies.append(self.potential(x))
        
        # Compute statistics
        displacement = np.linalg.norm(trajectory[-1] - trajectory[0])
        rmsd = np.sqrt(np.mean([np.sum((t - x0)**2) for t in trajectory]))
        
        return {
            "final_position": x,
            "final_velocity": v,
            "displacement": displacement,
            "rmsd": rmsd,
            "mean_energy": np.mean(energies),
            "std_energy": np.std(energies),
            "stable": rmsd < 2.0,  # Angstroms - stays in binding pocket
            "n_steps": n_steps,
            "temperature": self.T
        }
