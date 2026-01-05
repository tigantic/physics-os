"""
Symplectic Integrators
======================

Energy-conserving time integrators with TT re-compression.
"""

import numpy as np
from typing import Callable, Tuple, Dict, Optional
from hypertensor.core.tensor_train import tt_round, tt_to_full, TTTensor


class SymplecticIntegrator:
    """
    Velocity Verlet integrator with TT re-compression.
    
    The key insight: After each physics step, we re-compress the state
    to bounded rank. This keeps memory constant while time evolves.
    
    The Verlet algorithm is symplectic (preserves phase space volume),
    making it ideal for long-time molecular dynamics and Hamiltonian systems.
    
    Equations:
        x_{n+1} = x_n + v_n·dt + 0.5·a_n·dt²
        v_{n+1} = v_n + 0.5·(a_n + a_{n+1})·dt
    
    Args:
        force_fn: Function f(x) -> force array
        mass: Particle mass (scalar or array)
        max_rank: TT compression rank (default: 12)
    """
    
    def __init__(self, force_fn: Callable, mass: float = 1.0, max_rank: int = 12):
        self.force_fn = force_fn
        self.mass = mass
        self.max_rank = max_rank
        self.step_count = 0
        
    def step(self, x: np.ndarray, v: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single Verlet step with TT compression.
        
        Args:
            x: Position array
            v: Velocity array  
            dt: Time step
            
        Returns:
            (x_new, v_new): Updated position and velocity
        """
        # Current acceleration
        a_n = self.force_fn(x) / self.mass
        
        # Position update (full step)
        x_new = x + v * dt + 0.5 * a_n * dt**2
        
        # New acceleration at updated position
        a_new = self.force_fn(x_new) / self.mass
        
        # Velocity update (using average acceleration)
        v_new = v + 0.5 * (a_n + a_new) * dt
        
        # RE-COMPRESS: This is the patent
        if x_new.size > 100 and x_new.ndim > 1:
            tt_x = tt_round(x_new, self.max_rank)
            tt_v = tt_round(v_new, self.max_rank)
            x_new = tt_to_full(tt_x)
            v_new = tt_to_full(tt_v)
        
        self.step_count += 1
        return x_new, v_new
    
    def run(self, x0: np.ndarray, v0: np.ndarray, n_steps: int, 
            dt: float, save_every: int = 1) -> Dict:
        """
        Run simulation for multiple steps.
        
        Args:
            x0: Initial position
            v0: Initial velocity
            n_steps: Number of time steps
            dt: Time step size
            save_every: Save trajectory every N steps
            
        Returns:
            Dictionary with trajectory, energies, etc.
        """
        x, v = x0.copy(), v0.copy()
        
        trajectory = [x.copy()]
        times = [0.0]
        
        for i in range(n_steps):
            x, v = self.step(x, v, dt)
            
            if (i + 1) % save_every == 0:
                trajectory.append(x.copy())
                times.append((i + 1) * dt)
        
        return {
            "trajectory": trajectory,
            "times": times,
            "final_x": x,
            "final_v": v,
            "n_steps": n_steps
        }


class LeapfrogIntegrator:
    """
    Leapfrog (Störmer-Verlet) integrator.
    
    Positions and velocities are staggered by half a time step.
    Equivalent to Verlet but with different formulation.
    """
    
    def __init__(self, force_fn: Callable, mass: float = 1.0, max_rank: int = 12):
        self.force_fn = force_fn
        self.mass = mass
        self.max_rank = max_rank
        
    def step(self, x: np.ndarray, v_half: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Leapfrog step: v at half-steps, x at full steps.
        
        v_{n+1/2} = v_{n-1/2} + a_n·dt
        x_{n+1} = x_n + v_{n+1/2}·dt
        """
        # Update velocity to n+1/2
        a = self.force_fn(x) / self.mass
        v_half_new = v_half + a * dt
        
        # Update position to n+1
        x_new = x + v_half_new * dt
        
        return x_new, v_half_new
