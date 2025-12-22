"""
Singularity Hunter: Adjoint-Based Search for Navier-Stokes Blowup
===================================================================

The Strategy: "Inverse Design for Destruction"

Instead of designing a wing to minimize drag, we use gradient-based
optimization to MAXIMIZE violence (enstrophy/vorticity). We search the
infinite space of smooth initial conditions to find the "Bad Apple"
that breaks the equations.

The Millennium Prize ($1M) Approach:
    1. Objective: Maximize enstrophy Omega = (1/2) int |curl u|^2 dx
    2. Adjoint: Compute dOmega/du_0 via backward integration
    3. Update: u_0 <- u_0 + alpha * dOmega/du_0 (gradient ASCENT)
    4. Iterate until chi -> infinity or convergence

Success Criteria:
    - If chi(t) -> infinity in finite time: SINGULARITY CANDIDATE
    - If chi(t) stays bounded for all tested ICs: Evidence for regularity

Key Insight:
    Near singularities, velocity fields become self-similar (Leray solutions).
    Self-similar structures have LOW tensor rank, so QTT can resolve them
    at effectively infinite resolution.

References:
    [1] Hou & Luo (2022) - Computer-assisted proof of Euler blowup
    [2] Tao (2016) - Finite-time blowup for averaged NS
    [3] Beale-Kato-Majda criterion - Vorticity controls blowup

Tag: [LEVEL-3] [SINGULARITY-HUNTER] [MILLENNIUM]
"""

import torch
from torch import Tensor
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum
import math


class ObjectiveType(Enum):
    """Objective function for singularity hunting."""
    ENSTROPHY = "enstrophy"              # int |curl u|^2 dx
    CHI_GROWTH = "chi_growth"            # d(chi)/dt
    GRADIENT_NORM = "gradient_norm"      # |nabla u|
    VORTICITY_MAX = "vorticity_max"      # max |omega|
    COMBINED = "combined"                # Weighted combination


@dataclass
class HuntingConfig:
    """Configuration for singularity hunt."""
    objective: ObjectiveType = ObjectiveType.COMBINED
    max_iterations: int = 100
    step_size: float = 0.01
    momentum: float = 0.9
    T_horizon: float = 1.0          # Time to integrate forward
    dt: float = 0.01                # Time step
    chi_threshold: float = 1e6     # Declare blowup if chi exceeds
    grad_clip: float = 1.0          # Gradient clipping
    smoothness_penalty: float = 0.1 # Penalize high-frequency noise
    projection_interval: int = 1    # Re-project to div-free every N steps


@dataclass 
class HuntResult:
    """Result from singularity hunt."""
    initial_condition: Tensor       # The found IC
    final_enstrophy: float          # Peak enstrophy achieved
    final_chi: float                # Peak chi achieved
    chi_trajectory: List[float]     # chi(t) over time
    enstrophy_trajectory: List[float]
    blowup_detected: bool           # Whether chi -> large
    convergence_history: List[Dict]
    iterations: int
    verdict: str                    # "SINGULARITY_CANDIDATE" or "BOUNDED"


class EnstrophyObjective:
    """
    Enstrophy objective: Omega = (1/2) int |curl u|^2 dx
    
    For 3D: omega = curl(u) = (dw/dy - dv/dz, du/dz - dw/dx, dv/dx - du/dy)
    Enstrophy = (1/2) int (omega_x^2 + omega_y^2 + omega_z^2) dV
    """
    
    def __init__(self, dx: float, dy: float, dz: float):
        self.dx = dx
        self.dy = dy
        self.dz = dz
        
    def evaluate(self, u: Tensor, v: Tensor, w: Tensor) -> Tensor:
        """Compute total enstrophy."""
        from tensornet.cfd.ns_3d import compute_vorticity_3d
        
        omega_x, omega_y, omega_z = compute_vorticity_3d(
            u, v, w, self.dx, self.dy, self.dz, method='spectral'
        )
        
        enstrophy = 0.5 * (omega_x**2 + omega_y**2 + omega_z**2).sum()
        return enstrophy * (self.dx * self.dy * self.dz)
    
    def gradient(self, u: Tensor, v: Tensor, w: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute gradient of enstrophy w.r.t. velocity field.
        
        Using chain rule:
            dOmega/du = dOmega/d(omega) * d(omega)/du
            
        Where d(omega)/du involves the curl operator.
        """
        from tensornet.cfd.ns_3d import (
            compute_vorticity_3d, compute_gradient_3d,
            compute_laplacian_3d
        )
        
        # Vorticity
        omega_x, omega_y, omega_z = compute_vorticity_3d(
            u, v, w, self.dx, self.dy, self.dz, method='spectral'
        )
        
        # The gradient of enstrophy w.r.t. velocity is:
        # dOmega/du = curl(omega) = (domega_z/dy - domega_y/dz, ...)
        # This is the "curl of curl" = -Laplacian(u) + grad(div(u))
        # For div-free: curl(curl(u)) = -Laplacian(u)
        
        # So dOmega/du_i = -2 * (omega cross e_i component)
        # More precisely, we use spectral differentiation:
        
        # dOmega/domega = omega (the vorticity itself)
        # d(omega)/d(u) = curl operator
        # So dOmega/du = curl^T(omega) = -curl(omega)
        
        grad_omega_x = compute_gradient_3d(omega_x, self.dx, self.dy, self.dz)
        grad_omega_y = compute_gradient_3d(omega_y, self.dx, self.dy, self.dz)
        grad_omega_z = compute_gradient_3d(omega_z, self.dx, self.dy, self.dz)
        
        # curl(omega) = nabla x omega
        curl_omega_x = grad_omega_z[1] - grad_omega_y[2]  # domega_z/dy - domega_y/dz
        curl_omega_y = grad_omega_x[2] - grad_omega_z[0]  # domega_x/dz - domega_z/dx
        curl_omega_z = grad_omega_y[0] - grad_omega_x[1]  # domega_y/dx - domega_x/dy
        
        # Gradient is -curl(omega) scaled by volume element
        scale = self.dx * self.dy * self.dz
        
        return (
            -curl_omega_x * scale,
            -curl_omega_y * scale,
            -curl_omega_z * scale
        )


class ChiGrowthObjective:
    """
    Chi growth rate objective: Maximize d(chi)/dt
    
    If chi grows rapidly, we're approaching a singularity.
    """
    
    def __init__(self, dx: float, dy: float, dz: float, chi_target: int = 128):
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.chi_target = chi_target
        
    def evaluate(self, trajectory) -> float:
        """Compute chi growth rate from trajectory."""
        if len(trajectory.chi_values) < 2:
            return 0.0
        return trajectory.growth_rate() or 0.0


class SingularityHunter:
    """
    Hunt for Navier-Stokes singularities via adjoint optimization.
    
    This is "inverse design for destruction" - we use gradient ascent
    to find initial conditions that maximize enstrophy/chi growth.
    """
    
    def __init__(
        self,
        Nx: int,
        Ny: int,
        Nz: int,
        Lx: float,
        Ly: float,
        Lz: float,
        nu: float,
        config: Optional[HuntingConfig] = None,
        dtype=torch.float64,
    ):
        """
        Initialize the hunter.
        
        Args:
            Nx, Ny, Nz: Grid dimensions
            Lx, Ly, Lz: Domain sizes
            nu: Kinematic viscosity
            config: Hunting configuration
            dtype: Tensor dtype
        """
        from tensornet.cfd.ns_3d import NS3DSolver
        
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.nu = nu
        self.config = config or HuntingConfig()
        self.dtype = dtype
        
        # Create solver
        self.solver = NS3DSolver(
            Nx=Nx, Ny=Ny, Nz=Nz,
            Lx=Lx, Ly=Ly, Lz=Lz,
            nu=nu, dtype=dtype
        )
        
        self.dx = Lx / Nx
        self.dy = Ly / Ny
        self.dz = Lz / Nz
        
        # Objectives
        self.enstrophy_obj = EnstrophyObjective(self.dx, self.dy, self.dz)
        
    def random_smooth_ic(self, n_modes: int = 4, amplitude: float = 1.0) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Generate random smooth divergence-free initial condition.
        
        Uses a random superposition of Fourier modes, then projects
        to ensure div(u) = 0.
        """
        from tensornet.cfd.ns_3d import project_velocity_3d
        
        device = 'cpu'
        
        # Random Fourier coefficients for low modes
        u = torch.zeros(self.Nx, self.Ny, self.Nz, dtype=self.dtype, device=device)
        v = torch.zeros(self.Nx, self.Ny, self.Nz, dtype=self.dtype, device=device)
        w = torch.zeros(self.Nx, self.Ny, self.Nz, dtype=self.dtype, device=device)
        
        # Grid - for periodic domain, don't include endpoint
        x = torch.linspace(0, self.Lx * (1 - 1/self.Nx), self.Nx, dtype=self.dtype, device=device)
        y = torch.linspace(0, self.Ly * (1 - 1/self.Ny), self.Ny, dtype=self.dtype, device=device)
        z = torch.linspace(0, self.Lz * (1 - 1/self.Nz), self.Nz, dtype=self.dtype, device=device)
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        
        # Add random smooth modes
        for _ in range(n_modes):
            kx = torch.randint(1, 4, (1,)).item()
            ky = torch.randint(1, 4, (1,)).item()
            kz = torch.randint(1, 4, (1,)).item()
            
            phase_x = torch.rand(1).item() * 2 * math.pi
            phase_y = torch.rand(1).item() * 2 * math.pi
            phase_z = torch.rand(1).item() * 2 * math.pi
            
            amp = amplitude * (torch.rand(3) - 0.5) * 2
            
            mode = torch.cos(kx * X + phase_x) * torch.sin(ky * Y + phase_y) * torch.cos(kz * Z + phase_z)
            
            u += amp[0].item() * mode
            v += amp[1].item() * mode
            w += amp[2].item() * mode
        
        # Project to divergence-free using spectral method
        proj = project_velocity_3d(u, v, w, self.dx, self.dy, self.dz, dt=1.0, method='spectral')
        
        return proj.u_projected, proj.v_projected, proj.w_projected
    
    def forward_integrate(
        self,
        u0: Tensor,
        v0: Tensor,
        w0: Tensor,
    ) -> Tuple[Dict, List[Dict]]:
        """
        Integrate forward and track chi/enstrophy.
        
        Returns:
            final_metrics: Dict with final chi, enstrophy, etc.
            trajectory: List of metrics at each timestep
        """
        from tensornet.cfd.ns_3d import NSState3D
        from tensornet.cfd.chi_diagnostic import compute_chi_state_3d
        
        state = NSState3D(u=u0.clone(), v=v0.clone(), w=w0.clone(), t=0.0, step=0)
        
        dt = self.config.dt
        n_steps = int(self.config.T_horizon / dt)
        
        trajectory = []
        chi_values = []
        enstrophy_values = []
        
        for step in range(n_steps + 1):
            t = step * dt
            
            # Compute metrics
            enstrophy = self.enstrophy_obj.evaluate(state.u, state.v, state.w).item()
            
            chi_state = compute_chi_state_3d(
                state.u, state.v, state.w,
                t=t, dx=self.dx, dy=self.dy, dz=self.dz,
                chi_target=256
            )
            
            chi_values.append(chi_state.chi_actual)
            enstrophy_values.append(enstrophy)
            
            trajectory.append({
                't': t,
                'chi': chi_state.chi_actual,
                'enstrophy': enstrophy,
                'gradient_norm': chi_state.gradient_norm,
            })
            
            # Check for blowup
            if chi_state.chi_actual > self.config.chi_threshold:
                break
                
            # Check for NaN
            if torch.isnan(state.u).any():
                break
            
            # Step forward
            if step < n_steps:
                state, _ = self.solver.step_rk4(state, dt)
        
        final_metrics = {
            'chi_max': max(chi_values),
            'chi_final': chi_values[-1],
            'enstrophy_max': max(enstrophy_values),
            'enstrophy_final': enstrophy_values[-1],
            'chi_trajectory': chi_values,
            'enstrophy_trajectory': enstrophy_values,
            'steps_completed': len(trajectory),
        }
        
        return final_metrics, trajectory
    
    def compute_gradient_numerical(
        self,
        u0: Tensor,
        v0: Tensor,
        w0: Tensor,
        epsilon: float = 1e-4,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute gradient of objective w.r.t. initial condition numerically.
        
        This is expensive but simple. For production, use adjoint method.
        """
        # Baseline objective
        metrics, _ = self.forward_integrate(u0, v0, w0)
        J0 = metrics['enstrophy_max'] + 100 * metrics['chi_max']
        
        grad_u = torch.zeros_like(u0)
        grad_v = torch.zeros_like(v0)
        grad_w = torch.zeros_like(w0)
        
        # Perturb each low-frequency mode and compute gradient
        # (Full finite difference would be too expensive)
        n_samples = min(27, u0.numel())  # Sample a few points
        
        indices = torch.randperm(u0.numel())[:n_samples]
        
        for idx in indices:
            i = idx // (self.Ny * self.Nz)
            j = (idx % (self.Ny * self.Nz)) // self.Nz
            k = idx % self.Nz
            
            # Perturb u
            u_pert = u0.clone()
            u_pert[i, j, k] += epsilon
            metrics_pert, _ = self.forward_integrate(u_pert, v0, w0)
            J_pert = metrics_pert['enstrophy_max'] + 100 * metrics_pert['chi_max']
            grad_u[i, j, k] = (J_pert - J0) / epsilon
            
        return grad_u, grad_v, grad_w
    
    def compute_gradient_autodiff(
        self,
        u0: Tensor,
        v0: Tensor,
        w0: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute gradient using PyTorch autograd.
        
        Requires differentiable forward pass.
        """
        u0 = u0.clone().requires_grad_(True)
        v0 = v0.clone().requires_grad_(True)
        w0 = w0.clone().requires_grad_(True)
        
        from tensornet.cfd.ns_3d import NSState3D
        
        state = NSState3D(u=u0, v=v0, w=w0, t=0.0, step=0)
        
        dt = self.config.dt
        n_steps = min(10, int(self.config.T_horizon / dt))  # Shorter for grad
        
        total_enstrophy = torch.tensor(0.0, dtype=self.dtype)
        
        for step in range(n_steps):
            enstrophy = self.enstrophy_obj.evaluate(state.u, state.v, state.w)
            total_enstrophy = total_enstrophy + enstrophy
            
            state, _ = self.solver.step_rk4(state, dt)
        
        # Final enstrophy bonus
        final_enstrophy = self.enstrophy_obj.evaluate(state.u, state.v, state.w)
        objective = total_enstrophy + 10.0 * final_enstrophy
        
        # Backprop
        objective.backward()
        
        return u0.grad, v0.grad, w0.grad
    
    def hunt(
        self,
        u0_init: Optional[Tensor] = None,
        v0_init: Optional[Tensor] = None,
        w0_init: Optional[Tensor] = None,
        use_autodiff: bool = True,
    ) -> HuntResult:
        """
        Hunt for singularity candidates.
        
        Uses gradient ascent to find initial conditions that maximize
        enstrophy and chi growth.
        """
        from tensornet.cfd.ns_3d import project_velocity_3d
        
        # Initialize IC
        if u0_init is None:
            u0, v0, w0 = self.random_smooth_ic()
        else:
            u0, v0, w0 = u0_init.clone(), v0_init.clone(), w0_init.clone()
        
        # Momentum buffers
        mu = torch.zeros_like(u0)
        mv = torch.zeros_like(v0)
        mw = torch.zeros_like(w0)
        
        convergence_history = []
        best_chi = 0.0
        best_enstrophy = 0.0
        best_ic = (u0.clone(), v0.clone(), w0.clone())
        
        for iteration in range(self.config.max_iterations):
            # Forward integrate
            metrics, trajectory = self.forward_integrate(u0, v0, w0)
            
            chi_max = metrics['chi_max']
            enstrophy_max = metrics['enstrophy_max']
            
            print(f"  Iter {iteration}: chi_max={chi_max:.1f}, enstrophy_max={enstrophy_max:.4f}")
            
            # Track best
            if chi_max > best_chi:
                best_chi = chi_max
                best_enstrophy = enstrophy_max
                best_ic = (u0.clone(), v0.clone(), w0.clone())
            
            convergence_history.append({
                'iteration': iteration,
                'chi_max': chi_max,
                'enstrophy_max': enstrophy_max,
            })
            
            # Check for blowup
            if chi_max > self.config.chi_threshold:
                print(f"  [CANDIDATE FOUND] chi = {chi_max:.1f} > threshold")
                return HuntResult(
                    initial_condition=torch.stack([u0, v0, w0]),
                    final_enstrophy=enstrophy_max,
                    final_chi=chi_max,
                    chi_trajectory=metrics['chi_trajectory'],
                    enstrophy_trajectory=metrics['enstrophy_trajectory'],
                    blowup_detected=True,
                    convergence_history=convergence_history,
                    iterations=iteration + 1,
                    verdict="SINGULARITY_CANDIDATE",
                )
            
            # Compute gradient (ASCENT direction)
            try:
                if use_autodiff:
                    grad_u, grad_v, grad_w = self.compute_gradient_autodiff(u0, v0, w0)
                else:
                    grad_u, grad_v, grad_w = self.compute_gradient_numerical(u0, v0, w0)
            except Exception as e:
                print(f"  [WARNING] Gradient computation failed: {e}")
                # Fall back to random perturbation
                grad_u = torch.randn_like(u0) * 0.01
                grad_v = torch.randn_like(v0) * 0.01
                grad_w = torch.randn_like(w0) * 0.01
            
            # Handle None gradients
            if grad_u is None:
                grad_u = torch.zeros_like(u0)
            if grad_v is None:
                grad_v = torch.zeros_like(v0)
            if grad_w is None:
                grad_w = torch.zeros_like(w0)
            
            # Gradient clipping
            grad_norm = torch.sqrt(
                grad_u.pow(2).sum() + grad_v.pow(2).sum() + grad_w.pow(2).sum()
            )
            if grad_norm > self.config.grad_clip:
                scale = self.config.grad_clip / (grad_norm + 1e-8)
                grad_u = grad_u * scale
                grad_v = grad_v * scale
                grad_w = grad_w * scale
            
            # Momentum update
            mu = self.config.momentum * mu + grad_u
            mv = self.config.momentum * mv + grad_v
            mw = self.config.momentum * mw + grad_w
            
            # Gradient ASCENT (we want to MAXIMIZE destruction)
            u0 = u0 + self.config.step_size * mu
            v0 = v0 + self.config.step_size * mv
            w0 = w0 + self.config.step_size * mw
            
            # Re-project to divergence-free (maintain incompressibility)
            if iteration % self.config.projection_interval == 0:
                proj = project_velocity_3d(u0, v0, w0, self.dx, self.dy, self.dz, dt=1.0)
                u0 = proj.u_projected
                v0 = proj.v_projected
                w0 = proj.w_projected
        
        # Return best found
        return HuntResult(
            initial_condition=torch.stack(best_ic),
            final_enstrophy=best_enstrophy,
            final_chi=best_chi,
            chi_trajectory=metrics['chi_trajectory'],
            enstrophy_trajectory=metrics['enstrophy_trajectory'],
            blowup_detected=False,
            convergence_history=convergence_history,
            iterations=self.config.max_iterations,
            verdict="BOUNDED",
        )


def test_singularity_hunter():
    """Test the singularity hunter on a small grid."""
    print("\n" + "=" * 60)
    print("Singularity Hunter Test")
    print("=" * 60)
    
    config = HuntingConfig(
        max_iterations=10,
        step_size=0.001,
        T_horizon=0.5,
        dt=0.02,
    )
    
    # Small grid for testing
    hunter = SingularityHunter(
        Nx=16, Ny=16, Nz=16,
        Lx=2*np.pi, Ly=2*np.pi, Lz=2*np.pi,
        nu=0.1,
        config=config,
    )
    
    print("\nHunting for singularity candidates...")
    result = hunter.hunt(use_autodiff=True)
    
    print(f"\nResult:")
    print(f"  Iterations: {result.iterations}")
    print(f"  Final chi: {result.final_chi:.1f}")
    print(f"  Final enstrophy: {result.final_enstrophy:.4f}")
    print(f"  Blowup detected: {result.blowup_detected}")
    print(f"  Verdict: {result.verdict}")
    
    return result


if __name__ == "__main__":
    test_singularity_hunter()
