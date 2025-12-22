"""
Adjoint-Based Vorticity Maximization for Blow-Up Profile Discovery.

This module finds the "bad apple" initial condition that maximizes
vorticity growth, which is the candidate for a self-similar singularity.

Mathematical Setup
==================

Objective: Find u₀ that maximizes ||ω(T)||_∞

Using the adjoint method:
1. Forward: Evolve NS from u₀ to get u(T)
2. Compute objective: J = ||ω(T)||_∞ or ∫|ω(T)|² dx (enstrophy)
3. Backward: Solve adjoint equations to get ∂J/∂u₀
4. Update: u₀ ← u₀ + α * gradient

The adjoint of NS in rescaled coordinates gives the sensitivity of 
vorticity growth to the initial profile.

Reference: 
- Kerswell (2018), "Nonlinear nonmodal stability theory"
- Hou & Li (2006), "Dynamic depletion of vortex stretching"
"""

from __future__ import annotations
import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Callable, List
import time

from tensornet.cfd.self_similar import (
    RescaledNSEquations,
    SelfSimilarScaling,
    SelfSimilarProfile,
    create_candidate_profile,
)


@dataclass
class OptimizationConfig:
    """Configuration for adjoint optimization."""
    N: int = 48                    # Grid resolution
    nu: float = 1e-3               # Viscosity
    T_final: float = 0.5           # Final time for forward evolution
    dt: float = 0.01               # Time step
    n_iter: int = 100              # Optimization iterations
    learning_rate: float = 0.1     # Step size
    momentum: float = 0.9          # Momentum coefficient
    objective: str = "enstrophy"   # "enstrophy" or "max_vorticity"
    regularization: float = 1e-4  # Smoothness penalty
    constraint_energy: float = 1.0 # Energy constraint ||u₀||² ≤ E₀
    verbose: bool = True


@dataclass
class OptimizationResult:
    """Result of adjoint optimization."""
    profile: torch.Tensor          # Optimized initial condition
    objective_history: List[float] # Objective values per iteration
    gradient_norm_history: List[float]
    final_enstrophy: float
    final_max_vorticity: float
    converged: bool
    n_iterations: int


class AdjointOptimizer:
    """
    Adjoint-based optimizer for finding blow-up candidates.
    
    This solves: max_{u₀} J(u(T; u₀))
    where u(T) is the solution at time T starting from u₀.
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.scaling = SelfSimilarScaling(alpha=0.5, beta=0.5, T_star=1.0)
        self.ns = RescaledNSEquations(
            self.scaling, 
            nu=config.nu, 
            N=config.N
        )
        
        # Spectral setup
        self.N = config.N
        self.L = 2 * np.pi
        self.dx = self.L / self.N
        
        k = torch.fft.fftfreq(self.N, self.dx) * 2 * np.pi
        self.kx, self.ky, self.kz = torch.meshgrid(k, k, k, indexing='ij')
        self.k_sq = self.kx**2 + self.ky**2 + self.kz**2
        self.k_sq[0, 0, 0] = 1.0
        
        # Dealiasing mask (2/3 rule)
        k_max = self.N // 3
        self.dealias = (
            (torch.abs(self.kx) < k_max) & 
            (torch.abs(self.ky) < k_max) & 
            (torch.abs(self.kz) < k_max)
        ).float()
    
    def forward_euler_step(
        self, 
        U: torch.Tensor, 
        dt: float
    ) -> torch.Tensor:
        """
        One step of forward NS evolution (RK4).
        
        Args:
            U: Current velocity field (N, N, N, 3)
            dt: Time step
            
        Returns:
            U_new: Velocity at next time step
        """
        def rhs(u):
            # Compute RHS of NS: -advection + viscous diffusion
            # Then project to divergence-free
            u_hat = torch.fft.fftn(u, dim=(0, 1, 2))
            
            # Derivatives
            dudx = torch.fft.ifftn(1j * self.kx.unsqueeze(-1) * u_hat, dim=(0, 1, 2)).real
            dudy = torch.fft.ifftn(1j * self.ky.unsqueeze(-1) * u_hat, dim=(0, 1, 2)).real
            dudz = torch.fft.ifftn(1j * self.kz.unsqueeze(-1) * u_hat, dim=(0, 1, 2)).real
            
            # Advection
            adv = torch.zeros_like(u)
            for i in range(3):
                adv[..., i] = u[..., 0] * dudx[..., i] + u[..., 1] * dudy[..., i] + u[..., 2] * dudz[..., i]
            
            # Laplacian
            lap = torch.fft.ifftn(-self.k_sq.unsqueeze(-1) * u_hat, dim=(0, 1, 2)).real
            
            # RHS = -advection + nu * laplacian
            f = -adv + self.config.nu * lap
            
            # Project to divergence-free
            f_hat = torch.fft.fftn(f, dim=(0, 1, 2))
            div_hat = 1j * self.kx * f_hat[..., 0] + 1j * self.ky * f_hat[..., 1] + 1j * self.kz * f_hat[..., 2]
            P_hat = div_hat / self.k_sq
            P_hat[0, 0, 0] = 0
            
            f_hat[..., 0] -= 1j * self.kx * P_hat
            f_hat[..., 1] -= 1j * self.ky * P_hat
            f_hat[..., 2] -= 1j * self.kz * P_hat
            
            # Dealias
            f_hat = f_hat * self.dealias.unsqueeze(-1)
            
            return torch.fft.ifftn(f_hat, dim=(0, 1, 2)).real
        
        # RK4
        k1 = rhs(U)
        k2 = rhs(U + 0.5 * dt * k1)
        k3 = rhs(U + 0.5 * dt * k2)
        k4 = rhs(U + dt * k3)
        
        return U + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    
    def evolve_forward(
        self, 
        U0: torch.Tensor, 
        store_trajectory: bool = True
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Evolve NS forward from initial condition.
        
        Args:
            U0: Initial velocity field
            store_trajectory: Whether to store all time steps (for adjoint)
            
        Returns:
            U_final: Solution at T_final
            trajectory: List of states at each time step (if store_trajectory)
        """
        n_steps = int(self.config.T_final / self.config.dt)
        U = U0.clone()
        
        trajectory = [U0.clone()] if store_trajectory else None
        
        for _ in range(n_steps):
            U = self.forward_euler_step(U, self.config.dt)
            if store_trajectory:
                trajectory.append(U.clone())
        
        return U, trajectory
    
    def compute_objective(self, U: torch.Tensor) -> torch.Tensor:
        """
        Compute objective function J(U).
        
        Options:
        - "enstrophy": J = ∫|ω|² dx
        - "max_vorticity": J = max|ω|
        """
        omega = self.ns.vorticity(U)
        omega_sq = (omega ** 2).sum(dim=-1)
        
        if self.config.objective == "enstrophy":
            return omega_sq.sum() * self.dx**3
        elif self.config.objective == "max_vorticity":
            return torch.sqrt(omega_sq).max()
        else:
            raise ValueError(f"Unknown objective: {self.config.objective}")
    
    def compute_objective_gradient(self, U: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient of objective w.r.t. U: ∂J/∂U.
        
        For enstrophy J = ∫|ω|² dx:
            ∂J/∂U = 2 * curl(ω) = -2 * ΔU (for div-free U)
            
        This is the initial condition for the adjoint solve.
        """
        if self.config.objective == "enstrophy":
            # ∂J/∂u = -2 * Δu (using vector identity for curl-curl)
            U_hat = torch.fft.fftn(U, dim=(0, 1, 2))
            lap_U = torch.fft.ifftn(-self.k_sq.unsqueeze(-1) * U_hat, dim=(0, 1, 2)).real
            return -2 * lap_U * self.dx**3
            
        elif self.config.objective == "max_vorticity":
            # Gradient of max is delta function at max location
            omega = self.ns.vorticity(U)
            omega_mag = torch.sqrt((omega ** 2).sum(dim=-1))
            max_val = omega_mag.max()
            max_loc = (omega_mag == max_val).float()
            
            # Approximate gradient using softmax
            softmax_weight = torch.exp(10 * (omega_mag - max_val))
            softmax_weight = softmax_weight / softmax_weight.sum()
            
            # Chain rule through vorticity
            grad = torch.zeros_like(U)
            for i in range(3):
                grad[..., i] = softmax_weight * omega[..., i] / (omega_mag + 1e-10)
            
            # Apply curl^T to get gradient w.r.t. U
            grad_hat = torch.fft.fftn(grad, dim=(0, 1, 2))
            curl_adj = torch.zeros_like(grad_hat)
            curl_adj[..., 0] = 1j * self.ky * grad_hat[..., 2] - 1j * self.kz * grad_hat[..., 1]
            curl_adj[..., 1] = 1j * self.kz * grad_hat[..., 0] - 1j * self.kx * grad_hat[..., 2]
            curl_adj[..., 2] = 1j * self.kx * grad_hat[..., 1] - 1j * self.ky * grad_hat[..., 0]
            
            return torch.fft.ifftn(curl_adj, dim=(0, 1, 2)).real
    
    def adjoint_step(
        self, 
        Lambda: torch.Tensor,
        U: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """
        One step of adjoint NS evolution (backward in time).
        
        The adjoint equation is:
            -∂λ/∂t + (∇u)^T λ - (u·∇)λ + ν Δλ + ∇p = 0
            
        Args:
            Lambda: Current adjoint field
            U: Forward velocity at this time
            dt: Time step (positive, we step backward)
            
        Returns:
            Lambda_prev: Adjoint at previous time
        """
        def adjoint_rhs(lam, u):
            lam_hat = torch.fft.fftn(lam, dim=(0, 1, 2))
            u_hat = torch.fft.fftn(u, dim=(0, 1, 2))
            
            # Derivatives of lambda
            dlam_dx = torch.fft.ifftn(1j * self.kx.unsqueeze(-1) * lam_hat, dim=(0, 1, 2)).real
            dlam_dy = torch.fft.ifftn(1j * self.ky.unsqueeze(-1) * lam_hat, dim=(0, 1, 2)).real
            dlam_dz = torch.fft.ifftn(1j * self.kz.unsqueeze(-1) * lam_hat, dim=(0, 1, 2)).real
            
            # Derivatives of u
            du_dx = torch.fft.ifftn(1j * self.kx.unsqueeze(-1) * u_hat, dim=(0, 1, 2)).real
            du_dy = torch.fft.ifftn(1j * self.ky.unsqueeze(-1) * u_hat, dim=(0, 1, 2)).real
            du_dz = torch.fft.ifftn(1j * self.kz.unsqueeze(-1) * u_hat, dim=(0, 1, 2)).real
            
            # (∇u)^T λ: adjoint of advection
            grad_u_T_lam = torch.zeros_like(lam)
            for i in range(3):
                grad_u_T_lam[..., i] = (
                    du_dx[..., i] * lam[..., 0] +
                    du_dy[..., i] * lam[..., 1] +
                    du_dz[..., i] * lam[..., 2]
                )
            
            # -(u·∇)λ
            u_dot_grad_lam = torch.zeros_like(lam)
            for i in range(3):
                u_dot_grad_lam[..., i] = (
                    u[..., 0] * dlam_dx[..., i] +
                    u[..., 1] * dlam_dy[..., i] +
                    u[..., 2] * dlam_dz[..., i]
                )
            
            # Laplacian
            lap_lam = torch.fft.ifftn(-self.k_sq.unsqueeze(-1) * lam_hat, dim=(0, 1, 2)).real
            
            # RHS (note: backward in time, so flip sign)
            f = grad_u_T_lam - u_dot_grad_lam + self.config.nu * lap_lam
            
            # Project to divergence-free
            f_hat = torch.fft.fftn(f, dim=(0, 1, 2))
            div_hat = 1j * self.kx * f_hat[..., 0] + 1j * self.ky * f_hat[..., 1] + 1j * self.kz * f_hat[..., 2]
            P_hat = div_hat / self.k_sq
            P_hat[0, 0, 0] = 0
            
            f_hat[..., 0] -= 1j * self.kx * P_hat
            f_hat[..., 1] -= 1j * self.ky * P_hat
            f_hat[..., 2] -= 1j * self.kz * P_hat
            
            return torch.fft.ifftn(f_hat, dim=(0, 1, 2)).real
        
        # Simple backward Euler for stability
        return Lambda + dt * adjoint_rhs(Lambda, U)
    
    def compute_gradient(
        self, 
        U0: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute full gradient ∂J/∂U₀ via forward-adjoint.
        
        Returns:
            grad: Gradient of objective w.r.t. initial condition
            J: Objective value
        """
        # Forward pass
        U_final, trajectory = self.evolve_forward(U0, store_trajectory=True)
        J = self.compute_objective(U_final)
        
        # Initialize adjoint with terminal condition
        Lambda = self.compute_objective_gradient(U_final)
        
        # Backward pass
        n_steps = len(trajectory) - 1
        for i in range(n_steps - 1, -1, -1):
            Lambda = self.adjoint_step(Lambda, trajectory[i], self.config.dt)
        
        # Gradient = adjoint at t=0
        grad = Lambda
        
        # Add regularization gradient
        if self.config.regularization > 0:
            U0_hat = torch.fft.fftn(U0, dim=(0, 1, 2))
            reg_grad = torch.fft.ifftn(
                self.k_sq.unsqueeze(-1) * U0_hat, 
                dim=(0, 1, 2)
            ).real
            grad = grad - self.config.regularization * reg_grad
        
        return grad, J.item()
    
    def project_energy_constraint(self, U: torch.Tensor) -> torch.Tensor:
        """
        Project U onto the energy constraint ||U||² ≤ E₀.
        """
        E = (U ** 2).sum() * self.dx**3
        E_max = self.config.constraint_energy
        
        if E > E_max:
            U = U * np.sqrt(E_max / E.item())
        
        return U
    
    def project_divergence_free(self, U: torch.Tensor) -> torch.Tensor:
        """Project U to be divergence-free."""
        U_hat = torch.fft.fftn(U, dim=(0, 1, 2))
        div_hat = 1j * self.kx * U_hat[..., 0] + 1j * self.ky * U_hat[..., 1] + 1j * self.kz * U_hat[..., 2]
        P_hat = div_hat / self.k_sq
        P_hat[0, 0, 0] = 0
        
        U_hat[..., 0] -= 1j * self.kx * P_hat
        U_hat[..., 1] -= 1j * self.ky * P_hat
        U_hat[..., 2] -= 1j * self.kz * P_hat
        
        return torch.fft.ifftn(U_hat, dim=(0, 1, 2)).real
    
    def optimize(
        self, 
        U0_init: Optional[torch.Tensor] = None
    ) -> OptimizationResult:
        """
        Run adjoint optimization to find blow-up candidate.
        
        Args:
            U0_init: Initial guess (default: tornado profile)
            
        Returns:
            OptimizationResult with optimized profile
        """
        if U0_init is None:
            U0 = create_candidate_profile(self.N, "tornado", strength=0.5)
        else:
            U0 = U0_init.clone()
        
        U0 = self.project_divergence_free(U0)
        U0 = self.project_energy_constraint(U0)
        
        velocity = torch.zeros_like(U0)  # Momentum buffer
        
        objective_history = []
        gradient_norm_history = []
        
        best_U0 = U0.clone()
        best_J = -float('inf')
        
        if self.config.verbose:
            print("=" * 60)
            print("ADJOINT OPTIMIZATION: Finding Blow-Up Candidate")
            print("=" * 60)
            print(f"Grid: {self.N}³, ν = {self.config.nu}, T = {self.config.T_final}")
            print(f"Objective: {self.config.objective}")
            print("-" * 60)
        
        start_time = time.time()
        
        for iteration in range(self.config.n_iter):
            # Compute gradient
            grad, J = self.compute_gradient(U0)
            grad_norm = torch.sqrt((grad ** 2).sum()).item()
            
            objective_history.append(J)
            gradient_norm_history.append(grad_norm)
            
            if J > best_J:
                best_J = J
                best_U0 = U0.clone()
            
            # Momentum update
            velocity = self.config.momentum * velocity + grad
            
            # Gradient ascent (maximize objective)
            U0 = U0 + self.config.learning_rate * velocity
            
            # Project constraints
            U0 = self.project_divergence_free(U0)
            U0 = self.project_energy_constraint(U0)
            
            if self.config.verbose and iteration % 10 == 0:
                omega_max = self.ns.max_vorticity(U0).item()
                print(f"Iter {iteration:4d}: J = {J:.6f}, ||∇J|| = {grad_norm:.4f}, ||ω||_∞ = {omega_max:.4f}")
        
        elapsed = time.time() - start_time
        
        # Final evaluation
        U_final, _ = self.evolve_forward(best_U0, store_trajectory=False)
        final_enstrophy = self.ns.enstrophy(U_final).item()
        final_max_vorticity = self.ns.max_vorticity(U_final).item()
        
        if self.config.verbose:
            print("-" * 60)
            print(f"Optimization complete in {elapsed:.1f}s")
            print(f"Final enstrophy: {final_enstrophy:.4f}")
            print(f"Final max vorticity: {final_max_vorticity:.4f}")
            print("=" * 60)
        
        return OptimizationResult(
            profile=best_U0,
            objective_history=objective_history,
            gradient_norm_history=gradient_norm_history,
            final_enstrophy=final_enstrophy,
            final_max_vorticity=final_max_vorticity,
            converged=len(objective_history) > 1 and gradient_norm_history[-1] < 1e-4,
            n_iterations=len(objective_history),
        )


def find_blowup_candidate(
    N: int = 48,
    nu: float = 1e-3,
    n_iter: int = 50,
    verbose: bool = True
) -> OptimizationResult:
    """
    Convenience function to find a blow-up candidate profile.
    
    This is the "Step A" of the Millennium proof pipeline:
    Find the initial condition that maximizes vorticity growth.
    
    Args:
        N: Grid resolution
        nu: Viscosity
        n_iter: Number of optimization iterations
        verbose: Print progress
        
    Returns:
        OptimizationResult containing the candidate profile
    """
    config = OptimizationConfig(
        N=N,
        nu=nu,
        T_final=0.3,
        dt=0.005,
        n_iter=n_iter,
        learning_rate=0.05,
        momentum=0.9,
        objective="enstrophy",
        regularization=1e-5,
        constraint_energy=1.0,
        verbose=verbose,
    )
    
    optimizer = AdjointOptimizer(config)
    return optimizer.optimize()


if __name__ == "__main__":
    # Quick test
    result = find_blowup_candidate(N=32, nu=1e-3, n_iter=30, verbose=True)
    
    print("\nCandidate profile statistics:")
    print(f"  Shape: {result.profile.shape}")
    print(f"  Final enstrophy: {result.final_enstrophy:.4f}")
    print(f"  Final max vorticity: {result.final_max_vorticity:.4f}")
