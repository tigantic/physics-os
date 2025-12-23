"""
Newton Refinement for Self-Similar Profile.

Once we have a candidate profile from the Hou-Luo ansatz or adjoint optimization,
we refine it using Newton iteration to find the exact fixed point: F(U*) = 0.

The Newton Method for Operator Equations
=========================================

Given F(U) = 0, Newton's method iterates:
    U_{k+1} = U_k - [DF(U_k)]^{-1} F(U_k)

For our rescaled Navier-Stokes:
    F(U) = -(U·∇)U - αU - β(ξ·∇)U + ν∇²U - ∇P = 0

The Jacobian DF is the linearized operator at U_k.

In practice, we solve:
    DF(U_k) · δU = -F(U_k)
    U_{k+1} = U_k + δU

The solve uses GMRES or a similar Krylov method.

Convergence Guarantee
=====================

If ||DF(U₀)^{-1} F(U₀)|| · ||DF(U₀)^{-1}|| · ||D²F|| < 0.5

then Newton converges quadratically to a true solution U*.

This is intimately connected to the Newton-Kantorovich theorem in kantorovich.py.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List, Callable
import time

from tensornet.cfd.self_similar import (
    RescaledNSEquations,
    SelfSimilarScaling,
)
from tensornet.cfd.kantorovich import (
    NewtonKantorovichVerifier,
    KantorovichBounds,
    VerificationStatus,
)


@dataclass
class NewtonConfig:
    """Configuration for Newton refinement."""
    max_iter: int = 50              # Maximum Newton steps
    tol: float = 1e-8               # Convergence tolerance on ||F||
    gmres_iter: int = 100           # GMRES iterations per Newton step
    gmres_tol: float = 1e-6         # GMRES convergence tolerance
    damping: float = 1.0            # Damping factor (1.0 = full Newton)
    line_search: bool = True        # Use line search for robustness
    verbose: bool = True


@dataclass
class NewtonResult:
    """Result of Newton refinement."""
    profile: torch.Tensor           # Refined profile
    residual_history: List[float]   # ||F|| at each iteration
    converged: bool
    n_iterations: int
    final_residual: float
    newton_decrement: float         # Estimate of ||U* - U||


class NewtonRefiner:
    """
    Newton iteration to refine self-similar profiles toward F(U*) = 0.
    """
    
    def __init__(
        self,
        N: int = 64,
        nu: float = 1e-3,
        tau: float = 10.0,  # Large tau for "steady state"
    ):
        self.N = N
        self.nu = nu
        self.tau = tau
        
        self.L = 2 * np.pi
        self.dx = self.L / N
        
        # Spectral grid
        k = torch.fft.fftfreq(N, self.dx) * 2 * np.pi
        self.kx, self.ky, self.kz = torch.meshgrid(k, k, k, indexing='ij')
        self.k_sq = self.kx**2 + self.ky**2 + self.kz**2
        self.k_sq_safe = self.k_sq.clone()
        self.k_sq_safe[0, 0, 0] = 1.0
        
        # Rescaled NS equations
        self.scaling = SelfSimilarScaling(alpha=0.5, beta=0.5, T_star=1.0)
        self.ns = RescaledNSEquations(self.scaling, nu=nu, N=N)
    
    def compute_residual(self, U: torch.Tensor) -> torch.Tensor:
        """Compute F(U) - the fixed-point residual."""
        tau_tensor = torch.tensor(self.tau, dtype=torch.float64)
        return self.ns.residual(U, tau_tensor)
    
    def residual_norm(self, U: torch.Tensor) -> float:
        """Compute ||F(U)||."""
        R = self.compute_residual(U)
        return torch.sqrt((R**2).sum() * self.dx**3).item()
    
    def jacobian_action(self, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Compute DF(U) · V - the Jacobian-vector product.
        
        DF(U)·V = -(V·∇)U - (U·∇)V - αV - β(ξ·∇)V + ν∇²V - ∇q
        """
        tau_tensor = torch.tensor(self.tau, dtype=torch.float64)
        nu_eff = self.ns.effective_viscosity(tau_tensor)
        alpha = self.scaling.alpha
        beta = self.scaling.beta
        
        # FFT
        U_hat = torch.fft.fftn(U, dim=(0, 1, 2))
        V_hat = torch.fft.fftn(V, dim=(0, 1, 2))
        
        # Derivatives
        dUdx = torch.fft.ifftn(1j * self.kx.unsqueeze(-1) * U_hat, dim=(0, 1, 2)).real
        dUdy = torch.fft.ifftn(1j * self.ky.unsqueeze(-1) * U_hat, dim=(0, 1, 2)).real
        dUdz = torch.fft.ifftn(1j * self.kz.unsqueeze(-1) * U_hat, dim=(0, 1, 2)).real
        
        dVdx = torch.fft.ifftn(1j * self.kx.unsqueeze(-1) * V_hat, dim=(0, 1, 2)).real
        dVdy = torch.fft.ifftn(1j * self.ky.unsqueeze(-1) * V_hat, dim=(0, 1, 2)).real
        dVdz = torch.fft.ifftn(1j * self.kz.unsqueeze(-1) * V_hat, dim=(0, 1, 2)).real
        
        # (V·∇)U
        V_grad_U = torch.zeros_like(U)
        for i in range(3):
            V_grad_U[..., i] = V[..., 0] * dUdx[..., i] + V[..., 1] * dUdy[..., i] + V[..., 2] * dUdz[..., i]
        
        # (U·∇)V
        U_grad_V = torch.zeros_like(U)
        for i in range(3):
            U_grad_V[..., i] = U[..., 0] * dVdx[..., i] + U[..., 1] * dVdy[..., i] + U[..., 2] * dVdz[..., i]
        
        # Stretching
        xi = self.ns.xi
        xi_x = xi.view(-1, 1, 1, 1).expand_as(V[..., 0:1])
        xi_y = xi.view(1, -1, 1, 1).expand_as(V[..., 0:1])
        xi_z = xi.view(1, 1, -1, 1).expand_as(V[..., 0:1])
        
        xi_grad_V = xi_x * dVdx + xi_y * dVdy + xi_z * dVdz
        stretching = beta * xi_grad_V + alpha * V
        
        # Laplacian
        lap_V = torch.fft.ifftn(-self.k_sq.unsqueeze(-1) * V_hat, dim=(0, 1, 2)).real
        
        # Unprojected
        DFV_unprojected = -V_grad_U - U_grad_V - stretching + nu_eff * lap_V
        
        # Project to divergence-free
        DFV = self.ns.pressure_projection(DFV_unprojected)
        
        return DFV
    
    def solve_linear_system(
        self,
        U: torch.Tensor,
        b: torch.Tensor,
        max_iter: int = 100,
        tol: float = 1e-4,
    ) -> Tuple[torch.Tensor, float, int]:
        """
        Solve DF(U) · x = b using preconditioned Richardson iteration.
        
        Uses viscous Laplacian as preconditioner: P = (αI - ν∇²)^{-1}
        This is much more stable than GMRES for this problem.
        
        Returns:
            x: Solution
            residual: Final residual norm
            n_iter: Number of iterations
        """
        alpha = self.scaling.alpha
        nu_eff = self.nu  # Simplified
        
        # Preconditioner: inverse of (α - ν∇²) in spectral space
        def precondition(r):
            r_hat = torch.fft.fftn(r, dim=(0, 1, 2))
            # (α + ν k²)^{-1}
            inv_op = 1.0 / (alpha + nu_eff * self.k_sq.unsqueeze(-1) + 0.1)
            return torch.fft.ifftn(r_hat * inv_op, dim=(0, 1, 2)).real
        
        x = torch.zeros_like(b)
        r = b.clone()
        b_norm = torch.sqrt((b**2).sum()).item()
        
        if b_norm < 1e-15:
            return x, 0.0, 0
        
        # Preconditioned Richardson iteration
        omega = 0.5  # Relaxation parameter
        
        for k in range(max_iter):
            # Apply preconditioner
            z = precondition(r)
            
            # Update
            x = x + omega * z
            
            # New residual
            Ax = self.jacobian_action(U, x)
            r = b - Ax
            
            r_norm = torch.sqrt((r**2).sum()).item()
            
            if r_norm / b_norm < tol:
                return x, r_norm, k + 1
        
        return x, r_norm, max_iter
    
    def line_search(
        self,
        U: torch.Tensor,
        delta: torch.Tensor,
        f0: float,
    ) -> float:
        """
        Backtracking line search to find good step size.
        
        Find α such that ||F(U + α·δ)|| < ||F(U)||
        """
        alpha = 1.0
        c = 1e-4  # Armijo constant
        
        for _ in range(10):
            U_new = U + alpha * delta
            f_new = self.residual_norm(U_new)
            
            if f_new < (1 - c * alpha) * f0:
                return alpha
            
            alpha *= 0.5
        
        return alpha
    
    def refine(
        self,
        U0: torch.Tensor,
        config: NewtonConfig = None,
    ) -> NewtonResult:
        """
        Refine a profile using pseudo-time evolution with stabilization.
        
        Key insight: The fixed point F(U*)=0 is a saddle, not a minimum.
        We use relaxation with momentum damping to find it.
        
        Args:
            U0: Initial guess
            config: Newton configuration
            
        Returns:
            NewtonResult with refined profile
        """
        if config is None:
            config = NewtonConfig()
        
        if config.verbose:
            print("=" * 60)
            print("PSEUDO-TIME EVOLUTION FOR SELF-SIMILAR PROFILE")
            print("=" * 60)
            print(f"  Grid: {self.N}³, ν = {self.nu}")
            print(f"  Target: ||F(U)|| < {config.tol:.2e}")
            print("-" * 60)
        
        U = U0.clone()
        residual_history = []
        
        # Fixed time step
        dt = 0.01
        nu_eff = max(self.nu, 0.01)  # Increased artificial viscosity
        
        start_time = time.time()
        best_U = U.clone()
        best_f = float('inf')
        
        for iteration in range(config.max_iter * 10):
            # Compute residual F(U)
            F = self.compute_residual(U)
            f_norm = torch.sqrt((F**2).sum() * self.dx**3).item()
            
            if iteration % 20 == 0:
                residual_history.append(f_norm)
                
                if f_norm < best_f:
                    best_f = f_norm
                    best_U = U.clone()
                
                if config.verbose:
                    print(f"  Iter {iteration:4d}: ||F|| = {f_norm:.6e}")
            
            # Check convergence
            if f_norm < config.tol:
                if config.verbose:
                    print("-" * 60)
                    print(f"  ★ CONVERGED in {iteration} iterations!")
                    print(f"  Final ||F|| = {f_norm:.6e}")
                    print("=" * 60)
                
                return NewtonResult(
                    profile=U,
                    residual_history=residual_history,
                    converged=True,
                    n_iterations=iteration,
                    final_residual=f_norm,
                    newton_decrement=f_norm,
                )
            
            # Simple explicit step with stabilizing diffusion
            # dU/dτ = -F(U) + ν_stab * ∇²U
            U_hat = torch.fft.fftn(U, dim=(0, 1, 2))
            laplacian = torch.fft.ifftn(-self.k_sq.unsqueeze(-1) * U_hat, dim=(0, 1, 2)).real
            
            # Take step
            U_new = U - dt * F + dt * nu_eff * laplacian
            
            # Project to divergence-free
            U_new = self.ns.pressure_projection(U_new)
            
            # Always accept (no line search - just damped evolution)
            U = U_new
            
            # Early termination if making no progress
            if iteration > 200 and len(residual_history) >= 3:
                recent = residual_history[-3:]
                if max(recent) - min(recent) < 1e-8 * best_f:
                    break
        
        elapsed = time.time() - start_time
        
        if config.verbose:
            print("-" * 60)
            print(f"  Evolution completed ({iteration} steps)")
            print(f"  Best ||F|| = {best_f:.6e}")
            print(f"  Time: {elapsed:.1f}s")
            print("=" * 60)
        
        return NewtonResult(
            profile=best_U,
            residual_history=residual_history,
            converged=False,
            n_iterations=iteration,
            final_residual=best_f,
            newton_decrement=best_f,
        )


def refine_blowup_candidate(
    profile: torch.Tensor,
    nu: float = 1e-3,
    max_iter: int = 30,
    verbose: bool = True,
) -> NewtonResult:
    """
    Convenience function to refine a blow-up candidate.
    
    Args:
        profile: Initial candidate (N, N, N, 3)
        nu: Viscosity
        max_iter: Maximum Newton iterations
        verbose: Print progress
        
    Returns:
        NewtonResult with refined profile
    """
    N = profile.shape[0]
    refiner = NewtonRefiner(N=N, nu=nu)
    
    config = NewtonConfig(
        max_iter=max_iter,
        tol=1e-6,
        gmres_iter=50,
        gmres_tol=1e-4,
        line_search=True,
        verbose=verbose,
    )
    
    return refiner.refine(profile, config)


if __name__ == "__main__":
    from tensornet.cfd.hou_luo_ansatz import create_hou_luo_profile, HouLuoConfig
    
    # Create Hou-Luo profile
    config = HouLuoConfig(N=32)
    U0 = create_hou_luo_profile(config)
    
    # Analyze residual with different alpha values
    print("="*60)
    print("SEARCHING FOR OPTIMAL RESCALING EXPONENT α")
    print("="*60)
    print()
    print("The self-similar fixed point F(U) = 0 depends on α.")
    print("Finding α that minimizes ||F(U)|| for this profile...")
    print()
    
    best_alpha = 0.5
    best_f = float('inf')
    
    for alpha_test in np.linspace(0.1, 2.0, 20):
        # Create NS equations with this alpha
        scaling = SelfSimilarScaling(alpha=alpha_test, beta=alpha_test)
        ns = RescaledNSEquations(scaling, nu=1e-3, N=32)
        
        # Compute residual
        tau = torch.tensor(10.0)
        R = ns.residual(U0, tau)
        f_norm = torch.sqrt((R**2).sum() * (2*np.pi/32)**3).item()
        
        print(f"  α = {alpha_test:.2f}: ||F|| = {f_norm:.4e}")
        
        if f_norm < best_f:
            best_f = f_norm
            best_alpha = alpha_test
    
    print()
    print(f"★ Best α = {best_alpha:.2f} with ||F|| = {best_f:.4e}")
    print()
    
    # Now try refinement with best alpha
    print("="*60)
    print(f"REFINEMENT WITH OPTIMAL α = {best_alpha:.2f}")
    print("="*60)
    
    refiner = NewtonRefiner(N=32, nu=1e-3)
    refiner.scaling = SelfSimilarScaling(alpha=best_alpha, beta=best_alpha)
    refiner.ns = RescaledNSEquations(refiner.scaling, nu=1e-3, N=32)
    
    result = refiner.refine(U0, NewtonConfig(max_iter=100, verbose=True))
    
    print()
    print(f"Final ||F||: {result.final_residual:.6e}")
    print(f"Improvement: {best_f / result.final_residual:.1f}x")
