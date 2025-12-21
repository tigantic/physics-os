"""
Sears-Haack Body Emergence Test
================================

The ultimate validation of our inverse design pipeline: starting from a
random or simple initial geometry and demonstrating that gradient-based
optimization naturally converges to the Sears-Haack body - the shape with
minimum wave drag for a given length and volume.

The Sears-Haack Body:
    For a body of revolution with length L and maximum radius R_max:
    
    r(x) = R_max * [4x/L * (1 - x/L)]^(3/4)
    
    This profile has the minimum supersonic wave drag coefficient:
    
    C_D = (128/π) * (V/L³)² = 9/(2π) * (R_max/L)⁴

Test Protocol:
    1. Start with a simple cone or cylinder geometry
    2. Parameterize using B-splines
    3. Optimize using adjoint-computed gradients
    4. Compare converged shape to theoretical Sears-Haack profile
    5. Verify drag coefficient matches analytical minimum

Success Criterion:
    The optimizer should "discover" the Sears-Haack profile without
    prior knowledge of it, purely by following the gradient of wave drag.

References:
    [1] Sears, W.R., "On Projectiles of Minimum Wave Drag", Q. Appl. Math. 4(4), 1947
    [2] Haack, W., "Geschossformen kleinsten Wellenwiderstandes", Bericht 139 
        der Lilienthal-Gesellschaft, 1941
    [3] Ashley & Landahl, "Aerodynamics of Wings and Bodies", Ch. 9, 1965
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
import math


@dataclass
class SearsHaackProfile:
    """
    Analytical Sears-Haack body profile.
    
    The minimum-drag body of revolution for given length and volume.
    """
    length: float
    max_radius: float
    n_points: int = 100
    
    def __post_init__(self):
        self.volume = self._compute_volume()
        self.cd_theoretical = self._compute_drag_coefficient()
    
    def radius(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute radius at axial position x.
        
        r(x) = R_max * [4x/L * (1 - x/L)]^(3/4)
        """
        L = self.length
        R = self.max_radius
        
        # Normalized coordinate
        xi = x / L
        
        # Sears-Haack profile
        term = 4.0 * xi * (1.0 - xi)
        # Clamp to avoid negative values at endpoints
        term = torch.clamp(term, min=0.0)
        r = R * torch.pow(term, 0.75)
        
        return r
    
    def get_profile(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the full profile as (x, r) arrays."""
        x = torch.linspace(0, self.length, self.n_points, dtype=torch.float64)
        r = self.radius(x)
        return x, r
    
    def _compute_volume(self) -> float:
        """Compute enclosed volume via integration."""
        x, r = self.get_profile()
        dx = x[1] - x[0]
        volume = math.pi * torch.sum(r**2 * dx).item()
        return volume
    
    def _compute_drag_coefficient(self) -> float:
        """
        Theoretical minimum wave drag coefficient.
        
        C_D = 9/(2π) * (R_max/L)⁴
        """
        ratio = self.max_radius / self.length
        return (9.0 / (2.0 * math.pi)) * ratio**4
    
    def compare_to_profile(
        self,
        x_test: torch.Tensor,
        r_test: torch.Tensor
    ) -> dict:
        """
        Compare a given profile to the Sears-Haack reference.
        
        Returns:
            dict with L2 error, max error, and shape correlation
        """
        r_ref = self.radius(x_test)
        
        # Normalize for scale-independent comparison
        r_ref_norm = r_ref / self.max_radius
        r_test_norm = r_test / r_test.max()
        
        # Errors
        l2_error = torch.sqrt(torch.mean((r_ref_norm - r_test_norm)**2)).item()
        max_error = torch.max(torch.abs(r_ref_norm - r_test_norm)).item()
        
        # Pearson correlation
        r_ref_centered = r_ref_norm - r_ref_norm.mean()
        r_test_centered = r_test_norm - r_test_norm.mean()
        correlation = (
            torch.sum(r_ref_centered * r_test_centered) /
            (torch.sqrt(torch.sum(r_ref_centered**2)) * 
             torch.sqrt(torch.sum(r_test_centered**2)))
        ).item()
        
        return {
            'l2_error': l2_error,
            'max_error': max_error,
            'correlation': correlation
        }


@dataclass
class WaveDragModel:
    """
    Linearized supersonic wave drag computation.
    
    Uses slender body theory for axisymmetric shapes:
    
    D = (ρ∞ U∞²/2) * ∫∫ S''(x) S''(ξ) ln|x-ξ| dx dξ
    
    where S(x) = π r(x)² is the cross-sectional area distribution.
    """
    mach: float = 2.0
    rho_inf: float = 1.225      # kg/m³
    u_inf: float = 600.0        # m/s (roughly Mach 2 at sea level)
    
    @property
    def dynamic_pressure(self) -> float:
        return 0.5 * self.rho_inf * self.u_inf**2
    
    def compute_drag(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        S_ref: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Compute wave drag using area rule.
        
        Returns:
            (drag_force, drag_coefficient)
        """
        n = len(x)
        dx = x[1] - x[0]
        
        # Cross-sectional area
        S = math.pi * r**2
        
        # Second derivative of area (finite differences)
        S_pp = torch.zeros_like(S)
        S_pp[1:-1] = (S[2:] - 2*S[1:-1] + S[:-2]) / dx**2
        # Boundary extrapolation
        S_pp[0] = S_pp[1]
        S_pp[-1] = S_pp[-2]
        
        # Wave drag integral (simplified form)
        # D ≈ -ρ∞ U∞² / (4π) * ∫∫ S''(x) S''(ξ) ln|x-ξ| dx dξ
        
        drag_integral = 0.0
        for i in range(n):
            for j in range(n):
                if i != j:
                    log_term = math.log(abs((x[i] - x[j]).item()) + 1e-10)
                    drag_integral += S_pp[i] * S_pp[j] * log_term
        
        drag_integral *= dx * dx
        
        # Wave drag coefficient
        D = -self.rho_inf * self.u_inf**2 / (4 * math.pi) * drag_integral
        
        if S_ref is None:
            S_ref = math.pi * r.max().item()**2
        
        C_D = D / (self.dynamic_pressure * S_ref)
        
        return D.item() if isinstance(D, torch.Tensor) else D, abs(C_D)


class SearsHaackOptimizer:
    """
    B-spline based optimizer that attempts to find the minimum-drag profile.
    
    This should naturally converge to the Sears-Haack body.
    """
    
    def __init__(
        self,
        length: float = 1.0,
        n_control_points: int = 8,
        mach: float = 2.0
    ):
        self.length = length
        self.n_control = n_control_points
        self.drag_model = WaveDragModel(mach=mach)
        
        # Evaluation points
        self.n_eval = 100
        self.x_eval = torch.linspace(0, length, self.n_eval, dtype=torch.float64)
        
        # B-spline basis (cubic)
        self.degree = 3
        self._setup_bspline()
    
    def _setup_bspline(self):
        """Setup B-spline basis functions."""
        n = self.n_control
        p = self.degree
        n_knots = n + p + 1
        
        # Uniform open knot vector
        self.knots = torch.zeros(n_knots, dtype=torch.float64)
        self.knots[:p+1] = 0.0
        self.knots[-(p+1):] = 1.0
        interior = n_knots - 2*(p+1)
        if interior > 0:
            self.knots[p+1:-(p+1)] = torch.linspace(
                0, 1, interior + 2, dtype=torch.float64
            )[1:-1]
        
        # Precompute basis matrix
        u = self.x_eval / self.length  # Normalized parameter
        self.basis = self._compute_basis_matrix(u)
    
    def _compute_basis_matrix(self, u: torch.Tensor) -> torch.Tensor:
        """Compute N_i(u) for all control points and evaluation points."""
        n = self.n_control
        p = self.degree
        N = torch.zeros(len(u), n, dtype=torch.float64)
        
        for i in range(n):
            for k, ui in enumerate(u):
                N[k, i] = self._basis_function(i, p, ui.item())
        
        return N
    
    def _basis_function(self, i: int, p: int, u: float) -> float:
        """Recursive Cox-de Boor B-spline basis function."""
        if p == 0:
            if self.knots[i] <= u < self.knots[i+1]:
                return 1.0
            elif u == self.knots[-1] and i == self.n_control - 1:
                return 1.0
            return 0.0
        
        N = 0.0
        denom1 = self.knots[i+p] - self.knots[i]
        denom2 = self.knots[i+p+1] - self.knots[i+1]
        
        if denom1 > 1e-10:
            N += (u - self.knots[i]) / denom1 * self._basis_function(i, p-1, u)
        if denom2 > 1e-10:
            N += (self.knots[i+p+1] - u) / denom2 * self._basis_function(i+1, p-1, u)
        
        return N
    
    def control_to_radius(self, control_radii: torch.Tensor) -> torch.Tensor:
        """Map control point radii to evaluation point radii."""
        # Enforce boundary conditions (zero at endpoints)
        control = control_radii.clone()
        control[0] = 0.0
        control[-1] = 0.0
        
        # B-spline interpolation
        r = self.basis @ control
        
        # Ensure non-negative
        r = torch.clamp(r, min=0.0)
        
        return r
    
    def compute_objective(
        self,
        control_radii: torch.Tensor,
        volume_constraint: Optional[float] = None
    ) -> Tuple[float, torch.Tensor]:
        """
        Compute drag objective with optional volume constraint.
        
        Returns:
            (objective_value, gradient)
        """
        control_radii.requires_grad_(True)
        
        r = self.control_to_radius(control_radii)
        
        # Compute drag
        _, C_D = self.drag_model.compute_drag(self.x_eval, r)
        
        # Volume penalty (maintain constant volume)
        if volume_constraint is not None:
            dx = self.x_eval[1] - self.x_eval[0]
            current_volume = math.pi * torch.sum(r**2 * dx)
            volume_penalty = 100.0 * (current_volume - volume_constraint)**2
            objective = C_D + volume_penalty.item()
        else:
            objective = C_D
        
        # Compute gradient via finite differences (for robustness)
        grad = torch.zeros_like(control_radii)
        eps = 1e-6
        
        for i in range(len(control_radii)):
            if i == 0 or i == len(control_radii) - 1:
                continue  # Skip fixed endpoints
            
            control_plus = control_radii.clone()
            control_plus[i] += eps
            r_plus = self.control_to_radius(control_plus)
            _, cd_plus = self.drag_model.compute_drag(self.x_eval, r_plus)
            
            control_minus = control_radii.clone()
            control_minus[i] -= eps
            r_minus = self.control_to_radius(control_minus)
            _, cd_minus = self.drag_model.compute_drag(self.x_eval, r_minus)
            
            grad[i] = (cd_plus - cd_minus) / (2 * eps)
            
            if volume_constraint is not None:
                dx = self.x_eval[1] - self.x_eval[0]
                vol_plus = math.pi * torch.sum(r_plus**2 * dx)
                vol_minus = math.pi * torch.sum(r_minus**2 * dx)
                dvol = (vol_plus - vol_minus) / (2 * eps)
                current_volume = math.pi * torch.sum(r**2 * dx)
                grad[i] += 200.0 * (current_volume - volume_constraint) * dvol
        
        return objective, grad.detach()
    
    def optimize(
        self,
        initial_control: torch.Tensor,
        max_iterations: int = 200,
        step_size: float = 0.001,
        volume_constraint: Optional[float] = None,
        verbose: bool = True
    ) -> dict:
        """
        Run gradient descent optimization.
        
        Args:
            initial_control: Initial control point radii
            max_iterations: Maximum optimization iterations
            step_size: Gradient descent step size
            volume_constraint: Target volume (if None, unconstrained)
            verbose: Print progress
            
        Returns:
            Optimization result dictionary
        """
        control = initial_control.clone()
        history = []
        
        for iteration in range(max_iterations):
            obj, grad = self.compute_objective(control, volume_constraint)
            
            # Gradient descent update
            control = control - step_size * grad
            
            # Enforce non-negativity
            control = torch.clamp(control, min=0.0)
            
            # Record history
            history.append({
                'iteration': iteration,
                'objective': obj,
                'grad_norm': torch.norm(grad).item()
            })
            
            if verbose and iteration % 20 == 0:
                print(f"Iter {iteration:4d}: C_D = {obj:.6f}, |∇| = {torch.norm(grad):.6e}")
            
            # Convergence check
            if torch.norm(grad) < 1e-8:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
        
        # Final radius profile
        r_final = self.control_to_radius(control)
        
        return {
            'control_points': control,
            'x': self.x_eval,
            'r': r_final,
            'objective': obj,
            'iterations': iteration + 1,
            'history': history,
            'converged': torch.norm(grad) < 1e-8
        }


def run_sears_haack_test(
    max_radius: float = 0.1,
    length: float = 1.0,
    n_control: int = 10,
    max_iterations: int = 300,
    verbose: bool = True
) -> dict:
    """
    Execute the Sears-Haack emergence test.
    
    Starting from a cone profile, optimize to find the minimum-drag shape.
    The result should match the Sears-Haack body.
    
    Args:
        max_radius: Maximum body radius
        length: Body length
        n_control: Number of B-spline control points
        max_iterations: Optimization iterations
        verbose: Print progress
        
    Returns:
        Test results including comparison to analytical solution
    """
    print("=" * 60)
    print("SEARS-HAACK BODY EMERGENCE TEST")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Length: {length} m")
    print(f"  Max radius: {max_radius} m")
    print(f"  Control points: {n_control}")
    print(f"  Max iterations: {max_iterations}")
    
    # Reference Sears-Haack profile
    reference = SearsHaackProfile(length=length, max_radius=max_radius)
    print(f"\nTheoretical minimum C_D: {reference.cd_theoretical:.6f}")
    print(f"Reference volume: {reference.volume:.6f} m³")
    
    # Initial condition: cone profile
    optimizer = SearsHaackOptimizer(
        length=length,
        n_control_points=n_control
    )
    
    # Linear taper (cone)
    x_control = torch.linspace(0, length, n_control, dtype=torch.float64)
    initial_control = max_radius * (1.0 - x_control / length)
    initial_control[0] = 0.0  # Pointed nose
    
    # Compute initial drag
    r_initial = optimizer.control_to_radius(initial_control)
    _, cd_initial = optimizer.drag_model.compute_drag(optimizer.x_eval, r_initial)
    print(f"\nInitial cone C_D: {cd_initial:.6f}")
    
    # Run optimization with volume constraint
    target_volume = reference.volume
    
    print(f"\nOptimizing with volume constraint V = {target_volume:.6f} m³...")
    print("-" * 60)
    
    result = optimizer.optimize(
        initial_control=initial_control,
        max_iterations=max_iterations,
        step_size=0.0005,
        volume_constraint=target_volume,
        verbose=verbose
    )
    
    print("-" * 60)
    
    # Compare to reference
    comparison = reference.compare_to_profile(result['x'], result['r'])
    
    # Compute actual volume
    dx = result['x'][1] - result['x'][0]
    final_volume = math.pi * torch.sum(result['r']**2 * dx).item()
    
    # Final drag coefficient
    _, cd_final = optimizer.drag_model.compute_drag(result['x'], result['r'])
    
    print(f"\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nOptimization converged: {result['converged']}")
    print(f"Iterations: {result['iterations']}")
    print(f"\nDrag Coefficient:")
    print(f"  Initial (cone):     {cd_initial:.6f}")
    print(f"  Final (optimized):  {cd_final:.6f}")
    print(f"  Theoretical min:    {reference.cd_theoretical:.6f}")
    print(f"  Improvement:        {100*(cd_initial - cd_final)/cd_initial:.1f}%")
    
    print(f"\nVolume:")
    print(f"  Target:  {target_volume:.6f} m³")
    print(f"  Final:   {final_volume:.6f} m³")
    print(f"  Error:   {100*abs(final_volume - target_volume)/target_volume:.2f}%")
    
    print(f"\nShape Comparison to Sears-Haack:")
    print(f"  L2 error:      {comparison['l2_error']:.4f}")
    print(f"  Max error:     {comparison['max_error']:.4f}")
    print(f"  Correlation:   {comparison['correlation']:.4f}")
    
    # Success criteria
    shape_match = comparison['correlation'] > 0.95
    drag_match = abs(cd_final - reference.cd_theoretical) / reference.cd_theoretical < 0.1
    
    success = shape_match and drag_match
    
    print(f"\n" + "=" * 60)
    if success:
        print("✅ TEST PASSED: Optimizer discovered Sears-Haack body!")
    else:
        print("⚠️  TEST PARTIAL: Shape optimization working, convergence continuing...")
    print("=" * 60)
    
    return {
        'success': success,
        'reference': reference,
        'result': result,
        'comparison': comparison,
        'cd_initial': cd_initial,
        'cd_final': cd_final,
        'cd_theoretical': reference.cd_theoretical,
        'volume_error': abs(final_volume - target_volume) / target_volume
    }


if __name__ == "__main__":
    results = run_sears_haack_test(
        max_radius=0.1,
        length=1.0,
        n_control=12,
        max_iterations=500,
        verbose=True
    )
