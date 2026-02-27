"""
TigantiCFD Grid Convergence & Verification
==========================================

Richardson extrapolation and Grid Convergence Index (GCI) for
mesh independence verification per ASME V&V 20-2009.

Capabilities:
- T2.01: Richardson extrapolation for observed order of convergence
- T2.02: GCI uncertainty quantification
- T2.03: Asymptotic range verification
- T2.04: Automated mesh refinement recommendation

Reference:
    Roache, P.J. (1998). "Verification and Validation in Computational 
    Science and Engineering." Hermosa Publishers.
    
    ASME V&V 20-2009: "Standard for Verification and Validation in 
    Computational Fluid Dynamics and Heat Transfer."
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class GridLevel:
    """Represents a single grid refinement level."""
    name: str
    cells: int
    h: float              # Characteristic cell size [m]
    result: float         # Solution quantity of interest
    wall_time: float = 0  # Computation time [s]


@dataclass
class GCIResult:
    """Grid Convergence Index analysis results."""
    # Grid refinement ratios
    r21: float  # h2/h1 (coarse/medium)
    r32: float  # h3/h2 (medium/fine)
    
    # Observed order of convergence
    p_observed: float
    
    # Extrapolated exact solution
    phi_extrapolated: float
    
    # Relative errors
    e21: float  # Between coarse and medium
    e32: float  # Between medium and fine
    
    # GCI values (uncertainty bounds)
    GCI_fine: float      # GCI for finest grid (as %)
    GCI_medium: float    # GCI for medium grid (as %)
    
    # Asymptotic range indicator
    asymptotic_ratio: float
    in_asymptotic_range: bool
    
    # Recommendation
    mesh_adequate: bool
    recommendation: str


class RichardsonExtrapolation:
    """
    Richardson extrapolation for grid convergence analysis.
    
    Uses three grid levels to compute:
    1. Observed order of convergence (p)
    2. Extrapolated exact solution (φ_ext)
    3. Grid Convergence Index (GCI)
    """
    
    def __init__(
        self,
        grids: List[GridLevel],
        expected_order: float = 2.0,
        safety_factor: float = 1.25
    ):
        """
        Initialize Richardson extrapolation analysis.
        
        Args:
            grids: List of GridLevel objects (coarse to fine)
            expected_order: Theoretical order of convergence
            safety_factor: GCI safety factor (1.25 for 3+ grids)
        """
        if len(grids) < 3:
            raise ValueError("Need at least 3 grid levels for Richardson extrapolation")
        
        # Sort by cell size (coarse to fine)
        self.grids = sorted(grids, key=lambda g: -g.h)
        self.expected_order = expected_order
        self.Fs = safety_factor
        
    def analyze(self) -> GCIResult:
        """
        Perform complete Richardson extrapolation and GCI analysis.
        
        Returns:
            GCIResult with extrapolated values and uncertainty
        """
        g1, g2, g3 = self.grids[0], self.grids[1], self.grids[2]  # coarse, medium, fine
        
        # Grid refinement ratios
        r21 = g1.h / g2.h
        r32 = g2.h / g3.h
        
        # Solution changes
        eps32 = g3.result - g2.result
        eps21 = g2.result - g1.result
        
        # Handle special cases
        if abs(eps32) < 1e-15 or abs(eps21) < 1e-15:
            # Solutions are essentially converged
            return GCIResult(
                r21=r21, r32=r32,
                p_observed=self.expected_order,
                phi_extrapolated=g3.result,
                e21=0.0, e32=0.0,
                GCI_fine=0.0, GCI_medium=0.0,
                asymptotic_ratio=1.0,
                in_asymptotic_range=True,
                mesh_adequate=True,
                recommendation="Grid is converged - solutions match within machine precision"
            )
        
        # Sign check for monotonic convergence
        s = np.sign(eps32 / eps21)
        if s < 0:
            # Oscillatory convergence - use absolute values
            eps32 = abs(eps32)
            eps21 = abs(eps21)
        
        # Observed order of convergence (iterative solution)
        p = self._compute_observed_order(eps21, eps32, r21, r32)
        
        # Extrapolated exact solution (Richardson)
        phi_ext = g3.result + eps32 / (r32**p - 1)
        
        # Relative errors
        e21 = abs((g2.result - g1.result) / g2.result) if g2.result != 0 else 0
        e32 = abs((g3.result - g2.result) / g3.result) if g3.result != 0 else 0
        
        # Grid Convergence Index
        GCI_fine = self.Fs * abs(e32) / (r32**p - 1) * 100
        GCI_medium = self.Fs * abs(e21) / (r21**p - 1) * 100
        
        # Asymptotic range check: GCI_medium / (r^p * GCI_fine) ≈ 1
        if GCI_fine > 0:
            asymptotic_ratio = GCI_medium / (r32**p * GCI_fine)
        else:
            asymptotic_ratio = 1.0
        
        in_asymptotic = 0.9 < asymptotic_ratio < 1.1
        
        # Mesh adequacy assessment
        mesh_ok = GCI_fine < 5.0  # Less than 5% uncertainty
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            GCI_fine, p, in_asymptotic, r32
        )
        
        return GCIResult(
            r21=r21,
            r32=r32,
            p_observed=p,
            phi_extrapolated=phi_ext,
            e21=e21,
            e32=e32,
            GCI_fine=GCI_fine,
            GCI_medium=GCI_medium,
            asymptotic_ratio=asymptotic_ratio,
            in_asymptotic_range=in_asymptotic,
            mesh_adequate=mesh_ok,
            recommendation=recommendation
        )
    
    def _compute_observed_order(
        self,
        eps21: float,
        eps32: float,
        r21: float,
        r32: float,
        tol: float = 1e-6,
        max_iter: int = 50
    ) -> float:
        """
        Compute observed order of convergence using fixed-point iteration.
        
        Solves: p = ln(eps21/eps32) / ln(r) for general r21 ≠ r32
        """
        if abs(eps21) < 1e-15:
            return self.expected_order
        
        ratio = abs(eps21 / eps32)
        
        if r21 == r32:
            # Simple case: constant refinement ratio
            if ratio > 0:
                return np.log(ratio) / np.log(r21)
            return self.expected_order
        
        # General case: iterative solution
        p = self.expected_order  # Initial guess
        
        for _ in range(max_iter):
            # Fixed-point iteration
            q = np.log((r21**p - 1) / (r32**p - 1))
            p_new = np.log(ratio) / np.log(r21) + q / np.log(r21)
            
            if abs(p_new - p) < tol:
                break
            p = p_new
        
        # Clamp to reasonable bounds
        return np.clip(p, 0.5, 4.0)
    
    def _generate_recommendation(
        self,
        gci: float,
        p: float,
        in_asymptotic: bool,
        r: float
    ) -> str:
        """Generate mesh refinement recommendation."""
        if gci < 1.0:
            return "✓ Excellent grid resolution. Numerical uncertainty < 1%."
        elif gci < 2.0:
            return "✓ Good grid resolution. Consider this mesh for production runs."
        elif gci < 5.0:
            if in_asymptotic:
                return "✓ Acceptable resolution in asymptotic range. Fine for engineering accuracy."
            else:
                return "⚠ Marginal resolution. Refine mesh by factor of {:.1f} for better accuracy.".format(r)
        else:
            refine_factor = (gci / 2.0) ** (1/p)
            return "✗ Insufficient resolution. Refine mesh by factor of {:.1f}×.".format(refine_factor)


def run_grid_study(
    results: List[Tuple[int, float, float]],
    quantity_name: str = "Result"
) -> GCIResult:
    """
    Convenience function for quick grid convergence study.
    
    Args:
        results: List of (cell_count, cell_size, result_value) tuples
        quantity_name: Name of quantity for reporting
        
    Returns:
        GCIResult with analysis
        
    Example:
        >>> results = [
        ...     (10000, 0.1, 25.3),   # Coarse
        ...     (40000, 0.05, 25.8),  # Medium
        ...     (160000, 0.025, 25.95) # Fine
        ... ]
        >>> gci = run_grid_study(results, "Temperature")
        >>> print(f"Extrapolated: {gci.phi_extrapolated:.2f}")
        >>> print(f"Uncertainty: ±{gci.GCI_fine:.1f}%")
    """
    grids = [
        GridLevel(
            name=f"Grid-{i+1}",
            cells=cells,
            h=h,
            result=result
        )
        for i, (cells, h, result) in enumerate(results)
    ]
    
    extrap = RichardsonExtrapolation(grids)
    return extrap.analyze()


def print_gci_report(result: GCIResult, quantity_name: str = "φ") -> str:
    """Generate formatted GCI report."""
    lines = [
        "=" * 60,
        "GRID CONVERGENCE INDEX (GCI) REPORT",
        "ASME V&V 20-2009 Compliant",
        "=" * 60,
        "",
        f"Refinement Ratios:",
        f"  r₂₁ (coarse/medium): {result.r21:.3f}",
        f"  r₃₂ (medium/fine):   {result.r32:.3f}",
        "",
        f"Observed Order of Convergence: p = {result.p_observed:.3f}",
        "",
        f"Extrapolated {quantity_name}: {result.phi_extrapolated:.6f}",
        "",
        f"Relative Errors:",
        f"  e₂₁: {result.e21*100:.4f}%",
        f"  e₃₂: {result.e32*100:.4f}%",
        "",
        f"Grid Convergence Index:",
        f"  GCI_fine:   {result.GCI_fine:.2f}%",
        f"  GCI_medium: {result.GCI_medium:.2f}%",
        "",
        f"Asymptotic Range Check:",
        f"  Ratio: {result.asymptotic_ratio:.3f} (ideal ≈ 1.0)",
        f"  Status: {'✓ In asymptotic range' if result.in_asymptotic_range else '⚠ Not yet asymptotic'}",
        "",
        f"Assessment: {result.recommendation}",
        "=" * 60,
    ]
    return "\n".join(lines)
