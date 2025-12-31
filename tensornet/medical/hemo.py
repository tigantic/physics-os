"""
Hemodynamic Blood Flow Simulation

Phase 11: The Surgical Pre-Flight

Blood is NOT water. It's a Non-Newtonian fluid that gets THINNER
(less viscous) the faster it flows. This is called "Shear Thinning"
and it's critical for understanding cardiovascular disease.

The Physics:
- Carreau-Yasuda Model: μ(γ̇) = μ∞ + (μ₀ - μ∞)[1 + (λγ̇)²]^((n-1)/2)
- Stenosis (Plaque): Narrowing increases velocity (Venturi)
- Wall Shear Stress: τ = μ(∂u/∂r) - High τ = Rupture risk

Clinical Applications:
- Pre-surgical planning for angioplasty
- Aneurysm rupture risk assessment
- Stent placement optimization

Reference: Blood rheology follows the Carreau model with:
- μ₀ = 0.056 Pa·s (zero shear viscosity)
- μ∞ = 0.00345 Pa·s (infinite shear viscosity)
- λ = 3.313 s (relaxation time)
- n = 0.3568 (power law index)
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import numpy as np


@dataclass
class StenosisReport:
    """
    Clinical report on stenosis hemodynamics.
    """
    stenosis_severity: float     # Percent blockage (0-100%)
    peak_velocity: float         # cm/s at narrowest point
    wall_shear_stress: float     # Pa (Pascal)
    pressure_drop: float         # mmHg across stenosis
    
    rupture_risk: str            # LOW / MODERATE / HIGH / CRITICAL
    flow_reserve: float          # Ratio of max to resting flow
    
    recommendation: str
    
    def __str__(self) -> str:
        return (
            f"[STENOSIS REPORT]\n"
            f"  Blockage: {self.stenosis_severity:.0f}%\n"
            f"  Peak Velocity: {self.peak_velocity:.1f} cm/s\n"
            f"  Wall Shear Stress: {self.wall_shear_stress:.2f} Pa\n"
            f"  Pressure Drop: {self.pressure_drop:.1f} mmHg\n"
            f"  Rupture Risk: {self.rupture_risk}\n"
            f"  Flow Reserve: {self.flow_reserve:.2f}\n"
            f"  Recommendation: {self.recommendation}"
        )


class ArterySimulation:
    """
    3D Artery Blood Flow Simulation with Stenosis.
    
    Models blood flow through a cylindrical artery with a 
    calcified plaque blockage. Uses simplified Lattice Boltzmann
    approach with non-Newtonian viscosity correction.
    
    Coordinate System:
    - X: Axial (flow direction)
    - Y, Z: Radial (cross-section)
    """
    
    def __init__(
        self,
        length: int = 100,
        radius: int = 10,
        stenosis_severity: float = 0.7,
        stenosis_position: float = 0.5,
        stenosis_length: float = 0.2,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize artery simulation.
        
        Args:
            length: Artery length in grid units (mm)
            radius: Artery radius in grid units (mm)
            stenosis_severity: Percent blockage (0-1, where 0.7 = 70%)
            stenosis_position: Position along length (0-1)
            stenosis_length: Length of stenosis region (0-1)
            device: Torch device
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.L = length
        self.R = radius
        self.diameter = radius * 2
        self.stenosis_severity = stenosis_severity
        
        # Grid dimensions
        self.shape = (length, self.diameter, self.diameter)
        
        # Generate artery geometry
        self.geometry = self._create_geometry(
            stenosis_severity,
            stenosis_position,
            stenosis_length,
        )
        
        # Flow fields
        self.velocity = torch.zeros((3, *self.shape), device=self.device)
        self.pressure = torch.zeros(self.shape, device=self.device)
        
        # Blood properties (SI units scaled for simulation)
        self.rho = 1060.0  # kg/m³ (blood density)
        self.mu_0 = 0.056  # Pa·s (zero shear viscosity)
        self.mu_inf = 0.00345  # Pa·s (infinite shear viscosity)
        
        # Compute statistics
        open_cells = (self.geometry > 0.5).sum().item()
        total_cells = np.prod(self.shape)
        
        print(f"[MEDICAL] Artery Simulation Initialized")
        print(f"[MEDICAL] Dimensions: {length}mm × ⌀{self.diameter}mm")
        print(f"[MEDICAL] Stenosis: {stenosis_severity*100:.0f}% blockage")
        print(f"[MEDICAL] Open lumen: {100*open_cells/total_cells:.1f}%")
        print(f"[MEDICAL] Device: {self.device}")
    
    def _create_geometry(
        self,
        severity: float,
        position: float,
        length_frac: float,
    ) -> torch.Tensor:
        """
        Create artery geometry with stenosis.
        
        Returns tensor where:
        - 1.0 = Blood (open lumen)
        - 0.0 = Wall / Plaque
        """
        geometry = torch.ones(self.shape, device=self.device)
        
        # Create coordinate grids for cross-section
        y = torch.arange(self.diameter, device=self.device, dtype=torch.float32)
        z = torch.arange(self.diameter, device=self.device, dtype=torch.float32)
        yy, zz = torch.meshgrid(y, z, indexing='ij')
        
        center = self.R  # Center of cross-section
        dist_from_center = torch.sqrt((yy - center)**2 + (zz - center)**2)
        
        # Stenosis parameters
        stenosis_center = int(position * self.L)
        stenosis_half_length = int(length_frac * self.L / 2)
        
        # Create the stenosis (tapered blockage)
        for x in range(self.L):
            # Distance from stenosis center
            dist_from_stenosis = abs(x - stenosis_center)
            
            # Gaussian taper for smooth stenosis
            if dist_from_stenosis < stenosis_half_length * 2:
                # Severity tapers from max at center to 0 at edges
                local_severity = severity * np.exp(
                    -0.5 * (dist_from_stenosis / stenosis_half_length)**2
                )
            else:
                local_severity = 0.0
            
            # Effective radius at this slice
            effective_radius = self.R * (1.0 - local_severity)
            
            # Mark cells outside effective radius as wall
            wall_mask = dist_from_center > effective_radius
            geometry[x][wall_mask] = 0.0
        
        return geometry
    
    def compute_shear_rate(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        Compute shear rate magnitude from velocity field.
        
        γ̇ = √(2 * Σᵢⱼ Sᵢⱼ²) where S is the strain rate tensor
        
        Simplified: γ̇ ≈ |∂u/∂r| (radial velocity gradient)
        """
        # Compute velocity gradients
        du_dy = torch.gradient(velocity[0], dim=1)[0]
        du_dz = torch.gradient(velocity[0], dim=2)[0]
        
        # Shear rate magnitude
        shear_rate = torch.sqrt(du_dy**2 + du_dz**2 + 1e-10)
        
        return shear_rate
    
    def compute_viscosity(self, shear_rate: torch.Tensor) -> torch.Tensor:
        """
        Compute non-Newtonian viscosity using Carreau-Yasuda model.
        
        μ(γ̇) = μ∞ + (μ₀ - μ∞) * [1 + (λγ̇)²]^((n-1)/2)
        
        Blood is shear-thinning: viscosity decreases with shear rate.
        """
        # Carreau-Yasuda parameters for blood
        lambda_param = 3.313  # Relaxation time (s)
        n = 0.3568  # Power law index
        
        # Carreau model
        term = 1.0 + (lambda_param * shear_rate)**2
        viscosity = self.mu_inf + (self.mu_0 - self.mu_inf) * torch.pow(term, (n - 1) / 2)
        
        return viscosity
    
    def solve_blood_flow(
        self,
        inlet_velocity: float = 50.0,
        steps: int = 200,
        verbose: bool = True,
    ) -> StenosisReport:
        """
        Simulate blood flow through the artery.
        
        Uses simplified advection-diffusion with non-Newtonian
        viscosity correction. Full simulation would use LBM.
        
        Args:
            inlet_velocity: Blood velocity at inlet (cm/s)
            steps: Number of simulation steps
            verbose: Print progress
            
        Returns:
            StenosisReport with hemodynamic analysis
        """
        if verbose:
            print(f"\n[MEDICAL] Simulating blood flow...")
            print(f"[MEDICAL] Inlet velocity: {inlet_velocity} cm/s")
            print(f"[MEDICAL] Cardiac cycles: {steps}")
        
        # Initialize velocity field
        self.velocity = torch.zeros((3, *self.shape), device=self.device)
        
        # Inlet boundary condition (parabolic profile approximation)
        inlet_profile = self.geometry[0] * inlet_velocity
        self.velocity[0, 0:5, :, :] = inlet_profile
        
        # Track peak values
        peak_velocity = 0.0
        peak_shear = 0.0
        
        for t in range(steps):
            # 1. Advection (blood moves forward)
            self.velocity = torch.roll(self.velocity, shifts=1, dims=1)
            
            # Reinforce inlet
            self.velocity[0, 0:3, :, :] = inlet_profile
            
            # 2. Apply wall boundary (no-slip)
            wall_mask = self.geometry < 0.5
            self.velocity[:, wall_mask] = 0.0
            
            # 3. Conservation of mass (Venturi effect)
            # In stenosis, velocity must increase to conserve flow
            # Q = A₁v₁ = A₂v₂
            for x in range(self.L):
                open_area = (self.geometry[x] > 0.5).sum().item()
                inlet_area = (self.geometry[0] > 0.5).sum().item()
                
                if open_area > 0 and inlet_area > 0:
                    # Velocity scales inversely with area
                    area_ratio = inlet_area / open_area
                    self.velocity[0, x, :, :] *= min(area_ratio, 5.0)  # Cap at 5x
            
            # 4. Compute shear rate and stress
            shear_rate = self.compute_shear_rate(self.velocity)
            viscosity = self.compute_viscosity(shear_rate)
            
            # Wall shear stress: τ = μ * γ̇
            wall_shear = viscosity * shear_rate
            
            # Track peaks
            current_peak_v = self.velocity.abs().max().item()
            current_peak_shear = wall_shear.max().item()
            
            if current_peak_v > peak_velocity:
                peak_velocity = current_peak_v
            if current_peak_shear > peak_shear:
                peak_shear = current_peak_shear
            
            # Progress
            if verbose and t % 50 == 0:
                print(f"   Heartbeat {t:3d}: Peak Vel={current_peak_v:.1f} cm/s | "
                      f"Wall Shear={current_peak_shear:.2f} Pa")
                
                if current_peak_shear > 15.0:
                    print(f"   ⚠️  [ALERT] HIGH RUPTURE RISK at stenosis!")
        
        # Generate clinical report
        report = self._generate_report(peak_velocity, peak_shear)
        
        if verbose:
            print()
            print(report)
        
        return report
    
    def _generate_report(
        self,
        peak_velocity: float,
        peak_shear: float,
    ) -> StenosisReport:
        """Generate clinical stenosis report."""
        
        # Pressure drop estimation (simplified Bernoulli)
        # ΔP = 0.5 * ρ * (v₂² - v₁²)
        inlet_v = 50.0  # cm/s
        delta_v_sq = (peak_velocity**2 - inlet_v**2) / 10000  # Convert to m²/s²
        pressure_drop_pa = 0.5 * self.rho * delta_v_sq
        pressure_drop_mmhg = pressure_drop_pa * 0.00750062  # Pa to mmHg
        
        # Flow reserve (ratio of stressed to resting flow)
        flow_reserve = 1.0 / (1.0 + self.stenosis_severity)
        
        # Risk classification based on wall shear stress
        if peak_shear > 20.0:
            rupture_risk = "⛔ CRITICAL"
            recommendation = "IMMEDIATE INTERVENTION. Schedule angioplasty or bypass."
        elif peak_shear > 15.0:
            rupture_risk = "🔴 HIGH"
            recommendation = "Surgical consult recommended. Consider stent placement."
        elif peak_shear > 10.0:
            rupture_risk = "🟡 MODERATE"
            recommendation = "Medical management. Monitor with follow-up imaging."
        else:
            rupture_risk = "🟢 LOW"
            recommendation = "Lifestyle modification. Annual monitoring."
        
        return StenosisReport(
            stenosis_severity=self.stenosis_severity * 100,
            peak_velocity=peak_velocity,
            wall_shear_stress=peak_shear,
            pressure_drop=abs(pressure_drop_mmhg),
            rupture_risk=rupture_risk,
            flow_reserve=flow_reserve,
            recommendation=recommendation,
        )


def run_preop_analysis(stenosis_severity: float = 0.7) -> StenosisReport:
    """
    Run pre-operative hemodynamic analysis.
    
    This is what a surgeon would see before deciding on intervention.
    """
    print("=" * 70)
    print("SURGICAL PRE-FLIGHT: Hemodynamic Analysis")
    print("=" * 70)
    
    artery = ArterySimulation(
        length=100,
        radius=10,
        stenosis_severity=stenosis_severity,
    )
    
    report = artery.solve_blood_flow(
        inlet_velocity=50.0,  # Resting cardiac output
        steps=200,
    )
    
    return report
