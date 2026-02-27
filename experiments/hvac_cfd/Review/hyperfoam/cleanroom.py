"""
TigantiCFD Cleanroom & Particle Tracking
=========================================

ISO 14644-1 compliant cleanroom simulation with Lagrangian particle tracking.

Capabilities:
- T3.01: ISO 14644 class mapping and compliance
- T3.02: Particle tracking (Lagrangian DPM)
- T3.03: Recovery time calculation (100:1 decay)
- T3.04: Contamination risk assessment
- T3.05: Air change effectiveness

Reference:
    ISO 14644-1:2015 "Cleanrooms and associated controlled environments"
    IEST-RP-CC012: "Considerations in Cleanroom Design"
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


class ISOClass(Enum):
    """ISO 14644-1 cleanroom classifications."""
    ISO_1 = 1
    ISO_2 = 2
    ISO_3 = 3
    ISO_4 = 4
    ISO_5 = 5   # ~Class 100 (semiconductor fab)
    ISO_6 = 6   # ~Class 1000
    ISO_7 = 7   # ~Class 10000 (pharma)
    ISO_8 = 8   # ~Class 100000
    ISO_9 = 9   # Normal room air


@dataclass
class ISO14644Limits:
    """
    ISO 14644-1:2015 particle concentration limits.
    
    Maximum particles/m³ for each particle size at each ISO class.
    """
    # Particle size thresholds [μm]
    sizes: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 1.0, 5.0)
    
    # Maximum concentration [particles/m³] for each ISO class
    # Based on formula: Cn = 10^N × (0.1/D)^2.08
    limits: Dict[ISOClass, Dict[float, int]] = field(default_factory=lambda: {
        ISOClass.ISO_1: {0.1: 10, 0.2: 2},
        ISOClass.ISO_2: {0.1: 100, 0.2: 24, 0.3: 10},
        ISOClass.ISO_3: {0.1: 1000, 0.2: 237, 0.3: 102, 0.5: 35},
        ISOClass.ISO_4: {0.1: 10000, 0.2: 2370, 0.3: 1020, 0.5: 352, 1.0: 83},
        ISOClass.ISO_5: {0.1: 100000, 0.2: 23700, 0.3: 10200, 0.5: 3520, 1.0: 832, 5.0: 29},
        ISOClass.ISO_6: {0.2: 237000, 0.3: 102000, 0.5: 35200, 1.0: 8320, 5.0: 293},
        ISOClass.ISO_7: {0.5: 352000, 1.0: 83200, 5.0: 2930},
        ISOClass.ISO_8: {0.5: 3520000, 1.0: 832000, 5.0: 29300},
        ISOClass.ISO_9: {0.5: 35200000, 1.0: 8320000, 5.0: 293000},
    })
    
    def get_limit(self, iso_class: ISOClass, particle_size: float) -> Optional[int]:
        """Get concentration limit for given class and particle size."""
        class_limits = self.limits.get(iso_class, {})
        return class_limits.get(particle_size)
    
    def check_compliance(
        self,
        iso_class: ISOClass,
        concentrations: Dict[float, float]
    ) -> Tuple[bool, Dict[float, Tuple[float, int, bool]]]:
        """
        Check if measured concentrations meet ISO class requirements.
        
        Args:
            iso_class: Target ISO class
            concentrations: Dict of {particle_size: measured_concentration}
            
        Returns:
            (compliant, details) where details maps size to (measured, limit, pass)
        """
        class_limits = self.limits.get(iso_class, {})
        details = {}
        compliant = True
        
        for size, measured in concentrations.items():
            limit = class_limits.get(size)
            if limit is not None:
                passes = measured <= limit
                details[size] = (measured, limit, passes)
                if not passes:
                    compliant = False
        
        return compliant, details


@dataclass(slots=True)
class Particle:
    """Single particle for Lagrangian tracking."""
    x: float      # Position [m]
    y: float
    z: float
    vx: float     # Velocity [m/s]
    vy: float
    vz: float
    diameter: float  # [μm]
    density: float   # [kg/m³]
    age: float = 0   # Time since injection [s]
    active: bool = True


@dataclass(slots=True)
class ParticleSource:
    """Particle injection source."""
    x: float
    y: float
    z: float
    rate: float          # Particles per second
    diameter_mean: float # Mean diameter [μm]
    diameter_std: float  # Std dev [μm]
    density: float = 1000  # kg/m³ (water droplet default)


class LagrangianTracker:
    """
    Lagrangian Discrete Phase Model (DPM) for particle tracking.
    
    Tracks individual particles through the flow field accounting for:
    - Drag force (Stokes regime)
    - Gravity
    - Brownian motion (for sub-micron particles)
    - Wall deposition
    """
    
    def __init__(
        self,
        domain_size: Tuple[float, float, float],
        air_viscosity: float = 1.81e-5,
        air_density: float = 1.2,
        gravity: float = 9.81
    ):
        self.Lx, self.Ly, self.Lz = domain_size
        self.mu = air_viscosity
        self.rho_air = air_density
        self.g = gravity
        
        self.particles: List[Particle] = []
        self.deposited: List[Particle] = []
        self.escaped: List[Particle] = []
    
    def inject_particles(self, source: ParticleSource, dt: float) -> None:
        """Inject particles from source over time step dt."""
        n_inject = int(source.rate * dt)
        remainder = source.rate * dt - n_inject
        if np.random.random() < remainder:
            n_inject += 1
        
        for _ in range(n_inject):
            # Random diameter from distribution
            d = np.random.normal(source.diameter_mean, source.diameter_std)
            d = max(0.01, d)  # Minimum 0.01 μm
            
            # Small random offset from source
            offset = 0.01
            p = Particle(
                x=source.x + np.random.uniform(-offset, offset),
                y=source.y + np.random.uniform(-offset, offset),
                z=source.z + np.random.uniform(-offset, offset),
                vx=0, vy=0, vz=0,
                diameter=d,
                density=source.density
            )
            self.particles.append(p)
    
    def step(
        self,
        u_field: np.ndarray,
        v_field: np.ndarray,
        w_field: np.ndarray,
        grid_spacing: Tuple[float, float, float],
        dt: float
    ) -> None:
        """
        Advance all particles one time step.
        
        Uses one-way coupling (particles don't affect flow).
        """
        dx, dy, dz = grid_spacing
        nx, ny, nz = u_field.shape
        
        still_active = []
        
        for p in self.particles:
            if not p.active:
                continue
            
            # Get local fluid velocity (trilinear interpolation)
            u_f, v_f, w_f = self._interpolate_velocity(
                p.x, p.y, p.z,
                u_field, v_field, w_field,
                dx, dy, dz
            )
            
            # Particle properties
            d_m = p.diameter * 1e-6  # Convert to meters
            rho_p = p.density
            
            # Relaxation time (Stokes drag)
            tau_p = rho_p * d_m**2 / (18 * self.mu)
            
            # Drag force: F_D = (u_f - u_p) / tau_p
            ax = (u_f - p.vx) / tau_p
            ay = (v_f - p.vy) / tau_p - self.g  # Gravity in -y
            az = (w_f - p.vz) / tau_p
            
            # Brownian motion for small particles (< 1 μm)
            if p.diameter < 1.0:
                kB = 1.38e-23
                T = 293  # K
                D_B = kB * T * self._cunningham(d_m) / (3 * np.pi * self.mu * d_m)
                sigma = np.sqrt(2 * D_B / dt)
                ax += np.random.normal(0, sigma) / dt
                ay += np.random.normal(0, sigma) / dt
                az += np.random.normal(0, sigma) / dt
            
            # Update velocity (explicit Euler)
            p.vx += ax * dt
            p.vy += ay * dt
            p.vz += az * dt
            
            # Update position
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.z += p.vz * dt
            
            p.age += dt
            
            # Check boundaries
            if p.x < 0 or p.x > self.Lx or p.z < 0 or p.z > self.Lz:
                # Escaped through outlet
                p.active = False
                self.escaped.append(p)
            elif p.y < 0 or p.y > self.Ly:
                # Deposited on floor/ceiling
                p.active = False
                self.deposited.append(p)
            else:
                still_active.append(p)
        
        self.particles = still_active
    
    def _interpolate_velocity(
        self,
        x: float, y: float, z: float,
        u: np.ndarray, v: np.ndarray, w: np.ndarray,
        dx: float, dy: float, dz: float
    ) -> Tuple[float, float, float]:
        """Trilinear interpolation of velocity at particle position."""
        nx, ny, nz = u.shape
        
        # Grid indices
        i = int(x / dx)
        j = int(y / dy)
        k = int(z / dz)
        
        # Clamp to valid range
        i = max(0, min(i, nx - 2))
        j = max(0, min(j, ny - 2))
        k = max(0, min(k, nz - 2))
        
        # Local coordinates
        xd = (x - i * dx) / dx
        yd = (y - j * dy) / dy
        zd = (z - k * dz) / dz
        
        # Trilinear interpolation
        def interp(f):
            c000 = f[i, j, k]
            c100 = f[i+1, j, k]
            c010 = f[i, j+1, k]
            c001 = f[i, j, k+1]
            c110 = f[i+1, j+1, k]
            c101 = f[i+1, j, k+1]
            c011 = f[i, j+1, k+1]
            c111 = f[i+1, j+1, k+1]
            
            c00 = c000 * (1 - xd) + c100 * xd
            c01 = c001 * (1 - xd) + c101 * xd
            c10 = c010 * (1 - xd) + c110 * xd
            c11 = c011 * (1 - xd) + c111 * xd
            
            c0 = c00 * (1 - yd) + c10 * yd
            c1 = c01 * (1 - yd) + c11 * yd
            
            return c0 * (1 - zd) + c1 * zd
        
        return interp(u), interp(v), interp(w)
    
    def _cunningham(self, d: float) -> float:
        """Cunningham slip correction for small particles."""
        # Mean free path of air
        lambda_air = 6.5e-8  # m
        Kn = 2 * lambda_air / d  # Knudsen number
        return 1 + Kn * (1.257 + 0.4 * np.exp(-1.1 / Kn))
    
    def get_concentration(
        self,
        grid_shape: Tuple[int, int, int],
        domain_size: Tuple[float, float, float],
        size_bins: List[float] = [0.3, 0.5, 1.0, 5.0]
    ) -> Dict[float, np.ndarray]:
        """
        Compute particle concentration field by size bin.
        
        Returns dict mapping particle size to concentration [particles/m³]
        """
        dx = domain_size[0] / grid_shape[0]
        dy = domain_size[1] / grid_shape[1]
        dz = domain_size[2] / grid_shape[2]
        cell_volume = dx * dy * dz
        
        concentrations = {size: np.zeros(grid_shape) for size in size_bins}
        
        for p in self.particles:
            # Find which bin
            for size in size_bins:
                if p.diameter >= size:
                    # Grid cell
                    i = min(int(p.x / dx), grid_shape[0] - 1)
                    j = min(int(p.y / dy), grid_shape[1] - 1)
                    k = min(int(p.z / dz), grid_shape[2] - 1)
                    
                    concentrations[size][i, j, k] += 1 / cell_volume
                    break
        
        return concentrations


@dataclass
class RecoveryTimeResult:
    """Results of cleanroom recovery time analysis."""
    initial_concentration: float
    final_concentration: float
    target_ratio: float
    recovery_time: float  # seconds
    air_changes_required: float
    meets_standard: bool


class RecoveryAnalysis:
    """
    Cleanroom recovery time calculation.
    
    Per IEST-RP-CC012 and ISO 14644-3, recovery time is the time
    required for particle concentration to decay by a specified
    ratio (typically 100:1) after a contamination event.
    """
    
    def __init__(
        self,
        room_volume: float,          # m³
        supply_flow_rate: float,     # m³/s
        filter_efficiency: float = 0.9999,  # HEPA
        mixing_factor: float = 0.7   # Ventilation effectiveness
    ):
        self.V = room_volume
        self.Q = supply_flow_rate
        self.eta = filter_efficiency
        self.k = mixing_factor
        
        # Effective air change rate
        self.ACH = 3600 * self.Q / self.V  # per hour
        
    def compute_recovery_time(
        self,
        initial_conc: float,
        target_ratio: float = 100.0,
        background_conc: float = 0.0
    ) -> RecoveryTimeResult:
        """
        Compute time to achieve target concentration reduction.
        
        Uses exponential decay model:
            C(t) = C_bg + (C_0 - C_bg) × exp(-k × ACH × t / 3600)
        
        Args:
            initial_conc: Initial particle concentration
            target_ratio: Required reduction ratio (e.g., 100:1)
            background_conc: Background/ambient concentration
            
        Returns:
            RecoveryTimeResult with analysis
        """
        C_0 = initial_conc
        C_target = C_0 / target_ratio
        
        # Time constant
        tau = 3600 / (self.k * self.ACH * self.eta)  # seconds
        
        # Time for exponential decay (ignoring background)
        if C_target > background_conc:
            t_recovery = tau * np.log((C_0 - background_conc) / 
                                       (C_target - background_conc))
        else:
            # Target below background - use pure decay
            t_recovery = tau * np.log(target_ratio)
        
        # Air changes required
        n_ach = t_recovery * self.ACH / 3600
        
        # Standard: ISO 14644-3 suggests 20 minutes for 100:1
        meets_standard = t_recovery <= 1200  # 20 minutes
        
        return RecoveryTimeResult(
            initial_concentration=C_0,
            final_concentration=C_target,
            target_ratio=target_ratio,
            recovery_time=t_recovery,
            air_changes_required=n_ach,
            meets_standard=meets_standard
        )
    
    def required_ach_for_recovery(
        self,
        target_ratio: float = 100.0,
        max_time: float = 1200.0  # 20 minutes
    ) -> float:
        """
        Compute required ACH to achieve target recovery within time limit.
        
        Returns:
            Required air changes per hour
        """
        # From: t = (3600 / k×ACH×η) × ln(ratio)
        # ACH = 3600 × ln(ratio) / (k × η × t)
        
        required_ach = 3600 * np.log(target_ratio) / (
            self.k * self.eta * max_time
        )
        
        return required_ach


def classify_room(
    measured_concentrations: Dict[float, float]
) -> Tuple[ISOClass, Dict[float, Tuple[float, int, str]]]:
    """
    Determine ISO class from measured particle concentrations.
    
    Args:
        measured_concentrations: Dict of {particle_size_um: concentration}
        
    Returns:
        (best_class, details) where details show compliance at each size
    """
    limits = ISO14644Limits()
    
    # Check each class from cleanest to dirtiest
    for iso_class in ISOClass:
        compliant, details = limits.check_compliance(iso_class, measured_concentrations)
        if compliant:
            return iso_class, {
                size: (meas, lim, "PASS" if p else "FAIL")
                for size, (meas, lim, p) in details.items()
            }
    
    return ISOClass.ISO_9, {}


def print_cleanroom_report(
    iso_class: ISOClass,
    concentrations: Dict[float, float],
    recovery: Optional[RecoveryTimeResult] = None
) -> str:
    """Generate formatted cleanroom compliance report."""
    limits = ISO14644Limits()
    compliant, details = limits.check_compliance(iso_class, concentrations)
    
    lines = [
        "=" * 60,
        "ISO 14644-1 CLEANROOM COMPLIANCE REPORT",
        "=" * 60,
        "",
        f"Target Classification: {iso_class.name}",
        "",
        "Particle Concentrations (particles/m³):",
        "-" * 40,
        f"{'Size (μm)':<12} {'Measured':<15} {'Limit':<15} {'Status':<10}",
        "-" * 40,
    ]
    
    for size, (meas, lim, passed) in sorted(details.items()):
        status = "✓ PASS" if passed else "✗ FAIL"
        lines.append(f"{size:<12.1f} {meas:<15.0f} {lim:<15} {status}")
    
    lines.extend([
        "-" * 40,
        "",
        f"Overall Compliance: {'✓ COMPLIANT' if compliant else '✗ NON-COMPLIANT'}",
    ])
    
    if recovery:
        lines.extend([
            "",
            "Recovery Time Analysis:",
            f"  Initial concentration: {recovery.initial_concentration:.0f} p/m³",
            f"  Target ratio: {recovery.target_ratio:.0f}:1",
            f"  Recovery time: {recovery.recovery_time:.1f} s ({recovery.recovery_time/60:.1f} min)",
            f"  Air changes required: {recovery.air_changes_required:.1f}",
            f"  Meets 20-min standard: {'✓ YES' if recovery.meets_standard else '✗ NO'}",
        ])
    
    lines.append("=" * 60)
    
    return "\n".join(lines)
