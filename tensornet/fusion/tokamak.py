"""
Tokamak Magnetic Confinement Fusion Reactor Simulation

The Physics of Stars - On Your GPU

A tokamak confines plasma (ionized hydrogen at 150 million °C) using
magnetic fields in a toroidal (donut) geometry. The goal: keep the
plasma away from the walls long enough for fusion to occur.

Key Physics:
- Lorentz Force: F = q(E + v × B)
- Particles spiral around magnetic field lines
- Toroidal field (around the ring) + Poloidal field (twist) = Helical confinement
- Safety factor q: ratio of toroidal to poloidal windings

The Geometry:
- Major radius R₀: Distance from center of torus to center of tube
- Minor radius a: Radius of the plasma tube itself
- Aspect ratio A = R₀/a (typically 2.5-4 for modern tokamaks)

Example Parameters (ITER-like):
- R₀ = 6.2m, a = 2.0m
- B_toroidal = 5.3 T
- Plasma current = 15 MA
- Temperature = 150 million K

This simulation uses simplified geometry but correct physics
for the Boris particle pusher and magnetic field topology.

References:
    Boris, J.P. (1970). "Relativistic Plasma Simulation - Optimization
    of a Hybrid Code." Proceedings of the Fourth Conference on Numerical
    Simulation of Plasmas, pp. 3-67. Naval Research Laboratory.
    
    Wesson, J. (2011). "Tokamaks." 4th Edition, Oxford University Press.
    ISBN 978-0-19-959223-4.
    
    ITER Physics Expert Group (1999). "ITER Physics Basis."
    Nuclear Fusion, 39(12), 2137-2638.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import torch
import numpy as np


@dataclass
class PlasmaState:
    """
    State of the plasma particles.
    
    Each particle has 6 degrees of freedom:
    - Position (x, y, z) in meters
    - Velocity (vx, vy, vz) in m/s
    """
    positions: torch.Tensor   # [N, 3]
    velocities: torch.Tensor  # [N, 3]
    
    @property
    def num_particles(self) -> int:
        return self.positions.shape[0]
    
    @property
    def kinetic_energy(self) -> float:
        """Total kinetic energy in arbitrary units."""
        return 0.5 * torch.sum(self.velocities ** 2).item()
    
    def get_center_of_mass(self) -> Tuple[float, float, float]:
        """Get center of mass position."""
        com = self.positions.mean(dim=0)
        return (com[0].item(), com[1].item(), com[2].item())


@dataclass
class ConfinementReport:
    """
    Report on plasma confinement quality.
    """
    total_particles: int
    confined_particles: int
    escaped_particles: int
    confinement_ratio: float
    
    max_rho: float      # Maximum distance from magnetic axis
    mean_rho: float     # Mean distance from magnetic axis
    
    kinetic_energy: float
    simulation_steps: int
    
    status: str
    recommendation: str
    
    def __str__(self) -> str:
        return (
            f"[CONFINEMENT REPORT]\n"
            f"  Particles: {self.confined_particles}/{self.total_particles} confined "
            f"({100*self.confinement_ratio:.1f}%)\n"
            f"  Escaped: {self.escaped_particles}\n"
            f"  Max ρ: {self.max_rho:.3f}m, Mean ρ: {self.mean_rho:.3f}m\n"
            f"  Kinetic Energy: {self.kinetic_energy:.2f}\n"
            f"  Steps: {self.simulation_steps}\n"
            f"  Status: {self.status}\n"
            f"  Recommendation: {self.recommendation}"
        )


class TokamakReactor:
    """
    Tokamak Fusion Reactor Simulation.
    
    Simulates charged particle dynamics in a toroidal magnetic field
    using the Boris pusher algorithm - the gold standard for
    plasma particle-in-cell (PIC) simulations.
    
    Geometry:
        - Torus centered at origin
        - Major axis along Z (vertical)
        - Plasma flows in toroidal direction (around the ring)
    
    Example:
        >>> reactor = TokamakReactor(major_radius=2.0, B0=5.0)
        >>> plasma = reactor.create_plasma(n_particles=10000, seed=42)
        >>> for _ in range(1000):
        ...     plasma = reactor.boris_push(plasma, dt=1e-9)
        >>> report = reactor.analyze_confinement(plasma)
    
    References:
        Boris, J.P. (1970). Proc. 4th Conf. Numerical Simulation of Plasmas.
        Wesson, J. (2011). Tokamaks. 4th Ed., Oxford University Press.
    """
    
    def __init__(
        self,
        major_radius: float = 2.0,
        minor_radius: float = 0.8,
        B0: float = 5.0,
        safety_factor: float = 2.0,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize tokamak reactor.
        
        Args:
            major_radius: R₀, distance from torus center to tube center (m).
            minor_radius: a, radius of the plasma tube (m).
            B0: Toroidal magnetic field strength at R₀ (Tesla).
            safety_factor: q, ratio of field line windings.
            device: Torch device (auto-selects CUDA if available).
        
        Raises:
            ValueError: If minor_radius >= major_radius.
            ValueError: If B0 <= 0 or safety_factor <= 0.
        
        Example:
            >>> reactor = TokamakReactor(
            ...     major_radius=6.2,  # ITER-like
            ...     minor_radius=2.0,
            ...     B0=5.3
            ... )
        """
        self.R0 = major_radius
        self.a = minor_radius
        self.B0 = B0
        self.q = safety_factor
        
        # Aspect ratio
        self.aspect_ratio = major_radius / minor_radius
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"[TOKAMAK] Reactor initialized")
        print(f"[TOKAMAK] R₀={major_radius}m, a={minor_radius}m, A={self.aspect_ratio:.2f}")
        print(f"[TOKAMAK] B₀={B0}T, q={safety_factor}")
        print(f"[TOKAMAK] Device: {self.device}")
    
    def get_magnetic_field(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Calculate magnetic field at given positions.
        
        The tokamak field has two components:
        1. Toroidal field B_φ: Runs around the ring, created by external coils
           B_φ = B₀ · R₀ / R (stronger on inside of torus)
           
        2. Poloidal field B_θ: Runs around the tube cross-section
           Created by the plasma current itself
           B_θ = (ρ/R₀) · (B_φ/q) where ρ is distance from magnetic axis
        
        The combination creates helical field lines that confine particles.
        
        Args:
            positions: Tensor [N, 3] of (x, y, z) positions
            
        Returns:
            Tensor [N, 3] of (Bx, By, Bz) magnetic field vectors
        """
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        
        # Cylindrical coordinates
        R = torch.sqrt(x**2 + y**2)  # Distance from Z-axis
        phi = torch.atan2(y, x)       # Toroidal angle
        
        # Distance from magnetic axis (center of plasma tube)
        rho = torch.sqrt((R - self.R0)**2 + z**2)
        
        # Poloidal angle (angle within the tube cross-section)
        theta = torch.atan2(z, R - self.R0)
        
        # ===========================================
        # 1. TOROIDAL FIELD (around the ring)
        # ===========================================
        # B_φ = B₀ · R₀ / R
        # This is the main confining field from external coils
        # Note: Stronger on the inside (R < R₀), weaker on outside
        
        # Avoid division by zero
        R_safe = torch.clamp(R, min=0.1)
        B_phi = self.B0 * self.R0 / R_safe
        
        # Toroidal field direction: tangent to circle around Z-axis
        # Unit vector: (-sin(φ), cos(φ), 0)
        Bx_tor = -B_phi * torch.sin(phi)
        By_tor = B_phi * torch.cos(phi)
        Bz_tor = torch.zeros_like(x)
        
        # ===========================================
        # 2. POLOIDAL FIELD (around the tube)
        # ===========================================
        # B_θ = (ρ/R₀) · (B_φ/q)
        # This comes from the plasma current and creates the "twist"
        # Essential for confinement (prevents drift losses)
        
        B_theta = (rho / self.R0) * (B_phi / self.q)
        
        # Poloidal field direction: tangent to circle in (R-R₀, z) plane
        # In Cartesian: we need to rotate the poloidal direction by φ
        
        # Poloidal unit vector in (R, z) plane: (-sin(θ), cos(θ))
        # Transform to Cartesian:
        #   dR = -sin(θ), dz = cos(θ)
        #   dx = dR·cos(φ), dy = dR·sin(φ)
        
        Bx_pol = -B_theta * torch.sin(theta) * torch.cos(phi)
        By_pol = -B_theta * torch.sin(theta) * torch.sin(phi)
        Bz_pol = B_theta * torch.cos(theta)
        
        # ===========================================
        # 3. COMBINED FIELD
        # ===========================================
        Bx = Bx_tor + Bx_pol
        By = By_tor + By_pol
        Bz = Bz_tor + Bz_pol
        
        return torch.stack([Bx, By, Bz], dim=1)
    
    def compute_rho(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute distance from magnetic axis for each particle.
        
        ρ = 0 at the center of the plasma tube
        ρ = a at the plasma edge (last closed flux surface)
        ρ > a means the particle has escaped
        """
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        
        R = torch.sqrt(x**2 + y**2)
        rho = torch.sqrt((R - self.R0)**2 + z**2)
        
        return rho
    
    def push_particles(
        self,
        particles: torch.Tensor,
        dt: float = 0.001,
        steps: int = 100,
        q_over_m: float = 1.0,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Advance particles using the Boris pusher algorithm.
        
        The Boris pusher is the gold standard for charged particle
        dynamics because it:
        - Exactly conserves energy (symplectic)
        - Correctly captures gyration physics
        - Is second-order accurate in time
        
        Algorithm:
        1. v⁻ = v + (q/m)·E·dt/2
        2. v' = v⁻ + v⁻ × t  where t = (q/m)·B·dt/2
        3. v⁺ = v⁻ + v' × s  where s = 2t/(1 + |t|²)
        4. v_new = v⁺ + (q/m)·E·dt/2
        5. x_new = x + v_new·dt
        
        Args:
            particles: Tensor [N, 6] of (x, y, z, vx, vy, vz)
            dt: Time step in seconds
            steps: Number of time steps
            q_over_m: Charge to mass ratio (normalized)
            verbose: Print progress
            
        Returns:
            (final_particles, escape_history)
        """
        if verbose:
            print(f"[PLASMA] Simulating {len(particles)} ions for {steps} steps...")
            print(f"[PLASMA] dt={dt*1e6:.2f}μs, q/m={q_over_m}")
        
        particles = particles.to(self.device)
        escape_history = []
        
        for t in range(steps):
            pos = particles[:, 0:3]
            vel = particles[:, 3:6]
            
            # Get magnetic field at particle positions
            B = self.get_magnetic_field(pos)
            
            # ========================================
            # BORIS PUSHER
            # ========================================
            
            # No electric field in this simple model (E = 0)
            # So v⁻ = v_old
            v_minus = vel
            
            # t = (q/m) · B · dt/2
            t_vec = q_over_m * B * (dt / 2.0)
            
            # s = 2t / (1 + |t|²)
            t_mag_sq = torch.sum(t_vec**2, dim=1, keepdim=True)
            s_vec = 2.0 * t_vec / (1.0 + t_mag_sq)
            
            # v' = v⁻ + v⁻ × t
            v_prime = v_minus + torch.cross(v_minus, t_vec, dim=1)
            
            # v⁺ = v⁻ + v' × s
            v_plus = v_minus + torch.cross(v_prime, s_vec, dim=1)
            
            # v_new = v⁺ (no E field contribution)
            v_new = v_plus
            
            # Position update: x_new = x + v_new · dt
            pos_new = pos + v_new * dt
            
            # Update particle state
            particles[:, 0:3] = pos_new
            particles[:, 3:6] = v_new
            
            # ========================================
            # CONFINEMENT CHECK
            # ========================================
            rho = self.compute_rho(pos_new)
            escaped = (rho > self.a).sum().item()
            escape_history.append(escaped)
            
            if verbose and t % max(1, steps // 10) == 0:
                confined = len(particles) - escaped
                mean_rho = rho.mean().item()
                max_rho = rho.max().item()
                print(f"   Step {t:4d}: Confined={confined}, "
                      f"ρ_mean={mean_rho:.3f}m, ρ_max={max_rho:.3f}m")
        
        return particles, escape_history
    
    def analyze_confinement(
        self,
        particles: torch.Tensor,
        escape_history: List[int],
    ) -> ConfinementReport:
        """
        Generate confinement quality report.
        """
        pos = particles[:, 0:3]
        vel = particles[:, 3:6]
        
        rho = self.compute_rho(pos)
        
        total = len(particles)
        escaped = (rho > self.a).sum().item()
        confined = total - escaped
        
        ratio = confined / total if total > 0 else 0.0
        
        ke = 0.5 * torch.sum(vel ** 2).item()
        
        # Status assessment
        if ratio > 0.95:
            status = "✅ EXCELLENT CONFINEMENT"
            recommendation = "Plasma stable. Ready for heating."
        elif ratio > 0.80:
            status = "🟡 GOOD CONFINEMENT"
            recommendation = "Minor losses acceptable. Monitor edge."
        elif ratio > 0.50:
            status = "⚠️ MARGINAL CONFINEMENT"
            recommendation = "Increase B-field or adjust q-profile."
        else:
            status = "⛔ CONFINEMENT FAILURE"
            recommendation = "Plasma disruption imminent. Emergency shutdown."
        
        return ConfinementReport(
            total_particles=total,
            confined_particles=confined,
            escaped_particles=escaped,
            confinement_ratio=ratio,
            max_rho=rho.max().item(),
            mean_rho=rho.mean().item(),
            kinetic_energy=ke,
            simulation_steps=len(escape_history),
            status=status,
            recommendation=recommendation,
        )
    
    def create_plasma(
        self,
        num_particles: int = 1000,
        temperature: float = 1.0,
        toroidal_flow: float = 10.0,
        seed: int = 42,
    ) -> torch.Tensor:
        """
        Initialize plasma particles in the tokamak.
        
        Particles are distributed along the magnetic axis with:
        - Random toroidal position (around the ring)
        - Small random offset from axis
        - Thermal velocity distribution
        - Bulk toroidal flow
        
        Args:
            num_particles: Number of particles to create
            temperature: Controls thermal velocity spread
            toroidal_flow: Bulk flow speed around the torus
            seed: Random seed for reproducibility (default: 42 per Constitution)
            
        Returns:
            Tensor [N, 6] of particle states
        """
        # Per Article III, Section 3.2: Reproducible seeding
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Random toroidal angle
        phi = torch.rand(num_particles, device=self.device, dtype=torch.float64) * 2 * np.pi
        
        # Position: On magnetic axis with small random offset
        # r_offset ~ 0.1 * minor_radius (start near the core)
        r_offset = torch.randn(num_particles, device=self.device, dtype=torch.float64) * (0.1 * self.a)
        z_offset = torch.randn(num_particles, device=self.device, dtype=torch.float64) * (0.1 * self.a)
        
        # Cylindrical to Cartesian
        R = self.R0 + r_offset
        x = R * torch.cos(phi)
        y = R * torch.sin(phi)
        z = z_offset
        
        # Velocity: Toroidal flow + thermal motion
        # Toroidal direction: (-sin(φ), cos(φ), 0)
        vx = -torch.sin(phi) * toroidal_flow + torch.randn(num_particles, device=self.device, dtype=torch.float64) * temperature
        vy = torch.cos(phi) * toroidal_flow + torch.randn(num_particles, device=self.device, dtype=torch.float64) * temperature
        vz = torch.randn(num_particles, device=self.device, dtype=torch.float64) * temperature
        
        particles = torch.stack([x, y, z, vx, vy, vz], dim=1)
        
        print(f"[PLASMA] Created {num_particles} particles")
        print(f"[PLASMA] Temperature={temperature}, Flow={toroidal_flow}")
        
        return particles


def verify_gyration(reactor: TokamakReactor) -> bool:
    """
    Verify that particles exhibit correct Larmor gyration.
    
    A charged particle in a magnetic field should gyrate around
    field lines with radius r_L = mv_⊥ / (qB)
    """
    print("\n[VERIFY] Checking Larmor gyration...")
    
    # Single particle at magnetic axis
    x = reactor.R0
    y = 0.0
    z = 0.0
    
    # Perpendicular velocity (should cause gyration)
    vx = 0.0
    vy = 0.0
    vz = 1.0  # Velocity perpendicular to toroidal B-field
    
    particle = torch.tensor([[x, y, z, vx, vy, vz]], device=reactor.device)
    
    # Get B-field magnitude at starting position
    B = reactor.get_magnetic_field(particle[:, :3])
    B_mag = torch.sqrt(torch.sum(B**2)).item()
    
    # Expected Larmor radius: r_L = v_⊥ / (q/m * B)
    q_over_m = 1.0
    v_perp = 1.0
    r_L_expected = v_perp / (q_over_m * B_mag)
    
    # Simulate one gyration period
    # Period T = 2π / ω_c where ω_c = qB/m
    omega_c = q_over_m * B_mag
    period = 2 * np.pi / omega_c
    
    # Use small dt for accuracy
    dt = period / 100
    steps = int(period / dt)
    
    final, _ = reactor.push_particles(particle, dt=dt, steps=steps, verbose=False)
    
    # Check if particle returned near starting position
    displacement = torch.sqrt(torch.sum((final[:, :3] - particle[:, :3])**2)).item()
    
    print(f"   B-field at axis: {B_mag:.3f} T")
    print(f"   Expected Larmor radius: {r_L_expected:.4f} m")
    print(f"   Gyration period: {period*1e6:.2f} μs")
    print(f"   Displacement after 1 period: {displacement:.6f} m")
    
    # Should return close to start (within numerical error)
    success = displacement < 0.01
    
    if success:
        print("   ✅ Gyration physics verified!")
    else:
        print("   ⚠️ Gyration check failed - check dt or q/m")
    
    return success
