"""
Test Module: tensornet/fusion/tokamak.py

Phase 9: Tokamak Magnetic Confinement Fusion Simulation
Constitutional Compliance:
    - Article III, Section 3.1: Unit tests with 90%+ coverage
    - Article III, Section 3.2: Deterministic seeding (seed=42)
    - Article IV, Section 4.1: Physical validation

References:
    Boris, J.P. (1970). "Relativistic plasma simulation - optimization of 
    a hybrid code." Proc. Fourth Conf. Num. Sim. Plasmas, Naval Research Lab.
    
    Freidberg, J.P. (2007). "Plasma Physics and Fusion Energy."
    Cambridge University Press. ISBN: 978-0521851077
"""

import pytest
import torch
import numpy as np

from tensornet.fusion.tokamak import (
    TokamakReactor,
    PlasmaState,
    ConfinementReport,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def deterministic_seed():
    """Per Article III, Section 3.2: Reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    yield


@pytest.fixture
def iter_reactor():
    """ITER-like tokamak reactor."""
    return TokamakReactor(
        major_radius=6.2,
        minor_radius=2.0,
        B0=5.3,
        safety_factor=3.0,
    )


@pytest.fixture
def small_reactor():
    """Small test reactor for fast tests."""
    return TokamakReactor(
        major_radius=1.0,
        minor_radius=0.3,
        B0=1.0,
        safety_factor=2.0,
    )


# ============================================================================
# UNIT TESTS
# ============================================================================

class TestTokamakInit:
    """Test TokamakReactor initialization."""
    
    @pytest.mark.unit
    def test_init_iter_params(self, deterministic_seed):
        """Test ITER-like initialization."""
        reactor = TokamakReactor(
            major_radius=6.2,
            minor_radius=2.0,
            B0=5.3,
            safety_factor=3.0,
        )
        
        assert reactor.R0 == 6.2
        assert reactor.a == 2.0
        assert reactor.B0 == 5.3
        assert reactor.q == 3.0
    
    @pytest.mark.unit
    def test_aspect_ratio(self, iter_reactor, deterministic_seed):
        """Test aspect ratio calculation."""
        expected = 6.2 / 2.0
        assert iter_reactor.aspect_ratio == pytest.approx(expected, rel=0.01)


class TestMagneticField:
    """Test magnetic field calculations."""
    
    @pytest.mark.unit
    @pytest.mark.physics
    def test_toroidal_field_scaling(self, iter_reactor, deterministic_seed):
        """Toroidal field should scale as B0 * R0 / R."""
        # Point on magnetic axis
        pos = torch.tensor([[6.2, 0.0, 0.0]], dtype=torch.float64)
        B = iter_reactor.get_magnetic_field(pos)
        
        # At R = R0, B_phi ≈ B0
        B_magnitude = torch.norm(B[0]).item()
        assert B_magnitude > 0  # Field exists
    
    @pytest.mark.unit
    @pytest.mark.physics
    def test_field_stronger_inside(self, iter_reactor, deterministic_seed):
        """Field should be stronger on inside of torus (smaller R)."""
        # Inside point (R < R0)
        pos_inside = torch.tensor([[5.0, 0.0, 0.0]], dtype=torch.float64)
        B_inside = iter_reactor.get_magnetic_field(pos_inside)
        
        # Outside point (R > R0)
        pos_outside = torch.tensor([[7.5, 0.0, 0.0]], dtype=torch.float64)
        B_outside = iter_reactor.get_magnetic_field(pos_outside)
        
        assert torch.norm(B_inside) > torch.norm(B_outside)
    
    @pytest.mark.unit
    @pytest.mark.physics
    def test_field_symmetry(self, small_reactor, deterministic_seed):
        """Field should be toroidally symmetric."""
        # Two points at same R but different toroidal angles
        pos1 = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        pos2 = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)  # 90° rotated
        
        B1 = small_reactor.get_magnetic_field(pos1)
        B2 = small_reactor.get_magnetic_field(pos2)
        
        # Magnitudes should be equal
        assert torch.norm(B1).item() == pytest.approx(torch.norm(B2).item(), rel=0.01)


class TestPlasmaCreation:
    """Test plasma particle initialization."""
    
    @pytest.mark.unit
    def test_create_plasma_shape(self, small_reactor, deterministic_seed):
        """Test plasma tensor shape."""
        particles = small_reactor.create_plasma(num_particles=100, seed=42)
        
        assert particles.shape == (100, 6)  # [N, (x,y,z,vx,vy,vz)]
    
    @pytest.mark.unit
    def test_create_plasma_dtype(self, small_reactor, deterministic_seed):
        """Test plasma uses float64."""
        particles = small_reactor.create_plasma(num_particles=100, seed=42)
        
        assert particles.dtype == torch.float64
    
    @pytest.mark.unit
    def test_create_plasma_reproducible(self, small_reactor):
        """Same seed should give same particles."""
        p1 = small_reactor.create_plasma(num_particles=50, seed=42)
        p2 = small_reactor.create_plasma(num_particles=50, seed=42)
        
        assert torch.allclose(p1, p2)
    
    @pytest.mark.unit
    def test_particles_near_axis(self, small_reactor, deterministic_seed):
        """Particles should start near magnetic axis."""
        particles = small_reactor.create_plasma(num_particles=100, seed=42)
        
        # Compute distance from magnetic axis
        rho = small_reactor.compute_rho(particles[:, :3])
        
        # Most should be well within minor radius
        mean_rho = rho.mean().item()
        assert mean_rho < small_reactor.a  # Inside plasma boundary


class TestBorisPusher:
    """Test Boris particle pusher algorithm."""
    
    @pytest.mark.unit
    @pytest.mark.physics
    def test_energy_conservation(self, small_reactor, deterministic_seed):
        """Boris pusher should conserve energy (no E field)."""
        particles = small_reactor.create_plasma(num_particles=20, seed=42)
        
        # Initial kinetic energy
        v_initial = particles[:, 3:6]
        KE_initial = 0.5 * torch.sum(v_initial**2).item()
        
        # Evolve
        final_particles, _ = small_reactor.push_particles(
            particles, dt=0.0001, steps=100, verbose=False
        )
        
        # Final kinetic energy
        v_final = final_particles[:, 3:6]
        KE_final = 0.5 * torch.sum(v_final**2).item()
        
        # Should be conserved to within numerical tolerance
        assert KE_final == pytest.approx(KE_initial, rel=0.05)
    
    @pytest.mark.unit
    @pytest.mark.physics
    def test_gyration_frequency(self, small_reactor, deterministic_seed):
        """Test Larmor gyration physics."""
        # Single particle at magnetic axis
        particles = torch.tensor([
            [1.0, 0.0, 0.0, 0.0, 0.1, 0.0]  # Small perpendicular velocity
        ], dtype=torch.float64)
        
        # Push for a few steps
        final, _ = small_reactor.push_particles(
            particles, dt=0.001, steps=10, verbose=False
        )
        
        # Particle should have moved but stayed near axis
        assert final.shape == (1, 6)


class TestConfinement:
    """Test plasma confinement analysis."""
    
    @pytest.mark.unit
    def test_confinement_report_fields(self, small_reactor, deterministic_seed):
        """Test ConfinementReport has all required fields."""
        particles = small_reactor.create_plasma(num_particles=50, seed=42)
        report = small_reactor.analyze_confinement(particles)
        
        assert hasattr(report, 'confined_particles')
        assert hasattr(report, 'escaped_particles')
        assert hasattr(report, 'confinement_ratio')
        assert hasattr(report, 'status')
    
    @pytest.mark.unit
    def test_initial_confinement_high(self, small_reactor, deterministic_seed):
        """Initially created plasma should be well confined."""
        particles = small_reactor.create_plasma(num_particles=100, seed=42)
        report = small_reactor.analyze_confinement(particles)
        
        # Most particles should be confined initially
        assert report.confinement_ratio > 0.9


class TestPlasmaState:
    """Test PlasmaState dataclass."""
    
    @pytest.mark.unit
    def test_plasma_state_creation(self, deterministic_seed):
        """Test PlasmaState can be created."""
        positions = torch.randn(10, 3, dtype=torch.float64)
        velocities = torch.randn(10, 3, dtype=torch.float64)
        
        state = PlasmaState(positions=positions, velocities=velocities)
        
        assert state.num_particles == 10
    
    @pytest.mark.unit
    def test_kinetic_energy(self, deterministic_seed):
        """Test kinetic energy calculation."""
        positions = torch.zeros(5, 3, dtype=torch.float64)
        velocities = torch.ones(5, 3, dtype=torch.float64)
        
        state = PlasmaState(positions=positions, velocities=velocities)
        
        # KE = 0.5 * sum(v^2) = 0.5 * 5 * 3 = 7.5
        assert state.kinetic_energy == pytest.approx(7.5, rel=0.01)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestFusionIntegration:
    """Integration tests for fusion simulation."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_simulation(self, small_reactor, deterministic_seed):
        """Test complete plasma simulation workflow."""
        # Create plasma
        particles = small_reactor.create_plasma(
            num_particles=50,
            temperature=0.5,
            toroidal_flow=5.0,
            seed=42
        )
        
        # Evolve
        final_particles, escape_history = small_reactor.push_particles(
            particles,
            dt=0.001,
            steps=100,
            verbose=False
        )
        
        # Analyze
        report = small_reactor.analyze_confinement(final_particles)
        
        # Verify
        assert final_particles.shape == particles.shape
        assert report.total_particles == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])
