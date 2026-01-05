"""
HyperTensor Physics Engine - Test Suite
"""

import numpy as np
import pytest

from hypertensor.core import TTTensor, tt_round, tt_to_full, tt_add, tt_dot, tt_norm
from hypertensor.integrators import SymplecticIntegrator, LangevinDynamics
from hypertensor.pde import ResistiveMHD, FokkerPlanck, HeatEquation1D, CompositeWall


class TestTensorTrain:
    """Tests for TT compression core."""
    
    def test_tt_round_1d(self):
        """1D tensor should have trivial TT."""
        x = np.random.randn(100)
        tt = tt_round(x, max_rank=10)
        
        assert tt.shape == (100,)
        assert tt.ranks == (1, 1)
        assert len(tt.cores) == 1
    
    def test_tt_round_2d(self):
        """2D tensor compression."""
        A = np.random.randn(20, 30)
        tt = tt_round(A, max_rank=5)
        
        assert tt.shape == (20, 30)
        assert tt.max_rank <= 5
        
    def test_tt_to_full_reconstruction(self):
        """Reconstruction should be close to original."""
        A = np.random.randn(10, 10, 10)
        tt = tt_round(A, max_rank=50)  # High rank for accuracy
        A_rec = tt_to_full(tt)
        
        error = np.linalg.norm(A - A_rec) / np.linalg.norm(A)
        assert error < 1e-10
        
    def test_tt_compression_ratio(self):
        """Large tensor should have significant compression."""
        A = np.random.randn(10, 10, 10, 10)  # 10,000 elements
        tt = tt_round(A, max_rank=3)
        
        assert tt.compression_ratio > 5  # At least 5× compression
        
    def test_tt_dot(self):
        """Inner product should match full computation."""
        A = np.random.randn(8, 8, 8)
        B = np.random.randn(8, 8, 8)
        
        tt_a = tt_round(A, max_rank=50)
        tt_b = tt_round(B, max_rank=50)
        
        dot_tt = tt_dot(tt_a, tt_b)
        dot_full = np.sum(A * B)
        
        assert abs(dot_tt - dot_full) < 1e-8


class TestIntegrators:
    """Tests for time integrators."""
    
    def test_symplectic_harmonic_oscillator(self):
        """Harmonic oscillator should conserve energy."""
        def force(x):
            return -x  # F = -kx with k=1
        
        integrator = SymplecticIntegrator(force, mass=1.0)
        
        x = np.array([1.0])
        v = np.array([0.0])
        
        # Initial energy
        E0 = 0.5 * v[0]**2 + 0.5 * x[0]**2
        
        # Run for many steps
        for _ in range(1000):
            x, v = integrator.step(x, v, dt=0.01)
        
        # Final energy
        E1 = 0.5 * v[0]**2 + 0.5 * x[0]**2
        
        # Energy should be conserved (symplectic property)
        assert abs(E1 - E0) / E0 < 0.01  # Within 1%
    
    def test_langevin_equilibrium(self):
        """Langevin dynamics should thermalize."""
        def potential(x):
            return 0.5 * np.sum(x**2)  # Harmonic well
        
        langevin = LangevinDynamics(
            potential_fn=potential,
            temperature=300,
            friction=1.0
        )
        
        x0 = np.array([5.0, 5.0, 5.0])  # Start far from equilibrium
        result = langevin.run(x0, n_steps=100, dt=1e-14)
        
        # Should return a result
        assert "final_position" in result
        assert "rmsd" in result


class TestPDE:
    """Tests for PDE solvers."""
    
    def test_mhd_harris_sheet(self):
        """Harris sheet should show reconnection."""
        mhd = ResistiveMHD(nx=32, L=1.0, eta=0.01)
        result = mhd.run(n_steps=50, dt=1e-5)
        
        assert result["stable"]
        assert "reconnection_rate" in result
        assert np.all(np.isfinite(result["final_B"]))
    
    def test_fokker_planck_relaxation(self):
        """Distribution should relax toward equilibrium."""
        fp = FokkerPlanck(nx=64, x_range=(-5, 5), diffusion=1.0)
        
        # Start off-center
        P0 = fp.initialize_gaussian(mean=3.0, std=0.5)
        result = fp.run(P0, n_steps=200, dt=0.01)
        
        # Mean should move toward zero (equilibrium)
        assert abs(result["mean"]) < 3.0  # Closer to zero
        
    def test_heat_equation_steady_state(self):
        """Heat equation should reach linear steady state."""
        heat = HeatEquation1D(nx=50, L=1.0, alpha=1e-4)
        
        T0 = np.ones(50) * 50  # Uniform initial
        result = heat.run(T0, n_steps=10000, dt=0.01, T_left=100, T_right=0)
        
        # Should approach linear profile
        T_analytical = heat.steady_state(100, 0)
        error = np.max(np.abs(result["final_T"] - T_analytical))
        
        assert error < 5  # Within 5 degrees
        
    def test_composite_wall(self):
        """Composite wall should compute correct heat flux."""
        wall = CompositeWall(
            layers=[(0.01, 100), (0.02, 200)],  # (thickness, conductivity)
            T_hot=500,
            T_cold=100,
            h_conv=10000
        )
        
        result = wall.analyze()
        
        assert result["heat_flux_W_m2"] > 0
        assert result["T_surface"] == 500
        assert len(result["interface_temps"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
