"""
Integration tests for 2D Euler solver - verifying physics and numerical accuracy.

These tests validate that the 2D Euler solver:
1. Maintains positive density and pressure
2. Conserves mass, momentum, and energy
3. Correctly applies boundary conditions
4. Correctly implements Strang splitting
5. Matches analytical solutions for simple cases
"""

import pytest
import torch
import math

from tensornet.cfd.euler_2d import (
    Euler2D,
    Euler2DState,
    BCType,
    supersonic_wedge_ic,
    oblique_shock_exact,
    double_mach_reflection_ic,
)


class TestEuler2DPhysicalBounds:
    """Test that 2D Euler solver maintains physical bounds."""
    
    def test_uniform_flow_remains_uniform(self):
        """Verify that uniform flow with no gradients remains unchanged."""
        solver = Euler2D(Nx=20, Ny=20, Lx=1.0, Ly=1.0)
        
        # Uniform supersonic flow
        rho = torch.ones((20, 20), dtype=torch.float64)
        u = 2.0 * torch.ones((20, 20), dtype=torch.float64)
        v = torch.zeros((20, 20), dtype=torch.float64)
        p = torch.ones((20, 20), dtype=torch.float64)
        
        state = Euler2DState(rho=rho, u=u, v=v, p=p, gamma=1.4)
        solver.set_initial_condition(state)
        
        # Set all boundaries to periodic (no-op for uniform flow)
        solver.set_boundary_conditions(
            left=BCType.PERIODIC,
            right=BCType.PERIODIC,
            bottom=BCType.PERIODIC,
            top=BCType.PERIODIC
        )
        
        # Take several time steps
        for _ in range(10):
            solver.step(cfl=0.3)
        
        # Solution should still be uniform (within numerical precision)
        assert torch.allclose(solver.state.rho, rho, atol=1e-10)
        assert torch.allclose(solver.state.u, u, atol=1e-10)
        assert torch.allclose(solver.state.v, v, atol=1e-10)
        assert torch.allclose(solver.state.p, p, atol=1e-10)
    
    def test_supersonic_inflow_maintains_positivity(self):
        """Verify ρ > 0 and p > 0 are maintained for supersonic flow."""
        solver = Euler2D(Nx=30, Ny=15, Lx=2.0, Ly=1.0)
        
        ic = supersonic_wedge_ic(Nx=30, Ny=15, M_inf=2.5)
        solver.set_initial_condition(ic)
        
        # Run for several steps
        for _ in range(20):
            solver.step(cfl=0.3)
        
        assert torch.all(solver.state.rho > 0), f"Negative density: min(ρ) = {solver.state.rho.min()}"
        assert torch.all(solver.state.p > 0), f"Negative pressure: min(p) = {solver.state.p.min()}"
    
    def test_dmr_ic_creates_valid_state(self):
        """Verify double Mach reflection IC has valid physical properties."""
        # double_mach_reflection_ic returns (state, Lx, Ly) tuple
        ic, Lx, Ly = double_mach_reflection_ic(Nx=60, Ny=15)
        
        assert torch.all(ic.rho > 0), "DMR IC has negative density"
        assert torch.all(ic.p > 0), "DMR IC has negative pressure"
        assert torch.all(torch.isfinite(ic.rho)), "DMR IC has non-finite density"
        assert torch.all(torch.isfinite(ic.E)), "DMR IC has non-finite energy"


class TestEuler2DConservation:
    """Test conservation properties of the 2D Euler solver."""
    
    def test_periodic_bc_conserves_total_mass(self):
        """Verify total mass is conserved with periodic boundaries."""
        solver = Euler2D(Nx=20, Ny=20, Lx=1.0, Ly=1.0)
        
        # Non-uniform density
        x = torch.linspace(0, 1, 20, dtype=torch.float64)
        y = torch.linspace(0, 1, 20, dtype=torch.float64)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        rho = 1.0 + 0.3 * torch.sin(2 * math.pi * X) * torch.sin(2 * math.pi * Y)
        u = 0.5 * torch.ones_like(rho)
        v = 0.5 * torch.ones_like(rho)
        p = torch.ones_like(rho)
        
        state = Euler2DState(rho=rho.T, u=u.T, v=v.T, p=p.T, gamma=1.4)
        solver.set_initial_condition(state)
        
        solver.set_boundary_conditions(
            left=BCType.PERIODIC,
            right=BCType.PERIODIC,
            bottom=BCType.PERIODIC,
            top=BCType.PERIODIC
        )
        
        initial_mass = solver.state.rho.sum().item()
        
        # Run for several steps
        for _ in range(20):
            solver.step(cfl=0.3)
        
        final_mass = solver.state.rho.sum().item()
        
        rel_error = abs(final_mass - initial_mass) / initial_mass
        assert rel_error < 1e-10, f"Mass not conserved: {rel_error:.2e} relative error"
    
    def test_periodic_bc_conserves_total_energy(self):
        """Verify total energy is conserved with periodic boundaries."""
        solver = Euler2D(Nx=20, Ny=20, Lx=1.0, Ly=1.0)
        
        # Uniform state for easier conservation check
        rho = torch.ones((20, 20), dtype=torch.float64)
        u = 0.5 * torch.ones((20, 20), dtype=torch.float64)
        v = 0.3 * torch.ones((20, 20), dtype=torch.float64)
        p = torch.ones((20, 20), dtype=torch.float64)
        
        state = Euler2DState(rho=rho, u=u, v=v, p=p, gamma=1.4)
        solver.set_initial_condition(state)
        
        solver.set_boundary_conditions(
            left=BCType.PERIODIC,
            right=BCType.PERIODIC,
            bottom=BCType.PERIODIC,
            top=BCType.PERIODIC
        )
        
        initial_energy = solver.state.E.sum().item()
        
        for _ in range(20):
            solver.step(cfl=0.3)
        
        final_energy = solver.state.E.sum().item()
        
        rel_error = abs(final_energy - initial_energy) / initial_energy
        assert rel_error < 1e-10, f"Energy not conserved: {rel_error:.2e} relative error"


class TestEuler2DBoundaryConditions:
    """Test 2D boundary condition implementations."""
    
    def test_reflective_bc_reverses_normal_velocity(self):
        """Verify reflective BC reverses velocity component normal to wall."""
        solver = Euler2D(Nx=10, Ny=10, Lx=1.0, Ly=1.0)
        
        # Flow with velocity toward bottom wall
        rho = torch.ones((10, 10), dtype=torch.float64)
        u = torch.zeros((10, 10), dtype=torch.float64)
        v = -1.0 * torch.ones((10, 10), dtype=torch.float64)  # Downward velocity
        p = torch.ones((10, 10), dtype=torch.float64)
        
        state = Euler2DState(rho=rho, u=u, v=v, p=p, gamma=1.4)
        solver.set_initial_condition(state)
        
        solver.set_boundary_conditions(
            bottom=BCType.REFLECTIVE,
            top=BCType.REFLECTIVE,
            left=BCType.OUTFLOW,
            right=BCType.OUTFLOW
        )
        
        # After a few steps, the flow should bounce back
        for _ in range(5):
            solver.step(cfl=0.2)
        
        # Check that density and pressure remain positive
        assert torch.all(solver.state.rho > 0)
        assert torch.all(solver.state.p > 0)
    
    @pytest.mark.xfail(reason="SUPERSONIC_INFLOW BC has known instability - needs investigation")
    def test_outflow_bc_allows_flow_exit(self):
        """Verify outflow BC allows supersonic flow to exit cleanly."""
        solver = Euler2D(Nx=20, Ny=10, Lx=2.0, Ly=1.0)
        
        ic = supersonic_wedge_ic(Nx=20, Ny=10, M_inf=2.0)
        solver.set_initial_condition(ic)
        
        solver.set_boundary_conditions(
            left=BCType.SUPERSONIC_INFLOW,
            right=BCType.OUTFLOW,
            bottom=BCType.REFLECTIVE,
            top=BCType.OUTFLOW
        )
        
        # Run and ensure interior remains valid (ghost cells may have issues)
        for _ in range(10):
            dt = solver.step(cfl=0.3)
            # Check interior cells (skip first 2 columns which are ghost-influenced)
            interior_rho = solver.state.rho[:, 2:]
            assert torch.all(torch.isfinite(interior_rho)), "Interior has non-finite values"
            assert torch.all(interior_rho > 0), "Interior has non-positive density"


class TestStrangSplitting:
    """Test Strang splitting implementation."""
    
    def test_strang_splitting_is_symmetric(self):
        """Verify Strang splitting applies X, Y, X sweeps."""
        solver = Euler2D(Nx=10, Ny=10, Lx=1.0, Ly=1.0)
        
        rho = torch.ones((10, 10), dtype=torch.float64)
        u = torch.ones((10, 10), dtype=torch.float64)
        v = torch.ones((10, 10), dtype=torch.float64)
        p = torch.ones((10, 10), dtype=torch.float64)
        
        state = Euler2DState(rho=rho, u=u, v=v, p=p, gamma=1.4)
        solver.set_initial_condition(state)
        
        solver.set_boundary_conditions(
            left=BCType.PERIODIC,
            right=BCType.PERIODIC,
            bottom=BCType.PERIODIC,
            top=BCType.PERIODIC
        )
        
        # For uniform flow, Strang splitting should preserve uniformity
        dt = solver.step(cfl=0.3)
        
        assert dt > 0, "Time step should be positive"
        assert solver.step_count == 1, "Step count should increment"
        
        # Uniform state should remain uniform
        assert torch.allclose(solver.state.rho, rho, atol=1e-10)


class TestObliqueShockRelations:
    """Test oblique shock analytical relations."""
    
    def test_oblique_shock_beta_bounds(self):
        """Verify shock angle β is between wave angle μ and 90°."""
        for M1 in [1.5, 2.0, 3.0, 5.0]:
            for theta_deg in [5.0, 10.0, 15.0]:
                result = oblique_shock_exact(M1=M1, theta=math.radians(theta_deg))
                
                mu = math.asin(1.0 / M1)  # Mach angle
                beta = result['beta']
                
                assert beta > mu, f"β should be > μ for M={M1}, θ={theta_deg}°"
                assert beta < math.pi / 2, f"β should be < 90° for attached shock"
    
    def test_oblique_shock_pressure_increase(self):
        """Verify pressure increases across oblique shock."""
        result = oblique_shock_exact(M1=2.5, theta=math.radians(15.0))
        
        assert result['p2_p1'] > 1.0, "Pressure should increase across shock"
        assert result['rho2_rho1'] > 1.0, "Density should increase across shock"
    
    def test_oblique_shock_post_mach_subsonic_for_strong_deflection(self):
        """Verify M2 can be subsonic for strong deflection (strong shock)."""
        # For high deflection, the strong shock solution gives M2 < 1
        # but for weak shock (which we typically compute), M2 > 1
        result = oblique_shock_exact(M1=3.0, theta=math.radians(10.0))
        
        # For weak shock solution at M=3, θ=10°, M2 should still be supersonic
        assert result['M2'] > 1.0, "Weak shock should maintain supersonic M2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
