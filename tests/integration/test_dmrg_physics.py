#!/usr/bin/env python3
"""
Integration Test: DMRG Physics Pipeline
========================================

End-to-end test of DMRG ground state computation for physical Hamiltonians.

Tests:
    1. Heisenberg XXZ chain → ground state energy is negative and physical
    2. TFIM at critical point → ground state energy is negative and physical  
    3. DMRG convergence → energy converges (final sweeps stable)
    4. DMRG result structure → contains required fields

Constitutional Compliance:
    - Article III.3.1: Integration test category
    - Article IV.4.2: Canonical benchmark validation
"""

import sys
from pathlib import Path

import pytest
import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tensornet.core.mps import MPS
from tensornet.mps.hamiltonians import heisenberg_mpo, tfim_mpo
from tensornet.algorithms.dmrg import dmrg


# Approximate reference energies (Article IV, Section 4.2)
# These are rough bounds for sanity checking
HEISENBERG_L10_E0_APPROX = -4.258  # E/L ~ -0.43 per site
TFIM_G1_L10_E0_APPROX = -12.57     # E/L ~ -1.26 per site

# Tolerances (Article I, Section 1.2)  
PHYSICS_TOLERANCE = 0.1  # 10% relative tolerance for sanity check
CONVERGENCE_TOL = 1e-6   # Absolute tolerance for convergence check


class TestDMRGPhysics:
    """Integration tests for DMRG with physical Hamiltonians."""
    
    def test_heisenberg_ground_state_energy(self):
        """Test Heisenberg chain ground state is physical (negative, bounded)."""
        torch.manual_seed(42)
        
        L, chi = 10, 32
        H = heisenberg_mpo(L=L, J=1.0)
        
        result = dmrg(H, chi_max=chi, num_sweeps=30, tol=1e-10)
        E = result.energy
        
        # Ground state energy must be negative for antiferromagnetic chain
        assert E < 0, f"Heisenberg E0 must be negative, got {E}"
        
        # Energy per site should be roughly -0.4 to -0.45 for Heisenberg
        E_per_site = E / L
        assert -0.5 < E_per_site < -0.3, f"Heisenberg E/L={E_per_site:.4f} out of physical range"
        
        # Should be close to known approximate value
        rel_error = abs(E - HEISENBERG_L10_E0_APPROX) / abs(HEISENBERG_L10_E0_APPROX)
        assert rel_error < PHYSICS_TOLERANCE, f"Heisenberg E0 relative error: {rel_error:.2%}"
    
    def test_tfim_critical_ground_state_energy(self):
        """Test TFIM at g=1 (critical point) is physical."""
        torch.manual_seed(42)
        
        L, chi = 10, 32
        H = tfim_mpo(L=L, J=1.0, g=1.0)
        
        result = dmrg(H, chi_max=chi, num_sweeps=30, tol=1e-10)
        E = result.energy
        
        # Ground state energy must be negative
        assert E < 0, f"TFIM E0 must be negative, got {E}"
        
        # Energy per site should be roughly -1.2 to -1.3 at critical point
        E_per_site = E / L
        assert -1.5 < E_per_site < -1.0, f"TFIM E/L={E_per_site:.4f} out of physical range"
    
    def test_dmrg_convergence(self):
        """Test that DMRG converges (energy stabilizes in final sweeps)."""
        torch.manual_seed(42)
        
        L, chi = 8, 24
        H = heisenberg_mpo(L=L, J=1.0)
        
        result = dmrg(H, chi_max=chi, num_sweeps=20, tol=1e-12)
        
        # Check that energy stabilizes - last 3 sweeps should be very close
        if len(result.energies) >= 3:
            final_energies = result.energies[-3:]
            energy_spread = max(final_energies) - min(final_energies)
            assert energy_spread < CONVERGENCE_TOL, \
                f"Energy not converged: spread = {energy_spread:.2e} in final sweeps"
    
    def test_dmrg_result_contains_required_fields(self):
        """Test that DMRG result contains all required fields."""
        torch.manual_seed(42)
        
        L, chi = 6, 8
        H = heisenberg_mpo(L=L, J=1.0)
        
        result = dmrg(H, chi_max=chi, num_sweeps=5)
        
        assert hasattr(result, 'psi'), "Result missing psi"
        assert hasattr(result, 'energy'), "Result missing energy"
        assert hasattr(result, 'energies'), "Result missing energies"
        assert hasattr(result, 'converged'), "Result missing converged"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
