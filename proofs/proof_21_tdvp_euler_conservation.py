"""
Proof 21.3: TDVP-Euler Conservation Verification (Simplified)
==============================================================

This proof verifies that the TT-CFD framework:
1. Correctly encodes/decodes primitive variables in MPS format
2. Computes conservation integrals correctly
3. Initializes the Euler solver properly

Constitution Compliance: Article I.1 (Proof Requirements)

NOTE: Full time-stepping tests are in proof_21_tdvp_euler_sod.py.
This proof focuses on the data structure correctness.
"""

import torch
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_mps_state_roundtrip():
    """Test that MPSState correctly encodes and decodes primitive variables."""
    from tensornet.cfd.tt_cfd import MPSState
    
    N = 16
    chi_max = 8
    
    # Create primitive variables
    rho = torch.ones(N, dtype=torch.float64) + 0.1 * torch.sin(torch.linspace(0, 2*torch.pi, N, dtype=torch.float64))
    u = 0.5 * torch.ones(N, dtype=torch.float64)
    p = torch.ones(N, dtype=torch.float64)
    
    gamma = 1.4
    
    # Create MPS state
    state = MPSState.from_primitive(rho, u, p, chi_max=chi_max, gamma=gamma)
    
    # Extract back - to_primitive returns (rho, u, p) NOT (rho, rhou, E)
    rho2, u2, p2 = state.to_primitive(gamma=gamma)
    
    # Check round-trip accuracy for primitive variables
    rho_err = torch.max(torch.abs(rho2 - rho)).item()
    u_err = torch.max(torch.abs(u2 - u)).item()
    p_err = torch.max(torch.abs(p2 - p)).item()
    
    passed = rho_err < 1e-10 and u_err < 1e-10 and p_err < 1e-10
    
    return {
        'test': 'mps_state_roundtrip',
        'N': N,
        'chi_max': chi_max,
        'rho_error': rho_err,
        'u_error': u_err,
        'p_error': p_err,
        'passed': passed
    }


def test_conservation_check():
    """Test that conservation quantities are computed correctly."""
    from tensornet.cfd.tt_cfd import MPSState
    
    N = 16
    chi_max = 8
    dx = 1.0 / N
    
    # Uniform state
    rho = torch.ones(N, dtype=torch.float64)
    u = torch.zeros(N, dtype=torch.float64)
    p = torch.ones(N, dtype=torch.float64)
    gamma = 1.4
    
    state = MPSState.from_primitive(rho, u, p, chi_max=chi_max, gamma=gamma)
    state.dx = dx
    
    # Test the total_mass, total_momentum, total_energy methods
    # Note: these return SUM, not integral (no dx multiplication)
    mass = state.total_mass()
    momentum = state.total_momentum()
    energy = state.total_energy()
    
    # Expected values (sum over grid points, NOT scaled by dx)
    expected_mass = float(rho.sum())
    expected_momentum = float((rho * u).sum())
    E_vals = p / (gamma - 1) + 0.5 * rho * u**2
    expected_energy = float(E_vals.sum())
    
    mass_err = abs(mass - expected_mass) / max(abs(expected_mass), 1e-10)
    mom_err = abs(momentum - expected_momentum)  # momentum is 0
    energy_err = abs(energy - expected_energy) / max(abs(expected_energy), 1e-10)
    
    passed = mass_err < 1e-10 and mom_err < 1e-10 and energy_err < 1e-10
    
    return {
        'test': 'conservation_check',
        'expected_mass': expected_mass,
        'computed_mass': mass,
        'mass_error': mass_err,
        'expected_momentum': expected_momentum,
        'computed_momentum': momentum,
        'momentum_error': mom_err,
        'expected_energy': expected_energy,
        'computed_energy': energy,
        'energy_error': energy_err,
        'passed': passed
    }


def test_euler_mpo_construction():
    """Test that EulerMPO can be constructed."""
    from tensornet.cfd.tt_cfd import EulerMPO
    
    N = 8
    dx = 0.1
    gamma = 1.4
    
    mpo = EulerMPO(N, dx, gamma)
    
    # Check basic properties
    has_cores = len(mpo.mpo_cores) == N
    correct_gamma = mpo.gamma == gamma
    correct_dx = mpo.dx == dx
    correct_n = mpo.n_sites == N
    
    passed = has_cores and correct_gamma and correct_dx and correct_n
    
    return {
        'test': 'euler_mpo_construction',
        'N': N,
        'has_cores': has_cores,
        'correct_gamma': correct_gamma,
        'correct_dx': correct_dx,
        'correct_n': correct_n,
        'passed': passed
    }


def test_tt_euler_1d_init():
    """Test TT_Euler1D initialization."""
    from tensornet.cfd.tt_cfd import TT_Euler1D
    
    N = 8
    chi_max = 4
    L = 1.0
    
    solver = TT_Euler1D(N=N, L=L, gamma=1.4, chi_max=chi_max)
    
    # Initialize with simple state
    rho = torch.ones(N, dtype=torch.float64)
    u = torch.zeros(N, dtype=torch.float64)
    p = torch.ones(N, dtype=torch.float64)
    
    solver.initialize(rho, u, p)
    
    # Check state exists
    has_state = solver.state is not None
    correct_n = solver.N == N
    
    # Check round-trip
    rho2, rhou2, E2 = solver.state.to_primitive()
    roundtrip_ok = torch.allclose(rho2, rho, atol=1e-10)
    
    passed = has_state and correct_n and roundtrip_ok
    
    return {
        'test': 'tt_euler_1d_init',
        'N': N,
        'chi_max': chi_max,
        'has_state': has_state,
        'correct_n': correct_n,
        'roundtrip_ok': roundtrip_ok,
        'passed': passed
    }


def test_sod_initialization():
    """Test Sod shock tube initialization."""
    from tensornet.cfd.tt_cfd import TT_Euler1D
    
    N = 16
    chi_max = 8
    L = 1.0
    
    solver = TT_Euler1D(N=N, L=L, gamma=1.4, chi_max=chi_max)
    solver.initialize_sod()
    
    # Check state exists and basic properties
    has_state = solver.state is not None
    
    # Get primitive values back
    rho, u, p = solver.state.to_primitive()
    
    # Sod tube: left state (rho=1), right state (rho=0.125)
    # Check that left side is higher density
    left_density = rho[:N//2].mean().item()
    right_density = rho[N//2:].mean().item()
    density_ratio_ok = left_density > right_density * 2
    
    # Check that mass sum is reasonable (not integral, just sum)
    mass = solver.state.total_mass()
    expected_mass_approx = N/2 * 1.0 + N/2 * 0.125  # Sum, not integral
    mass_reasonable = abs(mass - expected_mass_approx) / expected_mass_approx < 0.3
    
    passed = has_state and density_ratio_ok and mass_reasonable
    
    return {
        'test': 'sod_initialization',
        'N': N,
        'chi_max': chi_max,
        'has_state': has_state,
        'left_density': left_density,
        'right_density': right_density,
        'density_ratio_ok': density_ratio_ok,
        'mass': mass,
        'expected_mass_approx': expected_mass_approx,
        'mass_reasonable': mass_reasonable,
        'passed': passed
    }


def run_all_proofs():
    """Run all conservation proofs."""
    results = {
        'proof_id': '21.3',
        'name': 'TDVP-Euler Conservation (Simplified)',
        'timestamp': datetime.now().isoformat(),
        'tests': []
    }
    
    print("=" * 60)
    print("Proof 21.3: TDVP-Euler Conservation Verification")
    print("=" * 60)
    
    # Run tests
    tests = [
        ('MPS State Round-Trip', test_mps_state_roundtrip),
        ('Conservation Check', test_conservation_check),
        ('Euler MPO Construction', test_euler_mpo_construction),
        ('TT Euler 1D Init', test_tt_euler_1d_init),
        ('Sod Initialization', test_sod_initialization),
    ]
    
    all_passed = True
    for name, test_fn in tests:
        print(f"\n{name}...")
        try:
            result = test_fn()
            results['tests'].append(result)
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"  {status}")
            for k, v in result.items():
                if k not in ['test', 'passed']:
                    if isinstance(v, float):
                        print(f"  {k}: {v:.2e}")
                    else:
                        print(f"  {k}: {v}")
            all_passed = all_passed and result['passed']
        except Exception as e:
            import traceback
            print(f"  ❌ ERROR: {e}")
            traceback.print_exc()
            results['tests'].append({'test': name, 'error': str(e), 'passed': False})
            all_passed = False
    
    results['all_passed'] = all_passed
    
    # Save results
    output_path = Path(__file__).parent / 'proof_21_tdvp_conservation_result.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "=" * 60)
    print(f"PROOF 21.3: {'✅ PASSED' if all_passed else '❌ FAILED'}")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    run_all_proofs()
