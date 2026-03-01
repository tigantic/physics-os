#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         GATE 1: LATTICE CONSTRUCTION                         ║
║                                                                              ║
║                     Yang-Mills Battle Plan - First Gate                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Gate 1 validates the foundational infrastructure for lattice gauge theory:

1. SU(2) Group Operations
   - Pauli matrices algebra
   - Commutation relations
   - Random element properties

2. Lattice Geometry
   - Hypercubic lattice with periodic BC
   - Link enumeration
   - Plaquette construction

3. Quantum Operators
   - Electric field operators E^a
   - Link operators U
   - Casimir operator E²

4. Kogut-Susskind Hamiltonian
   - H = (g²/2a)ΣE² - (1/g²a)ΣRe Tr(P)
   - Hermiticity
   - Spectral structure

5. Gauss Law
   - G_x operators at each site
   - Gauge invariance of trivial state

SUCCESS CRITERIA (from Battle Plan):
    - All Pauli algebra tests: max error < 10^{-14}
    - Lattice plaquettes in valid range
    - Hamiltonian Hermitian
    - Trivial state gauge-invariant

Author: TiganticLabz Yang-Mills Project
Date: 2026-01-15
"""

import sys
import numpy as np
import time
from datetime import datetime

# Import modules to test
sys.path.insert(0, '/home/brad/TiganticLabz/Main_Projects/physics-os')

from yangmills.su2 import (
    PAULI, TAU, SIGMA_0, 
    random_su2, spin_j_generators, casimir_eigenvalue,
    pauli_matrices, su2_generators
)
from yangmills.lattice import (
    Lattice, LatticeLink, GaugeConfiguration
)
from yangmills.operators import (
    TruncatedHilbertSpace, ElectricFieldOperator, LinkOperator
)
from yangmills.hamiltonian import SinglePlaquetteHamiltonian
from yangmills.gauss import SinglePlaquetteGauss, GaussOperator


# =============================================================================
# TEST RESULTS TRACKING
# =============================================================================

class GateResults:
    """Track test results for Gate 1."""
    
    def __init__(self):
        self.tests = []
        self.passed = 0
        self.failed = 0
        
    def record(self, name: str, passed: bool, details: str = ""):
        self.tests.append({
            'name': name,
            'passed': passed,
            'details': details
        })
        if passed:
            self.passed += 1
        else:
            self.failed += 1
    
    def summary(self) -> str:
        total = self.passed + self.failed
        lines = [
            "=" * 70,
            "GATE 1 TEST SUMMARY",
            "=" * 70,
            f"Total: {total}  |  Passed: {self.passed}  |  Failed: {self.failed}",
            "-" * 70
        ]
        
        for test in self.tests:
            status = "✓" if test['passed'] else "✗"
            lines.append(f"  [{status}] {test['name']}")
            if test['details']:
                lines.append(f"      {test['details']}")
        
        lines.append("=" * 70)
        
        if self.failed == 0:
            lines.append("  ★★★ GATE 1 PASSED - LATTICE CONSTRUCTION VALIDATED ★★★")
        else:
            lines.append(f"  ✗ GATE 1 FAILED - {self.failed} tests need attention")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


# =============================================================================
# SU(2) ALGEBRA TESTS
# =============================================================================

def test_su2_algebra(results: GateResults):
    """Test 1: SU(2) group algebra."""
    
    print("\n--- Test 1: SU(2) Algebra ---")
    
    sigma = pauli_matrices()
    tau = su2_generators()
    
    # 1.1: Pauli commutation relations [σ_a, σ_b] = 2i ε_{abc} σ_c
    max_comm_err = 0.0
    for a in range(3):
        for b in range(3):
            comm = sigma[a] @ sigma[b] - sigma[b] @ sigma[a]
            expected = 2j * sum(
                levi_civita(a, b, c) * sigma[c] 
                for c in range(3)
            )
            err = np.linalg.norm(comm - expected)
            max_comm_err = max(max_comm_err, err)
    
    results.record(
        "SU(2) Commutation Relations", 
        max_comm_err < 1e-14,
        f"max error = {max_comm_err:.2e}"
    )
    
    # 1.2: Pauli anticommutation {σ_a, σ_b} = 2δ_{ab} I
    max_anti_err = 0.0
    I = np.eye(2)
    for a in range(3):
        for b in range(3):
            anti = sigma[a] @ sigma[b] + sigma[b] @ sigma[a]
            expected = 2 * (1 if a == b else 0) * I
            err = np.linalg.norm(anti - expected)
            max_anti_err = max(max_anti_err, err)
    
    results.record(
        "SU(2) Anticommutation Relations",
        max_anti_err < 1e-14,
        f"max error = {max_anti_err:.2e}"
    )
    
    # 1.3: Hermiticity σ† = σ
    max_herm_err = 0.0
    for a in range(3):
        err = np.linalg.norm(sigma[a] - sigma[a].conj().T)
        max_herm_err = max(max_herm_err, err)
    
    results.record(
        "Pauli Hermiticity",
        max_herm_err < 1e-14,
        f"max error = {max_herm_err:.2e}"
    )
    
    # 1.4: Random SU(2) elements
    n_samples = 100
    det_errors = []
    unit_errors = []
    
    for _ in range(n_samples):
        U_obj = random_su2()
        U = U_obj.matrix  # Get the matrix from SU2 object
        det_errors.append(abs(np.linalg.det(U) - 1.0))
        unit_errors.append(np.linalg.norm(U @ U.conj().T - np.eye(2)))
    
    results.record(
        "Random SU(2) det(U) = 1",
        max(det_errors) < 1e-12,
        f"max |det-1| = {max(det_errors):.2e}"
    )
    
    results.record(
        "Random SU(2) UU† = I",
        max(unit_errors) < 1e-12,
        f"max ||UU†-I|| = {max(unit_errors):.2e}"
    )
    
    # 1.5: Casimir eigenvalues j(j+1)
    casimir_correct = True
    for j in [0, 0.5, 1, 1.5, 2]:
        expected = j * (j + 1)
        computed = casimir_eigenvalue(j)
        if abs(computed - expected) > 1e-14:
            casimir_correct = False
    
    results.record(
        "Casimir eigenvalues j(j+1)",
        casimir_correct,
        "Verified for j = 0, 1/2, 1, 3/2, 2"
    )


def levi_civita(i, j, k):
    """Levi-Civita symbol ε_{ijk}."""
    return ((i - j) * (j - k) * (k - i)) // 2


# =============================================================================
# LATTICE GEOMETRY TESTS
# =============================================================================

def test_lattice_geometry(results: GateResults):
    """Test 2: Lattice construction."""
    
    print("\n--- Test 2: Lattice Geometry ---")
    
    # 2.1: 2D lattice construction
    L2 = Lattice(dims=(4, 4))  # 4x4 2D lattice
    expected_sites = 16
    expected_links = 32  # 2 * L^2 for 2D
    
    results.record(
        "2D Lattice Sites",
        L2.n_sites == expected_sites,
        f"Sites: {L2.n_sites} (expected {expected_sites})"
    )
    
    results.record(
        "2D Lattice Links",
        L2.n_links == expected_links,
        f"Links: {L2.n_links} (expected {expected_links})"
    )
    
    # 2.2: 3D lattice
    L3 = Lattice(dims=(3, 3, 3))  # 3x3x3 3D lattice
    expected_sites_3d = 27
    expected_links_3d = 81  # 3 * 3^3
    
    results.record(
        "3D Lattice Sites",
        L3.n_sites == expected_sites_3d,
        f"Sites: {L3.n_sites} (expected {expected_sites_3d})"
    )
    
    # 2.3: Plaquette count for 2D
    plaq_count = L2.n_plaquettes
    expected_plaq = 16  # L^2 plaquettes in 2D (1 plane)
    
    results.record(
        "2D Plaquette Count",
        plaq_count == expected_plaq,
        f"Plaquettes: {plaq_count} (expected {expected_plaq})"
    )
    
    # 2.4: Gauge configuration - identity has zero action
    config = GaugeConfiguration.identity(L2)
    action = config.wilson_action(beta=1.0)
    
    results.record(
        "Identity Config Action = 0",
        abs(action) < 1e-14,
        f"Action: {action:.2e}"
    )
    
    # 2.5: Random config plaquettes in valid range
    config_rand = GaugeConfiguration.random(L2)
    traces = []
    for plaq in L2.all_plaquettes():
        P = config_rand.plaquette(plaq)
        traces.append(np.real(np.trace(P)))
    
    valid_range = all(-2.01 < t < 2.01 for t in traces)
    
    results.record(
        "Plaquette Traces in [-2, 2]",
        valid_range,
        f"Range: [{min(traces):.3f}, {max(traces):.3f}]"
    )


# =============================================================================
# QUANTUM OPERATOR TESTS
# =============================================================================

def test_quantum_operators(results: GateResults):
    """Test 3: Quantum operators."""
    
    print("\n--- Test 3: Quantum Operators ---")
    
    # 3.1: Hilbert space dimension
    H = TruncatedHilbertSpace(j_max=0.5)
    expected_dim = 1 + 4  # j=0: 1, j=1/2: 4
    
    results.record(
        "Hilbert Space Dimension (j_max=0.5)",
        H.total_dim == expected_dim,
        f"Dim: {H.total_dim} (expected {expected_dim})"
    )
    
    H1 = TruncatedHilbertSpace(j_max=1.0)
    expected_dim_1 = 1 + 4 + 9  # j=0: 1, j=1/2: 4, j=1: 9
    
    results.record(
        "Hilbert Space Dimension (j_max=1.0)",
        H1.total_dim == expected_dim_1,
        f"Dim: {H1.total_dim} (expected {expected_dim_1})"
    )
    
    # 3.2: Casimir operator eigenvalues
    E_op = ElectricFieldOperator(H)
    E2 = E_op.E_squared
    
    # Check eigenvalues
    eigenvalues = np.diag(E2.toarray())
    
    # j=0 state should have eigenvalue 0
    # j=1/2 states should have eigenvalue 0.75
    j0_eigenvalue = eigenvalues[0]
    j_half_eigenvalue = eigenvalues[1]  # First j=1/2 state
    
    results.record(
        "E² eigenvalue for j=0",
        abs(j0_eigenvalue) < 1e-14,
        f"E²|j=0⟩ = {j0_eigenvalue:.2e} (expected 0)"
    )
    
    results.record(
        "E² eigenvalue for j=1/2",
        abs(j_half_eigenvalue - 0.75) < 1e-14,
        f"E²|j=1/2⟩ = {j_half_eigenvalue:.4f} (expected 0.75)"
    )
    
    # 3.3: Link operator exists and has entries
    U_op = LinkOperator(H)
    U = U_op.get_matrix()
    
    has_entries = U.nnz > 0
    
    results.record(
        "Link Operator Has Non-Zero Entries",
        has_entries,
        f"Non-zeros: {U.nnz}"
    )


# =============================================================================
# HAMILTONIAN TESTS
# =============================================================================

def test_hamiltonian(results: GateResults):
    """Test 4: Kogut-Susskind Hamiltonian."""
    
    print("\n--- Test 4: Hamiltonian ---")
    
    # 4.1: Hermiticity
    H_sys = SinglePlaquetteHamiltonian(j_max=0.5, g=1.0)
    herm_err = H_sys.verify_hermiticity()
    
    results.record(
        "Hamiltonian Hermiticity",
        herm_err < 1e-14,
        f"||H - H†|| = {herm_err:.2e}"
    )
    
    # 4.2: Ground state energy structure
    E0, gs = H_sys.strong_coupling_ground_state()
    
    results.record(
        "Strong Coupling Ground State E₀ = 0",
        abs(E0) < 1e-14,
        f"E₀ = {E0}"
    )
    
    # 4.3: Spectrum computation
    try:
        vals, vecs = H_sys.compute_spectrum(k=4)
        spectrum_works = True
        min_val = vals[0]
    except:
        spectrum_works = False
        min_val = None
    
    results.record(
        "Spectrum Computation",
        spectrum_works,
        f"Lowest eigenvalue: {min_val}" if min_val is not None else "Failed"
    )
    
    # 4.4: Coupling dependence
    H_weak = SinglePlaquetteHamiltonian(j_max=0.5, g=0.5)
    H_strong = SinglePlaquetteHamiltonian(j_max=0.5, g=2.0)
    
    vals_weak, _ = H_weak.compute_spectrum(k=2)
    vals_strong, _ = H_strong.compute_spectrum(k=2)
    
    # Strong coupling should have larger gap (g² dominates)
    coupling_correct = vals_strong[1] > vals_weak[1]
    
    results.record(
        "Coupling Dependence",
        coupling_correct,
        f"Gap(g=2): {vals_strong[1]-vals_strong[0]:.4f}, Gap(g=0.5): {vals_weak[1]-vals_weak[0]:.4f}"
    )


# =============================================================================
# GAUSS LAW TESTS
# =============================================================================

def test_gauss_law(results: GateResults):
    """Test 5: Gauss law and gauge invariance."""
    
    print("\n--- Test 5: Gauss Law ---")
    
    H = TruncatedHilbertSpace(j_max=0.5)
    gauss = SinglePlaquetteGauss(H)
    
    # 5.1: Trivial state satisfies Gauss law
    dim = H.total_dim ** 4
    trivial = np.zeros(dim)
    trivial[0] = 1.0
    
    G2 = gauss.total_gauss_squared()
    violation = np.real(trivial @ G2 @ trivial)
    
    results.record(
        "Trivial State Gauge Invariant",
        abs(violation) < 1e-14,
        f"⟨0|G²|0⟩ = {violation:.2e}"
    )
    
    # 5.2: Gauss operator structure - use j_max=1.0 for richer space
    H1 = TruncatedHilbertSpace(j_max=1.0)
    gauss1 = SinglePlaquetteGauss(H1)
    G0 = gauss1.gauss_at_site(0)
    
    # For j_max > 0, E operators should have structure
    # The Gauss operator combines E ops on multiple links
    # Check that the total Gauss squared has non-trivial entries
    G2_check = G0.G_squared()
    has_structure = G2_check.nnz > 0
    
    results.record(
        "Gauss Operator Structure (j_max=1.0)",
        has_structure,
        f"G² non-zeros: {G2_check.nnz}"
    )


# =============================================================================
# MAIN GATE 1 TEST
# =============================================================================

def run_gate1():
    """
    Execute Gate 1: Lattice Construction.
    
    This is the first checkpoint in the Yang-Mills Battle Plan.
    """
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "GATE 1: LATTICE CONSTRUCTION" + " " * 20 + "║")
    print("║" + " " * 20 + "Yang-Mills Battle Plan" + " " * 26 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print(f"Execution Time: {datetime.now().isoformat()}")
    print(f"Tolerance: 10^{{-14}} for algebraic tests")
    
    results = GateResults()
    
    start = time.time()
    
    # Run all test suites
    test_su2_algebra(results)
    test_lattice_geometry(results)
    test_quantum_operators(results)
    test_hamiltonian(results)
    test_gauss_law(results)
    
    elapsed = time.time() - start
    
    print()
    print(results.summary())
    print(f"\nExecution time: {elapsed:.2f} seconds")
    
    # Return exit code
    return 0 if results.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_gate1())
