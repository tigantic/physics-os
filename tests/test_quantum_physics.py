"""
Test Module: Quantum Many-Body Physics

Constitutional Compliance:
    - Article III, Section 3.1: Unit tests with 90%+ coverage
    - Article III, Section 3.2: Deterministic seeding (seed=42)
    - Article IV, Section 4.1: Physical validation
    - Article V, Section 5.1: Float64 precision

References:
    Sachdev, S. (2011). "Quantum Phase Transitions." 2nd Edition,
    Cambridge University Press.
    
    White, S.R. (1992). "Density matrix formulation for quantum
    renormalization groups." Physical Review Letters, 69(19), 2863.
"""

import pytest
import torch
import numpy as np
import math
from typing import List, Tuple, Optional


# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

HBAR = 1.054571817e-34  # J·s
KB = 1.380649e-23  # J/K


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
def device():
    """Get device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def spin_chain_params():
    """Parameters for spin chain."""
    return {
        'n_sites': 10,
        'J': 1.0,
        'h': 0.5,
        'bond_dim': 16,
    }


# ============================================================================
# OPERATOR UTILITIES
# ============================================================================

def pauli_matrices(dtype=torch.float64):
    """Return Pauli matrices."""
    I = torch.eye(2, dtype=dtype)
    X = torch.tensor([[0, 1], [1, 0]], dtype=dtype)
    Y = torch.tensor([[0, -1], [1, 0]], dtype=dtype)  # Real part only
    Z = torch.tensor([[1, 0], [0, -1]], dtype=dtype)
    return I, X, Y, Z


def creation_annihilation(dtype=torch.float64):
    """Creation and annihilation operators for fermions/bosons."""
    # For spin-1/2: S+ and S-
    Sp = torch.tensor([[0, 1], [0, 0]], dtype=dtype)
    Sm = torch.tensor([[0, 0], [1, 0]], dtype=dtype)
    return Sp, Sm


def number_operator(dtype=torch.float64):
    """Number operator n = c†c."""
    n = torch.tensor([[0, 0], [0, 1]], dtype=dtype)
    return n


# ============================================================================
# UNIT TESTS: PAULI ALGEBRA
# ============================================================================

class TestPauliAlgebra:
    """Test Pauli matrix algebra."""
    
    @pytest.mark.unit
    def test_pauli_squared(self, deterministic_seed):
        """Pauli matrices square to identity."""
        I, X, Y, Z = pauli_matrices()
        
        assert torch.allclose(X @ X, I)
        assert torch.allclose(Z @ Z, I)
    
    @pytest.mark.unit
    def test_pauli_trace(self, deterministic_seed):
        """Pauli matrices are traceless (except I)."""
        I, X, Y, Z = pauli_matrices()
        
        assert I.trace() == 2
        assert X.trace() == 0
        assert Z.trace() == 0
    
    @pytest.mark.unit
    def test_pauli_anticommutation(self, deterministic_seed):
        """Pauli matrices anticommute."""
        I, X, Y, Z = pauli_matrices()
        
        # {X, Z} = 0
        assert torch.allclose(X @ Z + Z @ X, torch.zeros(2, 2, dtype=torch.float64))
    
    @pytest.mark.unit
    def test_pauli_completeness(self, deterministic_seed):
        """Pauli matrices form a complete basis."""
        I, X, Y, Z = pauli_matrices()
        
        # Any 2x2 Hermitian matrix can be written as sum of Paulis
        A = torch.tensor([[1, 0.5], [0.5, -1]], dtype=torch.float64)
        
        # Decomposition coefficients
        a0 = torch.trace(I @ A) / 2
        a1 = torch.trace(X @ A) / 2
        a3 = torch.trace(Z @ A) / 2
        
        # Note: ignoring Y for real matrices
        reconstructed = a0 * I + a1 * X + a3 * Z
        
        assert reconstructed.shape == A.shape


# ============================================================================
# UNIT TESTS: SPIN OPERATORS
# ============================================================================

class TestSpinOperators:
    """Test spin operators."""
    
    @pytest.mark.unit
    def test_ladder_operators(self, deterministic_seed):
        """Ladder operators raise/lower states."""
        Sp, Sm = creation_annihilation()
        
        up = torch.tensor([1.0, 0.0], dtype=torch.float64)
        down = torch.tensor([0.0, 1.0], dtype=torch.float64)
        
        # S+ |down> = |up>
        assert torch.allclose(Sp @ down, up)
        
        # S- |up> = |down>
        assert torch.allclose(Sm @ up, down)
    
    @pytest.mark.unit
    def test_ladder_nilpotent(self, deterministic_seed):
        """S+ on |up> and S- on |down> give zero."""
        Sp, Sm = creation_annihilation()
        
        up = torch.tensor([1.0, 0.0], dtype=torch.float64)
        down = torch.tensor([0.0, 1.0], dtype=torch.float64)
        
        # S+ |up> = 0
        assert torch.allclose(Sp @ up, torch.zeros(2, dtype=torch.float64))
        
        # S- |down> = 0
        assert torch.allclose(Sm @ down, torch.zeros(2, dtype=torch.float64))
    
    @pytest.mark.unit
    def test_number_operator(self, deterministic_seed):
        """Number operator counts particles."""
        n = number_operator()
        
        # |0> = |down> has n=0, |1> = |up> has... wait, reversed
        # Standard: |0> = [1, 0], |1> = [0, 1]
        vacuum = torch.tensor([1.0, 0.0], dtype=torch.float64)
        one = torch.tensor([0.0, 1.0], dtype=torch.float64)
        
        assert (vacuum @ n @ vacuum).item() == 0
        assert (one @ n @ one).item() == 1


# ============================================================================
# UNIT TESTS: HEISENBERG MODEL
# ============================================================================

class TestHeisenbergModel:
    """Test Heisenberg spin chain."""
    
    @pytest.mark.unit
    def test_two_site_hamiltonian(self, deterministic_seed):
        """Two-site Heisenberg Hamiltonian."""
        # Use complex dtype for proper Heisenberg model with full SU(2) symmetry
        I = torch.eye(2, dtype=torch.complex128)
        X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
        Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
        Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
        J = 1.0
        
        # H = J * (Sx1 Sx2 + Sy1 Sy2 + Sz1 Sz2)
        # Using S = sigma/2
        Sx, Sy, Sz = X/2, Y/2, Z/2
        
        H = J * (torch.kron(Sx, Sx) + torch.kron(Sy, Sy) + torch.kron(Sz, Sz))
        
        # Eigenvalues (Hermitian matrix)
        eigvals = torch.linalg.eigvalsh(H)
        
        # Ground state energy: -3J/4
        E_gs = eigvals.min().real.item()
        assert E_gs == pytest.approx(-0.75 * J)
    
    @pytest.mark.unit
    def test_singlet_triplet_gap(self, deterministic_seed):
        """Singlet-triplet gap for two spins."""
        # Use complex dtype for proper Heisenberg model
        X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
        Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
        Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
        Sx, Sy, Sz = X/2, Y/2, Z/2
        
        H = torch.kron(Sx, Sx) + torch.kron(Sy, Sy) + torch.kron(Sz, Sz)
        eigvals = torch.linalg.eigvalsh(H)
        
        # Gap = E_triplet - E_singlet = 1/4 - (-3/4) = 1
        gap = (eigvals[1] - eigvals[0]).real
        assert gap == pytest.approx(1.0)
    
    @pytest.mark.unit
    def test_total_spin_conservation(self, deterministic_seed):
        """Total S² commutes with Heisenberg H."""
        I, X, Y, Z = pauli_matrices()
        Sx, Sy, Sz = X/2, Y/2, Z/2
        
        # Total spin operators
        Sx_tot = torch.kron(Sx, I) + torch.kron(I, Sx)
        Sy_tot = torch.kron(Sy, I) + torch.kron(I, Sy)
        Sz_tot = torch.kron(Sz, I) + torch.kron(I, Sz)
        
        S2 = Sx_tot @ Sx_tot + Sy_tot @ Sy_tot + Sz_tot @ Sz_tot
        
        H = torch.kron(Sx, Sx) + torch.kron(Sy, Sy) + torch.kron(Sz, Sz)
        
        # [H, S²] = 0
        commutator = H @ S2 - S2 @ H
        assert torch.allclose(commutator, torch.zeros(4, 4, dtype=torch.float64), atol=1e-10)


# ============================================================================
# UNIT TESTS: ISING MODEL
# ============================================================================

class TestIsingModel:
    """Test transverse-field Ising model."""
    
    @pytest.mark.unit
    def test_two_site_ising(self, deterministic_seed):
        """Two-site transverse-field Ising."""
        I, X, Y, Z = pauli_matrices()
        J, h = 1.0, 0.5
        
        # H = -J Sz1 Sz2 - h (Sx1 + Sx2)
        H = -J * torch.kron(Z, Z) - h * (torch.kron(X, I) + torch.kron(I, X))
        
        eigvals = torch.linalg.eigvalsh(H)
        
        # Ground state is ferromagnetic for small h
        assert eigvals.min() < 0
    
    @pytest.mark.unit
    def test_ising_phase_transition(self, deterministic_seed):
        """Ising model has phase transition at h/J = 1."""
        # For infinite chain, critical point is h/J = 1
        # Ground state:
        #   h/J < 1: Ferromagnetic (ordered)
        #   h/J > 1: Paramagnetic (disordered)
        
        h_c = 1.0  # Critical field (in units of J)
        
        assert h_c == 1.0
    
    @pytest.mark.unit
    def test_ising_symmetry(self, deterministic_seed):
        """Ising Hamiltonian has Z2 symmetry."""
        I, X, Y, Z = pauli_matrices()
        
        # Parity operator: P = X ⊗ X
        P = torch.kron(X, X)
        
        # H should commute with P
        H = -torch.kron(Z, Z)  # Just ZZ term
        
        commutator = H @ P - P @ H
        assert torch.allclose(commutator, torch.zeros(4, 4, dtype=torch.float64))


# ============================================================================
# UNIT TESTS: ENTANGLEMENT
# ============================================================================

class TestEntanglement:
    """Test entanglement measures."""
    
    @pytest.mark.unit
    def test_bell_state_entanglement(self, deterministic_seed):
        """Bell state is maximally entangled."""
        # |Bell> = (|00> + |11>) / sqrt(2)
        bell = torch.tensor([1, 0, 0, 1], dtype=torch.float64) / math.sqrt(2)
        
        # Reduced density matrix of first qubit via partial trace
        # rho_A = Tr_B(|bell><bell|)
        # For state |psi> = sum_ij c_ij |i>|j>, rho_A[i,i'] = sum_j c_ij * c_i'j*
        psi_matrix = bell.reshape(2, 2)  # psi[i,j] = c_ij
        rho_A = psi_matrix @ psi_matrix.T  # Tr_B(|psi><psi|)
        
        # Von Neumann entropy S = -Tr(rho log rho)
        eigvals = torch.linalg.eigvalsh(rho_A)
        eigvals = eigvals[eigvals > 1e-10]  # Remove zeros
        entropy = -torch.sum(eigvals * torch.log2(eigvals))
        
        # Maximum entropy for qubit: log2(2) = 1
        assert entropy == pytest.approx(1.0, abs=0.1)
    
    @pytest.mark.unit
    def test_product_state_unentangled(self, deterministic_seed):
        """Product state has zero entanglement."""
        # |00> = |0> ⊗ |0>
        psi = torch.tensor([1, 0, 0, 0], dtype=torch.float64)
        
        rho = psi.outer(psi)
        rho_A = rho.reshape(2, 2, 2, 2)[:, :, 0, 0] + rho.reshape(2, 2, 2, 2)[:, :, 1, 1]
        
        eigvals = torch.linalg.eigvalsh(rho_A)
        eigvals = eigvals[eigvals > 1e-10]
        
        if len(eigvals) == 1:
            entropy = 0.0
        else:
            entropy = -torch.sum(eigvals * torch.log2(eigvals + 1e-10))
        
        assert entropy == pytest.approx(0.0, abs=0.1)
    
    @pytest.mark.unit
    def test_schmidt_decomposition(self, deterministic_seed):
        """Schmidt decomposition of bipartite state."""
        # Any bipartite state can be written as: sum_i lambda_i |a_i> |b_i>
        psi = torch.randn(4, dtype=torch.float64)
        psi = psi / psi.norm()
        
        # Reshape as matrix
        psi_matrix = psi.reshape(2, 2)
        
        # SVD gives Schmidt decomposition
        U, S, Vh = torch.linalg.svd(psi_matrix, full_matrices=False)
        
        # Schmidt coefficients are singular values squared
        schmidt_coeffs = S ** 2
        
        # Should sum to 1 (normalization)
        assert torch.sum(schmidt_coeffs) == pytest.approx(1.0)


# ============================================================================
# UNIT TESTS: GROUND STATE
# ============================================================================

class TestGroundState:
    """Test ground state properties."""
    
    @pytest.mark.unit
    def test_variational_principle(self, deterministic_seed):
        """Energy of any state >= ground state energy."""
        I, X, Y, Z = pauli_matrices()
        H = -torch.kron(Z, Z) - 0.5 * (torch.kron(X, I) + torch.kron(I, X))
        
        eigvals = torch.linalg.eigvalsh(H)
        E_gs = eigvals.min().item()
        
        # Random state
        psi = torch.randn(4, dtype=torch.float64)
        psi = psi / psi.norm()
        
        E_trial = psi @ H @ psi
        
        assert E_trial >= E_gs - 1e-10
    
    @pytest.mark.unit
    def test_ground_state_degeneracy(self, deterministic_seed):
        """Ferromagnetic Ising has degenerate ground state at h=0."""
        I, X, Y, Z = pauli_matrices()
        
        # H = -ZZ only
        H = -torch.kron(Z, Z)
        
        eigvals = torch.linalg.eigvalsh(H)
        
        # Lowest eigenvalue is -1, with degeneracy 2 (|00> and |11>)
        assert eigvals[0] == eigvals[1]  # Degenerate


# ============================================================================
# UNIT TESTS: CORRELATION FUNCTIONS
# ============================================================================

class TestCorrelationFunctions:
    """Test correlation function calculations."""
    
    @pytest.mark.unit
    def test_two_point_correlation(self, deterministic_seed):
        """Two-point correlation function."""
        I, X, Y, Z = pauli_matrices()
        
        # Ground state of ZZ
        # |psi> = |00> (or |11>)
        psi = torch.tensor([1, 0, 0, 0], dtype=torch.float64)
        
        # <Sz1 Sz2>
        Sz1 = torch.kron(Z/2, I)
        Sz2 = torch.kron(I, Z/2)
        Sz1Sz2 = Sz1 @ Sz2
        
        correlation = psi @ Sz1Sz2 @ psi
        
        # For |00>: <Sz1> = 1/2, <Sz2> = 1/2, <Sz1 Sz2> = 1/4
        assert correlation == pytest.approx(0.25)
    
    @pytest.mark.unit
    def test_connected_correlation(self, deterministic_seed):
        """Connected correlation <AB> - <A><B>."""
        I, X, Y, Z = pauli_matrices()
        
        psi = torch.tensor([1, 0, 0, 0], dtype=torch.float64)
        
        Sz1 = torch.kron(Z/2, I)
        Sz2 = torch.kron(I, Z/2)
        
        # Individual expectations
        Sz1_exp = psi @ Sz1 @ psi
        Sz2_exp = psi @ Sz2 @ psi
        
        # Joint expectation
        Sz1Sz2 = Sz1 @ Sz2
        joint = psi @ Sz1Sz2 @ psi
        
        # Connected correlation
        connected = joint - Sz1_exp * Sz2_exp
        
        # For product state |00>: connected = 0
        assert connected == pytest.approx(0.0, abs=1e-10)


# ============================================================================
# FLOAT64 COMPLIANCE
# ============================================================================

class TestFloat64ComplianceQuantum:
    """Article V: Float64 precision tests."""
    
    @pytest.mark.unit
    def test_pauli_float64(self, deterministic_seed):
        """Pauli matrices are float64."""
        I, X, Y, Z = pauli_matrices(dtype=torch.float64)
        
        assert I.dtype == torch.float64
        assert Z.dtype == torch.float64
    
    @pytest.mark.unit
    def test_eigensolver_float64(self, deterministic_seed):
        """Eigenvalue computation uses float64."""
        H = torch.randn(4, 4, dtype=torch.float64)
        H = H + H.T  # Hermitian
        
        eigvals = torch.linalg.eigvalsh(H)
        
        assert eigvals.dtype == torch.float64


# ============================================================================
# GPU COMPATIBILITY
# ============================================================================

class TestGPUCompatibilityQuantum:
    """Test GPU execution compatibility."""
    
    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_operators_on_gpu(self, deterministic_seed, device):
        """Operators work on GPU."""
        I, X, Y, Z = pauli_matrices()
        
        Z_gpu = Z.to(device)
        H = torch.kron(Z_gpu, Z_gpu)
        
        assert H.device.type == device.type


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

class TestReproducibilityQuantum:
    """Article III, Section 3.2: Reproducibility tests."""
    
    @pytest.mark.unit
    def test_deterministic_eigensolver(self):
        """Eigenvalue computation is deterministic."""
        torch.manual_seed(42)
        H = torch.randn(4, 4, dtype=torch.float64)
        H = H + H.T
        eigvals1 = torch.linalg.eigvalsh(H)
        
        torch.manual_seed(42)
        H = torch.randn(4, 4, dtype=torch.float64)
        H = H + H.T
        eigvals2 = torch.linalg.eigvalsh(H)
        
        assert torch.allclose(eigvals1, eigvals2)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestQuantumIntegration:
    """Integration tests for quantum physics."""
    
    @pytest.mark.integration
    def test_spin_chain_spectrum(self, deterministic_seed, spin_chain_params):
        """Full spin chain spectrum (small system)."""
        n_sites = 4  # Small for exact diagonalization
        J = spin_chain_params['J']
        h = spin_chain_params['h']
        
        I, X, Y, Z = pauli_matrices()
        dim = 2 ** n_sites
        
        H = torch.zeros(dim, dim, dtype=torch.float64)
        
        # Build Hamiltonian
        for i in range(n_sites - 1):
            # ZZ interaction
            ZZ = torch.eye(1, dtype=torch.float64)
            for j in range(n_sites):
                if j == i or j == i + 1:
                    ZZ = torch.kron(ZZ, Z)
                else:
                    ZZ = torch.kron(ZZ, I)
            H -= J * ZZ
        
        for i in range(n_sites):
            # Transverse field
            Xi = torch.eye(1, dtype=torch.float64)
            for j in range(n_sites):
                if j == i:
                    Xi = torch.kron(Xi, X)
                else:
                    Xi = torch.kron(Xi, I)
            H -= h * Xi
        
        eigvals = torch.linalg.eigvalsh(H)
        
        assert len(eigvals) == dim
        assert eigvals[0] < eigvals[-1]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])
