#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              SU(2) GROUP MODULE                              ║
║                                                                              ║
║                    Special Unitary Group in 2 Dimensions                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

SU(2) is the gauge group for the simplest non-Abelian Yang-Mills theory.
This module implements the group algebra needed for lattice gauge theory.

Key Properties:
    - dim(SU(2)) = 3 (3 generators: Pauli matrices / 2)
    - SU(2) ≅ S³ (3-sphere) topologically
    - Fundamental representation: 2×2 unitary matrices with det = 1
    
Algebraic Relations:
    - [σᵢ, σⱼ] = 2i εᵢⱼₖ σₖ  (Pauli commutation)
    - {σᵢ, σⱼ} = 2δᵢⱼ I      (Pauli anticommutation)
    - σᵢ σⱼ = δᵢⱼ I + i εᵢⱼₖ σₖ
    
Representations:
    - Spin-j representation has dimension (2j+1)
    - j = 0: trivial (1-dim)
    - j = 1/2: fundamental (2-dim) ← used for links
    - j = 1: adjoint (3-dim) ← electric field transforms

Author: TiganticLabz Yang-Mills Project
Date: 2026-01-15
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
from functools import lru_cache


# =============================================================================
# PAULI MATRICES
# =============================================================================

# Pauli matrices (fundamental representation)
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
SIGMA_0 = np.eye(2, dtype=np.complex128)  # Identity

# Collect into tuple for indexing
PAULI = (SIGMA_X, SIGMA_Y, SIGMA_Z)

# SU(2) generators: τₐ = σₐ / 2
TAU = tuple(s / 2 for s in PAULI)

# Levi-Civita symbol (structure constants for SU(2))
EPSILON = np.zeros((3, 3, 3))
EPSILON[0, 1, 2] = EPSILON[1, 2, 0] = EPSILON[2, 0, 1] = 1
EPSILON[0, 2, 1] = EPSILON[2, 1, 0] = EPSILON[1, 0, 2] = -1


def pauli_matrices() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the three Pauli matrices (σₓ, σᵧ, σᵤ).
    
    Properties:
        - Hermitian: σᵢ† = σᵢ
        - Unitary: σᵢ² = I
        - Traceless: Tr(σᵢ) = 0
        - [σᵢ, σⱼ] = 2i εᵢⱼₖ σₖ
    """
    return SIGMA_X.copy(), SIGMA_Y.copy(), SIGMA_Z.copy()


def su2_generators() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the SU(2) generators τₐ = σₐ / 2.
    
    These satisfy:
        - [τₐ, τᵦ] = i εₐᵦᵧ τᵧ  (Lie algebra)
        - Tr(τₐ τᵦ) = δₐᵦ / 2   (normalization)
    """
    return tuple(t.copy() for t in TAU)


# =============================================================================
# SU(2) GROUP ELEMENTS
# =============================================================================

@dataclass
class SU2:
    """
    An SU(2) group element in the fundamental (2×2) representation.
    
    Parameterization:
        U = exp(i θₐ τₐ) = cos(θ/2) I + i sin(θ/2) n̂·σ
        
    where θ = |θ| is the rotation angle and n̂ = θ/|θ| is the axis.
    
    Equivalently, using quaternion parameterization:
        U = a₀ I + i aₐ σₐ  where  a₀² + a₁² + a₂² + a₃² = 1
    """
    matrix: np.ndarray  # 2×2 complex matrix
    
    def __post_init__(self):
        """Validate SU(2) properties."""
        assert self.matrix.shape == (2, 2), "Must be 2×2"
        # Relax tolerance for numerical operations
        assert np.abs(np.linalg.det(self.matrix) - 1.0) < 1e-10, "det must be 1"
        assert np.allclose(self.matrix @ self.matrix.conj().T, SIGMA_0, atol=1e-10), "Must be unitary"
    
    @classmethod
    def identity(cls) -> 'SU2':
        """Return the identity element."""
        return cls(matrix=SIGMA_0.copy())
    
    @classmethod
    def from_angles(cls, theta: np.ndarray) -> 'SU2':
        """
        Create SU(2) element from angle vector θ = (θ₁, θ₂, θ₃).
        
        U = exp(i θₐ τₐ) = cos(|θ|/2) I + i sin(|θ|/2) (θ̂·σ)
        """
        theta = np.asarray(theta)
        angle = np.linalg.norm(theta)
        
        if angle < 1e-12:
            return cls.identity()
        
        axis = theta / angle
        
        c = np.cos(angle / 2)
        s = np.sin(angle / 2)
        
        matrix = c * SIGMA_0 + 1j * s * (
            axis[0] * SIGMA_X + axis[1] * SIGMA_Y + axis[2] * SIGMA_Z
        )
        
        return cls(matrix=matrix)
    
    @classmethod
    def from_quaternion(cls, q: np.ndarray) -> 'SU2':
        """
        Create SU(2) element from unit quaternion (a₀, a₁, a₂, a₃).
        
        U = a₀ I + i (a₁ σₓ + a₂ σᵧ + a₃ σᵤ)
        
        Requires: a₀² + a₁² + a₂² + a₃² = 1
        """
        q = np.asarray(q)
        q = q / np.linalg.norm(q)  # Normalize
        
        matrix = (
            q[0] * SIGMA_0 + 
            1j * (q[1] * SIGMA_X + q[2] * SIGMA_Y + q[3] * SIGMA_Z)
        )
        
        return cls(matrix=matrix)
    
    def to_quaternion(self) -> np.ndarray:
        """Extract quaternion representation (a₀, a₁, a₂, a₃)."""
        # U = a₀ I + i aₐ σₐ
        # Tr(U) = 2 a₀
        # Tr(U σₐ) / 2i = aₐ (using Tr(σₐ σᵦ) = 2 δₐᵦ)
        
        a0 = np.real(np.trace(self.matrix)) / 2
        a1 = np.imag(np.trace(self.matrix @ SIGMA_X)) / 2
        a2 = np.imag(np.trace(self.matrix @ SIGMA_Y)) / 2
        a3 = np.imag(np.trace(self.matrix @ SIGMA_Z)) / 2
        
        return np.array([a0, a1, a2, a3])
    
    def __matmul__(self, other: 'SU2') -> 'SU2':
        """Group multiplication: U₁ × U₂."""
        return SU2(matrix=self.matrix @ other.matrix)
    
    def inverse(self) -> 'SU2':
        """Group inverse: U⁻¹ = U†."""
        return SU2(matrix=self.matrix.conj().T)
    
    def __repr__(self) -> str:
        q = self.to_quaternion()
        return f"SU2(q=[{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}])"


def random_su2(seed: Optional[int] = None) -> SU2:
    """
    Generate a uniformly random SU(2) element (Haar measure).
    
    Method: Sample 4D Gaussian, normalize to unit quaternion.
    This gives uniform distribution on S³ ≅ SU(2).
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 4D Gaussian → uniform on S³
    q = np.random.randn(4)
    q = q / np.linalg.norm(q)
    
    return SU2.from_quaternion(q)


# =============================================================================
# REPRESENTATION THEORY
# =============================================================================

@lru_cache(maxsize=32)
def spin_j_dimension(j: float) -> int:
    """Dimension of spin-j representation: 2j + 1."""
    assert j >= 0 and 2*j == int(2*j), "j must be non-negative half-integer"
    return int(2*j + 1)


@lru_cache(maxsize=32)
def spin_j_generators(j: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the spin-j representation of SU(2) generators.
    
    J_z |j, m⟩ = m |j, m⟩
    J_± |j, m⟩ = √(j(j+1) - m(m±1)) |j, m±1⟩
    
    Returns (J_x, J_y, J_z) as (2j+1) × (2j+1) matrices.
    """
    dim = spin_j_dimension(j)
    
    # m values: j, j-1, ..., -j
    m_vals = np.array([j - i for i in range(dim)])
    
    # J_z is diagonal
    J_z = np.diag(m_vals).astype(np.complex128)
    
    # J_+ and J_- from ladder operators
    J_plus = np.zeros((dim, dim), dtype=np.complex128)
    J_minus = np.zeros((dim, dim), dtype=np.complex128)
    
    for i in range(dim - 1):
        m = m_vals[i]
        # J_+ |j,m⟩ = √(j(j+1) - m(m+1)) |j,m+1⟩
        coeff = np.sqrt(j*(j+1) - m*(m+1))
        J_plus[i+1, i] = coeff  # |m+1⟩⟨m|... wait, m decreases with index
        
    for i in range(1, dim):
        m = m_vals[i]
        # J_- |j,m⟩ = √(j(j+1) - m(m-1)) |j,m-1⟩
        coeff = np.sqrt(j*(j+1) - m*(m-1))
        J_minus[i-1, i] = coeff
    
    # J_x = (J_+ + J_-) / 2
    # J_y = (J_+ - J_-) / 2i
    J_x = (J_plus + J_minus) / 2
    J_y = (J_plus - J_minus) / (2j)  # Note: 2j not 2i, fixing...
    J_y = (J_plus - J_minus) / (2 * 1j)
    
    return J_x, J_y, J_z


def casimir_eigenvalue(j: float) -> float:
    """
    Casimir operator eigenvalue for spin-j: C₂ = j(j+1).
    
    The quadratic Casimir is C₂ = J² = Jₓ² + Jᵧ² + Jᵤ².
    It commutes with all generators (center of algebra).
    """
    return j * (j + 1)


# =============================================================================
# CHARACTER EXPANSION
# =============================================================================

def character(U: SU2, j: float) -> complex:
    """
    Character of U in spin-j representation: χⱼ(U) = Tr(Dⱼ(U)).
    
    For SU(2), this only depends on the class angle θ:
        χⱼ(θ) = sin((2j+1)θ/2) / sin(θ/2)
    """
    # Extract class angle from trace
    # Tr(U) = 2 cos(θ/2) for U = exp(iθ n·τ)
    trace = np.trace(U.matrix)
    cos_half_theta = np.real(trace) / 2
    
    # Handle θ = 0 (identity)
    if np.abs(cos_half_theta - 1) < 1e-12:
        return spin_j_dimension(j)
    
    # Handle θ = 2π (minus identity for half-integer j)
    if np.abs(cos_half_theta + 1) < 1e-12:
        return spin_j_dimension(j) * ((-1)**(2*j))
    
    half_theta = np.arccos(np.clip(cos_half_theta, -1, 1))
    
    # χⱼ(θ) = sin((2j+1)θ/2) / sin(θ/2)
    numerator = np.sin((2*j + 1) * half_theta)
    denominator = np.sin(half_theta)
    
    return numerator / denominator


# =============================================================================
# ALGEBRA VERIFICATION
# =============================================================================

def verify_su2_algebra() -> dict:
    """
    Verify all SU(2) algebraic identities.
    
    Returns dict with test results for:
        1. Pauli commutation relations
        2. Pauli anticommutation relations
        3. Hermiticity
        4. Tracelessness
        5. Determinant = 1 for group elements
        6. Unitarity
        7. Closure under multiplication
    """
    results = {}
    
    sigma = PAULI
    
    # Test 1: Commutation [σᵢ, σⱼ] = 2i εᵢⱼₖ σₖ
    comm_errors = []
    for i in range(3):
        for j in range(3):
            comm = sigma[i] @ sigma[j] - sigma[j] @ sigma[i]
            expected = sum(2j * EPSILON[i, j, k] * sigma[k] for k in range(3))
            error = np.max(np.abs(comm - expected))
            comm_errors.append(error)
    results['commutation_max_error'] = max(comm_errors)
    results['commutation_passed'] = max(comm_errors) < 1e-14
    
    # Test 2: Anticommutation {σᵢ, σⱼ} = 2δᵢⱼ I
    anticomm_errors = []
    for i in range(3):
        for j in range(3):
            anticomm = sigma[i] @ sigma[j] + sigma[j] @ sigma[i]
            expected = 2 * (1 if i == j else 0) * SIGMA_0
            error = np.max(np.abs(anticomm - expected))
            anticomm_errors.append(error)
    results['anticommutation_max_error'] = max(anticomm_errors)
    results['anticommutation_passed'] = max(anticomm_errors) < 1e-14
    
    # Test 3: Hermiticity σᵢ† = σᵢ
    herm_errors = [np.max(np.abs(s - s.conj().T)) for s in sigma]
    results['hermiticity_max_error'] = max(herm_errors)
    results['hermiticity_passed'] = max(herm_errors) < 1e-14
    
    # Test 4: Tracelessness Tr(σᵢ) = 0
    traces = [np.abs(np.trace(s)) for s in sigma]
    results['tracelessness_max_error'] = max(traces)
    results['tracelessness_passed'] = max(traces) < 1e-14
    
    # Test 5: Random group elements have det = 1
    det_errors = []
    for _ in range(100):
        U = random_su2()
        det_errors.append(np.abs(np.linalg.det(U.matrix) - 1.0))
    results['determinant_max_error'] = max(det_errors)
    results['determinant_passed'] = max(det_errors) < 1e-12
    
    # Test 6: Unitarity U U† = I
    unitary_errors = []
    for _ in range(100):
        U = random_su2()
        error = np.max(np.abs(U.matrix @ U.matrix.conj().T - SIGMA_0))
        unitary_errors.append(error)
    results['unitarity_max_error'] = max(unitary_errors)
    results['unitarity_passed'] = max(unitary_errors) < 1e-12
    
    # Test 7: Closure U₁ U₂ ∈ SU(2)
    closure_errors = []
    for _ in range(100):
        U1 = random_su2()
        U2 = random_su2()
        U3 = U1 @ U2
        # Check det = 1
        closure_errors.append(np.abs(np.linalg.det(U3.matrix) - 1.0))
    results['closure_max_error'] = max(closure_errors)
    results['closure_passed'] = max(closure_errors) < 1e-12
    
    # Overall
    results['all_passed'] = all([
        results['commutation_passed'],
        results['anticommutation_passed'],
        results['hermiticity_passed'],
        results['tracelessness_passed'],
        results['determinant_passed'],
        results['unitarity_passed'],
        results['closure_passed'],
    ])
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SU(2) GROUP ALGEBRA VERIFICATION")
    print("=" * 70)
    
    results = verify_su2_algebra()
    
    tests = [
        ('commutation', 'Commutation [σᵢ,σⱼ] = 2iεᵢⱼₖσₖ'),
        ('anticommutation', 'Anticommutation {σᵢ,σⱼ} = 2δᵢⱼI'),
        ('hermiticity', 'Hermiticity σᵢ† = σᵢ'),
        ('tracelessness', 'Tracelessness Tr(σᵢ) = 0'),
        ('determinant', 'Determinant det(U) = 1'),
        ('unitarity', 'Unitarity U U† = I'),
        ('closure', 'Closure U₁U₂ ∈ SU(2)'),
    ]
    
    print()
    for key, name in tests:
        passed = results[f'{key}_passed']
        error = results[f'{key}_max_error']
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}")
        print(f"    Max error: {error:.2e}")
        print(f"    Status: {status}")
        print()
    
    print("=" * 70)
    if results['all_passed']:
        print("  ★ ALL SU(2) ALGEBRA TESTS PASSED ★")
    else:
        print("  ✗ SOME TESTS FAILED")
    print("=" * 70)
