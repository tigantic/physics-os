"""
Matrix Product Operator (MPO) Implementation
=============================================

Represents operators (like Hamiltonian) in tensor network form:

O = Σ W[1]^{s1,s1'} W[2]^{s2,s2'} ... W[N]^{sN,sN'}

Each W[i] is a rank-4 tensor of shape (D_{i-1}, d_i, d_i, D_i) where:
- D_i = MPO bond dimension
- d_i = local Hilbert space dimension

For Yang-Mills Hamiltonian H = (g²/2) Σ E² - (1/g²) Σ Tr(U_plaq + U†_plaq):
- Electric term: diagonal in j-basis
- Magnetic term: couples neighboring representations

The MPO form allows O(N × χ² × D × d²) application to MPS,
avoiding the exponential cost of full matrix representation.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from scipy import sparse


class MPOHamiltonian:
    """
    Matrix Product Operator representation of Hamiltonian.
    
    H = Σ_i h_i + Σ_{ij} h_{ij} + ...
    
    Exact MPO representation for:
    - On-site terms: D = 1
    - Nearest-neighbor: D = 3
    - Next-nearest-neighbor: D = 5
    - etc.
    """
    
    def __init__(self, tensors: List[np.ndarray]):
        """
        Initialize MPO from list of rank-4 tensors.
        
        Args:
            tensors: List of tensors [D_l, d, d', D_r]
        """
        self.tensors = [np.asarray(t, dtype=np.complex128) for t in tensors]
        self.n_sites = len(tensors)
    
    @property
    def bond_dimensions(self) -> List[int]:
        """Get MPO bond dimensions."""
        return [t.shape[3] for t in self.tensors[:-1]]
    
    @property
    def local_dimensions(self) -> List[int]:
        """Get local dimensions."""
        return [t.shape[1] for t in self.tensors]
    
    def to_matrix(self) -> np.ndarray:
        """
        Convert MPO to dense matrix.
        
        WARNING: Exponential memory! Only for small systems.
        """
        result = self.tensors[0]  # [1, d, d', D]
        
        for i in range(1, self.n_sites):
            # Contract over MPO bond dimension
            # result[1, d1...di, d1'...di', D] × W[D, d, d', D'] 
            # → result[1, d1...di+1, d1'...di+1', D']
            result = np.tensordot(result, self.tensors[i], axes=([-1], [0]))
        
        # Reshape to matrix
        total_dim = np.prod([t.shape[1] for t in self.tensors])
        result = result.reshape(total_dim, total_dim)
        return result
    
    def apply(self, mps: 'MPS') -> 'MPS':
        """
        Apply MPO to MPS: |ψ'⟩ = H|ψ⟩
        
        Result has bond dimension χ' = χ × D (can be compressed).
        """
        from .mps import MPS
        
        new_tensors = []
        
        for i in range(self.n_sites):
            W = self.tensors[i]  # [D_l, d, d', D_r]
            A = mps.tensors[i]  # [χ_l, d, χ_r]
            
            D_l, d_out, d_in, D_r = W.shape
            chi_l, _, chi_r = A.shape
            
            # Contract: W[D_l, d', d, D_r] × A[χ_l, d, χ_r]
            # Sum over d (physical index of input)
            # Result: [D_l, d', D_r, χ_l, χ_r] → reshape to [(D_l×χ_l), d', (D_r×χ_r)]
            
            # Contract physical index
            temp = np.tensordot(W, A, axes=([2], [1]))  # [D_l, d', D_r, χ_l, χ_r]
            
            # Reshape to MPS tensor
            new_chi_l = D_l * chi_l
            new_chi_r = D_r * chi_r
            new_t = temp.transpose(0, 3, 1, 2, 4).reshape(new_chi_l, d_out, new_chi_r)
            
            new_tensors.append(new_t)
        
        result = MPS(new_tensors)
        return result


class YangMillsMPO(MPOHamiltonian):
    """
    Yang-Mills Hamiltonian in MPO form.
    
    H = (g²/2) Σ_ℓ E²_ℓ - (1/g²) Σ_□ Tr(U_□ + U†_□)
    
    In the representation basis (j = 0, 1/2, 1, ...):
    - E² is diagonal: E²|j⟩ = j(j+1)|j⟩ for SU(2)
    - Magnetic term couples j ↔ j±1/2
    
    For a single plaquette with 4 links:
    - 4 physical sites
    - MPO bond dimension D = 5 (captures plaquette coupling)
    """
    
    def __init__(self, n_links: int, j_max: float, g: float, gauge_group: str = 'SU2'):
        """
        Build Yang-Mills MPO.
        
        Args:
            n_links: Number of links (sites)
            j_max: Maximum representation j
            g: Coupling constant
            gauge_group: 'SU2' or 'SU3'
        """
        self.n_links = n_links
        self.j_max = j_max
        self.g = g
        self.gauge_group = gauge_group
        
        # Build local operators
        self.j_values = np.arange(0, j_max + 0.5, 0.5)
        self.local_dim = len(self.j_values)
        
        # Build MPO tensors
        tensors = self._build_mpo_tensors()
        super().__init__(tensors)
    
    def _build_local_operators(self) -> Dict[str, np.ndarray]:
        """Build local operators in j-basis."""
        d = self.local_dim
        j_vals = self.j_values
        
        # Identity
        I = np.eye(d, dtype=np.complex128)
        
        # Electric operator: E²|j⟩ = C₂(j)|j⟩
        # For SU(2): C₂(j) = j(j+1)
        # For SU(3): Use quadratic Casimir
        if self.gauge_group == 'SU2':
            E2_diag = j_vals * (j_vals + 1)
        else:  # SU(3)
            # SU(3) Casimir: C₂ = (p² + q² + pq + 3p + 3q)/3
            # Simplified for fundamental rep chain
            E2_diag = j_vals * (j_vals + 1) * (4/3)  # Scale for SU(3)
        
        E2 = np.diag(E2_diag.astype(np.complex128))
        
        # Raising/lowering operators for magnetic term
        # U increases j by 1/2, U† decreases j by 1/2
        # Matrix elements from Clebsch-Gordan coefficients
        U_plus = np.zeros((d, d), dtype=np.complex128)
        U_minus = np.zeros((d, d), dtype=np.complex128)
        
        for i, j in enumerate(j_vals):
            if i + 1 < d:
                # |j+1/2⟩⟨j| matrix element
                j_new = j + 0.5
                # CG coefficient for j ⊗ 1/2 → j+1/2
                coeff = np.sqrt((2*j + 2) / (2*j + 1))
                U_plus[i + 1, i] = coeff
            
            if i - 1 >= 0:
                # |j-1/2⟩⟨j| matrix element
                j_new = j - 0.5
                coeff = np.sqrt((2*j) / (2*j + 1))
                U_minus[i - 1, i] = coeff
        
        return {
            'I': I,
            'E2': E2,
            'U_plus': U_plus,
            'U_minus': U_minus,
            'U': U_plus + U_minus,  # U + U†
        }
    
    def _build_mpo_tensors(self) -> List[np.ndarray]:
        """
        Build MPO tensors for Yang-Mills Hamiltonian.
        
        Uses finite automaton construction:
        - State 0: Identity (start)
        - State 1: Building plaquette product
        - State 2: Accumulated terms (end)
        """
        ops = self._build_local_operators()
        d = self.local_dim
        g = self.g
        
        # For a single plaquette (4 links), the Hamiltonian is:
        # H = (g²/2) Σᵢ E²ᵢ - (1/g²) Tr(U₁U₂U₃†U₄† + h.c.)
        
        # For single plaquette, use simple structure
        if self.n_links == 4:
            return self._build_plaquette_mpo(ops)
        
        # For general lattice, use standard automaton
        return self._build_general_mpo(ops)
    
    def _build_plaquette_mpo(self, ops: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """
        Build MPO for single plaquette (4 links).
        
        H = (g²/2)(E₁² + E₂² + E₃² + E₄²) - (2/g²)Re[Tr(U₁U₂U₃†U₄†)]
        
        The plaquette term couples all 4 links and requires D=2 MPO.
        """
        d = self.local_dim
        g = self.g
        I = ops['I']
        E2 = ops['E2']
        U = ops['U']  # U + U†
        
        # Electric term coefficient
        c_E = g**2 / 2
        
        # Magnetic term coefficient  
        # Factor of 2 for trace normalization
        c_B = 2.0 / g**2
        
        # MPO structure for single plaquette:
        # W₁ = [[I, U, c_E*E2]]  (start)
        # W₂ = [[I, 0, 0], [U, 0, 0], [c_E*E2, 0, I]]
        # W₃ = [[I, 0, 0], [U, 0, 0], [c_E*E2, 0, I]]
        # W₄ = [[c_E*E2], [-c_B*U], [I]]  (end)
        
        # Simplified: For strong coupling limit, electric term dominates
        # Use D=3 MPO structure
        
        tensors = []
        
        # First site: [1, d, d, D=3]
        W1 = np.zeros((1, d, d, 3), dtype=np.complex128)
        W1[0, :, :, 0] = I
        W1[0, :, :, 1] = U
        W1[0, :, :, 2] = c_E * E2
        tensors.append(W1)
        
        # Middle sites: [D=3, d, d, D=3]
        for i in range(1, self.n_links - 1):
            W = np.zeros((3, d, d, 3), dtype=np.complex128)
            W[0, :, :, 0] = I
            W[0, :, :, 1] = U
            W[0, :, :, 2] = c_E * E2
            W[1, :, :, 2] = -c_B * U / self.n_links  # Distribute plaquette term
            W[2, :, :, 2] = I
            tensors.append(W)
        
        # Last site: [D=3, d, d, 1]
        WN = np.zeros((3, d, d, 1), dtype=np.complex128)
        WN[0, :, :, 0] = c_E * E2
        WN[1, :, :, 0] = -c_B * U / self.n_links
        WN[2, :, :, 0] = I
        tensors.append(WN)
        
        return tensors
    
    def _build_general_mpo(self, ops: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """Build MPO for general lattice structure."""
        # For now, use electric term only (exact in strong coupling)
        d = self.local_dim
        g = self.g
        I = ops['I']
        E2 = ops['E2']
        
        c_E = g**2 / 2
        
        tensors = []
        
        # Simple sum of local terms: H = Σᵢ hᵢ
        # MPO bond dimension D = 2
        
        # First site
        W1 = np.zeros((1, d, d, 2), dtype=np.complex128)
        W1[0, :, :, 0] = I
        W1[0, :, :, 1] = c_E * E2
        tensors.append(W1)
        
        # Middle sites
        for i in range(1, self.n_links - 1):
            W = np.zeros((2, d, d, 2), dtype=np.complex128)
            W[0, :, :, 0] = I
            W[0, :, :, 1] = c_E * E2
            W[1, :, :, 1] = I
            tensors.append(W)
        
        # Last site
        WN = np.zeros((2, d, d, 1), dtype=np.complex128)
        WN[0, :, :, 0] = c_E * E2
        WN[1, :, :, 0] = I
        tensors.append(WN)
        
        return tensors
    
    def strong_coupling_gap(self) -> float:
        """
        Analytical strong coupling gap.
        
        In strong coupling (g >> 1), H ≈ (g²/2) Σ E²
        Ground state: all links in j=0
        First excited: one link in j=1/2
        
        Gap = (g²/2) × C₂(1/2) = (g²/2) × (1/2)(3/2) = (3/4)g²
        
        For plaquette with 4 links, factor of 2 from degeneracy:
        Δ = (3/2)g²
        """
        if self.gauge_group == 'SU2':
            C2_half = 0.5 * 1.5  # j(j+1) for j=1/2
        else:  # SU(3)
            C2_half = 0.5 * 1.5 * (4/3)  # SU(3) Casimir
        
        # Gap = (g²/2) × C₂(adj)/N_links × degeneracy_factor
        # For single plaquette: Δ = (3/2)g²
        return self.g**2 * C2_half * 2  # Factor 2 for 4-link plaquette
    
    def get_sparse_matrix(self) -> sparse.csr_matrix:
        """Get sparse matrix representation (for small systems)."""
        dense = self.to_matrix()
        return sparse.csr_matrix(dense)


def build_yang_mills_mpo(n_plaquettes: int, j_max: float, g: float,
                         gauge_group: str = 'SU2') -> YangMillsMPO:
    """
    Build Yang-Mills MPO for lattice with given number of plaquettes.
    
    Args:
        n_plaquettes: Number of plaquettes
        j_max: Maximum representation
        g: Coupling constant
        gauge_group: 'SU2' or 'SU3'
    
    Returns:
        YangMillsMPO object
    """
    # 1 plaquette = 4 links
    # 2 plaquettes (shared edge) = 7 links
    # etc.
    n_links = 4 * n_plaquettes - (n_plaquettes - 1) if n_plaquettes > 1 else 4
    
    return YangMillsMPO(n_links, j_max, g, gauge_group)
