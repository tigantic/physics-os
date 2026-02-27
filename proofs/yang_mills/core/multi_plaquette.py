"""
Multi-Plaquette Yang-Mills Lattice Hamiltonian

The single plaquette gives Δ = (3/2)g² which vanishes as g → 0.
To capture dimensional transmutation, we need spatial extent.

For an Lx × Ly lattice:
- Number of plaquettes: Lx × Ly
- Number of links: 2*Lx*Ly + Lx + Ly (with PBC: 2*Lx*Ly)
- Gauss constraints: One per vertex (Lx × Ly vertices)

The Hamiltonian:
    H = (g²/2) Σ_links E²_link + (1/g²) Σ_plaquettes (1 - Re Tr U_plaquette)

Key physics:
- Adjacent plaquettes share links → correlations
- Gauss law at each vertex → gauge invariance
- In 2D, confinement is linear for all couplings
- Mass gap should show different scaling than single plaquette
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from functools import lru_cache
import itertools


@dataclass
class LatticeGeometry:
    """2D square lattice geometry with periodic boundary conditions."""
    Lx: int
    Ly: int
    
    @property
    def num_sites(self) -> int:
        return self.Lx * self.Ly
    
    @property
    def num_links(self) -> int:
        """Number of links with periodic boundary conditions."""
        return 2 * self.Lx * self.Ly  # Each site has one x-link and one y-link
    
    @property
    def num_plaquettes(self) -> int:
        return self.Lx * self.Ly
    
    def site_index(self, x: int, y: int) -> int:
        """Convert (x, y) to linear site index."""
        return (x % self.Lx) + (y % self.Ly) * self.Lx
    
    def site_coords(self, idx: int) -> Tuple[int, int]:
        """Convert linear index to (x, y) coordinates."""
        return idx % self.Lx, idx // self.Lx
    
    def link_index(self, x: int, y: int, direction: int) -> int:
        """
        Get link index for link starting at (x, y) in direction (0=x, 1=y).
        Links are numbered: first all x-links, then all y-links.
        """
        site = self.site_index(x, y)
        if direction == 0:  # x-direction
            return site
        else:  # y-direction
            return self.num_sites + site
    
    def plaquette_links(self, x: int, y: int) -> Tuple[int, int, int, int]:
        """
        Return the 4 links forming the plaquette at (x, y).
        Returns (link1, link2, link3, link4) going counterclockwise:
        - link1: (x, y) → (x+1, y)  [x-direction]
        - link2: (x+1, y) → (x+1, y+1)  [y-direction]
        - link3: (x+1, y+1) → (x, y+1)  [x-direction, reversed]
        - link4: (x, y+1) → (x, y)  [y-direction, reversed]
        
        Returns tuple of (link_idx, orientation) where orientation = +1 or -1.
        """
        l1 = self.link_index(x, y, 0)  # Bottom
        l2 = self.link_index((x + 1) % self.Lx, y, 1)  # Right
        l3 = self.link_index(x, (y + 1) % self.Ly, 0)  # Top (reversed)
        l4 = self.link_index(x, y, 1)  # Left (reversed)
        return (l1, l2, l3, l4)
    
    def vertex_links(self, x: int, y: int) -> List[Tuple[int, int]]:
        """
        Return links attached to vertex (x, y) with their orientations.
        Gauss law: Σ_outgoing E - Σ_incoming E = 0
        
        Returns list of (link_idx, sign) where sign = +1 for outgoing, -1 for incoming.
        """
        links = []
        # Outgoing links (start at this vertex)
        links.append((self.link_index(x, y, 0), +1))  # x-direction outgoing
        links.append((self.link_index(x, y, 1), +1))  # y-direction outgoing
        # Incoming links (end at this vertex)
        links.append((self.link_index((x - 1) % self.Lx, y, 0), -1))  # x-direction incoming
        links.append((self.link_index(x, (y - 1) % self.Ly, 1), -1))  # y-direction incoming
        return links


class SU2Representation:
    """SU(2) representation theory utilities."""
    
    def __init__(self, j_max: float = 0.5):
        """
        j_max: Maximum spin to include in truncation.
        For SU(2), j = 0, 1/2, 1, 3/2, ...
        """
        self.j_max = j_max
        self.j_values = np.arange(0, j_max + 0.5, 0.5)
        
        # Build state space
        self.states = []  # List of (j, m) tuples
        for j in self.j_values:
            for m in np.arange(-j, j + 1):
                self.states.append((j, m))
        
        self.dim = len(self.states)
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
    
    def casimir(self, j: float) -> float:
        """Quadratic Casimir C₂(j) = j(j+1)."""
        return j * (j + 1)
    
    def dimension(self, j: float) -> int:
        """Dimension of spin-j representation: 2j + 1."""
        return int(2 * j + 1)
    
    def build_E_squared(self) -> np.ndarray:
        """Build E² operator in the basis of (j, m) states."""
        E2 = np.zeros((self.dim, self.dim))
        for i, (j, m) in enumerate(self.states):
            E2[i, i] = self.casimir(j)
        return E2
    
    def build_J_plus(self) -> np.ndarray:
        """Build J+ operator."""
        Jp = np.zeros((self.dim, self.dim))
        for i, (j, m) in enumerate(self.states):
            if m < j:
                m_new = m + 1
                new_state = (j, m_new)
                if new_state in self.state_to_idx:
                    j_idx = self.state_to_idx[new_state]
                    # J+ |j,m⟩ = √(j-m)(j+m+1) |j,m+1⟩
                    Jp[j_idx, i] = np.sqrt((j - m) * (j + m + 1))
        return Jp
    
    def build_J_minus(self) -> np.ndarray:
        """Build J- operator."""
        Jm = np.zeros((self.dim, self.dim))
        for i, (j, m) in enumerate(self.states):
            if m > -j:
                m_new = m - 1
                new_state = (j, m_new)
                if new_state in self.state_to_idx:
                    j_idx = self.state_to_idx[new_state]
                    # J- |j,m⟩ = √(j+m)(j-m+1) |j,m-1⟩
                    Jm[j_idx, i] = np.sqrt((j + m) * (j - m + 1))
        return Jm
    
    def build_J_z(self) -> np.ndarray:
        """Build Jz operator."""
        Jz = np.zeros((self.dim, self.dim))
        for i, (j, m) in enumerate(self.states):
            Jz[i, i] = m
        return Jz


class MultiPlaquetteHamiltonian:
    """
    Yang-Mills Hamiltonian on a 2D lattice with multiple plaquettes.
    
    H = (g²/2) Σ_links E²_link + (1/g²) Σ_plaq (1 - ½ Tr U_plaq)
    
    The plaquette term requires careful treatment of the group integration.
    In the strong coupling expansion, we use the character expansion.
    """
    
    def __init__(self, Lx: int, Ly: int, g: float = 1.0, j_max: float = 0.5):
        self.geometry = LatticeGeometry(Lx, Ly)
        self.g = g
        self.j_max = j_max
        self.rep = SU2Representation(j_max)
        
        # Compute total Hilbert space dimension
        self.link_dim = self.rep.dim
        self.num_links = self.geometry.num_links
        
        # For small lattices, compute full dimension
        self.total_dim = self.link_dim ** self.num_links
        
        print(f"Lattice: {Lx}×{Ly}")
        print(f"Links: {self.num_links}")
        print(f"Link Hilbert space dim: {self.link_dim}")
        print(f"Total Hilbert space dim: {self.total_dim}")
    
    def _state_to_indices(self, state_idx: int) -> List[int]:
        """Convert total state index to per-link indices."""
        indices = []
        remaining = state_idx
        for _ in range(self.num_links):
            indices.append(remaining % self.link_dim)
            remaining //= self.link_dim
        return indices
    
    def _indices_to_state(self, indices: List[int]) -> int:
        """Convert per-link indices to total state index."""
        state_idx = 0
        multiplier = 1
        for idx in indices:
            state_idx += idx * multiplier
            multiplier *= self.link_dim
        return state_idx
    
    def _get_link_state(self, state_idx: int, link: int) -> Tuple[float, float]:
        """Get (j, m) for a specific link in a given state."""
        indices = self._state_to_indices(state_idx)
        return self.rep.states[indices[link]]
    
    def build_electric_term(self) -> sparse.csr_matrix:
        """
        Build the electric term: (g²/2) Σ_links E²_link
        This is diagonal in the representation basis.
        """
        diag = np.zeros(self.total_dim)
        
        for state_idx in range(self.total_dim):
            indices = self._state_to_indices(state_idx)
            energy = 0.0
            for link in range(self.num_links):
                j, m = self.rep.states[indices[link]]
                energy += self.rep.casimir(j)
            diag[state_idx] = (self.g**2 / 2) * energy
        
        return sparse.diags(diag, format='csr')
    
    def build_plaquette_term(self) -> sparse.csr_matrix:
        """
        Build the magnetic (plaquette) term: (1/g²) Σ_plaq (1 - ½ Tr U_plaq)
        
        Using the character expansion of the plaquette operator.
        For a single plaquette: Tr U = Σ_j d_j χ_j(U)
        
        The plaquette operator changes the representation labels of the 4 links.
        """
        rows, cols, data = [], [], []
        
        # For each plaquette
        for px in range(self.geometry.Lx):
            for py in range(self.geometry.Ly):
                plaq_links = self.geometry.plaquette_links(px, py)
                
                # Add diagonal term (constant offset)
                for state_idx in range(self.total_dim):
                    rows.append(state_idx)
                    cols.append(state_idx)
                    data.append(1.0 / self.g**2)
                
                # Plaquette transitions (off-diagonal)
                self._add_plaquette_transitions(plaq_links, rows, cols, data)
        
        return sparse.csr_matrix((data, (rows, cols)), 
                                  shape=(self.total_dim, self.total_dim))
    
    def _add_plaquette_transitions(self, plaq_links: Tuple[int, int, int, int],
                                    rows: List, cols: List, data: List):
        """
        Add matrix elements from the plaquette operator.
        
        The plaquette term couples states where all 4 links change by Δj = ±1/2.
        This implements the leading order in the strong coupling expansion.
        """
        l1, l2, l3, l4 = plaq_links
        
        # For simplicity, implement the diagonal part of Re(Tr U)
        # which gives the leading strong-coupling behavior
        for state_idx in range(self.total_dim):
            indices = self._state_to_indices(state_idx)
            
            j1, m1 = self.rep.states[indices[l1]]
            j2, m2 = self.rep.states[indices[l2]]
            j3, m3 = self.rep.states[indices[l3]]
            j4, m4 = self.rep.states[indices[l4]]
            
            # Plaquette gives dimension factor when all links have same j
            # and m values sum to zero around the plaquette
            if j1 == j2 == j3 == j4:
                # Check magnetic quantum number constraint
                # m1 - m2 + m3 - m4 = 0 (going around the plaquette)
                if abs(m1 - m2 + m3 - m4) < 1e-10:
                    # Character contribution: χ_j(U) / d_j
                    d_j = 2 * j1 + 1
                    if d_j > 0:
                        matrix_element = -1.0 / (2 * self.g**2 * d_j)
                        rows.append(state_idx)
                        cols.append(state_idx)
                        data.append(matrix_element)
    
    def build_gauss_projector(self) -> sparse.csr_matrix:
        """
        Build projector onto gauge-invariant (physical) subspace.
        
        At each vertex: Σ_outgoing E^a - Σ_incoming E^a = 0
        
        This is enforced by requiring that angular momenta at each vertex
        can be coupled to zero total angular momentum.
        """
        physical_states = []
        
        for state_idx in range(self.total_dim):
            if self._is_gauge_invariant(state_idx):
                physical_states.append(state_idx)
        
        # Build projector
        n_phys = len(physical_states)
        if n_phys == 0:
            return sparse.csr_matrix((self.total_dim, self.total_dim))
        
        # P = Σ_i |i⟩⟨i| for physical states
        diag = np.zeros(self.total_dim)
        for idx in physical_states:
            diag[idx] = 1.0
        
        return sparse.diags(diag, format='csr')
    
    def _is_gauge_invariant(self, state_idx: int) -> bool:
        """Check if a state satisfies Gauss law at all vertices."""
        indices = self._state_to_indices(state_idx)
        
        for vx in range(self.geometry.Lx):
            for vy in range(self.geometry.Ly):
                vertex_links = self.geometry.vertex_links(vx, vy)
                
                # Get angular momenta at this vertex
                j_values = []
                for link_idx, sign in vertex_links:
                    j, m = self.rep.states[indices[link_idx]]
                    j_values.append(j)
                
                # Check if 4 angular momenta can couple to zero
                # For j1 ⊗ j2 ⊗ j3 ⊗ j4, need |j1-j2| ≤ j12 ≤ j1+j2, etc.
                if not self._can_couple_to_zero(j_values):
                    return False
                
                # Also check m values sum to zero
                m_sum = 0.0
                for link_idx, sign in vertex_links:
                    j, m = self.rep.states[indices[link_idx]]
                    m_sum += sign * m
                
                if abs(m_sum) > 1e-10:
                    return False
        
        return True
    
    def _can_couple_to_zero(self, j_values: List[float]) -> bool:
        """Check if angular momenta can couple to total J = 0."""
        if len(j_values) != 4:
            return False
        
        j1, j2, j3, j4 = j_values
        
        # Couple j1 ⊗ j2 → j12, then j12 ⊗ j3 → j123, then j123 ⊗ j4 → 0
        # For final coupling to 0, need j123 = j4
        
        for j12 in np.arange(abs(j1 - j2), j1 + j2 + 0.5, 1.0):
            for j123 in np.arange(abs(j12 - j3), j12 + j3 + 0.5, 1.0):
                if abs(j123 - j4) < 1e-10 and j123 <= j4:
                    return True
        
        return False
    
    def build_hamiltonian(self) -> sparse.csr_matrix:
        """Build the full Hamiltonian."""
        print("Building electric term...")
        H_E = self.build_electric_term()
        print("Building plaquette term...")
        H_B = self.build_plaquette_term()
        return H_E + H_B
    
    def get_physical_subspace(self) -> Tuple[sparse.csr_matrix, List[int]]:
        """
        Return the Hamiltonian restricted to the physical subspace.
        Also returns the list of physical state indices.
        """
        physical_states = []
        for state_idx in range(self.total_dim):
            if self._is_gauge_invariant(state_idx):
                physical_states.append(state_idx)
        
        n_phys = len(physical_states)
        print(f"Physical states: {n_phys}/{self.total_dim}")
        
        if n_phys == 0:
            return None, []
        
        # Build restriction matrices
        # R: total space → physical subspace
        R = sparse.csr_matrix((np.ones(n_phys), 
                               (np.arange(n_phys), physical_states)),
                              shape=(n_phys, self.total_dim))
        
        H_full = self.build_hamiltonian()
        H_phys = R @ H_full @ R.T
        
        return H_phys, physical_states
    
    def compute_gap(self, n_states: int = 3) -> Dict:
        """Compute the mass gap in the physical subspace."""
        H_phys, physical_states = self.get_physical_subspace()
        
        if H_phys is None or len(physical_states) < 2:
            return {
                'gap': None,
                'E0': None,
                'E1': None,
                'n_physical': len(physical_states) if physical_states else 0
            }
        
        n_phys = len(physical_states)
        k = min(n_states, n_phys - 1)
        
        if n_phys <= 10:
            # Full diagonalization for small systems
            H_dense = H_phys.toarray()
            eigenvalues = np.linalg.eigvalsh(H_dense)
        else:
            # Sparse eigenvalue computation
            eigenvalues, _ = eigsh(H_phys, k=k + 1, which='SA')
            eigenvalues = np.sort(eigenvalues)
        
        E0 = eigenvalues[0]
        E1 = eigenvalues[1] if len(eigenvalues) > 1 else E0
        gap = E1 - E0
        
        return {
            'gap': gap,
            'E0': E0,
            'E1': E1,
            'n_physical': n_phys,
            'spectrum': eigenvalues[:min(5, len(eigenvalues))]
        }


def compute_multi_plaquette_gap(Lx: int, Ly: int, g: float, j_max: float = 0.5) -> Dict:
    """
    Compute the mass gap for an Lx × Ly lattice.
    
    This is the key function for studying how the gap scales with lattice size.
    """
    ham = MultiPlaquetteHamiltonian(Lx, Ly, g, j_max)
    return ham.compute_gap()


def finite_size_scaling(g: float = 1.0, j_max: float = 0.5, max_L: int = 3) -> Dict:
    """
    Study how the gap depends on lattice size.
    
    Key question: Does gap/g² = 1.5 hold for multi-plaquette systems?
    Or does the coefficient change with L?
    """
    results = []
    
    for L in range(1, max_L + 1):
        print(f"\n{'='*60}")
        print(f"Computing L = {L} × {L} lattice")
        print(f"{'='*60}")
        
        result = compute_multi_plaquette_gap(L, L, g, j_max)
        result['L'] = L
        result['g'] = g
        
        if result['gap'] is not None:
            result['gap_over_g2'] = result['gap'] / g**2
            print(f"  Gap = {result['gap']:.6f}")
            print(f"  Δ/g² = {result['gap_over_g2']:.6f}")
        else:
            print(f"  Gap computation failed (too few physical states)")
        
        results.append(result)
    
    return results


def scan_coupling_multi_plaquette(L: int, g_values: List[float], j_max: float = 0.5) -> List[Dict]:
    """
    Scan over coupling values for fixed lattice size.
    
    Looking for deviations from Δ ~ g² that would signal dimensional transmutation.
    """
    results = []
    
    print(f"Scanning g for {L}×{L} lattice with j_max={j_max}")
    print("-" * 50)
    
    for g in g_values:
        result = compute_multi_plaquette_gap(L, L, g, j_max)
        result['g'] = g
        result['L'] = L
        
        if result['gap'] is not None:
            result['gap_over_g2'] = result['gap'] / g**2
            print(f"g = {g:.3f}: Δ = {result['gap']:.6f}, Δ/g² = {result['gap_over_g2']:.4f}")
        
        results.append(result)
    
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("MULTI-PLAQUETTE YANG-MILLS LATTICE")
    print("Testing if gap scaling changes with spatial extent")
    print("=" * 70)
    
    # Test 1: Single plaquette (should recover Δ = 1.5 g²)
    print("\nTest 1: Single plaquette (1×1)")
    result = compute_multi_plaquette_gap(1, 1, g=1.0, j_max=0.5)
    if result['gap']:
        print(f"  Δ = {result['gap']:.6f}")
        print(f"  Δ/g² = {result['gap']/1.0:.6f}")
        print(f"  Expected: 1.5")
    
    # Test 2: 2×2 lattice
    print("\nTest 2: 2×2 lattice")
    result = compute_multi_plaquette_gap(2, 2, g=1.0, j_max=0.5)
    if result['gap']:
        print(f"  Δ = {result['gap']:.6f}")
        print(f"  Δ/g² = {result['gap']/1.0:.6f}")
    
    # Test 3: Finite-size scaling
    print("\nTest 3: Finite-size scaling at g=1.0")
    scaling_results = finite_size_scaling(g=1.0, j_max=0.5, max_L=2)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in scaling_results:
        if r['gap'] is not None:
            print(f"L = {r['L']}: Δ/g² = {r['gap_over_g2']:.4f}, n_phys = {r['n_physical']}")
