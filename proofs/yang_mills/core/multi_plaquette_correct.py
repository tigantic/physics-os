"""
Multi-Plaquette Yang-Mills - Correct Implementation

Key insight from proven single plaquette:
- 4 links form a square
- 4 vertices at corners  
- Gauss law at each vertex: Σ ±E^a = 0
- Physical states: G² = 0 everywhere
- Strong coupling gap: Δ = (3/2)g²

The 3/2 comes from:
- Ground state: all links in j=0
- First excited: one link at j=1/2 → E² = (1/2)(3/2) = 3/4
- With 4 links constrained by Gauss: excited state has j=1/2 on all 4 links
- Total E² = 4 × (3/4) = 3
- Gap = (g²/2) × 3 = (3/2)g²

For multi-plaquette: extend this correctly with proper Gauss constraints.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import sys

sys.path.insert(0, '/home/brad/TiganticLabz/Main_Projects/physics-os')

from yangmills.operators import TruncatedHilbertSpace, ElectricFieldOperator
from yangmills.gauss import SinglePlaquetteGauss


def verify_single_plaquette_gap():
    """
    Verify we can reproduce the proven Δ = (3/2)g² result.
    This is our reference implementation.
    """
    from yangmills.hamiltonian import SinglePlaquetteHamiltonian
    
    print("=" * 70)
    print("VERIFICATION: Single Plaquette Gap Δ = (3/2)g²")
    print("=" * 70)
    
    g = 1.0
    j_max = 0.5
    
    H_sys = SinglePlaquetteHamiltonian(j_max=j_max, g=g)
    H = H_sys.build_hamiltonian()
    
    gauss = SinglePlaquetteGauss(H_sys.hilbert)
    G2 = gauss.total_gauss_squared()
    
    # Diagonalize
    H_dense = H.toarray() if sparse.issparse(H) else H
    eigenvalues, eigenvectors = np.linalg.eigh(H_dense)
    
    # Find physical states
    physical_E = []
    physical_indices = []
    for i in range(len(eigenvalues)):
        psi = eigenvectors[:, i]
        g2_val = np.abs(psi.conj() @ G2 @ psi)
        if g2_val < 1e-6:
            physical_E.append(eigenvalues[i])
            physical_indices.append(i)
    
    gap = physical_E[1] - physical_E[0]
    gap_over_g2 = gap / g**2
    
    print(f"Coupling g = {g}")
    print(f"j_max = {j_max}")
    print(f"Hilbert dim = {H_sys.total_dim} (per link: {H_sys.link_dim})")
    print(f"Physical states: {len(physical_E)}")
    print(f"E₀ = {physical_E[0]:.6f}")
    print(f"E₁ = {physical_E[1]:.6f}")
    print(f"Gap Δ = {gap:.6f}")
    print(f"Δ/g² = {gap_over_g2:.6f}")
    print(f"Expected: 1.5")
    print(f"Match: {'✓' if abs(gap_over_g2 - 1.5) < 0.01 else '✗'}")
    
    return gap_over_g2


class MultiPlaquetteLattice:
    """
    Multi-plaquette lattice with correct Gauss law structure.
    
    For Lx × Ly plaquettes with OPEN boundaries:
    - Vertices: (Lx+1) × (Ly+1)  
    - Horizontal links: (Ly+1) × Lx
    - Vertical links: Ly × (Lx+1)
    - Plaquettes: Lx × Ly
    
    With PBC:
    - Vertices: Lx × Ly
    - Links: 2 × Lx × Ly
    - Plaquettes: Lx × Ly
    """
    
    def __init__(self, Lx: int, Ly: int, pbc: bool = True):
        self.Lx = Lx
        self.Ly = Ly
        self.pbc = pbc
        
        if pbc:
            self.n_vertices = Lx * Ly
            self.n_links = 2 * Lx * Ly  # x-links + y-links
            self.n_plaquettes = Lx * Ly
        else:
            # Open boundaries like single plaquette
            self.n_vertices = (Lx + 1) * (Ly + 1)
            self.n_links = Lx * (Ly + 1) + (Lx + 1) * Ly  # horizontal + vertical
            self.n_plaquettes = Lx * Ly
    
    def vertex_index(self, x: int, y: int) -> int:
        """Convert (x, y) to vertex index."""
        if self.pbc:
            return (x % self.Lx) + (y % self.Ly) * self.Lx
        else:
            return x + y * (self.Lx + 1)
    
    def link_index_h(self, x: int, y: int) -> int:
        """Index of horizontal link from (x,y) to (x+1,y)."""
        if self.pbc:
            return (x % self.Lx) + (y % self.Ly) * self.Lx
        else:
            # Horizontal links: indexed first
            return x + y * self.Lx
    
    def link_index_v(self, x: int, y: int) -> int:
        """Index of vertical link from (x,y) to (x,y+1)."""
        if self.pbc:
            return self.Lx * self.Ly + (x % self.Lx) + (y % self.Ly) * self.Lx
        else:
            # Vertical links: after horizontal
            n_horiz = self.Lx * (self.Ly + 1)
            return n_horiz + x + y * (self.Lx + 1)
    
    def get_vertex_links(self, x: int, y: int) -> List[Tuple[int, int]]:
        """
        Get links touching vertex (x, y) with orientations.
        Returns [(link_index, sign), ...] where sign = +1 for outgoing, -1 for incoming.
        """
        links = []
        
        if self.pbc:
            # Outgoing: x-link at (x,y), y-link at (x,y)
            links.append((self.link_index_h(x, y), +1))
            links.append((self.link_index_v(x, y), +1))
            # Incoming: x-link at (x-1,y), y-link at (x,y-1)
            links.append((self.link_index_h(x-1, y), -1))
            links.append((self.link_index_v(x, y-1), -1))
        else:
            # Open boundaries - check existence
            if x < self.Lx:  # Outgoing horizontal
                links.append((self.link_index_h(x, y), +1))
            if y < self.Ly:  # Outgoing vertical
                links.append((self.link_index_v(x, y), +1))
            if x > 0:  # Incoming horizontal
                links.append((self.link_index_h(x-1, y), -1))
            if y > 0:  # Incoming vertical
                links.append((self.link_index_v(x, y-1), -1))
        
        return links
    
    def get_plaquette_links(self, px: int, py: int) -> List[Tuple[int, int]]:
        """
        Get links forming plaquette at (px, py).
        Returns [(link_index, orientation), ...] going counterclockwise.
        """
        # Bottom → Right → Top → Left (CCW)
        return [
            (self.link_index_h(px, py), +1),      # bottom (px,py) → (px+1,py)
            (self.link_index_v(px+1, py), +1),    # right (px+1,py) → (px+1,py+1)
            (self.link_index_h(px, py+1), -1),    # top (px+1,py+1) ← (px,py+1)
            (self.link_index_v(px, py), -1),      # left (px,py+1) ← (px,py)
        ]


class MultiPlaquetteHamiltonian:
    """
    Multi-plaquette Hamiltonian with correct Gauss law implementation.
    
    H = (g²/2) Σ_l E²_l + (1/g²) Σ_p (1 - Re Tr U_p / 2)
    
    Physical states: G^a_v |ψ⟩ = 0 for all vertices v, colors a.
    """
    
    def __init__(self, Lx: int, Ly: int, g: float, j_max: float, pbc: bool = True):
        self.lattice = MultiPlaquetteLattice(Lx, Ly, pbc)
        self.g = g
        self.j_max = j_max
        
        self.hilbert = TruncatedHilbertSpace(j_max=j_max)
        self.link_dim = self.hilbert.total_dim
        self.n_links = self.lattice.n_links
        self.total_dim = self.link_dim ** self.n_links
        
        self.E_op = ElectricFieldOperator(self.hilbert)
        
        print(f"Multi-Plaquette: {Lx}×{Ly} lattice, PBC={pbc}")
        print(f"  Links: {self.n_links}, Vertices: {self.lattice.n_vertices}")
        print(f"  Plaquettes: {self.lattice.n_plaquettes}")
        print(f"  Link dim: {self.link_dim}, Total dim: {self.total_dim}")
    
    def _extend_operator(self, op, link_idx: int):
        """Extend single-link operator to full Hilbert space."""
        I = sparse.eye(self.link_dim, format='csr')
        
        result = op if link_idx == 0 else I
        for i in range(1, self.n_links):
            if i == link_idx:
                result = sparse.kron(result, op, format='csr')
            else:
                result = sparse.kron(result, I, format='csr')
        return result
    
    def build_electric_term(self) -> sparse.csr_matrix:
        """H_E = (g²/2) Σ_l E²_l"""
        coeff = self.g**2 / 2
        E2 = self.E_op.E_squared
        
        H_E = sparse.csr_matrix((self.total_dim, self.total_dim))
        for l in range(self.n_links):
            H_E = H_E + self._extend_operator(E2, l)
        
        return coeff * H_E
    
    def build_gauss_squared(self, vertex_idx: int) -> sparse.csr_matrix:
        """Build G² at a specific vertex."""
        if self.lattice.pbc:
            x = vertex_idx % self.lattice.Lx
            y = vertex_idx // self.lattice.Lx
        else:
            x = vertex_idx % (self.lattice.Lx + 1)
            y = vertex_idx // (self.lattice.Lx + 1)
        
        vertex_links = self.lattice.get_vertex_links(x, y)
        
        G2 = sparse.csr_matrix((self.total_dim, self.total_dim))
        
        for a in range(3):  # SU(2) has 3 generators
            E_a = self.E_op.E_a(a)
            
            G_a = sparse.csr_matrix((self.total_dim, self.total_dim))
            for link_idx, sign in vertex_links:
                G_a = G_a + sign * self._extend_operator(E_a, link_idx)
            
            G2 = G2 + G_a @ G_a
        
        return G2
    
    def total_gauss_squared(self) -> sparse.csr_matrix:
        """Total Gauss constraint: Σ_v G²_v"""
        G2_total = sparse.csr_matrix((self.total_dim, self.total_dim))
        
        for v in range(self.lattice.n_vertices):
            G2_total = G2_total + self.build_gauss_squared(v)
        
        return G2_total
    
    def build_hamiltonian(self) -> sparse.csr_matrix:
        """Build full Hamiltonian (electric term only for strong coupling)."""
        return self.build_electric_term()
    
    def compute_physical_gap(self) -> Dict:
        """Compute gap in gauge-invariant sector."""
        H = self.build_hamiltonian()
        G2 = self.total_gauss_squared()
        
        # Convert to dense for small systems
        if self.total_dim <= 1000:
            H_dense = H.toarray()
            G2_dense = G2.toarray()
            
            eigenvalues, eigenvectors = np.linalg.eigh(H_dense)
            
            # Find physical states (G² ≈ 0)
            physical_E = []
            for i in range(len(eigenvalues)):
                psi = eigenvectors[:, i]
                g2_val = np.abs(psi.conj() @ G2_dense @ psi)
                if g2_val < 1e-6:
                    physical_E.append(eigenvalues[i])
            
            if len(physical_E) >= 2:
                gap = physical_E[1] - physical_E[0]
                E0, E1 = physical_E[0], physical_E[1]
            else:
                gap, E0, E1 = np.nan, np.nan, np.nan
                
        else:
            # Sparse for larger systems
            # First find lowest eigenvalues
            n_eig = min(20, self.total_dim - 2)
            eigenvalues, eigenvectors = eigsh(H, k=n_eig, which='SA')
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            G2_dense = G2.toarray()
            
            physical_E = []
            for i in range(len(eigenvalues)):
                psi = eigenvectors[:, i]
                g2_val = np.abs(psi.conj() @ G2_dense @ psi)
                if g2_val < 1e-6:
                    physical_E.append(eigenvalues[i])
            
            if len(physical_E) >= 2:
                gap = physical_E[1] - physical_E[0]
                E0, E1 = physical_E[0], physical_E[1]
            else:
                gap, E0, E1 = np.nan, np.nan, np.nan
        
        return {
            'gap': gap,
            'E0': E0,
            'E1': E1,
            'gap_over_g2': gap / self.g**2 if gap > 0 else np.nan,
            'n_physical': len(physical_E),
            'total_dim': self.total_dim
        }


def test_single_plaquette_match():
    """
    Test that 1×1 open boundary matches proven single plaquette.
    
    Single plaquette = 1×1 with open boundaries:
    - 4 vertices (corners)
    - 4 links (edges)
    - 1 plaquette
    """
    print("\n" + "=" * 70)
    print("TEST: 1×1 Open Boundary = Single Plaquette")
    print("=" * 70)
    
    g = 1.0
    j_max = 0.5
    
    # Multi-plaquette with open boundaries
    ham = MultiPlaquetteHamiltonian(Lx=1, Ly=1, g=g, j_max=j_max, pbc=False)
    result = ham.compute_physical_gap()
    
    print(f"\n1×1 Open BC Multi-Plaquette:")
    print(f"  n_physical = {result['n_physical']}")
    print(f"  E₀ = {result['E0']:.6f}")
    print(f"  E₁ = {result['E1']:.6f}")
    print(f"  Δ = {result['gap']:.6f}")
    print(f"  Δ/g² = {result['gap_over_g2']:.6f}")
    print(f"  Expected: 1.5")
    print(f"  Match: {'✓' if abs(result['gap_over_g2'] - 1.5) < 0.01 else '✗'}")
    
    return result


def scaling_analysis():
    """Study how gap scales with lattice size."""
    print("\n" + "=" * 70)
    print("SCALING ANALYSIS: Gap vs Lattice Size")
    print("=" * 70)
    
    g = 1.0
    j_max = 0.5
    
    results = []
    
    # Test different sizes
    configs = [
        (1, 1, False, "1×1 OBC"),
        (1, 1, True, "1×1 PBC"),
        (1, 2, True, "1×2 PBC"),
        (2, 2, True, "2×2 PBC"),
    ]
    
    print(f"\n{'Config':<12} {'n_phys':<8} {'Δ':<10} {'Δ/g²':<10} {'Δ/(L²g²)':<10}")
    print("-" * 60)
    
    for Lx, Ly, pbc, name in configs:
        try:
            ham = MultiPlaquetteHamiltonian(Lx, Ly, g, j_max, pbc)
            result = ham.compute_physical_gap()
            
            n_plaq = Lx * Ly
            gap_per_plaq = result['gap_over_g2'] / n_plaq if n_plaq > 0 else np.nan
            
            print(f"{name:<12} {result['n_physical']:<8} {result['gap']:<10.4f} "
                  f"{result['gap_over_g2']:<10.4f} {gap_per_plaq:<10.4f}")
            
            results.append({
                'name': name, 'Lx': Lx, 'Ly': Ly, 'pbc': pbc,
                **result, 'gap_per_plaq': gap_per_plaq
            })
        except Exception as e:
            print(f"{name:<12} ERROR: {e}")
    
    return results


def coupling_scan():
    """Scan coupling strength for fixed lattice."""
    print("\n" + "=" * 70)
    print("COUPLING SCAN: 1×1 Open BC")
    print("=" * 70)
    
    j_max = 0.5
    g_values = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    
    print(f"\n{'g':<8} {'Δ':<12} {'Δ/g²':<12}")
    print("-" * 35)
    
    for g in g_values:
        ham = MultiPlaquetteHamiltonian(1, 1, g, j_max, pbc=False)
        result = ham.compute_physical_gap()
        print(f"{g:<8.2f} {result['gap']:<12.6f} {result['gap_over_g2']:<12.6f}")


if __name__ == "__main__":
    # First verify we match the proven result
    print("\n" + "=" * 70)
    print("REFERENCE: Proven Single Plaquette Implementation")
    print("=" * 70)
    ref_gap = verify_single_plaquette_gap()
    
    # Test our multi-plaquette matches
    test_single_plaquette_match()
    
    # Scaling analysis
    scaling_analysis()
    
    # Coupling scan
    coupling_scan()
