"""
Multi-Plaquette Yang-Mills - Corrected Implementation

Key insight: The single plaquette with 4 links gives Δ = (3/2)g².
For PBC in 2D, the 1×1 lattice has only 2 independent links!
This changes the calculation significantly.

Let me implement a proper treatment with correct link counting.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from typing import Tuple, List, Dict
from dataclasses import dataclass
import itertools


@dataclass 
class CorrectLatticeGeometry:
    """
    2D square lattice with correct PBC treatment.
    
    For an Lx × Ly lattice with periodic BC:
    - Sites: Lx × Ly
    - Links: 2 × Lx × Ly (each site has one x-link and one y-link)
    - Plaquettes: Lx × Ly
    - Independent Gauss constraints: Lx × Ly - 1 (one is redundant)
    """
    Lx: int
    Ly: int
    
    @property
    def n_sites(self) -> int:
        return self.Lx * self.Ly
    
    @property  
    def n_links(self) -> int:
        return 2 * self.Lx * self.Ly
    
    @property
    def n_plaquettes(self) -> int:
        return self.Lx * self.Ly
    
    def site(self, x: int, y: int) -> int:
        """Site index with PBC."""
        return (x % self.Lx) + (y % self.Ly) * self.Lx
    
    def link_x(self, x: int, y: int) -> int:
        """x-direction link starting at (x,y)."""
        return self.site(x, y)
    
    def link_y(self, x: int, y: int) -> int:
        """y-direction link starting at (x,y)."""
        return self.n_sites + self.site(x, y)
    
    def plaquette(self, x: int, y: int) -> Tuple[Tuple[int, int], ...]:
        """
        Links of plaquette at (x,y): (link_idx, orientation).
        Going counterclockwise: bottom, right, top, left.
        Orientation: +1 = forward, -1 = backward.
        """
        return (
            (self.link_x(x, y), +1),           # bottom: (x,y) → (x+1,y)
            (self.link_y(x+1, y), +1),         # right: (x+1,y) → (x+1,y+1)
            (self.link_x(x, y+1), -1),         # top: (x+1,y+1) ← (x,y+1) 
            (self.link_y(x, y), -1)            # left: (x,y+1) ← (x,y)
        )


class SU2LinkBasis:
    """
    SU(2) link variable basis using angular momentum representation.
    
    Each link carries representation label j and magnetic quantum number m.
    Electric field: E² has eigenvalue j(j+1).
    """
    
    def __init__(self, j_max: float):
        self.j_max = j_max
        
        # Build basis: |j, m⟩ for j = 0, 1/2, 1, ... up to j_max
        self.states = []
        j = 0.0
        while j <= j_max + 1e-10:
            for m in np.arange(-j, j + 1):
                self.states.append((j, m))
            j += 0.5
        
        self.dim = len(self.states)
        self.idx = {s: i for i, s in enumerate(self.states)}
        
        # Precompute Casimir values
        self.casimir = np.array([j * (j + 1) for j, m in self.states])
        
    def E_squared_matrix(self) -> np.ndarray:
        """E² operator - diagonal in this basis."""
        return np.diag(self.casimir)


class MultiPlaquetteHamiltonianV2:
    """
    Corrected multi-plaquette Hamiltonian.
    
    H = (g²/2) Σ_l E²_l + (1/g²) Σ_p (1 - ½ Re Tr U_p)
    
    In strong coupling (large g), the plaquette term is a perturbation.
    The gap comes from the electric term: Δ ~ g² × (minimal Casimir change).
    """
    
    def __init__(self, Lx: int, Ly: int, g: float, j_max: float = 0.5):
        self.geom = CorrectLatticeGeometry(Lx, Ly)
        self.g = g
        self.j_max = j_max
        self.basis = SU2LinkBasis(j_max)
        
        self.n_links = self.geom.n_links
        self.link_dim = self.basis.dim
        
        print(f"Lattice: {Lx}×{Ly}, Links: {self.n_links}, States/link: {self.link_dim}")
        
    def _multi_index_to_int(self, indices: Tuple[int, ...]) -> int:
        """Convert tuple of per-link indices to single integer."""
        result = 0
        multiplier = 1
        for idx in indices:
            result += idx * multiplier
            multiplier *= self.link_dim
        return result
    
    def _int_to_multi_index(self, state: int) -> Tuple[int, ...]:
        """Convert single integer to tuple of per-link indices."""
        indices = []
        for _ in range(self.n_links):
            indices.append(state % self.link_dim)
            state //= self.link_dim
        return tuple(indices)
    
    def total_dim(self) -> int:
        return self.link_dim ** self.n_links
    
    def is_gauge_invariant(self, state: int) -> bool:
        """
        Check Gauss law at all vertices.
        
        At each vertex, the incoming and outgoing E fields must couple to zero.
        For SU(2): the four j values at a vertex must satisfy triangle inequalities
        allowing total J = 0, and the m values must sum to zero.
        """
        indices = self._int_to_multi_index(state)
        
        for x in range(self.geom.Lx):
            for y in range(self.geom.Ly):
                # Links at vertex (x, y):
                # Outgoing: x-link at (x,y), y-link at (x,y)  
                # Incoming: x-link at (x-1,y), y-link at (x,y-1)
                
                links = [
                    (self.geom.link_x(x, y), +1),       # outgoing x
                    (self.geom.link_y(x, y), +1),       # outgoing y
                    (self.geom.link_x(x-1, y), -1),     # incoming x
                    (self.geom.link_y(x, y-1), -1),     # incoming y
                ]
                
                # Get (j, m) for each link
                jm_values = []
                for link_idx, sign in links:
                    j, m = self.basis.states[indices[link_idx]]
                    jm_values.append((j, m, sign))
                
                # Check m conservation: Σ sign_i × m_i = 0
                m_sum = sum(sign * m for j, m, sign in jm_values)
                if abs(m_sum) > 1e-10:
                    return False
                
                # Check angular momentum coupling to zero
                j_vals = [j for j, m, sign in jm_values]
                if not self._can_couple_four_to_zero(j_vals):
                    return False
        
        return True
    
    def _can_couple_four_to_zero(self, j_list: List[float]) -> bool:
        """Check if 4 angular momenta can couple to J = 0."""
        j1, j2, j3, j4 = j_list
        
        # j1 ⊗ j2 → j12, j12 ⊗ j3 → j123, j123 ⊗ j4 → 0
        # Need j123 = j4 for final coupling to zero
        
        for j12 in self._allowed_j(j1, j2):
            for j123 in self._allowed_j(j12, j3):
                if abs(j123 - j4) < 1e-10:
                    return True
        return False
    
    def _allowed_j(self, j1: float, j2: float) -> List[float]:
        """Return allowed total J values for j1 ⊗ j2."""
        j_min = abs(j1 - j2)
        j_max = j1 + j2
        return [j for j in np.arange(j_min, j_max + 0.5, 1.0) if j <= j_max + 1e-10]
    
    def find_physical_states(self) -> List[int]:
        """Find all gauge-invariant states."""
        total = self.total_dim()
        physical = []
        for state in range(total):
            if self.is_gauge_invariant(state):
                physical.append(state)
        return physical
    
    def build_H_physical(self) -> Tuple[np.ndarray, List[int]]:
        """Build Hamiltonian in physical subspace."""
        physical = self.find_physical_states()
        n_phys = len(physical)
        
        print(f"Physical states: {n_phys}/{self.total_dim()}")
        
        if n_phys == 0:
            return None, []
        
        H = np.zeros((n_phys, n_phys))
        
        # Electric term (diagonal)
        for i, state in enumerate(physical):
            indices = self._int_to_multi_index(state)
            E_total = 0.0
            for link_idx in range(self.n_links):
                j, m = self.basis.states[indices[link_idx]]
                E_total += j * (j + 1)  # Casimir
            H[i, i] += (self.g**2 / 2) * E_total
        
        # Plaquette term (for now, just the diagonal part from strong coupling)
        # The leading order in strong coupling is just a constant shift
        # We add the character contribution from Re(Tr U)
        for i, state in enumerate(physical):
            indices = self._int_to_multi_index(state)
            
            for px in range(self.geom.Lx):
                for py in range(self.geom.Ly):
                    plaq = self.geom.plaquette(px, py)
                    
                    # Constant offset
                    H[i, i] += 1.0 / self.g**2
                    
                    # Diagonal contribution from Re(Tr U)
                    # When all links have same j and m's match up
                    j_vals = []
                    m_vals = []
                    for link_idx, orient in plaq:
                        j, m = self.basis.states[indices[link_idx]]
                        j_vals.append(j)
                        m_vals.append(m * orient)
                    
                    if len(set(j_vals)) == 1:  # All same j
                        j = j_vals[0]
                        if abs(sum(m_vals)) < 1e-10:  # m's sum to 0
                            dim_j = 2 * j + 1
                            if dim_j > 0:
                                # Character expansion coefficient
                                H[i, i] -= 1.0 / (2 * self.g**2 * dim_j)
        
        return H, physical
    
    def compute_spectrum(self, n_states: int = 5) -> Dict:
        """Compute low-lying spectrum."""
        H, physical = self.build_H_physical()
        
        if H is None:
            return {'gap': None, 'n_physical': 0}
        
        eigenvalues = np.linalg.eigvalsh(H)
        eigenvalues = np.sort(eigenvalues)
        
        E0 = eigenvalues[0]
        E1 = eigenvalues[1] if len(eigenvalues) > 1 else E0
        gap = E1 - E0
        
        return {
            'gap': gap,
            'E0': E0,
            'E1': E1,
            'n_physical': len(physical),
            'spectrum': eigenvalues[:min(n_states, len(eigenvalues))]
        }


def analyze_gap_scaling():
    """Analyze how gap scales with coupling and lattice size."""
    
    print("=" * 70)
    print("MULTI-PLAQUETTE GAP ANALYSIS")
    print("=" * 70)
    
    results = []
    
    # Test different lattice sizes
    for Lx, Ly in [(1, 1), (1, 2), (2, 2)]:
        print(f"\n{'='*60}")
        print(f"Lattice {Lx}×{Ly}")
        print(f"{'='*60}")
        
        g_values = [0.5, 1.0, 1.5, 2.0]
        
        for g in g_values:
            try:
                ham = MultiPlaquetteHamiltonianV2(Lx, Ly, g, j_max=0.5)
                result = ham.compute_spectrum()
                
                if result['gap'] is not None:
                    gap_over_g2 = result['gap'] / g**2
                    print(f"  g={g:.1f}: Δ={result['gap']:.4f}, Δ/g²={gap_over_g2:.4f}, n_phys={result['n_physical']}")
                    results.append({
                        'Lx': Lx, 'Ly': Ly, 'g': g,
                        'gap': result['gap'],
                        'gap_over_g2': gap_over_g2,
                        'n_physical': result['n_physical']
                    })
            except Exception as e:
                print(f"  g={g:.1f}: Error - {e}")
    
    return results


def study_strong_coupling_formula():
    """
    Derive the strong coupling gap formula for multi-plaquette lattice.
    
    In strong coupling, H ≈ (g²/2) Σ E² + perturbations.
    
    Ground state: all links in j = 0 representation.
    First excited state: one link in j = 1/2.
    
    But wait - Gauss law constrains which configurations are allowed!
    For a single plaquette (1×1 with PBC), only 2 independent links.
    The physical ground state has specific structure.
    """
    
    print("\n" + "=" * 70)
    print("STRONG COUPLING ANALYSIS")
    print("=" * 70)
    
    # For 1×1 lattice with PBC: 2 links, 1 plaquette
    # Physical states must satisfy Gauss law at the single vertex
    
    print("\n1×1 LATTICE:")
    print("  - 2 links (one x, one y)")
    print("  - 1 vertex, 1 plaquette")  
    print("  - Gauss law: j_x = j_y (both links meet at same vertex twice)")
    
    # Build explicit states
    j_max = 0.5
    states_11 = []
    for jx in [0.0, 0.5]:
        for mx in np.arange(-jx, jx + 1):
            for jy in [0.0, 0.5]:
                for my in np.arange(-jy, jy + 1):
                    # Gauss law: at each vertex, all 4 links are the same 2 links!
                    # x,y outgoing = x,y incoming (due to PBC wrapping)
                    # So constraint is trivially satisfied for any j,m
                    states_11.append((jx, mx, jy, my))
    
    print(f"  Total states: {len(states_11)}")
    
    # Build Hamiltonian matrix
    g = 1.0
    H_11 = np.zeros((len(states_11), len(states_11)))
    
    for i, (jx, mx, jy, my) in enumerate(states_11):
        # Electric term
        E_sq = jx * (jx + 1) + jy * (jy + 1)
        H_11[i, i] = (g**2 / 2) * E_sq
        
        # Plaquette: goes around x-link twice, y-link twice
        # Re(Tr U) diagonal part when jx = jy and mx, my consistent
        H_11[i, i] += 1.0 / g**2  # constant
        
        if abs(jx - jy) < 1e-10:  # same j on both links
            j = jx
            dim_j = 2 * j + 1
            if dim_j > 0:
                # Check m constraint around plaquette
                # Plaquette: x(+1), y(+1), x(-1), y(-1)
                m_sum = mx + my - mx - my  # = 0 always
                H_11[i, i] -= 1.0 / (2 * g**2 * dim_j)
    
    eigenvalues_11 = np.linalg.eigvalsh(H_11)
    eigenvalues_11 = np.sort(eigenvalues_11)
    
    print(f"  Eigenvalues: {eigenvalues_11[:5]}")
    print(f"  Gap: {eigenvalues_11[1] - eigenvalues_11[0]:.6f}")
    print(f"  Δ/g²: {(eigenvalues_11[1] - eigenvalues_11[0])/g**2:.6f}")
    
    # The correct single plaquette with 4 INDEPENDENT links
    print("\n4-LINK SINGLE PLAQUETTE (correct model):")
    
    states_4link = []
    for j1 in [0.0, 0.5]:
        for m1 in np.arange(-j1, j1 + 1):
            for j2 in [0.0, 0.5]:
                for m2 in np.arange(-j2, j2 + 1):
                    for j3 in [0.0, 0.5]:
                        for m3 in np.arange(-j3, j3 + 1):
                            for j4 in [0.0, 0.5]:
                                for m4 in np.arange(-j4, j4 + 1):
                                    states_4link.append((j1, m1, j2, m2, j3, m3, j4, m4))
    
    # Filter by Gauss law at 4 vertices
    def gauss_4link(state):
        j1, m1, j2, m2, j3, m3, j4, m4 = state
        
        # Vertex 1 (bottom-left): outgoing j1, j4; incoming j2, j3
        # For j coupling to zero: need j1=j4 and j2=j3 pairs or similar
        # And m1 + m4 - m2 - m3 = 0
        
        # Actually for single plaquette with 4 vertices:
        # V1: link1(+), link4(-) → links 1 and 4 at this vertex
        # V2: link1(-), link2(+) → links 1 and 2 at this vertex
        # V3: link2(-), link3(+) → links 2 and 3 at this vertex
        # V4: link3(-), link4(+) → links 3 and 4 at this vertex
        
        # At each vertex with 2 links: j1 ⊗ j2 must contain j=0
        # This requires j1 = j2
        
        # V1: j1 = j4
        # V2: j1 = j2
        # V3: j2 = j3
        # V4: j3 = j4
        # → All j equal!
        
        if not (abs(j1 - j2) < 1e-10 and abs(j2 - j3) < 1e-10 and 
                abs(j3 - j4) < 1e-10):
            return False
        
        # m constraints at each vertex
        # V1: m1 - m4 = 0
        # V2: -m1 + m2 = 0  
        # V3: -m2 + m3 = 0
        # V4: -m3 + m4 = 0
        # → All m equal!
        
        if not (abs(m1 - m4) < 1e-10 and abs(m1 - m2) < 1e-10 and
                abs(m2 - m3) < 1e-10 and abs(m3 - m4) < 1e-10):
            return False
        
        return True
    
    physical_4link = [s for s in states_4link if gauss_4link(s)]
    print(f"  Total states: {len(states_4link)}")
    print(f"  Physical states: {len(physical_4link)}")
    
    # Build Hamiltonian
    n_phys = len(physical_4link)
    H_4link = np.zeros((n_phys, n_phys))
    
    for i, state in enumerate(physical_4link):
        j1, m1, j2, m2, j3, m3, j4, m4 = state
        
        # Electric term
        E_sq = sum(j * (j + 1) for j in [j1, j2, j3, j4])
        H_4link[i, i] = (g**2 / 2) * E_sq
        
        # Plaquette term
        H_4link[i, i] += 1.0 / g**2
        
        # All j's are equal (by Gauss law)
        j = j1
        dim_j = 2 * j + 1
        if dim_j > 0:
            # Check m around plaquette: m1 + m2 - m3 - m4
            # All m's are equal, so sum = 0
            H_4link[i, i] -= 1.0 / (2 * g**2 * dim_j)
    
    eigenvalues_4link = np.linalg.eigvalsh(H_4link)
    eigenvalues_4link = np.sort(eigenvalues_4link)
    
    print(f"  Eigenvalues: {eigenvalues_4link[:5]}")
    print(f"  Gap: {eigenvalues_4link[1] - eigenvalues_4link[0]:.6f}")
    print(f"  Δ/g²: {(eigenvalues_4link[1] - eigenvalues_4link[0])/g**2:.6f}")
    
    # Verify this matches 3/2
    expected = 1.5
    actual = (eigenvalues_4link[1] - eigenvalues_4link[0]) / g**2
    print(f"\n  Expected Δ/g² = {expected}")
    print(f"  Computed Δ/g² = {actual:.6f}")
    print(f"  Match: {'✓' if abs(actual - expected) < 0.01 else '✗'}")


if __name__ == "__main__":
    study_strong_coupling_formula()
    print("\n")
    results = analyze_gap_scaling()
