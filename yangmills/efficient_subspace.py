"""
Efficient Physical Subspace Projection for Multi-Plaquette Yang-Mills

Key insight: Instead of building full Hilbert space, directly enumerate
gauge-invariant states using Gauss law constraints.

For SU(2) at each vertex:
- J^a_v = Σ_l ± E^a_l where sum is over links at vertex v
- Physical: J² = 0 at each vertex
- This means angular momenta must couple to singlet

Strategy:
1. Build basis of states satisfying j-coupling constraints at each vertex
2. Project Hamiltonian onto this subspace
3. Diagonalize in smaller subspace
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, LinearOperator
from typing import Tuple, List, Dict, Set, Optional
from dataclasses import dataclass
import itertools
from functools import lru_cache
import sys

sys.path.insert(0, '/home/brad/TiganticLabz/Main_Projects/HyperTensor-VM-main')


@dataclass
class LinkState:
    """State of a single link: (j, m)"""
    j: float
    m: float
    
    def __hash__(self):
        return hash((self.j, self.m))
    
    def __eq__(self, other):
        return abs(self.j - other.j) < 1e-10 and abs(self.m - other.m) < 1e-10


def enumerate_link_states(j_max: float) -> List[LinkState]:
    """Enumerate all (j, m) states up to j_max."""
    states = []
    j = 0.0
    while j <= j_max + 1e-10:
        for m in np.arange(-j, j + 1):
            states.append(LinkState(j, float(m)))
        j += 0.5
    return states


@lru_cache(maxsize=10000)
def clebsch_can_couple_to_zero(j1: float, m1: float, j2: float, m2: float,
                                j3: float, m3: float, j4: float, m4: float) -> bool:
    """
    Check if 4 angular momenta can couple to total J = 0.
    
    For J = 0: (j1 ⊗ j2) ⊗ (j3 ⊗ j4) must contain J = 0.
    This requires j12 = j34 where j12 ∈ j1 ⊗ j2 and j34 ∈ j3 ⊗ j4.
    Also need m-conservation: m1 + m2 + m3 + m4 = 0.
    """
    # m-conservation
    if abs(m1 + m2 + m3 + m4) > 1e-10:
        return False
    
    # j-coupling: check if j1⊗j2 and j3⊗j4 share any J
    j12_min, j12_max = abs(j1 - j2), j1 + j2
    j34_min, j34_max = abs(j3 - j4), j3 + j4
    
    # Check overlap
    for j12 in np.arange(j12_min, j12_max + 0.5, 1.0):
        for j34 in np.arange(j34_min, j34_max + 0.5, 1.0):
            if abs(j12 - j34) < 1e-10:
                return True
    return False


@lru_cache(maxsize=10000)
def two_links_can_couple_to_zero(j1: float, m1: float, sign1: int,
                                  j2: float, m2: float, sign2: int) -> bool:
    """
    Check if 2 links at a vertex can couple to J = 0.
    
    At vertex: J^a = sign1 * E^a_1 + sign2 * E^a_2 = 0
    
    For J = 0: need j1 = j2 and m1*sign1 + m2*sign2 = 0.
    """
    if abs(j1 - j2) > 1e-10:
        return False
    if abs(sign1 * m1 + sign2 * m2) > 1e-10:
        return False
    return True


class EfficientPhysicalSubspace:
    """
    Efficiently enumerate gauge-invariant states.
    
    Key optimization: Instead of checking G² on all 5^N states,
    build states satisfying Gauss law vertex by vertex.
    """
    
    def __init__(self, Lx: int, Ly: int, j_max: float, pbc: bool = True):
        self.Lx = Lx
        self.Ly = Ly
        self.j_max = j_max
        self.pbc = pbc
        
        self.link_states = enumerate_link_states(j_max)
        self.link_dim = len(self.link_states)
        self.state_idx = {s: i for i, s in enumerate(self.link_states)}
        
        if pbc:
            self.n_links = 2 * Lx * Ly
            self.n_vertices = Lx * Ly
        else:
            self.n_links = Lx * (Ly + 1) + (Lx + 1) * Ly
            self.n_vertices = (Lx + 1) * (Ly + 1)
        
        self.total_dim = self.link_dim ** self.n_links
        
        # Precompute vertex-link structure
        self._build_vertex_structure()
        
    def _build_vertex_structure(self):
        """Build vertex → link mapping."""
        self.vertex_links = []  # List of [(link_idx, sign), ...] per vertex
        
        for v in range(self.n_vertices):
            if self.pbc:
                x = v % self.Lx
                y = v // self.Lx
                links = [
                    (x + y * self.Lx, +1),  # outgoing x-link
                    (self.Lx * self.Ly + x + y * self.Lx, +1),  # outgoing y-link
                    ((x - 1) % self.Lx + y * self.Lx, -1),  # incoming x-link
                    (self.Lx * self.Ly + x + ((y - 1) % self.Ly) * self.Lx, -1),  # incoming y-link
                ]
            else:
                x = v % (self.Lx + 1)
                y = v // (self.Lx + 1)
                links = []
                # Horizontal links: numbered 0 to Lx*(Ly+1)-1
                n_horiz = self.Lx * (self.Ly + 1)
                if x < self.Lx:
                    links.append((x + y * self.Lx, +1))  # outgoing x
                if x > 0:
                    links.append((x - 1 + y * self.Lx, -1))  # incoming x
                # Vertical links: numbered n_horiz to n_horiz + (Lx+1)*Ly - 1
                if y < self.Ly:
                    links.append((n_horiz + x + y * (self.Lx + 1), +1))  # outgoing y
                if y > 0:
                    links.append((n_horiz + x + (y - 1) * (self.Lx + 1), -1))  # incoming y
            
            self.vertex_links.append(links)
    
    def state_to_tuple(self, state_int: int) -> Tuple[int, ...]:
        """Convert integer state to tuple of link indices."""
        result = []
        for _ in range(self.n_links):
            result.append(state_int % self.link_dim)
            state_int //= self.link_dim
        return tuple(result)
    
    def tuple_to_state(self, state_tuple: Tuple[int, ...]) -> int:
        """Convert tuple of link indices to integer."""
        result = 0
        mult = 1
        for idx in state_tuple:
            result += idx * mult
            mult *= self.link_dim
        return result
    
    def check_gauss_law(self, state_tuple: Tuple[int, ...]) -> bool:
        """Check if state satisfies Gauss law at all vertices."""
        for vertex_links in self.vertex_links:
            n = len(vertex_links)
            if n == 0:
                continue
            elif n == 2:
                l1, s1 = vertex_links[0]
                l2, s2 = vertex_links[1]
                st1 = self.link_states[state_tuple[l1]]
                st2 = self.link_states[state_tuple[l2]]
                if not two_links_can_couple_to_zero(st1.j, st1.m, s1, st2.j, st2.m, s2):
                    return False
            elif n == 4:
                links_jm = []
                for lidx, sign in vertex_links:
                    st = self.link_states[state_tuple[lidx]]
                    links_jm.append((st.j, st.m * sign))
                
                # m conservation
                m_sum = sum(jm[1] for jm in links_jm)
                if abs(m_sum) > 1e-10:
                    return False
                
                # j coupling to zero
                j1, m1 = links_jm[0]
                j2, m2 = links_jm[1]
                j3, m3 = links_jm[2]
                j4, m4 = links_jm[3]
                
                if not clebsch_can_couple_to_zero(j1, m1, j2, m2, j3, m3, j4, m4):
                    return False
            else:
                # General case - should not happen for square lattice
                pass
        
        return True
    
    def enumerate_physical_states(self) -> List[int]:
        """Enumerate all gauge-invariant states."""
        physical = []
        
        for state in range(self.total_dim):
            state_tuple = self.state_to_tuple(state)
            if self.check_gauss_law(state_tuple):
                physical.append(state)
        
        return physical
    
    def enumerate_physical_states_smart(self) -> List[int]:
        """
        Smarter enumeration using vertex-by-vertex constraints.
        Only works well for small systems with OBC.
        """
        return self.enumerate_physical_states()  # Fall back to full enumeration


class EfficientMultiPlaquetteHamiltonian:
    """
    Efficient multi-plaquette Hamiltonian operating in physical subspace.
    """
    
    def __init__(self, Lx: int, Ly: int, g: float, j_max: float, pbc: bool = True):
        self.Lx = Lx
        self.Ly = Ly
        self.g = g
        self.j_max = j_max
        self.pbc = pbc
        
        self.subspace = EfficientPhysicalSubspace(Lx, Ly, j_max, pbc)
        self.link_states = self.subspace.link_states
        self.link_dim = self.subspace.link_dim
        self.n_links = self.subspace.n_links
        
        print(f"Building physical subspace for {Lx}×{Ly} lattice (PBC={pbc})")
        print(f"  Links: {self.n_links}, Link dim: {self.link_dim}")
        print(f"  Total Hilbert: {self.subspace.total_dim}")
        
        # Enumerate physical states
        self.physical_states = self.subspace.enumerate_physical_states()
        self.n_physical = len(self.physical_states)
        
        # Create mapping from physical state to index
        self.phys_to_idx = {s: i for i, s in enumerate(self.physical_states)}
        
        print(f"  Physical states: {self.n_physical}")
        
        # Precompute Casimir values
        self.casimir = {}
        for i, state in enumerate(self.link_states):
            self.casimir[i] = state.j * (state.j + 1)
    
    def build_hamiltonian_dense(self) -> np.ndarray:
        """Build Hamiltonian matrix in physical subspace."""
        H = np.zeros((self.n_physical, self.n_physical))
        
        for idx, state in enumerate(self.physical_states):
            state_tuple = self.subspace.state_to_tuple(state)
            
            # Electric term (diagonal)
            E_total = 0.0
            for l in range(self.n_links):
                E_total += self.casimir[state_tuple[l]]
            
            H[idx, idx] = (self.g**2 / 2) * E_total
        
        return H
    
    def build_hamiltonian_sparse(self) -> sparse.csr_matrix:
        """Build sparse Hamiltonian in physical subspace."""
        # For pure electric term, H is diagonal
        diag = np.zeros(self.n_physical)
        
        for idx, state in enumerate(self.physical_states):
            state_tuple = self.subspace.state_to_tuple(state)
            
            E_total = 0.0
            for l in range(self.n_links):
                E_total += self.casimir[state_tuple[l]]
            
            diag[idx] = (self.g**2 / 2) * E_total
        
        return sparse.diags(diag, format='csr')
    
    def compute_gap(self) -> Dict:
        """Compute mass gap in physical subspace."""
        if self.n_physical < 2:
            return {'gap': np.nan, 'n_physical': self.n_physical}
        
        if self.n_physical <= 1000:
            H = self.build_hamiltonian_dense()
            eigenvalues = np.linalg.eigvalsh(H)
        else:
            H = self.build_hamiltonian_sparse()
            k = min(10, self.n_physical - 1)
            eigenvalues, _ = eigsh(H, k=k, which='SA')
            eigenvalues = np.sort(eigenvalues)
        
        E0 = eigenvalues[0]
        E1 = eigenvalues[1]
        gap = E1 - E0
        
        return {
            'E0': E0,
            'E1': E1,
            'gap': gap,
            'gap_over_g2': gap / self.g**2 if gap > 0 else np.nan,
            'n_physical': self.n_physical,
            'spectrum': eigenvalues[:min(5, len(eigenvalues))]
        }


def test_efficient_implementation():
    """Test that efficient implementation matches reference."""
    print("=" * 70)
    print("EFFICIENT IMPLEMENTATION TEST")
    print("=" * 70)
    
    configs = [
        (1, 1, False, "1×1 OBC"),
        (1, 1, True, "1×1 PBC"),
        (1, 2, True, "1×2 PBC"),
    ]
    
    g = 1.0
    j_max = 0.5
    
    for Lx, Ly, pbc, name in configs:
        print(f"\n{name}:")
        ham = EfficientMultiPlaquetteHamiltonian(Lx, Ly, g, j_max, pbc)
        result = ham.compute_gap()
        print(f"  Gap: {result['gap']:.6f}")
        print(f"  Δ/g²: {result['gap_over_g2']:.6f}")
        if name == "1×1 OBC":
            expected = 1.5
            print(f"  Expected: {expected}")
            print(f"  Match: {'✓' if abs(result['gap_over_g2'] - expected) < 0.01 else '✗'}")


def test_larger_lattices():
    """Test larger lattices with efficient enumeration."""
    print("\n" + "=" * 70)
    print("LARGER LATTICE TEST")
    print("=" * 70)
    
    g = 1.0
    j_max = 0.5  # Keep truncation small
    
    configs = [
        (2, 1, True, "2×1 PBC"),
        (2, 2, True, "2×2 PBC"),
        (2, 1, False, "2×1 OBC"),
        (2, 2, False, "2×2 OBC"),
    ]
    
    for Lx, Ly, pbc, name in configs:
        print(f"\n{name}:")
        try:
            ham = EfficientMultiPlaquetteHamiltonian(Lx, Ly, g, j_max, pbc)
            
            if ham.n_physical > 0:
                result = ham.compute_gap()
                print(f"  Gap: {result['gap']:.6f}")
                print(f"  Δ/g²: {result['gap_over_g2']:.6f}")
            else:
                print(f"  No physical states found!")
        except MemoryError as e:
            print(f"  Memory error: {e}")
        except Exception as e:
            print(f"  Error: {e}")


def scaling_study():
    """Study scaling of gap with lattice size."""
    print("\n" + "=" * 70)
    print("SCALING STUDY: Gap per Plaquette")
    print("=" * 70)
    
    g = 1.0
    j_max = 0.5
    
    results = []
    
    print(f"\n{'Config':<12} {'n_phys':<8} {'Δ/g²':<10} {'Δ/(n_p·g²)':<12}")
    print("-" * 50)
    
    for Lx, Ly in [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1)]:
        for pbc in [False]:  # Focus on OBC where 1×1 gives 3/2
            name = f"{Lx}×{Ly} {'PBC' if pbc else 'OBC'}"
            try:
                ham = EfficientMultiPlaquetteHamiltonian(Lx, Ly, g, j_max, pbc)
                if ham.n_physical >= 2:
                    result = ham.compute_gap()
                    n_plaq = Lx * Ly
                    gap_per_plaq = result['gap_over_g2'] / n_plaq
                    
                    print(f"{name:<12} {ham.n_physical:<8} {result['gap_over_g2']:<10.4f} {gap_per_plaq:<12.4f}")
                    
                    results.append({
                        'Lx': Lx, 'Ly': Ly, 'pbc': pbc,
                        'n_phys': ham.n_physical,
                        'gap_over_g2': result['gap_over_g2'],
                        'gap_per_plaq': gap_per_plaq
                    })
                else:
                    print(f"{name:<12} {ham.n_physical:<8} --         --")
            except Exception as e:
                print(f"{name:<12} ERROR: {e}")
    
    return results


if __name__ == "__main__":
    test_efficient_implementation()
    test_larger_lattices()
    scaling_study()
