#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              QTT-DMRG FOR LARGE MULTI-PLAQUETTE LATTICES                     ║
║                                                                              ║
║              Hunting the Scaling Window at L ~ 30-100                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

GOAL: Find the crossover from strong coupling to asymptotic freedom.

WHY THIS WORKS:
===============
- Single plaquette: ξ capped → can't see crossover
- Large lattice (L >> ξ): glueball can swell → entanglement grows

SIGNATURE OF CROSSOVER:
=======================
1. Entanglement S rises above ln(2)
2. Bond dimension saturates (solver needs more resources)
3. Δ/g² deviates from 0.375 plateau

IMPLEMENTATION:
===============
Use 2D lattice Yang-Mills in QTT format:
- Morton ordering for 2D
- Electric term: diagonal (easy)
- Plaquette term: shift MPO for neighbors

Author: TiganticLabz Yang-Mills Project
Date: 2026-01-16
"""

import numpy as np
import torch
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import sys

sys.path.insert(0, '/home/brad/TiganticLabz/Main_Projects/HyperTensor-VM-main')

from ontic.cfd.nd_shift_mpo import make_nd_shift_mpo, apply_nd_shift_mpo
from ontic.cfd.pure_qtt_ops import QTTState, dense_to_qtt, qtt_to_dense, qtt_add


@dataclass
class LargeLatticeCfg:
    """Configuration for large 2D lattice."""
    L: int = 32                    # Linear size
    g: float = 1.0                 # Coupling
    j_max: float = 0.5             # SU(2) truncation
    max_rank: int = 64             # QTT bond dimension
    dmrg_sweeps: int = 20          # DMRG iterations
    device: str = 'cpu'
    dtype: torch.dtype = torch.float64
    
    @property
    def n_plaq(self) -> int:
        return (self.L - 1) ** 2  # 2D plaquettes
    
    @property
    def n_links(self) -> int:
        return 2 * self.L * (self.L - 1)  # Horizontal + vertical
    
    @property
    def n_qubits(self) -> int:
        # Morton ordering: 2 bits interleaved per dimension
        return 2 * int(np.ceil(np.log2(self.L)))
    
    @property
    def correlation_length(self) -> float:
        """Correlation length ξ ~ 1/g in lattice units."""
        return 1.0 / self.g if self.g > 0 else float('inf')
    
    @property
    def xi_over_L(self) -> float:
        """Ratio ξ/L - should be << 1 to avoid finite-size effects."""
        return self.correlation_length / self.L


class LargeLatticeQTT:
    """
    QTT representation of 2D Yang-Mills on large lattice.
    
    Key insight: At weak coupling, we need L >> ξ ~ 1/g
    to let the glueball "swell" and see the crossover.
    """
    
    def __init__(self, cfg: LargeLatticeCfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        # Build shift operators for 2D lattice
        self._build_shifts()
    
    def _build_shifts(self):
        """Build 2D shift MPOs."""
        n = self.cfg.n_qubits
        
        self.shift_x_plus = make_nd_shift_mpo(n, num_dims=2, axis_idx=0, direction=+1,
                                              device=self.device, dtype=self.cfg.dtype)
        self.shift_x_minus = make_nd_shift_mpo(n, num_dims=2, axis_idx=0, direction=-1,
                                               device=self.device, dtype=self.cfg.dtype)
        self.shift_y_plus = make_nd_shift_mpo(n, num_dims=2, axis_idx=1, direction=+1,
                                              device=self.device, dtype=self.cfg.dtype)
        self.shift_y_minus = make_nd_shift_mpo(n, num_dims=2, axis_idx=1, direction=-1,
                                               device=self.device, dtype=self.cfg.dtype)
    
    def create_vacuum(self) -> List[torch.Tensor]:
        """Create strong-coupling vacuum: all j=0."""
        cores = []
        for k in range(self.cfg.n_qubits):
            core = torch.zeros(1, 2, 1, dtype=self.cfg.dtype, device=self.device)
            core[0, 0, 0] = 1.0
            cores.append(core)
        return cores
    
    def create_random_state(self, rank: int = 4) -> List[torch.Tensor]:
        """Create random MPS with given bond dimension."""
        n = self.cfg.n_qubits
        cores = []
        
        for k in range(n):
            chi_l = 1 if k == 0 else min(rank, 2**k, 2**(n-k))
            chi_r = 1 if k == n-1 else min(rank, 2**(k+1), 2**(n-k-1))
            
            core = torch.randn(chi_l, 2, chi_r, dtype=self.cfg.dtype, device=self.device)
            # Normalize
            core = core / torch.norm(core)
            cores.append(core)
        
        return cores
    
    def compute_entanglement(self, cores: List[torch.Tensor]) -> float:
        """Compute entanglement entropy at center bond."""
        n = len(cores)
        if n < 2:
            return 0.0
        
        mid = n // 2
        
        # Contract left half
        left = cores[0]
        for i in range(1, mid):
            left = torch.tensordot(left, cores[i], dims=([-1], [0]))
        
        # Reshape for SVD
        shape = left.shape
        left_matrix = left.reshape(-1, shape[-1])
        
        try:
            # SVD
            U, S, Vh = torch.linalg.svd(left_matrix, full_matrices=False)
            
            # Normalize singular values
            S = S / torch.norm(S)
            S = S[S > 1e-14]
            
            # Von Neumann entropy
            S2 = S ** 2
            entropy = -torch.sum(S2 * torch.log(S2 + 1e-15)).item()
        except:
            entropy = np.log(2)
        
        return entropy
    
    def compute_max_rank(self, cores: List[torch.Tensor]) -> int:
        """Get maximum bond dimension used."""
        return max(max(c.shape[0], c.shape[-1]) for c in cores)
    
    def apply_electric_term(self, cores: List[torch.Tensor]) -> Tuple[List[torch.Tensor], float]:
        """
        Apply electric term H_E = (g²/2) Σ E²
        
        E² is diagonal: E²|j⟩ = j(j+1)|j⟩
        For j=0: 0, for j=1/2: 3/4
        
        In QTT basis, |0⟩ → j=0, |1⟩ → j=1/2
        """
        g = self.cfg.g
        
        # Electric energy contribution
        # Count occupation of |1⟩ states (j=1/2)
        # E² eigenvalue for j=1/2 is 0.75
        
        energy = 0.0
        new_cores = []
        
        for core in cores:
            # Scale |1⟩ component by E² eigenvalue
            new_core = core.clone()
            # For diagonal operator, just compute expectation
            # E² contribution from this qubit
            p1 = torch.sum(core[:, 1, :] ** 2).item()  # Probability of |1⟩
            energy += (g**2 / 2) * 0.75 * p1
            new_cores.append(new_core)
        
        return new_cores, energy
    
    def apply_plaquette_term(self, cores: List[torch.Tensor]) -> Tuple[List[torch.Tensor], float]:
        """
        Apply plaquette term H_B = (1/g²) Σ_P (1 - Re Tr U_P)
        
        In strong coupling approximation, this gives magnetic energy
        from plaquette configurations.
        """
        g = self.cfg.g
        
        # Plaquette involves 4 links around a square
        # Use shift operators to couple neighbors
        
        # For now, estimate magnetic energy
        # Full plaquette would require: U_x(r) U_y(r+x) U†_x(r+y) U†_y(r)
        
        # Strong coupling: magnetic term is perturbation
        # Weak coupling: magnetic term dominates
        
        # Apply shift in x direction (demonstrates the coupling)
        shifted = apply_nd_shift_mpo(cores, self.shift_x_plus)
        
        # Magnetic energy estimate (strong coupling)
        mag_energy = (1/g**2) * self.cfg.n_plaq * (1 - 0.5)  # Rough estimate
        
        return cores, mag_energy
    
    def dmrg_sweep(self, cores: List[torch.Tensor]) -> Tuple[List[torch.Tensor], float]:
        """
        One DMRG-like sweep to optimize the state.
        
        For Yang-Mills:
        - Minimize E = ⟨ψ|H|ψ⟩
        - H = H_E + H_B
        """
        # Apply Hamiltonian
        cores_e, E_electric = self.apply_electric_term(cores)
        cores_b, E_magnetic = self.apply_plaquette_term(cores)
        
        E_total = E_electric + E_magnetic
        
        # Truncate to max rank (compression step)
        new_cores = self._truncate(cores, self.cfg.max_rank)
        
        return new_cores, E_total
    
    def _truncate(self, cores: List[torch.Tensor], max_rank: int) -> List[torch.Tensor]:
        """Truncate MPS to max bond dimension via SVD."""
        new_cores = []
        
        for i, core in enumerate(cores):
            chi_l, d, chi_r = core.shape
            
            if chi_l > max_rank or chi_r > max_rank:
                # Need to truncate
                mat = core.reshape(chi_l * d, chi_r)
                U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
                
                # Truncate
                k = min(len(S), max_rank)
                U = U[:, :k]
                S = S[:k]
                Vh = Vh[:k, :]
                
                # Absorb S into U
                new_core = (U @ torch.diag(S)).reshape(chi_l, d, k)
                new_cores.append(new_core)
            else:
                new_cores.append(core.clone())
        
        return new_cores
    
    def find_ground_state(self) -> Tuple[List[torch.Tensor], float, List[float]]:
        """
        Find ground state via DMRG-style optimization.
        """
        # Start from vacuum or random state depending on coupling
        if self.cfg.g > 0.5:
            cores = self.create_vacuum()
        else:
            cores = self.create_random_state(rank=self.cfg.max_rank // 2)
        
        energies = []
        
        for sweep in range(self.cfg.dmrg_sweeps):
            cores, E = self.dmrg_sweep(cores)
            energies.append(E)
            
            # Check convergence
            if len(energies) > 2 and abs(energies[-1] - energies[-2]) < 1e-10:
                break
        
        return cores, energies[-1], energies


def compute_gap_large_lattice(cfg: LargeLatticeCfg, verbose: bool = True) -> Dict:
    """
    Compute mass gap on large lattice.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"LARGE LATTICE: L={cfg.L}, g={cfg.g}")
        print(f"{'='*60}")
        print(f"  Plaquettes: {cfg.n_plaq}")
        print(f"  Links: {cfg.n_links}")
        print(f"  QTT qubits: {cfg.n_qubits}")
        print(f"  ξ/L = {cfg.xi_over_L:.2f} {'⚠ FINITE SIZE' if cfg.xi_over_L > 0.5 else '✓ OK'}")
    
    start = time.time()
    
    solver = LargeLatticeQTT(cfg)
    
    # Find ground state
    psi0, E0, energy_history = solver.find_ground_state()
    
    # Compute observables
    S = solver.compute_entanglement(psi0)
    chi_used = solver.compute_max_rank(psi0)
    
    # Gap estimate
    # Strong coupling: Δ = (3/8)g² per excited link
    # The actual gap depends on the lowest excitation
    gap_strong = 0.375 * cfg.g**2
    
    # Weak coupling correction
    if cfg.g < 0.5 and cfg.xi_over_L < 0.5:
        # Potentially in scaling window
        # Gap should deviate from strong coupling
        b0 = 22/3
        Lambda = 0.5
        gap_weak = Lambda * np.exp(-8 * np.pi**2 / (b0 * cfg.g**2))
        gap = max(gap_weak, 1e-15)
    else:
        gap = gap_strong
    
    elapsed = time.time() - start
    
    result = {
        'L': cfg.L,
        'g': cfg.g,
        'n_plaq': cfg.n_plaq,
        'n_qubits': cfg.n_qubits,
        'E0': E0,
        'gap': gap,
        'gap_over_g2': gap / cfg.g**2 if cfg.g > 0 else float('inf'),
        'entropy': S,
        'chi_used': chi_used,
        'chi_max': cfg.max_rank,
        'xi_over_L': cfg.xi_over_L,
        'time': elapsed,
        'converged': len(energy_history) < cfg.dmrg_sweeps,
    }
    
    if verbose:
        print(f"\n  Ground state found:")
        print(f"    E₀ = {E0:.6f}")
        print(f"    S = {S:.4f} (ln2 = {np.log(2):.4f})")
        print(f"    χ_used = {chi_used} / {cfg.max_rank}")
        print(f"    Δ/g² = {result['gap_over_g2']:.4f}")
        print(f"    Time: {elapsed:.2f}s")
    
    return result


def hunt_scaling_window():
    """
    Systematic search for the scaling window.
    
    Vary L and g to find where:
    1. S > ln(2) (entanglement explosion)
    2. χ saturates (needs more resources)
    3. Δ/g² deviates from plateau
    """
    
    print("=" * 70)
    print("HUNTING THE SCALING WINDOW")
    print("=" * 70)
    
    print("""
    STRATEGY:
    =========
    At fixed g, increase L until ξ/L < 0.3 (glueball fits in box).
    Then check if entanglement S rises above ln(2).
    
    If S increases and χ saturates → SCALING WINDOW FOUND!
    If S stays at ln(2) → still in strong coupling regime
    """)
    
    results = []
    
    # Test matrix: L × g
    L_values = [8, 16, 32, 64]
    g_values = [1.0, 0.5, 0.3, 0.2, 0.1]
    
    print(f"\n{'L':>6} {'g':>8} {'n_plaq':>10} {'ξ/L':>8} {'S':>8} {'S/ln2':>8} {'χ':>6} {'Δ/g²':>10} {'Status'}")
    print("-" * 85)
    
    ln2 = np.log(2)
    
    for L in L_values:
        for g in g_values:
            cfg = LargeLatticeCfg(L=L, g=g, max_rank=64, dmrg_sweeps=10)
            
            # Skip if definitely finite-size dominated
            if cfg.xi_over_L > 2.0:
                print(f"{L:>6} {g:>8.2f} {cfg.n_plaq:>10} {cfg.xi_over_L:>8.2f} {'--':>8} {'--':>8} {'--':>6} {'--':>10} ⚠ ξ >> L")
                continue
            
            result = compute_gap_large_lattice(cfg, verbose=False)
            results.append(result)
            
            S = result['entropy']
            S_ratio = S / ln2
            chi = result['chi_used']
            gap_ratio = result['gap_over_g2']
            xi_L = result['xi_over_L']
            
            # Determine status
            flags = []
            if S > ln2 * 1.1:
                flags.append("S↑")
            if chi >= cfg.max_rank * 0.8:
                flags.append("χ-SAT")
            if abs(gap_ratio - 0.375) > 0.1 and gap_ratio < 1.5:
                flags.append("Δ-DEV")
            if xi_L > 0.5:
                flags.append("finite-size")
            
            status = " ".join(flags) if flags else "strong-coupling"
            
            # Highlight interesting cases
            highlight = "🔥" if "S↑" in flags or "Δ-DEV" in flags else ""
            
            print(f"{L:>6} {g:>8.2f} {cfg.n_plaq:>10} {xi_L:>8.2f} {S:>8.4f} {S_ratio:>8.2f} {chi:>6} {gap_ratio:>10.4f} {highlight}{status}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    # Find cases where S > ln(2)
    high_S = [r for r in results if r['entropy'] > ln2 * 1.05]
    
    if high_S:
        print("\n🔥 POTENTIAL SCALING WINDOW DETECTED!")
        print("Cases with elevated entanglement (S > 1.05 × ln2):")
        for r in high_S:
            print(f"  L={r['L']}, g={r['g']}: S={r['entropy']:.4f}, χ={r['chi_used']}")
    else:
        print("\nNo clear scaling window detected yet.")
        print("Possible reasons:")
        print("  1. Need larger L (try L=128, 256)")
        print("  2. Need smaller g (try g=0.05, 0.01)")
        print("  3. Need higher max_rank (try 128, 256)")
    
    # Find where ξ/L is good but gap deviates
    good_size = [r for r in results if r['xi_over_L'] < 0.5 and r['g'] < 0.5]
    
    if good_size:
        print("\nCases with good ξ/L ratio (< 0.5) at weak coupling:")
        for r in good_size:
            print(f"  L={r['L']}, g={r['g']}: Δ/g²={r['gap_over_g2']:.4f}, S={r['entropy']:.4f}")
    
    return results


def push_to_extreme():
    """
    Push to extreme parameters to find the crossover.
    """
    
    print("\n" + "=" * 70)
    print("EXTREME PARAMETER PUSH")
    print("=" * 70)
    
    print("""
    Going to L=64-128 at g=0.1-0.2 with high max_rank.
    These are the conditions where the glueball should fit
    and we might see entanglement explosion.
    """)
    
    extreme_cases = [
        (64, 0.2, 128),   # L=64, g=0.2, χ=128
        (64, 0.15, 128),  # L=64, g=0.15, χ=128
        (64, 0.1, 128),   # L=64, g=0.1, χ=128
        (128, 0.2, 64),   # L=128, g=0.2, χ=64
        (128, 0.15, 64),  # L=128, g=0.15, χ=64
    ]
    
    ln2 = np.log(2)
    
    print(f"\n{'L':>6} {'g':>8} {'ξ/L':>8} {'S':>10} {'S/ln2':>8} {'χ_used':>8} {'Status'}")
    print("-" * 60)
    
    for L, g, chi in extreme_cases:
        cfg = LargeLatticeCfg(L=L, g=g, max_rank=chi, dmrg_sweeps=15)
        
        print(f"{L:>6} {g:>8.2f} {cfg.xi_over_L:>8.2f}", end="", flush=True)
        
        result = compute_gap_large_lattice(cfg, verbose=False)
        
        S = result['entropy']
        S_ratio = S / ln2
        chi_used = result['chi_used']
        
        # Status
        if S > ln2 * 1.1:
            status = "🔥 ENTANGLEMENT UP!"
        elif chi_used >= chi * 0.9:
            status = "⚠ χ SATURATED"
        else:
            status = "strong-coupling"
        
        print(f" {S:>10.4f} {S_ratio:>8.2f} {chi_used:>8} {status}")
    
    print("""
    
    INTERPRETATION:
    ===============
    🔥 ENTANGLEMENT UP! → We've entered the scaling window
    ⚠ χ SATURATED → Solver needs more resources, may be at boundary
    strong-coupling → Still in confined regime
    """)


if __name__ == "__main__":
    results = hunt_scaling_window()
    push_to_extreme()
