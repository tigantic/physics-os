"""
Yang-Mills Tensor Network Module
================================

Implements Matrix Product States (MPS) and Density Matrix Renormalization Group (DMRG)
for accessing the weak coupling regime of Yang-Mills theory.

The strong coupling result Δ = (3/2)g² vanishes in the continuum limit because:
- Asymptotic freedom: g → 0 as a → 0
- Physical gap: Δ_physical ~ g² × exp(-1/(2β₀g²)) → 0

To solve the Millennium Prize, we need tensor networks that can:
1. Handle exponentially growing entanglement at weak coupling
2. Detect dimensional transmutation: Δ ~ Λ_QCD
3. Access the non-perturbative regime

This module provides:
- MPS: Matrix Product State representation
- MPO: Matrix Product Operator for Hamiltonian
- DMRG: Ground state optimization
"""

from .mps import MPS
from .mpo import MPOHamiltonian, YangMillsMPO
from .dmrg import DMRG, compute_gap_tensor_network, scan_coupling_range

__all__ = ['MPS', 'MPOHamiltonian', 'YangMillsMPO', 'DMRG', 'compute_gap_tensor_network', 'scan_coupling_range']
