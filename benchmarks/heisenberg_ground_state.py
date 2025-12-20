#!/usr/bin/env python
"""
Heisenberg ground state energy benchmark.

Computes E0 for the Heisenberg XXX chain H = Σ S_i · S_{i+1}
and compares to exact/reference values.
"""

import torch
from tensornet import dmrg, heisenberg_mpo, MPS


def exact_heisenberg_E0(L: int) -> float:
    """
    Exact ground state energy for Heisenberg chain via Bethe ansatz.
    For small L, we use exact diagonalization values.
    """
    # Exact values from diagonalization (periodic BC would differ)
    exact = {
        2: -0.75,
        4: -1.6160254037844388,
        6: -2.493577133567863,
        8: -3.374932598230364,
        10: -4.258035207282883,
    }
    return exact.get(L, None)


def main():
    torch.manual_seed(42)
    
    for L in [6, 8, 10, 12, 14]:
        H = heisenberg_mpo(L=L, J=1.0, h=0.0)
        psi = MPS.random(L=L, d=2, chi=32)
        
        psi_opt, E, info = dmrg(psi, H, num_sweeps=20, chi_max=32, tol=1e-10)
        
        exact = exact_heisenberg_E0(L)
        if exact:
            error = abs(E - exact)
            print(f"L={L:2d}: E = {E:.10f}, exact = {exact:.10f}, error = {error:.2e}")
        else:
            print(f"L={L:2d}: E = {E:.10f}")


if __name__ == "__main__":
    main()
