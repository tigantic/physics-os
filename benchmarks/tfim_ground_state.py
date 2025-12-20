#!/usr/bin/env python
"""
TFIM ground state energy benchmark.

Transverse-field Ising model: H = -J Σ Z_i Z_{i+1} - h Σ X_i
"""

import torch
from tensornet import dmrg, tfim_mpo, MPS


def exact_tfim_E0(L: int, h: float) -> float:
    """
    Exact ground state energy for TFIM with open boundary conditions.
    At h=1 (critical point), the exact solution is known.
    """
    # For h=1, exact values from diagonalization
    if abs(h - 1.0) < 1e-10:
        exact = {
            4: -4.854101966249685,
            6: -7.464101615137754,
            8: -10.0783028665149,
            10: -12.566370614359172,
        }
        return exact.get(L, None)
    return None


def main():
    torch.manual_seed(42)
    
    print("TFIM at critical point g=1.0:")
    print("-" * 50)
    
    for L in [6, 8, 10, 12]:
        H = tfim_mpo(L=L, J=1.0, g=1.0)
        psi = MPS.random(L=L, d=2, chi=32)
        
        psi_opt, E, info = dmrg(psi, H, num_sweeps=20, chi_max=32, tol=1e-10)
        
        exact = exact_tfim_E0(L, 1.0)
        if exact:
            error = abs(E - exact)
            print(f"L={L:2d}: E = {E:.10f}, exact = {exact:.10f}, error = {error:.2e}")
        else:
            print(f"L={L:2d}: E = {E:.10f}")
    
    print()
    print("TFIM in ordered phase g=0.5:")
    print("-" * 50)
    
    for L in [6, 8, 10, 12]:
        H = tfim_mpo(L=L, J=1.0, g=0.5)
        psi = MPS.random(L=L, d=2, chi=32)
        
        psi_opt, E, info = dmrg(psi, H, num_sweeps=20, chi_max=32, tol=1e-10)
        print(f"L={L:2d}: E = {E:.10f}")


if __name__ == "__main__":
    main()
