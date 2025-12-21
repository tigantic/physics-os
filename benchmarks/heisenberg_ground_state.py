#!/usr/bin/env python
"""
Heisenberg ground state energy benchmark.

Computes E0 for the Heisenberg XXX chain H = Σ S_i · S_{i+1}
and compares to exact/reference values.

Constitutional Compliance:
    - Article III.3.4: Benchmark with hardware specs
"""

import platform
import psutil
import torch
from tensornet import dmrg, heisenberg_mpo, MPS


def get_hardware_specs() -> dict:
    """Get hardware specifications for benchmark reproducibility (Article III.3.4)."""
    specs = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        specs["gpu_name"] = torch.cuda.get_device_name(0)
        specs["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    return specs


def print_hardware_specs():
    """Print hardware specifications."""
    specs = get_hardware_specs()
    print("=" * 60)
    print("HARDWARE SPECIFICATIONS")
    print("=" * 60)
    for key, value in specs.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    print()


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
    print_hardware_specs()
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
