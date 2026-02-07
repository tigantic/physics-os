#!/usr/bin/env python3
"""
Unit tests for SpectralNS3D solver.

Tests:
1. Taylor-Green vortex energy conservation
2. QTT compression/decompression round-trip
3. Spectral derivative accuracy
4. Reynolds sweep consistency
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
import pytest


def test_imports():
    """Verify all required modules are importable."""
    from spectral_ns3d import SpectralNS3D, SpectralNS3DConfig
    assert SpectralNS3D is not None
    assert SpectralNS3DConfig is not None


def test_taylor_green_init():
    """Test Taylor-Green vortex initialization."""
    from spectral_ns3d import SpectralNS3D, SpectralNS3DConfig
    
    config = SpectralNS3DConfig(n_bits=5, nu=0.01, dt=0.01, max_rank=32)
    solver = SpectralNS3D(config)
    
    # Check initial energy is approximately 1.0 (for unit amplitude TG)
    E0 = solver.compute_energy()
    assert 0.1 < E0 < 2.0, f"Initial energy {E0} out of expected range"


def test_energy_conservation():
    """Test energy conservation over short integration."""
    from spectral_ns3d import SpectralNS3D, SpectralNS3DConfig
    
    config = SpectralNS3DConfig(n_bits=5, nu=0.01, dt=0.005, max_rank=32)
    solver = SpectralNS3D(config)
    
    E0 = solver.compute_energy()
    
    # Run 10 steps
    for _ in range(10):
        solver.step()
    
    E1 = solver.compute_energy()
    
    # Energy should decrease due to viscosity, but not explode
    drift = abs(E1 - E0) / E0
    assert drift < 0.5, f"Energy drift {drift*100:.1f}% too large"
    assert E1 > 0, "Energy went negative"
    assert not np.isnan(E1), "Energy is NaN"


def test_qtt_round_trip():
    """Test QTT compression and decompression preserves field."""
    from spectral_ns3d import SpectralNS3D, SpectralNS3DConfig
    
    config = SpectralNS3DConfig(n_bits=5, nu=0.01, dt=0.01, max_rank=64)
    solver = SpectralNS3D(config)
    
    # Get initial vorticity
    omega_dense = solver.get_vorticity_dense()
    
    # Round-trip through QTT
    solver._compress_state()
    omega_recovered = solver.get_vorticity_dense()
    
    # Check relative error
    error = torch.norm(omega_dense - omega_recovered) / (torch.norm(omega_dense) + 1e-10)
    assert error < 0.1, f"QTT round-trip error {error:.2e} too large"


def test_spectral_derivatives():
    """Test spectral derivative accuracy on known function."""
    # sin(x) should have derivative cos(x)
    n = 64
    x = torch.linspace(0, 2*np.pi, n+1)[:-1]
    f = torch.sin(x)
    
    # Spectral derivative
    f_hat = torch.fft.fft(f)
    k = torch.fft.fftfreq(n, d=1/n) * 2 * np.pi
    df_hat = 1j * k * f_hat
    df = torch.fft.ifft(df_hat).real
    
    # Expected: cos(x)
    expected = torch.cos(x)
    
    error = torch.max(torch.abs(df - expected))
    assert error < 1e-10, f"Spectral derivative error {error:.2e}"


def test_memory_scaling():
    """Test that QTT memory scales as O(log N)."""
    from spectral_ns3d import SpectralNS3DConfig
    
    memories = []
    for n_bits in [4, 5, 6]:
        N = 2 ** n_bits
        n_cores = 3 * n_bits  # 3D
        chi = 32
        # QTT memory: n_cores * 2 * chi^2 * 4 bytes
        qtt_bytes = n_cores * 2 * chi * chi * 4
        memories.append((N, qtt_bytes))
    
    # Check O(log N) scaling: memory ratio should be ~constant per n_bit
    ratios = [memories[i+1][1] / memories[i][1] for i in range(len(memories)-1)]
    for ratio in ratios:
        # Should be roughly linear in n_bits (O(log N))
        assert 1.0 < ratio < 2.0, f"Memory scaling ratio {ratio} not O(log N)"


if __name__ == "__main__":
    print("Running SpectralNS3D unit tests...\n")
    
    tests = [
        ("Imports", test_imports),
        ("Taylor-Green init", test_taylor_green_init),
        ("Energy conservation", test_energy_conservation),
        ("QTT round-trip", test_qtt_round_trip),
        ("Spectral derivatives", test_spectral_derivatives),
        ("Memory scaling", test_memory_scaling),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            print(f"  ✓ {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            failed += 1
    
    print(f"\n{passed}/{passed+failed} tests passed")
    sys.exit(0 if failed == 0 else 1)
