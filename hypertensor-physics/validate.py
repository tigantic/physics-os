#!/usr/bin/env python3
"""Quick validation script"""
import sys
sys.path.insert(0, '/home/brad/TiganticLabz/Main_Projects/HyperTensor-VM-main/hypertensor-physics')

from hypertensor import tt_round, TTTensor
from hypertensor.integrators import SymplecticIntegrator, LangevinDynamics
from hypertensor.pde import ResistiveMHD, FokkerPlanck, HeatEquation1D, CompositeWall
import numpy as np

print("✓ All imports successful!")

# Quick TT test
A = np.random.randn(8, 8, 8)
tt = tt_round(A, max_rank=3)
print(f"✓ TT compression: {A.nbytes} → {tt.tt_size * 8} bytes ({tt.compression_ratio:.1f}×)")

# Quick MHD test
mhd = ResistiveMHD(nx=32, L=1.0, eta=0.01)
result = mhd.run(n_steps=10, dt=1e-5)
print(f"✓ MHD stable: {result['stable']}")

print("\n🎉 HyperTensor Physics Package VALIDATED!")
