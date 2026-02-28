"""
Quick test to measure MPO solver performance.
"""

import time

import torch

from ontic.cfd.qtt_2d import QTT2DState
from ontic.mpo import MPOAtmosphericSolver

# Initialize
device = torch.device("cuda")
solver = MPOAtmosphericSolver(
    grid_size=(64, 64), viscosity=0.001, dt=0.01, dtype=torch.float32, device=device
)

print("Warming up...")
for _ in range(10):
    solver.step()

print("\nMeasuring MPO performance over 100 steps...")
torch.cuda.synchronize()
start = time.perf_counter()

for _ in range(100):
    solver.step()

torch.cuda.synchronize()
elapsed = (time.perf_counter() - start) * 1000  # ms
per_step = elapsed / 100

print(f"Total time: {elapsed:.2f}ms")
print(f"Per step: {per_step:.2f}ms")
print(f"Target: 0.65ms")
print(f"Status: {'✓ PASS' if per_step < 1.0 else '✗ FAIL (needs optimization)'}")

# Test core extraction
print("\nTesting core extraction...")
torch.cuda.synchronize()
start = time.perf_counter()

for _ in range(100):
    u_cores, v_cores = solver.get_cores()

torch.cuda.synchronize()
elapsed = (time.perf_counter() - start) * 1000
per_call = elapsed / 100

print(f"Per call: {per_call:.3f}ms")
print(f"Target: <0.001ms")

# Test QTT2DState wrapping
print("\nTesting QTT2DState wrapping...")
u_cores, v_cores = solver.get_cores()
qtt_state = QTT2DState(cores=u_cores, nx=6, ny=6)
print(f"✓ QTT2DState created")
print(f"  Cores: {len(qtt_state.cores)}")
print(f"  Shape: {qtt_state.shape_2d}")
print(f"  Max rank: {qtt_state.max_rank}")
