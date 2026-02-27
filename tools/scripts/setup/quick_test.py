#!/usr/bin/env python3
"""Quick performance test of orbital command center with MPO."""

import time

import torch

from tensornet.engine.gateway.orbital_command import OrbitalCommandCenter

print("Initializing OrbitalCommandCenter...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

occ = OrbitalCommandCenter(device=device)
print("✓ Initialized\n")

# Warm up
print("Warming up...")
for i in range(3):
    occ.update_physics(0.01)
    occ.render_frame()
print("✓ Warm up complete\n")

# Benchmark
print("Benchmarking 3 frames...")
timings = []

for i in range(3):
    start = time.perf_counter()
    occ.update_physics(0.01)
    occ.render_frame()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    timings.append(elapsed)
    print(f"  Frame {i+1:2d}: {elapsed:7.2f}ms")

import numpy as np

mean_time = np.mean(timings)
min_time = np.min(timings)
max_time = np.max(timings)

print(f"\n{'Metric':<20} {'Value':>10}")
print("-" * 35)
print(f"{'Mean':<20} {mean_time:>10.2f} ms")
print(f"{'Min':<20} {min_time:>10.2f} ms")
print(f"{'Max':<20} {max_time:>10.2f} ms")
print(f"{'FPS (mean)':<20} {1000/mean_time:>10.1f}")
print(f"{'Target (60 FPS)':<20} {'16.67':>10} ms")

if mean_time <= 16.67:
    print("\n✓✓✓ 60 FPS TARGET ACHIEVED ✓✓✓")
elif mean_time <= 25:
    print(f"\n✓ Good: ~{1000/mean_time:.0f} FPS (gap: {mean_time-16.67:.2f}ms)")
else:
    print(
        f"\n⚠ Current: ~{1000/mean_time:.0f} FPS (need: {mean_time-16.67:.2f}ms more)"
    )
