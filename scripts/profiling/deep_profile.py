"""
Deep profiling with torch.profiler to see CPU/GPU execution.
"""
import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

device = 'cuda:0'

# Simulate actual workload
print("Profiling actual operations...")
print("="*60)

# Test 1: Fluid solver operations
shape = (64, 64, 64)
u = torch.randn(shape, device=device)

# Test 2: Interpolation (CRITICAL for 4K)
scalar = torch.randn(64, 64, device=device)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True
) as prof:
    with record_function("fluid_laplacian"):
        for _ in range(10):
            lap = (
                torch.roll(u, 1, 0) + torch.roll(u, -1, 0) +
                torch.roll(u, 1, 1) + torch.roll(u, -1, 1) +
                torch.roll(u, 1, 2) + torch.roll(u, -1, 2) -
                6 * u
            )
    
    with record_function("interpolate_4k"):
        for _ in range(5):
            resized = F.interpolate(
                scalar.unsqueeze(0).unsqueeze(0),
                size=(3840, 2160),
                mode='bilinear',
                align_corners=False
            )

# Print profiler results
print("\nProfiler Results:")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Check if operations ran on CUDA
print("\n" + "="*60)
print("CUDA Events:")
events = prof.events()
cuda_events = [e for e in events if e.device_type == torch.device('cuda').type]
print(f"Total events: {len(events)}")
print(f"CUDA events: {len(cuda_events)}")
print(f"CPU events: {len(events) - len(cuda_events)}")

if len(cuda_events) == 0:
    print("\n⚠️ WARNING: No CUDA events detected!")
    print("Operations are running on CPU despite tensors being on GPU!")
