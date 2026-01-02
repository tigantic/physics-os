"""
Profile individual operations to find CPU/GPU bottleneck.
"""
import torch
import time

device = 'cuda:0'
print("="*60)
print("OPERATION PROFILING")
print("="*60)

# Simulate fluid solver operations
print("\n1. Testing fluid solver (64³ grid)...")
shape = (64, 64, 64)
u = torch.randn(shape, device=device)
v = torch.randn(shape, device=device)
w = torch.randn(shape, device=device)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(100):
    # Laplacian with roll (from PCG solver)
    lap = (
        torch.roll(u, 1, 0) + torch.roll(u, -1, 0) +
        torch.roll(u, 1, 1) + torch.roll(u, -1, 1) +
        torch.roll(u, 1, 2) + torch.roll(u, -1, 2) -
        6 * u
    )
end.record()
torch.cuda.synchronize()
print(f"   Laplacian (100 iters): {start.elapsed_time(end):.2f}ms")

# Test interpolation (64x64 → 3840x2160)
print("\n2. Testing interpolation (64x64 → 3840x2160)...")
scalar = torch.randn(64, 64, device=device)

start.record()
for _ in range(10):
    resized = torch.nn.functional.interpolate(
        scalar.unsqueeze(0).unsqueeze(0),
        size=(3840, 2160),
        mode='bilinear',
        align_corners=False
    )
end.record()
torch.cuda.synchronize()
print(f"   Interpolate (10 iters): {start.elapsed_time(end):.2f}ms")
print(f"   Per frame: {start.elapsed_time(end)/10:.2f}ms")

# Test indexing operation (plasma LUT)
print("\n3. Testing LUT indexing (3840x2160)...")
lut = torch.randn(256, 3, device=device)
indices = torch.randint(0, 256, (3840, 2160), device=device)

start.record()
for _ in range(10):
    rgb = lut[indices.flatten()].view(3840, 2160, 3)
end.record()
torch.cuda.synchronize()
print(f"   LUT indexing (10 iters): {start.elapsed_time(end):.2f}ms")
print(f"   Per frame: {start.elapsed_time(end)/10:.2f}ms")

# Test layer compositing
print("\n4. Testing layer blending (3840x2160)...")
layers = [torch.randn(3840, 2160, 4, device=device, dtype=torch.float16) for _ in range(5)]

start.record()
for _ in range(10):
    result = layers[0].clone()
    for layer in layers[1:]:
        # Alpha blend
        alpha = layer[:, :, 3:4]
        result = result * (1 - alpha) + layer * alpha
end.record()
torch.cuda.synchronize()
print(f"   Layer blend (10 iters): {start.elapsed_time(end):.2f}ms")
print(f"   Per frame: {start.elapsed_time(end)/10:.2f}ms")

print("\n" + "="*60)
print("Done. Check if GPU showed any utilization during this.")
