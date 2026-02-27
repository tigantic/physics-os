"""
Profile render operations at 4K to find bottleneck.
"""

import torch
import torch.nn.functional as F

device = "cuda:0"
height, width = 2160, 3840

print("=" * 60)
print("4K RENDER PROFILING")
print("=" * 60)

# 1. Interpolation (64x64 → 4K)
print("\n1. Interpolation (64x64 → 3840x2160)...")
scalar = torch.randn(64, 64, device=device)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(10):
    resized = F.interpolate(
        scalar.unsqueeze(0).unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze()
end.record()
torch.cuda.synchronize()
print(f"   Per frame: {start.elapsed_time(end)/10:.2f}ms")

# 2. LUT indexing
print("\n2. Plasma LUT indexing...")
lut = torch.randn(256, 3, device=device)
indices = torch.randint(0, 256, (height, width), device=device)

start.record()
for _ in range(10):
    rgb = lut[indices.flatten()].view(height, width, 3)
end.record()
torch.cuda.synchronize()
print(f"   Per frame: {start.elapsed_time(end)/10:.2f}ms")

# 3. Layer blending (5 layers)
print("\n3. Layer compositing (5 layers)...")
layers = [
    torch.randn(height, width, 4, device=device, dtype=torch.float16) for _ in range(5)
]

start.record()
for _ in range(10):
    result = layers[0].clone()
    for layer in layers[1:]:
        alpha = layer[:, :, 3:4]
        result = result * (1 - alpha) + layer * alpha
end.record()
torch.cuda.synchronize()
print(f"   Per frame: {start.elapsed_time(end)/10:.2f}ms")

# 4. Grid copy
print("\n4. Grid mask copy...")
grid_mask = torch.randn(height, width, 4, device=device, dtype=torch.float16)
layer_buffer = torch.empty_like(grid_mask)

start.record()
for _ in range(10):
    layer_buffer.copy_(grid_mask)
end.record()
torch.cuda.synchronize()
print(f"   Per frame: {start.elapsed_time(end)/10:.2f}ms")

# 5. HUD rendering (slicing + assignment)
print("\n5. HUD rendering...")
hud_layer = torch.zeros(height, width, 4, device=device, dtype=torch.float16)
color = torch.tensor([0.1, 0.1, 0.1], device=device, dtype=torch.float16)

start.record()
for _ in range(10):
    hud_layer[0:100, 0:200, :3] = color
end.record()
torch.cuda.synchronize()
print(f"   Per frame: {start.elapsed_time(end)/10:.2f}ms")

print("\n" + "=" * 60)
print("Total estimated render time: Sum of above operations")
print("=" * 60)
