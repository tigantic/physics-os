"""
Component-Level Performance Profiling
======================================

Isolate and measure each pipeline component to validate audit assumptions.
"""

import torch
import time
import numpy as np

# Setup
device = torch.device('cuda:0')
width, height = 3840, 2160
n_trials = 100

print("="*60)
print("COMPONENT PROFILING @ 4K (3840×2160)")
print("="*60)
print(f"Trials: {n_trials}")
print(f"Device: {torch.cuda.get_device_name(0)}\n")

# Warm-up GPU
_ = torch.randn(1024, 1024, device=device) @ torch.randn(1024, 1024, device=device)
torch.cuda.synchronize()

# ============================================================================
# Test 1: Compositor (5-layer blend)
# ============================================================================
print("[Test 1] Compositor (5-layer alpha blend)")

# Create mock layers
layers = []
for i in range(5):
    layer = torch.randn(height, width, 4, dtype=torch.float16, device=device).clamp(0, 1)
    layers.append(layer)

final_buffer = torch.zeros(height, width, 4, dtype=torch.float32, device=device)

torch.cuda.synchronize()
times = []

for trial in range(n_trials):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    
    # Copy base layer
    final_buffer.copy_(layers[0])
    
    # Blend remaining layers (ADDITIVE)
    for layer in layers[1:]:
        src = layer
        alpha = src[:, :, 3:4]
        final_buffer[:, :, :3].add_(src[:, :, :3] * alpha).clamp_(0, 1)
    
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))

avg_time = np.mean(times)
std_time = np.std(times)
print(f"  Average: {avg_time:.2f}ms ± {std_time:.2f}ms")
print(f"  FPS: {1000/avg_time:.1f}")
print(f"  Memory traffic: {5*66 + 132:.0f}MB read + {4*132:.0f}MB write = {5*66 + 5*132:.0f}MB")
print()

# ============================================================================
# Test 2: Grid Rendering (memcpy + slice assignment)
# ============================================================================
print("[Test 2] Grid Rendering (pre-computed memcpy)")

grid_mask = torch.randn(height, width, 4, dtype=torch.float16, device=device)
grid_buffer = torch.zeros(height, width, 4, dtype=torch.float16, device=device)

torch.cuda.synchronize()
times = []

for trial in range(n_trials):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    grid_buffer.copy_(grid_mask)
    end.record()
    
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))

avg_time = np.mean(times)
std_time = np.std(times)
print(f"  Average: {avg_time:.2f}ms ± {std_time:.2f}ms")
print(f"  FPS: {1000/avg_time:.1f}")
print(f"  Memory traffic: 66MB read + 66MB write = 132MB")
print()

# ============================================================================
# Test 3: HUD Rendering (filled rectangle)
# ============================================================================
print("[Test 3] HUD Rendering (filled rectangle)")

hud_buffer = torch.zeros(height, width, 4, dtype=torch.float16, device=device)
color = torch.tensor([0.1, 0.1, 0.15], device=device)

torch.cuda.synchronize()
times = []

for trial in range(n_trials):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    
    # Fill 200×200 rectangle in top-left
    hud_buffer[0:200, 0:200, :3] = color
    hud_buffer[0:200, 0:200, 3] = 0.9
    
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))

avg_time = np.mean(times)
std_time = np.std(times)
print(f"  Average: {avg_time:.2f}ms ± {std_time:.2f}ms")
print(f"  FPS: {1000/avg_time:.1f}")
print()

# ============================================================================
# Test 4: Float16 → Float32 Conversion
# ============================================================================
print("[Test 4] Float16 → Float32 Conversion (layer → final)")

src_f16 = torch.randn(height, width, 4, dtype=torch.float16, device=device)
dst_f32 = torch.zeros(height, width, 4, dtype=torch.float32, device=device)

torch.cuda.synchronize()
times = []

for trial in range(n_trials):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    dst_f32.copy_(src_f16)
    end.record()
    
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))

avg_time = np.mean(times)
std_time = np.std(times)
print(f"  Average: {avg_time:.2f}ms ± {std_time:.2f}ms")
print(f"  FPS: {1000/avg_time:.1f}")
print(f"  Overhead vs same-precision: ~{avg_time - 1.0:.2f}ms per layer")
print()

# ============================================================================
# Test 5: Bicubic Interpolation (256×256 → 4K)
# ============================================================================
print("[Test 5] Bicubic Interpolation (256×256 → 3840×2160)")

sparse = torch.randn(1, 1, 256, 256, device=device)

torch.cuda.synchronize()
times = []

for trial in range(n_trials):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    dense = torch.nn.functional.interpolate(sparse, size=(height, width), mode='bicubic', align_corners=False)
    end.record()
    
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))

avg_time = np.mean(times)
std_time = np.std(times)
print(f"  Average: {avg_time:.2f}ms ± {std_time:.2f}ms")
print(f"  FPS: {1000/avg_time:.1f}")
print()

# ============================================================================
# Test 6: Colormap Application (LUT indexing)
# ============================================================================
print("[Test 6] Colormap Application (256-entry LUT)")

scalar_field = torch.rand(height, width, device=device)
lut = torch.randn(256, 3, device=device)

torch.cuda.synchronize()
times = []

for trial in range(n_trials):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    
    indices = (scalar_field * 255).long().clamp(0, 255)
    rgb = lut[indices]
    
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))

avg_time = np.mean(times)
std_time = np.std(times)
print(f"  Average: {avg_time:.2f}ms ± {std_time:.2f}ms")
print(f"  FPS: {1000/avg_time:.1f}")
print()

# ============================================================================
# Test 7: Opacity Mapping (mean + custom curve)
# ============================================================================
print("[Test 7] Opacity Mapping (luminance + power curve)")

rgb_field = torch.rand(height, width, 3, device=device)

torch.cuda.synchronize()
times = []

for trial in range(n_trials):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    
    luminance = rgb_field.mean(dim=-1)
    alpha = torch.pow(luminance, 0.5)  # Simple power curve
    
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))

avg_time = np.mean(times)
std_time = np.std(times)
print(f"  Average: {avg_time:.2f}ms ± {std_time:.2f}ms")
print(f"  FPS: {1000/avg_time:.1f}")
print()

print("="*60)
print("PROFILING COMPLETE")
print("="*60)
