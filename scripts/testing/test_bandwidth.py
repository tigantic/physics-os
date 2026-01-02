"""
Test if operations actually run on GPU by measuring memory bandwidth.
If operations run on CPU, they won't saturate GPU memory bandwidth.
"""
import torch
import time

device = 'cuda:0'

print("="*60)
print("MEMORY BANDWIDTH TEST")
print("="*60)

# Test 1: Large memory copy (should saturate GPU bandwidth ~600 GB/s for RTX 5070)
size = 1024 * 1024 * 1024 // 4  # 1GB of float32
print(f"\n1. Testing GPU memory bandwidth (1GB copy)...")

a = torch.randn(size, device=device)
b = torch.empty_like(a)

# Warm-up
for _ in range(10):
    b.copy_(a)
torch.cuda.synchronize()

# Measure
start = time.perf_counter()
iterations = 100
for _ in range(iterations):
    b.copy_(a)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

bandwidth_gbs = (size * 4 * iterations * 2) / elapsed / 1e9  # Read + write
print(f"   Bandwidth: {bandwidth_gbs:.1f} GB/s")
print(f"   Expected GPU: ~500-700 GB/s")
print(f"   Expected CPU: ~50-100 GB/s")

if bandwidth_gbs < 200:
    print("   ⚠️  LOW BANDWIDTH - Operations likely on CPU!")
else:
    print("   ✓ High bandwidth - GPU is working!")

# Test 2: Compute throughput (matmul)
print(f"\n2. Testing compute throughput (matrix multiply)...")
size = 4096
a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)

# Warm-up
for _ in range(5):
    c = torch.matmul(a, b)
torch.cuda.synchronize()

# Measure
start = time.perf_counter()
iterations = 20
for _ in range(iterations):
    c = torch.matmul(a, b)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

flops = 2 * size**3 * iterations  # 2 * N^3 for matmul
tflops = flops / elapsed / 1e12
print(f"   Throughput: {tflops:.2f} TFLOPS")
print(f"   Expected GPU (RTX 5070): ~30-50 TFLOPS")
print(f"   Expected CPU: <1 TFLOPS")

if tflops < 5:
    print("   ⚠️  LOW THROUGHPUT - Operations on CPU!")
else:
    print("   ✓ High throughput - GPU is working!")

print("\n" + "="*60)
