"""
Quick GPU utilization test - verify RTX 5070 is actually being used.
"""
import torch
import time

print("="*60)
print("GPU UTILIZATION TEST")
print("="*60)

# Check device
device = torch.device('cuda:0')
print(f"\nDevice: {torch.cuda.get_device_name(0)}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}")

# Create large tensors to force GPU usage
print("\nAllocating 1GB tensor on GPU...")
size = 128 * 1024 * 1024  # 1GB of float32
x = torch.randn(size, device=device)
print(f"Tensor device: {x.device}")
print(f"VRAM used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# Run heavy computation
print("\nRunning 1000 matrix multiplications...")
a = torch.randn(4096, 4096, device=device)
b = torch.randn(4096, 4096, device=device)

print("GO! Check nvidia-smi now...")
for i in range(1000):
    c = torch.matmul(a, b)
    if i % 100 == 0:
        torch.cuda.synchronize()
        print(f"  Iteration {i}/1000")
        time.sleep(0.5)  # Give time to check nvidia-smi

torch.cuda.synchronize()
print("\nDone. Check nvidia-smi output.")
print(f"Final VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
