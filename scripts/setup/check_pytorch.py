import torch

print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
print("Built with CUDA:", torch.backends.cuda.is_built())
print("cuDNN enabled:", torch.backends.cudnn.enabled)

# Check if operations actually run on GPU
print("\nTesting GPU execution...")
x = torch.randn(1000, 1000, device="cuda")
y = torch.randn(1000, 1000, device="cuda")
z = torch.matmul(x, y)
print(f"Result device: {z.device}")
print(f"Result shape: {z.shape}")
