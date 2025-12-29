#!/usr/bin/env python3
"""Check CUDA device assignment"""

import torch

print("CUDA Device Information")
print("="*70)
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print()

for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    props = torch.cuda.get_device_properties(i)
    print(f"  Compute capability: {props.major}.{props.minor}")
    print(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")
    print(f"  Multi-processors: {props.multi_processor_count}")
    print()

print(f"Current device: {torch.cuda.current_device()}")
print(f"Current device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
print()

# Test which device actually gets used
print("Creating tensor with device='cuda:0'...")
x = torch.randn(100, 100, device='cuda:0')
print(f"Tensor device: {x.device}")
print()

print("="*70)
print("RECOMMENDATION:")
print("="*70)
if torch.cuda.device_count() > 1:
    print("Multiple GPUs detected. Use device='cuda:1' for RTX 5070")
else:
    print("Single GPU detected. Already using correct device.")
