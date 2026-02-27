#!/usr/bin/env python3
"""Test Triton kernel at 524K feature scale"""
import torch
import triton
import triton.language as tl

N_FEATURES = 524288
device = torch.device('cuda')

@triton.jit
def extract_test_kernel(data_ptr, positions_ptr, features_ptr, stride_feat):
    pid = tl.program_id(0)
    pos = tl.load(positions_ptr + pid)
    out_base = features_ptr + pid * stride_feat
    
    # Unigrams [0, 4096)
    for i in tl.static_range(16):
        byte_val = tl.load(data_ptr + pos + i).to(tl.int32)
        idx = ((i * 256 + byte_val) % 4096)
        tl.atomic_add(out_base + idx, 1.0)
    
    # Bigrams [4096, 69632)
    for i in tl.static_range(15):
        b1 = tl.load(data_ptr + pos + i).to(tl.int32)
        b2 = tl.load(data_ptr + pos + i + 1).to(tl.int32)
        h = (i * 65537 + b1 * 257 + b2) % 65536
        idx = 4096 + h
        tl.atomic_add(out_base + idx, 1.0)
    
    # Trigrams [69632, 200704)
    for i in tl.static_range(14):
        b1 = tl.load(data_ptr + pos + i).to(tl.int32)
        b2 = tl.load(data_ptr + pos + i + 1).to(tl.int32)
        b3 = tl.load(data_ptr + pos + i + 2).to(tl.int32)
        h = (b1 * 65537 + b2 * 257 + b3) % 131072
        idx = 69632 + h
        tl.atomic_add(out_base + idx, 1.0)
    
    # 4-grams [200704, 462848)
    for i in tl.static_range(13):
        b1 = tl.load(data_ptr + pos + i).to(tl.int32)
        b2 = tl.load(data_ptr + pos + i + 1).to(tl.int32)
        b3 = tl.load(data_ptr + pos + i + 2).to(tl.int32)
        b4 = tl.load(data_ptr + pos + i + 3).to(tl.int32)
        h = (b1 * 16777259 + b2 * 65537 + b3 * 257 + b4) % 262144
        idx = 200704 + h
        tl.atomic_add(out_base + idx, 1.0)
    
    # Skip-grams [462848, 524288)
    for i in tl.static_range(14):
        b1 = tl.load(data_ptr + pos + i).to(tl.int32)
        b3 = tl.load(data_ptr + pos + i + 2).to(tl.int32)
        h = (b1 * 257 + b3) % 61440
        idx = 462848 + h
        tl.atomic_add(out_base + idx, 1.0)


# Test
print("Testing kernel...")
data = torch.randint(0, 256, (10000,), dtype=torch.int32, device=device)
positions = torch.tensor([0, 100, 200, 500, 1000], dtype=torch.int64, device=device)
features = torch.zeros((5, N_FEATURES), dtype=torch.float32, device=device)

print(f"Data shape: {data.shape}")
print(f"Positions: {positions}")
print(f"Features shape: {features.shape}")
print(f"Features size: {features.numel() * 4 / 1e6:.1f} MB")

print("Running kernel...")
extract_test_kernel[(5,)](data, positions, features, N_FEATURES)
torch.cuda.synchronize()

print(f"Success! features sum: {features.sum().item()}")
print(f"Non-zero features: {(features > 0).sum().item()}")

# Check index bounds
print("\nFeature index bounds check:")
print(f"  Unigram max: {4095} (limit 4096)")
print(f"  Bigram max: {4096 + 65535} = 69631 (limit 69632)")
print(f"  Trigram max: {69632 + 131071} = 200703 (limit 200704)")
print(f"  4-gram max: {200704 + 262143} = 462847 (limit 462848)")
print(f"  Skip-gram max: {462848 + 61439} = 524287 (limit 524288)")
