#!/usr/bin/env python3
"""Test FluidElite Hybrid with actual training data contexts."""
import struct
import hashlib
from pathlib import Path

def context_hash(context: bytes) -> int:
    """Match the Rust hash function."""
    return int(hashlib.sha256(context).hexdigest()[:16], 16)

# Load model to get lookup table
model_path = Path("/home/brad/TiganticLabz/Main_Projects/HyperTensor-VM-main/fluidelite/data/fluidelite_hybrid.bin")

with open(model_path, 'rb') as f:
    magic = f.read(4)
    assert magic == b'FLHB', f"Bad magic: {magic}"
    
    version = struct.unpack('<I', f.read(4))[0]
    L = struct.unpack('<I', f.read(4))[0]
    vocab = struct.unpack('<I', f.read(4))[0]
    chi = struct.unpack('<I', f.read(4))[0]
    feature_dim = struct.unpack('<I', f.read(4))[0]
    lookup_size = struct.unpack('<Q', f.read(8))[0]
    
    print(f"Model: L={L}, vocab={vocab}, chi={chi}, features={feature_dim}")
    print(f"Lookup table: {lookup_size:,} entries")
    
    # Read first few lookup entries for verification
    lookup = {}
    for _ in range(min(10, lookup_size)):
        h = struct.unpack('<Q', f.read(8))[0]
        v = struct.unpack('B', f.read(1))[0]
        lookup[h] = v

# Load training data and check some contexts
train_path = Path("/home/brad/TiganticLabz/Main_Projects/HyperTensor-VM-main/fluidelite/data/wikitext2_train.txt")
data = train_path.read_bytes()

print(f"\nTraining data: {len(data):,} bytes")
print(f"\nFirst 100 bytes: {data[:100]}")

# Test some actual contexts from training data
print("\n" + "="*60)
print("Testing contexts from training data:")
print("="*60)

for i in [0, 100, 1000, 10000, 100000]:
    if i + L + 1 <= len(data):
        ctx = data[i:i+L]
        target = data[i+L]
        h = context_hash(ctx)
        
        print(f"\nPosition {i}: context='{ctx.decode('utf-8', errors='replace')}' -> target='{chr(target) if 32 <= target < 127 else '?'}'")
        print(f"  Hash: {h}")
        
        # These should be in the lookup table since they're from training data
        # (We can't verify directly here without reading the whole table, but the Rust prover should find them)
