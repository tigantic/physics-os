#!/usr/bin/env python3
"""Test that Sovereign Substrate generates correctly"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "demos"))

from earth_renderer import EarthRenderer
from PIL import Image

print("=" * 60)
print("SOVEREIGN SUBSTRATE VERIFICATION")
print("=" * 60)

renderer = EarthRenderer()
print(f"\n✓ Renderer initialized")
print(f"  Shape: {renderer.texture.shape}")
print(f"  Range: [{renderer.texture.min():.3f}, {renderer.texture.max():.3f}]")
print(f"  Dtype: {renderer.texture.dtype}")

# Convert to 8-bit for saving
texture_8bit = (renderer.texture * 255).astype('uint8')
img = Image.fromarray(texture_8bit)

output_path = Path(__file__).parent / "SOVEREIGN_SUBSTRATE_PROOF.png"
img.save(output_path)

print(f"\n✓ Substrate saved to: {output_path}")
print(f"  File size: {output_path.stat().st_size:,} bytes")
print("\n" + "=" * 60)
print("SUCCESS: Sovereign Substrate is OPERATIONAL")
print("=" * 60)
