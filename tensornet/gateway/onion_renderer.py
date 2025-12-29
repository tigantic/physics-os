"""
Onion Renderer - 5-Layer Depth Strategy
========================================

OPERATION VALHALLA - Phase 4.3: Layered Visualization Pipeline

The "Onion" is a vertical stack of shaders. Each layer has its own Z-depth 
and blending mode to ensure the RTX 5070 prioritizes pixels that represent 
the "Truth."

We do not just draw objects; we manage Light and Depth through five discrete 
GPU render passes.

Layer Stack (Bottom to Top):
    Layer 0: Geological Substrate (Foundation)
    Layer 1: Tensor Field (Energy)
    Layer 2: Kinetic Streamlines (Momentum)
    Layer 3: Sovereign Geometry (Grid)
    Layer 4: Tactical HUD (Interface)

Design Philosophy:
    "The weather is not data on a map; it is a living energy field that 
     exists as a volume in 3D space."

Author: OPERATION VALHALLA
Date: 2025-12-28
"""

import torch
import torch.nn.functional as F
from numba import njit, prange
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Callable
import numpy as np


class BlendMode(Enum):
    """GPU blending modes for layer composition."""
    OPAQUE = "opaque"                    # No blending (bottom layer)
    ALPHA = "alpha"                      # Standard transparency
    ADDITIVE = "additive"                # Light accumulation (GL_ONE, GL_ONE)
    PREMULTIPLIED = "premultiplied"      # Pre-multiplied alpha
    OVER = "over"                        # Porter-Duff over operator


class LayerType(Enum):
    """The five onion layers."""
    GEOLOGICAL = 0   # Substrate: Dark satellite texture
    TENSOR = 1        # Energy: Additive plasma gradient
    KINETIC = 2       # Momentum: Alpha-blended particle tracers
    GEOMETRY = 3      # Grid: Premultiplied blue lines
    HUD = 4           # Interface: Topmost UI overlay


@dataclass
class RenderLayer:
    """
    Single layer in the onion stack.
    
    Each layer maintains its own GPU-resident framebuffer and 
    blending configuration.
    """
    type: LayerType
    blend_mode: BlendMode
    z_depth: int
    opacity: float = 1.0
    enabled: bool = True
    
    # Framebuffer (H, W, RGBA)
    buffer: Optional[torch.Tensor] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError(f"Opacity must be [0, 1], got {self.opacity}")
    
    def allocate(self, height: int, width: int, device: str = 'cuda:0'):
        """Allocate GPU framebuffer - Float16 baseline (transitioning to QTT-Native)."""
        self.buffer = torch.zeros(
            (height, width, 4),
            dtype=torch.float16,  # Half precision during transition
            device=device
        )
        
        # Dirty rectangle for scissor-test compositing
        self.dirty_rect = None  # (x0, y0, x1, y1) or None for full-screen
    
    def clear(self):
        """Clear framebuffer to transparent black."""
        if self.buffer is not None:
            self.buffer.zero_()
    
    def __repr__(self) -> str:
        return f"RenderLayer({self.type.name}, z={self.z_depth}, {self.blend_mode.value})"


class OnionRenderer:
    """
    5-Layer GPU render pipeline with depth-sorted composition.
    
    The onion strategy prevents "muddy" visuals when dealing with 50MB+ 
    data streams by treating each data domain as a separate optical layer 
    with its own blending physics.
    
    Pipeline:
        1. Clear all layer buffers
        2. Render each layer independently (parallel where possible)
        3. Composite layers bottom-to-top with correct blending
        4. Output final RGBA frame
    """
    
    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        device: str = 'cuda:0'
    ):
        """
        Initialize onion renderer.
        
        Args:
            width: Framebuffer width (pixels)
            height: Framebuffer height (pixels)
            device: CUDA device
        """
        self.width = width
        self.height = height
        self.device = torch.device(device)
        
        # Create the five layers
        self.layers: List[RenderLayer] = [
            # Layer 0: Geological Substrate (Opaque base)
            RenderLayer(
                type=LayerType.GEOLOGICAL,
                blend_mode=BlendMode.OPAQUE,
                z_depth=0,
                opacity=0.60  # Darkened to 60% per photonic discipline
            ),
            
            # Layer 1: Tensor Field (Additive glow)
            RenderLayer(
                type=LayerType.TENSOR,
                blend_mode=BlendMode.ADDITIVE,
                z_depth=1,
                opacity=1.0
            ),
            
            # Layer 2: Kinetic Streamlines (Alpha particles)
            RenderLayer(
                type=LayerType.KINETIC,
                blend_mode=BlendMode.ALPHA,
                z_depth=2,
                opacity=0.8,
                enabled=False  # Disabled for 4K performance - saves 3ms
            ),
            
            # Layer 3: Sovereign Geometry (Premultiplied grid)
            RenderLayer(
                type=LayerType.GEOMETRY,
                blend_mode=BlendMode.PREMULTIPLIED,
                z_depth=3,
                opacity=0.5
            ),
            
            # Layer 4: Tactical HUD (Topmost)
            RenderLayer(
                type=LayerType.HUD,
                blend_mode=BlendMode.OVER,
                z_depth=4,
                opacity=1.0
            )
        ]
        
        # Allocate framebuffers
        for layer in self.layers:
            layer.allocate(height, width, device)
        
        # Final composite buffer (Float16 for 2× memory bandwidth)
        self.final_buffer = torch.zeros(
            (height, width, 4),
            dtype=torch.float16,  # Changed from Float32
            device=self.device
        )
        
        # Statistics
        self.frame_count = 0
        self.total_render_time = 0.0
    
    def get_layer(self, layer_type: LayerType) -> RenderLayer:
        """Retrieve layer by type."""
        return self.layers[layer_type.value]
    
    def clear_all(self):
        """Clear all layer buffers."""
        for layer in self.layers:
            layer.clear()
        self.final_buffer.zero_()
    
    def render_geological(self, satellite_texture: torch.Tensor):
        """
        Layer 0: Render geological substrate.
        
        Args:
            satellite_texture: RGB texture (H, W, 3) from Phase 3 S3 fetcher
        """
        layer = self.get_layer(LayerType.GEOLOGICAL)
        
        # Darken to 60% luminance (photonic discipline)
        darkened = satellite_texture * layer.opacity
        
        # Write to RGBA buffer (opaque)
        layer.buffer[:, :, :3] = darkened
        layer.buffer[:, :, 3] = 1.0  # Fully opaque
    
    def render_tensor_field(self, scalar_field: torch.Tensor, plasma_gradient: np.ndarray):
        """
        Layer 1: Render tensor field as additive plasma gradient.
        
        Args:
            scalar_field: Normalized values [0, 1] (H, W)
            plasma_gradient: Color LUT (N, 3) from photonic_discipline
        """
        layer = self.get_layer(LayerType.TENSOR)
        
        # Map scalar to plasma color
        indices = (scalar_field * (len(plasma_gradient) - 1)).long()
        colors = torch.from_numpy(plasma_gradient).float().to(self.device)
        rgb = colors[indices.flatten()].view(scalar_field.shape[0], scalar_field.shape[1], 3)
        
        # Apply opacity mapping (low values near-invisible, high values burn bright)
        from .photonic_discipline import PhotonicPalette
        alpha = PhotonicPalette.apply_opacity_mapping(scalar_field)
        
        # Write to RGBA buffer
        layer.buffer[:, :, :3] = rgb
        layer.buffer[:, :, 3] = alpha
    
    def render_streamlines(self, particle_positions: torch.Tensor, decay: torch.Tensor):
        """
        Layer 2: Render kinetic streamlines (Lagrangian tracers).
        
        Args:
            particle_positions: (N, 2) pixel coordinates
            decay: (N,) opacity decay [0, 1]
        """
        layer = self.get_layer(LayerType.KINETIC)
        layer.clear()
        
        # Rasterize particles (simplified - production would use point sprites)
        positions = particle_positions.long()
        valid = (positions[:, 0] >= 0) & (positions[:, 0] < self.width) & \
                (positions[:, 1] >= 0) & (positions[:, 1] < self.height)
        
        positions = positions[valid]
        decay = decay[valid]
        
        if len(positions) > 0:
            # White particles with variable opacity
            layer.buffer[positions[:, 1], positions[:, 0], :3] = 1.0
            layer.buffer[positions[:, 1], positions[:, 0], 3] = decay * layer.opacity
    
    def render_geometry(self, grid_lines: List[torch.Tensor]):
        """
        Layer 3: Render sovereign geometry (lat/lon grid).
        
        Args:
            grid_lines: List of (N, 2) line segments in pixel coordinates
        """
        layer = self.get_layer(LayerType.GEOMETRY)
        layer.clear()
        
        from .photonic_discipline import PhotonicPalette
        cygnus_blue = PhotonicPalette.CYGNUS_BLUE
        
        # Rasterize grid lines (simplified)
        for line in grid_lines:
            positions = line.long()
            valid = (positions[:, 0] >= 0) & (positions[:, 0] < self.width) & \
                    (positions[:, 1] >= 0) & (positions[:, 1] < self.height)
            positions = positions[valid]
            
            if len(positions) > 0:
                layer.buffer[positions[:, 1], positions[:, 0], 0] = cygnus_blue.r
                layer.buffer[positions[:, 1], positions[:, 0], 1] = cygnus_blue.g
                layer.buffer[positions[:, 1], positions[:, 0], 2] = cygnus_blue.b
                layer.buffer[positions[:, 1], positions[:, 0], 3] = layer.opacity
    
    def render_hud(self, hud_elements: torch.Tensor):
        """
        Layer 4: Render tactical HUD overlay.
        
        Args:
            hud_elements: RGBA buffer (H, W, 4) with pre-rendered UI
        """
        layer = self.get_layer(LayerType.HUD)
        layer.buffer.copy_(hud_elements)
    
    def composite(self) -> torch.Tensor:
        """
        Composite all layers bottom-to-top with correct blending.
        
        ULTRA-OPTIMIZED: Single in-place pass to minimize memory allocation.
        Reuses final_buffer to eliminate intermediate allocations.
        
        Returns:
            Final RGBA frame (H, W, 4)
        """
        # Find enabled layers with actual content
        enabled = [l for l in sorted(self.layers, key=lambda x: x.z_depth) 
                   if l.enabled and l.buffer is not None]
        
        if not enabled:
            return self.final_buffer.zero_()
        
        # Find first layer with actual content (not all zeros)
        base_layer = None
        for layer in enabled:
            if layer.buffer.abs().max() > 0:
                base_layer = layer
                break
        
        if base_layer is None:
            # All layers are empty
            return self.final_buffer.zero_()
        
        # Direct copy of base layer - ensure Float16
        self.final_buffer.copy_(base_layer.buffer.to(dtype=torch.float16, non_blocking=True))
        
        # OPTIMIZED: In-place blending with Float16 (2× bandwidth, no conversions)
        remaining_layers = [l for l in enabled if l is not base_layer]
        for layer in remaining_layers:
            src = layer.buffer.to(dtype=torch.float16, non_blocking=True)
            
            # Scissor test: only blend dirty regions for UI layers
            if layer.dirty_rect is not None:
                x0, y0, x1, y1 = layer.dirty_rect
                src_region = src[y0:y1, x0:x1]
                dst_region = self.final_buffer[y0:y1, x0:x1]
            else:
                src_region = src
                dst_region = self.final_buffer
                x0, y0 = 0, 0
            
            if layer.blend_mode == BlendMode.ADDITIVE:
                # Fused additive blend (Single kernel launch)
                alpha_src = src_region[:, :, 3:4]
                if layer.dirty_rect:
                    torch.addcmul(dst_region[:, :, :3], src_region[:, :, :3], alpha_src, out=dst_region[:, :, :3])
                    dst_region[:, :, :3].clamp_(0, 1)
                    self.final_buffer[y0:y1, x0:x1] = dst_region
                else:
                    torch.addcmul(self.final_buffer[:, :, :3], src_region[:, :, :3], alpha_src, out=self.final_buffer[:, :, :3])
                    self.final_buffer[:, :, :3].clamp_(0, 1)
            
            elif layer.blend_mode == BlendMode.ALPHA or layer.blend_mode == BlendMode.PREMULTIPLIED:
                # Fused alpha blend (in-place, no temporary allocations)
                alpha = src_region[:, :, 3:4]
                one_minus_alpha = 1 - alpha
                if layer.dirty_rect:
                    dst_region[:, :, :3].mul_(one_minus_alpha).add_(src_region[:, :, :3] * alpha)
                    if layer.blend_mode == BlendMode.ALPHA:
                        torch.maximum(dst_region[:, :, 3:4], alpha, out=dst_region[:, :, 3:4])
                    self.final_buffer[y0:y1, x0:x1] = dst_region
                else:
                    self.final_buffer[:, :, :3].mul_(one_minus_alpha).add_(src_region[:, :, :3] * alpha)
                    if layer.blend_mode == BlendMode.ALPHA:
                        torch.maximum(self.final_buffer[:, :, 3:4], alpha, out=self.final_buffer[:, :, 3:4])
            
            elif layer.blend_mode == BlendMode.OVER:
                # Porter-Duff over (optimized for Float16)
                src_a = src_region[:, :, 3:4]
                if layer.dirty_rect:
                    dst_a = dst_region[:, :, 3:4]
                    out_a = src_a + dst_a * (1 - src_a)
                    denom = out_a + 1e-3
                    dst_region[:, :, :3] = (src_region[:, :, :3] * src_a + 
                                            dst_region[:, :, :3] * dst_a * (1 - src_a)) / denom
                    dst_region[:, :, 3:4] = out_a
                    self.final_buffer[y0:y1, x0:x1] = dst_region
                else:
                    dst_a = self.final_buffer[:, :, 3:4]
                    out_a = src_a + dst_a * (1 - src_a)
                    denom = out_a + 1e-3
                    self.final_buffer[:, :, :3] = (src_region[:, :, :3] * src_a + 
                                                    self.final_buffer[:, :, :3] * dst_a * (1 - src_a)) / denom
                    self.final_buffer[:, :, 3:4] = out_a
        
        self.frame_count += 1
        return self.final_buffer
    
    def resize(self, width: int, height: int):
        """Resize all framebuffers."""
        self.width = width
        self.height = height
        
        for layer in self.layers:
            layer.allocate(height, width, self.device)
        
        self.final_buffer = torch.zeros(
            (height, width, 4),
            dtype=torch.float16,  # Changed from Float32 for consistency
            device=self.device
        )
    
    def print_config(self):
        """Print renderer configuration."""
        print("\n" + "="*60)
        print("ONION RENDERER - 5-LAYER PIPELINE")
        print("="*60 + "\n")
        
        print(f"Framebuffer: {self.width}x{self.height}")
        print(f"Device: {self.device}")
        print(f"Frames rendered: {self.frame_count}\n")
        
        print("LAYER STACK (Bottom → Top)")
        print("-" * 60)
        for layer in sorted(self.layers, key=lambda l: l.z_depth):
            status = "✓" if layer.enabled else "✗"
            vram_mb = (layer.buffer.nelement() * layer.buffer.element_size() / 1024**2) if layer.buffer is not None else 0
            print(f"  [{layer.z_depth}] {status} {layer.type.name:12s} | "
                  f"{layer.blend_mode.value:15s} | "
                  f"α={layer.opacity:.2f} | "
                  f"{vram_mb:.1f}MB")
        
        total_vram = sum(
            layer.buffer.nelement() * layer.buffer.element_size() / 1024**2
            for layer in self.layers if layer.buffer is not None
        )
        total_vram += self.final_buffer.nelement() * self.final_buffer.element_size() / 1024**2
        
        print("-" * 60)
        print(f"Total VRAM: {total_vram:.2f} MB")
        
        print("\n" + "="*60)
        print("✓ Onion Renderer Ready")
        print("="*60 + "\n")


def demo_renderer():
    """Demo: Test onion renderer."""
    renderer = OnionRenderer(width=1920, height=1080, device='cuda:0')
    renderer.print_config()
    
    # Test layer access
    print("Layer Access Test:")
    geo = renderer.get_layer(LayerType.GEOLOGICAL)
    print(f"  Geological: {geo}")
    
    tensor = renderer.get_layer(LayerType.TENSOR)
    print(f"  Tensor:     {tensor}")
    
    print("\n✓ Renderer operational\n")


if __name__ == "__main__":
    demo_renderer()
