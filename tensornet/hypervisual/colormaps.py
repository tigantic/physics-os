"""
Colormaps and Transfer Functions
================================

Scientific colormaps for field visualization.
All colormaps designed for perceptual uniformity.

Features:
    - Perceptually uniform colormaps (viridis, plasma, etc.)
    - Transfer functions for volume rendering
    - Custom colormap creation
    - GPU-accelerated colormap application
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, Callable, Union


# =============================================================================
# COLORMAP DATA
# =============================================================================

# Viridis colormap (matplotlib)
_VIRIDIS_DATA = np.array([
    [0.267004, 0.004874, 0.329415],
    [0.282327, 0.140926, 0.457517],
    [0.253935, 0.265254, 0.529983],
    [0.206756, 0.371758, 0.553117],
    [0.163625, 0.471133, 0.558148],
    [0.127568, 0.566949, 0.550556],
    [0.134692, 0.658636, 0.517649],
    [0.266941, 0.748751, 0.440573],
    [0.477504, 0.821444, 0.318195],
    [0.741388, 0.873449, 0.149561],
    [0.993248, 0.906157, 0.143936],
], dtype=np.float32)

# Plasma colormap
_PLASMA_DATA = np.array([
    [0.050383, 0.029803, 0.527975],
    [0.254627, 0.013882, 0.615419],
    [0.417642, 0.000564, 0.658390],
    [0.562738, 0.051545, 0.641509],
    [0.692840, 0.165141, 0.564522],
    [0.798216, 0.280197, 0.469538],
    [0.881443, 0.392529, 0.383229],
    [0.949217, 0.517763, 0.295662],
    [0.988362, 0.652325, 0.211364],
    [0.988648, 0.809579, 0.145357],
    [0.940015, 0.975158, 0.131326],
], dtype=np.float32)

# Inferno colormap
_INFERNO_DATA = np.array([
    [0.001462, 0.000466, 0.013866],
    [0.087411, 0.044556, 0.224813],
    [0.232077, 0.059889, 0.437695],
    [0.390384, 0.100379, 0.501864],
    [0.550287, 0.161158, 0.505719],
    [0.699524, 0.234534, 0.450329],
    [0.826549, 0.329485, 0.353205],
    [0.918858, 0.449241, 0.246137],
    [0.970919, 0.584547, 0.147314],
    [0.988362, 0.731594, 0.076395],
    [0.988648, 0.998364, 0.644924],
], dtype=np.float32)

# Magma colormap
_MAGMA_DATA = np.array([
    [0.001462, 0.000466, 0.013866],
    [0.078815, 0.054184, 0.211667],
    [0.208030, 0.080287, 0.381706],
    [0.346636, 0.086785, 0.489364],
    [0.490262, 0.105033, 0.522599],
    [0.633676, 0.138808, 0.515474],
    [0.769402, 0.197236, 0.475424],
    [0.878168, 0.291320, 0.424143],
    [0.951344, 0.419994, 0.381306],
    [0.988422, 0.574417, 0.436061],
    [0.987053, 0.991438, 0.749504],
], dtype=np.float32)

# Turbo colormap (improved rainbow)
_TURBO_DATA = np.array([
    [0.18995, 0.07176, 0.23217],
    [0.25107, 0.25237, 0.63374],
    [0.15991, 0.49620, 0.86789],
    [0.09672, 0.68706, 0.81389],
    [0.21651, 0.82064, 0.60925],
    [0.42804, 0.89642, 0.35926],
    [0.64715, 0.91959, 0.17356],
    [0.85399, 0.85066, 0.07992],
    [0.96199, 0.68132, 0.05320],
    [0.94879, 0.47827, 0.05653],
    [0.82399, 0.26855, 0.05623],
], dtype=np.float32)

# Coolwarm (diverging)
_COOLWARM_DATA = np.array([
    [0.229739, 0.298717, 0.753683],
    [0.366253, 0.469320, 0.871726],
    [0.520676, 0.631008, 0.952907],
    [0.677423, 0.769534, 0.990859],
    [0.821069, 0.876257, 0.988497],
    [0.927635, 0.928249, 0.917335],
    [0.963298, 0.845163, 0.820021],
    [0.959573, 0.716010, 0.666761],
    [0.916387, 0.561909, 0.489509],
    [0.826793, 0.392344, 0.321941],
    [0.705673, 0.015556, 0.150233],
], dtype=np.float32)

# Jet (legacy rainbow - not perceptually uniform!)
_JET_DATA = np.array([
    [0.0, 0.0, 0.5],
    [0.0, 0.0, 1.0],
    [0.0, 0.5, 1.0],
    [0.0, 1.0, 1.0],
    [0.5, 1.0, 0.5],
    [1.0, 1.0, 0.0],
    [1.0, 0.5, 0.0],
    [1.0, 0.0, 0.0],
    [0.5, 0.0, 0.0],
], dtype=np.float32)

# Grayscale
_GRAYSCALE_DATA = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0],
], dtype=np.float32)


# =============================================================================
# COLORMAP CLASS
# =============================================================================

@dataclass
class ColorMap:
    """
    A colormap for mapping scalar values to colors.
    
    Usage:
        cmap = VIRIDIS
        rgba = apply_colormap(data, cmap)
    """
    name: str
    colors: np.ndarray  # (N, 3) RGB values in [0, 1]
    
    # Range
    vmin: float = 0.0
    vmax: float = 1.0
    
    # Options
    reversed: bool = False
    under_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    over_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    nan_color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    
    def __post_init__(self):
        if self.colors.ndim != 2 or self.colors.shape[1] != 3:
            raise ValueError("colors must be (N, 3) array")
    
    def __call__(self, value: float) -> Tuple[float, float, float]:
        """Map a single value to RGB."""
        if np.isnan(value):
            return self.nan_color
        
        if value < self.vmin:
            return self.under_color
        if value > self.vmax:
            return self.over_color
        
        # Normalize to [0, 1]
        t = (value - self.vmin) / (self.vmax - self.vmin + 1e-10)
        
        if self.reversed:
            t = 1.0 - t
        
        # Interpolate
        return self._interpolate(t)
    
    def _interpolate(self, t: float) -> Tuple[float, float, float]:
        """Interpolate color at position t in [0, 1]."""
        n = len(self.colors)
        t = np.clip(t, 0, 1)
        
        # Find segment
        idx = t * (n - 1)
        i0 = int(idx)
        i1 = min(i0 + 1, n - 1)
        frac = idx - i0
        
        # Linear interpolation
        c0 = self.colors[i0]
        c1 = self.colors[i1]
        c = c0 + frac * (c1 - c0)
        
        return tuple(c)
    
    def reverse(self) -> 'ColorMap':
        """Return reversed colormap."""
        return ColorMap(
            name=f"{self.name}_r",
            colors=self.colors[::-1].copy(),
            vmin=self.vmin,
            vmax=self.vmax,
        )
    
    def rescale(self, vmin: float, vmax: float) -> 'ColorMap':
        """Return colormap with new range."""
        return ColorMap(
            name=self.name,
            colors=self.colors.copy(),
            vmin=vmin,
            vmax=vmax,
        )


# =============================================================================
# PREDEFINED COLORMAPS
# =============================================================================

VIRIDIS = ColorMap(name='viridis', colors=_VIRIDIS_DATA)
PLASMA = ColorMap(name='plasma', colors=_PLASMA_DATA)
INFERNO = ColorMap(name='inferno', colors=_INFERNO_DATA)
MAGMA = ColorMap(name='magma', colors=_MAGMA_DATA)
TURBO = ColorMap(name='turbo', colors=_TURBO_DATA)
COOLWARM = ColorMap(name='coolwarm', colors=_COOLWARM_DATA)
JET = ColorMap(name='jet', colors=_JET_DATA)  # Not recommended
GRAYSCALE = ColorMap(name='grayscale', colors=_GRAYSCALE_DATA)


# =============================================================================
# TRANSFER FUNCTION
# =============================================================================

@dataclass
class ControlPoint:
    """Control point for transfer function."""
    position: float  # Value position [0, 1]
    color: Tuple[float, float, float]  # RGB
    opacity: float  # Alpha


class TransferFunction:
    """
    Transfer function for volume rendering.
    
    Maps scalar values to RGBA (color + opacity).
    
    Usage:
        tf = TransferFunction()
        tf.add_point(0.0, (0, 0, 0), 0.0)
        tf.add_point(0.5, (1, 0, 0), 0.5)
        tf.add_point(1.0, (1, 1, 0), 1.0)
        
        rgba = tf.apply(data)
    """
    
    def __init__(self, colormap: Optional[ColorMap] = None):
        self.points: List[ControlPoint] = []
        self.colormap = colormap or VIRIDIS
        
        # Default: linear opacity ramp with colormap
        self.add_point(0.0, self.colormap(0.0), 0.0)
        self.add_point(1.0, self.colormap(1.0), 1.0)
    
    def add_point(
        self,
        position: float,
        color: Tuple[float, float, float],
        opacity: float,
    ):
        """Add a control point."""
        self.points.append(ControlPoint(position, color, opacity))
        self.points.sort(key=lambda p: p.position)
    
    def clear(self):
        """Clear all control points."""
        self.points = []
    
    def __call__(self, value: float) -> Tuple[float, float, float, float]:
        """Map value to RGBA."""
        if not self.points:
            return (0, 0, 0, 0)
        
        if value <= self.points[0].position:
            p = self.points[0]
            return (*p.color, p.opacity)
        
        if value >= self.points[-1].position:
            p = self.points[-1]
            return (*p.color, p.opacity)
        
        # Find segment
        for i in range(len(self.points) - 1):
            p0 = self.points[i]
            p1 = self.points[i + 1]
            
            if p0.position <= value <= p1.position:
                t = (value - p0.position) / (p1.position - p0.position + 1e-10)
                
                r = p0.color[0] + t * (p1.color[0] - p0.color[0])
                g = p0.color[1] + t * (p1.color[1] - p0.color[1])
                b = p0.color[2] + t * (p1.color[2] - p0.color[2])
                a = p0.opacity + t * (p1.opacity - p0.opacity)
                
                return (r, g, b, a)
        
        return (0, 0, 0, 0)
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply transfer function to array."""
        shape = data.shape
        flat = data.flatten()
        
        result = np.zeros((len(flat), 4), dtype=np.float32)
        for i, v in enumerate(flat):
            result[i] = self(v)
        
        return result.reshape(*shape, 4)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def apply_colormap(
    data: np.ndarray,
    colormap: ColorMap = VIRIDIS,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """
    Apply colormap to 2D scalar data.
    
    Args:
        data: 2D scalar array
        colormap: ColorMap to use
        vmin, vmax: Data range (auto if None)
        
    Returns:
        RGBA image (H, W, 4) as uint8
    """
    # Auto range
    if vmin is None:
        vmin = float(np.nanmin(data))
    if vmax is None:
        vmax = float(np.nanmax(data))
    
    # Normalize to [0, 1]
    if vmax - vmin < 1e-10:
        normalized = np.zeros_like(data)
    else:
        normalized = (data - vmin) / (vmax - vmin)
    
    normalized = np.clip(normalized, 0, 1)
    
    # Build LUT
    n_colors = len(colormap.colors)
    lut = np.zeros((256, 4), dtype=np.uint8)
    
    for i in range(256):
        t = i / 255.0
        idx = t * (n_colors - 1)
        i0 = int(idx)
        i1 = min(i0 + 1, n_colors - 1)
        frac = idx - i0
        
        c0 = colormap.colors[i0]
        c1 = colormap.colors[i1]
        c = c0 + frac * (c1 - c0)
        
        lut[i, :3] = (c * 255).astype(np.uint8)
        lut[i, 3] = 255
    
    # Apply LUT
    indices = (normalized * 255).astype(np.uint8)
    rgba = lut[indices]
    
    return rgba


def create_colormap(
    name: str,
    colors: List[Tuple[float, float, float]],
) -> ColorMap:
    """
    Create a custom colormap from a list of colors.
    
    Args:
        name: Colormap name
        colors: List of RGB tuples in [0, 1]
        
    Returns:
        ColorMap instance
    """
    return ColorMap(
        name=name,
        colors=np.array(colors, dtype=np.float32),
    )


def diverging_colormap(
    name: str,
    low: Tuple[float, float, float],
    mid: Tuple[float, float, float],
    high: Tuple[float, float, float],
    n_colors: int = 11,
) -> ColorMap:
    """
    Create a diverging colormap.
    
    Args:
        name: Colormap name
        low: Color for minimum value
        mid: Color for middle value
        high: Color for maximum value
        n_colors: Number of colors
        
    Returns:
        ColorMap instance
    """
    colors = []
    half = n_colors // 2
    
    # Low to mid
    for i in range(half):
        t = i / half
        c = tuple(low[j] + t * (mid[j] - low[j]) for j in range(3))
        colors.append(c)
    
    # Mid
    colors.append(mid)
    
    # Mid to high
    for i in range(half):
        t = (i + 1) / half
        c = tuple(mid[j] + t * (high[j] - mid[j]) for j in range(3))
        colors.append(c)
    
    return ColorMap(
        name=name,
        colors=np.array(colors, dtype=np.float32),
    )
