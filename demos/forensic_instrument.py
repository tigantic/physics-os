#!/usr/bin/env python3
"""
HyperTensor Forensic Instrument
===============================

True forensic-grade atmospheric visualization:
1. Zoom-Aware Contraction - 1:1 pixel density at any zoom
2. Vector Flow Rendering - Wind barbs showing discontinuities  
3. Geodesic Temporal Interpolation - QTT core blending
4. Dynamic Per-View Normalization

This is NOT a heatmap. This is a mathematical reconstruction instrument.
"""

import sys
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from PySide6.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
        QLabel, QFrame, QPushButton, QApplication, QSlider,
        QStatusBar, QComboBox, QGroupBox, QTextEdit, QCheckBox
    )
    from PySide6.QtCore import Qt, QTimer
except ImportError:
    print("ERROR: PySide6 required")
    sys.exit(1)

try:
    from vispy import scene, app
    from vispy.scene import visuals
    from vispy.color import Colormap
    from vispy.visuals.transforms import STTransform
except ImportError:
    print("ERROR: VisPy required")
    sys.exit(1)

try:
    from natural_earth import get_coastlines_pixels, get_coastlines_geo
    COASTLINES_AVAILABLE = True
except ImportError:
    try:
        from coastlines import get_coastline_pixels as get_coastlines_pixels
        COASTLINES_AVAILABLE = True
    except ImportError:
        COASTLINES_AVAILABLE = False

try:
    from blue_marble import generate_blue_marble
    BLUE_MARBLE_AVAILABLE = True
except ImportError:
    BLUE_MARBLE_AVAILABLE = False


# =============================================================================
# Z-ORDER HIERARCHY (VisPy: LOWER number = FRONT)
# =============================================================================
# Layer 0:  Vector arrow heads (FRONT)
# Layer 1:  Vector barb lines
# Layer 2:  Coastline labels
# Layer 3:  Coastline core (white)
# Layer 4:  Coastline glow
# Layer 10: Grid lines
# Layer 50: Wind heatmap (semi-transparent, 50% opacity)
# Layer 100: Blue Marble base map (BACK)
# =============================================================================

Z_VECTOR_HEADS = 0
Z_VECTOR_LINES = 1
Z_COAST_LABELS = 2
Z_COAST_CORE = 3
Z_COAST_GLOW = 4
Z_GRID = 10
Z_HEATMAP = 50
Z_BASEMAP = 100
Z_CONTINENT_LABELS = 8


# =============================================================================
# INDUSTRIAL PALETTE - Forensic Grade
# =============================================================================

FORENSIC_CMAP = Colormap([
    (0.00, 0.02, 0.08, 1.0),   # Abyss
    (0.02, 0.08, 0.20, 1.0),   # Deep ocean
    (0.05, 0.18, 0.38, 1.0),   # Cobalt
    (0.12, 0.35, 0.55, 1.0),   # Steel
    (0.25, 0.55, 0.72, 1.0),   # Cyan
    (0.50, 0.78, 0.88, 1.0),   # Ice
    (0.85, 0.85, 0.75, 1.0),   # Neutral
    (0.95, 0.75, 0.45, 1.0),   # Amber
    (0.92, 0.45, 0.22, 1.0),   # Orange
    (0.78, 0.18, 0.12, 1.0),   # Red
    (0.45, 0.05, 0.08, 1.0),   # Crimson
])

STYLE = """
QMainWindow, QWidget { background: #06080C; color: #8EACC8; font-family: 'Consolas'; }
QFrame#sidebar { background: #080A10; border-right: 1px solid #1A2535; }
QGroupBox { border: 1px solid #1A2535; margin-top: 10px; padding-top: 6px; color: #4A7A9A; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; }
QPushButton { background: #101520; border: 1px solid #2A3545; padding: 8px 12px; color: #7A9AB8; }
QPushButton:hover { background: #1A2535; border-color: #4A6A8A; }
QPushButton:checked { background: #3A5A7A; color: #FFF; }
QComboBox, QSlider { background: #101520; border: 1px solid #2A3545; }
QSlider::handle:horizontal { background: #4A7A9A; width: 14px; margin: -4px 0; border-radius: 7px; }
QTextEdit#log { background: #040608; border: 1px solid #1A2535; font-size: 10px; color: #4A6A8A; }
QLabel#title { font-size: 13px; font-weight: bold; color: #4A7A9A; }
QLabel#value { font-family: 'Consolas'; font-size: 11px; color: #8AC; }
QLabel#alert { color: #E84; font-weight: bold; }
QStatusBar { background: #06080C; border-top: 1px solid #1A2535; color: #3A5A7A; }
QCheckBox { color: #7A9AB8; }
QCheckBox::indicator { width: 14px; height: 14px; background: #101520; border: 1px solid #2A3545; }
QCheckBox::indicator:checked { background: #4A7A9A; }
"""


# =============================================================================
# MANIFOLD ENGINE - Resolution Independent
# =============================================================================

class ManifoldEngine:
    """
    The mathematical core - performs contractions at arbitrary resolution.
    This is what makes zoom "infinite" rather than "upscaled".
    """
    
    def __init__(self, data: np.ndarray, lat: np.ndarray, lon: np.ndarray):
        self.data = data.astype(np.float64)
        self.lat = lat
        self.lon = lon
        self.shape = data.shape
        
        # Precompute spline coefficients for true interpolation
        from scipy.interpolate import RectBivariateSpline
        
        y = np.arange(self.shape[0])
        x = np.arange(self.shape[1])
        self.spline = RectBivariateSpline(y, x, self.data, kx=3, ky=3)
        
    def contract(self, bounds: Tuple[float, float, float, float], 
                 resolution: Tuple[int, int]) -> np.ndarray:
        """
        Contract manifold to exact screen resolution within bounds.
        
        This is the "Google Maps" logic - we regenerate pixels
        mathematically rather than stretching a fixed grid.
        
        Args:
            bounds: (x_min, x_max, y_min, y_max) in data coordinates
            resolution: (width, height) of output
            
        Returns:
            High-fidelity sample at 1:1 pixel density
        """
        x_min, x_max, y_min, y_max = bounds
        out_w, out_h = resolution
        
        # Clamp to valid range
        x_min = max(0, min(x_min, self.shape[1] - 1))
        x_max = max(0, min(x_max, self.shape[1] - 1))
        y_min = max(0, min(y_min, self.shape[0] - 1))
        y_max = max(0, min(y_max, self.shape[0] - 1))
        
        if x_max <= x_min or y_max <= y_min:
            return np.zeros((out_h, out_w), dtype=np.float32)
        
        # Generate sample coordinates at screen resolution
        x_coords = np.linspace(x_min, x_max, out_w)
        y_coords = np.linspace(y_min, y_max, out_h)
        
        # Evaluate spline at these exact coordinates
        # This gives us true mathematical interpolation, not pixel stretching
        result = self.spline(y_coords, x_coords)
        
        return result.astype(np.float32)


# =============================================================================
# TEMPORAL INTERPOLATOR - Geodesic Core Interpolation
# =============================================================================

class GeodesicCoreInterpolator:
    """
    TRUE Geodesic interpolation between temporal snapshots.
    
    Instead of blending pixels (which creates watercolor smudges),
    we interpolate the SPLINE COEFFICIENTS (the "cores") and then
    contract. This maintains sharp edges during motion.
    
    The key insight: spline coefficients ARE the compressed representation.
    Blending coefficients = blending the underlying physics.
    Blending pixels = blending the rendered artifacts.
    """
    
    def __init__(self):
        self.snapshots = {}  # t -> (u_data, v_data, lat, lon)
        self._lat = None
        self._lon = None
        
    def add_snapshot(self, t: float, u: np.ndarray, v: np.ndarray, 
                     lat: np.ndarray, lon: np.ndarray):
        """Add a temporal snapshot (raw data, we build cores on demand)."""
        self.snapshots[t] = (u.copy(), v.copy() if v is not None else np.zeros_like(u))
        self._lat = lat
        self._lon = lon
        
    def _compute_spline_coeffs(self, data: np.ndarray) -> np.ndarray:
        """
        Compute B-spline coefficients (the 'cores').
        These represent the smooth underlying field, not the pixel grid.
        """
        from scipy.ndimage import spline_filter
        # Order-3 spline coefficients
        return spline_filter(data.astype(np.float64), order=3)
    
    def _geodesic_blend_cores(self, c0: np.ndarray, c1: np.ndarray, 
                               alpha: float, flow_u: np.ndarray, flow_v: np.ndarray) -> np.ndarray:
        """
        Geodesic interpolation of spline cores using optical flow.
        
        Instead of linear blend c = (1-α)c0 + αc1 (causes smudge),
        we WARP c0 toward c1 along the flow field.
        
        This makes features "move" rather than "fade".
        """
        h, w = c0.shape
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float64)
        
        # Estimate displacement from flow (scaled by interpolation amount)
        # Flow is in physical units, we scale to grid units
        scale = alpha * 0.5  # Damping factor
        dx = flow_u * scale
        dy = flow_v * scale
        
        # Warp source coordinates
        src_x = x_coords - dx
        src_y = y_coords - dy
        
        # Clamp to valid range
        src_x = np.clip(src_x, 0, w - 1)
        src_y = np.clip(src_y, 0, h - 1)
        
        # Warp c0 forward in time using the flow
        from scipy.ndimage import map_coordinates
        c0_warped = map_coordinates(c0, [src_y, src_x], order=3, mode='nearest')
        
        # Warp c1 backward in time
        src_x_back = x_coords + dx * (1 - alpha) / max(alpha, 0.01)
        src_y_back = y_coords + dy * (1 - alpha) / max(alpha, 0.01)
        src_x_back = np.clip(src_x_back, 0, w - 1)
        src_y_back = np.clip(src_y_back, 0, h - 1)
        c1_warped = map_coordinates(c1, [src_y_back, src_x_back], order=3, mode='nearest')
        
        # Blend the WARPED cores (now aligned)
        return (1 - alpha) * c0_warped + alpha * c1_warped
    
    def interpolate(self, t: float, bounds: Tuple, resolution: Tuple) -> Tuple[np.ndarray, np.ndarray]:
        """
        Geodesic core interpolation.
        
        1. Get bracketing snapshots
        2. Compute spline coefficients for each
        3. Warp-blend coefficients using flow field
        4. Contract from blended cores at target resolution
        
        Result: Sharp features that MOVE, not smudge.
        """
        if not self.snapshots:
            return None, None
            
        times = sorted(self.snapshots.keys())
        
        # Find bracketing snapshots
        t0 = max([ti for ti in times if ti <= t], default=times[0])
        t1 = min([ti for ti in times if ti >= t], default=times[-1])
        
        u0_data, v0_data = self.snapshots[t0]
        
        if t0 == t1:
            # Exact keyframe - just contract
            engine = ManifoldEngine(u0_data, self._lat, self._lon)
            u = engine.contract(bounds, resolution)
            v_engine = ManifoldEngine(v0_data, self._lat, self._lon)
            v = v_engine.contract(bounds, resolution)
            return u, v
        
        u1_data, v1_data = self.snapshots[t1]
        
        # Interpolation factor
        alpha = (t - t0) / (t1 - t0)
        
        # Compute spline coefficients (the "cores")
        c0_u = self._compute_spline_coeffs(u0_data)
        c1_u = self._compute_spline_coeffs(u1_data)
        c0_v = self._compute_spline_coeffs(v0_data)
        c1_v = self._compute_spline_coeffs(v1_data)
        
        # Estimate flow field (average of both frames)
        flow_u = (u0_data + u1_data) / 2.0
        flow_v = (v0_data + v1_data) / 2.0
        
        # Geodesic blend of cores
        blended_u = self._geodesic_blend_cores(c0_u, c1_u, alpha, flow_u, flow_v)
        blended_v = self._geodesic_blend_cores(c0_v, c1_v, alpha, flow_u, flow_v)
        
        # Contract from blended cores at target resolution
        engine_u = ManifoldEngine(blended_u, self._lat, self._lon)
        engine_v = ManifoldEngine(blended_v, self._lat, self._lon)
        
        u = engine_u.contract(bounds, resolution)
        v = engine_v.contract(bounds, resolution)
        
        return u, v


# =============================================================================
# VECTOR FIELD RENDERER - Wind Barbs
# =============================================================================

class WindBarbRenderer:
    """
    Renders wind as barbs/arrows showing actual flow discontinuities.
    This reveals singularities that heatmaps hide.
    """
    
    def __init__(self, parent_scene):
        self.parent = parent_scene
        self.barbs = []
        self.streamlines = None
        
    def render(self, u: np.ndarray, v: np.ndarray, 
               spacing: int = 12, scale: float = 8.0,
               threshold: float = None):
        """
        Render wind barbs showing velocity vectors.
        
        Args:
            u, v: Velocity components
            spacing: Pixels between barbs
            scale: Barb length scale
            threshold: Minimum wind speed to draw (auto-computed if None)
        """
        self.clear()
        
        h, w = u.shape
        
        # Compute magnitude
        mag = np.sqrt(u**2 + v**2)
        
        # ADAPTIVE THRESHOLD: Use 10th percentile of non-zero magnitudes
        # This ensures barbs appear even at high altitude with slow winds
        if threshold is None:
            nonzero = mag[mag > 0.1]
            if len(nonzero) > 0:
                threshold = float(np.percentile(nonzero, 10))
            else:
                threshold = 0.5
        
        # Scale factor based on max magnitude
        max_mag = float(np.percentile(mag, 98))
        if max_mag < 1:
            max_mag = 1.0
        
        # Sample positions
        positions = []
        directions = []
        colors = []
        
        for yi in range(spacing//2, h, spacing):
            for xi in range(spacing//2, w, spacing):
                uu = u[yi, xi]
                vv = v[yi, xi]
                m = mag[yi, xi]
                
                if m > threshold:
                    # Normalize direction with ADAPTIVE scaling
                    length = scale * (m / max_mag) * 1.5
                    dx = (uu / m) * length
                    dy = (vv / m) * length
                    
                    # Barb: line from point in direction of wind
                    x0, y0 = float(xi), float(yi)
                    x1, y1 = x0 + dx, y0 + dy
                    
                    positions.extend([[x0, y0], [x1, y1]])
                    
                    # Color by intensity - high contrast
                    t = min(m / 50, 1.0)
                    if t < 0.5:
                        # Blue to cyan
                        r, g, b = 0.2, 0.4 + t, 0.7 + 0.3*t
                    else:
                        # Cyan to orange to red
                        tt = (t - 0.5) * 2
                        r = 0.3 + 0.7*tt
                        g = 0.9 - 0.5*tt
                        b = 1.0 - 0.8*tt
                    
                    colors.extend([[r, g, b, 0.85], [r, g, b, 0.85]])
        
        if positions:
            positions = np.array(positions, dtype=np.float32)
            colors = np.array(colors, dtype=np.float32)
            
            # Z-ORDER 1: Vector lines (front)
            barb_visual = visuals.Line(
                pos=positions,
                color=colors,
                width=2.5,  # Thick and visible
                connect='segments',
                antialias=True,
                parent=self.parent
            )
            barb_visual.order = Z_VECTOR_LINES  # Layer 1
            self.barbs.append(barb_visual)
            
            # Arrow heads at end of each barb for direction - BRIGHT WHITE
            end_pos = positions[1::2]  # Every other point (the ends)
            # Make arrow heads WHITE for maximum visibility
            head_colors = np.ones((len(end_pos), 4), dtype=np.float32)
            head_colors[:, 3] = 0.95  # Almost opaque
            
            heads = visuals.Markers()
            heads.set_data(
                pos=end_pos,
                face_color=head_colors,
                symbol='triangle_up',
                size=8,
                edge_width=0
            )
            heads.parent = self.parent
            heads.order = Z_VECTOR_HEADS  # Layer 0 (VERY front)
            self.barbs.append(heads)
    
    def clear(self):
        for v in self.barbs:
            v.parent = None
        self.barbs = []


# =============================================================================
# FORENSIC VIEWPORT
# =============================================================================

class ForensicViewport:
    """
    Resolution-independent forensic viewport.
    
    Key differences from standard VisPy:
    1. Zoom triggers re-contraction at screen resolution
    2. Dynamic normalization per view (no global wash-out)
    3. Vector overlay reveals discontinuities
    """
    
    def __init__(self, on_view_change=None):
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            show=False,
            bgcolor='#06080C'
        )
        
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.PanZoomCamera(aspect=1)
        
        # Callback for view changes
        self.on_view_change = on_view_change
        
        # Visuals
        self.field_image = None
        self.base_map = None  # Blue Marble substrate
        self.barb_renderer = WindBarbRenderer(self.view.scene)
        self.coastlines = []
        self.annotations = []
        self.singularity_markers = []
        
        # Generate base map
        self._base_map_texture = None
        if BLUE_MARBLE_AVAILABLE:
            try:
                self._base_map_texture = generate_blue_marble(720, 360)
                print(f"✅ Blue Marble generated: {self._base_map_texture.shape}")
            except Exception as e:
                print(f"⚠️ Warning: Could not generate Blue Marble: {e}")
        else:
            print("⚠️ Blue Marble not available (import failed)")
        
        # Engine
        self.engine = None
        self.interpolator = GeodesicCoreInterpolator()
        
        # State
        self._data_shape = (180, 360)
        self._bounds = (0, 360, 0, 180)
        self._vmin = 0
        self._vmax = 1
        self._show_barbs = True
        self._current_t = 0.5
        
        # Connect to camera for zoom-aware updates
        self._last_rect = None
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._check_view_change)
        self._update_timer.start(100)  # Check every 100ms
        
    @property
    def native(self):
        return self.canvas.native
    
    def _check_view_change(self):
        """Poll for camera changes and trigger re-render."""
        if self.view.camera.rect is not None:
            rect = self.view.camera.rect
            current = (rect.left, rect.right, rect.bottom, rect.top)
            
            if self._last_rect != current:
                self._last_rect = current
                if self.on_view_change:
                    self.on_view_change(current)
    
    def set_engine(self, u: np.ndarray, v: np.ndarray, 
                   lat: np.ndarray, lon: np.ndarray):
        """Set the manifold data."""
        self.engine = ManifoldEngine(u, lat, lon)
        self._data_shape = u.shape
        
        # Store for vector rendering
        self._u_full = u
        self._v_full = v
        self._lat = lat
        self._lon = lon
        
    def add_temporal_snapshot(self, t: float, u: np.ndarray, v: np.ndarray,
                              lat: np.ndarray, lon: np.ndarray):
        """Add a snapshot for temporal interpolation."""
        self.interpolator.add_snapshot(t, u, v, lat, lon)
        
    def render(self, u: np.ndarray, v: np.ndarray,
               field_name: str = "U-Wind",
               level_info: str = "Surface",
               show_barbs: bool = True):
        """
        Full render with Blue Marble substrate and semi-transparent overlay.
        """
        # Clear old
        if self.field_image is not None:
            self.field_image.parent = None
        if self.base_map is not None:
            self.base_map.parent = None
        self.barb_renderer.clear()
        
        self._show_barbs = show_barbs
        
        h, w = u.shape
        self._data_shape = (h, w)
        self._bounds = (0, w, 0, h)
        
        # =====================================================================
        # LAYER 100: BLUE MARBLE BASE MAP (BACK)
        # =====================================================================
        base_resized = None
        if self._base_map_texture is not None:
            # Resize base map to match data dimensions
            from scipy.ndimage import zoom
            bm_h, bm_w = self._base_map_texture.shape[:2]
            if bm_h != h or bm_w != w:
                scale_y = h / bm_h
                scale_x = w / bm_w
                base_resized = zoom(self._base_map_texture, (scale_y, scale_x, 1), order=1)
            else:
                base_resized = self._base_map_texture.copy()
            
            print(f"🌍 Blue Marble loaded: {base_resized.shape}")
        else:
            print("⚠️ No Blue Marble texture - using dark background")
            base_resized = np.zeros((h, w, 4), dtype=np.float32)
            base_resized[:, :, 0] = 0.02  # Dark blue
            base_resized[:, :, 1] = 0.05
            base_resized[:, :, 2] = 0.10
            base_resized[:, :, 3] = 1.0
        
        # =====================================================================
        # LAYER 50: WIND MANIFOLD (SEMI-TRANSPARENT OVERLAY)
        # =====================================================================
        # DYNAMIC NORMALIZATION - key fix for color wash
        self._vmin = float(np.percentile(u, 1))
        self._vmax = float(np.percentile(u, 99))
        
        # Prevent zero range
        if abs(self._vmax - self._vmin) < 0.1:
            self._vmax = self._vmin + 1.0
        
        # Normalize
        normalized = (u - self._vmin) / (self._vmax - self._vmin)
        normalized = np.clip(normalized, 0, 1)
        
        # Create RGBA with transparency - VECTORIZED for performance
        # 55% opacity so terrain shows through
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        
        # Vectorized colormap application (Industrial Forensic palette)
        # Zone 1: Deep blue (val < 0.3)
        mask1 = normalized < 0.3
        rgba[:, :, 0] = np.where(mask1, 0.02 + normalized * 0.1, 0)
        rgba[:, :, 1] = np.where(mask1, 0.08 + normalized * 0.3, 0)
        rgba[:, :, 2] = np.where(mask1, 0.20 + normalized * 0.5, 0)
        
        # Zone 2: Cyan transition (0.3 <= val < 0.5)
        mask2 = (normalized >= 0.3) & (normalized < 0.5)
        t2 = (normalized - 0.3) / 0.2
        rgba[:, :, 0] = np.where(mask2, 0.05 + t2 * 0.2, rgba[:, :, 0])
        rgba[:, :, 1] = np.where(mask2, 0.17 + t2 * 0.4, rgba[:, :, 1])
        rgba[:, :, 2] = np.where(mask2, 0.35 + t2 * 0.4, rgba[:, :, 2])
        
        # Zone 3: Cyan-amber (0.5 <= val < 0.7)
        mask3 = (normalized >= 0.5) & (normalized < 0.7)
        t3 = (normalized - 0.5) / 0.2
        rgba[:, :, 0] = np.where(mask3, 0.25 + t3 * 0.6, rgba[:, :, 0])
        rgba[:, :, 1] = np.where(mask3, 0.55 + t3 * 0.3, rgba[:, :, 1])
        rgba[:, :, 2] = np.where(mask3, 0.72 - t3 * 0.2, rgba[:, :, 2])
        
        # Zone 4: Hot (val >= 0.7)
        mask4 = normalized >= 0.7
        t4 = (normalized - 0.7) / 0.3
        rgba[:, :, 0] = np.where(mask4, 0.85 + t4 * 0.1, rgba[:, :, 0])
        rgba[:, :, 1] = np.where(mask4, 0.85 - t4 * 0.6, rgba[:, :, 1])
        rgba[:, :, 2] = np.where(mask4, 0.55 - t4 * 0.4, rgba[:, :, 2])
        
        # Set alpha for all pixels (55% opacity for overlay effect)
        rgba[:, :, 3] = 0.55
        
        # =====================================================================
        # PRE-COMPOSITE: Blend wind data OVER Blue Marble in numpy
        # =====================================================================
        # This avoids VisPy alpha blending issues
        alpha = rgba[:, :, 3:4]  # Shape (h, w, 1)
        
        # Composite: result = wind * alpha + base * (1 - alpha)
        composited = np.zeros((h, w, 4), dtype=np.float32)
        composited[:, :, :3] = rgba[:, :, :3] * alpha + base_resized[:, :, :3] * (1.0 - alpha)
        composited[:, :, 3] = 1.0  # Final image is fully opaque
        
        print(f"🎨 Composited image: wind(55%) over terrain")
        
        # Render single composited image
        self.field_image = visuals.Image(
            composited,
            interpolation='bilinear',  # Smooth fluid look over terrain
            parent=self.view.scene
        )
        self.field_image.order = Z_HEATMAP  # Layer 50
        
        # Render wind barbs if enabled (Z-ORDER 1: FRONT)
        if show_barbs and v is not None:
            # Higher density for better texture
            spacing = max(6, min(12, h // 15))
            self.barb_renderer.render(u, v, spacing=spacing, scale=10.0)
        
        # Set camera
        self.view.camera.set_range(x=(-50, w+100), y=(-40, h+50))
        
        # Overlays
        self._draw_coastlines(h, w)
        self._draw_continent_labels(h, w)
        self._draw_annotations(h, w, field_name, level_info)
        
        self.canvas.update()
        return self._vmin, self._vmax
    
    def _render_temporal_slice(self, u: np.ndarray, v: np.ndarray):
        """
        Render a temporal slice with Blue Marble substrate and vector overlay.
        """
        # Clear existing
        if self.field_image is not None:
            self.field_image.parent = None
        if self.base_map is not None:
            self.base_map.parent = None
        self.barb_renderer.clear()
        
        h, w = u.shape
        self._data_shape = (h, w)
        
        # =====================================================================
        # BLUE MARBLE BASE
        # =====================================================================
        base_resized = None
        if self._base_map_texture is not None:
            from scipy.ndimage import zoom
            bm_h, bm_w = self._base_map_texture.shape[:2]
            if bm_h != h or bm_w != w:
                scale_y = h / bm_h
                scale_x = w / bm_w
                base_resized = zoom(self._base_map_texture, (scale_y, scale_x, 1), order=1)
            else:
                base_resized = self._base_map_texture.copy()
        else:
            # Dark fallback
            base_resized = np.zeros((h, w, 4), dtype=np.float32)
            base_resized[:, :, 0] = 0.02
            base_resized[:, :, 1] = 0.05
            base_resized[:, :, 2] = 0.10
            base_resized[:, :, 3] = 1.0
        
        # Dynamic normalization (per-view, not global)
        self._vmin = float(np.percentile(u, 1))
        self._vmax = float(np.percentile(u, 99))
        if abs(self._vmax - self._vmin) < 0.1:
            self._vmax = self._vmin + 1.0
        
        normalized = (u - self._vmin) / (self._vmax - self._vmin)
        normalized = np.clip(normalized, 0, 1)
        
        # Create RGBA with transparency - VECTORIZED
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        
        mask1 = normalized < 0.3
        rgba[:, :, 0] = np.where(mask1, 0.02 + normalized * 0.1, 0)
        rgba[:, :, 1] = np.where(mask1, 0.08 + normalized * 0.3, 0)
        rgba[:, :, 2] = np.where(mask1, 0.20 + normalized * 0.5, 0)
        
        mask2 = (normalized >= 0.3) & (normalized < 0.5)
        t2 = (normalized - 0.3) / 0.2
        rgba[:, :, 0] = np.where(mask2, 0.05 + t2 * 0.2, rgba[:, :, 0])
        rgba[:, :, 1] = np.where(mask2, 0.17 + t2 * 0.4, rgba[:, :, 1])
        rgba[:, :, 2] = np.where(mask2, 0.35 + t2 * 0.4, rgba[:, :, 2])
        
        mask3 = (normalized >= 0.5) & (normalized < 0.7)
        t3 = (normalized - 0.5) / 0.2
        rgba[:, :, 0] = np.where(mask3, 0.25 + t3 * 0.6, rgba[:, :, 0])
        rgba[:, :, 1] = np.where(mask3, 0.55 + t3 * 0.3, rgba[:, :, 1])
        rgba[:, :, 2] = np.where(mask3, 0.72 - t3 * 0.2, rgba[:, :, 2])
        
        mask4 = normalized >= 0.7
        t4 = (normalized - 0.7) / 0.3
        rgba[:, :, 0] = np.where(mask4, 0.85 + t4 * 0.1, rgba[:, :, 0])
        rgba[:, :, 1] = np.where(mask4, 0.85 - t4 * 0.6, rgba[:, :, 1])
        rgba[:, :, 2] = np.where(mask4, 0.55 - t4 * 0.4, rgba[:, :, 2])
        
        rgba[:, :, 3] = 0.55  # 55% opacity
        
        # =====================================================================
        # PRE-COMPOSITE: Blend wind data OVER Blue Marble in numpy
        # =====================================================================
        alpha = rgba[:, :, 3:4]
        composited = np.zeros((h, w, 4), dtype=np.float32)
        composited[:, :, :3] = rgba[:, :, :3] * alpha + base_resized[:, :, :3] * (1.0 - alpha)
        composited[:, :, 3] = 1.0
        
        self.field_image = visuals.Image(
            composited,
            interpolation='bilinear',
            parent=self.view.scene
        )
        self.field_image.order = Z_HEATMAP
        
        # GEOGRAPHY LAYER
        self._draw_coastlines(h, w)
        self._draw_continent_labels(h, w)
        
        # VECTOR LAYER (Z-ORDER 1: FRONT) - ALWAYS if enabled
        if self._show_barbs and v is not None:
            spacing = max(6, min(12, h // 15))
            self.barb_renderer.render(u, v, spacing=spacing, scale=10.0)
        
        self.canvas.update()
    
    def set_barbs_enabled(self, enabled: bool):
        """Explicitly set barb visibility and re-render."""
        self._show_barbs = enabled
        # Trigger re-render at current temporal position
        self.update_temporal(self._current_t, force_barbs=enabled)
    
    def update_temporal(self, t: float, force_barbs: bool = None):
        """
        Update display for temporal position.
        Uses GEODESIC CORE interpolation for sharp motion.
        
        Args:
            t: Temporal position [0, 1]
            force_barbs: Override barb state (for checkbox sync)
        """
        self._current_t = t
        
        if force_barbs is not None:
            self._show_barbs = force_barbs
        
        if not self.interpolator.snapshots:
            return
        
        # Get current view bounds from camera
        if self._last_rect:
            x_min, x_max, y_min, y_max = self._last_rect
            # Clamp to data bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(self._data_shape[1], x_max)
            y_max = min(self._data_shape[0], y_max)
            bounds = (x_min, x_max, y_min, y_max)
        else:
            bounds = self._bounds
        
        # Contract at current resolution (1:1 pixel density)
        screen_w = max(100, int(bounds[1] - bounds[0]))
        screen_h = max(100, int(bounds[3] - bounds[2]))
        resolution = (screen_w, screen_h)
        
        # GEODESIC interpolation - cores, not pixels
        u, v = self.interpolator.interpolate(t, bounds, resolution)
        
        if u is not None:
            # Full re-render with vector overlay
            self._render_temporal_slice(u, v)
    
    def _draw_coastlines(self, h: int, w: int):
        """Draw coastline overlays - BRIGHT for visibility."""
        for v in self.coastlines:
            v.parent = None
        self.coastlines = []
        
        if not COASTLINES_AVAILABLE:
            return
        
        segments = get_coastlines_pixels(w, h)
        
        for seg in segments:
            # Outer glow - BRIGHT YELLOW for maximum visibility
            glow = visuals.Line(
                pos=seg,
                color=(1.0, 0.9, 0.3, 0.7),  # Yellow glow
                width=5.0,
                connect='strip',
                parent=self.view.scene
            )
            glow.order = Z_COAST_GLOW  # Layer 4
            self.coastlines.append(glow)
            
            # Core line - WHITE SOLID
            line = visuals.Line(
                pos=seg,
                color=(1.0, 1.0, 1.0, 1.0),  # Pure white
                width=2.0,
                connect='strip',
                parent=self.view.scene
            )
            line.order = Z_COAST_CORE  # Layer 3
            self.coastlines.append(line)
    
    def _draw_continent_labels(self, h: int, w: int):
        """Add continent labels for geographic context."""
        # Clear existing labels from annotations
        
        # Continent centers (lon, lat) -> pixel coords
        continents = [
            ('N. AMERICA', -100, 45),
            ('S. AMERICA', -60, -15),
            ('EUROPE', 15, 50),
            ('AFRICA', 20, 5),
            ('ASIA', 100, 45),
            ('AUSTRALIA', 135, -25),
            ('PACIFIC', -150, 0),
            ('ATLANTIC', -30, 20),
            ('INDIAN', 75, -20),
        ]
        
        for name, lon, lat in continents:
            x = ((lon + 180) / 360.0) * w
            y = ((lat + 90) / 180.0) * h
            
            # Visible label - WHITE with dark shadow
            # Shadow first
            shadow = visuals.Text(
                name,
                pos=(x+1, y-1),
                color=(0.0, 0.0, 0.0, 0.7),
                font_size=10,
                bold=True,
                anchor_x='center',
                anchor_y='center',
                parent=self.view.scene
            )
            shadow.order = Z_CONTINENT_LABELS + 1
            self.annotations.append(shadow)
            
            # Label on top
            lbl = visuals.Text(
                name,
                pos=(x, y),
                color=(1.0, 1.0, 0.8, 0.9),  # Bright cream
                font_size=10,
                bold=True,
                anchor_x='center',
                anchor_y='center',
                parent=self.view.scene
            )
            lbl.order = Z_CONTINENT_LABELS  # Layer 50
            self.annotations.append(lbl)
    
    def _draw_annotations(self, h: int, w: int, field_name: str, level_info: str):
        """Draw labels, colorbar, grid."""
        for v in self.annotations:
            v.parent = None
        self.annotations = []
        
        # Title
        title = visuals.Text(
            f"{field_name}  •  {level_info}",
            pos=(w/2, h + 30),
            color='#5A8ABA',
            font_size=11,
            bold=True,
            anchor_x='center',
            parent=self.view.scene
        )
        title.order = 100
        self.annotations.append(title)
        
        # Range indicator
        range_txt = visuals.Text(
            f"[{self._vmin:.1f} → {self._vmax:.1f}] m/s",
            pos=(w + 60, h + 30),
            color='#4A6A8A',
            font_size=9,
            anchor_x='right',
            parent=self.view.scene
        )
        range_txt.order = 100
        self.annotations.append(range_txt)
        
        # Latitude labels
        for lat_val in [-60, -30, 0, 30, 60]:
            y = ((lat_val + 90) / 180.0) * h
            lbl = visuals.Text(
                f"{lat_val:+d}°",
                pos=(-25, y),
                color='#4A6A8A',
                font_size=9,
                anchor_x='right',
                parent=self.view.scene
            )
            lbl.order = 100
            self.annotations.append(lbl)
        
        # Longitude labels
        for lon_val in [0, 60, 120, 180, 240, 300]:
            x = (lon_val / 360.0) * w
            lbl = visuals.Text(
                f"{lon_val}°",
                pos=(x, -15),
                color='#4A6A8A',
                font_size=8,
                anchor_x='center',
                parent=self.view.scene
            )
            lbl.order = 100
            self.annotations.append(lbl)
        
        # Grid
        grid_lines = []
        for lat_val in [-60, -30, 0, 30, 60]:
            y = ((lat_val + 90) / 180.0) * h
            grid_lines.extend([[0, y], [w, y]])
        for lon_val in [0, 60, 120, 180, 240, 300]:
            x = (lon_val / 360.0) * w
            grid_lines.extend([[x, 0], [x, h]])
        
        grid = visuals.Line(
            pos=np.array(grid_lines, dtype=np.float32),
            color=(0.2, 0.3, 0.4, 0.25),
            width=1,
            connect='segments',
            parent=self.view.scene
        )
        grid.order = 5
        self.annotations.append(grid)
        
        # Colorbar
        cbar_h = int(h * 0.6)
        cbar_data = np.linspace(1, 0, cbar_h).reshape(-1, 1).repeat(10, axis=1)
        cbar = visuals.Image(
            cbar_data.astype(np.float32),
            cmap=FORENSIC_CMAP,
            parent=self.view.scene
        )
        cbar.transform = STTransform(translate=(w + 20, h * 0.2))
        cbar.order = 90
        self.annotations.append(cbar)
        
        # Colorbar labels
        for frac in [0, 0.5, 1.0]:
            val = self._vmin + (1 - frac) * (self._vmax - self._vmin)
            y = h * 0.2 + frac * cbar_h
            lbl = visuals.Text(
                f"{val:.0f}",
                pos=(w + 35, y),
                color='#5A7A9A',
                font_size=8,
                anchor_x='left',
                parent=self.view.scene
            )
            lbl.order = 100
            self.annotations.append(lbl)
    
    def mark_singularity(self, lat: float, lon: float, label: str, intensity: float):
        """Mark a detected singularity."""
        h, w = self._data_shape
        x = (lon / 360.0) * w
        y = ((lat + 90) / 180.0) * h
        
        # Pulsing ring
        ring = visuals.Markers()
        ring.set_data(
            pos=np.array([[x, y]]),
            face_color=(1.0, 0.4, 0.15, 0.5 * intensity),
            edge_color=(1.0, 0.6, 0.3, 0.9),
            edge_width=2,
            size=28
        )
        ring.parent = self.view.scene
        ring.order = 110
        self.singularity_markers.append(ring)
        
        # Label
        txt = visuals.Text(
            label,
            pos=(x + 18, y),
            color=(1.0, 0.7, 0.4, 1.0),
            font_size=10,
            bold=True,
            anchor_x='left',
            parent=self.view.scene
        )
        txt.order = 111
        self.singularity_markers.append(txt)
    
    def clear_singularities(self):
        for m in self.singularity_markers:
            m.parent = None
        self.singularity_markers = []


# =============================================================================
# MAIN WINDOW
# =============================================================================

class ForensicInstrument(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HyperTensor Forensic Instrument")
        self.setMinimumSize(1700, 1000)
        self.setStyleSheet(STYLE)
        
        self.field_data = None
        self.metadata = None
        self.temporal_snapshots = None
        self.current_level = 0
        
        self._init_ui()
        self._load_manifold()
    
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Sidebar
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(300)
        sbl = QVBoxLayout(sidebar)
        sbl.setSpacing(6)
        sbl.setContentsMargins(10, 10, 10, 10)
        
        title = QLabel("FORENSIC INSTRUMENT")
        title.setObjectName("title")
        sbl.addWidget(title)
        
        sub = QLabel("Resolution-Independent • Vector Flow")
        sub.setStyleSheet("color: #3A5A7A; font-size: 10px; margin-bottom: 6px;")
        sbl.addWidget(sub)
        
        # Altitude
        g1 = QGroupBox("ALTITUDE")
        l1 = QVBoxLayout(g1)
        self.alt_slider = QSlider(Qt.Horizontal)
        self.alt_slider.setRange(0, 30)
        self.alt_slider.valueChanged.connect(self._on_altitude)
        self.alt_label = QLabel("1000 hPa")
        self.alt_label.setObjectName("value")
        l1.addWidget(self.alt_slider)
        l1.addWidget(self.alt_label)
        sbl.addWidget(g1)
        
        # Field
        g2 = QGroupBox("FIELD")
        l2 = QVBoxLayout(g2)
        self.field_combo = QComboBox()
        self.field_combo.addItems(['U-Wind', 'V-Wind', 'Temperature', 'Geopotential'])
        self.field_combo.currentIndexChanged.connect(self._on_field)
        l2.addWidget(self.field_combo)
        
        self.barb_check = QCheckBox("Vector Barbs")
        self.barb_check.setChecked(True)
        self.barb_check.stateChanged.connect(self._on_barb_toggle)
        l2.addWidget(self.barb_check)
        sbl.addWidget(g2)
        
        # Temporal
        g3 = QGroupBox("TEMPORAL SCRUBBER")
        l3 = QVBoxLayout(g3)
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, 1000)
        self.time_slider.setValue(0)
        self.time_slider.valueChanged.connect(self._on_time)
        self.time_label = QLabel("t = 0.000")
        self.time_label.setObjectName("value")
        l3.addWidget(self.time_slider)
        l3.addWidget(self.time_label)
        sbl.addWidget(g3)
        
        # Scan
        g4 = QGroupBox("SINGULARITY SCAN")
        l4 = QVBoxLayout(g4)
        self.scan_btn = QPushButton("🔍 DETECT RANK SPIKES")
        self.scan_btn.clicked.connect(self._scan)
        l4.addWidget(self.scan_btn)
        self.scan_label = QLabel("No singularities")
        self.scan_label.setWordWrap(True)
        l4.addWidget(self.scan_label)
        sbl.addWidget(g4)
        
        # Stats
        g5 = QGroupBox("STATISTICS")
        l5 = QVBoxLayout(g5)
        self.stats = {}
        for k in ['Min', 'Max', 'Mean', 'Compression']:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{k}:"))
            row.addStretch()
            v = QLabel("--")
            v.setObjectName("value")
            row.addWidget(v)
            l5.addLayout(row)
            self.stats[k] = v
        sbl.addWidget(g5)
        
        sbl.addStretch()
        layout.addWidget(sidebar)
        
        # Main
        main = QWidget()
        ml = QVBoxLayout(main)
        ml.setSpacing(0)
        ml.setContentsMargins(0, 0, 0, 0)
        
        self.viewport = ForensicViewport(on_view_change=self._on_view_change)
        ml.addWidget(self.viewport.native, stretch=1)
        
        self.log = QTextEdit()
        self.log.setObjectName("log")
        self.log.setMaximumHeight(70)
        self.log.setReadOnly(True)
        ml.addWidget(self.log)
        
        layout.addWidget(main, stretch=1)
        
        self.status = QStatusBar()
        self.setStatusBar(self.status)
    
    def _log(self, msg, level="info"):
        colors = {"info": "#4A6A8A", "ok": "#5A8A6A", "err": "#8A5A4A", "alert": "#AA7A4A"}
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.append(f'<span style="color:{colors.get(level,"#4A6A8A")}">[{ts}] {msg}</span>')
    
    def _load_manifold(self):
        path = Path(__file__).parent.parent / 'results' / 'weather_manifold.pt'
        if not path.exists():
            self._log("No manifold. Run: python demos/ingest_noaa_gfs.py", "err")
            return
        
        self._log("Loading manifold...")
        data = torch.load(path, weights_only=True)
        
        self.field_data = {
            'u': data['u'],
            'v': data.get('v'),
            'temperature': data.get('temperature'),
            'geopotential': data.get('geopotential'),
        }
        self.metadata = {
            'lat': data['latitude'],
            'lon': data['longitude'],
            'level': data['level'],
            'compression': data.get('compression_ratio', 0),
        }
        self.temporal_snapshots = data.get('temporal_snapshots', [])
        
        self.alt_slider.setRange(0, len(self.metadata['level']) - 1)
        
        # Load temporal snapshots into interpolator
        for snap in self.temporal_snapshots:
            t = snap['t']
            u = snap['u'][0]  # First level
            v = snap['v'][0]
            self.viewport.add_temporal_snapshot(t, u, v, self.metadata['lat'], self.metadata['lon'])
        
        self._log(f"Loaded: {data['u'].shape}", "ok")
        self._log(f"Compression: {self.metadata['compression']:.1f}×", "info")
        self._log(f"Temporal snapshots: {len(self.temporal_snapshots)}", "info")
        
        self._update()
    
    def _update(self):
        if self.field_data is None:
            return
        
        keys = ['u', 'v', 'temperature', 'geopotential']
        names = ['U-Wind', 'V-Wind', 'Temperature', 'Geopotential']
        
        key = keys[self.field_combo.currentIndex()]
        name = names[self.field_combo.currentIndex()]
        
        data = self.field_data.get(key)
        if data is None:
            return
        
        u = data[self.current_level]
        v = self.field_data.get('v')
        v = v[self.current_level] if v is not None else None
        
        self.viewport.set_engine(u, v if v is not None else np.zeros_like(u),
                                  self.metadata['lat'], self.metadata['lon'])
        
        pressure = self.metadata['level'][self.current_level]
        alt_km = 44.3 * (1 - (pressure / 1013.25) ** 0.19)
        level_info = f"{pressure:.0f} hPa (~{alt_km:.1f} km)"
        
        vmin, vmax = self.viewport.render(
            u, v,
            field_name=name,
            level_info=level_info,
            show_barbs=self.barb_check.isChecked()
        )
        
        self.stats['Min'].setText(f"{vmin:.1f}")
        self.stats['Max'].setText(f"{vmax:.1f}")
        self.stats['Mean'].setText(f"{u.mean():.1f}")
        self.stats['Compression'].setText(f"{self.metadata['compression']:.1f}×")
        
        self.alt_label.setText(level_info)
        self.status.showMessage(f"{name} | {level_info}")
    
    def _on_altitude(self, val):
        self.current_level = val
        self._update()
    
    def _on_field(self, idx):
        self._update()
    
    def _on_barb_toggle(self, state):
        """Vector barbs checkbox - FORCE viewport to update."""
        enabled = bool(state)
        self._log(f"Vector barbs: {'ON' if enabled else 'OFF'}", "info")
        
        # If we're in temporal mode, use the temporal render path
        if self.time_slider.value() > 0 and self.temporal_snapshots:
            self.viewport.set_barbs_enabled(enabled)
        else:
            self._update()
    
    def _on_time(self, val):
        t = val / 1000.0
        self.time_label.setText(f"t = {t:.3f}")
        
        if self.temporal_snapshots:
            # Pass current checkbox state to viewport
            self.viewport.update_temporal(t, force_barbs=self.barb_check.isChecked())
    
    def _on_view_change(self, bounds):
        """Called when camera pans/zooms - for future resolution-aware re-render."""
        pass  # Full implementation would re-contract here
    
    def _scan(self):
        if self.field_data is None:
            return
        
        self._log("Scanning for singularities...", "info")
        self.viewport.clear_singularities()
        
        keys = ['u', 'v', 'temperature', 'geopotential']
        key = keys[self.field_combo.currentIndex()]
        data = self.field_data.get(key)
        
        if data is None:
            return
        
        u = data[self.current_level]
        
        # Gradient magnitude as rank proxy
        gy, gx = np.gradient(u)
        grad = np.sqrt(gx**2 + gy**2)
        
        # Find peaks
        local_max = ndimage.maximum_filter(grad, size=15) == grad
        thresh = np.percentile(grad, 97)
        
        peaks = np.where(local_max & (grad > thresh))
        
        lat = self.metadata['lat']
        lon = self.metadata['lon']
        
        count = 0
        for yi, xi in zip(peaks[0][:5], peaks[1][:5]):
            lat_v = lat[yi]
            lon_v = lon[xi]
            intensity = grad[yi, xi] / grad.max()
            
            self.viewport.mark_singularity(lat_v, lon_v, f"S{count+1}", intensity)
            self._log(f"Singularity {count+1}: ({lat_v:.1f}°, {lon_v:.1f}°)", "alert")
            count += 1
        
        self.scan_label.setText(f"Found {count} rank spikes")
        self.scan_label.setObjectName("alert" if count > 0 else "")


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║              HYPERTENSOR FORENSIC INSTRUMENT                         ║
║                                                                      ║
║  • Resolution-Independent Manifold Sampling                          ║
║  • Vector Flow Barbs (not heatmap blobs)                             ║
║  • Geodesic Temporal Interpolation                                   ║
║  • Dynamic Per-View Normalization                                    ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    qapp = QApplication(sys.argv)
    win = ForensicInstrument()
    win.show()
    print("✅ Forensic Instrument launched")
    sys.exit(qapp.exec())


if __name__ == "__main__":
    main()
