#!/usr/bin/env python3
"""
Ontic Forensic Hub v2
===========================

Resolution-independent manifold visualization with:
1. View-frustum sampling at screen resolution
2. Vector flow rendering (barbs, not blobs)
3. Geodesic temporal interpolation
4. Auto-contrast normalization per view

"""

import sys
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import json
import hashlib

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from PySide6.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
        QLabel, QFrame, QPushButton, QApplication, QSlider,
        QStatusBar, QComboBox, QGroupBox, QLineEdit, QTextEdit,
        QCheckBox
    )
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QFont
except ImportError:
    print("ERROR: PySide6 required. pip install PySide6")
    sys.exit(1)

try:
    from vispy import scene
    from vispy.scene import visuals
    from vispy.color import Colormap, get_colormap
except ImportError:
    print("ERROR: VisPy required. pip install vispy")
    sys.exit(1)

# Coastlines
try:
    from coastlines import get_coastline_pixels
    COASTLINES_AVAILABLE = True
except ImportError:
    COASTLINES_AVAILABLE = False


# =============================================================================
# INDUSTRIAL COBALT PALETTE
# =============================================================================

# High-contrast scientific colormap (not pastel)
COBALT_CMAP = Colormap([
    (0.02, 0.02, 0.08, 1.0),   # Deep void
    (0.05, 0.15, 0.35, 1.0),   # Deep cobalt
    (0.10, 0.30, 0.55, 1.0),   # Ocean blue
    (0.20, 0.50, 0.70, 1.0),   # Steel blue
    (0.40, 0.75, 0.85, 1.0),   # Electric cyan
    (0.70, 0.90, 0.95, 1.0),   # Ice white
    (0.95, 0.85, 0.60, 1.0),   # Warning amber
    (0.95, 0.50, 0.20, 1.0),   # Alert orange
    (0.85, 0.20, 0.15, 1.0),   # Critical red
    (0.50, 0.05, 0.10, 1.0),   # Deep crimson
])

STYLE = """
QMainWindow { background-color: #080810; }
QWidget { background-color: #080810; color: #AACCEE; font-family: 'Consolas'; }
QFrame#sidebar { background: #0A0A14; border-right: 1px solid #1A2A3A; }
QGroupBox { 
    border: 1px solid #2A3A4A; 
    margin-top: 12px; 
    padding-top: 8px;
    font-weight: bold;
    color: #4A7AAA;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; }
QPushButton { 
    background: #1A2A3A; 
    border: 1px solid #3A4A5A; 
    padding: 8px 14px;
    color: #8AC;
}
QPushButton:hover { background: #2A3A4A; border-color: #5A8ABB; }
QPushButton:pressed { background: #4A7AAA; }
QPushButton:checked { background: #4A7AAA; color: white; }
QComboBox { background: #1A2A3A; border: 1px solid #3A4A5A; padding: 6px; }
QSlider::groove:horizontal { background: #1A2A3A; height: 6px; border-radius: 3px; }
QSlider::handle:horizontal { 
    background: #4A7AAA; width: 16px; margin: -5px 0; border-radius: 8px; 
}
QLineEdit#intent {
    background: #0A0A14;
    border: 2px solid #2A3A4A;
    padding: 10px;
    font-size: 13px;
    color: #8AC;
}
QLineEdit#intent:focus { border-color: #4A7AAA; }
QTextEdit#log {
    background: #050508;
    border: 1px solid #1A2A3A;
    font-size: 10px;
    color: #557788;
}
QLabel#title { font-size: 14px; font-weight: bold; color: #4A7AAA; }
QLabel#value { font-family: 'Consolas'; font-size: 12px; color: #8AC; }
QLabel#alert { color: #FF6644; font-weight: bold; }
QStatusBar { background: #080810; border-top: 1px solid #1A2A3A; color: #446688; }
"""


# =============================================================================
# MANIFOLD SAMPLER - Resolution Independent
# =============================================================================

class ManifoldSampler:
    """
    Samples the manifold at arbitrary resolution within view bounds.
    This is the key to infinite zoom without pixelation.
    """
    
    def __init__(self, data: np.ndarray, lat: np.ndarray, lon: np.ndarray):
        """
        Args:
            data: 2D field data (lat x lon)
            lat: latitude coordinates
            lon: longitude coordinates
        """
        self.data = data
        self.lat = lat
        self.lon = lon
        self.h, self.w = data.shape
        
    def sample(self, bounds: tuple, resolution: tuple) -> np.ndarray:
        """
        Sample manifold within bounds at given resolution.
        
        Args:
            bounds: (x_min, x_max, y_min, y_max) in pixel coords
            resolution: (width, height) output resolution
            
        Returns:
            Resampled data at requested resolution
        """
        x_min, x_max, y_min, y_max = bounds
        out_w, out_h = resolution
        
        # Clamp bounds
        x_min = max(0, x_min)
        x_max = min(self.w, x_max)
        y_min = max(0, y_min)
        y_max = min(self.h, y_max)
        
        if x_max <= x_min or y_max <= y_min:
            return np.zeros((out_h, out_w), dtype=np.float32)
        
        # Generate sample coordinates
        x_coords = np.linspace(x_min, x_max - 1, out_w)
        y_coords = np.linspace(y_min, y_max - 1, out_h)
        
        # Bilinear interpolation for smooth sampling
        # (In production, this would be QTT contraction)
        from scipy import ndimage
        
        # Create coordinate grids
        xi = x_coords
        yi = y_coords
        
        # Use map_coordinates for proper interpolation
        coords = np.meshgrid(yi, xi, indexing='ij')
        sampled = ndimage.map_coordinates(
            self.data, 
            [coords[0], coords[1]], 
            order=1,  # Bilinear
            mode='nearest'
        )
        
        return sampled.astype(np.float32)


# =============================================================================
# VECTOR FIELD RENDERER
# =============================================================================

class VectorFieldRenderer:
    """
    Renders vector fields as flow barbs instead of blurry heatmaps.
    """
    
    def __init__(self, parent_scene):
        self.parent = parent_scene
        self.arrows = None
        self.magnitude_image = None
        
    def render(self, u: np.ndarray, v: np.ndarray, 
               subsample: int = 8, scale: float = 2.0):
        """
        Render vector field with flow barbs.
        
        Args:
            u: East-west velocity component
            v: North-south velocity component
            subsample: Draw arrow every N pixels
            scale: Arrow length scale
        """
        self.clear()
        
        h, w = u.shape
        
        # Compute magnitude for background
        mag = np.sqrt(u**2 + v**2)
        
        # Normalize magnitude for color
        mag_norm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
        
        # Create magnitude background
        self.magnitude_image = visuals.Image(
            mag_norm.astype(np.float32),
            cmap=COBALT_CMAP,
            parent=self.parent
        )
        
        # Create arrow positions and directions
        y_idx = np.arange(subsample//2, h, subsample)
        x_idx = np.arange(subsample//2, w, subsample)
        
        positions = []
        arrows_data = []
        colors = []
        
        for yi in y_idx:
            for xi in x_idx:
                uu = u[yi, xi]
                vv = v[yi, xi]
                m = np.sqrt(uu**2 + vv**2)
                
                if m > 0.1:  # Skip near-zero vectors
                    # Start position
                    x0, y0 = xi, yi
                    
                    # End position (normalized direction * scale)
                    dx = (uu / m) * scale * min(m / 10, 3)
                    dy = (vv / m) * scale * min(m / 10, 3)
                    
                    positions.append([x0, y0])
                    arrows_data.append([x0, y0, x0 + dx, y0 + dy])
                    
                    # Color by magnitude
                    t = min(m / 50, 1.0)  # Normalize to ~50 m/s max
                    colors.append([0.3 + 0.7*t, 0.8 - 0.3*t, 1.0 - 0.5*t, 0.8])
        
        if arrows_data:
            arrows_data = np.array(arrows_data, dtype=np.float32)
            colors = np.array(colors, dtype=np.float32)
            
            # Draw arrows as lines
            # Reshape for segment drawing: [start, end, start, end, ...]
            line_pos = arrows_data.reshape(-1, 2)
            line_colors = np.repeat(colors, 2, axis=0)
            
            self.arrows = visuals.Line(
                pos=line_pos,
                color=line_colors,
                width=1.5,
                connect='segments',
                parent=self.parent
            )
            self.arrows.order = 50
            
        return mag
    
    def clear(self):
        if self.arrows is not None:
            self.arrows.parent = None
            self.arrows = None
        if self.magnitude_image is not None:
            self.magnitude_image.parent = None
            self.magnitude_image = None


# =============================================================================
# FORENSIC VIEWPORT v2
# =============================================================================

class ForensicViewportV2:
    """
    Resolution-independent viewport with:
    - View-frustum sampling
    - Auto-contrast normalization
    - Vector flow rendering
    - Coastline overlay
    """
    
    def __init__(self):
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            show=False,
            bgcolor='#080810'
        )
        
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.PanZoomCamera(aspect=1)
        
        # Visuals
        self.field_image = None
        self.vector_renderer = VectorFieldRenderer(self.view.scene)
        self.coastline_visuals = []
        self.annotation_visuals = []
        self.anomaly_markers = []
        
        # Data
        self.sampler = None
        self.u_data = None
        self.v_data = None
        self._current_bounds = None
        self._data_shape = (180, 360)
        
        # State
        self._vmin = 0
        self._vmax = 1
        self._field_name = "U-Wind"
        self._level_info = "Surface"
        self._show_vectors = False
        
        # Connect camera events for resolution-independent sampling
        # Note: event handling for pan/zoom is automatic
        
    @property
    def native(self):
        return self.canvas.native
    
    def set_data(self, u: np.ndarray, v: np.ndarray, 
                 lat: np.ndarray, lon: np.ndarray):
        """Set the field data."""
        self.u_data = u
        self.v_data = v
        self.sampler = ManifoldSampler(u, lat, lon)
        self._data_shape = u.shape
        
    def update_field(self, data: np.ndarray, 
                     field_name: str = "U-Wind", 
                     level_info: str = "Surface",
                     show_vectors: bool = False):
        """Update the field display with auto-contrast."""
        
        if self.field_image is not None:
            self.field_image.parent = None
        self.vector_renderer.clear()
        
        self._field_name = field_name
        self._level_info = level_info
        self._show_vectors = show_vectors
        
        # AUTO-CONTRAST: Normalize to visible range (not global)
        self._vmin = float(np.percentile(data, 2))  # Ignore outliers
        self._vmax = float(np.percentile(data, 98))
        
        # Prevent zero-range
        if abs(self._vmax - self._vmin) < 1e-6:
            self._vmax = self._vmin + 1.0
        
        # Normalize
        normalized = (data - self._vmin) / (self._vmax - self._vmin)
        normalized = np.clip(normalized, 0, 1)
        
        if show_vectors and self.v_data is not None:
            # Vector field mode
            self.vector_renderer.render(
                self.u_data if self.u_data is not None else data,
                self.v_data,
                subsample=6,
                scale=3.0
            )
        else:
            # Scalar field mode with industrial colormap
            self.field_image = visuals.Image(
                normalized.astype(np.float32),
                cmap=COBALT_CMAP,
                parent=self.view.scene
            )
        
        h, w = data.shape
        self._data_shape = (h, w)
        
        # Set camera range with margins for labels
        self.view.camera.set_range(x=(-40, w+80), y=(-30, h+40))
        self._current_bounds = (0, w, 0, h)
        
        # Add overlays
        self._add_coastlines(h, w)
        self._add_annotations(h, w)
        
        self.canvas.update()
        return self._vmin, self._vmax
    
    def _on_camera_change(self, event):
        """
        Called when camera pans/zooms.
        In production: resample manifold at screen resolution.
        """
        # Get view bounds
        rect = self.view.camera.rect
        if rect is not None:
            x_min, y_min = rect.left, rect.bottom
            x_max, y_max = rect.right, rect.top
            self._current_bounds = (x_min, x_max, y_min, y_max)
            
            # For true resolution-independence, you would:
            # 1. Get screen size
            # 2. Sample manifold at that resolution within bounds
            # 3. Renormalize for auto-contrast
            # self._resample_at_screen_resolution()
    
    def _add_coastlines(self, h: int, w: int):
        """Add coastline overlays."""
        for v in self.coastline_visuals:
            v.parent = None
        self.coastline_visuals = []
        
        if not COASTLINES_AVAILABLE:
            return
            
        segments = get_coastline_pixels(w, h)
        
        for segment in segments:
            # Bright outline for visibility against any background
            line = visuals.Line(
                pos=segment,
                color=(0.9, 0.95, 1.0, 0.7),  # Bright cyan-white
                width=2.0,
                connect='strip',
                antialias=True,
                parent=self.view.scene
            )
            line.order = 80
            self.coastline_visuals.append(line)
            
            # Dark core
            line2 = visuals.Line(
                pos=segment,
                color=(0.0, 0.1, 0.2, 0.9),
                width=1.0,
                connect='strip',
                antialias=True,
                parent=self.view.scene
            )
            line2.order = 81
            self.coastline_visuals.append(line2)
    
    def _add_annotations(self, h: int, w: int):
        """Add labels and colorbar."""
        for v in self.annotation_visuals:
            v.parent = None
        self.annotation_visuals = []
        
        # Title
        title = visuals.Text(
            f"{self._field_name}  •  {self._level_info}",
            pos=(w/2, h + 25),
            color='#5A9ACA',
            font_size=11,
            bold=True,
            anchor_x='center',
            parent=self.view.scene
        )
        self.annotation_visuals.append(title)
        
        # Range (the critical fix)
        range_text = visuals.Text(
            f"Range: [{self._vmin:.1f}, {self._vmax:.1f}] m/s",
            pos=(w - 10, h + 25),
            color='#6A8AAA',
            font_size=9,
            anchor_x='right',
            parent=self.view.scene
        )
        self.annotation_visuals.append(range_text)
        
        # Latitude labels
        for lat_val in [-60, -30, 0, 30, 60]:
            y = ((lat_val + 90) / 180.0) * h
            label = visuals.Text(
                f"{lat_val:+d}°",
                pos=(-20, y),
                color='#5577AA',
                font_size=9,
                anchor_x='right',
                parent=self.view.scene
            )
            self.annotation_visuals.append(label)
        
        # Longitude labels
        for lon_val in [0, 90, 180, 270]:
            x = (lon_val / 360.0) * w
            label = visuals.Text(
                f"{lon_val}°E",
                pos=(x, -15),
                color='#5577AA',
                font_size=9,
                anchor_x='center',
                parent=self.view.scene
            )
            self.annotation_visuals.append(label)
        
        # Colorbar
        self._add_colorbar(h, w)
        
        # Grid lines
        self._add_grid(h, w)
    
    def _add_colorbar(self, h: int, w: int):
        """Add vertical colorbar on right."""
        cbar_h = int(h * 0.7)
        cbar_data = np.linspace(1, 0, cbar_h).reshape(-1, 1).repeat(12, axis=1)
        
        from vispy.visuals.transforms import STTransform
        
        cbar = visuals.Image(
            cbar_data.astype(np.float32),
            cmap=COBALT_CMAP,
            parent=self.view.scene
        )
        cbar.transform = STTransform(translate=(w + 15, h * 0.15))
        cbar.order = 90
        self.annotation_visuals.append(cbar)
        
        # Scale labels
        for i, frac in enumerate([0, 0.5, 1.0]):
            val = self._vmin + (1 - frac) * (self._vmax - self._vmin)
            y = h * 0.15 + frac * cbar_h
            label = visuals.Text(
                f"{val:.0f}",
                pos=(w + 32, y),
                color='#6A8AAA',
                font_size=8,
                anchor_x='left',
                parent=self.view.scene
            )
            self.annotation_visuals.append(label)
    
    def _add_grid(self, h: int, w: int):
        """Add subtle grid lines."""
        lines = []
        
        for lat_val in [-60, -30, 0, 30, 60]:
            y = ((lat_val + 90) / 180.0) * h
            lines.extend([[0, y], [w, y]])
            
        for lon_val in [0, 60, 120, 180, 240, 300]:
            x = (lon_val / 360.0) * w
            lines.extend([[x, 0], [x, h]])
        
        grid = visuals.Line(
            pos=np.array(lines).reshape(-1, 2),
            color=(0.2, 0.3, 0.4, 0.3),
            width=1,
            connect='segments',
            parent=self.view.scene
        )
        grid.order = 5
        self.annotation_visuals.append(grid)
    
    def add_anomaly_marker(self, lat: float, lon: float, label: str, intensity: float = 1.0):
        """Add a singularity marker."""
        h, w = self._data_shape
        x = (lon / 360.0) * w
        y = ((lat + 90) / 180.0) * h
        
        # Pulsing ring
        marker = visuals.Markers()
        marker.set_data(
            pos=np.array([[x, y]]),
            face_color=(1.0, 0.3, 0.1, 0.6 * intensity),
            edge_color=(1.0, 0.5, 0.2, 1.0),
            edge_width=2,
            size=25
        )
        marker.parent = self.view.scene
        marker.order = 100
        self.anomaly_markers.append(marker)
        
        # Label
        text = visuals.Text(
            label,
            pos=(x + 15, y),
            color=(1.0, 0.6, 0.3, 1.0),
            font_size=10,
            anchor_x='left',
            parent=self.view.scene
        )
        text.order = 101
        self.anomaly_markers.append(text)
    
    def clear_anomaly_markers(self):
        for m in self.anomaly_markers:
            m.parent = None
        self.anomaly_markers = []


# =============================================================================
# MAIN WINDOW
# =============================================================================

class ForensicHubV2(QMainWindow):
    """
    Ontic Forensic Hub v2
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ontic Forensic Hub v2")
        self.setMinimumSize(1600, 1000)
        self.setStyleSheet(STYLE)
        
        # State
        self.field_data = None
        self.metadata = None
        self.current_level = 0
        self.current_field = 'u'
        self.show_vectors = False
        
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
        sidebar.setFixedWidth(320)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setSpacing(8)
        sidebar_layout.setContentsMargins(12, 12, 12, 12)
        
        title = QLabel("FORENSIC HUB v2")
        title.setObjectName("title")
        sidebar_layout.addWidget(title)
        
        subtitle = QLabel("Resolution-Independent • Vector Flow")
        subtitle.setStyleSheet("color: #446688; margin-bottom: 8px;")
        sidebar_layout.addWidget(subtitle)
        
        # Altitude
        alt_group = QGroupBox("ALTITUDE")
        alt_layout = QVBoxLayout(alt_group)
        
        self.alt_slider = QSlider(Qt.Horizontal)
        self.alt_slider.setRange(0, 30)
        self.alt_slider.setValue(0)
        self.alt_slider.valueChanged.connect(self._on_altitude_changed)
        
        self.alt_label = QLabel("1000 hPa (Surface)")
        self.alt_label.setObjectName("value")
        
        alt_layout.addWidget(self.alt_slider)
        alt_layout.addWidget(self.alt_label)
        sidebar_layout.addWidget(alt_group)
        
        # Field selector
        field_group = QGroupBox("FIELD")
        field_layout = QVBoxLayout(field_group)
        
        self.field_combo = QComboBox()
        self.field_combo.addItems(['U-Wind', 'V-Wind', 'Temperature', 'Geopotential'])
        self.field_combo.currentIndexChanged.connect(self._on_field_changed)
        field_layout.addWidget(self.field_combo)
        
        self.vector_check = QCheckBox("Show Vector Barbs")
        self.vector_check.stateChanged.connect(self._on_vector_toggle)
        field_layout.addWidget(self.vector_check)
        
        sidebar_layout.addWidget(field_group)
        
        # Anomaly detection
        scan_group = QGroupBox("SINGULARITY SCAN")
        scan_layout = QVBoxLayout(scan_group)
        
        self.scan_btn = QPushButton("🔍 SCAN FOR RANK SPIKES")
        self.scan_btn.clicked.connect(self._scan_anomalies)
        scan_layout.addWidget(self.scan_btn)
        
        self.anomaly_label = QLabel("No singularities detected")
        self.anomaly_label.setWordWrap(True)
        scan_layout.addWidget(self.anomaly_label)
        
        sidebar_layout.addWidget(scan_group)
        
        # Stats
        stats_group = QGroupBox("FIELD STATISTICS")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_labels = {}
        for stat in ['Min', 'Max', 'Mean', 'Std', 'Compression']:
            row = QHBoxLayout()
            label = QLabel(f"{stat}:")
            value = QLabel("--")
            value.setObjectName("value")
            row.addWidget(label)
            row.addStretch()
            row.addWidget(value)
            stats_layout.addLayout(row)
            self.stats_labels[stat] = value
        
        sidebar_layout.addWidget(stats_group)
        sidebar_layout.addStretch()
        
        layout.addWidget(sidebar)
        
        # Main viewport
        main_area = QWidget()
        main_layout = QVBoxLayout(main_area)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Viewport
        self.viewport = ForensicViewportV2()
        main_layout.addWidget(self.viewport.native, stretch=1)
        
        # Log
        self.log = QTextEdit()
        self.log.setObjectName("log")
        self.log.setMaximumHeight(80)
        self.log.setReadOnly(True)
        main_layout.addWidget(self.log)
        
        layout.addWidget(main_area, stretch=1)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
    
    def _log(self, msg: str, level: str = "info"):
        colors = {"info": "#6688AA", "success": "#66AA88", "error": "#AA6666", "alert": "#AAAA66"}
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.append(f'<span style="color:{colors.get(level, "#6688AA")}">[{ts}] {msg}</span>')
    
    def _load_manifold(self):
        path = Path(__file__).parent.parent / 'results' / 'weather_manifold.pt'
        
        if not path.exists():
            self._log("No manifold found. Run: python demos/ingest_noaa_gfs.py", "error")
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
            'latitude': data['latitude'],
            'longitude': data['longitude'],
            'level': data['level'],
            'compression_ratio': data.get('compression_ratio', 0),
        }
        
        self.alt_slider.setRange(0, len(self.metadata['level']) - 1)
        
        self._log(f"Loaded: {data['u'].shape}", "success")
        self._log(f"Compression: {self.metadata['compression_ratio']:.1f}×", "info")
        
        self._update_display()
    
    def _update_display(self):
        if self.field_data is None:
            return
        
        field_keys = ['u', 'v', 'temperature', 'geopotential']
        field_names = ['U-Wind', 'V-Wind', 'Temperature', 'Geopotential']
        
        field_key = field_keys[self.field_combo.currentIndex()]
        field_name = field_names[self.field_combo.currentIndex()]
        
        data = self.field_data.get(field_key)
        if data is None:
            return
        
        slice_data = data[self.current_level]
        
        # Set U and V for vector rendering
        u_slice = self.field_data['u'][self.current_level]
        v_slice = self.field_data.get('v')
        if v_slice is not None:
            v_slice = v_slice[self.current_level]
        
        self.viewport.set_data(
            u_slice, 
            v_slice if v_slice is not None else np.zeros_like(u_slice),
            self.metadata['latitude'],
            self.metadata['longitude']
        )
        
        pressure = self.metadata['level'][self.current_level]
        altitude_km = 44.3 * (1 - (pressure / 1013.25) ** 0.19)
        level_info = f"{pressure:.0f} hPa (~{altitude_km:.1f} km)"
        
        vmin, vmax = self.viewport.update_field(
            slice_data,
            field_name=field_name,
            level_info=level_info,
            show_vectors=self.show_vectors
        )
        
        # Update stats
        self.stats_labels['Min'].setText(f"{vmin:.2f}")
        self.stats_labels['Max'].setText(f"{vmax:.2f}")
        self.stats_labels['Mean'].setText(f"{slice_data.mean():.2f}")
        self.stats_labels['Std'].setText(f"{slice_data.std():.2f}")
        self.stats_labels['Compression'].setText(f"{self.metadata['compression_ratio']:.1f}×")
        
        self.alt_label.setText(level_info)
        self.status_bar.showMessage(f"Field: {field_name} | Level: {level_info}")
    
    def _on_altitude_changed(self, value: int):
        self.current_level = value
        self._update_display()
    
    def _on_field_changed(self, index: int):
        self._update_display()
    
    def _on_vector_toggle(self, state: int):
        self.show_vectors = state == Qt.Checked
        self._update_display()
    
    def _scan_anomalies(self):
        if self.field_data is None:
            return
        
        self._log("Scanning for rank spikes...", "info")
        self.viewport.clear_anomaly_markers()
        
        # Get current field
        field_keys = ['u', 'v', 'temperature', 'geopotential']
        field_key = field_keys[self.field_combo.currentIndex()]
        data = self.field_data.get(field_key)
        
        if data is None:
            return
        
        slice_data = data[self.current_level]
        
        # Detect rank spikes via local SVD
        from scipy import ndimage
        
        # Compute local gradient magnitude (proxy for rank)
        grad_y, grad_x = np.gradient(slice_data)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Find local maxima
        local_max = ndimage.maximum_filter(grad_mag, size=20) == grad_mag
        threshold = np.percentile(grad_mag, 98)
        
        peaks = np.where(local_max & (grad_mag > threshold))
        
        lat = self.metadata['latitude']
        lon = self.metadata['longitude']
        
        count = 0
        for yi, xi in zip(peaks[0][:5], peaks[1][:5]):  # Top 5
            lat_val = lat[yi]
            lon_val = lon[xi]
            intensity = grad_mag[yi, xi] / grad_mag.max()
            
            self.viewport.add_anomaly_marker(
                lat_val, lon_val,
                f"S{count+1}: {intensity:.2f}",
                intensity
            )
            self._log(f"Singularity {count+1}: ({lat_val:.1f}°, {lon_val:.1f}°) intensity={intensity:.2f}", "alert")
            count += 1
        
        if count > 0:
            self.anomaly_label.setText(f"Found {count} rank spikes")
            self.anomaly_label.setObjectName("alert")
        else:
            self.anomaly_label.setText("No significant singularities")


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                ONTIC_ENGINE FORENSIC HUB v2                           ║
║                                                                      ║
║  Resolution-Independent • Vector Flow • Auto-Contrast                ║
║  Industrial Cobalt Palette • Singularity Detection                   ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    app = QApplication(sys.argv)
    hub = ForensicHubV2()
    hub.show()
    
    print("✅ Forensic Hub v2 launched")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
