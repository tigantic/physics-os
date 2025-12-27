#!/usr/bin/env python3
"""
HyperTensor Professional Visualization System
==============================================

Broadcast-quality atmospheric visualization using NASA Blue Marble imagery.

Features:
    • NASA Blue Marble 8K satellite imagery substrate
    • Professional streamline/arrow vector rendering
    • Perceptually uniform colormaps
    • Proper alpha compositing
    • Broadcast-quality HUD overlays
    • Smooth temporal animation with geodesic interpolation
    
Usage:
    python hypertensor_pro.py
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# IMPORTS - GUI
# =============================================================================
try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QFrame, QPushButton, QSlider, QComboBox, QCheckBox,
        QGroupBox, QSplitter, QStatusBar, QToolBar, QSizePolicy
    )
    from PySide6.QtCore import Qt, QTimer, Signal
    from PySide6.QtGui import QFont, QAction, QIcon
except ImportError:
    print("ERROR: PySide6 required. Install with: pip install PySide6")
    sys.exit(1)

# =============================================================================
# IMPORTS - VISUALIZATION
# =============================================================================
try:
    from vispy import scene, app
    from vispy.scene import visuals
    from vispy.color import Colormap
except ImportError:
    print("ERROR: VisPy required. Install with: pip install vispy")
    sys.exit(1)

# =============================================================================
# IMPORTS - LOCAL
# =============================================================================
from earth_renderer import EarthRenderer, VectorFieldRenderer, Compositor

# =============================================================================
# CONSTANTS
# =============================================================================
WINDOW_TITLE = "HyperTensor Professional Visualization"
WINDOW_SIZE = (1600, 900)

# Professional colormaps (perceptually uniform)
COLORMAPS = {
    "Plasma": "plasma",
    "Viridis": "viridis",
    "Inferno": "inferno",
    "Magma": "magma",
    "Turbo": "turbo",
    "Coolwarm": "coolwarm",
}

# Wind intensity colormap (custom)
WIND_COLORS = [
    (0.02, 0.15, 0.40, 1.0),   # 0.0: Deep blue (calm)
    (0.10, 0.35, 0.65, 1.0),   # 0.2: Ocean blue
    (0.20, 0.60, 0.70, 1.0),   # 0.4: Cyan
    (0.50, 0.80, 0.50, 1.0),   # 0.5: Seafoam
    (0.90, 0.85, 0.40, 1.0),   # 0.6: Yellow
    (0.95, 0.55, 0.25, 1.0),   # 0.7: Orange
    (0.90, 0.25, 0.20, 1.0),   # 0.85: Red
    (0.70, 0.15, 0.35, 1.0),   # 0.95: Magenta
    (0.50, 0.10, 0.40, 1.0),   # 1.0: Purple (extreme)
]


# =============================================================================
# COLOR LEGEND BAR
# =============================================================================
class ColorLegendBar(QFrame):
    """
    Horizontal color legend bar showing colormap with value labels.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(50)
        self.setStyleSheet("""
            QFrame {
                background-color: rgba(0, 0, 0, 0.7);
                border-radius: 4px;
            }
        """)
        
        self._vmin = 0
        self._vmax = 1
        self._units = "m/s"
        self._colormap = "plasma"
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(2)
        
        # Color bar (will be drawn with gradient)
        self.color_bar = QLabel()
        self.color_bar.setFixedHeight(15)
        self.color_bar.setStyleSheet("border-radius: 2px;")
        layout.addWidget(self.color_bar)
        
        # Labels row
        labels_layout = QHBoxLayout()
        labels_layout.setContentsMargins(0, 0, 0, 0)
        
        self.min_label = QLabel("0.0")
        self.min_label.setStyleSheet("color: white; font-size: 10px;")
        labels_layout.addWidget(self.min_label)
        
        labels_layout.addStretch()
        
        self.mid_label = QLabel("0.5")
        self.mid_label.setStyleSheet("color: white; font-size: 10px;")
        labels_layout.addWidget(self.mid_label)
        
        labels_layout.addStretch()
        
        self.max_label = QLabel("1.0")
        self.max_label.setStyleSheet("color: white; font-size: 10px;")
        labels_layout.addWidget(self.max_label)
        
        layout.addLayout(labels_layout)
        
        self._update_gradient()
        
    def set_range(self, vmin: float, vmax: float, units: str = "m/s"):
        self._vmin = vmin
        self._vmax = vmax
        self._units = units
        
        mid = (vmin + vmax) / 2
        self.min_label.setText(f"{vmin:.1f}")
        self.mid_label.setText(f"{mid:.1f} {units}")
        self.max_label.setText(f"{vmax:.1f}")
        
    def set_colormap(self, cmap: str):
        self._colormap = cmap
        self._update_gradient()
        
    def _update_gradient(self):
        """Update the gradient bar based on colormap."""
        try:
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap(self._colormap)
            
            # Build Qt gradient stylesheet
            self.color_bar.setStyleSheet(f"""
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0.0 rgb({int(cmap(0.0)[0]*255)},{int(cmap(0.0)[1]*255)},{int(cmap(0.0)[2]*255)}),
                    stop:0.25 rgb({int(cmap(0.25)[0]*255)},{int(cmap(0.25)[1]*255)},{int(cmap(0.25)[2]*255)}),
                    stop:0.5 rgb({int(cmap(0.5)[0]*255)},{int(cmap(0.5)[1]*255)},{int(cmap(0.5)[2]*255)}),
                    stop:0.75 rgb({int(cmap(0.75)[0]*255)},{int(cmap(0.75)[1]*255)},{int(cmap(0.75)[2]*255)}),
                    stop:1.0 rgb({int(cmap(1.0)[0]*255)},{int(cmap(1.0)[1]*255)},{int(cmap(1.0)[2]*255)})
                );
                border-radius: 2px;
            """)
        except Exception:
            # Fallback gradient
            self.color_bar.setStyleSheet("""
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0d0887, stop:0.25 #7e03a8, stop:0.5 #cc4778,
                    stop:0.75 #f89540, stop:1 #f0f921);
                border-radius: 2px;
            """)


# =============================================================================
# PROFESSIONAL HUD OVERLAY
# =============================================================================
class HUDOverlay(QFrame):
    """
    Broadcast-quality heads-up display overlay.
    
    Shows:
        - Data source and initialization time
        - Current display time and forecast hour
        - Variable name, level, units
        - Color legend with values
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            HUDOverlay {
                background-color: transparent;
            }
            QLabel {
                color: white;
                font-family: 'Segoe UI', 'Roboto', sans-serif;
            }
        """)
        
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Top bar
        self.top_bar = QFrame()
        self.top_bar.setStyleSheet("""
            QFrame {
                background-color: rgba(0, 0, 0, 0.7);
                border-radius: 4px;
            }
        """)
        top_layout = QHBoxLayout(self.top_bar)
        top_layout.setContentsMargins(12, 8, 12, 8)
        
        # Left: Data source
        self.source_label = QLabel("HyperTensor Simulation")
        self.source_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        top_layout.addWidget(self.source_label)
        
        top_layout.addStretch()
        
        # Center: Variable info
        self.variable_label = QLabel("U-Wind Component | 10m AGL")
        self.variable_label.setStyleSheet("font-size: 13px;")
        top_layout.addWidget(self.variable_label)
        
        top_layout.addStretch()
        
        # Right: Time
        self.time_label = QLabel("2025-12-27 00:00 UTC")
        self.time_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        top_layout.addWidget(self.time_label)
        
        layout.addWidget(self.top_bar)
        
        # Spacer
        layout.addStretch()
        
        # Color legend bar
        self.legend_bar = ColorLegendBar()
        layout.addWidget(self.legend_bar)
        
        # Bottom bar with stats
        self.bottom_bar = QFrame()
        self.bottom_bar.setStyleSheet("""
            QFrame {
                background-color: rgba(0, 0, 0, 0.7);
                border-radius: 4px;
            }
        """)
        bottom_layout = QHBoxLayout(self.bottom_bar)
        bottom_layout.setContentsMargins(12, 8, 12, 8)
        
        # Left: Stats
        self.stats_label = QLabel("Range: -15.2 to 32.8 m/s")
        self.stats_label.setStyleSheet("font-size: 12px;")
        bottom_layout.addWidget(self.stats_label)
        
        bottom_layout.addStretch()
        
        # Center: Forecast hour
        self.forecast_label = QLabel("Analysis (T+0)")
        self.forecast_label.setStyleSheet("font-size: 13px; font-weight: bold;")
        bottom_layout.addWidget(self.forecast_label)
        
        bottom_layout.addStretch()
        
        # Right: Coordinates
        self.coord_label = QLabel("Cursor: ---.--°, ---.--°")
        self.coord_label.setStyleSheet("font-size: 12px;")
        bottom_layout.addWidget(self.coord_label)
        
        layout.addWidget(self.bottom_bar)
        
    def update_source(self, source: str):
        self.source_label.setText(source)
        
    def update_variable(self, name: str, level: str, units: str = "m/s"):
        self.variable_label.setText(f"{name} | {level}")
        
    def update_time(self, dt: datetime):
        self.time_label.setText(dt.strftime("%Y-%m-%d %H:%M UTC"))
        
    def update_stats(self, vmin: float, vmax: float, units: str = "m/s"):
        self.stats_label.setText(f"Range: {vmin:.1f} to {vmax:.1f} {units}")
        self.legend_bar.set_range(vmin, vmax, units)
        
    def update_colormap(self, cmap: str):
        self.legend_bar.set_colormap(cmap)
        
    def update_forecast(self, hours: int):
        if hours == 0:
            self.forecast_label.setText("Analysis (T+0)")
        else:
            self.forecast_label.setText(f"Forecast T+{hours}h")
            
    def update_cursor(self, lat: float, lon: float):
        self.coord_label.setText(f"Cursor: {lat:.2f}°, {lon:.2f}°")


# =============================================================================
# CONTROL PANEL
# =============================================================================
class ControlPanel(QFrame):
    """
    Professional control panel with all visualization options.
    """
    
    # Signals
    opacity_changed = Signal(float)
    colormap_changed = Signal(str)
    vector_mode_changed = Signal(str)
    temporal_changed = Signal(float)
    play_toggled = Signal(bool)
    grid_toggled = Signal(bool)
    coastlines_toggled = Signal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(280)
        self.setStyleSheet("""
            ControlPanel {
                background-color: #1a1a2e;
                border-left: 1px solid #333;
            }
            QLabel {
                color: #e0e0e0;
                font-size: 12px;
            }
            QGroupBox {
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #444;
                border-radius: 4px;
                margin-top: 12px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #333;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4a9eff;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: #4a9eff;
                border-radius: 3px;
            }
            QComboBox {
                background: #2a2a4e;
                color: white;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 5px 10px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background: #2a2a4e;
                color: white;
                selection-background-color: #4a9eff;
            }
            QPushButton {
                background: #4a9eff;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #6ab0ff;
            }
            QPushButton:pressed {
                background: #3a8eef;
            }
            QPushButton:checked {
                background: #ff6b6b;
            }
            QCheckBox {
                color: #e0e0e0;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 3px;
                border: 2px solid #555;
            }
            QCheckBox::indicator:checked {
                background: #4a9eff;
                border-color: #4a9eff;
            }
        """)
        
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("VISUALIZATION CONTROLS")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #4a9eff;")
        layout.addWidget(title)
        
        # === DISPLAY OPTIONS ===
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)
        
        # Colormap
        cmap_row = QHBoxLayout()
        cmap_row.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(list(COLORMAPS.keys()))
        self.colormap_combo.currentTextChanged.connect(
            lambda t: self.colormap_changed.emit(COLORMAPS.get(t, "plasma"))
        )
        cmap_row.addWidget(self.colormap_combo)
        display_layout.addLayout(cmap_row)
        
        # Opacity
        opacity_row = QVBoxLayout()
        opacity_header = QHBoxLayout()
        opacity_header.addWidget(QLabel("Overlay Opacity:"))
        self.opacity_value = QLabel("50%")
        opacity_header.addWidget(self.opacity_value)
        opacity_row.addLayout(opacity_header)
        
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(10, 90)
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_row.addWidget(self.opacity_slider)
        display_layout.addLayout(opacity_row)
        
        layout.addWidget(display_group)
        
        # === VECTOR FIELD ===
        vector_group = QGroupBox("Vector Field")
        vector_layout = QVBoxLayout(vector_group)
        
        # Vector mode
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.vector_combo = QComboBox()
        self.vector_combo.addItems(["Streamlines", "Arrows", "Both", "None"])
        self.vector_combo.currentTextChanged.connect(self.vector_mode_changed.emit)
        mode_row.addWidget(self.vector_combo)
        vector_layout.addLayout(mode_row)
        
        # Show coastlines
        self.coastlines_check = QCheckBox("Show Coastlines")
        self.coastlines_check.setChecked(True)
        self.coastlines_check.toggled.connect(self.coastlines_toggled.emit)
        vector_layout.addWidget(self.coastlines_check)
        
        # Show grid
        self.grid_check = QCheckBox("Show Lat/Lon Grid")
        self.grid_check.setChecked(False)
        self.grid_check.toggled.connect(self.grid_toggled.emit)
        vector_layout.addWidget(self.grid_check)
        
        layout.addWidget(vector_group)
        
        # === TEMPORAL ===
        temporal_group = QGroupBox("Temporal Control")
        temporal_layout = QVBoxLayout(temporal_group)
        
        # Time slider
        time_header = QHBoxLayout()
        time_header.addWidget(QLabel("Time:"))
        self.time_value = QLabel("T+0h")
        time_header.addWidget(self.time_value)
        temporal_layout.addLayout(time_header)
        
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, 100)
        self.time_slider.setValue(0)
        self.time_slider.valueChanged.connect(self._on_time_changed)
        temporal_layout.addWidget(self.time_slider)
        
        # Playback controls
        playback_row = QHBoxLayout()
        
        self.step_back_btn = QPushButton("◀◀")
        self.step_back_btn.setFixedWidth(50)
        playback_row.addWidget(self.step_back_btn)
        
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setCheckable(True)
        self.play_btn.clicked.connect(self._on_play_clicked)
        playback_row.addWidget(self.play_btn)
        
        self.step_fwd_btn = QPushButton("▶▶")
        self.step_fwd_btn.setFixedWidth(50)
        playback_row.addWidget(self.step_fwd_btn)
        
        temporal_layout.addLayout(playback_row)
        
        layout.addWidget(temporal_group)
        
        # === DATA ===
        data_group = QGroupBox("Data")
        data_layout = QVBoxLayout(data_group)
        
        # Variable selector
        var_row = QHBoxLayout()
        var_row.addWidget(QLabel("Variable:"))
        self.variable_combo = QComboBox()
        self.variable_combo.addItems(["U-Wind", "V-Wind", "Wind Speed", "Temperature", "Pressure"])
        var_row.addWidget(self.variable_combo)
        data_layout.addLayout(var_row)
        
        # Level selector
        level_row = QHBoxLayout()
        level_row.addWidget(QLabel("Level:"))
        self.level_combo = QComboBox()
        self.level_combo.addItems(["10m AGL", "850 hPa", "500 hPa", "250 hPa", "200 hPa"])
        level_row.addWidget(self.level_combo)
        data_layout.addLayout(level_row)
        
        layout.addWidget(data_group)
        
        # Spacer
        layout.addStretch()
        
        # Export button
        self.export_btn = QPushButton("📷 Export Image")
        layout.addWidget(self.export_btn)
        
    def _on_opacity_changed(self, value: int):
        self.opacity_value.setText(f"{value}%")
        self.opacity_changed.emit(value / 100.0)
        
    def _on_time_changed(self, value: int):
        hours = int(value * 0.48)  # 0-48 hours range
        self.time_value.setText(f"T+{hours}h")
        self.temporal_changed.emit(value / 100.0)
        
    def _on_play_clicked(self, checked: bool):
        self.play_btn.setText("⏸ Pause" if checked else "▶ Play")
        self.play_toggled.emit(checked)


# =============================================================================
# MAIN VIEWPORT
# =============================================================================
class ProfessionalViewport(QWidget):
    """
    Main visualization viewport with NASA Blue Marble substrate.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create VisPy canvas
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            bgcolor='#0a0a14',
            parent=self,
            show=False
        )
        
        # Set up view
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.PanZoomCamera(aspect=1)
        
        # Layers
        self.base_image = None      # Blue Marble
        self.data_image = None      # Wind overlay
        self.streamlines = []       # Streamline visuals
        self.arrows = None          # Arrow visuals
        self.coastlines = None      # Coastline overlay
        
        # Renderers
        self.earth_renderer = EarthRenderer()
        self.earth_renderer.load()
        self.vector_renderer = VectorFieldRenderer()
        
        # State
        self._opacity = 0.5
        self._colormap = "plasma"
        self._vector_mode = "Streamlines"
        self._show_grid = False
        self._show_coastlines = True
        self._data_u = None
        self._data_v = None
        self._vmin = 0
        self._vmax = 1
        
        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas.native)
        
    def set_data(self, u: np.ndarray, v: np.ndarray):
        """Set wind field data and render."""
        self._data_u = u
        self._data_v = v
        self.render()
        
    def set_opacity(self, opacity: float):
        """Set overlay opacity."""
        self._opacity = opacity
        self.render()
        
    def set_colormap(self, cmap: str):
        """Set colormap."""
        self._colormap = cmap
        self.render()
        
    def set_vector_mode(self, mode: str):
        """Set vector display mode."""
        self._vector_mode = mode
        self.render()
        
    def set_grid_visible(self, visible: bool):
        """Toggle lat/lon grid visibility."""
        self._show_grid = visible
        self.render()
        
    def set_coastlines_visible(self, visible: bool):
        """Toggle coastlines visibility."""
        self._show_coastlines = visible
        self.render()
        
    def render(self):
        """Full render of all layers."""
        if self._data_u is None:
            return
            
        u, v = self._data_u, self._data_v
        h, w = u.shape
        
        # Clear existing
        self._clear_layers()
        
        # =====================================================================
        # LAYER 1: NASA BLUE MARBLE (BACK)
        # =====================================================================
        earth_texture = self.earth_renderer.get_texture(w, h)
        
        # =====================================================================
        # LAYER 2: WIND MAGNITUDE OVERLAY
        # =====================================================================
        magnitude = np.sqrt(u**2 + v**2)
        self._vmin = float(np.percentile(magnitude, 1))
        self._vmax = float(np.percentile(magnitude, 99))
        
        wind_overlay = Compositor.apply_colormap(
            magnitude, 
            self._colormap,
            vmin=self._vmin,
            vmax=self._vmax,
            alpha=self._opacity
        )
        
        # =====================================================================
        # COMPOSITE IN NUMPY (RELIABLE ALPHA BLENDING)
        # =====================================================================
        composited = Compositor.composite(earth_texture, wind_overlay)
        
        # Render composited image
        self.data_image = visuals.Image(
            composited,
            interpolation='bilinear',
            parent=self.view.scene
        )
        self.data_image.order = 100  # Back
        
        # =====================================================================
        # LAYER 3: VECTOR FIELD (FRONT)
        # =====================================================================
        if self._vector_mode in ["Streamlines", "Both"]:
            self._draw_streamlines(u, v, h, w)
            
        if self._vector_mode in ["Arrows", "Both"]:
            self._draw_arrows(u, v)
            
        # =====================================================================
        # LAYER 4: GRID (OPTIONAL)
        # =====================================================================
        if self._show_grid:
            self._draw_grid(h, w)
            
        # =====================================================================
        # LAYER 5: COASTLINES (FRONT)
        # =====================================================================
        if self._show_coastlines:
            self._draw_coastlines(h, w)
        
        # Set camera range
        self.view.camera.set_range(x=(0, w), y=(0, h))
        
        self.canvas.update()
        
    def _clear_layers(self):
        """Clear all visual layers."""
        if self.data_image is not None:
            self.data_image.parent = None
            self.data_image = None
            
        for sl in self.streamlines:
            sl.parent = None
        self.streamlines.clear()
        
        if self.arrows is not None:
            self.arrows.parent = None
            self.arrows = None
            
        if self.coastlines is not None:
            self.coastlines.parent = None
            self.coastlines = None
            
    def _draw_streamlines(self, u: np.ndarray, v: np.ndarray, h: int, w: int):
        """Draw streamlines with color-coded magnitude and arrow heads."""
        streamlines = self.vector_renderer.compute_streamlines(u, v, density=1.0)
        
        # Get colormap for magnitude coloring
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(self._colormap)
        
        for points, mags in streamlines:
            if len(points) < 5:
                continue
                
            # Color based on magnitude along line (use max for visibility)
            max_mag = np.max(mags)
            color = list(cmap(max_mag))
            color[3] = 0.85  # Slightly transparent
            
            # Create line visual
            line = visuals.Line(
                pos=points,
                color=color,
                width=1.8,
                antialias=True,
                parent=self.view.scene
            )
            line.order = 10  # Front
            self.streamlines.append(line)
            
            # Add arrow head at end of streamline
            if len(points) >= 3:
                end = points[-1]
                prev = points[-3]
                direction = end - prev
                length = np.linalg.norm(direction)
                if length > 0.1:
                    direction = direction / length
                    
                    # Triangle arrow head
                    head_size = 4.0
                    perp = np.array([-direction[1], direction[0]])
                    
                    p1 = end
                    p2 = end - direction * head_size + perp * head_size * 0.5
                    p3 = end - direction * head_size - perp * head_size * 0.5
                    
                    arrow_head = visuals.Markers(
                        pos=np.array([end]),
                        face_color=color,
                        edge_color=color,
                        size=6,
                        symbol='triangle_up',
                        parent=self.view.scene
                    )
                    arrow_head.order = 5
                    self.streamlines.append(arrow_head)
            
    def _draw_arrows(self, u: np.ndarray, v: np.ndarray):
        """Draw arrow glyphs."""
        positions, directions, magnitudes = self.vector_renderer.compute_arrows(
            u, v, spacing=25, scale=15.0
        )
        
        if len(positions) == 0:
            return
            
        # Get colormap
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(self._colormap)
        colors = cmap(magnitudes)
        
        # Create arrow shafts and heads
        all_lines = []
        all_colors = []
        head_positions = []
        head_colors = []
        
        for i, (pos, direction, mag) in enumerate(zip(positions, directions, magnitudes)):
            # Arrow shaft (slightly shortened to make room for head)
            end = pos + direction * 0.7
            all_lines.append([pos[0], pos[1]])
            all_lines.append([end[0], end[1]])
            all_colors.append(colors[i])
            all_colors.append(colors[i])
            
            # Arrow head position (at full length)
            head_pos = pos + direction
            head_positions.append(head_pos)
            head_colors.append(colors[i])
        
        if all_lines:
            # Draw shafts
            self.arrows = visuals.Line(
                pos=np.array(all_lines),
                color=np.array(all_colors),
                width=2.0,
                connect='segments',
                antialias=True,
                parent=self.view.scene
            )
            self.arrows.order = 5
            
            # Draw arrow heads as markers
            if head_positions:
                arrow_heads = visuals.Markers(
                    pos=np.array(head_positions),
                    face_color=np.array(head_colors),
                    edge_color=(1, 1, 1, 0.8),
                    edge_width=0.5,
                    size=8,
                    symbol='triangle_up',
                    parent=self.view.scene
                )
                arrow_heads.order = 4
                self.streamlines.append(arrow_heads)  # For cleanup
            
    def _draw_coastlines(self, h: int, w: int):
        """Draw coastline overlay."""
        try:
            from natural_earth import get_coastlines_pixels
            segments = get_coastlines_pixels(w, h)
            
            if segments:
                # Flatten segments for Line visual
                all_points = []
                connects = []
                
                for seg in segments:
                    start_idx = len(all_points)
                    for pt in seg:
                        all_points.append(pt)
                    # Connect consecutive points within segment
                    for i in range(start_idx, len(all_points) - 1):
                        connects.append([i, i + 1])
                
                if all_points:
                    self.coastlines = visuals.Line(
                        pos=np.array(all_points),
                        color=(1, 1, 1, 0.7),
                        width=1.0,
                        antialias=True,
                        parent=self.view.scene
                    )
                    self.coastlines.order = 2
        except ImportError:
            pass  # No coastlines available
            
    def _draw_grid(self, h: int, w: int):
        """Draw lat/lon grid overlay."""
        grid_lines = []
        
        # Longitude lines (every 30 degrees)
        for lon in range(-180, 181, 30):
            x = (lon + 180) / 360 * w
            grid_lines.append([[x, 0], [x, h]])
        
        # Latitude lines (every 30 degrees)
        for lat in range(-90, 91, 30):
            y = (90 - lat) / 180 * h
            grid_lines.append([[0, y], [w, y]])
        
        # Create line visuals
        for line in grid_lines:
            grid_visual = visuals.Line(
                pos=np.array(line),
                color=(1, 1, 1, 0.15),
                width=0.5,
                parent=self.view.scene
            )
            grid_visual.order = 80
            self.streamlines.append(grid_visual)  # Reuse streamlines list for cleanup
            
    def get_value_range(self) -> Tuple[float, float]:
        """Get current data value range."""
        return self._vmin, self._vmax


# =============================================================================
# MAIN WINDOW
# =============================================================================
class HyperTensorPro(QMainWindow):
    """
    Professional HyperTensor visualization application.
    """
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(*WINDOW_SIZE)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0a0a14;
            }
        """)
        
        self._setup_ui()
        self._connect_signals()
        self._load_demo_data()
        
        # Animation timer
        self._play_timer = QTimer()
        self._play_timer.timeout.connect(self._on_play_tick)
        
    def _setup_ui(self):
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # === VIEWPORT ===
        viewport_container = QWidget()
        viewport_layout = QVBoxLayout(viewport_container)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        
        self.viewport = ProfessionalViewport()
        viewport_layout.addWidget(self.viewport)
        
        # HUD overlay (positioned over viewport)
        self.hud = HUDOverlay(self.viewport)
        
        main_layout.addWidget(viewport_container, 1)
        
        # === CONTROL PANEL ===
        self.controls = ControlPanel()
        main_layout.addWidget(self.controls)
        
        # Status bar
        self.status = QStatusBar()
        self.status.setStyleSheet("""
            QStatusBar {
                background: #1a1a2e;
                color: #888;
                border-top: 1px solid #333;
            }
        """)
        self.setStatusBar(self.status)
        self.status.showMessage("Ready")
        
    def _connect_signals(self):
        # Control panel signals
        self.controls.opacity_changed.connect(self.viewport.set_opacity)
        self.controls.colormap_changed.connect(self._on_colormap_changed)
        self.controls.vector_mode_changed.connect(self.viewport.set_vector_mode)
        self.controls.temporal_changed.connect(self._on_temporal_changed)
        self.controls.play_toggled.connect(self._on_play_toggled)
        self.controls.grid_toggled.connect(self.viewport.set_grid_visible)
        self.controls.coastlines_toggled.connect(self.viewport.set_coastlines_visible)
        self.controls.export_btn.clicked.connect(self._on_export)
        
    def _on_colormap_changed(self, cmap: str):
        """Handle colormap change."""
        self.viewport.set_colormap(cmap)
        self.hud.update_colormap(cmap)
        
    def _load_demo_data(self):
        """Load demonstration wind field data with realistic weather patterns."""
        self.status.showMessage("Loading demonstration data...")
        
        # Generate realistic wind patterns
        h, w = 180, 360  # 1-degree resolution
        
        y = np.linspace(-90, 90, h)
        x = np.linspace(-180, 180, w)
        X, Y = np.meshgrid(x, y)
        
        # =====================================================================
        # BASE FLOW: Hadley cells + jet streams
        # =====================================================================
        # Trade winds (easterlies in tropics)
        u = -8 * np.cos(np.radians(Y * 6)) * np.exp(-Y**2 / 800)
        
        # Northern hemisphere jet stream (wavy pattern)
        jet_lat = 45 + 10 * np.sin(np.radians(X * 3))
        u += 35 * np.exp(-((Y - jet_lat)**2) / 100)
        
        # Southern hemisphere jet stream
        jet_lat_s = -50 + 5 * np.sin(np.radians(X * 4))
        u += 30 * np.exp(-((Y - jet_lat_s)**2) / 100)
        
        # Initialize v component
        v = np.zeros_like(u)
        
        # =====================================================================
        # CYCLONE 1: Major North Atlantic hurricane
        # =====================================================================
        cx1, cy1 = -50, 35  # Near Bermuda
        dist1 = np.sqrt((X - cx1)**2 + (Y - cy1)**2)
        r1 = 15  # Eye radius
        strength1 = 45 * np.exp(-dist1**2 / (r1**2 * 4))
        # Cyclonic rotation (counter-clockwise in NH)
        u -= strength1 * (Y - cy1) / (dist1 + 3)
        v += strength1 * (X - cx1) / (dist1 + 3)
        
        # =====================================================================
        # CYCLONE 2: Pacific storm
        # =====================================================================
        cx2, cy2 = 150, 42  # Near Japan
        dist2 = np.sqrt((X - cx2)**2 + (Y - cy2)**2)
        strength2 = 30 * np.exp(-dist2**2 / 300)
        u -= strength2 * (Y - cy2) / (dist2 + 3)
        v += strength2 * (X - cx2) / (dist2 + 3)
        
        # =====================================================================
        # CYCLONE 3: Southern hemisphere (clockwise rotation)
        # =====================================================================
        cx3, cy3 = 80, -35  # Indian Ocean
        dist3 = np.sqrt((X - cx3)**2 + (Y - cy3)**2)
        strength3 = 25 * np.exp(-dist3**2 / 250)
        # Clockwise in SH
        u += strength3 * (Y - cy3) / (dist3 + 3)
        v -= strength3 * (X - cx3) / (dist3 + 3)
        
        # =====================================================================
        # MONSOON: Asian summer monsoon flow
        # =====================================================================
        monsoon_region = ((X > 60) & (X < 120) & (Y > 5) & (Y < 30))
        v += np.where(monsoon_region, 8, 0)
        u += np.where(monsoon_region, -5, 0)
        
        # =====================================================================
        # POLAR VORTEX: Arctic circulation
        # =====================================================================
        polar_dist = np.sqrt(X**2 + (Y - 85)**2)
        polar_strength = 15 * np.exp(-polar_dist**2 / 1000)
        u -= polar_strength * (Y - 85) / (polar_dist + 5)
        v += polar_strength * X / (polar_dist + 5)
        
        # =====================================================================
        # Add realistic turbulence (noise)
        # =====================================================================
        np.random.seed(42)
        # Multi-scale noise
        from scipy.ndimage import gaussian_filter
        noise_large = gaussian_filter(np.random.randn(h, w), sigma=10)
        noise_medium = gaussian_filter(np.random.randn(h, w), sigma=5)
        noise_small = np.random.randn(h, w) * 0.5
        
        u += noise_large * 3 + noise_medium * 2 + noise_small
        v += noise_large * 2 + noise_medium * 1.5 + noise_small
        
        # Store data
        self._demo_u = u.astype(np.float32)
        self._demo_v = v.astype(np.float32)
        
        # Set viewport data
        self.viewport.set_data(self._demo_u, self._demo_v)
        
        # Update HUD
        vmin, vmax = self.viewport.get_value_range()
        self.hud.update_stats(vmin, vmax)
        self.hud.update_source("HyperTensor Demonstration")
        self.hud.update_variable("Wind Speed", "10m AGL")
        self.hud.update_time(datetime.now())
        
        self.status.showMessage("Demonstration data loaded")
        
    def _on_temporal_changed(self, t: float):
        """Handle temporal slider change with realistic storm motion."""
        from scipy.ndimage import gaussian_filter
        
        h, w = 180, 360
        y = np.linspace(-90, 90, h)
        x = np.linspace(-180, 180, w)
        X, Y = np.meshgrid(x, y)
        
        # Time offset for animation (storms move over 48 hours)
        hours = t * 48
        
        # =====================================================================
        # BASE FLOW
        # =====================================================================
        u = -8 * np.cos(np.radians(Y * 6)) * np.exp(-Y**2 / 800)
        jet_lat = 45 + 10 * np.sin(np.radians(X * 3))
        u += 35 * np.exp(-((Y - jet_lat)**2) / 100)
        jet_lat_s = -50 + 5 * np.sin(np.radians(X * 4))
        u += 30 * np.exp(-((Y - jet_lat_s)**2) / 100)
        v = np.zeros_like(u)
        
        # =====================================================================
        # MOVING CYCLONE 1: Atlantic hurricane tracking NE
        # =====================================================================
        cx1 = -50 + hours * 0.4  # Moves east
        cy1 = 35 + hours * 0.2   # Curves north
        dist1 = np.sqrt((X - cx1)**2 + (Y - cy1)**2)
        strength1 = 45 * np.exp(-dist1**2 / 225)
        u -= strength1 * (Y - cy1) / (dist1 + 3)
        v += strength1 * (X - cx1) / (dist1 + 3)
        
        # =====================================================================
        # MOVING CYCLONE 2: Pacific storm tracking NE
        # =====================================================================
        cx2 = 150 + hours * 0.3
        cy2 = 42 + hours * 0.15
        dist2 = np.sqrt((X - cx2)**2 + (Y - cy2)**2)
        strength2 = 30 * np.exp(-dist2**2 / 300)
        u -= strength2 * (Y - cy2) / (dist2 + 3)
        v += strength2 * (X - cx2) / (dist2 + 3)
        
        # =====================================================================
        # MOVING CYCLONE 3: Indian Ocean (tracks west)
        # =====================================================================
        cx3 = 80 - hours * 0.25
        cy3 = -35 + hours * 0.1
        dist3 = np.sqrt((X - cx3)**2 + (Y - cy3)**2)
        strength3 = 25 * np.exp(-dist3**2 / 250)
        u += strength3 * (Y - cy3) / (dist3 + 3)
        v -= strength3 * (X - cx3) / (dist3 + 3)
        
        # Monsoon (steady)
        monsoon_region = ((X > 60) & (X < 120) & (Y > 5) & (Y < 30))
        v += np.where(monsoon_region, 8, 0)
        u += np.where(monsoon_region, -5, 0)
        
        # Polar vortex
        polar_dist = np.sqrt(X**2 + (Y - 85)**2)
        polar_strength = 15 * np.exp(-polar_dist**2 / 1000)
        u -= polar_strength * (Y - 85) / (polar_dist + 5)
        v += polar_strength * X / (polar_dist + 5)
        
        # Turbulence (consistent with time)
        np.random.seed(42 + int(hours))
        noise_large = gaussian_filter(np.random.randn(h, w), sigma=10)
        noise_medium = gaussian_filter(np.random.randn(h, w), sigma=5)
        u += noise_large * 3 + noise_medium * 2
        v += noise_large * 2 + noise_medium * 1.5
        
        self.viewport.set_data(u.astype(np.float32), v.astype(np.float32))
        
        # Update HUD
        self.hud.update_forecast(int(hours))
        self.hud.update_time(datetime.now() + timedelta(hours=hours))
        
        vmin, vmax = self.viewport.get_value_range()
        self.hud.update_stats(vmin, vmax)
        
    def _on_play_toggled(self, playing: bool):
        """Handle play/pause toggle."""
        if playing:
            self._play_timer.start(100)  # 10 FPS
        else:
            self._play_timer.stop()
            
    def _on_play_tick(self):
        """Animation tick."""
        current = self.controls.time_slider.value()
        next_val = (current + 1) % 101
        self.controls.time_slider.setValue(next_val)
        
    def _on_export(self):
        """Export current visualization as PNG."""
        from PySide6.QtWidgets import QFileDialog
        from datetime import datetime
        
        # Default filename with timestamp
        default_name = f"hypertensor_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Visualization",
            default_name,
            "PNG Images (*.png);;JPEG Images (*.jpg);;All Files (*)"
        )
        
        if filename:
            try:
                # Capture VisPy canvas
                img = self.viewport.canvas.render()
                
                # Save with PIL
                from PIL import Image
                pil_img = Image.fromarray(img)
                pil_img.save(filename)
                
                self.status.showMessage(f"Exported: {filename}")
            except Exception as e:
                self.status.showMessage(f"Export failed: {e}")
        
    def resizeEvent(self, event):
        """Handle resize to update HUD position."""
        super().resizeEvent(event)
        # Resize HUD to match viewport
        self.hud.setGeometry(0, 0, self.viewport.width(), self.viewport.height())


# =============================================================================
# ENTRY POINT
# =============================================================================
def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║          HYPERTENSOR PROFESSIONAL VISUALIZATION                     ║
║                                                                      ║
║  • NASA Blue Marble 8K Satellite Imagery                             ║
║  • Professional Streamline/Arrow Rendering                           ║
║  • Perceptually Uniform Colormaps                                    ║
║  • Broadcast-Quality HUD Overlays                                    ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = HyperTensorPro()
    window.show()
    
    print("✅ Application launched successfully")
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
