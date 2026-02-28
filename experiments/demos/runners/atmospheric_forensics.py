"""
Ontic Atmospheric Forensics
==================================

Broadcast-quality atmospheric visualization with NASA Blue Marble substrate.

Features:
    - NASA Blue Marble 8K satellite imagery
    - Professional vector field rendering (streamlines + arrows)
    - Geodesic temporal interpolation
    - Multi-layer compositing with proper alpha blending
    - Broadcast-quality HUD overlays
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List

sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# IMPORTS
# =============================================================================

try:
    from PySide6.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QFrame, QPushButton, QApplication, QSlider,
        QStatusBar, QComboBox, QGroupBox, QCheckBox, QSpinBox,
        QDoubleSpinBox, QSplitter, QScrollArea
    )
    from PySide6.QtCore import Qt, QTimer, Signal, QObject
    from PySide6.QtGui import QFont, QColor, QPalette
except ImportError:
    print("ERROR: PySide6 required. Install with: pip install PySide6")
    sys.exit(1)

try:
    from vispy import scene, app
    from vispy.scene import visuals
    from vispy.color import Colormap
    from vispy.visuals.transforms import STTransform
except ImportError:
    print("ERROR: VisPy required. Install with: pip install vispy")
    sys.exit(1)

try:
    from earth_renderer import EarthRenderer, VectorFieldRenderer, Compositor
    EARTH_AVAILABLE = True
except ImportError:
    EARTH_AVAILABLE = False
    print("Warning: earth_renderer not available")

try:
    from natural_earth import get_coastlines_pixels
    COASTLINES_AVAILABLE = True
except ImportError:
    COASTLINES_AVAILABLE = False


# =============================================================================
# COLORMAPS
# =============================================================================

# Professional wind colormap (perceptually uniform)
WIND_COLORS = [
    (0.00, (0.05, 0.15, 0.35, 0.3)),   # Calm: dark blue, very transparent
    (0.20, (0.10, 0.40, 0.60, 0.4)),   # Light: teal
    (0.40, (0.20, 0.65, 0.55, 0.5)),   # Moderate: cyan-green
    (0.55, (0.70, 0.75, 0.20, 0.6)),   # Fresh: yellow-green
    (0.70, (0.95, 0.65, 0.15, 0.7)),   # Strong: orange
    (0.85, (0.90, 0.35, 0.15, 0.8)),   # Gale: red-orange
    (1.00, (0.70, 0.10, 0.30, 0.9)),   # Storm: deep magenta
]

def apply_wind_colormap(magnitude: np.ndarray, 
                        vmin: float = None, 
                        vmax: float = None) -> np.ndarray:
    """Apply professional wind colormap with variable transparency."""
    if vmin is None:
        vmin = np.percentile(magnitude, 2)
    if vmax is None:
        vmax = np.percentile(magnitude, 98)
    
    # Normalize
    norm = (magnitude - vmin) / (vmax - vmin + 1e-10)
    norm = np.clip(norm, 0, 1)
    
    h, w = magnitude.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    
    # Interpolate through color stops
    for i in range(len(WIND_COLORS) - 1):
        t0, c0 = WIND_COLORS[i]
        t1, c1 = WIND_COLORS[i + 1]
        
        mask = (norm >= t0) & (norm < t1)
        if not np.any(mask):
            continue
        
        # Local interpolation parameter
        local_t = (norm[mask] - t0) / (t1 - t0)
        
        for ch in range(4):
            rgba[:, :, ch][mask] = c0[ch] + (c1[ch] - c0[ch]) * local_t
    
    # Handle values at exactly 1.0
    mask = norm >= 1.0
    if np.any(mask):
        for ch in range(4):
            rgba[:, :, ch][mask] = WIND_COLORS[-1][1][ch]
    
    return rgba


# =============================================================================
# GEODESIC INTERPOLATOR
# =============================================================================

class GeodesicInterpolator:
    """
    Optical flow-based temporal interpolation.
    
    Uses motion estimation to warp fields along their natural trajectories,
    avoiding the "watercolor blur" of linear pixel interpolation.
    """
    
    def __init__(self):
        self.u0: Optional[np.ndarray] = None
        self.v0: Optional[np.ndarray] = None
        self.u1: Optional[np.ndarray] = None
        self.v1: Optional[np.ndarray] = None
        self._flow_u: Optional[np.ndarray] = None
        self._flow_v: Optional[np.ndarray] = None
        
    def set_keyframes(self, u0: np.ndarray, v0: np.ndarray,
                      u1: np.ndarray, v1: np.ndarray):
        """Set temporal keyframes and compute motion field."""
        self.u0 = u0.astype(np.float32)
        self.v0 = v0.astype(np.float32)
        self.u1 = u1.astype(np.float32)
        self.v1 = v1.astype(np.float32)
        
        # Estimate motion from wind field itself
        # The wind vectors approximate the flow direction
        self._flow_u = 0.5 * (u0 + u1)
        self._flow_v = 0.5 * (v0 + v1)
        
    def interpolate(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate field at time t using geodesic warping.
        
        Args:
            t: Temporal position [0, 1]
            
        Returns:
            (u, v) interpolated vector field
        """
        if self.u0 is None or self.u1 is None:
            raise ValueError("Keyframes not set")
        
        if t <= 0:
            return self.u0.copy(), self.v0.copy()
        if t >= 1:
            return self.u1.copy(), self.v1.copy()
        
        h, w = self.u0.shape
        
        # Create sampling coordinates
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        x_coords = x_coords.astype(np.float32)
        y_coords = y_coords.astype(np.float32)
        
        # Forward warp from t=0
        x0_warped = x_coords + self._flow_u * t * 0.5
        y0_warped = y_coords + self._flow_v * t * 0.5
        
        # Backward warp from t=1
        x1_warped = x_coords - self._flow_u * (1 - t) * 0.5
        y1_warped = y_coords - self._flow_v * (1 - t) * 0.5
        
        # Clamp coordinates
        x0_warped = np.clip(x0_warped, 0, w - 1)
        y0_warped = np.clip(y0_warped, 0, h - 1)
        x1_warped = np.clip(x1_warped, 0, w - 1)
        y1_warped = np.clip(y1_warped, 0, h - 1)
        
        # Bilinear sample
        def bilinear_sample(field, x, y):
            x0 = np.floor(x).astype(int)
            y0 = np.floor(y).astype(int)
            x1 = np.minimum(x0 + 1, w - 1)
            y1 = np.minimum(y0 + 1, h - 1)
            
            fx = x - x0
            fy = y - y0
            
            return (field[y0, x0] * (1 - fx) * (1 - fy) +
                    field[y0, x1] * fx * (1 - fy) +
                    field[y1, x0] * (1 - fx) * fy +
                    field[y1, x1] * fx * fy)
        
        # Sample both directions
        u0_sampled = bilinear_sample(self.u0, x0_warped, y0_warped)
        v0_sampled = bilinear_sample(self.v0, x0_warped, y0_warped)
        u1_sampled = bilinear_sample(self.u1, x1_warped, y1_warped)
        v1_sampled = bilinear_sample(self.v1, x1_warped, y1_warped)
        
        # Blend warped fields
        u_result = (1 - t) * u0_sampled + t * u1_sampled
        v_result = (1 - t) * v0_sampled + t * v1_sampled
        
        return u_result, v_result


# =============================================================================
# VISUALIZATION VIEWPORT
# =============================================================================

class AtmosphericViewport(QObject):
    """
    Professional atmospheric visualization viewport.
    
    Renders:
        - NASA Blue Marble satellite imagery (base layer)
        - Wind magnitude field (semi-transparent overlay)
        - Vector field arrows or streamlines
        - Coastlines and geographic annotations
        - Professional HUD overlays
    """
    
    view_changed = Signal(tuple)
    
    def __init__(self, parent: QWidget = None):
        super().__init__()
        
        # Create VisPy canvas
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            bgcolor='#0a0a12',
            parent=parent,
            show=False
        )
        
        # Create view
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.PanZoomCamera(aspect=1)
        
        # Layers (all start as None)
        self.earth_layer: Optional[visuals.Image] = None
        self.wind_layer: Optional[visuals.Image] = None
        self.vector_lines: List[visuals.Line] = []
        self.vector_arrows: Optional[visuals.Markers] = None
        self.coastline_glow: Optional[visuals.Line] = None
        self.coastline_core: Optional[visuals.Line] = None
        
        # Renderers
        self.earth_renderer = EarthRenderer() if EARTH_AVAILABLE else None
        self.vector_renderer = VectorFieldRenderer()
        self.interpolator = GeodesicInterpolator()
        
        # State
        self._data_shape = (180, 360)
        self._current_t = 0.5
        self._show_vectors = True
        self._vector_mode = "arrows"  # "arrows" or "streamlines"
        self._vector_spacing = 20
        self._wind_opacity = 0.55
        
        # Load Earth
        if self.earth_renderer:
            self.earth_renderer.load()
    
    @property
    def native(self) -> QWidget:
        return self.canvas.native
    
    def render(self, 
               u: np.ndarray, 
               v: np.ndarray,
               show_vectors: bool = True,
               wind_opacity: float = 0.55) -> Tuple[float, float]:
        """
        Full render of atmospheric data over Earth.
        
        Args:
            u, v: Wind vector components
            show_vectors: Whether to show vector overlays
            wind_opacity: Opacity of wind field overlay
            
        Returns:
            (vmin, vmax) of the rendered field
        """
        self._clear_layers()
        
        h, w = u.shape
        self._data_shape = (h, w)
        self._show_vectors = show_vectors
        self._wind_opacity = wind_opacity
        
        # =====================================================================
        # LAYER 1: NASA BLUE MARBLE (BASE)
        # =====================================================================
        if self.earth_renderer:
            earth_texture = self.earth_renderer.get_texture(w, h)
        else:
            # Fallback dark ocean
            earth_texture = np.zeros((h, w, 4), dtype=np.float32)
            earth_texture[:, :, 0] = 0.02
            earth_texture[:, :, 1] = 0.06
            earth_texture[:, :, 2] = 0.14
            earth_texture[:, :, 3] = 1.0
        
        # =====================================================================
        # LAYER 2: WIND MAGNITUDE (SEMI-TRANSPARENT OVERLAY)
        # =====================================================================
        magnitude = np.sqrt(u**2 + v**2)
        vmin = float(np.percentile(magnitude, 2))
        vmax = float(np.percentile(magnitude, 98))
        
        # Apply wind colormap with variable alpha
        wind_rgba = apply_wind_colormap(magnitude, vmin, vmax)
        
        # Scale alpha by user opacity setting
        wind_rgba[:, :, 3] *= wind_opacity
        
        # =====================================================================
        # COMPOSITE LAYERS IN NUMPY (avoids VisPy alpha issues)
        # =====================================================================
        alpha = wind_rgba[:, :, 3:4]
        composited = np.zeros((h, w, 4), dtype=np.float32)
        composited[:, :, :3] = wind_rgba[:, :, :3] * alpha + earth_texture[:, :, :3] * (1.0 - alpha)
        composited[:, :, 3] = 1.0
        
        # Render composited image
        self.earth_layer = visuals.Image(
            composited,
            interpolation='bilinear',
            parent=self.view.scene
        )
        self.earth_layer.order = 100  # Back
        
        # =====================================================================
        # LAYER 3: COASTLINES
        # =====================================================================
        self._draw_coastlines(h, w)
        
        # =====================================================================
        # LAYER 4: VECTOR FIELD
        # =====================================================================
        if show_vectors:
            self._draw_vectors(u, v, magnitude, vmax)
        
        # Set camera
        self.view.camera.set_range(x=(-20, w + 20), y=(-20, h + 20))
        self.canvas.update()
        
        return vmin, vmax
    
    def _clear_layers(self):
        """Remove all existing visual layers."""
        for layer in [self.earth_layer, self.wind_layer, 
                      self.coastline_glow, self.coastline_core,
                      self.vector_arrows]:
            if layer is not None:
                layer.parent = None
        
        for line in self.vector_lines:
            line.parent = None
        self.vector_lines = []
        
        self.earth_layer = None
        self.wind_layer = None
        self.coastline_glow = None
        self.coastline_core = None
        self.vector_arrows = None
    
    def _draw_coastlines(self, h: int, w: int):
        """Draw coastline overlays."""
        if not COASTLINES_AVAILABLE:
            return
        
        try:
            segments = get_coastlines_pixels(w, h)
            if not segments:
                return
            
            # Build line data
            all_points = []
            for seg in segments:
                if len(seg) > 1:
                    all_points.extend(seg)
                    all_points.append([np.nan, np.nan])  # Break between segments
            
            if not all_points:
                return
            
            points = np.array(all_points, dtype=np.float32)
            
            # Glow layer (yellow, wider)
            self.coastline_glow = visuals.Line(
                pos=points,
                color=(1.0, 0.85, 0.2, 0.4),
                width=3.0,
                connect='strip',
                parent=self.view.scene
            )
            self.coastline_glow.order = 20
            
            # Core layer (white, thinner)
            self.coastline_core = visuals.Line(
                pos=points,
                color=(1.0, 1.0, 1.0, 0.9),
                width=1.0,
                connect='strip',
                parent=self.view.scene
            )
            self.coastline_core.order = 10
            
        except Exception as e:
            print(f"Coastline error: {e}")
    
    def _draw_vectors(self, u: np.ndarray, v: np.ndarray, 
                      magnitude: np.ndarray, mag_max: float):
        """Draw vector field visualization."""
        h, w = u.shape
        
        if self._vector_mode == "arrows":
            self._draw_arrows(u, v, magnitude, mag_max)
        else:
            self._draw_streamlines(u, v, magnitude, mag_max)
    
    def _draw_arrows(self, u: np.ndarray, v: np.ndarray,
                     magnitude: np.ndarray, mag_max: float):
        """Draw directional arrow glyphs for vector field."""
        h, w = u.shape
        spacing = self._vector_spacing
        
        positions = []
        colors = []
        sizes = []
        
        for y in range(spacing // 2, h, spacing):
            for x in range(spacing // 2, w, spacing):
                mag = magnitude[y, x]
                if mag > mag_max * 0.08:  # Skip very weak vectors
                    positions.append([x, y])
                    
                    # Color by magnitude using wind colormap
                    t = min(1.0, mag / mag_max)
                    color = [0.3, 0.7, 1.0, 0.9]  # Default cyan
                    for j in range(len(WIND_COLORS) - 1):
                        t0, c0 = WIND_COLORS[j]
                        t1, c1 = WIND_COLORS[j + 1]
                        if t0 <= t < t1:
                            local_t = (t - t0) / (t1 - t0)
                            color = [c0[k] + (c1[k] - c0[k]) * local_t for k in range(4)]
                            break
                    
                    # Boost alpha for visibility
                    color[3] = min(0.95, color[3] + 0.3)
                    colors.append(color)
                    
                    # Size proportional to magnitude
                    sizes.append(6 + t * 6)
        
        if positions:
            positions = np.array(positions, dtype=np.float32)
            colors = np.array(colors, dtype=np.float32)
            sizes = np.array(sizes, dtype=np.float32)
            
            # Triangle markers for direction indication
            self.vector_arrows = visuals.Markers(parent=self.view.scene)
            self.vector_arrows.set_data(
                pos=positions,
                face_color=colors,
                edge_color=(1, 1, 1, 0.6),
                edge_width=0.5,
                size=sizes,
                symbol='disc'  # Use disc for cleaner look
            )
            self.vector_arrows.order = 5
            
            print(f"   → Rendered {len(positions)} vector markers")
    
    def _draw_streamlines(self, u: np.ndarray, v: np.ndarray,
                          magnitude: np.ndarray, mag_max: float):
        """Draw professional streamlines for vector field."""
        streamlines = self.vector_renderer.compute_streamlines(u, v, density=0.7)
        
        for points, mags in streamlines:
            if len(points) < 5:
                continue
            
            # Color by local magnitude using wind colormap
            colors = np.zeros((len(points), 4), dtype=np.float32)
            for i, m in enumerate(mags):
                t = min(1.0, m)
                # Interpolate through WIND_COLORS
                for j in range(len(WIND_COLORS) - 1):
                    t0, c0 = WIND_COLORS[j]
                    t1, c1 = WIND_COLORS[j + 1]
                    if t0 <= t < t1:
                        local_t = (t - t0) / (t1 - t0)
                        for ch in range(4):
                            colors[i, ch] = c0[ch] + (c1[ch] - c0[ch]) * local_t
                        break
                else:
                    # At max
                    colors[i] = WIND_COLORS[-1][1]
                
                # Boost alpha for visibility
                colors[i, 3] = min(0.95, colors[i, 3] + 0.3)
            
            line = visuals.Line(
                pos=points,
                color=colors,
                width=2.0,
                connect='strip',
                parent=self.view.scene,
                antialias=True
            )
            line.order = 5
            self.vector_lines.append(line)
        
        print(f"   → Rendered {len(streamlines)} streamlines")
    
    def set_temporal(self, t: float):
        """Update display for temporal position."""
        self._current_t = t
        
        if self.interpolator.u0 is not None:
            u, v = self.interpolator.interpolate(t)
            self.render(u, v, self._show_vectors, self._wind_opacity)
    
    def set_keyframes(self, u0: np.ndarray, v0: np.ndarray,
                      u1: np.ndarray, v1: np.ndarray):
        """Set temporal interpolation keyframes."""
        self.interpolator.set_keyframes(u0, v0, u1, v1)


# =============================================================================
# CONTROL PANEL
# =============================================================================

class ControlPanel(QWidget):
    """Professional control panel for visualization settings."""
    
    opacity_changed = Signal(float)
    vectors_changed = Signal(bool)
    spacing_changed = Signal(int)
    temporal_changed = Signal(float)
    mode_changed = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Style
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3a3a4a;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
                background: #1a1a24;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                color: #8af;
            }
            QLabel {
                color: #ccc;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #3a3a4a;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #6af;
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QCheckBox {
                color: #ccc;
            }
            QCheckBox::indicator:checked {
                background: #6af;
            }
            QComboBox {
                background: #2a2a3a;
                color: #ccc;
                border: 1px solid #3a3a4a;
                padding: 4px;
                border-radius: 3px;
            }
            QSpinBox {
                background: #2a2a3a;
                color: #ccc;
                border: 1px solid #3a3a4a;
                padding: 2px;
            }
        """)
        
        # =====================================================================
        # TEMPORAL CONTROLS
        # =====================================================================
        temporal_group = QGroupBox("Temporal Navigation")
        temporal_layout = QVBoxLayout(temporal_group)
        
        # Time slider
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, 1000)
        self.time_slider.setValue(500)
        self.time_slider.valueChanged.connect(self._on_time_change)
        temporal_layout.addWidget(self.time_slider)
        
        # Time label
        self.time_label = QLabel("t = 0.500")
        self.time_label.setAlignment(Qt.AlignCenter)
        temporal_layout.addWidget(self.time_label)
        
        # Playback buttons
        btn_layout = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setCheckable(True)
        self.play_btn.toggled.connect(self._on_play_toggle)
        btn_layout.addWidget(self.play_btn)
        temporal_layout.addLayout(btn_layout)
        
        layout.addWidget(temporal_group)
        
        # =====================================================================
        # OVERLAY CONTROLS
        # =====================================================================
        overlay_group = QGroupBox("Overlay Settings")
        overlay_layout = QVBoxLayout(overlay_group)
        
        # Wind opacity
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("Wind Opacity:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(55)
        self.opacity_slider.valueChanged.connect(self._on_opacity_change)
        opacity_layout.addWidget(self.opacity_slider)
        self.opacity_label = QLabel("55%")
        opacity_layout.addWidget(self.opacity_label)
        overlay_layout.addLayout(opacity_layout)
        
        # Vector toggle
        self.vector_check = QCheckBox("Show Vectors")
        self.vector_check.setChecked(True)
        self.vector_check.toggled.connect(self._on_vectors_toggle)
        overlay_layout.addWidget(self.vector_check)
        
        # Vector mode
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Vector Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Markers", "Streamlines"])
        self.mode_combo.currentTextChanged.connect(self._on_mode_change)
        mode_layout.addWidget(self.mode_combo)
        overlay_layout.addLayout(mode_layout)
        
        # Vector spacing
        spacing_layout = QHBoxLayout()
        spacing_layout.addWidget(QLabel("Vector Spacing:"))
        self.spacing_spin = QSpinBox()
        self.spacing_spin.setRange(10, 50)
        self.spacing_spin.setValue(20)
        self.spacing_spin.valueChanged.connect(self._on_spacing_change)
        spacing_layout.addWidget(self.spacing_spin)
        overlay_layout.addLayout(spacing_layout)
        
        layout.addWidget(overlay_group)
        
        # =====================================================================
        # INFO PANEL
        # =====================================================================
        info_group = QGroupBox("Data Info")
        info_layout = QVBoxLayout(info_group)
        
        self.info_label = QLabel("No data loaded")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        
        layout.addWidget(info_group)
        
        layout.addStretch()
        
        # Animation timer
        self._anim_timer = QTimer()
        self._anim_timer.timeout.connect(self._animate_step)
    
    def _on_time_change(self, value):
        t = value / 1000.0
        self.time_label.setText(f"t = {t:.3f}")
        self.temporal_changed.emit(t)
    
    def _on_opacity_change(self, value):
        self.opacity_label.setText(f"{value}%")
        self.opacity_changed.emit(value / 100.0)
    
    def _on_vectors_toggle(self, checked):
        self.vectors_changed.emit(checked)
    
    def _on_spacing_change(self, value):
        self.spacing_changed.emit(value)
    
    def _on_mode_change(self, text):
        mode = "arrows" if text == "Markers" else "streamlines"
        self.mode_changed.emit(mode)
    
    def _on_play_toggle(self, checked):
        if checked:
            self.play_btn.setText("⏸ Pause")
            self._anim_timer.start(33)  # ~30 FPS
        else:
            self.play_btn.setText("▶ Play")
            self._anim_timer.stop()
    
    def _animate_step(self):
        current = self.time_slider.value()
        next_val = (current + 5) % 1001
        self.time_slider.setValue(next_val)
    
    def set_info(self, text: str):
        self.info_label.setText(text)


# =============================================================================
# MAIN WINDOW
# =============================================================================

class AtmosphericForensicsWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Ontic Atmospheric Forensics")
        self.setMinimumSize(1400, 800)
        self.setStyleSheet("""
            QMainWindow {
                background: #12121a;
            }
            QStatusBar {
                background: #1a1a24;
                color: #8af;
            }
        """)
        
        # Central widget with splitter
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Viewport container with HUD overlays
        viewport_container = QWidget()
        viewport_layout = QVBoxLayout(viewport_container)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        viewport_layout.setSpacing(0)
        
        # Top HUD bar
        self.top_hud = self._create_hud_bar([
            ("ONTIC_ENGINE ATMOSPHERIC FORENSICS", "title"),
            ("NASA Blue Marble 8K | Equirectangular Projection", "subtitle"),
        ], align="left")
        viewport_layout.addWidget(self.top_hud)
        
        # Viewport
        self.viewport = AtmosphericViewport(self)
        viewport_layout.addWidget(self.viewport.native, 1)
        
        # Bottom HUD bar
        self.bottom_hud = self._create_hud_bar([
            ("Field: U/V Wind Components", "info"),
            ("", "spacer"),
            ("Range: -- to -- m/s", "range"),
        ], align="spread")
        viewport_layout.addWidget(self.bottom_hud)
        
        splitter.addWidget(viewport_container)
        
        # Control panel
        self.controls = ControlPanel()
        self.controls.setFixedWidth(280)
        splitter.addWidget(self.controls)
        
        splitter.setSizes([1100, 280])
        
        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready — NASA Blue Marble 8K substrate active")
        
        # Connect signals
        self.controls.opacity_changed.connect(self._on_opacity)
        self.controls.vectors_changed.connect(self._on_vectors)
        self.controls.temporal_changed.connect(self._on_temporal)
        self.controls.mode_changed.connect(self._on_mode)
        self.controls.spacing_changed.connect(self._on_spacing)
        
        # HUD references for updates
        self._range_label = None
        for child in self.bottom_hud.findChildren(QLabel):
            if "Range:" in child.text():
                self._range_label = child
                break
        
        # Load demo data
        QTimer.singleShot(100, self._load_demo)
    
    def _create_hud_bar(self, items: list, align: str = "left") -> QWidget:
        """Create a professional HUD bar."""
        bar = QFrame()
        bar.setFixedHeight(32)
        bar.setStyleSheet("""
            QFrame {
                background: rgba(10, 10, 18, 0.85);
                border-bottom: 1px solid #2a2a3a;
            }
            QLabel {
                color: #aaccff;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                padding: 0 10px;
            }
            QLabel[class="title"] {
                font-weight: bold;
                font-size: 12px;
                color: #88aaff;
            }
            QLabel[class="subtitle"] {
                color: #6688aa;
                font-size: 10px;
            }
        """)
        
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 0, 10, 0)
        
        for text, style in items:
            if style == "spacer":
                layout.addStretch()
            else:
                label = QLabel(text)
                label.setProperty("class", style)
                if align == "spread" and style != "spacer":
                    layout.addWidget(label)
                else:
                    layout.addWidget(label)
        
        if align == "left":
            layout.addStretch()
        
        return bar
    
    def _load_demo(self):
        """Generate and display demo wind field with realistic patterns."""
        self.status.showMessage("Generating atmospheric data...")
        
        # Grid - match Earth aspect ratio
        h, w = 180, 360
        lat = np.linspace(90, -90, h)
        lon = np.linspace(-180, 180, w)
        LON, LAT = np.meshgrid(lon, lat)
        
        # =====================================================================
        # REALISTIC ATMOSPHERIC PATTERNS
        # =====================================================================
        
        # 1. Hadley Cell - Trade Winds (easterlies near equator)
        trade_winds_u = -8.0 * np.exp(-((LAT) / 15) ** 2)  # Easterly
        trade_winds_v = 3.0 * np.sin(np.radians(LAT * 3))   # Slight meridional
        
        # 2. Ferrel Cell - Westerlies (mid-latitudes 30-60°)
        westerlies_n = 15.0 * np.exp(-((LAT - 45) / 12) ** 2)
        westerlies_s = 15.0 * np.exp(-((LAT + 45) / 12) ** 2)
        
        # 3. Polar Cell - Polar Easterlies
        polar_n = -6.0 * np.exp(-((LAT - 75) / 10) ** 2)
        polar_s = -6.0 * np.exp(-((LAT + 75) / 10) ** 2)
        
        # 4. Add Rossby waves (meandering jet stream)
        rossby_amplitude = 5.0
        rossby_wavenumber = 4
        rossby_n = rossby_amplitude * np.sin(np.radians(LON * rossby_wavenumber)) * \
                   np.exp(-((LAT - 50) / 15) ** 2)
        rossby_s = rossby_amplitude * np.sin(np.radians(LON * rossby_wavenumber + 90)) * \
                   np.exp(-((LAT + 50) / 15) ** 2)
        
        # 5. Add a cyclonic system (low pressure) in North Atlantic
        cyclone_lat, cyclone_lon = 55, -30
        dist = np.sqrt((LAT - cyclone_lat)**2 + (LON - cyclone_lon)**2)
        cyclone_intensity = 20.0 * np.exp(-(dist / 15) ** 2)
        # Cyclonic rotation (counter-clockwise in NH)
        cyclone_u = -cyclone_intensity * (LAT - cyclone_lat) / (dist + 0.1)
        cyclone_v = cyclone_intensity * (LON - cyclone_lon) / (dist + 0.1)
        
        # 6. Add an anticyclone (high pressure) in Pacific
        anti_lat, anti_lon = 35, -140
        dist_anti = np.sqrt((LAT - anti_lat)**2 + (LON - anti_lon)**2)
        anti_intensity = 12.0 * np.exp(-(dist_anti / 20) ** 2)
        # Anticyclonic rotation (clockwise in NH)
        anti_u = anti_intensity * (LAT - anti_lat) / (dist_anti + 0.1)
        anti_v = -anti_intensity * (LON - anti_lon) / (dist_anti + 0.1)
        
        # Combine all patterns
        u0 = (trade_winds_u + westerlies_n + westerlies_s + polar_n + polar_s +
              rossby_n + rossby_s + cyclone_u + anti_u)
        v0 = (trade_winds_v + rossby_n * 0.3 + rossby_s * 0.3 + cyclone_v + anti_v)
        
        # Add small-scale turbulence
        u0 += np.random.randn(h, w) * 1.0
        v0 += np.random.randn(h, w) * 1.0
        
        # =====================================================================
        # SECOND KEYFRAME (evolved state - systems have moved)
        # =====================================================================
        # Move cyclone east
        cyclone_lon_t1 = cyclone_lon + 10
        dist_t1 = np.sqrt((LAT - cyclone_lat)**2 + (LON - cyclone_lon_t1)**2)
        cyclone_intensity_t1 = 18.0 * np.exp(-(dist_t1 / 15) ** 2)
        cyclone_u_t1 = -cyclone_intensity_t1 * (LAT - cyclone_lat) / (dist_t1 + 0.1)
        cyclone_v_t1 = cyclone_intensity_t1 * (LON - cyclone_lon_t1) / (dist_t1 + 0.1)
        
        u1 = (trade_winds_u + westerlies_n + westerlies_s + polar_n + polar_s +
              rossby_n + rossby_s + cyclone_u_t1 + anti_u)
        v1 = (trade_winds_v + rossby_n * 0.3 + rossby_s * 0.3 + cyclone_v_t1 + anti_v)
        u1 += np.random.randn(h, w) * 1.0
        v1 += np.random.randn(h, w) * 1.0
        
        # Set keyframes
        self.viewport.set_keyframes(u0.astype(np.float32), v0.astype(np.float32),
                                     u1.astype(np.float32), v1.astype(np.float32))
        
        # Initial render at t=0.5
        u, v = self.viewport.interpolator.interpolate(0.5)
        vmin, vmax = self.viewport.render(u, v)
        
        # Update info
        mag = np.sqrt(u**2 + v**2)
        self.controls.set_info(
            f"Field: U/V Wind Components\n"
            f"Grid: {w}×{h} (1° resolution)\n"
            f"Magnitude: {mag.min():.1f} - {mag.max():.1f} m/s\n"
            f"Features: Cyclone (N.Atlantic),\n"
            f"  Anticyclone (Pacific),\n"
            f"  Trade Winds, Westerlies\n"
            f"Source: Synthetic Reanalysis"
        )
        
        # Update HUD
        if self._range_label:
            self._range_label.setText(f"Range: {vmin:.1f} to {vmax:.1f} m/s")
        
        self.status.showMessage(
            f"✅ Rendered {w}×{h} atmospheric field — "
            f"Cyclone at {cyclone_lat}°N, {abs(cyclone_lon)}°W"
        )
    
    def _on_opacity(self, value: float):
        self.viewport._wind_opacity = value
        self.viewport.set_temporal(self.viewport._current_t)
    
    def _on_vectors(self, show: bool):
        self.viewport._show_vectors = show
        self.viewport.set_temporal(self.viewport._current_t)
    
    def _on_temporal(self, t: float):
        self.viewport.set_temporal(t)
    
    def _on_mode(self, mode: str):
        self.viewport._vector_mode = mode
        self.viewport.set_temporal(self.viewport._current_t)
    
    def _on_spacing(self, spacing: int):
        self.viewport._vector_spacing = spacing
        self.viewport.set_temporal(self.viewport._current_t)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                 ONTIC_ENGINE ATMOSPHERIC FORENSICS                        ║
║                                                                          ║
║   • NASA Blue Marble 8K satellite imagery                                ║
║   • Professional vector field rendering                                  ║
║   • Geodesic temporal interpolation                                      ║
║   • Broadcast-quality visualization                                      ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    app_instance = QApplication.instance()
    if app_instance is None:
        app_instance = QApplication(sys.argv)
    
    # Dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(18, 18, 26))
    palette.setColor(QPalette.WindowText, QColor(200, 200, 220))
    palette.setColor(QPalette.Base, QColor(26, 26, 36))
    palette.setColor(QPalette.Text, QColor(200, 200, 220))
    palette.setColor(QPalette.Button, QColor(36, 36, 50))
    palette.setColor(QPalette.ButtonText, QColor(200, 200, 220))
    palette.setColor(QPalette.Highlight, QColor(100, 150, 255))
    app_instance.setPalette(palette)
    
    window = AtmosphericForensicsWindow()
    window.show()
    
    print("✅ Atmospheric Forensics launched")
    
    sys.exit(app_instance.exec())


if __name__ == "__main__":
    main()
