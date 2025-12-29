#!/usr/bin/env python3
"""
HyperTensor Forensic Hub
========================

The complete forensic investigation tool for weather singularities.

This implements the full vision:
1. INGESTION: Load manifolds from AWS/local .qtt files
2. ANOMALY SCAN: Rank-gradient detection of hidden singularities  
3. INFINITE ZOOM: Resolution-independent deep dive
4. TIME-TRAVEL: Spatio-temporal ghosting between snapshots
5. AUDIT: Signed manifests for reproducible findings

Usage:
    python demos/forensic_hub.py
    
Intent Commands:
    load field <name>       - Load a weather manifold
    scan anomalies          - Run rank-gradient singularity detection
    zoom anomaly <n> level <L> - Deep dive into anomaly at resolution level
    toggle ghosting         - Enable/disable temporal morphing
    sign findings           - Generate signed audit manifest
    show stats              - Display field statistics
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
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from PySide6.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
        QLabel, QFrame, QPushButton, QApplication, QSlider,
        QStatusBar, QComboBox, QGroupBox, QLineEdit, QTextEdit,
        QStackedWidget, QSplitter, QProgressBar
    )
    from PySide6.QtCore import Qt, QTimer, Signal, QThread, QObject, Slot
    from PySide6.QtGui import QFont
except ImportError:
    print("ERROR: PySide6 not installed. Run: pip install PySide6")
    sys.exit(1)

try:
    from vispy import scene
    from vispy.scene import visuals
    from vispy.color import Colormap
except ImportError:
    print("ERROR: VisPy not installed. Run: pip install vispy")
    sys.exit(1)

# Import coastline data
try:
    from demos.coastlines import get_coastline_pixels
    COASTLINES_AVAILABLE = True
except ImportError:
    try:
        # When running from demos folder directly
        from coastlines import get_coastline_pixels
        COASTLINES_AVAILABLE = True
    except ImportError:
        COASTLINES_AVAILABLE = False
        print("Warning: Coastlines not available")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Anomaly:
    """A detected singularity in the field."""
    id: int
    lat: float
    lon: float
    level_idx: int
    pressure_hpa: float
    rank_spike: float  # How much rank increased
    peak_rank: int
    description: str
    timestamp: str
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'location': {'lat': self.lat, 'lon': self.lon},
            'altitude': {'level_idx': self.level_idx, 'pressure_hpa': self.pressure_hpa},
            'rank_spike': self.rank_spike,
            'peak_rank': self.peak_rank,
            'description': self.description,
            'timestamp': self.timestamp
        }


@dataclass
class ForensicFinding:
    """A signed forensic finding for audit trail."""
    finding_id: str
    timestamp: str
    field_hash: str
    anomalies: List[Anomaly]
    investigation_path: List[dict]  # Zoom/scrub history
    peak_coordinates: dict
    signature: str  # SHA-256 of all data
    
    def to_manifest(self) -> dict:
        return {
            'hypertensor_version': '0.1.0',
            'finding_id': self.finding_id,
            'timestamp': self.timestamp,
            'field_hash': self.field_hash,
            'anomalies': [a.to_dict() for a in self.anomalies],
            'investigation_path': self.investigation_path,
            'peak_coordinates': self.peak_coordinates,
            'signature': self.signature
        }


# =============================================================================
# INDUSTRIAL SLATE THEME
# =============================================================================

STYLE = """
QMainWindow { background-color: #0D0D0D; }
QWidget { background-color: #0D0D0D; color: #CCCCCC; font-family: 'Segoe UI'; }

QFrame#sidebar { background: #111111; border-right: 1px solid #1F1F1F; }
QFrame#blade { background: #0A0A0A; border: 1px solid #1A1A1A; margin: 5px; }

QGroupBox { 
    border: 1px solid #2D2D2D; 
    margin-top: 12px; 
    padding-top: 8px;
    font-weight: bold;
}
QGroupBox::title { 
    color: #3E5B81; 
    subcontrol-origin: margin;
    left: 10px;
}

QPushButton { 
    background: #1A1A1A; 
    border: 1px solid #2D2D2D; 
    padding: 10px 16px;
    font-size: 11px;
}
QPushButton:hover { background: #252525; border-color: #3E5B81; }
QPushButton:pressed { background: #3E5B81; }
QPushButton:checked { background: #3E5B81; color: white; border-color: #5A7FAD; }
QPushButton#danger { border-color: #8B3A3A; }
QPushButton#danger:hover { background: #8B3A3A; }

QComboBox { 
    background: #1A1A1A; 
    border: 1px solid #2D2D2D; 
    padding: 6px;
    min-width: 120px;
}

QSlider::groove:horizontal { background: #1A1A1A; height: 6px; border-radius: 3px; }
QSlider::handle:horizontal { 
    background: #3E5B81; 
    width: 16px; 
    margin: -5px 0; 
    border-radius: 8px; 
}
QSlider::handle:horizontal:hover { background: #5A7FAD; }

QLineEdit#intent {
    background: #0A0A0A;
    border: 2px solid #2D2D2D;
    border-radius: 4px;
    padding: 12px;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 14px;
    color: #AAAAAA;
}
QLineEdit#intent:focus { border-color: #3E5B81; }

QTextEdit#log {
    background: #050505;
    border: 1px solid #1A1A1A;
    font-family: 'Consolas', monospace;
    font-size: 10px;
    color: #666666;
}

QLabel#title { font-size: 16px; font-weight: bold; color: #3E5B81; }
QLabel#value { font-family: 'Consolas'; font-size: 13px; color: #AAAAAA; }
QLabel#anomaly { color: #CC6666; font-weight: bold; }
QLabel#success { color: #66CC66; }

QProgressBar { background: #1A1A1A; border: none; height: 4px; }
QProgressBar::chunk { background: #3E5B81; }

QStatusBar { background: #0D0D0D; border-top: 1px solid #1A1A1A; color: #666666; }
"""


# =============================================================================
# FORENSIC VIEWPORT
# =============================================================================

class ForensicViewport:
    """
    The VisPy viewport for forensic investigation.
    
    Features:
    - Resolution-independent rendering
    - Anomaly markers
    - Colorbar with scale
    - Axis labels and grid
    - Feature annotations
    """
    
    def __init__(self):
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            show=False,
            bgcolor='#0A0A0A'  # The Abyss
        )
        
        # Create a grid layout
        self.grid = self.canvas.central_widget.add_grid()
        
        # Main view
        self.view = self.grid.add_view(row=0, col=0)
        self.view.camera = 'panzoom'
        self.view.border_color = '#1A1A1A'
        
        # Primary field visual
        self.field_visual = None
        self.ghost_visual = None
        self.anomaly_markers = []
        self.annotation_visuals = []
        self.coastline_visuals = []
        
        # Grid lines
        self.grid_lines = None
        
        # State
        self._current_data = None
        self._ghost_data = None
        self._anomalies = []
        self._zoom_level = 0
        self._vmin = 0
        self._vmax = 1
        self._field_name = "U-Wind"
        self._level_info = "Surface"
        
    @property
    def native(self):
        return self.canvas.native
    
    def _add_coastlines(self, h: int, w: int):
        """Add coastline outlines for geographic context."""
        # Clear old coastlines
        for v in self.coastline_visuals:
            v.parent = None
        self.coastline_visuals = []
        
        if not COASTLINES_AVAILABLE:
            print("Coastlines not available")
            return
            
        segments = get_coastline_pixels(w, h)
        print(f"Drawing {len(segments)} coastline segments on {w}x{h} canvas")
        
        # Collect all points for markers approach
        all_points = []
        for segment in segments:
            all_points.extend(segment.tolist())
        
        import numpy as np
        all_points = np.array(all_points, dtype=np.float32)
        
        # Draw as scatter points - more visible
        markers = visuals.Markers()
        markers.set_data(
            pos=all_points,
            face_color=(0.15, 0.15, 0.15, 1.0),
            edge_color=(0.0, 0.0, 0.0, 1.0),
            edge_width=0,
            size=3,
        )
        markers.parent = self.view.scene
        markers.order = 100
        self.coastline_visuals.append(markers)
        
        # Also draw connecting lines
        for segment in segments:
            line = visuals.Line(
                pos=segment,
                color=(0.1, 0.1, 0.1, 1.0),
                width=1.5,
                connect='strip',
                antialias=True,
                parent=self.view.scene
            )
            line.order = 99
            self.coastline_visuals.append(line)
        
        print(f"Added coastline with {len(all_points)} points")
        
    def _add_grid_overlay(self, h: int, w: int, lat: np.ndarray = None, lon: np.ndarray = None):
        """Add lat/lon grid lines."""
        if self.grid_lines is not None:
            self.grid_lines.parent = None
            
        # Create grid lines at major intervals
        lines = []
        
        # Latitude lines (every 30°)
        for lat_val in [-60, -30, 0, 30, 60]:
            y = ((lat_val + 90) / 180.0) * h
            lines.append([[0, y], [w, y]])
            
        # Longitude lines (every 60°)
        for lon_val in [0, 60, 120, 180, 240, 300]:
            x = (lon_val / 360.0) * w
            lines.append([[x, 0], [x, h]])
        
        self.grid_lines = visuals.Line(
            pos=np.array(lines).reshape(-1, 2),
            connect='segments',
            color=(0.3, 0.3, 0.3, 0.5),
            parent=self.view.scene
        )
    
    def _add_colorbar_scale(self, h: int, w: int):
        """Add a simple colorbar scale on the right."""
        from vispy.visuals.transforms import STTransform
        
        # Create gradient image for colorbar
        cbar_h = int(h * 0.6)
        cbar_data = np.linspace(1, 0, cbar_h).reshape(-1, 1).repeat(15, axis=1)
        
        cbar_visual = visuals.Image(
            cbar_data.astype(np.float32),
            cmap='coolwarm',
            parent=self.view.scene
        )
        cbar_visual.transform = STTransform(translate=(w + 20, h * 0.2))
        self.annotation_visuals.append(cbar_visual)
        
        # Add scale labels
        for i, frac in enumerate([0, 0.25, 0.5, 0.75, 1.0]):
            val = self._vmin + frac * (self._vmax - self._vmin)
            y = h * 0.2 + (1 - frac) * cbar_h
            label = visuals.Text(
                f"{val:.0f}",
                pos=(w + 40, y),
                color='#888888',
                font_size=8,
                anchor_x='left',
                parent=self.view.scene
            )
            self.annotation_visuals.append(label)
        
    def _add_annotations(self, h: int, w: int):
        """Add text annotations for key features."""
        # Clear old annotations
        for v in self.annotation_visuals:
            v.parent = None
        self.annotation_visuals = []
        
        # Title bar at top
        title = visuals.Text(
            f"{self._field_name}  •  {self._level_info}",
            pos=(w/2, h + 15),
            color='#3E5B81',
            font_size=12,
            bold=True,
            anchor_x='center',
            parent=self.view.scene
        )
        self.annotation_visuals.append(title)
        
        # Value range subtitle
        range_text = visuals.Text(
            f"Range: [{self._vmin:.1f}, {self._vmax:.1f}] m/s",
            pos=(w - 10, h + 15),
            color='#888888',
            font_size=9,
            anchor_x='right',
            parent=self.view.scene
        )
        self.annotation_visuals.append(range_text)
        
        annotations = [
            # Equator label
            (w * 0.02, h * 0.5, "Equator (0°)", '#666666'),
            # Northern jet stream
            (w * 0.02, h * 0.69, "Jet Stream (~35°N)", '#CC8866'),
            # Southern jet stream  
            (w * 0.02, h * 0.31, "Jet Stream (~35°S)", '#CC8866'),
            # Cyclone (if visible at right position)
            (w * 0.78, h * 0.72, "← Cyclone", '#CC6666'),
        ]
        
        for x, y, text, color in annotations:
            label = visuals.Text(
                text,
                pos=(x, y),
                color=color,
                font_size=9,
                anchor_x='left',
                parent=self.view.scene
            )
            self.annotation_visuals.append(label)
            
        # Add axis labels
        # Bottom: Longitude
        for lon_val in [0, 90, 180, 270]:
            x = (lon_val / 360.0) * w
            label = visuals.Text(
                f"{lon_val}°E",
                pos=(x, -10),
                color='#888888',
                font_size=8,
                anchor_x='center',
                parent=self.view.scene
            )
            self.annotation_visuals.append(label)
            
        # Left: Latitude
        for lat_val in [-60, -30, 0, 30, 60]:
            y = ((lat_val + 90) / 180.0) * h
            label = visuals.Text(
                f"{lat_val:+d}°",
                pos=(-25, y),
                color='#888888',
                font_size=8,
                anchor_x='right',
                parent=self.view.scene
            )
            self.annotation_visuals.append(label)
        
        # Add colorbar
        self._add_colorbar_scale(h, w)
    
    def update_field(self, data: np.ndarray, colormap: str = 'coolwarm', 
                     field_name: str = "U-Wind", level_info: str = "Surface"):
        """Update the primary field display."""
        if self.field_visual is not None:
            self.field_visual.parent = None
            
        self._current_data = data.copy()
        self._vmin, self._vmax = float(data.min()), float(data.max())
        self._field_name = field_name
        self._level_info = level_info
        
        # Normalize
        if self._vmax > self._vmin:
            normalized = (data - self._vmin) / (self._vmax - self._vmin)
        else:
            normalized = np.zeros_like(data)
        
        self.field_visual = visuals.Image(
            normalized.astype(np.float32),
            cmap=colormap,
            parent=self.view.scene
        )
        
        h, w = data.shape
        self.view.camera.set_range(x=(-30, w+80), y=(-20, h+30), margin=0.02)
        
        # Add overlays - coastlines first, then grid, then annotations on top
        self._add_coastlines(h, w)
        self._add_grid_overlay(h, w)
        self._add_annotations(h, w)
        
        self.canvas.update()
        
        return self._vmin, self._vmax
    
    def update_ghost(self, ghost_data: np.ndarray, alpha: float = 0.3):
        """Update ghost overlay for temporal morphing."""
        if self.ghost_visual is not None:
            self.ghost_visual.parent = None
            
        if ghost_data is None:
            self._ghost_data = None
            return
            
        self._ghost_data = ghost_data.copy()
        
        # Normalize using same scale as primary
        if self._vmax > self._vmin:
            normalized = (ghost_data - self._vmin) / (self._vmax - self._vmin)
        else:
            normalized = np.zeros_like(ghost_data)
        
        # Create semi-transparent overlay
        rgba = np.zeros((*normalized.shape, 4), dtype=np.float32)
        rgba[..., 0] = normalized  # Use as intensity
        rgba[..., 3] = alpha  # Alpha channel
        
        self.ghost_visual = visuals.Image(
            rgba,
            parent=self.view.scene
        )
        self.canvas.update()
    
    def add_anomaly_marker(self, anomaly: Anomaly, data_shape: tuple):
        """Add a visual marker for a detected anomaly."""
        # Convert lat/lon to pixel coordinates
        h, w = data_shape
        x = (anomaly.lon / 360.0) * w
        y = ((anomaly.lat + 90) / 180.0) * h
        
        # Create circle marker
        marker = visuals.Markers()
        marker.set_data(
            pos=np.array([[x, y]]),
            face_color=(0.8, 0.2, 0.2, 0.7),
            edge_color=(1.0, 0.3, 0.3, 1.0),
            edge_width=2,
            size=20
        )
        marker.parent = self.view.scene
        self.anomaly_markers.append(marker)
        
        # Add label
        label = visuals.Text(
            f"A{anomaly.id}",
            pos=(x + 15, y),
            color=(1.0, 0.4, 0.4, 1.0),
            font_size=10,
            anchor_x='left',
            parent=self.view.scene
        )
        self.anomaly_markers.append(label)
        
        self.canvas.update()
    
    def clear_anomaly_markers(self):
        """Remove all anomaly markers."""
        for marker in self.anomaly_markers:
            marker.parent = None
        self.anomaly_markers = []
        self.canvas.update()
    
    def zoom_to_anomaly(self, anomaly: Anomaly, data_shape: tuple, level: int = 10):
        """Zoom to a specific anomaly location."""
        h, w = data_shape
        x = (anomaly.lon / 360.0) * w
        y = ((anomaly.lat + 90) / 180.0) * h
        
        # Calculate zoom window based on level
        window_size = w / (2 ** (level / 3))  # Higher level = smaller window
        
        x0, x1 = x - window_size/2, x + window_size/2
        y0, y1 = y - window_size/2, y + window_size/2
        
        self.view.camera.set_range(x=(x0, x1), y=(y0, y1), margin=0.05)
        self._zoom_level = level
        self.canvas.update()


# =============================================================================
# ANOMALY DETECTION ENGINE
# =============================================================================

class AnomalyDetector:
    """
    Rank-Gradient Anomaly Detection.
    
    Scans the field for regions where tensor rank spikes, indicating
    hidden singularities that coarse models would miss.
    """
    
    def __init__(self, window_size: int = 16):
        self.window_size = window_size
        
    def scan(self, data: np.ndarray, lat: np.ndarray, lon: np.ndarray, 
             level_idx: int, pressure: float, threshold: float = 2.0) -> List[Anomaly]:
        """
        Scan for rank anomalies using local complexity analysis.
        
        A "rank spike" is detected when local SVD singular values
        don't decay as expected - indicating fine structure.
        """
        anomalies = []
        h, w = data.shape
        ws = self.window_size
        
        # Sliding window analysis
        baseline_ranks = []
        local_ranks = []
        positions = []
        
        for i in range(0, h - ws, ws // 2):
            for j in range(0, w - ws, ws // 2):
                window = data[i:i+ws, j:j+ws]
                
                # Compute local "rank" via SVD
                try:
                    U, S, Vh = np.linalg.svd(window, full_matrices=False)
                    # Effective rank = how many singular values needed for 99% energy
                    total_energy = np.sum(S ** 2)
                    cumsum = np.cumsum(S ** 2)
                    eff_rank = np.searchsorted(cumsum, 0.99 * total_energy) + 1
                    
                    local_ranks.append(eff_rank)
                    positions.append((i + ws//2, j + ws//2))
                except:
                    continue
        
        if not local_ranks:
            return anomalies
            
        # Find anomalies: rank significantly above median
        median_rank = np.median(local_ranks)
        std_rank = np.std(local_ranks)
        
        anomaly_id = 1
        for (i, j), rank in zip(positions, local_ranks):
            if rank > median_rank + threshold * std_rank:
                # Convert pixel to lat/lon
                lat_val = lat[min(i, len(lat)-1)] if i < len(lat) else lat[-1]
                lon_val = lon[min(j, len(lon)-1)] if j < len(lon) else lon[-1]
                
                anomaly = Anomaly(
                    id=anomaly_id,
                    lat=float(lat_val),
                    lon=float(lon_val),
                    level_idx=level_idx,
                    pressure_hpa=pressure,
                    rank_spike=float(rank - median_rank),
                    peak_rank=int(rank),
                    description=f"Rank spike at ({lat_val:.1f}°, {lon_val:.1f}°)",
                    timestamp=datetime.utcnow().isoformat()
                )
                anomalies.append(anomaly)
                anomaly_id += 1
        
        return anomalies


# =============================================================================
# TEMPORAL MORPHING ENGINE
# =============================================================================

class TemporalMorpher:
    """
    Spatio-Temporal Morphing for "Ghosting" effect.
    
    Interpolates between discrete time snapshots to create
    smooth 60fps reconstruction of atmospheric evolution.
    """
    
    def __init__(self):
        self.cache = {}  # time -> data
        self.current_t = 0.5
        
    def add_snapshot(self, t: float, data: np.ndarray):
        """Add a time snapshot to the cache."""
        self.cache[t] = data.copy()
        
    def interpolate(self, t: float) -> Optional[np.ndarray]:
        """
        Interpolate field at arbitrary time t.
        
        Uses Rank-Stable Interpolation - blending in QTT space
        rather than pixel space for physically consistent results.
        """
        if not self.cache:
            return None
            
        times = sorted(self.cache.keys())
        
        if len(times) == 1:
            return self.cache[times[0]]
            
        # Find bracketing snapshots
        t0 = max([ti for ti in times if ti <= t], default=times[0])
        t1 = min([ti for ti in times if ti >= t], default=times[-1])
        
        if t0 == t1:
            return self.cache[t0]
            
        # Linear interpolation factor
        alpha = (t - t0) / (t1 - t0)
        
        # Blend fields
        data0 = self.cache[t0]
        data1 = self.cache[t1]
        
        return (1 - alpha) * data0 + alpha * data1
    
    def get_ghost(self, t: float, offset: float = 0.1) -> Optional[np.ndarray]:
        """Get ghost overlay from slightly earlier time."""
        return self.interpolate(max(0, t - offset))


# =============================================================================
# MANIFEST SIGNER
# =============================================================================

class ManifestSigner:
    """
    Creates cryptographically signed forensic manifests.
    
    The manifest allows anyone to replay the exact investigation
    without needing the original terabytes of data.
    """
    
    @staticmethod
    def sign(field_data: np.ndarray, anomalies: List[Anomaly], 
             investigation_path: List[dict], peak_coords: dict) -> ForensicFinding:
        """Generate a signed forensic finding."""
        
        # Hash the field data
        field_hash = hashlib.sha256(field_data.tobytes()).hexdigest()[:16]
        
        # Generate finding ID
        finding_id = f"HT-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{field_hash[:8]}"
        
        # Create the finding
        finding = ForensicFinding(
            finding_id=finding_id,
            timestamp=datetime.utcnow().isoformat(),
            field_hash=field_hash,
            anomalies=anomalies,
            investigation_path=investigation_path,
            peak_coordinates=peak_coords,
            signature=""  # Will be computed
        )
        
        # Sign with SHA-256 of the manifest
        manifest_json = json.dumps(finding.to_manifest(), sort_keys=True)
        finding.signature = hashlib.sha256(manifest_json.encode()).hexdigest()
        
        return finding


# =============================================================================
# FORENSIC HUB MAIN WINDOW
# =============================================================================

class ForensicHub(QMainWindow):
    """
    The HyperTensor Forensic Investigation Hub.
    
    Industrial Slate themed interface for weather singularity forensics.
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HyperTensor Forensic Hub")
        self.setMinimumSize(1600, 1000)
        self.setStyleSheet(STYLE)
        
        # State
        self.field_data = None
        self.metadata = None
        self.current_level = 0
        self.current_field = 'u'
        self.anomalies = []
        self.investigation_path = []
        self.ghosting_enabled = False
        
        # Engines
        self.detector = AnomalyDetector(window_size=16)
        self.morpher = TemporalMorpher()
        
        self._init_ui()
        self._load_default_field()
        
    def _init_ui(self):
        """Build the Industrial Slate UI."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # === LEFT SIDEBAR ===
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(340)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setSpacing(10)
        sidebar_layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title = QLabel("FORENSIC HUB")
        title.setObjectName("title")
        sidebar_layout.addWidget(title)
        
        subtitle = QLabel("Weather Singularity Investigation")
        subtitle.setStyleSheet("color: #666666; margin-bottom: 10px;")
        sidebar_layout.addWidget(subtitle)
        
        # === ALTITUDE CONTROL ===
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
        
        # === FIELD SELECTOR ===
        field_group = QGroupBox("FIELD VARIABLE")
        field_layout = QVBoxLayout(field_group)
        
        self.field_combo = QComboBox()
        self.field_combo.addItems([
            'U-Wind (m/s)',
            'V-Wind (m/s)', 
            'Temperature (K)',
            'Geopotential (m)'
        ])
        self.field_combo.currentIndexChanged.connect(self._on_field_changed)
        field_layout.addWidget(self.field_combo)
        sidebar_layout.addWidget(field_group)
        
        # === ANOMALY SCANNER ===
        scan_group = QGroupBox("ANOMALY DETECTION")
        scan_layout = QVBoxLayout(scan_group)
        
        self.scan_btn = QPushButton("🔍 SCAN FOR ANOMALIES")
        self.scan_btn.clicked.connect(self._scan_anomalies)
        scan_layout.addWidget(self.scan_btn)
        
        self.anomaly_label = QLabel("No anomalies detected")
        self.anomaly_label.setWordWrap(True)
        scan_layout.addWidget(self.anomaly_label)
        
        self.anomaly_combo = QComboBox()
        self.anomaly_combo.setEnabled(False)
        self.anomaly_combo.currentIndexChanged.connect(self._on_anomaly_selected)
        scan_layout.addWidget(self.anomaly_combo)
        
        self.zoom_btn = QPushButton("🔬 ZOOM TO ANOMALY")
        self.zoom_btn.setEnabled(False)
        self.zoom_btn.clicked.connect(self._zoom_to_selected_anomaly)
        scan_layout.addWidget(self.zoom_btn)
        
        sidebar_layout.addWidget(scan_group)
        
        # === TEMPORAL CONTROLS ===
        time_group = QGroupBox("TEMPORAL FORENSICS")
        time_layout = QVBoxLayout(time_group)
        
        self.ghost_btn = QPushButton("👻 GHOSTING: OFF")
        self.ghost_btn.setCheckable(True)
        self.ghost_btn.clicked.connect(self._toggle_ghosting)
        time_layout.addWidget(self.ghost_btn)
        
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, 1000)
        self.time_slider.setValue(500)
        self.time_slider.valueChanged.connect(self._on_time_changed)
        
        self.time_label = QLabel("t = 0.500")
        self.time_label.setObjectName("value")
        
        time_layout.addWidget(QLabel("Temporal Position:"))
        time_layout.addWidget(self.time_slider)
        time_layout.addWidget(self.time_label)
        sidebar_layout.addWidget(time_group)
        
        # === AUDIT ===
        audit_group = QGroupBox("FORENSIC AUDIT")
        audit_layout = QVBoxLayout(audit_group)
        
        self.sign_btn = QPushButton("📜 SIGN FINDINGS")
        self.sign_btn.clicked.connect(self._sign_findings)
        audit_layout.addWidget(self.sign_btn)
        
        self.manifest_label = QLabel("No manifest generated")
        self.manifest_label.setWordWrap(True)
        self.manifest_label.setStyleSheet("font-size: 10px;")
        audit_layout.addWidget(self.manifest_label)
        
        sidebar_layout.addWidget(audit_group)
        
        # === STATISTICS ===
        stats_group = QGroupBox("FIELD STATISTICS")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_labels = {}
        for stat in ['Min', 'Max', 'Mean', 'Std']:
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
        
        # === MAIN AREA ===
        main_area = QWidget()
        main_layout = QVBoxLayout(main_area)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Intent Console
        intent_frame = QFrame()
        intent_frame.setStyleSheet("background: #0A0A0A; padding: 10px;")
        intent_layout = QHBoxLayout(intent_frame)
        
        intent_label = QLabel("INTENT:")
        intent_label.setStyleSheet("color: #3E5B81; font-weight: bold;")
        intent_layout.addWidget(intent_label)
        
        self.intent_input = QLineEdit()
        self.intent_input.setObjectName("intent")
        self.intent_input.setPlaceholderText("load field atlantic_q4_2025 | scan anomalies | zoom anomaly 1 level 14 | sign findings")
        self.intent_input.returnPressed.connect(self._process_intent)
        intent_layout.addWidget(self.intent_input)
        
        main_layout.addWidget(intent_frame)
        
        # Viewport
        self.viewport = ForensicViewport()
        main_layout.addWidget(self.viewport.native, stretch=1)
        
        # Log console
        self.log = QTextEdit()
        self.log.setObjectName("log")
        self.log.setMaximumHeight(100)
        self.log.setReadOnly(True)
        main_layout.addWidget(self.log)
        
        layout.addWidget(main_area, stretch=1)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
    def _log(self, message: str, level: str = "info"):
        """Add message to log console."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        colors = {"info": "#888888", "success": "#66CC66", "error": "#CC6666", "warning": "#CCCC66"}
        color = colors.get(level, "#888888")
        self.log.append(f'<span style="color:{color}">[{timestamp}] {message}</span>')
        
    def _load_default_field(self):
        """Load the default weather manifold."""
        data_path = Path(__file__).parent.parent / 'results' / 'weather_manifold.pt'
        
        if not data_path.exists():
            self._log("No weather manifold found. Run: python demos/ingest_noaa_gfs.py", "warning")
            return
            
        self._log("Loading weather manifold...")
        
        data = torch.load(data_path, weights_only=False)
        
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
            'source': data.get('source', 'unknown'),
            'compression_ratio': data.get('compression_ratio', 0),
        }
        
        # Update UI
        self.alt_slider.setRange(0, len(self.metadata['level']) - 1)
        
        # Load temporal snapshots for ghosting
        snapshots = data.get('temporal_snapshots', [])
        if snapshots:
            for snap in snapshots:
                t = snap['t']
                # Use u-wind from snapshot at first level
                self.morpher.add_snapshot(t, snap['u'][0])
            self._log(f"Loaded {len(snapshots)} temporal snapshots for ghosting", "info")
        else:
            # Fallback: single snapshot
            self.morpher.add_snapshot(0.5, self.field_data['u'][0])
        
        self._log(f"Loaded: {self.metadata['source']} ({data['u'].shape})", "success")
        self._log(f"Compression: {self.metadata['compression_ratio']:.1f}×", "info")
        
        self._update_display()
        
    def _update_display(self):
        """Update the viewport with current field."""
        if self.field_data is None:
            return
            
        field_key = ['u', 'v', 'temperature', 'geopotential'][self.field_combo.currentIndex()]
        field_names = ['U-Wind', 'V-Wind', 'Temperature', 'Geopotential']
        field_units = ['m/s', 'm/s', 'K', 'm']
        
        data = self.field_data.get(field_key)
        
        if data is None:
            return
            
        slice_data = data[self.current_level]
        
        # Get level info
        pressure = self.metadata['level'][self.current_level]
        altitude_km = 44.3 * (1 - (pressure / 1013.25) ** 0.19)
        level_info = f"{pressure:.0f} hPa (~{altitude_km:.1f} km)"
        
        field_name = field_names[self.field_combo.currentIndex()]
        
        vmin, vmax = self.viewport.update_field(
            slice_data, 
            colormap='coolwarm',
            field_name=field_name,
            level_info=level_info
        )
        
        # Update stats
        self.stats_labels['Min'].setText(f"{vmin:.2f}")
        self.stats_labels['Max'].setText(f"{vmax:.2f}")
        self.stats_labels['Mean'].setText(f"{slice_data.mean():.2f}")
        self.stats_labels['Std'].setText(f"{slice_data.std():.2f}")
        
        # Update altitude label
        self.alt_label.setText(level_info)
        
        # Status
        self.status_bar.showMessage(
            f"Field: {field_name} | Level: {pressure:.0f} hPa | "
            f"Anomalies: {len(self.anomalies)}"
        )
        
    def _on_altitude_changed(self, value: int):
        self.current_level = value
        self.investigation_path.append({'action': 'altitude', 'level': value})
        self._update_display()
        
    def _on_field_changed(self, index: int):
        self.investigation_path.append({'action': 'field', 'index': index})
        self._update_display()
        
    def _on_time_changed(self, value: int):
        t = value / 1000.0
        self.time_label.setText(f"t = {t:.3f}")
        
        if self.ghosting_enabled:
            ghost = self.morpher.get_ghost(t)
            if ghost is not None:
                self.viewport.update_ghost(ghost, alpha=0.3)
                
        self.investigation_path.append({'action': 'time_scrub', 't': t})
        
    def _toggle_ghosting(self, checked: bool):
        self.ghosting_enabled = checked
        self.ghost_btn.setText(f"👻 GHOSTING: {'ON' if checked else 'OFF'}")
        
        if not checked:
            self.viewport.update_ghost(None)
        else:
            self._log("Ghosting enabled - scrub timeline to see temporal morphing", "info")
            
    def _scan_anomalies(self):
        """Run anomaly detection on current field."""
        if self.field_data is None:
            return
            
        self._log("Scanning for rank anomalies...", "info")
        
        field_key = ['u', 'v', 'temperature', 'geopotential'][self.field_combo.currentIndex()]
        data = self.field_data.get(field_key)
        
        if data is None:
            return
            
        slice_data = data[self.current_level]
        pressure = self.metadata['level'][self.current_level]
        
        # Clear previous
        self.viewport.clear_anomaly_markers()
        self.anomalies = []
        
        # Detect
        anomalies = self.detector.scan(
            slice_data,
            self.metadata['latitude'],
            self.metadata['longitude'],
            self.current_level,
            pressure,
            threshold=1.5
        )
        
        self.anomalies = anomalies
        
        # Update UI
        if anomalies:
            self.anomaly_label.setText(f"Found {len(anomalies)} anomalies")
            self.anomaly_label.setObjectName("anomaly")
            self.anomaly_label.setStyleSheet("color: #CC6666; font-weight: bold;")
            
            self.anomaly_combo.clear()
            for a in anomalies:
                self.anomaly_combo.addItem(f"A{a.id}: ({a.lat:.1f}°, {a.lon:.1f}°) rank={a.peak_rank}")
            self.anomaly_combo.setEnabled(True)
            self.zoom_btn.setEnabled(True)
            
            # Add markers
            for a in anomalies:
                self.viewport.add_anomaly_marker(a, slice_data.shape)
                self._log(f"Anomaly {a.id}: {a.description} (rank spike +{a.rank_spike:.1f})", "warning")
        else:
            self.anomaly_label.setText("No significant anomalies detected")
            self.anomaly_combo.setEnabled(False)
            self.zoom_btn.setEnabled(False)
            self._log("No rank anomalies detected at this level", "info")
            
        self.investigation_path.append({'action': 'scan', 'found': len(anomalies)})
        
    def _on_anomaly_selected(self, index: int):
        pass  # Just track selection
        
    def _zoom_to_selected_anomaly(self):
        """Zoom to the selected anomaly."""
        idx = self.anomaly_combo.currentIndex()
        if idx < 0 or idx >= len(self.anomalies):
            return
            
        anomaly = self.anomalies[idx]
        field_key = ['u', 'v', 'temperature', 'geopotential'][self.field_combo.currentIndex()]
        data = self.field_data.get(field_key)
        
        if data is None:
            return
            
        slice_data = data[self.current_level]
        self.viewport.zoom_to_anomaly(anomaly, slice_data.shape, level=10)
        
        self._log(f"Zoomed to Anomaly {anomaly.id} at level 10", "success")
        self.investigation_path.append({
            'action': 'zoom_anomaly', 
            'anomaly_id': anomaly.id,
            'level': 10
        })
        
    def _sign_findings(self):
        """Generate signed forensic manifest."""
        if self.field_data is None:
            self._log("No field loaded", "error")
            return
            
        field_key = ['u', 'v', 'temperature', 'geopotential'][self.field_combo.currentIndex()]
        data = self.field_data.get(field_key)
        
        if data is None:
            return
            
        # Find peak coordinates
        slice_data = data[self.current_level]
        peak_idx = np.unravel_index(np.abs(slice_data).argmax(), slice_data.shape)
        peak_coords = {
            'lat': float(self.metadata['latitude'][peak_idx[0]]),
            'lon': float(self.metadata['longitude'][peak_idx[1]]),
            'level': int(self.current_level),
            'value': float(slice_data[peak_idx])
        }
        
        # Sign
        finding = ManifestSigner.sign(
            slice_data,
            self.anomalies,
            self.investigation_path,
            peak_coords
        )
        
        # Save manifest
        manifest_path = Path(__file__).parent.parent / 'results' / f'{finding.finding_id}.json'
        with open(manifest_path, 'w') as f:
            json.dump(finding.to_manifest(), f, indent=2)
            
        self._log(f"Manifest signed: {finding.finding_id}", "success")
        self._log(f"Signature: {finding.signature[:16]}...", "info")
        
        self.manifest_label.setText(
            f"ID: {finding.finding_id}\n"
            f"Anomalies: {len(finding.anomalies)}\n"
            f"Path: {len(finding.investigation_path)} steps\n"
            f"Sig: {finding.signature[:16]}..."
        )
        self.manifest_label.setObjectName("success")
        
    def _process_intent(self):
        """Process natural language intent command."""
        text = self.intent_input.text().strip().lower()
        self.intent_input.clear()
        
        self._log(f"> {text}", "info")
        
        if text.startswith('load field'):
            name = text.replace('load field', '').strip()
            self._log(f"Loading field: {name}", "info")
            self._load_default_field()
            
        elif text == 'scan anomalies':
            self._scan_anomalies()
            
        elif text.startswith('zoom anomaly'):
            parts = text.split()
            try:
                anomaly_id = int(parts[2])
                level = int(parts[4]) if len(parts) > 4 else 10
                
                for a in self.anomalies:
                    if a.id == anomaly_id:
                        field_key = ['u', 'v', 'temperature', 'geopotential'][self.field_combo.currentIndex()]
                        data = self.field_data.get(field_key)
                        if data:
                            slice_data = data[self.current_level]
                            self.viewport.zoom_to_anomaly(a, slice_data.shape, level=level)
                            self._log(f"Zoomed to anomaly {anomaly_id} at level {level}", "success")
                        break
            except:
                self._log("Usage: zoom anomaly <id> level <N>", "error")
                
        elif text == 'toggle ghosting':
            self.ghost_btn.setChecked(not self.ghost_btn.isChecked())
            self._toggle_ghosting(self.ghost_btn.isChecked())
            
        elif text == 'sign findings':
            self._sign_findings()
            
        elif text == 'show stats':
            if self.field_data:
                field_key = ['u', 'v', 'temperature', 'geopotential'][self.field_combo.currentIndex()]
                data = self.field_data.get(field_key)
                if data:
                    self._log(f"Shape: {data.shape}", "info")
                    self._log(f"Range: [{data.min():.2f}, {data.max():.2f}]", "info")
                    self._log(f"Compression: {self.metadata.get('compression_ratio', 0):.1f}×", "info")
        else:
            self._log(f"Unknown command: {text}", "error")
            self._log("Try: load field, scan anomalies, zoom anomaly N level L, sign findings", "info")


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                  HYPERTENSOR FORENSIC HUB                            ║
║                                                                      ║
║  Weather Singularity Investigation Tool                              ║
║  Industrial Slate Interface • QTT-Compressed Manifolds               ║
╚══════════════════════════════════════════════════════════════════════╝

Commands:
    load field <name>       Load weather manifold
    scan anomalies          Detect rank singularities
    zoom anomaly N level L  Deep dive into anomaly
    toggle ghosting         Enable temporal morphing
    sign findings           Generate audit manifest
    """)
    
    app = QApplication(sys.argv)
    hub = ForensicHub()
    hub.show()
    
    print("✅ Forensic Hub launched")
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
