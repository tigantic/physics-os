#!/usr/bin/env python3
"""
Ontic CFD Viewer
======================

Visualizes REAL CFD simulation data from The Ontic Engine.

Loads actual physics simulation results:
- Mach 5 Oblique Shock (wedge flow)
- Double Mach Reflection

Fields you can view:
- rho: Density (kg/m³)
- p: Pressure (Pa) 
- M: Mach number
- u, v: Velocity components (m/s)

This is NOT synthetic random data - these are solutions to the Euler equations.
"""

import sys
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from PySide6.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
        QLabel, QFrame, QPushButton, QApplication, QSlider,
        QStatusBar, QComboBox, QGroupBox
    )
    from PySide6.QtCore import Qt, QTimer
except ImportError:
    print("ERROR: PySide6 not installed. Run: pip install PySide6")
    sys.exit(1)

try:
    from vispy import scene
    from vispy.scene import visuals
    VISPY_AVAILABLE = True
except ImportError:
    print("ERROR: VisPy not installed. Run: pip install vispy")
    sys.exit(1)


# Dark theme
STYLE = """
QMainWindow { background-color: #0D0D0D; }
QWidget { background-color: #0D0D0D; color: #CCCCCC; font-family: 'Segoe UI'; }
QGroupBox { border: 1px solid #2D2D2D; margin-top: 10px; padding-top: 10px; }
QGroupBox::title { color: #3E5B81; }
QPushButton { background: #1A1A1A; border: 1px solid #2D2D2D; padding: 8px 16px; }
QPushButton:hover { background: #252525; border-color: #3E5B81; }
QPushButton:checked { background: #3E5B81; color: white; }
QComboBox { background: #1A1A1A; border: 1px solid #2D2D2D; padding: 5px; }
QLabel#title { font-size: 18px; font-weight: bold; color: #3E5B81; }
QLabel#value { font-family: 'Consolas'; font-size: 14px; color: #AAAAAA; }
QStatusBar { background: #0D0D0D; border-top: 1px solid #1A1A1A; }
"""


class CFDCanvas:
    """VisPy canvas for CFD field visualization."""
    
    def __init__(self):
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            show=False,
            bgcolor='#0A0A0A'
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'panzoom'
        self.view.camera.aspect = 1
        
        self.image_visual = None
        self.colorbar_data = None
        self._data = None
        self._vmin = 0
        self._vmax = 1
        
    @property
    def native(self):
        return self.canvas.native
    
    def update_field(self, data: np.ndarray, title: str = "", colormap: str = 'inferno'):
        """Update the displayed field."""
        if self.image_visual is not None:
            self.image_visual.parent = None
            
        self._data = data.copy()
        self._vmin, self._vmax = float(data.min()), float(data.max())
        
        # Normalize
        if self._vmax > self._vmin:
            normalized = (data - self._vmin) / (self._vmax - self._vmin)
        else:
            normalized = np.zeros_like(data)
        
        # Create image
        self.image_visual = visuals.Image(
            normalized.astype(np.float32),
            cmap=colormap,
            parent=self.view.scene
        )
        
        # Fit to view
        h, w = data.shape
        self.view.camera.set_range(x=(0, w), y=(0, h), margin=0.05)
        self.canvas.update()
        
        return self._vmin, self._vmax


class CFDViewer(QMainWindow):
    """Main window for CFD data visualization."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ontic CFD Viewer")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet(STYLE)
        
        # Load CFD data
        self.data = self._load_cfd_data()
        self.current_field = 'M'  # Start with Mach number
        
        self._init_ui()
        self._update_display()
        
    def _load_cfd_data(self) -> dict:
        """Load the Mach 5 wedge CFD results."""
        data_path = Path(__file__).parent.parent / 'results' / 'mach5_wedge_field.pt'
        
        if not data_path.exists():
            print(f"ERROR: CFD data not found at {data_path}")
            print("Run a CFD benchmark first: python tools/scripts/mach5_wedge.py")
            sys.exit(1)
            
        data = torch.load(data_path, weights_only=True)
        
        # Convert to numpy
        result = {}
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                result[key] = val.numpy()
            else:
                result[key] = val
                
        print(f"✅ Loaded CFD data: {data_path.name}")
        print(f"   Grid: {result['rho'].shape[1]}×{result['rho'].shape[0]}")
        print(f"   Fields: {[k for k in result.keys() if k not in ['x', 'y']]}")
        
        return result
    
    def _init_ui(self):
        """Initialize the user interface."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Left panel - controls
        left_panel = QFrame()
        left_panel.setFixedWidth(280)
        left_panel.setStyleSheet("background: #111111; border-right: 1px solid #1F1F1F;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)
        left_layout.setContentsMargins(15, 20, 15, 15)
        
        # Title
        title = QLabel("MACH 5 WEDGE FLOW")
        title.setObjectName("title")
        left_layout.addWidget(title)
        
        # Description
        desc = QLabel(
            "Oblique shock simulation\n"
            "M∞ = 5.0, θ = 15°\n"
            "100×50 grid, Euler equations"
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666666; margin-bottom: 10px;")
        left_layout.addWidget(desc)
        
        # Field selector
        field_group = QGroupBox("FIELD VARIABLE")
        field_layout = QVBoxLayout(field_group)
        
        self.field_combo = QComboBox()
        self.field_combo.addItems([
            'M - Mach Number',
            'p - Pressure (Pa)',
            'rho - Density (kg/m³)',
            'u - X-Velocity (m/s)',
            'v - Y-Velocity (m/s)'
        ])
        self.field_combo.currentIndexChanged.connect(self._on_field_changed)
        field_layout.addWidget(self.field_combo)
        left_layout.addWidget(field_group)
        
        # Statistics
        stats_group = QGroupBox("FIELD STATISTICS")
        stats_layout = QVBoxLayout(stats_group)
        
        self.min_label = QLabel("Min: --")
        self.min_label.setObjectName("value")
        self.max_label = QLabel("Max: --")
        self.max_label.setObjectName("value")
        self.mean_label = QLabel("Mean: --")
        self.mean_label.setObjectName("value")
        
        stats_layout.addWidget(self.min_label)
        stats_layout.addWidget(self.max_label)
        stats_layout.addWidget(self.mean_label)
        left_layout.addWidget(stats_group)
        
        # Physics info
        physics_group = QGroupBox("SHOCK PHYSICS")
        physics_layout = QVBoxLayout(physics_group)
        
        physics_info = QLabel(
            "• Upstream: M = 5.0\n"
            "• Wedge angle: 15°\n"
            "• Expected shock angle: 24.3°\n"
            "• Post-shock M ≈ 3.5\n"
            "• Pressure ratio ≈ 4.8×"
        )
        physics_info.setStyleSheet("color: #888888; font-size: 11px;")
        physics_layout.addWidget(physics_info)
        left_layout.addWidget(physics_group)
        
        # Colormap selector
        cmap_group = QGroupBox("COLORMAP")
        cmap_layout = QVBoxLayout(cmap_group)
        
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(['inferno', 'viridis', 'plasma', 'magma', 'coolwarm', 'jet'])
        self.cmap_combo.currentTextChanged.connect(self._on_colormap_changed)
        cmap_layout.addWidget(self.cmap_combo)
        left_layout.addWidget(cmap_group)
        
        left_layout.addStretch()
        
        # What you can learn
        learn_group = QGroupBox("WHAT THIS SHOWS")
        learn_layout = QVBoxLayout(learn_group)
        self.learn_label = QLabel("")
        self.learn_label.setWordWrap(True)
        self.learn_label.setStyleSheet("color: #AAAAAA; font-size: 10px;")
        learn_layout.addWidget(self.learn_label)
        left_layout.addWidget(learn_group)
        
        layout.addWidget(left_panel)
        
        # Main viewport
        self.canvas = CFDCanvas()
        layout.addWidget(self.canvas.native, stretch=1)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
    def _on_field_changed(self, index: int):
        """Handle field selection change."""
        fields = ['M', 'p', 'rho', 'u', 'v']
        self.current_field = fields[index]
        self._update_display()
        
    def _on_colormap_changed(self, cmap: str):
        """Handle colormap change."""
        self._update_display()
        
    def _update_display(self):
        """Update the visualization with current field."""
        field_data = self.data[self.current_field]
        cmap = self.cmap_combo.currentText()
        
        vmin, vmax = self.canvas.update_field(field_data, self.current_field, cmap)
        
        # Update statistics
        self.min_label.setText(f"Min: {vmin:.4f}")
        self.max_label.setText(f"Max: {vmax:.4f}")
        self.mean_label.setText(f"Mean: {field_data.mean():.4f}")
        
        # Update "what you can learn" based on field
        explanations = {
            'M': (
                "MACH NUMBER shows flow speed relative to sound speed.\n\n"
                "• Yellow regions: Supersonic (M > 1)\n"
                "• Dark regions: Slower flow\n"
                "• Sharp gradient: Oblique shock wave\n"
                "• The shock turns the flow and slows it down"
            ),
            'p': (
                "PRESSURE shows the force per unit area.\n\n"
                "• Bright = High pressure (behind shock)\n"
                "• Dark = Low pressure (upstream)\n"
                "• Shock compression ratio ≈ 4.8×\n"
                "• Pressure jump validates shock physics"
            ),
            'rho': (
                "DENSITY shows mass per unit volume.\n\n"
                "• Bright = Dense gas (compressed)\n"
                "• Dark = Upstream density\n"
                "• Compression across shock ≈ 2.75×\n"
                "• Density increases with pressure"
            ),
            'u': (
                "X-VELOCITY shows horizontal flow speed.\n\n"
                "• Upstream: ~1700 m/s (Mach 5)\n"
                "• Post-shock: Reduced velocity\n"
                "• Flow deflects around wedge\n"
                "• Conservation of momentum"
            ),
            'v': (
                "Y-VELOCITY shows vertical flow speed.\n\n"
                "• Upstream: ~0 (horizontal flow)\n"
                "• Post-shock: Deflected downward\n"
                "• Matches wedge angle (15°)\n"
                "• Shows flow turning behavior"
            )
        }
        
        self.learn_label.setText(explanations.get(self.current_field, ""))
        
        # Status
        h, w = field_data.shape
        self.status_bar.showMessage(
            f"Field: {self.current_field} | Grid: {w}×{h} | "
            f"Range: [{vmin:.3f}, {vmax:.3f}]"
        )


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    ONTIC_ENGINE CFD VIEWER                            ║
║                                                                      ║
║  Visualizing REAL physics simulation data                            ║
║  Mach 5 oblique shock over a 15° wedge                               ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    app = QApplication(sys.argv)
    viewer = CFDViewer()
    viewer.show()
    
    print("\n✅ CFD Viewer launched")
    print("   Use the dropdown to switch between field variables")
    print("   Pan/zoom with mouse to explore the shock structure")
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
