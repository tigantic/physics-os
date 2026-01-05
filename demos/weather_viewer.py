#!/usr/bin/env python3
"""
HyperTensor Weather Manifold Viewer
===================================

Visualizes 3D atmospheric data compressed into QTT manifolds.

Data: NOAA GFS-style weather fields
- U-Wind: East-West wind component (m/s)
- V-Wind: North-South wind component (m/s)
- Temperature: Atmospheric temperature (K)
- Geopotential: Height of pressure surfaces (m)

Dimensions:
- Altitude: Pressure levels (1000hPa = surface, 10hPa = ~31km)
- Latitude: -90° to 90°
- Longitude: 0° to 360°

Features:
- Slice through any altitude (pressure level)
- View cyclone structures and jet streams
- See real atmospheric circulation patterns
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
    from PySide6.QtCore import Qt
except ImportError:
    print("ERROR: PySide6 not installed. Run: pip install PySide6")
    sys.exit(1)

try:
    from vispy import scene
    from vispy.scene import visuals
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
QComboBox { background: #1A1A1A; border: 1px solid #2D2D2D; padding: 5px; }
QSlider::groove:horizontal { background: #1A1A1A; height: 6px; border-radius: 3px; }
QSlider::handle:horizontal { background: #3E5B81; width: 14px; margin: -4px 0; border-radius: 7px; }
QLabel#title { font-size: 18px; font-weight: bold; color: #3E5B81; }
QLabel#value { font-family: 'Consolas'; font-size: 14px; color: #AAAAAA; }
QLabel#info { font-size: 11px; color: #666666; }
QStatusBar { background: #0D0D0D; border-top: 1px solid #1A1A1A; }
"""


class WeatherCanvas:
    """VisPy canvas for weather field visualization."""
    
    def __init__(self):
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            show=False,
            bgcolor='#0A0A0A'
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'panzoom'
        
        self.image_visual = None
        self._data = None
        self._vmin = 0
        self._vmax = 1
        
    @property
    def native(self):
        return self.canvas.native
    
    def update_field(self, data: np.ndarray, colormap: str = 'coolwarm'):
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
        self.view.camera.set_range(x=(0, w), y=(0, h), margin=0.02)
        self.canvas.update()
        
        return self._vmin, self._vmax


class WeatherViewer(QMainWindow):
    """Main window for weather data visualization."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HyperTensor Weather Manifold Viewer")
        self.setMinimumSize(1400, 900)
        self.setStyleSheet(STYLE)
        
        # Load weather data
        self.data = self._load_weather_data()
        self.current_field = 'u'
        self.current_level = 0
        
        self._init_ui()
        self._update_display()
        
    def _load_weather_data(self) -> dict:
        """Load the weather manifold."""
        data_path = Path(__file__).parent.parent / 'results' / 'weather_manifold.pt'
        
        if not data_path.exists():
            print(f"Weather manifold not found at {data_path}")
            print("Running ingestion first...")
            
            # Run ingestion
            import subprocess
            result = subprocess.run(
                [sys.executable, str(Path(__file__).parent / 'ingest_noaa_gfs.py')],
                cwd=str(Path(__file__).parent.parent)
            )
            
            if not data_path.exists():
                print("ERROR: Ingestion failed")
                sys.exit(1)
        
        data = torch.load(data_path, weights_only=False)
        
        print(f"✅ Loaded weather manifold")
        print(f"   Shape: {data['u'].shape} (levels × lat × lon)")
        print(f"   Levels: {len(data['level'])} pressure levels")
        print(f"   Source: {data.get('source', 'unknown')}")
        
        if 'compression_ratio' in data:
            print(f"   Compression: {data['compression_ratio']:.1f}×")
        
        return data
    
    def _init_ui(self):
        """Initialize the user interface."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Left panel - controls
        left_panel = QFrame()
        left_panel.setFixedWidth(320)
        left_panel.setStyleSheet("background: #111111; border-right: 1px solid #1F1F1F;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)
        left_layout.setContentsMargins(15, 20, 15, 15)
        
        # Title
        title = QLabel("ATMOSPHERIC MANIFOLD")
        title.setObjectName("title")
        left_layout.addWidget(title)
        
        # Source info
        source = self.data.get('source', 'Unknown')
        cyclone = self.data.get('cyclone_center', None)
        
        info_text = f"Source: {source}\n"
        info_text += f"Grid: {self.data['u'].shape[2]}×{self.data['u'].shape[1]}×{self.data['u'].shape[0]}\n"
        info_text += f"(lon × lat × levels)"
        if cyclone:
            info_text += f"\n\n🌀 Cyclone at: {cyclone[0]}°N, {cyclone[1]}°W"
        
        info_label = QLabel(info_text)
        info_label.setObjectName("info")
        info_label.setWordWrap(True)
        left_layout.addWidget(info_label)
        
        # Altitude slider
        alt_group = QGroupBox("ALTITUDE (PRESSURE LEVEL)")
        alt_layout = QVBoxLayout(alt_group)
        
        self.alt_slider = QSlider(Qt.Horizontal)
        self.alt_slider.setRange(0, len(self.data['level']) - 1)
        self.alt_slider.setValue(0)
        self.alt_slider.valueChanged.connect(self._on_altitude_changed)
        
        self.alt_label = QLabel("")
        self.alt_label.setObjectName("value")
        self._update_alt_label()
        
        alt_layout.addWidget(self.alt_slider)
        alt_layout.addWidget(self.alt_label)
        left_layout.addWidget(alt_group)
        
        # Field selector
        field_group = QGroupBox("FIELD VARIABLE")
        field_layout = QVBoxLayout(field_group)
        
        self.field_combo = QComboBox()
        fields = []
        if 'u' in self.data: fields.append('U-Wind (East-West, m/s)')
        if 'v' in self.data: fields.append('V-Wind (North-South, m/s)')
        if 'temperature' in self.data: fields.append('Temperature (K)')
        if 'geopotential' in self.data: fields.append('Geopotential Height (m)')
        
        self.field_combo.addItems(fields)
        self.field_combo.currentIndexChanged.connect(self._on_field_changed)
        field_layout.addWidget(self.field_combo)
        left_layout.addWidget(field_group)
        
        # Statistics
        stats_group = QGroupBox("SLICE STATISTICS")
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
        
        # Colormap
        cmap_group = QGroupBox("COLORMAP")
        cmap_layout = QVBoxLayout(cmap_group)
        
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(['coolwarm', 'viridis', 'plasma', 'inferno', 'RdBu_r', 'jet'])
        self.cmap_combo.currentTextChanged.connect(self._on_colormap_changed)
        cmap_layout.addWidget(self.cmap_combo)
        left_layout.addWidget(cmap_group)
        
        # What to look for
        look_group = QGroupBox("WHAT TO LOOK FOR")
        look_layout = QVBoxLayout(look_group)
        self.look_label = QLabel("")
        self.look_label.setWordWrap(True)
        self.look_label.setObjectName("info")
        look_layout.addWidget(self.look_label)
        left_layout.addWidget(look_group)
        
        # Compression info
        if 'compression_ratio' in self.data:
            comp_group = QGroupBox("QTT COMPRESSION")
            comp_layout = QVBoxLayout(comp_group)
            comp_info = QLabel(
                f"Ratio: {self.data['compression_ratio']:.1f}×\n"
                f"Error: {self.data.get('reconstruction_error', 0)*100:.4f}%"
            )
            comp_info.setObjectName("value")
            comp_layout.addWidget(comp_info)
            left_layout.addWidget(comp_group)
        
        left_layout.addStretch()
        
        layout.addWidget(left_panel)
        
        # Main viewport
        self.canvas = WeatherCanvas()
        layout.addWidget(self.canvas.native, stretch=1)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
    def _update_alt_label(self):
        """Update the altitude label."""
        level_idx = self.alt_slider.value()
        pressure = self.data['level'][level_idx]
        
        # Approximate altitude from pressure
        altitude_km = 44.3 * (1 - (pressure / 1013.25) ** 0.19)
        
        self.alt_label.setText(f"{pressure:.0f} hPa (~{altitude_km:.1f} km)")
        
    def _on_altitude_changed(self, value: int):
        """Handle altitude slider change."""
        self.current_level = value
        self._update_alt_label()
        self._update_display()
        
    def _on_field_changed(self, index: int):
        """Handle field selection change."""
        fields = ['u', 'v', 'temperature', 'geopotential']
        available = [f for f in fields if f in self.data]
        self.current_field = available[index]
        self._update_display()
        
    def _on_colormap_changed(self, cmap: str):
        """Handle colormap change."""
        self._update_display()
        
    def _update_display(self):
        """Update the visualization."""
        field_data = self.data[self.current_field][self.current_level]
        cmap = self.cmap_combo.currentText()
        
        vmin, vmax = self.canvas.update_field(field_data, cmap)
        
        # Update statistics
        self.min_label.setText(f"Min: {vmin:.2f}")
        self.max_label.setText(f"Max: {vmax:.2f}")
        self.mean_label.setText(f"Mean: {field_data.mean():.2f}")
        
        # Update "what to look for"
        level_idx = self.current_level
        pressure = self.data['level'][level_idx]
        
        look_for = {
            'u': self._get_u_wind_info(pressure),
            'v': self._get_v_wind_info(pressure),
            'temperature': self._get_temp_info(pressure),
            'geopotential': self._get_geopot_info(pressure),
        }
        
        self.look_label.setText(look_for.get(self.current_field, ""))
        
        # Status
        h, w = field_data.shape
        self.status_bar.showMessage(
            f"Field: {self.current_field} | Level: {pressure:.0f} hPa | "
            f"Grid: {w}×{h} | Range: [{vmin:.1f}, {vmax:.1f}]"
        )
    
    def _get_u_wind_info(self, pressure: float) -> str:
        if pressure < 300:
            return (
                "🌬️ UPPER ATMOSPHERE\n\n"
                "• Jet stream cores (>30 m/s)\n"
                "• Polar jet: ~60°N/S\n"
                "• Subtropical jet: ~30°N/S\n"
                "• Red = Westerly (→)\n"
                "• Blue = Easterly (←)"
            )
        elif pressure < 700:
            return (
                "☁️ MID-TROPOSPHERE\n\n"
                "• Steering level for storms\n"
                "• Weaker than upper levels\n"
                "• Cyclone circulation visible\n"
                "• Look for wind shear patterns"
            )
        else:
            return (
                "🌍 NEAR SURFACE\n\n"
                "• Trade winds (tropical easterlies)\n"
                "• Westerlies in mid-latitudes\n"
                "• Surface friction slows winds\n"
                "• Cyclone inflow visible"
            )
    
    def _get_v_wind_info(self, pressure: float) -> str:
        return (
            "↕️ NORTH-SOUTH WIND\n\n"
            "• Generally weaker than U-wind\n"
            "• Red = Northward (↑)\n"
            "• Blue = Southward (↓)\n"
            "• Strong in cyclone regions\n"
            "• Shows meridional transport"
        )
    
    def _get_temp_info(self, pressure: float) -> str:
        if pressure < 300:
            return (
                "❄️ UPPER ATMOSPHERE\n\n"
                "• Tropopause: ~200-220K\n"
                "• Very cold (-60 to -80°C)\n"
                "• Stratospheric warming visible\n"
                "• Temperature inversions"
            )
        elif pressure < 700:
            return (
                "🌡️ MID-TROPOSPHERE\n\n"
                "• Lapse rate: ~6.5K/km\n"
                "• Frontal boundaries visible\n"
                "• Warm/cold air masses\n"
                "• Cyclone thermal structure"
            )
        else:
            return (
                "🌞 NEAR SURFACE\n\n"
                "• Warmest near equator\n"
                "• Land/sea contrast\n"
                "• Diurnal variations\n"
                "• ~288K (15°C) global average"
            )
    
    def _get_geopot_info(self, pressure: float) -> str:
        return (
            "📐 GEOPOTENTIAL HEIGHT\n\n"
            "• Height of pressure surface\n"
            "• Higher = warmer air column\n"
            "• Ridges (high) = fair weather\n"
            "• Troughs (low) = storms\n"
            "• Contours show wind direction"
        )


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║              HYPERTENSOR WEATHER MANIFOLD VIEWER                     ║
║                                                                      ║
║  Visualizing atmospheric data compressed into QTT manifolds          ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    app = QApplication(sys.argv)
    viewer = WeatherViewer()
    viewer.show()
    
    print("\n✅ Weather Viewer launched")
    print("   Move the altitude slider to slice through pressure levels")
    print("   Switch fields to see U-wind, V-wind, Temperature, etc.")
    print("   Look for the cyclone structure in the North Atlantic!")
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
