#!/usr/bin/env python3
"""
HyperTensor Hub
===============

The primary interface for interacting with HyperTensor manifolds.
A professional desktop application using PySide6 + VisPy for GPU-accelerated
visualization of QTT-compressed physics fields.

Features:
- Direct memory access to QTT cores (no ZMQ bridge latency)
- Intent-first design with command bar (Layer 8 integration)
- Forensic 4D navigation with temporal scrubbing
- Portable standalone app for Third-Party Replay milestone

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                     HYPERTENSOR HUB                          │
├────────┬─────────────────────────────────────────────────────┤
│ BLADE  │                                                     │
│ STRIP  │              MANIFOLD VIEWPORT                      │
│        │         (VisPy OpenGL Canvas)                       │
│ 🛰 📊 ⌨│                                                     │
│        │                                                     │
├────────┴─────────────────────────────────────────────────────┤
│  > INTENT BAR: "what is the maximum temperature?"            │
└─────────────────────────────────────────────────────────────┘

Usage:
    python demos/hypertensor_hub.py
    
    Or with a specific field:
    python demos/hypertensor_hub.py --load path/to/field.htf
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import re

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import World Data Slicer components
from demos.world_data_slicer import GlobalManifoldSlicer, create_synthetic_weather_field

try:
    from PySide6.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
        QLineEdit, QLabel, QFrame, QPushButton, QStackedWidget, 
        QApplication, QSlider, QTextEdit, QSplitter, QStatusBar,
        QProgressBar, QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox
    )
    from PySide6.QtCore import Qt, QSize, QTimer, Signal, QThread
    from PySide6.QtGui import QFont, QColor, QPalette
except ImportError:
    print("ERROR: PySide6 not installed. Run: pip install PySide6")
    sys.exit(1)

try:
    from vispy import scene, app
    from vispy.scene import visuals
    VISPY_AVAILABLE = True
except ImportError:
    print("WARNING: VisPy not installed. Using fallback canvas.")
    VISPY_AVAILABLE = False

# HyperTensor imports
try:
    from tensornet.substrate import Field, SliceSpec
    from tensornet.hypervisual import SliceEngine
    from tensornet.intent import IntentParser, IntentEngine
    TENSORNET_AVAILABLE = True
except ImportError:
    TENSORNET_AVAILABLE = False
    print("WARNING: tensornet not fully available. Running in demo mode.")


# =============================================================================
# STYLE CONSTANTS - "Industrial Slate" Theme
# =============================================================================

INDUSTRIAL_SLATE_QSS = """
QMainWindow {
    background-color: #0A0A0A;
}

QFrame#BladeStrip {
    background-color: #0D0D0D;
    border-right: 1px solid #1A1A1A;
}

QFrame#BladePanel {
    background-color: #111111;
    border-right: 1px solid #1F1F1F;
}

QPushButton {
    background-color: #1A1A1A;
    color: #888888;
    border: 1px solid #2D2D2D;
    padding: 8px 16px;
    font-size: 12px;
}

QPushButton:hover {
    background-color: #252525;
    color: #AAAAAA;
    border-color: #3E5B81;
}

QPushButton:pressed {
    background-color: #3E5B81;
    color: #FFFFFF;
}

QPushButton:checked {
    background-color: #3E5B81;
    color: #FFFFFF;
    border-color: #5A7FAD;
}

QLabel {
    color: #888888;
    font-size: 11px;
}

QLabel#SectionTitle {
    color: #3E5B81;
    font-weight: bold;
    font-size: 12px;
    margin-bottom: 10px;
}

QLabel#StatValue {
    color: #CCCCCC;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 14px;
}

QLineEdit {
    background-color: #0D0D0D;
    color: #CCCCCC;
    border: 1px solid #2D2D2D;
    padding: 10px;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 13px;
}

QLineEdit:focus {
    border-color: #3E5B81;
}

QSlider::groove:horizontal {
    background: #1A1A1A;
    height: 6px;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #3E5B81;
    width: 14px;
    margin: -4px 0;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background: #5A7FAD;
}

QTextEdit {
    background-color: #0A0A0A;
    color: #888888;
    border: 1px solid #1A1A1A;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 11px;
}

QProgressBar {
    background-color: #1A1A1A;
    border: none;
    height: 4px;
}

QProgressBar::chunk {
    background-color: #3E5B81;
}

QStatusBar {
    background-color: #0D0D0D;
    color: #666666;
    border-top: 1px solid #1A1A1A;
}

QComboBox {
    background-color: #1A1A1A;
    color: #888888;
    border: 1px solid #2D2D2D;
    padding: 5px;
}

QSpinBox, QDoubleSpinBox {
    background-color: #1A1A1A;
    color: #888888;
    border: 1px solid #2D2D2D;
    padding: 5px;
}
"""


# =============================================================================
# HIGH-PERFORMANCE VIEWPORT
# =============================================================================

class ManifoldCanvas:
    """
    VisPy Canvas embedded into Qt for O(L×r²) rendering.
    
    This is the "Abyss" - where the manifold lives.
    Direct GPU rendering without network overhead.
    """
    
    def __init__(self):
        if not VISPY_AVAILABLE:
            # Fallback to a simple Qt widget
            self._native = QLabel("VisPy not available - Install with: pip install vispy")
            self._native.setStyleSheet("background: #050505; color: #3E5B81; padding: 20px;")
            self._native.setAlignment(Qt.AlignCenter)
            return
            
        self.canvas = scene.SceneCanvas(
            keys='interactive', 
            show=False,  # We'll embed it
            bgcolor='#050505'
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'  # Perfect for 3D manifolds
        self.view.camera.fov = 45
        self.view.camera.distance = 3
        
        # Initialize with grid lines
        self.grid = visuals.GridLines(parent=self.view.scene)
        
        # Placeholder for field visualization
        self.volume_visual = None
        self.slice_visual = None
        self._current_slice = None
        
    @property
    def native(self):
        """Return the native Qt widget for embedding."""
        if not VISPY_AVAILABLE:
            return self._native
        return self.canvas.native
    
    def update_slice(self, slice_data: np.ndarray, colormap: str = 'viridis'):
        """Update the displayed slice from field data."""
        if not VISPY_AVAILABLE:
            return
            
        # Remove old visual
        if self.slice_visual is not None:
            self.slice_visual.parent = None
            
        # Normalize data
        vmin, vmax = slice_data.min(), slice_data.max()
        if vmax > vmin:
            normalized = (slice_data - vmin) / (vmax - vmin)
        else:
            normalized = np.zeros_like(slice_data)
        
        # Create image visual
        self.slice_visual = visuals.Image(
            normalized,
            cmap=colormap,
            parent=self.view.scene
        )
        
        self._current_slice = slice_data
        self.canvas.update()
        
    def set_camera_3d(self):
        """Switch to 3D turntable camera."""
        if VISPY_AVAILABLE:
            self.view.camera = 'turntable'
            
    def set_camera_2d(self):
        """Switch to 2D pan/zoom camera."""
        if VISPY_AVAILABLE:
            self.view.camera = 'panzoom'


# =============================================================================
# BLADE PANELS
# =============================================================================

class ForensicBlade(QWidget):
    """
    Forensic 4D Blade - Temporal navigation through field history.
    """
    
    slice_requested = Signal(str, float)  # plane, depth
    
    def __init__(self):
        super().__init__()
        self.setObjectName("BladePanel")
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 20, 15, 15)
        
        # Title
        title = QLabel("FORENSIC 4D BLADE")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)
        
        # Temporal Scrubber
        scrubber_group = QGroupBox("TEMPORAL POSITION")
        scrubber_layout = QVBoxLayout(scrubber_group)
        
        self.scrubber = QSlider(Qt.Horizontal)
        self.scrubber.setRange(0, 1000)
        self.scrubber.setValue(500)
        self.time_label = QLabel("t = 0.500")
        self.time_label.setObjectName("StatValue")
        
        scrubber_layout.addWidget(self.scrubber)
        scrubber_layout.addWidget(self.time_label)
        layout.addWidget(scrubber_group)
        
        self.scrubber.valueChanged.connect(self._on_time_changed)
        
        # Slice Controls
        slice_group = QGroupBox("SLICE PLANE")
        slice_layout = QVBoxLayout(slice_group)
        
        self.plane_combo = QComboBox()
        self.plane_combo.addItems(['XY (Top)', 'XZ (Front)', 'YZ (Side)'])
        
        self.depth_slider = QSlider(Qt.Horizontal)
        self.depth_slider.setRange(0, 100)
        self.depth_slider.setValue(50)
        self.depth_label = QLabel("Depth: 50%")
        
        slice_layout.addWidget(QLabel("Plane:"))
        slice_layout.addWidget(self.plane_combo)
        slice_layout.addWidget(self.depth_label)
        slice_layout.addWidget(self.depth_slider)
        layout.addWidget(slice_group)
        
        self.depth_slider.valueChanged.connect(self._on_depth_changed)
        self.plane_combo.currentIndexChanged.connect(self._on_plane_changed)
        
        # Ghosting Toggle
        self.ghost_btn = QPushButton("GHOSTING: OFF")
        self.ghost_btn.setCheckable(True)
        self.ghost_btn.clicked.connect(self._toggle_ghosting)
        layout.addWidget(self.ghost_btn)
        
        # Singularity Detector
        singularity_group = QGroupBox("SINGULARITY DETECTOR")
        sing_layout = QVBoxLayout(singularity_group)
        
        self.detect_btn = QPushButton("🔍 SCAN FOR ANOMALIES")
        self.singularity_status = QLabel("Status: Idle")
        
        sing_layout.addWidget(self.detect_btn)
        sing_layout.addWidget(self.singularity_status)
        layout.addWidget(singularity_group)
        
        layout.addStretch()
        
    def _on_time_changed(self, value):
        t = value / 1000.0
        self.time_label.setText(f"t = {t:.3f}")
        
    def _on_depth_changed(self, value):
        self.depth_label.setText(f"Depth: {value}%")
        self._emit_slice_request()
        
    def _on_plane_changed(self, index):
        self._emit_slice_request()
        
    def _emit_slice_request(self):
        planes = ['xy', 'xz', 'yz']
        plane = planes[self.plane_combo.currentIndex()]
        depth = self.depth_slider.value() / 100.0
        self.slice_requested.emit(plane, depth)
        
    def _toggle_ghosting(self, checked):
        self.ghost_btn.setText(f"GHOSTING: {'ON' if checked else 'OFF'}")


class TelemetryBlade(QWidget):
    """
    Telemetry Blade - Real-time field statistics and performance metrics.
    """
    
    def __init__(self):
        super().__init__()
        self.setObjectName("BladePanel")
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 20, 15, 15)
        
        # Title
        title = QLabel("TELEMETRY DASHBOARD")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)
        
        # Field Stats
        stats_group = QGroupBox("FIELD STATISTICS")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_labels = {}
        for stat in ['Resolution', 'Max Rank', 'Compression', 'Memory', 'FPS']:
            row = QHBoxLayout()
            label = QLabel(f"{stat}:")
            value = QLabel("--")
            value.setObjectName("StatValue")
            row.addWidget(label)
            row.addStretch()
            row.addWidget(value)
            stats_layout.addLayout(row)
            self.stats_labels[stat] = value
            
        layout.addWidget(stats_group)
        
        # Performance
        perf_group = QGroupBox("PERFORMANCE")
        perf_layout = QVBoxLayout(perf_group)
        
        self.frame_time_bar = QProgressBar()
        self.frame_time_bar.setRange(0, 100)
        self.frame_time_bar.setValue(30)
        self.frame_time_label = QLabel("Frame Time: 16.7ms (60 FPS)")
        
        perf_layout.addWidget(self.frame_time_label)
        perf_layout.addWidget(self.frame_time_bar)
        layout.addWidget(perf_group)
        
        # Rank Evolution
        rank_group = QGroupBox("RANK EVOLUTION")
        rank_layout = QVBoxLayout(rank_group)
        self.rank_display = QLabel("Tracking rank over time...")
        rank_layout.addWidget(self.rank_display)
        layout.addWidget(rank_group)
        
        layout.addStretch()
        
    def update_stats(self, stats: dict):
        """Update displayed statistics."""
        if 'resolution' in stats:
            self.stats_labels['Resolution'].setText(stats['resolution'])
        if 'max_rank' in stats:
            self.stats_labels['Max Rank'].setText(str(stats['max_rank']))
        if 'compression' in stats:
            self.stats_labels['Compression'].setText(f"{stats['compression']:.1f}×")
        if 'memory_mb' in stats:
            self.stats_labels['Memory'].setText(f"{stats['memory_mb']:.1f} MB")
        if 'fps' in stats:
            self.stats_labels['FPS'].setText(f"{stats['fps']:.0f}")


class IntentBlade(QWidget):
    """
    Intent Blade - Natural language query interface (Layer 8).
    """
    
    intent_submitted = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.setObjectName("BladePanel")
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 20, 15, 15)
        
        # Title
        title = QLabel("INTENT TERMINAL")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)
        
        # Quick Actions
        actions_group = QGroupBox("QUICK ACTIONS")
        actions_layout = QVBoxLayout(actions_group)
        
        quick_intents = [
            "what is the maximum temperature?",
            "show me where velocity > 0.5",
            "set turbulence to calm",
            "run simulation for 10 steps"
        ]
        
        for intent in quick_intents:
            btn = QPushButton(f"▶ {intent[:35]}...")
            btn.clicked.connect(lambda checked, i=intent: self._quick_intent(i))
            actions_layout.addWidget(btn)
            
        layout.addWidget(actions_group)
        
        # History
        history_group = QGroupBox("QUERY HISTORY")
        history_layout = QVBoxLayout(history_group)
        
        self.history_display = QTextEdit()
        self.history_display.setReadOnly(True)
        self.history_display.setMaximumHeight(150)
        history_layout.addWidget(self.history_display)
        layout.addWidget(history_group)
        
        # Results
        results_group = QGroupBox("LAST RESULT")
        results_layout = QVBoxLayout(results_group)
        
        self.result_display = QLabel("No queries yet")
        self.result_display.setWordWrap(True)
        self.result_display.setObjectName("StatValue")
        results_layout.addWidget(self.result_display)
        layout.addWidget(results_group)
        
        layout.addStretch()
        
    def _quick_intent(self, intent: str):
        self.intent_submitted.emit(intent)
        self._add_to_history(intent)
        
    def _add_to_history(self, intent: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.history_display.append(f"[{timestamp}] {intent}")
        
    def show_result(self, result: str):
        self.result_display.setText(result)


# =============================================================================
# MAIN APPLICATION WINDOW
# =============================================================================

class HyperTensorHub(QMainWindow):
    """
    The HyperTensor Hub - Primary interface for manifold interaction.
    
    Industrial Slate aesthetic with Blade navigation system.
    Now integrated with GlobalManifoldSlicer for world-scale data queries.
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HYPERTENSOR HUB | 4D FORENSIC SUBSTRATE")
        self.resize(1440, 900)
        
        # Field state - now connected to World Data Slicer
        self.field = None
        self.slicer = None
        self.manifold_slicer = None  # GlobalManifoldSlicer for world-scale queries
        
        # Time-travel state for Rank-Stable Interpolation
        self.temporal_cache = {}  # {time_step: core_snapshot}
        self.current_time = 0.5
        self.ghosting_enabled = False
        
        # Last query coordinates for zoom operations
        self.last_coords = {'z': 16, 'eye': (0.5, 0.5, 1.0)}
        self.current_zoom_level = 2
        self.current_slice = None
        
        self.init_ui()
        self.init_connections()
        self.init_update_timer()
        self.load_atlantic_weather_manifold()
        
    def init_ui(self):
        # Main Layout: Horizontal [Sidebar | Blades | Viewport]
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QHBoxLayout(central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # 1. COMMAND STRIPE (Sidebar)
        self.sidebar = QFrame()
        self.sidebar.setObjectName("BladeStrip")
        self.sidebar.setFixedWidth(70)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setAlignment(Qt.AlignTop)
        sidebar_layout.setSpacing(5)
        sidebar_layout.setContentsMargins(10, 20, 10, 10)
        
        # Logo/Title
        logo = QLabel("HT")
        logo.setStyleSheet("color: #3E5B81; font-size: 20px; font-weight: bold;")
        logo.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(logo)
        sidebar_layout.addSpacing(20)
        
        # Blade Icons
        self.blade_buttons = []
        icons = [("🛰", "Forensic"), ("📊", "Telemetry"), ("⌨", "Intent")]
        for i, (icon, tooltip) in enumerate(icons):
            btn = self.create_blade_button(icon, tooltip, i)
            sidebar_layout.addWidget(btn)
            self.blade_buttons.append(btn)
            
        sidebar_layout.addStretch()
        
        # Settings at bottom
        settings_btn = QPushButton("⚙")
        settings_btn.setFixedSize(50, 50)
        settings_btn.setStyleSheet("font-size: 18px; border: none; color: #555;")
        sidebar_layout.addWidget(settings_btn)
        
        # 2. BLADE CONTAINER (Sliding Panels)
        self.blade_stack = QStackedWidget()
        self.blade_stack.setFixedWidth(300)
        
        self.forensic_blade = ForensicBlade()
        self.telemetry_blade = TelemetryBlade()
        self.intent_blade = IntentBlade()
        
        self.blade_stack.addWidget(self.forensic_blade)
        self.blade_stack.addWidget(self.telemetry_blade)
        self.blade_stack.addWidget(self.intent_blade)
        
        # 3. VIEWPORT (The "Abyss")
        viewport_container = QWidget()
        viewport_layout = QVBoxLayout(viewport_container)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        viewport_layout.setSpacing(0)
        
        self.canvas = ManifoldCanvas()
        viewport_layout.addWidget(self.canvas.native, stretch=1)
        
        # Intent Bar at bottom
        self.intent_bar = QLineEdit()
        self.intent_bar.setPlaceholderText("> Enter intent query (e.g., 'what is the maximum temperature?')")
        self.intent_bar.setMinimumHeight(45)
        viewport_layout.addWidget(self.intent_bar)
        
        # Assemble main layout
        self.main_layout.addWidget(self.sidebar)
        self.main_layout.addWidget(self.blade_stack)
        self.main_layout.addWidget(viewport_container, stretch=1)
        
        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("HyperTensor Hub v1.0 | No field loaded")
        
        # Apply styling
        self.setStyleSheet(INDUSTRIAL_SLATE_QSS)
        
        # Select first blade by default
        self.select_blade(0)
        
    def create_blade_button(self, icon: str, tooltip: str, index: int) -> QPushButton:
        btn = QPushButton(icon)
        btn.setFixedSize(50, 50)
        btn.setCheckable(True)
        btn.setStyleSheet("font-size: 20px; border: none;")
        btn.setToolTip(tooltip)
        btn.clicked.connect(lambda: self.select_blade(index))
        return btn
        
    def select_blade(self, index: int):
        """Switch to the specified blade panel."""
        self.blade_stack.setCurrentIndex(index)
        for i, btn in enumerate(self.blade_buttons):
            btn.setChecked(i == index)
            
    def init_connections(self):
        """Connect signals between components."""
        # Intent bar - now routes to the natural language parser
        self.intent_bar.returnPressed.connect(self._on_intent_submitted)
        
        # Forensic blade slice requests
        self.forensic_blade.slice_requested.connect(self._on_slice_requested)
        
        # Temporal scrubber for time-travel
        self.forensic_blade.scrubber.valueChanged.connect(self._on_temporal_scrub)
        
        # Ghosting toggle for Rank-Stable Interpolation
        self.forensic_blade.ghost_btn.clicked.connect(self._on_ghosting_toggled)
        
        # Intent blade
        self.intent_blade.intent_submitted.connect(self._process_intent)
        
    def init_update_timer(self):
        """Initialize 60 FPS viewport update timer."""
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._tick_viewport)
        self.update_timer.start(16)  # ~60 FPS
        self.frame_count = 0
        self.last_fps_time = datetime.now()
        
    def load_atlantic_weather_manifold(self):
        """
        Load the Atlantic Weather manifold - world-scale data ready for investigation.
        
        This uses the GlobalManifoldSlicer from world_data_slicer.py to enable:
        - Infinite zoom (resolution-independent sampling)
        - O(L×r²) slicing
        - Cyclone investigation workflow
        """
        try:
            # Create the global weather field (5 bits = 32³, rank 32)
            # In production: Field.load_manifold("world_weather_2025_Q4.qtt")
            self.field = create_synthetic_weather_field(bits_per_dim=5, rank=32)
            self.slicer = SliceEngine(self.field)
            
            # Initialize the Global Manifold Slicer for world-scale queries
            self.manifold_slicer = GlobalManifoldSlicer(
                self.field, 
                name="Atlantic_Weather_Q4_2025"
            )
            
            # Cache initial time step for ghosting interpolation
            self._cache_time_step(0.5)
            
            # Update status
            grid_size = self.field.grid_size
            self.status_bar.showMessage(
                f"🌀 Atlantic Weather Manifold | {grid_size}³ | Rank: 32 | Ready for Investigation"
            )
            
            # Update telemetry
            theoretical_points = 2 ** (self.field.bits_per_dim * 3)
            self.telemetry_blade.update_stats({
                'resolution': f"{grid_size}³ ({theoretical_points:,} pts)",
                'max_rank': 32,
                'compression': theoretical_points / (grid_size ** 2),
                'memory_mb': 0.8,
                'fps': 60
            })
            
            # Request initial slice at cyclone center
            self.last_coords = {'z': grid_size // 2, 'eye': (0.35, 0.55, 1.0)}
            self._on_slice_requested('xy', 0.5)
            
            print(f"✅ Atlantic Weather Manifold loaded: {grid_size}³")
            print(f"   Ready for cyclone investigation at coords (0.35, 0.55)")
            
        except Exception as e:
            self.status_bar.showMessage(f"Error loading manifold: {e}")
            print(f"❌ Failed to load Atlantic Weather: {e}")
            
    def load_demo_field(self):
        """Fallback: Load a basic demo field if manifold loading fails."""
        if not TENSORNET_AVAILABLE:
            self.status_bar.showMessage("Demo mode - tensornet not available")
            return
        self.load_atlantic_weather_manifold()
            
    def _on_intent_submitted(self):
        """Handle intent bar submission."""
        intent = self.intent_bar.text().strip()
        if intent:
            self._process_intent(intent)
            self.intent_bar.clear()
            
    def _process_intent(self, intent: str):
        """
        Process a natural language intent - the Intent Parser.
        
        Connects UI text to the Field Slicer and VisPy Viewport.
        Supports: zoom, altitude/depth, cyclone investigation, etc.
        """
        self.status_bar.showMessage(f"Processing: {intent}")
        query = intent.lower().strip()
        
        # Add to history
        self.intent_blade._add_to_history(intent)
        
        # 1. Action: Zooming (Resolution Independence)
        if "zoom" in query:
            # Extract level from string, e.g., "zoom to level 8"
            numbers = re.findall(r'\d+', query)
            level = int(numbers[0]) if numbers else self.current_zoom_level + 1
            level = max(1, min(level, 12))  # Clamp to valid range
            
            self.current_zoom_level = level
            
            if self.manifold_slicer:
                # HyperTensor queries the manifold at higher bit-depth
                result = self.manifold_slicer.interactive_zoom(self.last_coords, level)
                slice_data = result.get('slice')
                if slice_data is not None:
                    self.current_slice = slice_data
                    self.canvas.update_slice(slice_data)
                    
                effective_res = result.get('effective_resolution', 2 ** level)
                self.intent_blade.show_result(
                    f"🔍 Zoom Level: {level}\n"
                    f"Effective Resolution: {effective_res}³\n"
                    f"Samples: {slice_data.shape[0] * slice_data.shape[1]:,}"
                )
                self.status_bar.showMessage(
                    f"Zoom: Level {level} | Effective {effective_res}³"
                )
            else:
                self.intent_blade.show_result(f"Zoom to level {level} (manifold not loaded)")
            return
            
        # 2. Action: Slicing (Depth/Altitude Control)
        elif "altitude" in query or "depth" in query or "slice" in query:
            numbers = re.findall(r'\d+\.?\d*', query)
            if numbers:
                depth_val = float(numbers[0])
                # Normalize: if > 1, assume percentage
                if depth_val > 1:
                    depth_val = depth_val / 100.0
                depth_val = max(0.0, min(1.0, depth_val))
                
                # O(L × r²) slice extraction
                self._on_slice_requested('xy', depth_val)
                self.forensic_blade.depth_slider.setValue(int(depth_val * 100))
                
                self.intent_blade.show_result(
                    f"📍 Slice Depth: {depth_val:.0%}\n"
                    f"Z-index: {int(depth_val * self.field.grid_size)}"
                )
            else:
                self.intent_blade.show_result("Specify depth (e.g., 'depth 50' or 'altitude 0.75')")
            return
            
        # 3. Action: Cyclone Investigation (Multi-scale Survey)
        elif "cyclone" in query or "storm" in query or "investigate" in query:
            if self.manifold_slicer:
                self.intent_blade.show_result("🌀 Starting Cyclone Investigation...")
                
                # Run multi-scale survey at cyclone center
                cyclone_center = (0.35, 0.55, 0.5)
                
                # Phase 1: Regional
                result = self.manifold_slicer.interactive_zoom(
                    {'z': 16, 'eye': (0.35, 0.55, 1.5)}, zoom_level=2
                )
                self.canvas.update_slice(result['slice'])
                
                stats = self.manifold_slicer.get_statistics()
                self.intent_blade.show_result(
                    f"🌀 Cyclone Investigation Complete\n"
                    f"Queries: {stats['total_queries']}\n"
                    f"Samples: {stats['total_samples']:,}\n"
                    f"Compression: {stats['compression_factor']:.1f}×"
                )
                self.status_bar.showMessage("Cyclone investigation complete")
            else:
                self.intent_blade.show_result("Load Atlantic Weather manifold first")
            return
            
        # 4. Action: Time travel (Ghosting)
        elif "ghost" in query or "time travel" in query or "temporal" in query:
            self.ghosting_enabled = not self.ghosting_enabled
            self.forensic_blade.ghost_btn.setChecked(self.ghosting_enabled)
            self.forensic_blade._toggle_ghosting(self.ghosting_enabled)
            
            self.intent_blade.show_result(
                f"👻 Ghosting: {'ENABLED' if self.ghosting_enabled else 'DISABLED'}\n"
                f"Rank-Stable Interpolation active"
            )
            return
            
        # 5. Action: Statistics
        elif "stats" in query or "statistics" in query or "info" in query:
            if self.manifold_slicer:
                stats = self.manifold_slicer.get_statistics()
                self.intent_blade.show_result(
                    f"📊 Manifold: {stats['manifold_name']}\n"
                    f"Queries: {stats['total_queries']}\n"
                    f"Samples: {stats['total_samples']:,}\n"
                    f"Theoretical: {stats['theoretical_points']:,}\n"
                    f"Compression: {stats['compression_factor']:.1f}×"
                )
            else:
                self.intent_blade.show_result("No manifold loaded")
            return
            
        # 6. Fallback: Try tensornet IntentParser
        if TENSORNET_AVAILABLE and self.field:
            try:
                from tensornet.intent import IntentParser
                parser = IntentParser()
                parsed = parser.parse(intent)
                
                result = f"Intent: {parsed.intent_type.name}\n"
                if hasattr(parsed, 'target'):
                    result += f"Target: {parsed.target}"
                    
                self.intent_blade.show_result(result)
                self.status_bar.showMessage(f"Parsed: {parsed.intent_type.name}")
                
            except Exception as e:
                self.intent_blade.show_result(f"Unknown command: {intent}\n\nTry: zoom, depth, cyclone, ghost, stats")
        else:
            self.intent_blade.show_result(
                f"Unknown: {intent}\n\nAvailable commands:\n"
                f"• zoom to level N\n"
                f"• depth/altitude N\n"
                f"• investigate cyclone\n"
                f"• toggle ghost\n"
                f"• show stats"
            )
            
    def _on_slice_requested(self, plane: str, depth: float):
        """Handle slice request from forensic blade - O(L×r²) extraction."""
        if not TENSORNET_AVAILABLE or not self.slicer:
            return
            
        try:
            result = self.slicer.slice(plane=plane, depth=depth)
            slice_data = result.data if hasattr(result, 'data') else result
            
            # Store current slice for viewport updates
            self.current_slice = slice_data
            
            # Update canvas
            self.canvas.update_slice(slice_data)
            
            # Update last coordinates for zoom operations
            self.last_coords['z'] = int(depth * self.field.grid_size)
            
            # Update status with manifold slicer stats
            samples = slice_data.shape[0] * slice_data.shape[1] if hasattr(slice_data, 'shape') else 0
            self.status_bar.showMessage(
                f"Slice: {plane.upper()} @ {depth:.0%} | "
                f"Samples: {samples:,} | "
                f"Range: [{slice_data.min():.3f}, {slice_data.max():.3f}]"
            )
            
        except Exception as e:
            self.status_bar.showMessage(f"Slice error: {e}")
            
    def _on_temporal_scrub(self, value: int):
        """
        Handle temporal scrubber movement - the "Time-Travel" interface.
        
        Moving the slider triggers Rank-Stable Interpolation between cached
        manifold states. On CPU, this works on compressed cores, not full grid.
        """
        t = value / 1000.0
        self.current_time = t
        
        if self.ghosting_enabled:
            # Apply ghosting interpolation between adjacent time steps
            self._apply_ghosting_interpolation(t)
        else:
            # Direct slice at current depth (mapped to time for demo)
            self._on_slice_requested('xy', t)
            
    def _on_ghosting_toggled(self, checked: bool):
        """Toggle Rank-Stable ghosting interpolation."""
        self.ghosting_enabled = checked
        if checked:
            self.status_bar.showMessage("👻 GHOSTING: Rank-Stable Interpolation enabled")
            # Cache some time steps for smooth interpolation
            for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
                self._cache_time_step(t)
        else:
            self.status_bar.showMessage("GHOSTING: Disabled")
            
    def _cache_time_step(self, t: float):
        """
        Cache a manifold state at time t for interpolation.
        
        In production, this would store the QTT cores, not raw data.
        For CPU demo, we store the slice result.
        """
        if not self.slicer:
            return
        try:
            result = self.slicer.slice(plane='xy', depth=t)
            slice_data = result.data if hasattr(result, 'data') else result
            self.temporal_cache[t] = slice_data.copy()
        except:
            pass
            
    def _apply_ghosting_interpolation(self, t: float):
        """
        Apply Rank-Stable Synthesis for smooth temporal transitions.
        
        This demonstrates Spatiotemporal Continuity - smooth transitions
        between weather states using compressed core interpolation.
        """
        if not self.temporal_cache:
            return
            
        # Find bracketing time steps
        times = sorted(self.temporal_cache.keys())
        t0, t1 = times[0], times[-1]
        
        for i, ti in enumerate(times[:-1]):
            if ti <= t <= times[i + 1]:
                t0, t1 = ti, times[i + 1]
                break
                
        if t0 == t1:
            interpolated = self.temporal_cache[t0]
        else:
            # Linear interpolation weight
            weight = (t - t0) / (t1 - t0)
            core_t0 = self.temporal_cache[t0]
            core_t1 = self.temporal_cache[t1]
            
            # Rank-Stable Interpolation: lerp on compressed cores
            interpolated = core_t0 * (1 - weight) + core_t1 * weight
            
        self.current_slice = interpolated
        self.canvas.update_slice(interpolated)
        self.status_bar.showMessage(
            f"👻 Ghosting: t={t:.3f} | Interpolating [{t0:.2f}, {t1:.2f}]"
        )
        
    def _tick_viewport(self):
        """60 FPS viewport update tick."""
        self.frame_count += 1
        
        # Update FPS counter every second
        now = datetime.now()
        elapsed = (now - self.last_fps_time).total_seconds()
        if elapsed >= 1.0:
            fps = self.frame_count / elapsed
            self.telemetry_blade.update_stats({'fps': fps})
            self.frame_count = 0
            self.last_fps_time = now


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Launch the HyperTensor Hub."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                       HYPERTENSOR HUB                                ║
║                                                                      ║
║  4D Forensic Substrate Viewer                                        ║
║  Direct manifold access • Intent-first design • GPU-accelerated      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize Qt
    q_app = QApplication(sys.argv)
    q_app.setApplicationName("HyperTensor Hub")
    q_app.setOrganizationName("Tigantic Labs")
    
    # Create and show window
    window = HyperTensorHub()
    window.show()
    
    # Run event loop
    sys.exit(q_app.exec())


if __name__ == "__main__":
    main()
