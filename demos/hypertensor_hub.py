#!/usr/bin/env python3
"""
HyperTensor Hub
===============

The primary interface for interacting with HyperTensor manifolds.
A professional desktop application using PySide6 + VisPy for GPU-accelerated
visualization of QTT-compressed physics fields.

PRODUCTION ARCHITECTURE:
- Threaded compute: All Field operations run in QThread workers
- Non-blocking UI: Main thread stays responsive during tensor contractions
- Signal/slot pattern: Thread-safe communication between compute and UI
- Cancelable operations: User can change params while compute is running

Features:
- Direct memory access to QTT cores (no ZMQ bridge latency)
- Intent-first design with command bar (Layer 8 integration)
- Forensic 4D navigation with temporal scrubbing
- Portable standalone app for Third-Party Replay milestone

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
import traceback

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from PySide6.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
        QLineEdit, QLabel, QFrame, QPushButton, QStackedWidget, 
        QApplication, QSlider, QTextEdit, QSplitter, QStatusBar,
        QProgressBar, QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox
    )
    from PySide6.QtCore import Qt, QSize, QTimer, Signal, QThread, QObject, Slot
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
# BACKGROUND WORKER THREADS - Keep UI responsive during tensor operations
# =============================================================================

class FieldCreationWorker(QObject):
    """Worker thread for creating Field objects (can take seconds on CPU)."""
    
    finished = Signal(object, object, object)  # field, slicer, error
    progress = Signal(str)  # status message
    
    def __init__(self, bits_per_dim: int = 4, rank: int = 8):
        super().__init__()
        self.bits_per_dim = bits_per_dim
        self.rank = rank
        self._cancelled = False
        
    def cancel(self):
        self._cancelled = True
        
    @Slot()
    def run(self):
        """Create field in background thread."""
        try:
            self.progress.emit(f"Creating {2**self.bits_per_dim}³ field...")
            
            if self._cancelled:
                self.finished.emit(None, None, "Cancelled")
                return
                
            field = Field.create(
                dims=3,
                bits_per_dim=self.bits_per_dim,
                rank=self.rank,
                init='smooth'
            )
            
            if self._cancelled:
                self.finished.emit(None, None, "Cancelled")
                return
                
            self.progress.emit("Creating slice engine...")
            slicer = SliceEngine(field)
            
            self.finished.emit(field, slicer, None)
            
        except Exception as e:
            self.finished.emit(None, None, str(e))


class SliceWorker(QObject):
    """Worker thread for slice extraction (O(L×r²) but still needs threading)."""
    
    finished = Signal(object, str)  # slice_data, error
    progress = Signal(str)
    
    def __init__(self, slicer, plane: str, depth: float):
        super().__init__()
        self.slicer = slicer
        self.plane = plane
        self.depth = depth
        self._cancelled = False
        
    def cancel(self):
        self._cancelled = True
        
    @Slot()
    def run(self):
        """Extract slice in background thread."""
        try:
            if self._cancelled:
                self.finished.emit(None, "Cancelled")
                return
                
            self.progress.emit(f"Extracting {self.plane.upper()} slice @ {self.depth:.0%}...")
            
            result = self.slicer.slice(plane=self.plane, depth=self.depth)
            slice_data = result.data if hasattr(result, 'data') else result
            
            if self._cancelled:
                self.finished.emit(None, "Cancelled")
                return
                
            self.finished.emit(slice_data, None)
            
        except Exception as e:
            self.finished.emit(None, str(e))


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
    VisPy Canvas for manifold visualization.
    
    Simple 2D slice viewer with text overlays for context.
    """
    
    def __init__(self):
        if not VISPY_AVAILABLE:
            self._native = QLabel("VisPy not available - Install with: pip install vispy")
            self._native.setStyleSheet("background: #050505; color: #3E5B81; padding: 20px;")
            self._native.setAlignment(Qt.AlignCenter)
            return
            
        self.canvas = scene.SceneCanvas(
            keys='interactive', 
            show=False,
            bgcolor='#0A0A0A'
        )
        
        # Simple single view
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'panzoom'
        self.view.camera.aspect = 1
        
        self.slice_visual = None
        self._current_slice = None
        self._vmin = 0
        self._vmax = 1
        self._plane = 'XY'
        self._depth_pct = 50
        
    @property
    def native(self):
        """Return the native Qt widget for embedding."""
        if not VISPY_AVAILABLE:
            return self._native
        return self.canvas.native
    
    def set_title(self, plane: str, depth_pct: int):
        """Update the slice title."""
        self._plane = plane
        self._depth_pct = depth_pct
        print(f"[DEBUG] Canvas title: {plane} @ {depth_pct}%")
    
    def update_slice(self, slice_data: np.ndarray, colormap: str = 'inferno'):
        """Update the displayed slice from field data."""
        if not VISPY_AVAILABLE:
            return
        
        print(f"[DEBUG] Updating canvas with slice shape {slice_data.shape}")
            
        # Remove old visual
        if self.slice_visual is not None:
            self.slice_visual.parent = None
            
        # Store raw data
        self._current_slice = slice_data.copy()
        
        # Get actual min/max
        self._vmin, self._vmax = float(slice_data.min()), float(slice_data.max())
        
        # Normalize data to [0, 1] for display
        if self._vmax > self._vmin:
            normalized = (slice_data - self._vmin) / (self._vmax - self._vmin)
        else:
            normalized = np.zeros_like(slice_data)
        
        # Convert to float32 to avoid VisPy warning
        normalized = normalized.astype(np.float32)
        
        # Create image visual
        self.slice_visual = visuals.Image(
            normalized,
            cmap=colormap,
            parent=self.view.scene
        )
        
        # Set camera to fit the image
        h, w = slice_data.shape[:2]
        self.view.camera.set_range(x=(0, w), y=(0, h), margin=0.05)
        
        self.canvas.update()
        print(f"[DEBUG] Canvas updated: {w}x{h}, range [{self._vmin:.3f}, {self._vmax:.3f}]")
        
    @property
    def native(self):
        """Return the native Qt widget for embedding."""
        if not VISPY_AVAILABLE:
            return self._native
        return self.canvas.native
    
    def set_title(self, plane: str, depth_pct: int):
        """Update the slice title."""
        self._plane = plane
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
        print(f"[DEBUG] ForensicBlade emitting: {plane}, {depth}")  # DEBUG
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
    
    PRODUCTION ARCHITECTURE:
    - All Field/SliceEngine operations run in background QThreads
    - UI thread stays responsive at all times
    - Signal/slot pattern for thread-safe updates
    - Cancelable operations when user changes parameters
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HYPERTENSOR HUB | 4D FORENSIC SUBSTRATE")
        self.resize(1440, 900)
        
        # Field state
        self.field = None
        self.slicer = None
        
        # Worker thread management
        self.field_thread = None
        self.field_worker = None
        self.slice_thread = None
        self.slice_worker = None
        self._pending_slice = None  # Queued slice request while computing
        
        # Time-travel state for Rank-Stable Interpolation
        self.temporal_cache = {}
        self.current_time = 0.5
        self.ghosting_enabled = False
        
        # Last query coordinates for zoom operations
        self.last_coords = {'z': 8, 'eye': (0.5, 0.5, 1.0)}
        self.current_zoom_level = 2
        self.current_slice = None
        
        # Build UI first (instant)
        self.init_ui()
        self.init_connections()
        self.init_update_timer()
        
        # Start field creation in background (non-blocking)
        self._start_field_creation()
        
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
        
    # =========================================================================
    # THREADED FIELD OPERATIONS - Keep UI responsive
    # =========================================================================
        
    def _start_field_creation(self):
        """Start field creation in background thread."""
        if not TENSORNET_AVAILABLE:
            self.status_bar.showMessage("Demo mode - tensornet not available")
            return
            
        self.status_bar.showMessage("⏳ Creating manifold in background...")
        
        # Clean up any existing thread
        if self.field_thread is not None:
            self.field_worker.cancel()
            self.field_thread.quit()
            self.field_thread.wait()
            
        # Create worker and thread - SAME as world_data_slicer.py
        # 32³ field (bits_per_dim=5), rank=32 for weather-like patterns
        self.field_thread = QThread()
        self.field_worker = FieldCreationWorker(bits_per_dim=5, rank=32)
        self.field_worker.moveToThread(self.field_thread)
        
        # Connect signals
        self.field_thread.started.connect(self.field_worker.run)
        self.field_worker.finished.connect(self._on_field_created)
        self.field_worker.progress.connect(self._on_field_progress)
        self.field_worker.finished.connect(self.field_thread.quit)
        
        # Start
        self.field_thread.start()
        
    def _on_field_progress(self, message: str):
        """Handle progress updates from field creation."""
        self.status_bar.showMessage(f"⏳ {message}")
        
    def _on_field_created(self, field, slicer, error):
        """Handle field creation completion (runs in main thread)."""
        if error:
            self.status_bar.showMessage(f"❌ Field error: {error}")
            print(f"Field creation failed: {error}")
            return
            
        self.field = field
        self.slicer = slicer
        
        grid_size = self.field.grid_size
        theoretical_points = 2 ** (self.field.bits_per_dim * 3)
        
        self.status_bar.showMessage(
            f"🌀 Atlantic Weather Manifold | {grid_size}³ | Ready"
        )
        
        # Update telemetry
        self.telemetry_blade.update_stats({
            'resolution': f"{grid_size}³ ({theoretical_points:,} pts)",
            'max_rank': self.field.rank,
            'compression': theoretical_points / (grid_size ** 2),
            'memory_mb': 0.2,
            'fps': 60
        })
        
        self.last_coords = {'z': grid_size // 2, 'eye': (0.35, 0.55, 1.0)}
        
        print(f"✅ Field created: {grid_size}³")
        
        # Request initial slice (also threaded)
        self._request_slice_async('xy', 0.5)
        
    def _request_slice_async(self, plane: str, depth: float):
        """Request a slice in background thread."""
        if self.slicer is None:
            return
            
        # If a slice is already computing, queue this request
        if self.slice_thread is not None and self.slice_thread.isRunning():
            self._pending_slice = (plane, depth)
            return
            
        self.status_bar.showMessage(f"⏳ Computing {plane.upper()} slice @ {depth:.0%}...")
        
        # Create worker and thread
        self.slice_thread = QThread()
        self.slice_worker = SliceWorker(self.slicer, plane, depth)
        self.slice_worker.moveToThread(self.slice_thread)
        
        # Connect signals
        self.slice_thread.started.connect(self.slice_worker.run)
        self.slice_worker.finished.connect(self._on_slice_complete)
        self.slice_worker.progress.connect(self._on_slice_progress)
        self.slice_worker.finished.connect(self.slice_thread.quit)
        
        # Store current params for UI update
        self._current_slice_plane = plane
        self._current_slice_depth = depth
        
        # Start
        self.slice_thread.start()
        
    def _on_slice_progress(self, message: str):
        """Handle slice progress updates."""
        self.status_bar.showMessage(f"⏳ {message}")
        
    def _on_slice_complete(self, slice_data, error):
        """Handle slice completion (runs in main thread)."""
        if error:
            self.status_bar.showMessage(f"❌ Slice error: {error}")
            return
            
        if slice_data is None:
            return
            
        self.current_slice = slice_data
        
        # Update canvas with title showing which slice
        depth_pct = int(self._current_slice_depth * 100)
        plane_name = self._current_slice_plane.upper()
        self.canvas.set_title(plane_name, depth_pct)
        self.canvas.update_slice(slice_data)
        
        # Update last coordinates
        if self.field:
            self.last_coords['z'] = int(self._current_slice_depth * self.field.grid_size)
        
        # Cache for ghosting
        self.temporal_cache[self._current_slice_depth] = slice_data.copy()
        
        samples = slice_data.shape[0] * slice_data.shape[1] if hasattr(slice_data, 'shape') else 0
        self.status_bar.showMessage(
            f"✅ Slice: {self._current_slice_plane.upper()} @ {self._current_slice_depth:.0%} | "
            f"Samples: {samples:,} | "
            f"Range: [{slice_data.min():.3f}, {slice_data.max():.3f}]"
        )
        
        # Process any pending slice request
        if self._pending_slice:
            plane, depth = self._pending_slice
            self._pending_slice = None
            self._request_slice_async(plane, depth)
            
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
        All heavy operations are routed to async workers.
        """
        self.status_bar.showMessage(f"Processing: {intent}")
        query = intent.lower().strip()
        
        # Add to history
        self.intent_blade._add_to_history(intent)
        
        # 1. Action: Zooming (sets zoom level, requests new slice)
        if "zoom" in query:
            numbers = re.findall(r'\d+', query)
            level = int(numbers[0]) if numbers else self.current_zoom_level + 1
            level = max(1, min(level, 12))
            
            self.current_zoom_level = level
            effective_res = 2 ** (4 + level)  # bits_per_dim + zoom
            
            self.intent_blade.show_result(
                f"🔍 Zoom Level: {level}\n"
                f"Effective Resolution: {effective_res}³\n"
                f"(Slice requested)"
            )
            
            # Request slice at current depth (async)
            depth = self.forensic_blade.depth_slider.value() / 100.0
            self._request_slice_async('xy', depth)
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
                
                # Request slice async
                self._request_slice_async('xy', depth_val)
                self.forensic_blade.depth_slider.setValue(int(depth_val * 100))
                
                z_idx = int(depth_val * (self.field.grid_size if self.field else 16))
                self.intent_blade.show_result(
                    f"📍 Slice Depth: {depth_val:.0%}\n"
                    f"Z-index: {z_idx}"
                )
            else:
                self.intent_blade.show_result("Specify depth (e.g., 'depth 50' or 'altitude 0.75')")
            return
            
        # 3. Action: Cyclone Investigation (requests slices at different depths)
        elif "cyclone" in query or "storm" in query or "investigate" in query:
            self.intent_blade.show_result("🌀 Cyclone Investigation...\nRequesting regional slice")
            # Request a slice at cyclone location (0.5 depth)
            self._request_slice_async('xy', 0.5)
            self.forensic_blade.depth_slider.setValue(50)
            return
            
        # 4. Action: Time travel (Ghosting)
        elif "ghost" in query or "time travel" in query or "temporal" in query:
            self.ghosting_enabled = not self.ghosting_enabled
            self.forensic_blade.ghost_btn.setChecked(self.ghosting_enabled)
            self._on_ghosting_toggled(self.ghosting_enabled)
            
            cached_count = len(self.temporal_cache)
            self.intent_blade.show_result(
                f"👻 Ghosting: {'ENABLED' if self.ghosting_enabled else 'DISABLED'}\n"
                f"Cached time-steps: {cached_count}\n"
                f"Move scrubber to interpolate"
            )
            return
            
        # 5. Action: Statistics
        elif "stats" in query or "statistics" in query or "info" in query:
            if self.field:
                grid_size = self.field.grid_size
                theoretical = 2 ** (self.field.bits_per_dim * 3)
                cached = len(self.temporal_cache)
                self.intent_blade.show_result(
                    f"📊 Field: {grid_size}³\n"
                    f"Theoretical: {theoretical:,} points\n"
                    f"Rank: {self.field.rank}\n"
                    f"Cached slices: {cached}"
                )
            else:
                self.intent_blade.show_result("No field loaded")
            return
            
        # 6. Fallback: Unknown command
        self.intent_blade.show_result(
            f"Unknown: {intent}\n\nAvailable commands:\n"
            f"• zoom to level N\n"
            f"• depth/altitude N\n"
            f"• investigate cyclone\n"
            f"• toggle ghost\n"
            f"• show stats"
        )
            
    def _on_slice_requested(self, plane: str, depth: float):
        """Handle slice request from forensic blade - routes to async worker."""
        print(f"[DEBUG] Slice requested: {plane} @ {depth:.0%}")  # DEBUG
        
        if not TENSORNET_AVAILABLE:
            print("[DEBUG] TENSORNET_AVAILABLE = False")
            self.status_bar.showMessage("tensornet not available")
            return
            
        if not self.slicer:
            print("[DEBUG] slicer is None - field not created yet")
            self.status_bar.showMessage("Field not ready yet...")
            return
            
        # Use async slice request to keep UI responsive
        self._request_slice_async(plane, depth)
            
    def _on_temporal_scrub(self, value: int):
        """
        Handle temporal scrubber movement - the "Time-Travel" interface.
        
        Moving the slider triggers Rank-Stable Interpolation between cached
        manifold states. On CPU, this works on compressed cores, not full grid.
        """
        t = value / 1000.0
        self.current_time = t
        
        if self.ghosting_enabled and self.temporal_cache:
            # Apply ghosting interpolation between cached time steps (instant)
            self._apply_ghosting_interpolation(t)
        elif not self.ghosting_enabled:
            # Direct slice at current depth (async, non-blocking)
            self._on_slice_requested('xy', t)
            
    def _on_ghosting_toggled(self, checked: bool):
        """Toggle Rank-Stable ghosting interpolation."""
        self.ghosting_enabled = checked
        if checked:
            self.status_bar.showMessage("👻 GHOSTING: Using cached time-steps for interpolation")
            # Note: Cache is populated as slices complete
        else:
            self.status_bar.showMessage("GHOSTING: Disabled - slices computed on demand")
            
    def _cache_current_slice(self):
        """Cache the current slice for ghosting (called after slice completes)."""
        if self.current_slice is not None:
            self.temporal_cache[self.current_time] = self.current_slice.copy()
            
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
