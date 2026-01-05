"""
HyperTensor Substrate Layer
===========================

The foundational layer that everything hangs on.

One API. One substrate. Everything else is a client.

Core Components:
    Field       - The Oracle API (sample, slice, step, stats, serialize)
    FieldStats  - Telemetry dashboard (rank, error, divergence, energy, timings)
    FieldBundle - Serialization format with provenance (.htf files)
    BoundedMode - Frame budget enforcement with graceful degradation

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                      CONSUMERS                               │
    │   Renderer    Simulator    AI Env    Audit    Training      │
    └───────────────────────┬─────────────────────────────────────┘
                            │
    ┌───────────────────────▼─────────────────────────────────────┐
    │                    FIELD ORACLE API                          │
    │   sample(points)  slice(spec)  step(dt)  stats()  serialize()│
    └───────────────────────┬─────────────────────────────────────┘
                            │
    ┌───────────────────────▼─────────────────────────────────────┐
    │                    QTT SUBSTRATE                             │
    │   Cores    Contractions    GPU Runtime    Bounded Mode      │
    └─────────────────────────────────────────────────────────────┘

Usage:
    from tensornet.substrate import Field, SliceSpec, StepControls

    # Create a trillion-point field
    field = Field.create(dims=2, bits_per_dim=20, rank=16)

    # Step physics
    field = field.step(dt=0.01)

    # Sample points
    points = torch.rand(1000, 2)
    values = field.sample(points)

    # Get 2D slice for rendering
    spec = SliceSpec(plane='xy', resolution=(1024, 1024))
    buffer = field.slice(spec)

    # Telemetry
    stats = field.stats()
    print(stats.summary())

    # Save/load
    bundle = field.serialize()
    bundle.save('simulation.htf')
"""

from .bounded import (
                      AdaptiveRankController,
                      BoundedMode,
                      BudgetConfig,
                      ContractionCache,
                      ContractionPath,
                      QualityLevel,
)
from .bundle import BundleMetadata, FieldBundle, OperatorLog, TruncationPolicy
from .field import Field, FieldType, SliceSpec, StepControls
from .morton_ops import (
                      SlicedQTT2D,
                      contract_2d_qtt,
                      morton_decode_3d,
                      morton_encode_3d,
                      slice_morton_3d_x_plane,
                      slice_morton_3d_y_plane,
                      slice_morton_3d_z_plane,
)
from .stats import FieldStats, KernelTiming, TelemetryDashboard

__all__ = [
    # Core Oracle
    "Field",
    "FieldType",
    "SliceSpec",
    "StepControls",
    # Telemetry
    "FieldStats",
    "KernelTiming",
    "TelemetryDashboard",
    # Serialization
    "FieldBundle",
    "BundleMetadata",
    "OperatorLog",
    "TruncationPolicy",
    # Bounded Mode
    "BoundedMode",
    "BudgetConfig",
    "ContractionCache",
    "ContractionPath",
    "QualityLevel",
    "AdaptiveRankController",
    # Morton Operations (resolution-independent slicing)
    "SlicedQTT2D",
    "slice_morton_3d_z_plane",
    "slice_morton_3d_y_plane",
    "slice_morton_3d_x_plane",
    "contract_2d_qtt",
    "morton_encode_3d",
    "morton_decode_3d",
]

__version__ = "0.1.0"
