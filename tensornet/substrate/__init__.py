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

from .field import Field, SliceSpec, StepControls, FieldType
from .stats import FieldStats, KernelTiming, TelemetryDashboard
from .bundle import FieldBundle, BundleMetadata, OperatorLog, TruncationPolicy
from .bounded import (
    BoundedMode, 
    BudgetConfig, 
    ContractionCache,
    ContractionPath,
    QualityLevel,
    AdaptiveRankController,
)

__all__ = [
    # Core Oracle
    'Field',
    'FieldType',
    'SliceSpec',
    'StepControls',
    
    # Telemetry
    'FieldStats',
    'KernelTiming',
    'TelemetryDashboard',
    
    # Serialization
    'FieldBundle',
    'BundleMetadata',
    'OperatorLog',
    'TruncationPolicy',
    
    # Bounded Mode
    'BoundedMode',
    'BudgetConfig',
    'ContractionCache',
    'ContractionPath',
    'QualityLevel',
    'AdaptiveRankController',
]

__version__ = '0.1.0'
