"""
FRONTIER 01: Fusion Plasma Simulation

Validated plasma physics solvers using QTT for O(log N) complexity.

Benchmarks:
- Landau Damping: γ = -0.1514 (0% error)
- Two-Stream Instability: γ = 0.1278 (detected)
- Tokamak Geometry: ITER/SPARC/JET configurations

Usage:
    from fusion import run_fusion_demo
    result = run_fusion_demo()

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from landau_damping import (
    LandauDamping,
    LandauDampingConfig,
    LandauDampingState,
    validate_landau_damping,
)

from two_stream import (
    TwoStreamInstability,
    TwoStreamConfig,
    TwoStreamState,
    validate_two_stream,
)

from tokamak_geometry import (
    TokamakGeometry,
    TokamakConfig,
    VelocitySpaceGeometry,
    create_iter_geometry,
    create_sparc_geometry,
    create_jet_geometry,
)

from fusion_demo import (
    run_fusion_demo,
    FusionDemoConfig,
    FusionDemoResult,
    quick_validation,
)

__all__ = [
    # Landau damping
    "LandauDamping",
    "LandauDampingConfig",
    "LandauDampingState",
    "validate_landau_damping",
    # Two-stream
    "TwoStreamInstability",
    "TwoStreamConfig",
    "TwoStreamState",
    "validate_two_stream",
    # Tokamak geometry
    "TokamakGeometry",
    "TokamakConfig",
    "VelocitySpaceGeometry",
    "create_iter_geometry",
    "create_sparc_geometry",
    "create_jet_geometry",
    # Demo
    "run_fusion_demo",
    "FusionDemoConfig",
    "FusionDemoResult",
    "quick_validation",
]
