"""Problem template system — public re-exports.

``physics_os.templates`` is the entry-point for the Problem Compiler,
which translates high-level physical specifications into executable
QTT simulation configs.
"""

from __future__ import annotations

from physics_os.templates.models import (
    BoundarySpec,
    FlowConditions,
    GeometrySpec,
    GeometryType,
    ProblemClass,
    ProblemSpec,
    ProblemResult,
)
from physics_os.templates.registry import TemplateRegistry
from physics_os.templates.compiler import compile_problem
from physics_os.templates.extractors import (
    BoundaryLayerResult,
    CorrelationComparison,
    DragResult,
    LiftResult,
    NusseltResult,
    PressureDropResult,
    RecirculationResult,
    SkinFrictionResult,
    StrouhalResult,
    VelocityProfileResult,
    compare_drag_cylinder,
    compare_nusselt_cylinder,
    compare_nusselt_flat_plate,
    compare_skin_friction_plate,
    compare_strouhal_cylinder,
    extract_boundary_layer,
    extract_drag,
    extract_lift,
    extract_nusselt,
    extract_pressure_drop,
    extract_recirculation,
    extract_skin_friction,
    extract_strouhal,
    extract_velocity_profile,
)

__all__ = [
    # Models
    "BoundarySpec",
    "FlowConditions",
    "GeometrySpec",
    "GeometryType",
    "ProblemClass",
    "ProblemSpec",
    "ProblemResult",
    # Registry & compiler
    "TemplateRegistry",
    "compile_problem",
    # Extractor results
    "BoundaryLayerResult",
    "CorrelationComparison",
    "DragResult",
    "LiftResult",
    "NusseltResult",
    "PressureDropResult",
    "RecirculationResult",
    "SkinFrictionResult",
    "StrouhalResult",
    "VelocityProfileResult",
    # Extractors
    "extract_boundary_layer",
    "extract_drag",
    "extract_lift",
    "extract_nusselt",
    "extract_pressure_drop",
    "extract_recirculation",
    "extract_skin_friction",
    "extract_strouhal",
    "extract_velocity_profile",
    # Correlation comparisons
    "compare_drag_cylinder",
    "compare_nusselt_cylinder",
    "compare_nusselt_flat_plate",
    "compare_skin_friction_plate",
    "compare_strouhal_cylinder",
]
