"""Problem template registry.

Maps ``ProblemClass`` → compiler function and provides discovery
endpoints listing available templates with their required/optional
parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from physics_os.templates.models import GeometryType, ProblemClass


@dataclass(frozen=True, slots=True)
class TemplateInfo:
    """Metadata for a registered problem template."""

    problem_class: ProblemClass
    label: str
    description: str
    supported_geometries: list[GeometryType]
    default_geometry: GeometryType
    required_flow_fields: list[str]
    optional_flow_fields: list[str] = field(default_factory=list)
    example_params: dict[str, Any] = field(default_factory=dict)


class TemplateRegistry:
    """Singleton registry of problem templates.

    Usage::

        registry = TemplateRegistry()
        info = registry.get(ProblemClass.EXTERNAL_FLOW)
        all_templates = registry.list_all()
    """

    _instance: TemplateRegistry | None = None
    _templates: dict[ProblemClass, TemplateInfo]

    def __new__(cls) -> TemplateRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._templates = {}
            cls._instance._populate()
        return cls._instance

    def _populate(self) -> None:
        """Register all built-in templates."""
        self._templates[ProblemClass.EXTERNAL_FLOW] = TemplateInfo(
            problem_class=ProblemClass.EXTERNAL_FLOW,
            label="External Flow",
            description=(
                "Flow over an immersed body (cylinder, airfoil, wedge, rectangle). "
                "Computes drag/lift, pressure distribution, wake structure."
            ),
            supported_geometries=[
                GeometryType.CIRCLE, GeometryType.ELLIPSE, GeometryType.RECTANGLE,
                GeometryType.NACA_AIRFOIL, GeometryType.WEDGE,
                GeometryType.ROUNDED_RECTANGLE,
            ],
            default_geometry=GeometryType.CIRCLE,
            required_flow_fields=["velocity", "fluid"],
            example_params={
                "problem_class": "external_flow",
                "geometry": {"shape": "circle", "params": {"radius": 0.01}},
                "flow": {"velocity": 1.0, "fluid": "air"},
                "quality": "standard",
            },
        )

        self._templates[ProblemClass.INTERNAL_FLOW] = TemplateInfo(
            problem_class=ProblemClass.INTERNAL_FLOW,
            label="Internal Flow",
            description=(
                "Flow through a channel, pipe bend, or backward-facing step. "
                "Computes pressure drop, recirculation, velocity profiles."
            ),
            supported_geometries=[
                GeometryType.BACKWARD_STEP, GeometryType.PIPE_BEND,
                GeometryType.ANNULUS, GeometryType.NONE,
            ],
            default_geometry=GeometryType.BACKWARD_STEP,
            required_flow_fields=["velocity", "fluid"],
            example_params={
                "problem_class": "internal_flow",
                "geometry": {
                    "shape": "backward_step",
                    "params": {"step_height": 0.01, "channel_height": 0.02},
                },
                "flow": {"velocity": 0.5, "fluid": "water"},
            },
        )

        self._templates[ProblemClass.HEAT_TRANSFER] = TemplateInfo(
            problem_class=ProblemClass.HEAT_TRANSFER,
            label="Heat Transfer",
            description=(
                "Forced or natural convection over a body or along a surface. "
                "Computes Nusselt number, heat flux, temperature field."
            ),
            supported_geometries=[
                GeometryType.CIRCLE, GeometryType.FLAT_PLATE,
                GeometryType.FIN_ARRAY, GeometryType.NONE,
            ],
            default_geometry=GeometryType.FLAT_PLATE,
            required_flow_fields=["velocity", "fluid"],
            optional_flow_fields=["temperature"],
            example_params={
                "problem_class": "heat_transfer",
                "geometry": {"shape": "flat_plate", "params": {"length": 0.1}},
                "flow": {"velocity": 2.0, "fluid": "air", "temperature": 350.0},
            },
        )

        self._templates[ProblemClass.WAVE_PROPAGATION] = TemplateInfo(
            problem_class=ProblemClass.WAVE_PROPAGATION,
            label="Wave Propagation",
            description=(
                "Electromagnetic or acoustic wave propagation. "
                "Computes field evolution, reflection, interference patterns."
            ),
            supported_geometries=[GeometryType.NONE, GeometryType.RECTANGLE],
            default_geometry=GeometryType.NONE,
            required_flow_fields=["velocity"],
            example_params={
                "problem_class": "wave_propagation",
                "flow": {"velocity": 343.0, "fluid": "air"},
            },
        )

        self._templates[ProblemClass.NATURAL_CONVECTION] = TemplateInfo(
            problem_class=ProblemClass.NATURAL_CONVECTION,
            label="Natural Convection",
            description=(
                "Buoyancy-driven flow in an enclosure or along a surface. "
                "Computes temperature and velocity fields, Nusselt number."
            ),
            supported_geometries=[GeometryType.NONE, GeometryType.RECTANGLE],
            default_geometry=GeometryType.NONE,
            required_flow_fields=["velocity", "fluid"],
            optional_flow_fields=["temperature"],
            example_params={
                "problem_class": "natural_convection",
                "flow": {"velocity": 0.01, "fluid": "air", "temperature": 350.0},
            },
        )

        self._templates[ProblemClass.BOUNDARY_LAYER] = TemplateInfo(
            problem_class=ProblemClass.BOUNDARY_LAYER,
            label="Boundary Layer",
            description=(
                "Flat-plate boundary layer analysis. "
                "Computes velocity profiles, skin friction, transition location."
            ),
            supported_geometries=[GeometryType.FLAT_PLATE, GeometryType.NONE],
            default_geometry=GeometryType.FLAT_PLATE,
            required_flow_fields=["velocity", "fluid"],
            example_params={
                "problem_class": "boundary_layer",
                "geometry": {"shape": "flat_plate", "params": {"length": 0.5}},
                "flow": {"velocity": 10.0, "fluid": "air"},
            },
        )

        self._templates[ProblemClass.VORTEX_DYNAMICS] = TemplateInfo(
            problem_class=ProblemClass.VORTEX_DYNAMICS,
            label="Vortex Dynamics",
            description=(
                "Vortex shedding, Kármán street, vortex merging. "
                "Computes Strouhal number, vorticity field, frequency spectrum."
            ),
            supported_geometries=[
                GeometryType.CIRCLE, GeometryType.RECTANGLE,
                GeometryType.ELLIPSE,
            ],
            default_geometry=GeometryType.CIRCLE,
            required_flow_fields=["velocity", "fluid"],
            example_params={
                "problem_class": "vortex_dynamics",
                "geometry": {"shape": "circle", "params": {"radius": 0.005}},
                "flow": {"velocity": 0.3, "fluid": "water"},
            },
        )

        self._templates[ProblemClass.CHANNEL_FLOW] = TemplateInfo(
            problem_class=ProblemClass.CHANNEL_FLOW,
            label="Channel Flow",
            description=(
                "Pressure- or shear-driven channel flow (Poiseuille / Couette). "
                "Computes velocity profiles, pressure gradient, friction factor."
            ),
            supported_geometries=[GeometryType.NONE],
            default_geometry=GeometryType.NONE,
            required_flow_fields=["velocity", "fluid"],
            example_params={
                "problem_class": "channel_flow",
                "flow": {"velocity": 1.0, "fluid": "water"},
            },
        )

    def get(self, problem_class: ProblemClass) -> TemplateInfo:
        """Look up template info by problem class."""
        if problem_class not in self._templates:
            raise KeyError(f"No template for {problem_class.value}")
        return self._templates[problem_class]

    def list_all(self) -> list[TemplateInfo]:
        """Return all registered templates."""
        return list(self._templates.values())

    def list_keys(self) -> list[str]:
        """Return sorted list of problem class keys."""
        return sorted(t.value for t in self._templates)
