"""
FieldOps - Physics Operator Library
====================================

Layer 1 of the HyperTensor substrate.

Standard library of field operators that compose to build simulations.
All operators work directly in QTT format - O(d × r²) not O(N).

Operator Categories:
    Differential: Grad, Div, Curl, Laplacian
    Transport: Advect, Diffuse
    Projection: Project (Poisson + divergence-free)
    Forces: Impulse, Buoyancy, Attractor, Stir

Usage:
    from ontic.infra.fieldops import Advect, Diffuse, Project, FieldGraph

    # Compose operators
    graph = FieldGraph()
    graph.add('advect', Advect())
    graph.add('diffuse', Diffuse(viscosity=0.01))
    graph.add('project', Project())
    graph.connect('advect', 'diffuse', 'project')

    # Execute
    field = graph.execute(field, dt=0.01)
"""

from .operators import (  # Base; Differential; Transport; Projection; Forces; Boundary Conditions; Preset graphs
    Advect,
    Attractor,
    BoundaryCondition,
    Buoyancy,
    Curl,
    Diffuse,
    DirichletBC,
    Div,
    FieldGraph,
    Grad,
    GraphNode,
    Impulse,
    Laplacian,
    NeumannBC,
    ObstacleMask,
    Operator,
    OperatorStats,
    PeriodicBC,
    PoissonSolver,
    Project,
    Stir,
    fluid_graph,
    heat_graph,
    smoke_graph,
)

__all__ = [
    # Base
    "Operator",
    "OperatorStats",
    "FieldGraph",
    "GraphNode",
    # Differential
    "Grad",
    "Div",
    "Curl",
    "Laplacian",
    # Transport
    "Advect",
    "Diffuse",
    # Projection
    "Project",
    "PoissonSolver",
    # Forces
    "Impulse",
    "Buoyancy",
    "Attractor",
    "Stir",
    # Boundary Conditions
    "BoundaryCondition",
    "PeriodicBC",
    "DirichletBC",
    "NeumannBC",
    "ObstacleMask",
    # Preset graphs
    "smoke_graph",
    "fluid_graph",
    "heat_graph",
]

__version__ = "0.1.0"
