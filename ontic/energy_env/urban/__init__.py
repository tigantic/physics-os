"""
TensorNet Urban Module - Urban Canyon Wind Physics

Phase 7: The Urban Canyon
=========================

Simulates micro-climate wind tunnel effects in urban environments.
Wind hits buildings, accelerates around corners (Venturi Effect),
and creates dangerous updrafts/downdrafts.

Target Markets:
- Logistics: Amazon Prime Air, Google Wing drone delivery
- Architecture: Wind load analysis, pedestrian comfort
- Urban Air Mobility: Flying taxis, eVTOL corridors

Physics:
- No-slip boundary conditions (zero velocity at walls)
- Conservation of mass (wind accelerates around obstacles)
- Venturi effect (flow speedup in narrow passages)
- Updraft/downdraft generation at building faces

Outputs:
- Kill Zones: High turbulence areas (no-fly)
- Green Lanes: Safe flight corridors
- Wind acceleration maps
"""

from ontic.energy_env.urban.city_gen import BuildingSpec, VoxelCity
from ontic.energy_env.urban.solver import (
                                    FlightSafetyReport,
                                    UrbanFlowSolver,
                                    analyze_flight_safety,
                                    solve_urban_flow,
)

__all__ = [
    "VoxelCity",
    "BuildingSpec",
    "solve_urban_flow",
    "UrbanFlowSolver",
    "analyze_flight_safety",
    "FlightSafetyReport",
]
