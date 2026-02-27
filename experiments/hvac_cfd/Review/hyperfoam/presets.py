"""
HyperFOAM Presets: Ready-to-use room configurations

These presets encapsulate validated HVAC configurations
for common room types.

Example:
    >>> import hyperfoam
    >>> solver = hyperfoam.Solver(hyperfoam.ConferenceRoom())
    >>> solver.solve(duration=300)
"""

from .solver import SolverConfig, Solver
from typing import Tuple


class ConferenceRoom(SolverConfig):
    """
    Standard conference room (12-person).
    
    Dimensions: 9m × 6m × 3m (30ft × 20ft × 10ft)
    Occupancy: 12 seated occupants around central table
    HVAC: 2 ceiling diffusers, floor returns
    
    Validated settings:
        - Supply: 0.8 m/s @ 60° angle
        - Temperature: 20°C supply
        - Result: All ASHRAE metrics pass
    """
    
    def __init__(self):
        super().__init__(
            # Room dimensions
            lx=9.0,
            ly=6.0,
            lz=3.0,
            
            # Grid resolution (balanced accuracy/speed)
            nx=64,
            ny=48,
            nz=24,
            
            # Time stepping
            dt=0.01,
            
            # HVAC (validated settings)
            supply_velocity=0.8,
            supply_angle=60.0,
            supply_temp=20.0,
            
            # Full physics
            enable_thermal=True,
            enable_buoyancy=True,
            enable_co2=True,
            enable_age_of_air=True,
            
            initial_temp=20.0,
        )


class OpenOffice(SolverConfig):
    """
    Open office floor (48 workstations).
    
    Dimensions: 24m × 18m × 3m (80ft × 60ft × 10ft)
    Occupancy: 48 occupants in 6×8 grid
    HVAC: 6 ceiling diffusers, perimeter returns
    """
    
    def __init__(self):
        super().__init__(
            lx=24.0,
            ly=18.0,
            lz=3.0,
            
            nx=96,
            ny=72,
            nz=24,
            
            dt=0.01,
            
            supply_velocity=1.0,
            supply_angle=45.0,
            supply_temp=18.0,  # Slightly cooler for heat load
            
            enable_thermal=True,
            enable_buoyancy=True,
            enable_co2=True,
            enable_age_of_air=True,
            
            initial_temp=22.0,
        )


class ServerRoom(SolverConfig):
    """
    Data center hot/cold aisle configuration.
    
    Dimensions: 12m × 8m × 3m
    Heat load: High (server racks)
    HVAC: Raised floor plenum supply, ceiling returns
    
    Focus: Preventing hot spots, maximizing cooling efficiency.
    """
    
    def __init__(self):
        super().__init__(
            lx=12.0,
            ly=8.0,
            lz=3.0,
            
            nx=64,
            ny=48,
            nz=24,
            
            dt=0.005,  # Smaller dt for high velocities
            
            supply_velocity=2.0,  # High velocity from floor tiles
            supply_angle=0.0,     # Straight up from floor
            supply_temp=16.0,     # Cold supply
            
            enable_thermal=True,
            enable_buoyancy=True,
            enable_co2=False,     # Not relevant for server room
            enable_age_of_air=False,
            
            initial_temp=22.0,
        )


def setup_conference_room(solver: Solver, n_occupants: int = 12) -> None:
    """
    Quick setup helper for conference room.
    
    Adds table, occupants, and HVAC in one call.
    
    Args:
        solver: Solver instance with ConferenceRoom config
        n_occupants: Number of people (even number, split between sides)
    """
    cfg = solver.config
    center = (cfg.lx / 2, cfg.ly / 2)
    
    # Add table
    solver.add_table(center, length=3.66, width=1.22, height=0.76)
    
    # Add occupants
    n_per_side = n_occupants // 2
    solver.add_occupants_around_table(center, table_length=3.66, n_per_side=n_per_side)
    
    # Add HVAC
    solver.add_ceiling_diffusers(n_vents=2)
    solver.add_floor_returns()
    
    print(f"Conference room setup complete: {n_occupants} occupants")


def setup_open_office(solver: Solver, rows: int = 6, cols: int = 8) -> None:
    """
    Quick setup helper for open office.
    
    Adds workstation grid and HVAC.
    """
    cfg = solver.config
    
    # Add workstations in grid
    spacing_x = cfg.lx / (cols + 1)
    spacing_y = cfg.ly / (rows + 1)
    
    for i in range(cols):
        for j in range(rows):
            x = spacing_x * (i + 1)
            y = spacing_y * (j + 1)
            solver.add_person(x, y, 1.0, f"Worker_{i}_{j}")
    
    # Add HVAC
    solver.add_ceiling_diffusers(n_vents=6)
    solver.add_floor_returns()
    
    print(f"Open office setup complete: {rows * cols} workstations")
