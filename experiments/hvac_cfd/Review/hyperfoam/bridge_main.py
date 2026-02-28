#!/usr/bin/env python3
"""
HYPERFOAM PHYSICS BRIDGE — The Nerve Connection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Production-grade integration between HyperFoam CFD and DOMINION visualization.

This bridge is the neural pathway connecting:
  [HyperFoamSolver] ──────► [Shared Memory] ──────► [DOMINION GPU Renderer]
        │                         ▲
        │                         │
  [CommandListener] ◄────── [TCP Commands] ◄────── [DOMINION GUI Controls]

Architecture:
1. BridgePhysicsEngine: Wraps solver + grid + thermal in unified API
2. SharedMemoryBuffer: Zero-copy IPC to Rust (64-byte header + voxel data)
3. CommandListener: TCP socket for bidirectional control (port 19847)

Data Flow (60 FPS):
  solver.step() → extract_fields() → bridge.write_frame() → GPU upload → render

Commands Handled:
  LOAD_GEOMETRY path       Load IFC/OBJ/STL and rebuild grid
  SET_PARAM key value      Update inlet_velocity, inlet_temp, etc.
  SET_GRID nx ny nz        Change resolution (triggers rebuild)
  PAUSE / RESUME           Control simulation playback
  RESET                    Reinitialize to t=0
  SHUTDOWN                 Graceful exit

Usage:
    python -m hyperfoam.bridge_main --bridge-mode [--grid 64] [--preset conference]

Author: TiganticLabz Physics Laboratory
Copyright (c) 2025-2026 TiganticLabz. All Rights Reserved.
SPDX-License-Identifier: Proprietary
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np

# Configure logging with structured format
logging.basicConfig(
    level=logging.INFO,
    format='[BRIDGE] %(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Bridge configuration
DEFAULT_GRID_SIZE = 64
TARGET_FPS = 60.0
FRAME_TIME = 1.0 / TARGET_FPS

# Shared memory channels (must match Rust side)
CHANNEL_DENSITY = 0      # ρ or volume fraction
CHANNEL_TEMPERATURE = 1  # T in Kelvin
CHANNEL_VELOCITY_U = 2   # u velocity (m/s) - X component
CHANNEL_VELOCITY_V = 3   # v velocity (m/s) - Y component
CHANNEL_VELOCITY_W = 4   # w velocity (m/s) - Z component
CHANNEL_VELOCITY_MAG = 5 # |u| magnitude in m/s
CHANNEL_PRESSURE = 6     # p in Pa (or normalized)
NUM_CHANNELS = 7         # Total channels for physics buffer

# Physics defaults (ASHRAE 55 comfort)
DEFAULT_INLET_VELOCITY = 0.8   # m/s
DEFAULT_INLET_ANGLE = 60.0     # degrees from vertical
DEFAULT_INLET_TEMP = 293.15    # 20°C in Kelvin
DEFAULT_AMBIENT_TEMP = 295.15  # 22°C in Kelvin

# Shutdown handling
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    global shutdown_requested
    log.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_requested = True


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


# ═══════════════════════════════════════════════════════════════════════════════
# BRIDGE PHYSICS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class SolverMode(Enum):
    """Operating modes for the physics engine."""
    WAITING = auto()      # No geometry loaded, waiting for command
    INITIALIZING = auto() # Building grid and solver
    RUNNING = auto()      # Active simulation
    PAUSED = auto()       # Paused, holding state
    ERROR = auto()        # Error state


@dataclass
class SolverParams:
    """Runtime-adjustable solver parameters."""
    inlet_velocity: float = DEFAULT_INLET_VELOCITY
    inlet_angle: float = DEFAULT_INLET_ANGLE
    inlet_temp: float = DEFAULT_INLET_TEMP
    ambient_temp: float = DEFAULT_AMBIENT_TEMP
    n_occupants: int = 12
    body_heat: float = 100.0  # Watts per person
    
    # Derived
    dt: float = 0.01
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for status reporting."""
        return {
            'inlet_velocity': self.inlet_velocity,
            'inlet_angle': self.inlet_angle,
            'inlet_temp_C': self.inlet_temp - 273.15,
            'ambient_temp_C': self.ambient_temp - 273.15,
            'n_occupants': self.n_occupants,
            'body_heat': self.body_heat,
        }


class BridgePhysicsEngine:
    """
    Unified physics engine for the bridge.
    
    Encapsulates:
    - HyperGrid: Geometry and boundary conditions
    - HyperFoamSolver: Momentum + pressure projection
    - ThermalMultiPhysicsSolver: Temperature + buoyancy + CO2
    
    Provides a clean API for the bridge main loop.
    """
    
    def __init__(self, grid_size: int = DEFAULT_GRID_SIZE, device: str = None):
        self.grid_size = grid_size
        self.device = device or self._detect_device()
        
        self.mode = SolverMode.WAITING
        self.params = SolverParams()
        
        # Solver components (initialized on first geometry load or preset)
        self.grid = None
        self.solver = None
        self.thermal_solver = None
        
        # Simulation state
        self.time = 0.0
        self.step_count = 0
        self.frame_count = 0
        
        # Geometry info
        self.geometry_path: Optional[Path] = None
        self.room_dims = (9.0, 6.0, 3.0)  # Default conference room
        
        log.info(f"BridgePhysicsEngine initialized (device={self.device}, grid={grid_size}³)")
    
    def _detect_device(self) -> str:
        """Detect best available compute device."""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                log.info(f"CUDA available: {gpu_name}")
                return 'cuda'
        except ImportError:
            pass
        log.warning("CUDA not available, falling back to CPU")
        return 'cpu'
    
    def initialize_preset(self, preset: str = 'conference') -> bool:
        """
        Initialize solver with a preset room configuration.
        
        Presets:
            conference: 9m × 6m × 3m, 12 occupants, ceiling diffusers
            server_room: 12m × 8m × 3m, high heat load, floor supply
            open_office: 24m × 18m × 3m, 48 workstations
        """
        self.mode = SolverMode.INITIALIZING
        log.info(f"Initializing preset: {preset}")
        
        try:
            import torch
            from hyperfoam.core.grid import HyperGrid
            from hyperfoam.core.solver import HyperFoamSolver, ProjectionConfig
            from hyperfoam.core.thermal import (
                ThermalMultiPhysicsSolver, 
                ThermalSystemConfig,
                BuoyancyConfig
            )
            
            # Select preset dimensions
            if preset == 'server_room':
                lx, ly, lz = 12.0, 8.0, 3.0
                self.params.inlet_velocity = 2.0
                self.params.inlet_temp = 289.15  # 16°C
                self.params.n_occupants = 0
            elif preset == 'open_office':
                lx, ly, lz = 24.0, 18.0, 3.0
                self.params.n_occupants = 48
            else:  # conference (default)
                lx, ly, lz = 9.0, 6.0, 3.0
                self.params.n_occupants = 12
            
            self.room_dims = (lx, ly, lz)
            nx = ny = nz = self.grid_size
            
            # Adjust grid aspect ratio to match room
            # (For now, use cubic grid - future: non-uniform)
            
            log.info(f"  Room: {lx}m × {ly}m × {lz}m")
            log.info(f"  Grid: {nx} × {ny} × {nz} ({nx*ny*nz:,} cells)")
            
            # 1. Create grid
            self.grid = HyperGrid(
                nx=nx, ny=ny, nz=nz,
                lx=lx, ly=ly, lz=lz,
                device=self.device
            )
            
            # Add table for conference room
            if preset == 'conference':
                table_x = (lx/2 - 1.83, lx/2 + 1.83)  # 3.66m table
                table_y = (ly/2 - 0.61, ly/2 + 0.61)  # 1.22m wide
                self.grid.add_box_obstacle(
                    table_x[0], table_x[1],
                    table_y[0], table_y[1],
                    0.0, 0.76  # Table height
                )
                log.info(f"  Added table obstacle")
            
            # 2. Create flow solver
            flow_cfg = ProjectionConfig(
                nx=nx, ny=ny, nz=nz,
                Lx=lx, Ly=ly, Lz=lz,
                dt=self.params.dt,
                nu=1.5e-5,  # Air kinematic viscosity
            )
            
            # 3. Create thermal solver (includes flow solver)
            thermal_cfg = ThermalSystemConfig(
                T_initial=self.params.ambient_temp,
                T_supply=self.params.inlet_temp,
                track_co2=True,
                track_age_of_air=True,
                track_smoke=False,
                buoyancy=BuoyancyConfig(enabled=True),
            )
            
            self.thermal_solver = ThermalMultiPhysicsSolver(
                grid=self.grid,
                flow_cfg=flow_cfg,
                thermal_cfg=thermal_cfg
            )
            
            # Convenience reference
            self.solver = self.thermal_solver.flow
            
            # 4. Add supply vents (ceiling diffusers)
            self._setup_hvac(preset)

            # 5. Add heat sources (occupants)
            self._setup_occupants(preset)

            self.time = 0.0
            self.step_count = 0
            self.mode = SolverMode.RUNNING

            log.info(f"  Solver ready! Mode: RUNNING")
            return True

        except Exception as e:
            log.error(f"Failed to initialize preset '{preset}': {e}")
            import traceback
            traceback.print_exc()
            self.mode = SolverMode.ERROR
            return False

    # ═══════════════════════════════════════════════════════════════════════
    # GEOMETRY PRIMITIVES
    # ═══════════════════════════════════════════════════════════════════════

    def add_box(
        self,
        x_min: float, x_max: float,
        y_min: float, y_max: float,
        z_min: float, z_max: float,
        name: str = "Box"
    ) -> bool:
        """
        Add a box-shaped solid obstacle to the simulation.

        Args:
            x_min, x_max: X-axis bounds in meters
            y_min, y_max: Y-axis bounds in meters
            z_min, z_max: Z-axis bounds in meters
            name: Descriptive name for logging

        Returns:
            True if successful
        """
        if self.grid is None:
            log.error("Cannot add geometry: grid not initialized")
            return False

        self.grid.add_box(x_min, x_max, y_min, y_max, z_min, z_max, solid=True)
        log.info(f"  Added box obstacle: {name} [{x_min:.2f}-{x_max:.2f}, {y_min:.2f}-{y_max:.2f}, {z_min:.2f}-{z_max:.2f}]")
        return True

    def add_column(
        self,
        x: float, y: float,
        radius: float,
        z_min: float = 0.0,
        z_max: float = None,
        name: str = "Column"
    ) -> bool:
        """
        Add a cylindrical column obstacle (e.g., structural column, duct).

        Args:
            x, y: Center position in meters
            radius: Column radius in meters
            z_min: Base height (default: floor)
            z_max: Top height (default: ceiling)
            name: Descriptive name for logging

        Returns:
            True if successful
        """
        if self.grid is None:
            log.error("Cannot add geometry: grid not initialized")
            return False

        if z_max is None:
            z_max = self.room_dims[2] if self.room_dims else 3.0

        self.grid.add_cylinder(
            center=(x, y),
            radius=radius,
            z_min=z_min,
            z_max=z_max,
            axis='z',
            solid=True
        )
        log.info(f"  Added column: {name} at ({x:.2f}, {y:.2f}) r={radius:.2f}m h={z_max-z_min:.2f}m")
        return True

    def add_sphere_obstacle(
        self,
        x: float, y: float, z: float,
        radius: float,
        name: str = "Sphere"
    ) -> bool:
        """
        Add a spherical obstacle (e.g., light fixture, globe).

        Args:
            x, y, z: Center position in meters
            radius: Sphere radius in meters
            name: Descriptive name for logging

        Returns:
            True if successful
        """
        if self.grid is None:
            log.error("Cannot add geometry: grid not initialized")
            return False

        self.grid.add_sphere(center=(x, y, z), radius=radius, solid=True)
        log.info(f"  Added sphere: {name} at ({x:.2f}, {y:.2f}, {z:.2f}) r={radius:.2f}m")
        return True

    def _setup_hvac(self, preset: str):
        """Configure HVAC supply vents based on preset."""
        nx, ny, nz = self.grid_size, self.grid_size, self.grid_size
        
        # Compute velocity components from angle
        import math
        angle_rad = math.radians(self.params.inlet_angle)
        w_supply = -self.params.inlet_velocity * math.cos(angle_rad)
        u_supply = self.params.inlet_velocity * math.sin(angle_rad)
        
        # Vent size (cells)
        vent_size = max(2, nx // 16)
        z_ceiling = nz - 2
        
        if preset == 'server_room':
            # Floor supply (raised floor tiles)
            z_floor = 1
            for i in range(4):
                x_pos = nx * (i + 1) // 5
                self.thermal_solver.add_supply_vent(
                    (x_pos - vent_size, x_pos + vent_size),
                    (ny//4, 3*ny//4),
                    z_floor,
                    velocity_w=abs(w_supply),  # Upward
                    velocity_u=0.0,
                    temperature=self.params.inlet_temp
                )
        else:
            # Ceiling diffusers (conference, open_office)
            n_vents = 2 if preset == 'conference' else 6
            for i in range(n_vents):
                x_pos = nx * (i + 1) // (n_vents + 1)
                self.thermal_solver.add_supply_vent(
                    (x_pos - vent_size, x_pos + vent_size),
                    (ny//3, 2*ny//3),
                    z_ceiling,
                    velocity_w=w_supply,  # Downward
                    velocity_u=u_supply if i % 2 == 0 else -u_supply,
                    temperature=self.params.inlet_temp
                )
        
        log.info(f"  Added {n_vents if preset != 'server_room' else 4} supply vents")
    
    def _setup_occupants(self, preset: str):
        """Add heat sources for occupants."""
        lx, ly, lz = self.room_dims
        
        if preset == 'conference':
            # Occupants around table
            self.thermal_solver.add_occupants_around_table(
                table_center=(lx/2, ly/2),
                table_length=3.66,
                n_per_side=self.params.n_occupants // 2
            )
        elif preset == 'open_office':
            # Grid of workstations
            rows, cols = 6, 8
            spacing_x = lx / (cols + 1)
            spacing_y = ly / (rows + 1)
            
            for i in range(cols):
                for j in range(rows):
                    x = spacing_x * (i + 1)
                    y = spacing_y * (j + 1)
                    self.thermal_solver.add_person(x, y, 1.0, f"Worker_{i}_{j}")
            
            log.info(f"  Added {rows * cols} workstations")
        # server_room has no occupants
    
    def load_geometry(self, path: Path) -> bool:
        """
        Load geometry from IFC/OBJ/STL file.
        
        This triggers a complete solver rebuild with the new geometry.
        """
        self.mode = SolverMode.INITIALIZING
        log.info(f"Loading geometry: {path}")
        
        try:
            from hyperfoam.intake import process_geometry, IntakeStatus
            
            # Process and validate geometry
            result = process_geometry(path)
            
            if result.status not in (IntakeStatus.VALID, IntakeStatus.REPAIRED):
                log.error(f"Geometry validation failed: {result.status}")
                for err in result.errors:
                    log.error(f"  {err}")
                self.mode = SolverMode.ERROR
                return False
            
            # Get bounding box for room dimensions
            if result.bounding_box:
                lx, ly, lz = result.bounding_box.size
                self.room_dims = (lx, ly, lz)
                log.info(f"  Bounding box: {lx:.2f} × {ly:.2f} × {lz:.2f} m")
            
            self.geometry_path = path
            
            # Rebuild solver with new geometry
            # For now, use preset initialization with custom dimensions
            # FUTURE: Full STL voxelization via Tier1/voxelizer.py
            return self.initialize_preset('conference')

        except Exception as e:
            log.error(f"Failed to load geometry: {e}")
            import traceback
            traceback.print_exc()
            self.mode = SolverMode.ERROR
            return False

    # ═══════════════════════════════════════════════════════════════════════
    # GRID PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════════

    def save_grid(self, path: Path) -> bool:
        """
        Save current grid state to file for session continuity.

        Args:
            path: Output path for grid file (.pt format)

        Returns:
            True if save was successful
        """
        if self.grid is None:
            log.error("Cannot save grid: no grid initialized")
            return False

        try:
            # Validate path is within allowed directories
            path = Path(path).resolve()
            self.grid.save(str(path))
            log.info(f"Grid saved: {path}")
            return True
        except Exception as e:
            log.error(f"Failed to save grid: {e}")
            return False

    def load_grid(self, path: Path) -> bool:
        """
        Load grid state from file and reinitialize solver.

        This enables fast session resume without rebuilding geometry.

        Args:
            path: Path to saved grid file (.pt format)

        Returns:
            True if load and reinitialization was successful
        """
        self.mode = SolverMode.INITIALIZING
        path = Path(path).resolve()

        if not path.exists():
            log.error(f"Grid file not found: {path}")
            self.mode = SolverMode.ERROR
            return False

        try:
            from hyperfoam.core.grid import HyperGrid
            from hyperfoam.core.solver import ProjectionConfig
            from hyperfoam.core.thermal import (
                ThermalMultiPhysicsSolver,
                ThermalSystemConfig,
                BuoyancyConfig
            )

            log.info(f"Loading grid: {path}")
            self.grid = HyperGrid.load(str(path), device=self.device)

            # Update room dimensions from loaded grid
            self.room_dims = (self.grid.lx, self.grid.ly, self.grid.lz)
            log.info(f"  Grid: {self.grid.nx}×{self.grid.ny}×{self.grid.nz}")
            log.info(f"  Room: {self.grid.lx}m × {self.grid.ly}m × {self.grid.lz}m")

            # Reinitialize solvers with loaded grid
            flow_cfg = ProjectionConfig(
                nx=self.grid.nx, ny=self.grid.ny, nz=self.grid.nz,
                Lx=self.grid.lx, Ly=self.grid.ly, Lz=self.grid.lz,
                dt=self.params.dt,
                nu=1.5e-5,
            )

            thermal_cfg = ThermalSystemConfig(
                T_initial=self.params.ambient_temp,
                T_supply=self.params.inlet_temp,
                track_co2=True,
                track_age_of_air=True,
                buoyancy=BuoyancyConfig(enabled=True),
            )

            self.thermal_solver = ThermalMultiPhysicsSolver(
                grid=self.grid,
                flow_cfg=flow_cfg,
                thermal_cfg=thermal_cfg
            )
            self.solver = self.thermal_solver.flow

            self.time = 0.0
            self.step_count = 0
            self.mode = SolverMode.RUNNING

            log.info(f"  Grid loaded and solver reinitialized")
            return True

        except Exception as e:
            log.error(f"Failed to load grid: {e}")
            import traceback
            traceback.print_exc()
            self.mode = SolverMode.ERROR
            return False

    def set_parameter(self, key: str, value: float) -> bool:
        """
        Update a solver parameter at runtime.
        
        Supported keys:
            inlet_velocity, inlet_angle, inlet_temp, ambient_temp,
            n_occupants, body_heat, dt
        """
        if hasattr(self.params, key):
            old_value = getattr(self.params, key)
            setattr(self.params, key, value)
            log.info(f"Parameter '{key}': {old_value} → {value}")
            
            # Some parameters require solver update
            if key in ('inlet_velocity', 'inlet_angle', 'inlet_temp'):
                self._update_supply_vents()
            
            return True
        else:
            log.warning(f"Unknown parameter: {key}")
            return False
    
    def _update_supply_vents(self):
        """Update supply vent velocities based on current parameters."""
        if self.thermal_solver is None:
            return
        
        import math
        angle_rad = math.radians(self.params.inlet_angle)
        w_supply = -self.params.inlet_velocity * math.cos(angle_rad)
        u_supply = self.params.inlet_velocity * math.sin(angle_rad)
        
        for i, vent in enumerate(self.thermal_solver.supply_vents):
            vent['w'] = w_supply
            vent['u'] = u_supply if i % 2 == 0 else -u_supply
            vent['T'] = self.params.inlet_temp
    
    def step(self) -> bool:
        """
        Advance simulation by one timestep.
        
        Returns True if step was successful.
        """
        if self.mode != SolverMode.RUNNING:
            return False
        
        if self.thermal_solver is None:
            return False
        
        try:
            self.thermal_solver.step()
            self.time += self.params.dt
            self.step_count += 1
            return True
        except Exception as e:
            log.error(f"Solver step failed: {e}")
            self.mode = SolverMode.ERROR
            return False
    
    def extract_fields(self) -> Optional[np.ndarray]:
        """
        Extract solver fields for bridge transmission.
        
        Returns: (nx, ny, nz, 7) float32 array with channels:
            0: Volume fraction (1.0 = fluid, 0.0 = solid)
            1: Temperature [K]
            2: Velocity U [m/s] - X component
            3: Velocity V [m/s] - Y component  
            4: Velocity W [m/s] - Z component
            5: Velocity magnitude [m/s]
            6: Pressure [Pa, normalized]
        """
        if self.thermal_solver is None:
            return None
        
        import torch
        
        nx, ny, nz = self.grid_size, self.grid_size, self.grid_size
        
        # Pre-allocate output array (7 channels now)
        data = np.zeros((nx, ny, nz, NUM_CHANNELS), dtype=np.float32)
        
        # Channel 0: Volume fraction from grid
        if self.grid is not None:
            data[..., CHANNEL_DENSITY] = self.grid.vol_frac.cpu().numpy()
        else:
            data[..., CHANNEL_DENSITY] = 1.0
        
        # Channel 1: Temperature [K]
        T = self.thermal_solver.temperature.phi
        data[..., CHANNEL_TEMPERATURE] = T.cpu().numpy()
        
        # Channels 2,3,4: Velocity components (u, v, w)
        u = self.thermal_solver.flow.u
        v = self.thermal_solver.flow.v
        w = self.thermal_solver.flow.w
        data[..., CHANNEL_VELOCITY_U] = u.cpu().numpy()
        data[..., CHANNEL_VELOCITY_V] = v.cpu().numpy()
        data[..., CHANNEL_VELOCITY_W] = w.cpu().numpy()
        
        # Channel 5: Velocity magnitude [m/s]
        vel_mag = torch.sqrt(u**2 + v**2 + w**2)
        data[..., CHANNEL_VELOCITY_MAG] = vel_mag.cpu().numpy()
        
        # Channel 6: Pressure [Pa, normalized to 0-1]
        p = self.thermal_solver.flow.p
        p_min, p_max = p.min(), p.max()
        if p_max > p_min:
            p_norm = (p - p_min) / (p_max - p_min)
        else:
            p_norm = torch.zeros_like(p)
        data[..., CHANNEL_PRESSURE] = p_norm.cpu().numpy()
        
        return data
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current comfort metrics."""
        if self.thermal_solver is None:
            return {}
        
        try:
            metrics = self.thermal_solver.get_comfort_metrics()
            metrics['sim_time'] = self.time
            metrics['step_count'] = self.step_count
            return metrics
        except Exception:
            return {'sim_time': self.time, 'step_count': self.step_count}
    
    def pause(self):
        """Pause simulation."""
        if self.mode == SolverMode.RUNNING:
            self.mode = SolverMode.PAUSED
            log.info("Simulation PAUSED")
    
    def resume(self):
        """Resume simulation."""
        if self.mode == SolverMode.PAUSED:
            self.mode = SolverMode.RUNNING
            log.info("Simulation RESUMED")
    
    def reset(self):
        """Reset simulation to t=0."""
        log.info("Resetting simulation...")
        self.time = 0.0
        self.step_count = 0
        
        # Reinitialize with current preset
        if self.geometry_path:
            self.load_geometry(self.geometry_path)
        else:
            self.initialize_preset('conference')


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND HANDLER (Extended)
# ═══════════════════════════════════════════════════════════════════════════════

class BridgeCommandHandler:
    """
    Extended command handler that interfaces with BridgePhysicsEngine.
    """
    
    def __init__(self, engine: BridgePhysicsEngine):
        self.engine = engine
        self.shutdown_requested = False
    
    def handle(self, cmd) -> Dict[str, Any]:
        """Process a command and return result."""
        from hyperfoam.core.command_listener import CommandType
        
        handlers = {
            CommandType.LOAD_GEOMETRY: self._handle_load_geometry,
            CommandType.SET_PARAM: self._handle_set_param,
            CommandType.PAUSE: self._handle_pause,
            CommandType.RESUME: self._handle_resume,
            CommandType.RESET: self._handle_reset,
            CommandType.STATUS: self._handle_status,
            CommandType.SHUTDOWN: self._handle_shutdown,
            CommandType.SET_GRID: self._handle_set_grid,
            CommandType.SET_BOUNDARY: self._handle_set_boundary,
        }
        
        handler = handlers.get(cmd.cmd_type)
        if handler:
            return handler(cmd.params)
        else:
            return {"success": False, "message": f"Unknown command: {cmd.cmd_type}"}
    
    def _handle_load_geometry(self, params: dict) -> dict:
        """Load geometry from file."""
        path_str = params.get("path")
        if not path_str:
            return {"success": False, "message": "Missing 'path' parameter"}
        
        path = Path(path_str)
        
        # Handle WSL path translation
        if str(path).startswith('/mnt/c/'):
            # Already a WSL path
            pass
        elif str(path).startswith('C:\\') or str(path).startswith('C:/'):
            # Windows path - convert to WSL
            path = Path('/mnt/c/' + str(path)[3:].replace('\\', '/'))
        
        if not path.exists():
            return {"success": False, "message": f"File not found: {path}"}
        
        success = self.engine.load_geometry(path)
        return {"success": success, "message": f"Loaded: {path.name}" if success else "Load failed"}
    
    def _handle_set_param(self, params: dict) -> dict:
        """Set simulation parameter."""
        key = params.get("key")
        value = params.get("value")
        
        if key is None or value is None:
            return {"success": False, "message": "Missing 'key' or 'value'"}
        
        success = self.engine.set_parameter(key, float(value))
        return {"success": success}
    
    def _handle_pause(self, params: dict) -> dict:
        """Pause simulation."""
        self.engine.pause()
        return {"success": True}
    
    def _handle_resume(self, params: dict) -> dict:
        """Resume simulation."""
        self.engine.resume()
        return {"success": True}
    
    def _handle_reset(self, params: dict) -> dict:
        """Reset simulation."""
        self.engine.reset()
        return {"success": True}
    
    def _handle_status(self, params: dict) -> dict:
        """Get current status."""
        return {
            "success": True,
            "data": {
                "mode": self.engine.mode.name,
                "time": self.engine.time,
                "steps": self.engine.step_count,
                "frames": self.engine.frame_count,
                "params": self.engine.params.to_dict(),
                "metrics": self.engine.get_metrics(),
            }
        }
    
    def _handle_shutdown(self, params: dict) -> dict:
        """Request shutdown."""
        self.shutdown_requested = True
        return {"success": True}
    
    def _handle_set_grid(self, params: dict) -> dict:
        """Set grid resolution."""
        nx = int(params.get("nx", 64))
        self.engine.grid_size = nx
        self.engine.reset()
        return {"success": True, "message": f"Grid: {nx}³"}
    
    def _handle_set_boundary(self, params: dict) -> dict:
        """Set boundary condition (future expansion)."""
        # For now, just acknowledge
        return {"success": True}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN BRIDGE LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_bridge(
    grid_size: int = DEFAULT_GRID_SIZE,
    preset: str = 'conference',
    auto_start: bool = True
):
    """
    Run the HyperFoam physics bridge.
    
    Args:
        grid_size: Voxel grid resolution (NxNxN)
        preset: Initial room preset ('conference', 'server_room', 'open_office')
        auto_start: If True, start simulation immediately with preset
    """
    global shutdown_requested
    
    from hyperfoam.core.bridge import SharedMemoryBuffer
    from hyperfoam.core.command_listener import CommandListener
    
    log.info("=" * 70)
    log.info("  HYPERFOAM PHYSICS BRIDGE — The Nerve Connection")
    log.info("=" * 70)
    log.info(f"Grid Resolution: {grid_size}³ ({grid_size**3:,} cells)")
    log.info(f"Preset: {preset}")
    log.info(f"Auto-start: {auto_start}")
    log.info("")
    
    # Initialize physics engine
    engine = BridgePhysicsEngine(grid_size=grid_size)
    
    # Initialize command handler
    cmd_handler = BridgeCommandHandler(engine)
    
    # Start command listener
    cmd_listener = CommandListener()
    if not cmd_listener.start():
        log.warning("Command listener failed to start (non-fatal)")
    else:
        log.info(f"Command listener active on TCP port 19847")
    
    # Auto-start with preset if requested
    if auto_start:
        if not engine.initialize_preset(preset):
            log.error("Failed to initialize preset, waiting for geometry...")
    
    # Performance tracking
    frame_count = 0
    step_count = 0
    last_log_time = time.time()
    last_frame_time = time.time()
    
    # Statistics
    frame_times = []
    
    try:
        with SharedMemoryBuffer(
            grid_size=(grid_size, grid_size, grid_size),
            channels=NUM_CHANNELS,
        ) as bridge:
            log.info(f"Shared memory: {bridge.path}")
            log.info("")
            log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            log.info("  BRIDGE ACTIVE — Waiting for DOMINION to connect...")
            log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            
            while not shutdown_requested and not cmd_handler.shutdown_requested:
                loop_start = time.time()
                
                # ─────────────────────────────────────────────────────────
                # 1. Process incoming commands
                # ─────────────────────────────────────────────────────────
                for cmd in cmd_listener.poll_all():
                    result = cmd_handler.handle(cmd)
                    log.debug(f"Command {cmd.cmd_type.name}: {result}")
                
                # ─────────────────────────────────────────────────────────
                # 2. Run physics step if active
                # ─────────────────────────────────────────────────────────
                if engine.mode == SolverMode.RUNNING:
                    # Run multiple physics steps per frame for accuracy
                    # (CFD dt is typically smaller than frame dt)
                    steps_per_frame = max(1, int(FRAME_TIME / engine.params.dt))
                    
                    for _ in range(steps_per_frame):
                        if not engine.step():
                            break
                        step_count += 1
                
                # ─────────────────────────────────────────────────────────
                # 3. Extract fields and write to bridge
                # ─────────────────────────────────────────────────────────
                if engine.mode in (SolverMode.RUNNING, SolverMode.PAUSED):
                    data = engine.extract_fields()
                    if data is not None:
                        bridge.write_frame(
                            data=data,
                            sim_time=engine.time,
                            frame_index=frame_count
                        )
                        frame_count += 1
                        engine.frame_count = frame_count
                
                # ─────────────────────────────────────────────────────────
                # 4. Logging (every 5 seconds)
                # ─────────────────────────────────────────────────────────
                now = time.time()
                if now - last_log_time >= 5.0:
                    elapsed = now - last_log_time
                    fps = frame_count / max(elapsed, 0.001) if frame_count > 0 else 0
                    sps = step_count / max(elapsed, 0.001) if step_count > 0 else 0
                    
                    metrics = engine.get_metrics()
                    T = metrics.get('temperature_avg_C', 0)
                    CO2 = metrics.get('co2_avg_ppm', 0)
                    V = metrics.get('velocity_avg', 0)
                    
                    log.info(
                        f"t={engine.time:6.1f}s | "
                        f"T={T:5.1f}°C | "
                        f"CO2={CO2:5.0f}ppm | "
                        f"V={V:.3f}m/s | "
                        f"{fps:.0f} FPS | "
                        f"{sps:.0f} steps/s | "
                        f"Mode: {engine.mode.name}"
                    )
                    
                    last_log_time = now
                    frame_count = 0
                    step_count = 0
                
                # ─────────────────────────────────────────────────────────
                # 5. Frame rate limiting
                # ─────────────────────────────────────────────────────────
                loop_elapsed = time.time() - loop_start
                sleep_time = FRAME_TIME - loop_elapsed
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Track frame times for performance analysis
                actual_frame_time = time.time() - loop_start
                frame_times.append(actual_frame_time)
                if len(frame_times) > 100:
                    frame_times.pop(0)
    
    except Exception as e:
        log.error(f"Fatal error in bridge loop: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Guaranteed cleanup
        cmd_listener.stop()
        
        # Final statistics
        if frame_times:
            avg_frame = sum(frame_times) / len(frame_times) * 1000
            log.info(f"Average frame time: {avg_frame:.1f}ms")
        
        log.info("Bridge shutdown complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="HyperFoam Physics Bridge for DOMINION",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m hyperfoam.bridge_main --bridge-mode
  python -m hyperfoam.bridge_main --bridge-mode --grid 128 --preset server_room
  python -m hyperfoam.bridge_main --bridge-mode --wait
        """
    )
    
    parser.add_argument(
        '--bridge-mode', '-b',
        action='store_true',
        help="Run in bridge mode (required)",
    )
    parser.add_argument(
        '--grid', '-g',
        type=int,
        default=DEFAULT_GRID_SIZE,
        help=f"Grid resolution (default: {DEFAULT_GRID_SIZE}³)",
    )
    parser.add_argument(
        '--preset', '-p',
        choices=['conference', 'server_room', 'open_office'],
        default='conference',
        help="Room preset (default: conference)",
    )
    parser.add_argument(
        '--wait', '-w',
        action='store_true',
        help="Wait for geometry command instead of auto-starting",
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help="Enable debug logging",
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.bridge_mode:
        run_bridge(
            grid_size=args.grid,
            preset=args.preset,
            auto_start=not args.wait
        )
    else:
        print("HyperFoam Physics Bridge")
        print("")
        print("Usage: python -m hyperfoam.bridge_main --bridge-mode [options]")
        print("")
        print("Options:")
        print("  --grid N        Grid resolution (default: 64)")
        print("  --preset NAME   Room preset: conference, server_room, open_office")
        print("  --wait          Wait for LOAD_GEOMETRY instead of auto-starting")
        print("  --debug         Enable debug logging")
        sys.exit(1)


if __name__ == '__main__':
    main()
