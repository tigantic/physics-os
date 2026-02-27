"""
Zone: The Fundamental Computational Unit

A Zone is a self-contained CFD domain:
- Owns its geometry (HyperGrid)
- Owns its physics state (velocity, pressure, temperature, CO2)
- Owns its solver (time integration)
- Exposes boundary faces for coupling

The key insight: HyperFOAM's existing Solver already works.
We just need to wrap it in a container that can talk to neighbors.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto


class Face(Enum):
    """Boundary faces of a zone (3D box)."""
    WEST = auto()   # -X
    EAST = auto()   # +X
    SOUTH = auto()  # -Y
    NORTH = auto()  # +Y
    BOTTOM = auto() # -Z
    TOP = auto()    # +Z


@dataclass
class ZoneConfig:
    """Configuration for a single zone."""
    
    # Identity
    name: str = "zone_0"
    zone_id: int = 0
    
    # Grid resolution
    nx: int = 64
    ny: int = 48
    nz: int = 24
    
    # Physical dimensions (meters)
    lx: float = 9.0
    ly: float = 6.0
    lz: float = 3.0
    
    # Position in world coordinates (for visualization)
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # Physics
    dt: float = 0.01
    nu: float = 1.5e-5      # Kinematic viscosity (m²/s)
    alpha: float = 2.2e-5   # Thermal diffusivity (m²/s)
    
    # Turbulence (Smagorinsky LES)
    enable_turbulence: bool = True
    cs: float = 0.17        # Smagorinsky constant (0.1-0.2 typical)
    
    # Velocity limits
    max_velocity: float = 10.0  # Maximum velocity magnitude (m/s)
    
    # Buoyancy (Boussinesq approximation)
    enable_buoyancy: bool = True
    beta: float = 3.4e-3    # Thermal expansion coefficient (1/K) for air at ~20°C
    g: float = 9.81         # Gravitational acceleration (m/s²)
    T_ref_c: float = 20.0   # Reference temperature for buoyancy (°C)
    
    # Initial conditions
    initial_temp_c: float = 22.0
    initial_co2_ppm: float = 400.0
    
    # Boundary conditions (default: walls)
    bc_west: str = "wall"
    bc_east: str = "wall"
    bc_south: str = "wall"
    bc_north: str = "wall"
    bc_bottom: str = "wall"
    bc_top: str = "wall"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        """Validate configuration parameters."""
        import warnings
        
        # Grid dimension validation
        if self.nx < 4 or self.ny < 4 or self.nz < 4:
            raise ValueError(f"Grid dimensions must be at least 4 (got nx={self.nx}, ny={self.ny}, nz={self.nz})")
        
        if self.lx <= 0 or self.ly <= 0 or self.lz <= 0:
            raise ValueError(f"Physical dimensions must be positive (got lx={self.lx}, ly={self.ly}, lz={self.lz})")
        
        # Physics validation
        if self.dt <= 0:
            raise ValueError(f"Timestep must be positive (got dt={self.dt})")
        
        if self.nu <= 0 or self.alpha <= 0:
            raise ValueError(f"Viscosity and thermal diffusivity must be positive")
        
        if self.max_velocity <= 0:
            raise ValueError(f"Maximum velocity must be positive (got {self.max_velocity})")
        
        # CFL warning
        dx = self.lx / self.nx
        dy = self.ly / self.ny
        dz = self.lz / self.nz
        min_dx = min(dx, dy, dz)
        cfl_limit = self.max_velocity * self.dt / min_dx
        
        if cfl_limit > 0.8:
            warnings.warn(
                f"CFL number may exceed stability limit ({cfl_limit:.2f} > 0.8). "
                f"Consider reducing dt or max_velocity.",
                UserWarning
            )
        
        # Temperature validation
        if self.initial_temp_c < -50 or self.initial_temp_c > 80:
            warnings.warn(
                f"Initial temperature {self.initial_temp_c}°C is outside typical HVAC range [-50, 80]°C",
                UserWarning
            )


class Zone:
    """
    A self-contained computational zone.
    
    This wraps the existing HyperFOAM solver with:
    - Explicit boundary face access for portal coupling
    - Mass/momentum/energy flux computation at faces
    - State injection from neighboring zones
    
    Usage:
        zone = Zone(ZoneConfig(name="hallway", nx=32))
        zone.add_inlet(face=Face.WEST, velocity=0.5, temp=18.0)
        zone.add_heat_source(position, power_watts)
        zone.step(dt=0.01)
        
        # Get flux to pass to neighbor
        flux = zone.get_face_flux(Face.EAST)
    """
    
    def __init__(self, config: ZoneConfig):
        self.config = config
        self.name = config.name
        self.zone_id = config.zone_id
        self.device = torch.device(config.device)
        
        # Grid dimensions
        self.nx, self.ny, self.nz = config.nx, config.ny, config.nz
        self.lx, self.ly, self.lz = config.lx, config.ly, config.lz
        self.dx = self.lx / self.nx
        self.dy = self.ly / self.ny
        self.dz = self.lz / self.nz
        
        # Origin for world positioning
        self.origin = torch.tensor(config.origin, device=self.device)
        
        # =====================================================================
        # PHYSICS STATE TENSORS
        # Each zone owns its own memory - no sharing
        # =====================================================================
        
        # Velocity field (m/s)
        self.u = torch.zeros((self.nx, self.ny, self.nz), device=self.device)
        self.v = torch.zeros((self.nx, self.ny, self.nz), device=self.device)
        self.w = torch.zeros((self.nx, self.ny, self.nz), device=self.device)
        
        # Pressure (Pa, relative)
        self.p = torch.zeros((self.nx, self.ny, self.nz), device=self.device)
        
        # Temperature (Kelvin)
        self.T = torch.full(
            (self.nx, self.ny, self.nz), 
            config.initial_temp_c + 273.15,
            device=self.device
        )
        
        # CO2 concentration (ppm)
        self.co2 = torch.full(
            (self.nx, self.ny, self.nz),
            config.initial_co2_ppm,
            device=self.device
        )
        
        # Geometry mask: 1.0 = fluid, 0.0 = solid
        self.solid_mask = torch.zeros((self.nx, self.ny, self.nz), device=self.device)
        self.fluid_mask = torch.ones((self.nx, self.ny, self.nz), device=self.device)
        
        # Heat sources (W/m³)
        self.heat_source = torch.zeros((self.nx, self.ny, self.nz), device=self.device)
        
        # CO2 sources (ppm/s)
        self.co2_source = torch.zeros((self.nx, self.ny, self.nz), device=self.device)
        
        # =====================================================================
        # BOUNDARY FACE STORAGE
        # These are the "portal interfaces" - data exchanged with neighbors
        # =====================================================================
        
        self.portal_faces: Dict[Face, 'PortalInterface'] = {}
        self.inlet_faces: Dict[Face, Dict[str, Any]] = {}
        self.outlet_faces: Dict[Face, Dict[str, Any]] = {}
        
        # Portal masks: track which boundary cells are portal openings (not walls)
        # These masks are 2D per face, 1.0 = portal opening, 0.0 = wall
        self.portal_masks: Dict[Face, torch.Tensor] = {}
        
        # Physics constants
        self.nu = config.nu
        self.alpha = config.alpha
        self.rho = 1.2  # kg/m³
        self.cp = 1005.0  # J/kg·K
        
        # Boussinesq buoyancy parameters
        self.enable_buoyancy = config.enable_buoyancy
        self.beta = config.beta      # Thermal expansion coefficient (1/K)
        self.g = config.g            # Gravitational acceleration (m/s²)
        self.T_ref = config.T_ref_c + 273.15  # Reference temperature (K)
        
        # Time tracking
        self.time = 0.0
        self.step_count = 0
        
        # Mass balance tracking
        self.mass_in = 0.0
        self.mass_out = 0.0
        
    # =========================================================================
    # GEOMETRY MODIFICATION
    # =========================================================================
    
    def add_box_obstacle(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z_range: Tuple[float, float],
        heat_flux: float = 0.0,  # W/m²
        co2_rate: float = 0.0   # ppm/s per m³
    ):
        """Add a solid obstacle (wall, furniture, occupant)."""
        
        # Convert physical coords to grid indices
        i0, i1 = int(x_range[0] / self.dx), int(x_range[1] / self.dx)
        j0, j1 = int(y_range[0] / self.dy), int(y_range[1] / self.dy)
        k0, k1 = int(z_range[0] / self.dz), int(z_range[1] / self.dz)
        
        # Clamp to grid
        i0, i1 = max(0, i0), min(self.nx, i1)
        j0, j1 = max(0, j0), min(self.ny, j1)
        k0, k1 = max(0, k0), min(self.nz, k1)
        
        # Mark as solid
        self.solid_mask[i0:i1, j0:j1, k0:k1] = 1.0
        self.fluid_mask[i0:i1, j0:j1, k0:k1] = 0.0
        
        # Add heat/CO2 at surface (simplified: add to volume)
        if heat_flux > 0:
            volume = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0]) * (z_range[1] - z_range[0])
            surface_area = 2 * ((x_range[1]-x_range[0])*(y_range[1]-y_range[0]) + 
                               (y_range[1]-y_range[0])*(z_range[1]-z_range[0]) +
                               (x_range[1]-x_range[0])*(z_range[1]-z_range[0]))
            # Distribute surface heat into adjacent fluid cells
            heat_per_volume = heat_flux * surface_area / max(volume, 0.01)
            self.heat_source[i0:i1, j0:j1, k0:k1] = heat_per_volume
            
        if co2_rate > 0:
            self.co2_source[i0:i1, j0:j1, k0:k1] = co2_rate
    
    def add_occupant(
        self,
        position: Tuple[float, float, float],
        height: float = 1.2,
        heat_watts: float = 100.0,
        co2_lps: float = 0.005  # liters per second
    ):
        """Add a human occupant as heat + CO2 source (NOT a solid block)."""
        x, y, z = position
        radius = 0.3  # meters
        
        # Convert to grid
        i0 = max(0, int((x - radius) / self.dx))
        i1 = min(self.nx, int((x + radius) / self.dx) + 1)
        j0 = max(0, int((y - radius) / self.dy))
        j1 = min(self.ny, int((y + radius) / self.dy) + 1)
        k0 = max(0, int(z / self.dz))
        k1 = min(self.nz, int((z + height) / self.dz) + 1)
        
        # Volume of source region
        n_cells = max(1, (i1 - i0) * (j1 - j0) * (k1 - k0))
        cell_volume = self.dx * self.dy * self.dz
        total_volume = n_cells * cell_volume
        
        # Heat source: W/m³
        heat_density = heat_watts / total_volume
        self.heat_source[i0:i1, j0:j1, k0:k1] += heat_density
        
        # CO2 source: ppm/s
        # co2_lps = liters CO2 per second
        # Convert to ppm/s in the source volume
        # 1 liter = 1000 cm³, room might be 162 m³ = 162e6 cm³
        # ppm = parts per million of volume
        co2_per_second = co2_lps * 1e6 / total_volume  # rough approximation
        self.co2_source[i0:i1, j0:j1, k0:k1] += co2_per_second

    def add_equipment(
        self,
        position: Tuple[float, float, float],
        size: Tuple[float, float, float],
        heat_watts: float,
        is_solid: bool = True
    ):
        """
        Add equipment as a heat source (T2.06).
        
        Args:
            position: (x, y, z) of corner in meters
            size: (width, depth, height) in meters
            heat_watts: Total heat output in Watts
            is_solid: If True, blocks airflow (e.g., server rack)
        """
        x, y, z = position
        w, d, h = size
        
        x_range = (x, x + w)
        y_range = (y, y + d)
        z_range = (z, z + h)
        
        # Convert to grid indices
        i0 = max(0, int(x / self.dx))
        i1 = min(self.nx, int((x + w) / self.dx) + 1)
        j0 = max(0, int(y / self.dy))
        j1 = min(self.ny, int((y + d) / self.dy) + 1)
        k0 = max(0, int(z / self.dz))
        k1 = min(self.nz, int((z + h) / self.dz) + 1)
        
        # Calculate volume
        n_cells = max(1, (i1 - i0) * (j1 - j0) * (k1 - k0))
        cell_volume = self.dx * self.dy * self.dz
        total_volume = n_cells * cell_volume
        
        # Heat density: W/m³
        heat_density = heat_watts / total_volume
        self.heat_source[i0:i1, j0:j1, k0:k1] += heat_density
        
        # Optionally mark as solid obstacle
        if is_solid:
            self.solid_mask[i0:i1, j0:j1, k0:k1] = 1.0
            self.fluid_mask[i0:i1, j0:j1, k0:k1] = 0.0

    def add_glazing(
        self,
        face: 'Face',
        region: Optional[Tuple[float, float, float, float]] = None,
        shgc: float = 0.25,
        u_value: float = 1.1,
        incident_solar: float = 500.0,
        exterior_temp_c: float = 35.0
    ):
        """
        Add a glazed window with solar heat gain (T2.07).
        
        Solar heat gain is modeled as volumetric heat source in cells
        adjacent to the glazing, simulating absorbed radiation.
        
        Args:
            face: Which wall face (NORTH, SOUTH, EAST, WEST)
            region: (y0, y1, z0, z1) or (x0, x1, z0, z1) in meters
            shgc: Solar Heat Gain Coefficient (0-1), typical 0.25-0.40
            u_value: Thermal transmittance W/(m²·K), typical 1.0-2.0
            incident_solar: Direct + diffuse solar irradiance W/m²
            exterior_temp_c: Outside air temperature °C
            
        ASHRAE Reference:
            Q_solar = SHGC × A × I_solar
            Q_conduction = U × A × (T_ext - T_int)
        """
        from hyperfoam.multizone.zone import Face
        
        # Determine glazing area based on face and region
        if face in (Face.NORTH, Face.SOUTH):
            # Face is perpendicular to Y axis
            if region:
                x0, x1, z0, z1 = region
            else:
                x0, x1 = 0, self.lx
                z0, z1 = 0, self.lz
            area = (x1 - x0) * (z1 - z0)
            depth = self.dy  # Heat applied in first cell layer
            
            # Grid indices for heat application
            i0, i1 = int(x0 / self.dx), int(x1 / self.dx)
            k0, k1 = int(z0 / self.dz), int(z1 / self.dz)
            if face == Face.SOUTH:
                j0, j1 = 0, 1
            else:  # NORTH
                j0, j1 = self.ny - 1, self.ny
                
        elif face in (Face.EAST, Face.WEST):
            # Face is perpendicular to X axis
            if region:
                y0, y1, z0, z1 = region
            else:
                y0, y1 = 0, self.ly
                z0, z1 = 0, self.lz
            area = (y1 - y0) * (z1 - z0)
            depth = self.dx
            
            j0, j1 = int(y0 / self.dy), int(y1 / self.dy)
            k0, k1 = int(z0 / self.dz), int(z1 / self.dz)
            if face == Face.WEST:
                i0, i1 = 0, 1
            else:  # EAST
                i0, i1 = self.nx - 1, self.nx
        else:
            raise ValueError(f"Glazing not supported on face {face}")
        
        # Clamp indices
        i0, i1 = max(0, i0), min(self.nx, i1)
        j0, j1 = max(0, j0), min(self.ny, j1)
        k0, k1 = max(0, k0), min(self.nz, k1)
        
        # Calculate heat gains
        # Solar: SHGC × Area × Irradiance
        q_solar = shgc * area * incident_solar  # Watts
        
        # Conduction: U × Area × ΔT (simplified - assumes interior at initial temp)
        interior_temp_c = self.T[self.nx//2, self.ny//2, self.nz//2].item() - 273.15
        q_conduction = u_value * area * (exterior_temp_c - interior_temp_c)  # Watts
        
        # Total heat to add
        q_total = q_solar + max(0, q_conduction)  # Only add if exterior is warmer
        
        # Volume of affected cells
        n_cells = max(1, (i1 - i0) * (j1 - j0) * (k1 - k0))
        cell_volume = self.dx * self.dy * self.dz
        affected_volume = n_cells * cell_volume
        
        # Heat density W/m³
        heat_density = q_total / affected_volume
        self.heat_source[i0:i1, j0:j1, k0:k1] += heat_density
        
        # Store glazing metadata for reporting
        if not hasattr(self, 'glazings'):
            self.glazings = []
        self.glazings.append({
            'face': face.name,
            'area_m2': area,
            'shgc': shgc,
            'u_value': u_value,
            'q_solar_w': q_solar,
            'q_cond_w': q_conduction,
            'q_total_w': q_total
        })
    
    # =========================================================================
    # INLET/OUTLET DEFINITION
    # =========================================================================
    
    def add_inlet(
        self,
        face: Face,
        velocity: float,
        temperature_c: float = 18.0,
        co2_ppm: float = 400.0,
        region: Optional[Tuple[float, float, float, float]] = None
    ):
        """
        Define an inlet boundary condition.
        
        Args:
            face: Which face (WEST, EAST, etc.)
            velocity: Inflow velocity magnitude (m/s), positive = into zone
            temperature_c: Supply air temperature
            co2_ppm: Supply air CO2
            region: Optional (y0, y1, z0, z1) to limit inlet to part of face
        """
        self.inlet_faces[face] = {
            'velocity': velocity,
            'temperature': temperature_c + 273.15,
            'co2': co2_ppm,
            'region': region
        }
        
    def add_outlet(
        self,
        face: Face,
        region: Optional[Tuple[float, float, float, float]] = None
    ):
        """Define an outlet (zero-gradient pressure BC)."""
        self.outlet_faces[face] = {
            'region': region
        }
    
    def register_portal_region(
        self,
        face: Face,
        region: Tuple[int, int, int, int]
    ):
        """
        Register a portal opening on a face.
        
        This marks the region as "not a wall" so apply_boundary_conditions()
        won't zero out the velocity there.
        
        Args:
            face: Which boundary face has the portal
            region: (j0, j1, k0, k1) cell indices of the opening
        """
        j0, j1, k0, k1 = region
        
        # Create mask if needed (default: all wall = 0)
        if face not in self.portal_masks:
            if face in (Face.WEST, Face.EAST):
                self.portal_masks[face] = torch.zeros((self.ny, self.nz), device=self.device)
            elif face in (Face.SOUTH, Face.NORTH):
                self.portal_masks[face] = torch.zeros((self.nx, self.nz), device=self.device)
            else:  # TOP, BOTTOM
                self.portal_masks[face] = torch.zeros((self.nx, self.ny), device=self.device)
        
        # Mark portal region as 1.0 (opening)
        self.portal_masks[face][j0:j1, k0:k1] = 1.0
    
    # =========================================================================
    # PORTAL COUPLING (THE HARD PART)
    # =========================================================================
    
    def get_face_slice(self, face: Face) -> Tuple[slice, slice, slice]:
        """Get the index slices for a boundary face."""
        if face == Face.WEST:
            return (slice(0, 1), slice(None), slice(None))
        elif face == Face.EAST:
            return (slice(-1, None), slice(None), slice(None))
        elif face == Face.SOUTH:
            return (slice(None), slice(0, 1), slice(None))
        elif face == Face.NORTH:
            return (slice(None), slice(-1, None), slice(None))
        elif face == Face.BOTTOM:
            return (slice(None), slice(None), slice(0, 1))
        elif face == Face.TOP:
            return (slice(None), slice(None), slice(-1, None))
    
    def get_face_normal_velocity(self, face: Face) -> torch.Tensor:
        """Get velocity component normal to face (positive = outflow)."""
        slc = self.get_face_slice(face)
        
        if face in (Face.WEST, Face.EAST):
            vel = self.u[slc]
            return vel if face == Face.EAST else -vel
        elif face in (Face.SOUTH, Face.NORTH):
            vel = self.v[slc]
            return vel if face == Face.NORTH else -vel
        else:  # TOP, BOTTOM
            vel = self.w[slc]
            return vel if face == Face.TOP else -vel
    
    def get_face_flux(self, face: Face) -> Dict[str, torch.Tensor]:
        """
        Compute mass/momentum/energy flux leaving a face.
        
        This is what gets passed to the neighboring zone.
        
        Returns:
            dict with 'mass_flux', 'momentum_flux', 'temperature', 'co2'
        """
        slc = self.get_face_slice(face)
        
        # Normal velocity (positive = leaving this zone)
        u_n = self.get_face_normal_velocity(face)
        
        # Face area per cell
        if face in (Face.WEST, Face.EAST):
            dA = self.dy * self.dz
        elif face in (Face.SOUTH, Face.NORTH):
            dA = self.dx * self.dz
        else:
            dA = self.dx * self.dy
        
        # Mass flux: rho * u_n * dA (kg/s per cell)
        mass_flux = self.rho * u_n * dA
        
        # Only count outflow (positive flux)
        outflow_mask = (u_n > 0).float()
        
        return {
            'normal_velocity': u_n.squeeze(),
            'mass_flux': (mass_flux * outflow_mask).squeeze(),
            'temperature': self.T[slc].squeeze(),
            'co2': self.co2[slc].squeeze(),
            'u': self.u[slc].squeeze(),
            'v': self.v[slc].squeeze(),
            'w': self.w[slc].squeeze(),
            'outflow_mask': outflow_mask.squeeze(),
            'face_area': dA
        }
    
    def inject_face_flux(
        self,
        face: Face,
        flux_data: Dict[str, torch.Tensor],
        portal_region: Optional[Tuple[int, int, int, int]] = None
    ):
        """
        Inject flux from a neighboring zone.
        
        This is the receiving end of portal coupling.
        The flux that LEFT zone A's EAST face becomes
        the INLET for zone B's WEST face.
        
        Args:
            face: The receiving face
            flux_data: Output from neighbor's get_face_flux()
            portal_region: (j0, j1, k0, k1) - which cells receive the flux
        """
        slc = self.get_face_slice(face)
        
        # Map the outflow from neighbor to inflow here
        # Neighbor's outflow velocity becomes our inflow velocity (sign flip)
        incoming_velocity = flux_data['normal_velocity']
        
        # Apply to boundary cells
        if face == Face.WEST:
            # Incoming from EAST of neighbor = positive X velocity here
            if portal_region:
                j0, j1, k0, k1 = portal_region
                self.u[0, j0:j1, k0:k1] = incoming_velocity
                self.T[0, j0:j1, k0:k1] = flux_data['temperature']
                self.co2[0, j0:j1, k0:k1] = flux_data['co2']
            else:
                self.u[0, :, :] = incoming_velocity
                self.T[0, :, :] = flux_data['temperature']
                self.co2[0, :, :] = flux_data['co2']
                
        elif face == Face.EAST:
            if portal_region:
                j0, j1, k0, k1 = portal_region
                self.u[-1, j0:j1, k0:k1] = -incoming_velocity  # Reverse for outflow
                self.T[-1, j0:j1, k0:k1] = flux_data['temperature']
                self.co2[-1, j0:j1, k0:k1] = flux_data['co2']
            else:
                self.u[-1, :, :] = -incoming_velocity
                self.T[-1, :, :] = flux_data['temperature']
                self.co2[-1, :, :] = flux_data['co2']
        
        # Similar for other faces...
        # (Y and Z directions follow same pattern)
    
    # =========================================================================
    # PHYSICS STEP
    # =========================================================================
    
    def apply_boundary_conditions(self):
        """
        Apply inlet/outlet/wall BCs, but SKIP portal regions.
        
        Portal regions are handled separately by Portal.apply_open_bc().
        """
        
        # Create wall masks (1 = wall, 0 = portal/inlet)
        # Start with all ones (all walls), then subtract portal/inlet regions
        
        # WEST face (x=0)
        west_wall_mask = torch.ones((self.ny, self.nz), device=self.device)
        if Face.WEST in self.portal_masks:
            west_wall_mask -= self.portal_masks[Face.WEST]
        if Face.WEST in self.inlet_faces:
            west_wall_mask.zero_()  # Entire face is inlet
        if Face.WEST in self.outlet_faces:
            west_wall_mask.zero_()  # Outlet is not a wall
        
        # EAST face (x=-1)
        east_wall_mask = torch.ones((self.ny, self.nz), device=self.device)
        if Face.EAST in self.portal_masks:
            east_wall_mask -= self.portal_masks[Face.EAST]
        if Face.EAST in self.inlet_faces:
            east_wall_mask.zero_()
        if Face.EAST in self.outlet_faces:
            east_wall_mask.zero_()  # Outlet is not a wall
        
        # SOUTH face (y=0)
        south_wall_mask = torch.ones((self.nx, self.nz), device=self.device)
        if Face.SOUTH in self.portal_masks:
            south_wall_mask -= self.portal_masks[Face.SOUTH]
        if Face.SOUTH in self.inlet_faces:
            south_wall_mask.zero_()
        if Face.SOUTH in self.outlet_faces:
            south_wall_mask.zero_()  # Outlet is not a wall
        
        # NORTH face (y=-1)
        north_wall_mask = torch.ones((self.nx, self.nz), device=self.device)
        if Face.NORTH in self.portal_masks:
            north_wall_mask -= self.portal_masks[Face.NORTH]
        if Face.NORTH in self.inlet_faces:
            north_wall_mask.zero_()
        if Face.NORTH in self.outlet_faces:
            north_wall_mask.zero_()  # Outlet is not a wall
        
        # BOTTOM face (z=0)
        bottom_wall_mask = torch.ones((self.nx, self.ny), device=self.device)
        if Face.BOTTOM in self.portal_masks:
            bottom_wall_mask -= self.portal_masks[Face.BOTTOM]
        if Face.BOTTOM in self.inlet_faces:
            bottom_wall_mask.zero_()
        if Face.BOTTOM in self.outlet_faces:
            bottom_wall_mask.zero_()  # Outlet is not a wall
        
        # TOP face (z=-1)
        top_wall_mask = torch.ones((self.nx, self.ny), device=self.device)
        if Face.TOP in self.portal_masks:
            top_wall_mask -= self.portal_masks[Face.TOP]
        if Face.TOP in self.inlet_faces:
            top_wall_mask.zero_()
        if Face.TOP in self.outlet_faces:
            top_wall_mask.zero_()  # Outlet is not a wall
        
        # Apply wall BCs only where mask = 1 (walls, not portals)
        # X boundaries
        self.u[0, :, :] = self.u[0, :, :] * (1 - west_wall_mask)
        self.u[-1, :, :] = self.u[-1, :, :] * (1 - east_wall_mask)
        
        # Y boundaries
        self.v[:, 0, :] = self.v[:, 0, :] * (1 - south_wall_mask)
        self.v[:, -1, :] = self.v[:, -1, :] * (1 - north_wall_mask)
        
        # Z boundaries
        self.w[:, :, 0] = self.w[:, :, 0] * (1 - bottom_wall_mask)
        self.w[:, :, -1] = self.w[:, :, -1] * (1 - top_wall_mask)
        
        # Apply defined inlets (override wall BC)
        for face, inlet in self.inlet_faces.items():
            vel = inlet['velocity']
            temp = inlet['temperature']
            co2 = inlet['co2']
            
            # For negative velocity (extraction/outlet), use interior T/CO2 (Neumann BC)
            is_extraction = vel < 0
            
            if face == Face.WEST:
                self.u[0, :, :] = vel
                if not is_extraction:
                    self.T[0, :, :] = temp
                    self.co2[0, :, :] = co2
                else:
                    # Neumann BC: copy from interior
                    self.T[0, :, :] = self.T[1, :, :]
                    self.co2[0, :, :] = self.co2[1, :, :]
            elif face == Face.EAST:
                self.u[-1, :, :] = -vel
                if not is_extraction:
                    self.T[-1, :, :] = temp
                    self.co2[-1, :, :] = co2
                else:
                    self.T[-1, :, :] = self.T[-2, :, :]
                    self.co2[-1, :, :] = self.co2[-2, :, :]
            elif face == Face.SOUTH:
                self.v[:, 0, :] = vel
                if not is_extraction:
                    self.T[:, 0, :] = temp
                    self.co2[:, 0, :] = co2
                else:
                    self.T[:, 0, :] = self.T[:, 1, :]
                    self.co2[:, 0, :] = self.co2[:, 1, :]
            elif face == Face.NORTH:
                self.v[:, -1, :] = -vel
                if not is_extraction:
                    self.T[:, -1, :] = temp
                    self.co2[:, -1, :] = co2
                else:
                    self.T[:, -1, :] = self.T[:, -2, :]
                    self.co2[:, -1, :] = self.co2[:, -2, :]
            elif face == Face.BOTTOM:
                self.w[:, :, 0] = vel
                if not is_extraction:
                    self.T[:, :, 0] = temp
                    self.co2[:, :, 0] = co2
                else:
                    self.T[:, :, 0] = self.T[:, :, 1]
                    self.co2[:, :, 0] = self.co2[:, :, 1]
            elif face == Face.TOP:
                self.w[:, :, -1] = -vel
                if not is_extraction:
                    self.T[:, :, -1] = temp
                    self.co2[:, :, -1] = co2
                else:
                    self.T[:, :, -1] = self.T[:, :, -2]
                    self.co2[:, :, -1] = self.co2[:, :, -2]
        
        # Apply outlet BCs (zero-gradient / Neumann BC for all quantities)
        # This allows flow to exit naturally based on interior values
        for face in self.outlet_faces.keys():
            if face == Face.WEST:
                # Zero-gradient: copy from interior
                self.u[0, :, :] = self.u[1, :, :]
                self.v[0, :, :] = self.v[1, :, :]
                self.w[0, :, :] = self.w[1, :, :]
                self.T[0, :, :] = self.T[1, :, :]
                self.co2[0, :, :] = self.co2[1, :, :]
            elif face == Face.EAST:
                self.u[-1, :, :] = self.u[-2, :, :]
                self.v[-1, :, :] = self.v[-2, :, :]
                self.w[-1, :, :] = self.w[-2, :, :]
                self.T[-1, :, :] = self.T[-2, :, :]
                self.co2[-1, :, :] = self.co2[-2, :, :]
            elif face == Face.SOUTH:
                self.u[:, 0, :] = self.u[:, 1, :]
                self.v[:, 0, :] = self.v[:, 1, :]
                self.w[:, 0, :] = self.w[:, 1, :]
                self.T[:, 0, :] = self.T[:, 1, :]
                self.co2[:, 0, :] = self.co2[:, 1, :]
            elif face == Face.NORTH:
                self.u[:, -1, :] = self.u[:, -2, :]
                self.v[:, -1, :] = self.v[:, -2, :]
                self.w[:, -1, :] = self.w[:, -2, :]
                self.T[:, -1, :] = self.T[:, -2, :]
                self.co2[:, -1, :] = self.co2[:, -2, :]
            elif face == Face.BOTTOM:
                self.u[:, :, 0] = self.u[:, :, 1]
                self.v[:, :, 0] = self.v[:, :, 1]
                self.w[:, :, 0] = self.w[:, :, 1]
                self.T[:, :, 0] = self.T[:, :, 1]
                self.co2[:, :, 0] = self.co2[:, :, 1]
            elif face == Face.TOP:
                self.u[:, :, -1] = self.u[:, :, -2]
                self.v[:, :, -1] = self.v[:, :, -2]
                self.w[:, :, -1] = self.w[:, :, -2]
                self.T[:, :, -1] = self.T[:, :, -2]
                self.co2[:, :, -1] = self.co2[:, :, -2]
        
        # Zero velocity inside solid obstacles
        self.u *= self.fluid_mask
        self.v *= self.fluid_mask
        self.w *= self.fluid_mask
    
    def advect_field_stable(self, phi: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Stable second-order TVD advection with van Leer limiter.
        
        Uses MUSCL (Monotone Upstream-centered Schemes for Conservation Laws)
        with van Leer flux limiter to achieve second-order accuracy while
        maintaining monotonicity (no spurious oscillations).
        """
        phi_new = phi.clone()
        
        # Compute CFL number for stability check
        max_vel = max(
            self.u.abs().max().item(),
            self.v.abs().max().item(), 
            self.w.abs().max().item(),
            1e-10
        )
        cfl = max_vel * dt / min(self.dx, self.dy, self.dz)
        
        if cfl > 0.5:
            # Subcycle for stability
            n_sub = int(np.ceil(cfl / 0.4))
            dt_sub = dt / n_sub
            for _ in range(n_sub):
                phi_new = self._advect_step_tvd(phi_new, dt_sub)
            return phi_new
        else:
            return self._advect_step_tvd(phi, dt)
    
    def _van_leer_limiter(self, r: torch.Tensor) -> torch.Tensor:
        """
        Van Leer flux limiter: ψ(r) = (r + |r|) / (1 + |r|)
        
        Returns 0 for negative r (reverting to first-order upwind),
        approaches 2 for large positive r (second-order central).
        """
        return (r + torch.abs(r)) / (1 + torch.abs(r) + 1e-10)
    
    def _advect_step_tvd(self, phi: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Single advection substep with TVD (van Leer) scheme.
        
        Uses a simpler MUSCL formulation that avoids dimension mismatches.
        """
        # Helper: get neighbors with replicate padding
        def get_neighbors(f, dim):
            """Get f[i-1], f[i], f[i+1], f[i+2] along dimension dim"""
            if dim == 0:
                fm1 = torch.cat([f[0:1], f[:-1]], dim=0)
                fp1 = torch.cat([f[1:], f[-1:]], dim=0)
                fp2 = torch.cat([f[2:], f[-1:], f[-1:]], dim=0)
            elif dim == 1:
                fm1 = torch.cat([f[:, 0:1], f[:, :-1]], dim=1)
                fp1 = torch.cat([f[:, 1:], f[:, -1:]], dim=1)
                fp2 = torch.cat([f[:, 2:], f[:, -1:], f[:, -1:]], dim=1)
            else:
                fm1 = torch.cat([f[:, :, 0:1], f[:, :, :-1]], dim=2)
                fp1 = torch.cat([f[:, :, 1:], f[:, :, -1:]], dim=2)
                fp2 = torch.cat([f[:, :, 2:], f[:, :, -1:], f[:, :, -1:]], dim=2)
            return fm1, f, fp1, fp2
        
        # X direction with van Leer limiter
        phi_xm1, phi_x, phi_xp1, phi_xp2 = get_neighbors(phi, 0)
        
        # Forward and backward differences
        d_fwd = phi_xp1 - phi_x  # phi[i+1] - phi[i]
        d_bwd = phi_x - phi_xm1  # phi[i] - phi[i-1]
        d_ffwd = phi_xp2 - phi_xp1  # phi[i+2] - phi[i+1]
        
        # Smoothness ratio for u > 0 (looking backward)
        r_pos = d_bwd / (d_fwd + 1e-10)
        # Smoothness ratio for u < 0 (looking forward) 
        r_neg = d_ffwd / (d_fwd + 1e-10)
        
        # Limited slopes
        psi_pos = self._van_leer_limiter(r_pos)
        psi_neg = self._van_leer_limiter(r_neg)
        
        # Reconstructed values at face i+1/2
        phi_L = phi_x + 0.5 * psi_pos * d_fwd  # Left state (from cell i)
        phi_R = phi_xp1 - 0.5 * psi_neg * d_fwd  # Right state (from cell i+1)
        
        # Upwind selection
        phi_face = torch.where(self.u > 0, phi_L, phi_R)
        
        # Flux at face i+1/2
        flux_x = self.u * phi_face
        
        # Get flux at face i-1/2 by shifting
        flux_xm = torch.cat([flux_x[0:1], flux_x[:-1]], dim=0)
        
        # Flux divergence
        flux_x_div = (flux_x - flux_xm) / self.dx
        
        # Y direction
        phi_ym1, phi_y, phi_yp1, phi_yp2 = get_neighbors(phi, 1)
        d_fwd_y = phi_yp1 - phi_y
        d_bwd_y = phi_y - phi_ym1
        d_ffwd_y = phi_yp2 - phi_yp1
        
        r_pos_y = d_bwd_y / (d_fwd_y + 1e-10)
        r_neg_y = d_ffwd_y / (d_fwd_y + 1e-10)
        psi_pos_y = self._van_leer_limiter(r_pos_y)
        psi_neg_y = self._van_leer_limiter(r_neg_y)
        
        phi_L_y = phi_y + 0.5 * psi_pos_y * d_fwd_y
        phi_R_y = phi_yp1 - 0.5 * psi_neg_y * d_fwd_y
        phi_face_y = torch.where(self.v > 0, phi_L_y, phi_R_y)
        
        flux_y = self.v * phi_face_y
        flux_ym = torch.cat([flux_y[:, 0:1], flux_y[:, :-1]], dim=1)
        flux_y_div = (flux_y - flux_ym) / self.dy
        
        # Z direction
        phi_zm1, phi_z, phi_zp1, phi_zp2 = get_neighbors(phi, 2)
        d_fwd_z = phi_zp1 - phi_z
        d_bwd_z = phi_z - phi_zm1
        d_ffwd_z = phi_zp2 - phi_zp1
        
        r_pos_z = d_bwd_z / (d_fwd_z + 1e-10)
        r_neg_z = d_ffwd_z / (d_fwd_z + 1e-10)
        psi_pos_z = self._van_leer_limiter(r_pos_z)
        psi_neg_z = self._van_leer_limiter(r_neg_z)
        
        phi_L_z = phi_z + 0.5 * psi_pos_z * d_fwd_z
        phi_R_z = phi_zp1 - 0.5 * psi_neg_z * d_fwd_z
        phi_face_z = torch.where(self.w > 0, phi_L_z, phi_R_z)
        
        flux_z = self.w * phi_face_z
        flux_zm = torch.cat([flux_z[:, :, 0:1], flux_z[:, :, :-1]], dim=2)
        flux_z_div = (flux_z - flux_zm) / self.dz
        
        # Update
        phi_new = phi - dt * (flux_x_div + flux_y_div + flux_z_div) * self.fluid_mask
        
        return phi_new
    
    def _advect_step(self, phi: torch.Tensor, dt: float) -> torch.Tensor:
        """Single advection substep with first-order upwind (fallback)."""
        # Pad with boundary values (not circular)
        phi_xm = torch.cat([phi[0:1, :, :], phi[:-1, :, :]], dim=0)
        phi_xp = torch.cat([phi[1:, :, :], phi[-1:, :, :]], dim=0)
        phi_ym = torch.cat([phi[:, 0:1, :], phi[:, :-1, :]], dim=1)
        phi_yp = torch.cat([phi[:, 1:, :], phi[:, -1:, :]], dim=1)
        phi_zm = torch.cat([phi[:, :, 0:1], phi[:, :, :-1]], dim=2)
        phi_zp = torch.cat([phi[:, :, 1:], phi[:, :, -1:]], dim=2)
        
        # Upwind in X
        flux_x = torch.where(
            self.u > 0,
            self.u * (phi - phi_xm) / self.dx,
            self.u * (phi_xp - phi) / self.dx
        )
        
        # Upwind in Y
        flux_y = torch.where(
            self.v > 0,
            self.v * (phi - phi_ym) / self.dy,
            self.v * (phi_yp - phi) / self.dy
        )
        
        # Upwind in Z
        flux_z = torch.where(
            self.w > 0,
            self.w * (phi - phi_zm) / self.dz,
            self.w * (phi_zp - phi) / self.dz
        )
        
        phi_new = phi - dt * (flux_x + flux_y + flux_z)
        
        # Clamp to reasonable values for temperature
        # (prevents numerical blowup)
        return phi_new
    
    def diffuse_field_stable(self, phi: torch.Tensor, diffusivity, dt: float) -> torch.Tensor:
        """
        Explicit diffusion with stability check and proper boundaries.
        
        Supports both scalar diffusivity (laminar) and tensor field (turbulent).
        """
        # Handle tensor diffusivity from turbulence model
        if isinstance(diffusivity, torch.Tensor):
            diff_max = diffusivity.max().item()
        else:
            diff_max = diffusivity
        
        # Stability limit: dt < dx^2 / (6 * diffusivity) for 3D
        dt_max = min(self.dx, self.dy, self.dz)**2 / (6 * diff_max + 1e-10)
        n_substeps = max(1, int(np.ceil(dt / dt_max)))
        dt_sub = dt / n_substeps
        
        result = phi.clone()
        for _ in range(n_substeps):
            # Use boundary padding (replicate) instead of periodic roll
            # X direction
            phi_xm = torch.cat([result[0:1, :, :], result[:-1, :, :]], dim=0)
            phi_xp = torch.cat([result[1:, :, :], result[-1:, :, :]], dim=0)
            # Y direction  
            phi_ym = torch.cat([result[:, 0:1, :], result[:, :-1, :]], dim=1)
            phi_yp = torch.cat([result[:, 1:, :], result[:, -1:, :]], dim=1)
            # Z direction
            phi_zm = torch.cat([result[:, :, 0:1], result[:, :, :-1]], dim=2)
            phi_zp = torch.cat([result[:, :, 1:], result[:, :, -1:]], dim=2)
            
            laplacian = (
                (phi_xm + phi_xp - 2*result) / self.dx**2 +
                (phi_ym + phi_yp - 2*result) / self.dy**2 +
                (phi_zm + phi_zp - 2*result) / self.dz**2
            )
            result = result + dt_sub * diffusivity * laplacian * self.fluid_mask
        
        return result
    
    def step(self, dt: Optional[float] = None):
        """
        Advance physics by one timestep.
        
        Algorithm (Fractional Step / Chorin Projection):
        1. Apply boundary conditions
        2. Advect velocity (upwind/TVD)
        3. Diffuse velocity (with optional turbulence)
        4. Add body forces (buoyancy)
        5. Pressure Poisson solve for incompressibility
        6. Velocity correction from pressure gradient
        7. Advect/diffuse scalars (T, CO2)
        """
        import warnings
        
        if dt is None:
            dt = self.config.dt
        
        # 1. Apply boundary conditions
        self.apply_boundary_conditions()
        
        # 2. Advect velocity (stable upwind)
        u_new = self.advect_field_stable(self.u, dt)
        v_new = self.advect_field_stable(self.v, dt)
        w_new = self.advect_field_stable(self.w, dt)
        
        # 3. Compute effective viscosity (with optional Smagorinsky turbulence model)
        nu_eff = self._compute_effective_viscosity()
        
        # 4. Diffuse velocity with effective viscosity
        self.u = self.diffuse_field_stable(u_new, nu_eff, dt)
        self.v = self.diffuse_field_stable(v_new, nu_eff, dt)
        self.w = self.diffuse_field_stable(w_new, nu_eff, dt)
        
        # 5. BOUSSINESQ BUOYANCY: Add thermal buoyancy force
        # F_buoyancy = -rho * beta * (T - T_ref) * g (in -z direction)
        # For velocity: dw/dt = -beta * g * (T - T_ref)
        # Sign: warm air (T > T_ref) → positive buoyancy → upward (+w)
        if self.enable_buoyancy:
            buoyancy_force = self.beta * self.g * (self.T - self.T_ref)
            self.w += buoyancy_force * dt * self.fluid_mask
        
        # 6. PRESSURE POISSON SOLVE (enforces incompressibility: ∇·u = 0)
        # Compute divergence of intermediate velocity field
        div = self._compute_divergence()
        
        # Solve pressure Poisson equation: ∇²p = ρ/dt * ∇·u*
        # Using iterative Jacobi relaxation (simple but effective)
        self.p = self._solve_pressure_poisson(div, dt, n_iter=20)
        
        # 7. VELOCITY CORRECTION: u = u* - dt/ρ * ∇p
        # This projects velocity onto divergence-free space
        self._apply_pressure_correction(dt)
        
        # 8. Advect and diffuse temperature (with turbulent Prandtl number)
        self.T = self.advect_field_stable(self.T, dt)
        alpha_eff = self._compute_effective_thermal_diffusivity(nu_eff)
        self.T = self.diffuse_field_stable(self.T, alpha_eff, dt)
        
        # 9. Add heat sources
        cell_volume = self.dx * self.dy * self.dz
        dT = self.heat_source * dt / (self.rho * self.cp)
        self.T += dT * self.fluid_mask
        
        # 10. Advect CO2
        self.co2 = self.advect_field_stable(self.co2, dt)
        self.co2 += self.co2_source * dt * self.fluid_mask
        
        # 11. STABILITY: Check for NaN and warn instead of silently masking
        nan_count = 0
        for name, field in [('u', self.u), ('v', self.v), ('w', self.w), ('T', self.T)]:
            if torch.isnan(field).any():
                nan_count += torch.isnan(field).sum().item()
                
        if nan_count > 0:
            warnings.warn(f"NaN detected in {nan_count} cells at step {self.step_count}. "
                         f"Consider reducing dt or checking boundary conditions.")
        
        # Replace NaN to prevent complete blowup
        self.u = torch.nan_to_num(self.u, nan=0.0)
        self.v = torch.nan_to_num(self.v, nan=0.0)
        self.w = torch.nan_to_num(self.w, nan=0.0)
        self.T = torch.nan_to_num(self.T, nan=self.config.initial_temp_c + 273.15)
        self.co2 = torch.nan_to_num(self.co2, nan=self.config.initial_co2_ppm)
        
        # 12. STABILITY: Clamp velocities to configurable limit
        max_vel = self.config.max_velocity
        self.u = self.u.clamp(-max_vel, max_vel)
        self.v = self.v.clamp(-max_vel, max_vel)
        self.w = self.w.clamp(-max_vel, max_vel)
        
        # 13. Clamp temperature and CO2 to physical bounds
        self.T = self.T.clamp(263.15, 323.15)  # -10°C to 50°C
        self.co2 = self.co2.clamp(0, 100000)  # 0 to 10% (deadly is ~4%)
        
        # 14. Reapply BCs and mask
        self.apply_boundary_conditions()
        
        # Update tracking
        self.time += dt
        self.step_count += 1
    
    def _compute_effective_viscosity(self) -> torch.Tensor:
        """
        Compute effective viscosity with Smagorinsky LES turbulence model.
        
        ν_eff = ν + ν_t
        ν_t = (Cs * Δ)² * |S|
        
        where:
        - Cs = Smagorinsky constant (0.1-0.2)
        - Δ = filter width (grid spacing)
        - |S| = magnitude of strain rate tensor
        """
        if not self.config.enable_turbulence:
            return self.nu  # Scalar, laminar only
        
        # Filter width (geometric mean of grid spacings)
        delta = (self.dx * self.dy * self.dz) ** (1/3)
        cs = self.config.cs
        
        # Compute strain rate components using central differences
        # S_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)
        
        # du/dx, du/dy, du/dz
        dudx = (torch.roll(self.u, -1, 0) - torch.roll(self.u, 1, 0)) / (2 * self.dx)
        dudy = (torch.roll(self.u, -1, 1) - torch.roll(self.u, 1, 1)) / (2 * self.dy)
        dudz = (torch.roll(self.u, -1, 2) - torch.roll(self.u, 1, 2)) / (2 * self.dz)
        
        # dv/dx, dv/dy, dv/dz
        dvdx = (torch.roll(self.v, -1, 0) - torch.roll(self.v, 1, 0)) / (2 * self.dx)
        dvdy = (torch.roll(self.v, -1, 1) - torch.roll(self.v, 1, 1)) / (2 * self.dy)
        dvdz = (torch.roll(self.v, -1, 2) - torch.roll(self.v, 1, 2)) / (2 * self.dz)
        
        # dw/dx, dw/dy, dw/dz
        dwdx = (torch.roll(self.w, -1, 0) - torch.roll(self.w, 1, 0)) / (2 * self.dx)
        dwdy = (torch.roll(self.w, -1, 1) - torch.roll(self.w, 1, 1)) / (2 * self.dy)
        dwdz = (torch.roll(self.w, -1, 2) - torch.roll(self.w, 1, 2)) / (2 * self.dz)
        
        # Strain rate magnitude: |S| = sqrt(2 * S_ij * S_ij)
        S_mag = torch.sqrt(
            2 * (dudx**2 + dvdy**2 + dwdz**2) +
            (dudy + dvdx)**2 + (dudz + dwdx)**2 + (dvdz + dwdy)**2 +
            1e-10  # Prevent sqrt(0)
        )
        
        # Turbulent viscosity
        nu_t = (cs * delta)**2 * S_mag
        
        # Effective viscosity field
        nu_eff = self.nu + nu_t
        
        return nu_eff
    
    def _compute_effective_thermal_diffusivity(self, nu_eff) -> torch.Tensor:
        """
        Compute effective thermal diffusivity with turbulent Prandtl number.
        
        α_eff = α + ν_t / Pr_t
        
        where Pr_t ≈ 0.85 for air (turbulent Prandtl number)
        """
        if isinstance(nu_eff, float):
            return self.alpha
        
        Pr_t = 0.85  # Turbulent Prandtl number for air
        nu_t = nu_eff - self.nu  # Extract turbulent component
        alpha_eff = self.alpha + nu_t / Pr_t
        
        return alpha_eff
    
    def _compute_divergence(self) -> torch.Tensor:
        """Compute divergence of velocity field: ∇·u = ∂u/∂x + ∂v/∂y + ∂w/∂z"""
        # Forward difference (consistent with pressure gradient)
        dudx = (torch.roll(self.u, -1, 0) - self.u) / self.dx
        dvdy = (torch.roll(self.v, -1, 1) - self.v) / self.dy
        dwdz = (torch.roll(self.w, -1, 2) - self.w) / self.dz
        
        # Zero out at boundaries to prevent wrap-around artifacts
        dudx[-1, :, :] = 0
        dvdy[:, -1, :] = 0
        dwdz[:, :, -1] = 0
        
        return dudx + dvdy + dwdz
    
    def _solve_pressure_poisson(self, div: torch.Tensor, dt: float, n_iter: int = 20) -> torch.Tensor:
        """
        Solve pressure Poisson equation: ∇²p = ρ/dt * ∇·u*
        
        Uses Jacobi iteration which is simple and GPU-friendly.
        """
        # RHS of Poisson equation
        rhs = self.rho / dt * div * self.fluid_mask
        
        # Jacobi iteration coefficients
        idx2 = 1.0 / (self.dx * self.dx)
        idy2 = 1.0 / (self.dy * self.dy)
        idz2 = 1.0 / (self.dz * self.dz)
        diag = -2.0 * (idx2 + idy2 + idz2)
        
        p = self.p.clone()
        
        for _ in range(n_iter):
            # Laplacian stencil (7-point)
            p_xp = torch.roll(p, -1, 0)
            p_xm = torch.roll(p, 1, 0)
            p_yp = torch.roll(p, -1, 1)
            p_ym = torch.roll(p, 1, 1)
            p_zp = torch.roll(p, -1, 2)
            p_zm = torch.roll(p, 1, 2)
            
            # Jacobi update: p_new = (rhs - off_diag * p_neighbors) / diag
            off_diag = (p_xp + p_xm) * idx2 + (p_yp + p_ym) * idy2 + (p_zp + p_zm) * idz2
            p_new = (rhs - off_diag) / diag
            
            # Apply boundary conditions (zero pressure at outlets, Neumann elsewhere)
            # Zero gradient at walls (copy from interior)
            p_new[0, :, :] = p_new[1, :, :]
            p_new[-1, :, :] = p_new[-2, :, :]
            p_new[:, 0, :] = p_new[:, 1, :]
            p_new[:, -1, :] = p_new[:, -2, :]
            p_new[:, :, 0] = p_new[:, :, 1]
            p_new[:, :, -1] = p_new[:, :, -2]
            
            # Zero pressure at outlets (reference pressure)
            for face in self.outlet_faces.keys():
                if face == Face.WEST:
                    p_new[0, :, :] = 0
                elif face == Face.EAST:
                    p_new[-1, :, :] = 0
                elif face == Face.SOUTH:
                    p_new[:, 0, :] = 0
                elif face == Face.NORTH:
                    p_new[:, -1, :] = 0
                elif face == Face.BOTTOM:
                    p_new[:, :, 0] = 0
                elif face == Face.TOP:
                    p_new[:, :, -1] = 0
            
            # Mask solid cells
            p = p_new * self.fluid_mask
        
        return p
    
    def _apply_pressure_correction(self, dt: float):
        """
        Apply pressure correction to make velocity divergence-free.
        
        u = u* - dt/ρ * ∂p/∂x
        v = v* - dt/ρ * ∂p/∂y
        w = w* - dt/ρ * ∂p/∂z
        """
        # Pressure gradient (backward difference for consistency)
        dpdx = (self.p - torch.roll(self.p, 1, 0)) / self.dx
        dpdy = (self.p - torch.roll(self.p, 1, 1)) / self.dy
        dpdz = (self.p - torch.roll(self.p, 1, 2)) / self.dz
        
        # Correct velocity
        correction_factor = dt / self.rho
        self.u -= correction_factor * dpdx * self.fluid_mask
        self.v -= correction_factor * dpdy * self.fluid_mask
        self.w -= correction_factor * dpdz * self.fluid_mask
    
    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================
    
    def get_metrics(self) -> Dict[str, float]:
        """Get bulk zone metrics."""
        fluid_cells = self.fluid_mask.sum()
        
        return {
            'temperature_c': (self.T * self.fluid_mask).sum().item() / fluid_cells.item() - 273.15,
            'co2_ppm': (self.co2 * self.fluid_mask).sum().item() / fluid_cells.item(),
            'velocity_max': torch.sqrt(self.u**2 + self.v**2 + self.w**2).max().item(),
            'velocity_avg': torch.sqrt(self.u**2 + self.v**2 + self.w**2).mean().item(),
            'mass_total': (self.rho * self.fluid_mask * self.dx * self.dy * self.dz).sum().item()
        }
    
    def compute_mass_balance(self) -> Dict[str, float]:
        """Compute mass flux in/out of all faces."""
        mass_in = 0.0
        mass_out = 0.0
        
        for face in Face:
            slc = self.get_face_slice(face)
            
            # Normal velocity (positive = leaving this zone)
            u_n = self.get_face_normal_velocity(face)
            
            # Face area per cell
            if face in (Face.WEST, Face.EAST):
                dA = self.dy * self.dz
            elif face in (Face.SOUTH, Face.NORTH):
                dA = self.dx * self.dz
            else:
                dA = self.dx * self.dy
            
            # Mass flux: rho * u_n * dA (kg/s per cell)
            mass_flux = self.rho * u_n * dA
            
            # Sum positive (outflow) and negative (inflow) separately
            outflow = mass_flux[mass_flux > 0].sum().item()
            inflow = mass_flux[mass_flux < 0].sum().item()
            
            mass_out += outflow
            mass_in += abs(inflow)
        
        return {
            'mass_in': mass_in,
            'mass_out': mass_out,
            'imbalance': mass_in - mass_out,
            'relative_error': abs(mass_in - mass_out) / max(mass_in, 1e-10)
        }
    
    def __repr__(self):
        return f"Zone('{self.name}', {self.nx}x{self.ny}x{self.nz}, {self.lx}x{self.ly}x{self.lz}m)"
