"""
Portal: The Inter-Zone Coupling Mechanism

A Portal is a door, opening, or HVAC duct that connects two Zones.
It handles:
1. Reading flux from Zone A's boundary
2. Transforming it (coordinate mapping if needed)
3. Injecting it into Zone B's boundary

Mass conservation: What leaves A MUST enter B.
"""

import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from enum import Enum

from .zone import Zone, Face


@dataclass
class PortalConfig:
    """Configuration for an inter-zone portal."""
    
    # Identity
    name: str = "portal_0"
    portal_id: int = 0
    
    # Connection topology
    zone_a_name: str = ""  # Source zone
    zone_b_name: str = ""  # Destination zone
    face_a: Face = Face.EAST   # Which face of zone A
    face_b: Face = Face.WEST   # Which face of zone B
    
    # Physical dimensions of the opening
    width: float = 1.0   # meters (in the face plane)
    height: float = 2.1  # meters (door height)
    
    # Position on face A (local coordinates)
    position_a: Tuple[float, float] = (0.0, 0.0)  # (y, z) offset from corner
    
    # Position on face B (local coordinates)
    position_b: Tuple[float, float] = (0.0, 0.0)  # (y, z) offset from corner
    
    # Flow resistance (for HVAC ducts)
    resistance: float = 0.0  # Pressure drop coefficient
    
    # Is this a door (bidirectional) or a duct (unidirectional)?
    bidirectional: bool = True


class Portal:
    """
    Couples two zones across a shared boundary.
    
    The Portal reads flux leaving zone_a through face_a,
    and injects it into zone_b through face_b.
    
    If bidirectional, it also does the reverse.
    
    Usage:
        portal = Portal(config, zone_a, zone_b)
        portal.exchange()  # Called each timestep
    """
    
    def __init__(self, config: PortalConfig, zone_a: Zone, zone_b: Zone):
        self.config = config
        self.name = config.name
        self.portal_id = config.portal_id
        
        self.zone_a = zone_a
        self.zone_b = zone_b
        self.face_a = config.face_a
        self.face_b = config.face_b
        
        # Compute grid cell regions for the portal
        self.region_a = self._compute_region(zone_a, config.face_a, 
                                              config.position_a, 
                                              config.width, config.height)
        self.region_b = self._compute_region(zone_b, config.face_b,
                                              config.position_b,
                                              config.width, config.height)
        
        # CRITICAL: Register portal regions with zones
        # This prevents apply_boundary_conditions() from zeroing the portal cells
        zone_a.register_portal_region(config.face_a, self.region_a)
        zone_b.register_portal_region(config.face_b, self.region_b)
        
        # Tracking
        self.last_flux_a_to_b = 0.0
        self.last_flux_b_to_a = 0.0
        self.cumulative_mass_transfer = 0.0
        
        self.bidirectional = config.bidirectional
        
        print(f"Portal '{self.name}': {zone_a.name}:{config.face_a.name} <-> {zone_b.name}:{config.face_b.name}")
        print(f"  Region A: cells {self.region_a}")
        print(f"  Region B: cells {self.region_b}")
    
    def _compute_region(
        self,
        zone: Zone,
        face: Face,
        position: Tuple[float, float],
        width: float,
        height: float
    ) -> Tuple[int, int, int, int]:
        """
        Compute which grid cells the portal covers on a face.
        
        Returns (j0, j1, k0, k1) - the cell range in the face plane.
        """
        if face in (Face.WEST, Face.EAST):
            # Face is in the Y-Z plane
            y0, z0 = position
            y1, z1 = y0 + width, z0 + height
            
            j0 = int(y0 / zone.dy)
            j1 = int(y1 / zone.dy)
            k0 = int(z0 / zone.dz)
            k1 = int(z1 / zone.dz)
            
        elif face in (Face.SOUTH, Face.NORTH):
            # Face is in the X-Z plane
            x0, z0 = position
            x1, z1 = x0 + width, z0 + height
            
            j0 = int(x0 / zone.dx)
            j1 = int(x1 / zone.dx)
            k0 = int(z0 / zone.dz)
            k1 = int(z1 / zone.dz)
            
        else:  # TOP, BOTTOM
            # Face is in the X-Y plane
            x0, y0 = position
            x1, y1 = x0 + width, y0 + height
            
            j0 = int(x0 / zone.dx)
            j1 = int(x1 / zone.dx)
            k0 = int(y0 / zone.dy)
            k1 = int(y1 / zone.dy)
        
        # Clamp to valid range
        if face in (Face.WEST, Face.EAST):
            j1 = min(j1, zone.ny)
            k1 = min(k1, zone.nz)
        elif face in (Face.SOUTH, Face.NORTH):
            j1 = min(j1, zone.nx)
            k1 = min(k1, zone.nz)
        else:
            j1 = min(j1, zone.nx)
            k1 = min(k1, zone.ny)
        
        return (max(0, j0), j1, max(0, k0), k1)
    
    def _extract_portal_flux(
        self,
        zone: Zone,
        face: Face,
        region: Tuple[int, int, int, int]
    ) -> Dict[str, torch.Tensor]:
        """Extract flux data from a portal region on a face."""
        
        j0, j1, k0, k1 = region
        
        # Get full face flux
        full_flux = zone.get_face_flux(face)
        
        # Slice to portal region
        portal_flux = {}
        for key, tensor in full_flux.items():
            if isinstance(tensor, torch.Tensor) and tensor.dim() >= 2:
                portal_flux[key] = tensor[j0:j1, k0:k1]
            else:
                portal_flux[key] = tensor
        
        return portal_flux
    
    def _inject_portal_flux(
        self,
        zone: Zone,
        face: Face,
        region: Tuple[int, int, int, int],
        flux_data: Dict[str, torch.Tensor],
        force_inlet: bool = False
    ):
        """
        Inject flux data into a portal region on a face.
        
        Args:
            force_inlet: If True, directly set velocity (no relaxation, no outflow allowed).
                        Use this for unidirectional portal destinations.
        """
        
        j0, j1, k0, k1 = region
        slc = zone.get_face_slice(face)
        
        # The velocity needs to be mapped correctly
        # Outflow from source becomes inflow to destination
        incoming_velocity = flux_data['normal_velocity']
        
        # STABILITY: Clamp incoming velocity to physical limits
        MAX_PORTAL_VELOCITY = 2.0
        incoming_velocity = incoming_velocity.clamp(-MAX_PORTAL_VELOCITY, MAX_PORTAL_VELOCITY)
        
        # For force_inlet mode, only allow positive inflow (no outflow)
        if force_inlet:
            incoming_velocity = incoming_velocity.clamp(min=0)
        
        # Handle size mismatch by interpolating
        target_shape = (j1 - j0, k1 - k0)
        if incoming_velocity.shape != target_shape:
            # Resize via interpolation
            incoming_velocity = torch.nn.functional.interpolate(
                incoming_velocity.unsqueeze(0).unsqueeze(0),
                size=target_shape,
                mode='bilinear',
                align_corners=True
            ).squeeze()
            
            flux_data['temperature'] = torch.nn.functional.interpolate(
                flux_data['temperature'].unsqueeze(0).unsqueeze(0),
                size=target_shape,
                mode='bilinear',
                align_corners=True
            ).squeeze()
            
            flux_data['co2'] = torch.nn.functional.interpolate(
                flux_data['co2'].unsqueeze(0).unsqueeze(0),
                size=target_shape,
                mode='bilinear',
                align_corners=True
            ).squeeze()
        
        # Apply to boundary
        # force_inlet: direct set, no relaxation
        # else: relaxed blend for stability
        INJECT_RELAX = 0.5 if not force_inlet else 1.0
        
        if face == Face.WEST:
            current = zone.u[0, j0:j1, k0:k1]
            zone.u[0, j0:j1, k0:k1] = INJECT_RELAX * incoming_velocity + (1 - INJECT_RELAX) * current
            zone.T[0, j0:j1, k0:k1] = flux_data['temperature']
            zone.co2[0, j0:j1, k0:k1] = flux_data['co2']
            
        elif face == Face.EAST:
            current = zone.u[-1, j0:j1, k0:k1]
            zone.u[-1, j0:j1, k0:k1] = INJECT_RELAX * (-incoming_velocity) + (1 - INJECT_RELAX) * current
            zone.T[-1, j0:j1, k0:k1] = flux_data['temperature']
            zone.co2[-1, j0:j1, k0:k1] = flux_data['co2']
            
        elif face == Face.SOUTH:
            current = zone.v[j0:j1, 0, k0:k1]
            zone.v[j0:j1, 0, k0:k1] = INJECT_RELAX * incoming_velocity + (1 - INJECT_RELAX) * current
            zone.T[j0:j1, 0, k0:k1] = flux_data['temperature']
            zone.co2[j0:j1, 0, k0:k1] = flux_data['co2']
            
        elif face == Face.NORTH:
            current = zone.v[j0:j1, -1, k0:k1]
            zone.v[j0:j1, -1, k0:k1] = INJECT_RELAX * (-incoming_velocity) + (1 - INJECT_RELAX) * current
            zone.T[j0:j1, -1, k0:k1] = flux_data['temperature']
            zone.co2[j0:j1, -1, k0:k1] = flux_data['co2']
            
        elif face == Face.BOTTOM:
            current = zone.w[j0:j1, k0:k1, 0]
            zone.w[j0:j1, k0:k1, 0] = INJECT_RELAX * incoming_velocity + (1 - INJECT_RELAX) * current
            zone.T[j0:j1, k0:k1, 0] = flux_data['temperature']
            zone.co2[j0:j1, k0:k1, 0] = flux_data['co2']
            
        elif face == Face.TOP:
            current = zone.w[j0:j1, k0:k1, -1]
            zone.w[j0:j1, k0:k1, -1] = INJECT_RELAX * (-incoming_velocity) + (1 - INJECT_RELAX) * current
            zone.T[j0:j1, k0:k1, -1] = flux_data['temperature']
            zone.co2[j0:j1, k0:k1, -1] = flux_data['co2']
    
    def exchange(self):
        """
        Perform bidirectional flux exchange between zones.
        
        This is called once per timestep AFTER both zones have stepped.
        It reads the flux from each zone and injects it into the neighbor.
        """
        
        # A -> B transfer
        flux_a = self._extract_portal_flux(self.zone_a, self.face_a, self.region_a)
        mass_flux_a_to_b = flux_a['mass_flux'].sum().item()
        
        # For unidirectional: force_inlet=True ensures only inflow at destination
        force_inlet = not self.bidirectional
        self._inject_portal_flux(self.zone_b, self.face_b, self.region_b, flux_a, 
                                  force_inlet=force_inlet)
        self.last_flux_a_to_b = max(0, mass_flux_a_to_b)
        
        # B -> A transfer (if bidirectional)
        if self.bidirectional:
            flux_b = self._extract_portal_flux(self.zone_b, self.face_b, self.region_b)
            mass_flux_b_to_a = flux_b['mass_flux'].sum().item()
            
            self._inject_portal_flux(self.zone_a, self.face_a, self.region_a, flux_b)
            self.last_flux_b_to_a = max(0, mass_flux_b_to_a)
        
        # Track cumulative transfer
        net_flux = self.last_flux_a_to_b - self.last_flux_b_to_a
        self.cumulative_mass_transfer += net_flux
    
    def apply_open_bc(self):
        """
        Apply open boundary conditions at portal regions.
        
        Call this BEFORE zones step to prevent wall BCs from killing portal flow.
        Uses damped Neumann BC with velocity clamping for stability.
        
        For unidirectional portals:
        - Zone A gets open BC that only allows OUTFLOW (clamp to positive)
        - Zone B gets the injected velocity from exchange(), not open BC
        """
        # Maximum allowed velocity at portals (m/s) - conservative for stability
        MAX_PORTAL_VELOCITY = 2.0
        RELAX = 0.5  # Blend factor between interior and existing BC value
        
        # Zone A portal region - allow outflow only for unidirectional
        outflow_only = not self.bidirectional
        self._apply_open_bc_to_zone(self.zone_a, self.face_a, self.region_a, 
                                     MAX_PORTAL_VELOCITY, RELAX, outflow_only=outflow_only)
        
        # Zone B portal region - only for bidirectional flow
        # For unidirectional, zone B's portal cells are handled by inject_portal_flux
        if self.bidirectional:
            self._apply_open_bc_to_zone(self.zone_b, self.face_b, self.region_b,
                                         MAX_PORTAL_VELOCITY, RELAX, outflow_only=False)
    
    def _apply_open_bc_to_zone(self, zone: Zone, face: Face, region: Tuple[int, int, int, int],
                                max_vel: float = 5.0, relax: float = 0.5, outflow_only: bool = False):
        """
        Apply damped Neumann BC with velocity clamping at portal region.
        
        Args:
            outflow_only: If True, clamp velocity to only allow outflow (positive normal velocity)
        """
        j0, j1, k0, k1 = region
        
        if face == Face.WEST:
            # Get interior velocity and clamp
            interior_u = zone.u[1, j0:j1, k0:k1].clamp(-max_vel, max_vel)
            if outflow_only:
                interior_u = interior_u.clamp(max=0)  # WEST outflow = negative u
            current_u = zone.u[0, j0:j1, k0:k1]
            # Relaxed update
            zone.u[0, j0:j1, k0:k1] = relax * interior_u + (1 - relax) * current_u
            zone.T[0, j0:j1, k0:k1] = zone.T[1, j0:j1, k0:k1]
            zone.co2[0, j0:j1, k0:k1] = zone.co2[1, j0:j1, k0:k1]
            
        elif face == Face.EAST:
            interior_u = zone.u[-2, j0:j1, k0:k1].clamp(-max_vel, max_vel)
            if outflow_only:
                # Force outflow only - no relaxation, directly clamp to positive
                interior_u = interior_u.clamp(min=0)  # EAST outflow = positive u
                zone.u[-1, j0:j1, k0:k1] = interior_u  # Direct set, no blend
            else:
                current_u = zone.u[-1, j0:j1, k0:k1]
                zone.u[-1, j0:j1, k0:k1] = relax * interior_u + (1 - relax) * current_u
            zone.T[-1, j0:j1, k0:k1] = zone.T[-2, j0:j1, k0:k1]
            zone.co2[-1, j0:j1, k0:k1] = zone.co2[-2, j0:j1, k0:k1]
            
        elif face == Face.SOUTH:
            interior_v = zone.v[j0:j1, 1, k0:k1].clamp(-max_vel, max_vel)
            if outflow_only:
                interior_v = interior_v.clamp(max=0)  # SOUTH outflow = negative v
            current_v = zone.v[j0:j1, 0, k0:k1]
            zone.v[j0:j1, 0, k0:k1] = relax * interior_v + (1 - relax) * current_v
            zone.T[j0:j1, 0, k0:k1] = zone.T[j0:j1, 1, k0:k1]
            zone.co2[j0:j1, 0, k0:k1] = zone.co2[j0:j1, 1, k0:k1]
            
        elif face == Face.NORTH:
            interior_v = zone.v[j0:j1, -2, k0:k1].clamp(-max_vel, max_vel)
            if outflow_only:
                interior_v = interior_v.clamp(min=0)  # NORTH outflow = positive v
            current_v = zone.v[j0:j1, -1, k0:k1]
            zone.v[j0:j1, -1, k0:k1] = relax * interior_v + (1 - relax) * current_v
            zone.T[j0:j1, -1, k0:k1] = zone.T[j0:j1, -2, k0:k1]
            zone.co2[j0:j1, -1, k0:k1] = zone.co2[j0:j1, -2, k0:k1]
    
    def get_stats(self) -> Dict[str, float]:
        """Get portal statistics."""
        return {
            'flux_a_to_b': self.last_flux_a_to_b,
            'flux_b_to_a': self.last_flux_b_to_a,
            'net_flux': self.last_flux_a_to_b - self.last_flux_b_to_a,
            'cumulative_transfer': self.cumulative_mass_transfer
        }
    
    def __repr__(self):
        return f"Portal('{self.name}', {self.zone_a.name} <-> {self.zone_b.name})"
