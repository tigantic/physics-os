"""
Building: The Graph of Zones and Portals

A Building is a collection of Zones connected by Portals.
It manages:
1. Global time stepping (all zones advance together)
2. Portal exchanges (after each step)
3. Mass/energy conservation validation

This is the foundation for scaling to arbitrary building sizes.
The graph structure allows:
- Parallel zone computation (independent CUDA streams)
- Distributed simulation (zones on different GPUs/nodes)
- Dynamic load balancing
"""

import torch
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .zone import Zone, ZoneConfig, Face
from .portal import Portal, PortalConfig


@dataclass
class BuildingGraph:
    """
    Graph representation of a building.
    
    Nodes: Zones (computational domains)
    Edges: Portals (inter-zone connections)
    
    This structure can be serialized to JSON and reconstructed,
    or generated from IFC/BIM files.
    """
    
    zones: Dict[str, ZoneConfig] = field(default_factory=dict)
    portals: List[PortalConfig] = field(default_factory=list)
    
    # Metadata
    name: str = "building"
    description: str = ""
    
    def add_zone(self, config: ZoneConfig):
        """Add a zone to the graph."""
        self.zones[config.name] = config
    
    def add_portal(self, config: PortalConfig):
        """Add a portal (connection) to the graph."""
        # Validate zones exist
        if config.zone_a_name not in self.zones:
            raise ValueError(f"Zone '{config.zone_a_name}' not found in graph")
        if config.zone_b_name not in self.zones:
            raise ValueError(f"Zone '{config.zone_b_name}' not found in graph")
        
        self.portals.append(config)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (for JSON export)."""
        return {
            'name': self.name,
            'description': self.description,
            'zones': {name: vars(cfg) for name, cfg in self.zones.items()},
            'portals': [vars(p) for p in self.portals]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BuildingGraph':
        """Deserialize from dictionary."""
        graph = cls(
            name=data.get('name', 'building'),
            description=data.get('description', '')
        )
        
        for name, zone_data in data.get('zones', {}).items():
            config = ZoneConfig(**zone_data)
            graph.add_zone(config)
        
        for portal_data in data.get('portals', []):
            # Convert face strings back to enum
            portal_data['face_a'] = Face[portal_data['face_a']]
            portal_data['face_b'] = Face[portal_data['face_b']]
            config = PortalConfig(**portal_data)
            graph.add_portal(config)
        
        return graph


class Building:
    """
    Runtime simulation of a multi-zone building.
    
    Usage:
        # From config
        graph = BuildingGraph()
        graph.add_zone(ZoneConfig(name="hallway", nx=32, lx=6.0))
        graph.add_zone(ZoneConfig(name="office", nx=64, lx=9.0))
        graph.add_portal(PortalConfig(
            zone_a_name="hallway",
            zone_b_name="office",
            face_a=Face.EAST,
            face_b=Face.WEST,
            width=1.0,
            height=2.1
        ))
        
        building = Building(graph)
        building.step(dt=0.01)
    """
    
    def __init__(self, graph: BuildingGraph, device: str = None):
        self.graph = graph
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"\n{'='*60}")
        print(f"BUILDING: {graph.name}")
        print(f"{'='*60}")
        
        # Instantiate zones
        self.zones: Dict[str, Zone] = {}
        for name, config in graph.zones.items():
            config.device = self.device
            zone = Zone(config)
            self.zones[name] = zone
            print(f"  Zone '{name}': {zone}")
        
        # Instantiate portals
        self.portals: List[Portal] = []
        for config in graph.portals:
            zone_a = self.zones[config.zone_a_name]
            zone_b = self.zones[config.zone_b_name]
            portal = Portal(config, zone_a, zone_b)
            self.portals.append(portal)
        
        print(f"\n  Total: {len(self.zones)} zones, {len(self.portals)} portals")
        print(f"{'='*60}\n")
        
        # Simulation state
        self.time = 0.0
        self.step_count = 0
        
        # Conservation tracking
        self.mass_history = []
        self.energy_history = []
    
    def get_zone(self, name: str) -> Zone:
        """Get zone by name."""
        return self.zones[name]
    
    def step(self, dt: float = 0.01):
        """
        Advance all zones by one timestep.
        
        Order of operations:
        1. Apply open BCs at portal SOURCE faces (for outflow preparation)
        2. Exchange fluxes through portals (extract from source, inject to destination)
        3. Step all zones (in parallel if on GPU)
        4. Apply open BCs again (to prepare source face for next step)
        
        CRITICAL: exchange() is called ONCE per step, not twice!
        Calling it twice would overwrite the injected destination velocity
        with a lower value from the source (which decays during zone.step).
        """
        
        # 1. Apply open boundary conditions at portal SOURCE faces
        # This prepares outflow velocity from interior for extraction
        for portal in self.portals:
            portal.apply_open_bc()
        
        # 2. Portal exchange - extract from source, inject to destination
        # This is the key coupling step - only done ONCE per timestep
        for portal in self.portals:
            portal.exchange()
        
        # 3. Step all zones
        for zone in self.zones.values():
            zone.step(dt)
        
        # 4. Apply open BCs again to prepare source face for next exchange
        for portal in self.portals:
            portal.apply_open_bc()
        
        # 5. Update time
        self.time += dt
        self.step_count += 1
    
    def simulate(
        self,
        duration: float,
        dt: float = 0.01,
        report_interval: float = 10.0,
        validate_conservation: bool = True
    ) -> Dict[str, Any]:
        """
        Run simulation for specified duration.
        
        Args:
            duration: Total simulation time (seconds)
            dt: Timestep
            report_interval: How often to print status
            validate_conservation: Check mass balance each report
        
        Returns:
            Dict with history and final metrics
        """
        n_steps = int(duration / dt)
        report_steps = int(report_interval / dt)
        
        print(f"Simulating {duration:.0f}s ({n_steps} steps)...")
        
        history = {
            'time': [],
            'zones': {name: {'T': [], 'CO2': [], 'V': []} for name in self.zones}
        }
        
        start_time = time.time()
        
        for i in range(n_steps):
            self.step(dt)
            
            if (i + 1) % report_steps == 0:
                elapsed = time.time() - start_time
                sim_time = self.time
                
                history['time'].append(sim_time)
                
                print(f"  t={sim_time:6.1f}s |", end="")
                
                for name, zone in self.zones.items():
                    metrics = zone.get_metrics()
                    history['zones'][name]['T'].append(metrics['temperature_c'])
                    history['zones'][name]['CO2'].append(metrics['co2_ppm'])
                    history['zones'][name]['V'].append(metrics['velocity_avg'])
                    
                    print(f" {name}: {metrics['temperature_c']:.1f}°C |", end="")
                
                if validate_conservation:
                    imbalance = self.check_mass_conservation()
                    print(f" Δm={imbalance:.2e}", end="")
                
                steps_per_sec = (i + 1) / elapsed
                print(f" [{steps_per_sec:.0f} steps/s]")
        
        total_time = time.time() - start_time
        print(f"\nCompleted in {total_time:.1f}s ({n_steps/total_time:.0f} steps/s)")
        
        return {
            'history': history,
            'elapsed': total_time,
            'steps_per_sec': n_steps / total_time
        }
    
    def check_mass_conservation(self) -> float:
        """
        Check global mass conservation.
        
        Returns total mass imbalance across all zones.
        For a closed system or a system with balanced external flows,
        portal flows between zones should cancel out.
        """
        total_imbalance = 0.0
        
        for zone in self.zones.values():
            balance = zone.compute_mass_balance()
            # Use signed imbalance - portal flows should cancel between zones
            total_imbalance += balance['imbalance']
        
        # Return absolute value of net imbalance
        return abs(total_imbalance)
    
    def get_global_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics across all zones."""
        total_volume = 0.0
        weighted_temp = 0.0
        weighted_co2 = 0.0
        max_velocity = 0.0
        
        for zone in self.zones.values():
            volume = zone.lx * zone.ly * zone.lz
            metrics = zone.get_metrics()
            
            total_volume += volume
            weighted_temp += metrics['temperature_c'] * volume
            weighted_co2 += metrics['co2_ppm'] * volume
            max_velocity = max(max_velocity, metrics['velocity_max'])
        
        return {
            'total_volume_m3': total_volume,
            'avg_temperature_c': weighted_temp / total_volume,
            'avg_co2_ppm': weighted_co2 / total_volume,
            'max_velocity_ms': max_velocity,
            'n_zones': len(self.zones),
            'n_portals': len(self.portals)
        }
    
    def get_portal_flows(self) -> Dict[str, Dict[str, float]]:
        """Get flow rates through all portals."""
        flows = {}
        for portal in self.portals:
            flows[portal.name] = portal.get_stats()
        return flows
    
    def __repr__(self):
        return f"Building('{self.graph.name}', {len(self.zones)} zones, {len(self.portals)} portals)"
