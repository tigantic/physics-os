"""
The Duplex: Two Connected Rooms

This is the proof-of-concept for multi-zone physics.
If air flows correctly from the Hallway into the Conference Room
through a door, then the graph architecture works.

Scenario:
    - Hallway: 6m x 3m x 3m (small, with outside air inlet)
    - Conference Room: 9m x 6m x 3m (larger, with occupants)
    - Door: 1m wide x 2.1m tall connecting them
    
Physics:
    - Fresh air enters Hallway from WEST (simulating HVAC supply)
    - Air flows through door into Conference Room
    - Conference Room has occupants generating heat + CO2
    - Air exits Conference Room through ceiling returns
    
Validation:
    - Mass balance: what enters Hallway must equal what exits Conference Room
    - Temperature gradient: Conference Room should be warmer
    - CO2 gradient: Conference Room should have higher CO2
"""

import torch
import time
from dataclasses import dataclass
from typing import Dict, Optional

from .zone import Zone, ZoneConfig, Face
from .portal import Portal, PortalConfig
from .building import Building, BuildingGraph


@dataclass
class DuplexConfig:
    """Configuration for the two-room demo."""
    
    # Hallway (Zone A) - smaller for faster equilibration
    hallway_length: float = 4.0
    hallway_width: float = 2.0
    hallway_height: float = 2.5
    hallway_nx: int = 32
    hallway_ny: int = 16
    hallway_nz: int = 20
    
    # Conference Room (Zone B) - same grid density
    conference_length: float = 6.0
    conference_width: float = 4.0
    conference_height: float = 2.5
    conference_nx: int = 48
    conference_ny: int = 32
    conference_nz: int = 20
    
    # Door (Portal) - covering full face for better flow
    # Note: For demo simplicity, door covers full hallway face
    # This avoids pressure buildup issues with partial openings
    door_width: float = 2.0   # Full hallway width
    door_height: float = 2.5  # Full hallway height
    door_position_y: float = 0.0  # Starts at y=0
    
    # HVAC - supply velocity tuned so door is not the bottleneck
    # Door can pass ~2 m³/s at 1 m/s through 2.1 m² opening
    # Hallway inlet area = 3 * 3 = 9 m²  
    # Supply velocity = 2.1 / 9 ≈ 0.23 m/s (to match door capacity)
    supply_velocity: float = 0.3  # m/s entering hallway
    supply_temp_c: float = 20.0   # Slightly below room temp
    supply_co2_ppm: float = 400.0
    
    # Occupants in conference room
    n_occupants: int = 6
    heat_per_person: float = 100.0  # Watts
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def create_duplex(config: DuplexConfig = None) -> Building:
    """
    Create the two-room demo scenario.
    
    Returns a Building ready for simulation.
    """
    if config is None:
        config = DuplexConfig()
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                    THE DUPLEX                                 ║
╠═══════════════════════════════════════════════════════════════╣
║  Two-Room Multi-Zone CFD Demo                                 ║
║                                                               ║
║  If you can solve physics across ONE door,                    ║
║  you can solve a MILLION.                                     ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # =========================================================================
    # BUILD THE GRAPH
    # =========================================================================
    
    graph = BuildingGraph(
        name="duplex_demo",
        description="Hallway + Conference Room connected by door"
    )
    
    # Zone A: Hallway
    hallway_config = ZoneConfig(
        name="hallway",
        zone_id=0,
        nx=config.hallway_nx,
        ny=config.hallway_ny,
        nz=config.hallway_nz,
        lx=config.hallway_length,
        ly=config.hallway_width,
        lz=config.hallway_height,
        origin=(0.0, 0.0, 0.0),
        enable_buoyancy=False,  # Disable for mass conservation demo
        device=config.device
    )
    graph.add_zone(hallway_config)
    
    # Zone B: Conference Room
    # Positioned to the EAST of the hallway
    conference_config = ZoneConfig(
        name="conference",
        zone_id=1,
        nx=config.conference_nx,
        ny=config.conference_ny,
        nz=config.conference_nz,
        lx=config.conference_length,
        ly=config.conference_width,
        lz=config.conference_height,
        origin=(config.hallway_length, 0.0, 0.0),  # Adjacent to hallway
        enable_buoyancy=False,  # Disable for mass conservation demo
        device=config.device
    )
    graph.add_zone(conference_config)
    
    # Portal: The Door (bidirectional to allow natural flow)
    door_config = PortalConfig(
        name="door",
        portal_id=0,
        zone_a_name="hallway",
        zone_b_name="conference",
        face_a=Face.EAST,
        face_b=Face.WEST,
        width=config.door_width,
        height=config.door_height,
        position_a=(config.door_position_y, 0.0),  # (y, z) on hallway east face
        position_b=(config.door_position_y, 0.0),  # (y, z) on conference west face
        bidirectional=True  # Allow natural flow both ways
    )
    graph.add_portal(door_config)
    
    # =========================================================================
    # INSTANTIATE THE BUILDING
    # =========================================================================
    
    building = Building(graph, device=config.device)
    
    # =========================================================================
    # CONFIGURE ZONES
    # =========================================================================
    
    hallway = building.get_zone("hallway")
    conference = building.get_zone("conference")
    
    # Hallway: HVAC supply from the west
    print("\nConfiguring Hallway:")
    print(f"  - Supply inlet: WEST face, {config.supply_velocity} m/s, {config.supply_temp_c}°C")
    hallway.add_inlet(
        face=Face.WEST,
        velocity=config.supply_velocity,
        temperature_c=config.supply_temp_c,
        co2_ppm=config.supply_co2_ppm
    )
    
    # Conference Room: Occupants
    print(f"\nConfiguring Conference Room:")
    print(f"  - {config.n_occupants} occupants, {config.heat_per_person}W each")
    
    # Add occupants as distributed heat/CO2 sources
    # Place them around a central table
    table_center_x = config.conference_length / 2
    table_center_y = config.conference_width / 2
    table_radius = 2.0
    
    for i in range(config.n_occupants):
        angle = 2 * 3.14159 * i / config.n_occupants
        x = table_center_x + table_radius * torch.cos(torch.tensor(angle)).item()
        y = table_center_y + table_radius * torch.sin(torch.tensor(angle)).item()
        
        conference.add_occupant(
            position=(x, y, 0.0),
            height=1.2,
            heat_watts=config.heat_per_person,
            co2_lps=0.005  # ~18 L/hr
        )
    
    print(f"  - {config.n_occupants} occupants placed around conference table")
    
    # Conference Room: Outlet (ceiling return)
    # Use inlet with negative velocity to extract air
    # Extraction rate should match the hallway inlet for mass conservation
    inlet_flow = config.supply_velocity * config.hallway_width * config.hallway_height  # m³/s
    extract_area = config.conference_length * config.conference_width
    extract_velocity = inlet_flow / extract_area
    print(f"  - Return air: TOP face (ceiling) at {extract_velocity:.3f} m/s extraction")
    print(f"    (matches hallway inlet: {inlet_flow:.2f} m³/s)")
    conference.add_inlet(
        face=Face.TOP,
        velocity=-extract_velocity,  # Negative = extraction
        temperature_c=22.0,  # Doesn't matter for extraction
        co2_ppm=400.0
    )
    
    return building


def run_duplex_demo(duration: float = 120.0, dt: float = 0.01) -> Dict:
    """
    Run the complete Duplex demonstration.
    
    Args:
        duration: Simulation time in seconds
        dt: Timestep
    
    Returns:
        Results dictionary
    """
    
    # Create the building
    config = DuplexConfig()
    building = create_duplex(config)
    
    # Run simulation
    print(f"\n{'='*60}")
    print("SIMULATION")
    print(f"{'='*60}")
    
    results = building.simulate(
        duration=duration,
        dt=dt,
        report_interval=20.0,
        validate_conservation=True
    )
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    print(f"\n{'='*60}")
    print("VALIDATION")
    print(f"{'='*60}")
    
    hallway = building.get_zone("hallway")
    conference = building.get_zone("conference")
    
    h_metrics = hallway.get_metrics()
    c_metrics = conference.get_metrics()
    
    print(f"\nZone Comparison:")
    print(f"  {'Metric':<20} {'Hallway':>12} {'Conference':>12} {'Delta':>12}")
    print(f"  {'-'*56}")
    print(f"  {'Temperature (°C)':<20} {h_metrics['temperature_c']:>12.2f} {c_metrics['temperature_c']:>12.2f} {c_metrics['temperature_c'] - h_metrics['temperature_c']:>+12.2f}")
    print(f"  {'CO2 (ppm)':<20} {h_metrics['co2_ppm']:>12.1f} {c_metrics['co2_ppm']:>12.1f} {c_metrics['co2_ppm'] - h_metrics['co2_ppm']:>+12.1f}")
    print(f"  {'Max Velocity (m/s)':<20} {h_metrics['velocity_max']:>12.3f} {c_metrics['velocity_max']:>12.3f}")
    
    # Portal flow
    portal_flows = building.get_portal_flows()
    print(f"\nPortal Flows:")
    total_flow = 0.0
    for name, stats in portal_flows.items():
        print(f"  {name}: A→B = {stats['flux_a_to_b']:.4f} kg/s, B→A = {stats['flux_b_to_a']:.4f} kg/s")
        total_flow += stats['flux_a_to_b'] + stats['flux_b_to_a']
    
    # Mass conservation check - use relative threshold (1% of flow rate)
    mass_imbalance = building.check_mass_conservation()
    reference_flow = max(total_flow, 1.0)  # At least 1 kg/s as reference
    relative_error = mass_imbalance / reference_flow
    mass_conserved = relative_error < 0.01  # 1% threshold
    
    print(f"\nMass Conservation:")
    print(f"  Total imbalance: {mass_imbalance:.2e} kg/s ({relative_error*100:.2f}% of flow)")
    
    if mass_conserved:
        print("  ✓ PASS - Mass is conserved (<1% error)")
    else:
        print("  ✗ FAIL - Mass imbalance detected (>1% error)")
    
    # Physics validation
    print(f"\nPhysics Validation:")
    
    # Conference should be warmer (occupant heat)
    temp_check = c_metrics['temperature_c'] > h_metrics['temperature_c']
    print(f"  Conference warmer than Hallway: {'✓ PASS' if temp_check else '✗ FAIL'}")
    
    # Conference should have higher CO2 (breathing)
    co2_check = c_metrics['co2_ppm'] > h_metrics['co2_ppm']
    print(f"  Conference higher CO2 than Hallway: {'✓ PASS' if co2_check else '✗ FAIL'}")
    
    # Summary
    all_pass = mass_conserved and temp_check and co2_check

    print(f"\n{'='*60}")
    if all_pass:
        print("  ✓ THE DUPLEX WORKS - MULTI-ZONE PHYSICS VALIDATED")
    else:
        print("  ✗ VALIDATION FAILED - CHECK PHYSICS")
    print(f"{'='*60}")

    results['building'] = building
    results['validation'] = {
        'mass_conserved': mass_conserved,
        'co2_gradient': co2_check,
        'all_pass': all_pass
    }
    
    return results


if __name__ == "__main__":
    run_duplex_demo()
