"""
HyperFOAM Multi-Zone Architecture

The foundation for scaling from 1 room to 1 million rooms.

Core Concepts:
    Zone:     A single computational domain with its own grid + physics
    Portal:   A boundary condition coupling two zones (doors, openings)
    Building: A graph of Zones connected by Portals

T3 Capabilities:
    T3.01-04: Multi-zone coupling with mass/temperature/CO2 transport
    T3.05:    Building-level comfort metrics (PMV/PPD/ADPI aggregation)
    T3.06:    VAV Terminal Box models
    T3.07:    Fan-Powered Box models (Series/Parallel)
    T3.08:    Ductwork pressure drop calculation
    T3.09:    AHU (Air Handling Unit) simplified model

T4 Capabilities (Data Center / Transient):
    T4.01:    CRAC Unit model (Computer Room Air Conditioner)
    T4.02:    Server Rack model (variable heat load)
    T4.03:    Raised Floor Plenum (underfloor air distribution)
    T4.04:    Hot/Cold Aisle Containment
    T4.05:    Transient Scenario Runner (CRAC failure events)
    T4.06:    Thermal Metrics (RCI, SHI, RTI per ASHRAE)
    T4.07:    Critical Rack Detection (>35°C inlet)
    T4.08:    Time-to-Critical Prediction
    T4.09:    Energy Balance Verification

T5 Capabilities (Fire/Smoke/Atrium):
    T5.01:    Fire Source Model (t² HRR curves)
    T5.02:    Smoke Species Transport (soot/CO/CO2)
    T5.03:    Visibility Calculation (S = K/OD)
    T5.04:    Tenability Metrics (NFPA 502/BS 7974)
    T5.05:    Jet Fan Model (thrust-based momentum)
    T5.06:    Smoke Extraction System
    T5.07:    Buoyant Plume Physics (Heskestad)
    T5.08:    ASET/RSET Analysis
    T5.09:    Egress Zone Monitoring
"""

from .zone import Zone, ZoneConfig, Face
from .portal import Portal, PortalConfig
from .building import Building, BuildingGraph
from .duplex import create_duplex, DuplexConfig, run_duplex_demo
from .equipment import (
    VAVConfig, VAVTerminal,
    FanPoweredBoxConfig, FanPoweredBox,
    DuctConfig, Duct,
    AHUConfig, AHU,
    compute_building_comfort_metrics
)
from .datacenter import (
    # T4.01: CRAC Unit
    CRACConfig, CRACUnit, CRACState,
    # T4.02: Server Rack
    ServerRackConfig, ServerRack, RackLoadProfile,
    # T4.03: Raised Floor Plenum
    PlenumConfig, RaisedFloorPlenum,
    # T4.04: Containment
    ContainmentConfig, AisleContainment, ContainmentType,
    # T4.05: Transient Scenario
    TransientConfig, TransientEvent, DataCenterSimulator,
    # T4.06: Thermal Metrics
    compute_rci, compute_shi, compute_rti, ASHRAEClass,
    # Factory functions
    create_data_center, run_crac_failure_scenario
)

from .fire_smoke import (
    # T5.01: Fire Source
    FireGrowthType, FireConfig, FireSource,
    # T5.03: Visibility
    compute_visibility, smoke_to_optical_density,
    # T5.04: Tenability
    TenabilityThresholds, TenabilityStatus, assess_tenability,
    # T5.05: Jet Fan
    JetFanConfig, JetFan,
    # T5.06: Smoke Extraction
    ExtractionPointConfig, SmokeExtraction,
    # T5.07: Plume Physics
    heskestad_plume_radius, heskestad_centerline_velocity,
    heskestad_centerline_temp_rise, plume_mass_flow_rate,
    # T5.08: ASET/RSET
    EgressConfig, compute_rset, ASETTracker,
    # T5.09: Egress Monitoring
    EgressZone, EgressMonitor
)

__all__ = [
    # Core
    'Zone', 'ZoneConfig', 'Face',
    'Portal', 'PortalConfig', 
    'Building', 'BuildingGraph',
    'create_duplex', 'DuplexConfig', 'run_duplex_demo',
    # Equipment (T3.06-09)
    'VAVConfig', 'VAVTerminal',
    'FanPoweredBoxConfig', 'FanPoweredBox',
    'DuctConfig', 'Duct',
    'AHUConfig', 'AHU',
    'compute_building_comfort_metrics',
    # Data Center (T4.01-09)
    'CRACConfig', 'CRACUnit', 'CRACState',
    'ServerRackConfig', 'ServerRack', 'RackLoadProfile',
    'PlenumConfig', 'RaisedFloorPlenum',
    'ContainmentConfig', 'AisleContainment', 'ContainmentType',
    'TransientConfig', 'TransientEvent', 'DataCenterSimulator',
    'compute_rci', 'compute_shi', 'compute_rti', 'ASHRAEClass',
    'create_data_center', 'run_crac_failure_scenario',
    # Fire/Smoke (T5.01-09)
    'FireGrowthType', 'FireConfig', 'FireSource',
    'compute_visibility', 'smoke_to_optical_density',
    'TenabilityThresholds', 'TenabilityStatus', 'assess_tenability',
    'JetFanConfig', 'JetFan',
    'ExtractionPointConfig', 'SmokeExtraction',
    'heskestad_plume_radius', 'heskestad_centerline_velocity',
    'heskestad_centerline_temp_rise', 'plume_mass_flow_rate',
    'EgressConfig', 'compute_rset', 'ASETTracker',
    'EgressZone', 'EgressMonitor'
]
