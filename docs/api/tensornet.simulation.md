# Module `tensornet.simulation`

Simulation Module =================

End-to-end simulation capabilities for hypersonic vehicle analysis,
including hardware-in-the-loop testing, flight data integration,
and mission simulation with Monte Carlo analysis.

Architecture:
    ┌──────────────────────────────────────────────────────────────────┐
    │                    Simulation Module                              │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
    │  │     HIL     │  │ Flight Data │  │ Real-Time   │  │ Mission │ │
    │  │  Interface  │  │  Interface  │  │    CFD      │  │   Sim   │ │
    │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘ │
    └──────────────────────────────────────────────────────────────────┘

Submodules:
    hil: Hardware-in-the-loop simulation interface
    flight_data: Telemetry parsing and trajectory reconstruction
    realtime_cfd: CFD-guidance coupling for real-time aero lookup
    mission: End-to-end mission simulation with uncertainty

**Contents:**

- [Submodules](#submodules)

## Submodules

- [`simulation.flight_data`](#simulation-flight_data)
- [`simulation.hil`](#simulation-hil)
- [`simulation.mission`](#simulation-mission)
- [`simulation.realtime_cfd`](#simulation-realtime_cfd)
