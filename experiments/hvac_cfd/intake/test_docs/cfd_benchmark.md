# HyperFOAM v2 Residential CFD Benchmark Case

## Project Info
Project Name: Residential Summer Cooling Test
Client: Internal Benchmark
Date: January 2025

## Domain Configuration

| Parameter | Value |
|-----------|-------|
| Geometry | Rectangular room: $3.66\text{m} \times 2.74\text{m} \times 4.57\text{m}$ (12ft × 9ft × 15ft) |
| Grid Resolution | 128 × 96 × 64 cells (786,432 cells) |
| Inlet | Ceiling supply diffuser at $x=1.83, y=2.74, z=2.28$ |

## Boundary Conditions

| Boundary | Type | Values |
|----------|------|--------|
| Supply inlet | Velocity inlet | $x=0, y=-3.28, z=0 \text{ m/s}$ (645 fpm downward), $T=285.93 \text{ K}$ (55°F) |
| Return | Pressure outlet | $P_{gauge} = 0$ |
| Floor | Wall | Adiabatic |
| Ceiling | Wall | Adiabatic |
| Walls | Mixed | Convective ($h=5 \text{ W/m}^2\text{K}$) |
| Window (south) | Wall | $T=305.37 \text{ K}$ (90°F) - simulates solar gain |

## Internal Heat Sources

| Source | Power | Location |
|--------|-------|----------|
| Occupant | $75.0 \text{ W}$ | Floor center |
| Laptop | $100.0 \text{ W}$ | Desk at south wall |

## Fluid Properties (Air at 293K)

| Property | Value |
|----------|-------|
| Density | $\rho = 1.225 \text{ kg/m}^3$ |
| Dynamic Viscosity | $\mu = 1.81 \times 10^{-5} \text{ Pa·s}$ |
| Thermal Conductivity | $k = 0.026 \text{ W/m·K}$ |
| Specific Heat | $c_p = 1006 \text{ J/kg·K}$ |
| Prandtl Number | $Pr = 0.71$ |

## Solver Settings

- **Turbulence Model**: k-ε realizable
- **Pressure-Velocity Coupling**: SIMPLE
- **Time Stepping**: Steady-state
- **Convergence Criteria**: Residuals < 10⁻⁵
