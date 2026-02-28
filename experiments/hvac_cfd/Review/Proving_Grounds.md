# HVAC CFD — Proving Grounds

```
████████╗██╗███████╗██████╗     ███████╗ ██████╗███████╗███╗   ██╗ █████╗ ██████╗ ██╗ ██████╗ ███████╗
╚══██╔══╝██║██╔════╝██╔══██╗    ██╔════╝██╔════╝██╔════╝████╗  ██║██╔══██╗██╔══██╗██║██╔═══██╗██╔════╝
   ██║   ██║█████╗  ██████╔╝    ███████╗██║     █████╗  ██╔██╗ ██║███████║██████╔╝██║██║   ██║███████╗
   ██║   ██║██╔══╝  ██╔══██╗    ╚════██║██║     ██╔══╝  ██║╚██╗██║██╔══██║██╔══██╗██║██║   ██║╚════██║
   ██║   ██║███████╗██║  ██║    ███████║╚██████╗███████╗██║ ╚████║██║  ██║██║  ██║██║╚██████╔╝███████║
   ╚═╝   ╚═╝╚══════╝╚═╝  ╚═╝    ╚══════╝ ╚═════╝╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝ ╚═════╝ ╚══════╝
                    E S C A L A T E D   V A L I D A T I O N   C H A L L E N G E S
```

**Created**: January 6, 2026  
**Version**: 0.1.0  
**Purpose**: Graduated complexity challenges for HVAC CFD capability validation  
**Philosophy**: Each tier builds upon the previous — no tier is skipped.

---

## Overview

| Tier | Scenario | Budget | Timeline | Complexity |
|:----:|----------|--------|----------|:----------:|
| **1** | Simple Room Study | $8K–15K | 4–6 weeks | ⬛⬜⬜⬜⬜⬜ |
| **2** | Occupied Space | $20K–40K | 6–10 weeks | ⬛⬛⬜⬜⬜⬜ |
| **3** | Complex Geometry | $40K–80K | 2–4 months | ⬛⬛⬛⬜⬜⬜ |
| **4** | Multi-Zone / Transient | $80K–150K | 4–6 months | ⬛⬛⬛⬛⬜⬜ |
| **5** | Smoke/Fire/Atrium | $100K–250K+ | 6–12 months | ⬛⬛⬛⬛⬛⬜ |
| **6** | Full Building + BIM | $200K–500K+ | 6–18 months | ⬛⬛⬛⬛⬛⬛ |

---

## TIER 1: Simple Room Study (Isothermal)

**Traditional Budget**: $10,000 – $15,000  
**Traditional Timeline**: 4–6 weeks  
**Traditional Deliverable**: PDF report with contours

**Physics OS Budget**: $2,000 – $4,000  
**Physics OS Timeline**: 48 hours  
**Physics OS Deliverable**: Same report + interactive visualization

---

### Client Scenario — James Chen

| Attribute | Details |
|-----------|---------|
| **Role** | Facilities Engineering Manager |
| **Problem** | Conference room "stuffy" at back wall despite HVAC running |
| **Room** | 9m × 3m × 3m |
| **Supply** | Ceiling diffuser at x = 0 |
| **Return** | Floor grille at far wall |

### Geometry

```
┌─────────────────────────────────────────────────────────────────┐
│  SUPPLY DIFFUSER (168mm slot, 0.455 m/s)                        │
│  ════════                                                       │
│  ↓↓↓↓↓↓↓↓                                                       │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                                                           │  │
│  │                      9.0 m                                │  │
│  │  ←──────────────────────────────────────────────────────→ │  │
│  │                                                           │  │ 3.0 m
│  │    x=0        x=3m        x=6m        x=9m                │  │
│  │     ↓          ↓           ↓           ↓                  │  │
│  │  [INLET]   [PROFILE]   [PROFILE]   [STUFFY ZONE]          │  │
│  │                                                           │  │
│  │                                                     ║     │  │
│  │                                                     ║     │  │
│  └─────────────────────────────────────────────────────╨─────┘  │
│                                                     RETURN      │
│                                                   (480mm, p=0)  │
└─────────────────────────────────────────────────────────────────┘
         Height: 3.0 m
         Volume: 81 m³
         L/H Ratio: 3.0
```

### Nielsen Benchmark (Validation Target)

**Reference**: Aalborg University CFD Benchmark  
**Data Source**: https://www.en.build.aau.dk/web/cfd-benchmarks/two-dimensional-benchmark-test

| Parameter | Value |
|-----------|-------|
| **Geometry** | 9m × 3m room (L/H = 3.0) |
| **Inlet** | 168mm slot, 0.455 m/s uniform |
| **Outlet** | 480mm height, pressure outlet |
| **Reynolds Number** | Re = 5000 (turbulent low-speed) |
| **Validation Criterion** | < 10% normalized RMS error vs experimental |

### Solver Status

| Metric | Current Value | Target |
|--------|---------------|--------|
| **Grid** | 128×128 | 256×128+ |
| **Inlet Recovery** | 94.4% ✓ | > 90% |
| **Convergence** | 50 iterations | — |
| **Wall Time** | 358s (pre-optimization) | **< 60s** |

### Physics Notes

**Formulation**: Vorticity-Streamfunction (2D)

$$\omega = \text{vorticity (scalar in 2D)}$$
$$\psi = \text{streamfunction}$$
$$u = \frac{\partial \psi}{\partial y}, \quad v = -\frac{\partial \psi}{\partial x}$$

**Boundary Conditions**:
- Inlet BC: Set ψ directly (not ω)
- Outlet BC: Pressure = 0 (reference)
- Walls: No-slip (ψ = const, ω from wall gradient)

**Steady-State**: Pseudo-time iteration until convergence

**Key Insight**: Jet throw distance and recirculation pattern determine whether back wall gets fresh air. If jet attaches to ceiling (Coanda effect), it overshoots occupants.

### Deliverables for James

1. **Velocity Contours** — Full room u, v magnitude
2. **Streamlines** — Flow pattern visualization  
3. **Velocity Profiles** at x = 3m, 6m from inlet (x/H = 1.0, 2.0)
4. **Breathing Zone Analysis** — 1.0m to 1.8m height band
5. **ASHRAE 55 Compliance Check** — Minimum 0.10 m/s in occupied zone
6. **Recommendations**:
   - Option A: Add mid-room diffuser
   - Option B: Increase inlet velocity
   - Option C: Switch to high-induction diffuser

### Execution Checklist

```
[ ] Run Nielsen benchmark at 256×128 or higher
[ ] Extract profiles at x/H = 1.0, 2.0 (x = 3m, 6m)
[ ] Compare to Aalborg experimental data
[ ] Verify <10% RMS error
[ ] Generate client-ready visualizations
[ ] Write 2-page summary with recommendations
[ ] Invoice: $2,500 (first client discount)
```

### Success Criteria

| Metric | Target |
|--------|--------|
| Mass conservation | < 0.1% imbalance |
| Inlet recovery | > 90% |
| RMS error vs Aalborg | < 10% |
| Simulation runtime | **< 60 seconds** |
| ASHRAE 55 compliance | Documented |

### Status

```
🔴 BLOCKED — Awaiting deferred truncation fix for <60s target
```

---

## TIER 2: Occupied Space

**Budget**: $20,000 – $40,000  
**Timeline**: 6–10 weeks

### Geometry

```
┌──────────────────────────────────────────────────────┐
│                      30.0 m                          │
│  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐ │
│  │ WS │ WS │ WS │ WS │ WS │ WS │ WS │ WS │ WS │ WS │ │
│  ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤ │
│  │ WS │ WS │ WS │ WS │ WS │ WS │ WS │ WS │ WS │ WS │ │ 20.0 m
│  ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤ │
│  │ WS │ WS │ WS │ WS │ WS │ WS │ WS │ WS │ WS │ WS │ │
│  ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤ │
│  │ WS │ WS │ WS │ WS │ WS │ WS │ WS │ WS │ WS │ WS │ │
│  ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤ │
│  │ WS │ WS │ WS │ WS │ WS │ WS │ WS │ WS │ WS │ WS │ │
│  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘ │
│                                                      │
│  ☀️ WEST-FACING GLAZING (Solar Gain)                  │
└──────────────────────────────────────────────────────┘
         Height: 3.0 m
         Volume: 1,800 m³
```

### Specifications

| Parameter | Value |
|-----------|-------|
| **Dimensions** | 30m × 20m × 3m |
| **Volume** | 1,800 m³ |
| **Supply** | 4–6 ceiling diffusers |
| **Return** | 2–3 grilles |
| **Workstations** | 50 (simplified blocks) |
| **Occupants** | 50 @ 75W sensible each = 3,750W |
| **Computers** | 50 @ 150W each = 7,500W |
| **Total Cooling Load** | 15 kW |
| **Solar** | West-facing glazing (time-dependent) |

### Physics Requirements

- Incompressible Navier-Stokes with energy equation
- Steady-state (peak load conditions)
- Buoyancy-driven flow (Boussinesq)
- Surface heat flux boundary conditions
- Solar radiation model (simplified or ray-tracing)

### Deliverables

- [ ] Temperature stratification analysis (floor-to-ceiling gradient)
- [ ] PMV/PPD comfort maps (ISO 7730 compliance)
- [ ] Dead zone identification (velocity < 0.1 m/s)
- [ ] Vertical temperature gradient assessment
- [ ] Air age distribution (optional)

### Success Criteria

| Metric | Target |
|--------|--------|
| Mass conservation | < 0.1% imbalance |
| Energy balance | < 1% imbalance |
| PMV range | -0.5 to +0.5 (Class A) |
| Dead zones | < 5% of occupied volume |

---

## TIER 3: Complex Geometry

**Budget**: $40,000 – $80,000  
**Timeline**: 2–4 months

### Geometry

```
┌────────────────────────────────────┐
│            8.0 m                   │
│  ┌──────────────────────────────┐  │
│  │  ╔══════════════════════╗    │  │
│  │  ║ LAMINAR FLOW CEILING ║    │  │
│  │  ║   (48 HEPA Diffusers)║    │  │
│  │  ╚══════════════════════╝    │  │
│  │                              │  │
│  │    💡        💡              │  │ 6.0 m
│  │  (500W)    (500W)            │  │
│  │                              │  │
│  │  ┌─────┐  ┌─────────────┐    │  │
│  │  │ANES.│  │SURGICAL TABLE│   │  │
│  │  │CART │  └─────────────┘    │  │
│  │  └─────┘      👤👤👤         │  │
│  │           (6 personnel)      │  │
│  │                              │  │
│  └──────────────────────────────┘  │
│                                    │
│  [+] POSITIVE PRESSURE vs CORRIDOR │
└────────────────────────────────────┘
         Height: 3.0 m
         Volume: 144 m³
```

### Specifications

| Parameter | Value |
|-----------|-------|
| **Dimensions** | 8m × 6m × 3m |
| **Volume** | 144 m³ |
| **Supply** | 48 HEPA diffusers (laminar flow array) |
| **Surgical Lights** | 2 × 500W = 1,000W |
| **Equipment** | Anesthesia cart, monitors, surgical table |
| **Personnel** | 6 (sterile gowns, ~100W each) |
| **Pressure** | +15 Pa vs corridor |
| **Classification** | ISO 14644-1 Class 5 (at rest) |

### Physics Requirements

- Incompressible Navier-Stokes with energy
- Lagrangian particle tracking (0.5–5 µm)
- Turbulence model suitable for low-velocity laminar flow
- Conjugate heat transfer (surgical lights)
- Pressure boundary at door gaps

### Deliverables

- [ ] ISO 14644-1 compliance verification
- [ ] Particle concentration maps (CFU/m³)
- [ ] Airflow visualization for sterile field protection
- [ ] Contamination risk assessment
- [ ] Recovery time analysis (door opening event)

### Success Criteria

| Metric | Target |
|--------|--------|
| Particle count (0.5 µm) | < 3,520/m³ (ISO Class 5) |
| Unidirectional flow | > 90% of ceiling area |
| Sterile field velocity | 0.25–0.35 m/s |
| Pressure differential | +15 ± 3 Pa |

---

## TIER 4: Multi-Zone / Transient

**Budget**: $80,000 – $150,000  
**Timeline**: 4–6 months

### Geometry

```
┌─────────────────────────────────────────────────────────────────┐
│                         ~25m × 20m = 500 m²                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ ███ ███ ███ ███ ███ ███ ███ ███ ███ ███ ███ ███ ███ ███ ███ ││
│  │ ███ ███ ███ ███ ███ ███ ███ ███ ███ ███ ███ ███ ███ ███ ███ ││
│  │  │   │   │   │   │   │   │   │   │   │   │   │   │   │   │  ││
│  │  ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼  ││
│  │ ═══════════════════════════════════════════════════════════ ││
│  │                    RAISED FLOOR PLENUM                      ││
│  │ ═══════════════════════════════════════════════════════════ ││
│  │  ▲   ▲   ▲   ▲   ▲   ▲   ▲   ▲   ▲   ▲   ▲   ▲   ▲   ▲   ▲  ││
│  │ [CRAC1] [CRAC2] [CRAC3] [CRAC4] [CRAC5] [CRAC6] ... [CRAC12]││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  HOT AISLE ↑↑↑↑↑↑↑↑↑↑↑↑ │ COLD AISLE ↓↓↓↓↓↓↓↓↓↓↓↓              │
│            (CONTAINMENT)                                        │
└─────────────────────────────────────────────────────────────────┘
         Height: 4.0 m (floor-to-ceiling) + 0.6 m raised floor
         200 server racks × 5–20 kW each
```

### Specifications

| Parameter | Value |
|-----------|-------|
| **Floor Area** | 500 m² |
| **Server Racks** | 200 |
| **Heat Load per Rack** | 5–20 kW (variable) |
| **Total Heat Load** | 1–4 MW |
| **CRAC Units** | 12 |
| **Containment** | Hot aisle / cold aisle |
| **Raised Floor** | 0.6 m plenum |

### Transient Scenario

**CRAC FAILURE EVENT**

> At $t = 0$, CRAC unit #4 fails.  
> Question: What is the thermal state at $t = 5$ minutes?

- Which racks exceed 35°C inlet temperature?
- What is the maximum temperature reached?
- How long until thermal runaway threshold?

### Physics Requirements

- Incompressible Navier-Stokes with energy (unsteady)
- Porous media model for server racks
- Pressure-driven raised floor plenum
- Variable heat source distribution
- Time-stepping: Δt ≤ 1 second for 300 seconds

### Deliverables

- [ ] Thermal runaway risk map
- [ ] Redundancy validation (N+1 cooling)
- [ ] Rack placement optimization recommendations
- [ ] Time-to-critical temperature curves
- [ ] CRAC failure cascade analysis

### Success Criteria

| Metric | Target |
|--------|--------|
| Energy balance | < 1% imbalance |
| Transient stability | CFL < 1.0 |
| Critical racks identified | 100% detection |
| Simulation runtime | < 8 hours for 5-min transient |

---

## TIER 5: Smoke/Fire/Atrium

**Budget**: $100,000 – $250,000+  
**Timeline**: 6–12 months

### Geometry

```
┌────────────────────────────────────────────────────────────────────────┐
│                              200 m                                     │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                                                                  │  │
│  │                          ☁️☁️☁️☁️☁️                              │  │
│  │                        SMOKE LAYER                               │  │
│  │                                                                  │  │ 25 m
│  │  [JET FAN]  [JET FAN]  [JET FAN]  [JET FAN]  [JET FAN]  [JET FAN]│  │
│  │                                                                  │  │
│  │  ┌──────────────────────────────────────────────────────────┐    │  │
│  │  │  GATE    GATE    GATE   🔥FIRE🔥  GATE    GATE    GATE    │    │  │
│  │  │   A1      A2      A3     B12      B13     B14     B15    │    │  │ 80 m
│  │  └──────────────────────────────────────────────────────────┘    │  │
│  │                                                                  │  │
│  │  👥👥👥👥👥👥👥👥👥👥👥👥👥👥👥👥👥👥👥👥👥👥👥👥👥👥👥👥👥👥│  │
│  │                    2000 OCCUPANTS EGRESSING                      │  │
│  │                                                                  │  │
│  │  [EXIT 1]                    [EXIT 2]                   [EXIT 3] │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  SMOKE EXTRACTION: 12 points @ ceiling                                 │
└────────────────────────────────────────────────────────────────────────┘
         Volume: 400,000 m³
```

### Specifications

| Parameter | Value |
|-----------|-------|
| **Dimensions** | 200m × 80m × 25m |
| **Volume** | 400,000 m³ |
| **Jet Fans** | 6 |
| **Smoke Extraction** | 12 points |
| **Fire Source** | 5 MW kiosk fire at Gate B12 |
| **Occupants** | 2,000 (egress simulation) |
| **Analysis Duration** | 15–30 minutes (fire timeline) |

### Tenability Criteria (NFPA 502 / BS 7974)

| Parameter | Tenable Limit | Untenable |
|-----------|---------------|-----------|
| **Visibility** | > 10 m | < 3 m |
| **Temperature** | < 60°C | > 80°C |
| **CO Concentration** | < 1,400 ppm | > 2,500 ppm |
| **Radiant Heat** | < 2.5 kW/m² | > 2.5 kW/m² |

### Physics Requirements

- Compressible Navier-Stokes (large temperature gradients)
- Species transport (smoke, CO, CO₂)
- Combustion model (prescribed HRR curve or fire model)
- Radiation (discrete ordinates or P1)
- Buoyancy-driven plume dynamics
- Coupled egress simulation (optional)

### Deliverables

- [ ] Smoke control system validation
- [ ] Time-to-untenable at each exit
- [ ] Visibility maps at 2.0 m height (egress level)
- [ ] Temperature stratification vs time
- [ ] Egress time certification
- [ ] Code compliance documentation (IBC, NFPA)

### Success Criteria

| Metric | Target |
|--------|--------|
| Available Safe Egress Time (ASET) | > Required Safe Egress Time (RSET) |
| Visibility at exits | > 10 m for duration of egress |
| Temperature at 2.0 m | < 60°C for egress duration |
| Smoke extraction effectiveness | > 80% of design flow achieved |

---

## TIER 6: Full Building CFD + BIM Integration

**Budget**: $200,000 – $500,000+  
**Timeline**: 6–18 months

### Geometry

```
                    ┌───────────┐
                    │  ROOF     │ ← Wind pressure mapping
                    │  PLANT    │
                    ├───────────┤
                    │ FLOOR 40  │ ← Stack effect calculation
                    ├───────────┤
                    │    ...    │
                    ├───────────┤
                    │ FLOOR 20  │ ← Mid-height pressure neutral plane
                    ├───────────┤
                    │    ...    │
                    ├───────────┤    ┌──────────────────┐
                    │ FLOOR 5   │    │  ELEVATOR SHAFT  │
                    ├───────────┤    │  & STAIRWELLS    │
                    │ FLOOR 4   │    │                  │
                    ├───────────┤    │  (Vertical       │
                    │ FLOOR 3   │    │   pathways for   │
                    ├───────────┤    │   smoke spread)  │
                    │ FLOOR 2   │    │                  │
                    ├───────────┤    └──────────────────┘
                    │ FLOOR 1   │
                    ├───────────┤
                    │  LOBBY    │ ← Infiltration / main entrance
                    ├───────────┤
                    │ BASEMENT  │ ← Central plant
                    └───────────┘
                    
           40-STORY MIXED-USE TOWER
```

### Specifications

| Parameter | Value |
|-----------|-------|
| **Stories** | 40 |
| **Use Type** | Mixed (retail, office, residential) |
| **Facade** | Pressure map from wind tunnel or external CFD |
| **HVAC** | Floor-by-floor with central plant |
| **Vertical Shafts** | Elevators, stairwells, MEP risers |
| **Fire Scenarios** | Multi-floor smoke spread |
| **Ventilation Study** | Natural ventilation potential |

### Analysis Domains

1. **External Aerodynamics**
   - Wind pressure coefficients on all facades
   - Fresh air intake location optimization
   - Exhaust re-entrainment risk

2. **Stack Effect**
   - Winter/summer pressure profiles
   - Neutral pressure plane location
   - Elevator shaft flow rates

3. **Floor-by-Floor HVAC**
   - Representative floor CFD (Tier 2 level)
   - Comfort validation per zone

4. **Smoke Spread**
   - Fire on Floor 15: where does smoke go?
   - Stairwell pressurization effectiveness
   - Elevator shaft as smoke pathway

5. **Natural Ventilation**
   - Cross-ventilation potential
   - Night purge cooling effectiveness
   - Window opening strategy

### Deliverables

- [ ] Full building commissioning support
- [ ] Digital twin integration (BIM ↔ CFD bidirectional)
- [ ] Operational optimization model
- [ ] Facade pressure database
- [ ] Stack effect mitigation recommendations
- [ ] Smoke spread timeline (fire scenario)
- [ ] Natural ventilation feasibility report

### Success Criteria

| Metric | Target |
|--------|--------|
| BIM geometry fidelity | < 5% deviation |
| Floor-to-floor pressure | Documented for all conditions |
| Smoke containment | Per code requirements |
| Digital twin sync latency | < 1 hour for geometry updates |

---

## Implementation Roadmap

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  TIER 1 ──► TIER 2 ──► TIER 3 ──► TIER 4 ──► TIER 5 ──► TIER 6         │
│    │          │          │          │          │          │             │
│    ▼          ▼          ▼          ▼          ▼          ▼             │
│  [Core]    [Thermal]  [Particle] [Transient] [Fire]    [Full]          │
│  [Flow]    [Comfort]  [Tracking] [Multi-Zone][Species] [Integration]   │
│                                                                         │
│  Each tier MUST pass validation before advancing.                       │
│  No tier may be skipped.                                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### QTT-Specific Considerations

| Tier | QTT Challenge | Mitigation |
|:----:|---------------|------------|
| 1 | Basic validation | Dense reference comparison (authorized) |
| 2 | Thermal coupling | QTT-native energy equation |
| 3 | Particle tracking | Lagrangian on QTT velocity field |
| 4 | Time-stepping | QTT-RK4 or QTT-IMEX schemes |
| 5 | Species transport | Multi-field QTT bundle |
| 6 | Multi-scale | Hierarchical QTT decomposition |

---

## Validation Strategy

Each tier requires:

1. **Analytical Benchmark** (where possible)
2. **Experimental Data** (published or measured)
3. **Cross-Solver Comparison** (OpenFOAM, Fluent)
4. **Proof Artifact** (per Constitution Article I)

---

*Proving Grounds — HVAC CFD — Project The Physics OS*
