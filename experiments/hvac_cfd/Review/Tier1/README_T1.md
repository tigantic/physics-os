# TigantiCFD - Tier 1 Conference Room Analysis

## Project: TGC-2026-001
**Client:** James Morrison, Morrison & Associates Law Firm  
**Scope:** Conference Room B - Thermal Comfort Assessment

---

## Quick Start

```bash
# Install dependencies
pip install torch numpy matplotlib

# Run analysis
python tier1_james_conference_room.py

# Or with custom output directory
python tier1_james_conference_room.py ./my_output
```

---

## Files Included

| File | Description |
|------|-------------|
| `tier1_james_conference_room.py` | Complete Tier 1 simulation with boundary injection fix |
| `qtt_nielsen_runner.py` | Nielsen benchmark validation (mass conservation + accuracy) |
| `qtt_ns_3d_fixed.py` | Core 3D solver with QTT infrastructure |
| `run_t1.py` | Quick-start runner script |

---

## Output Deliverables

After running, you'll find in `./tiganti_output/TGC-2026-001/`:

```
TGC-2026-001/
├── CFD_Analysis_Report.txt     # Full technical report (client-ready)
├── analysis_summary.json       # Machine-readable results
├── velocity_xy_plane.png       # Top view at breathing height (1.2m)
├── velocity_xz_plane.png       # Side view centerline
└── convergence_history.png     # Solver convergence plot
```

---

## Room Configuration

```
┌─────────────────────────────────────────────────────────┐
│  CONFERENCE ROOM B                                       │
│  26' × 25' × 10' (650 sq ft)                            │
│                                                          │
│  ┌──────┐                                    ┌──────┐   │
│  │SUPPLY│ ←── Ceiling Diffuser (450 FPM)     │RETURN│   │
│  └──────┘                                    └──────┘   │
│                                                          │
│           ╔═══════════════════════════╗                 │
│           ║                           ║                 │
│           ║    CONFERENCE TABLE       ║                 │
│           ║       (12 seats)          ║                 │
│           ║                           ║                 │
│           ╚═══════════════════════════╝                 │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Technical Details

### Solver
- 3D Incompressible Navier-Stokes
- Skew-symmetric advection (energy conserving)
- **Boundary injection fix** (prevents periodic wrap)
- Effective viscosity turbulence model

### Grid
- Default: 64×64×32 = 131,072 cells
- Resolution: ~12cm (adequate for room-scale HVAC)

### Validation
- Nielsen IEA Annex 20 benchmark: **<10% RMS error**
- Mass conservation: **<5% change with inlet/outlet flux**
- ASHRAE 55-2020 comfort criteria

---

## Comfort Metrics (ASHRAE 55-2020)

| Metric | Target | Measured |
|--------|--------|----------|
| Air speed (occupied) | 0.05-0.25 m/s | See report |
| Draft risk (>0.25 m/s) | Minimal | See report |
| Stagnation (<0.05 m/s) | <20% | See report |
| Comfort Score | >70 | See report |

---

## Customization

Edit `tier1_james_conference_room.py` to modify:

### Client Info
```python
@dataclass
class ClientInfo:
    client_name: str = "James Morrison"
    company: str = "Morrison & Associates Law Firm"
    # ...
```

### Room Geometry
```python
@dataclass  
class RoomGeometry:
    length_ft: float = 26.0
    width_ft: float = 25.0
    height_ft: float = 10.0
    supply_velocity_fpm: float = 450
    # ...
```

### Solver Settings
```python
@dataclass
class SolverConfig:
    nx: int = 64
    ny: int = 64
    nz: int = 32
    t_end: float = 120.0  # Simulation time (s)
    # ...
```

---

## Tier Comparison

| Tier | Grid | Features | Delivery | Price |
|------|------|----------|----------|-------|
| **T1** | 64³ | Isothermal, steady-state | 48 hrs | $2,500 |
| T2 | 128³ | Thermal, solar loads | 72 hrs | $4,500 |
| T3 | 256³ | Full transient, contaminants | 1 week | $8,000 |
| T4 | 512³ | Multi-zone, parametric | 2 weeks | $15,000 |
| T5 | Custom | Real-time digital twin | Ongoing | $50,000+ |

---

## Support

**Tigantic Holdings LLC**  
Email: support@tigantic.com  
Web: https://tigantic.com

---

## License

Proprietary - © 2026 Tigantic Holdings LLC  
All Rights Reserved

This software is provided for the exclusive use of licensed TigantiCFD customers.
Unauthorized distribution or use is prohibited.
