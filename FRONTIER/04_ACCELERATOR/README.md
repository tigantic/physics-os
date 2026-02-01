# Frontier 04: Particle Accelerator Design

**$50 Billion in Accelerators Need Better Beam Physics**

## The Problem

Particle accelerators are the most precise machines ever built:
- **LHC**: 27 km ring, $10B, Nobel-winning physics
- **LCLS-II**: World's most powerful X-ray laser, $1B
- **EIC**: Electron-ion collider, $2.5B under construction
- **Muon Collider**: Next-gen concept, $10B+ projected

The bottleneck: **beam-beam interactions** and **collective effects**.

When two particle bunches collide (or even pass nearby):
- 10¹¹ particles interact with 10¹¹ particles
- Each particle feels the collective field of all others
- This is a **6D kinetic problem** — not tractable with current methods

### The Cost of Getting It Wrong

| Failure Mode | Consequence | Example |
|--------------|-------------|---------|
| Beam instability | Machine damage | LHC beam dump: €100M+ repairs |
| Luminosity loss | Wasted beam time | $1M/hour opportunity cost |
| Emittance growth | Physics reach limited | Reduced discovery potential |

## The Market

| Segment | Investment | Key Players |
|---------|------------|-------------|
| **High-Energy Physics** | $10B/decade | CERN, FNAL, SLAC, KEK |
| **Light Sources** | $5B/decade | LCLS, ESRF, PETRA, SPRING-8 |
| **Medical Accelerators** | $3B/year | Proton therapy, isotope production |
| **Industrial** | $2B/year | Ion implantation, sterilization |
| **Defense** | Classified | FEL weapons, ADS |

## Validation Roadmap

### Phase 1: Single Beam Dynamics (Week 1-2)

| Benchmark | Description | Success Metric |
|-----------|-------------|----------------|
| **Matched Beam** | Self-consistent equilibrium | Emittance preservation |
| **Filamentation** | Mismatched beam evolution | Phase space structure |
| **Synchrotron Oscillation** | Longitudinal dynamics | Tune measurement |

### Phase 2: Instabilities (Week 3-4)

| Benchmark | Description | Success Metric |
|-----------|-------------|----------------|
| **Transverse Mode Coupling** | TMCI threshold | Current limit match |
| **Head-Tail Instability** | Chromaticity-driven | Growth rate |
| **Microwave Instability** | Longitudinal coasting | CSR effects |

### Phase 3: Beam-Beam Effects (Month 2)

| Benchmark | Description | Success Metric |
|-----------|-------------|----------------|
| **Beam-Beam Tune Shift** | ΔQ from collision | σ dependence |
| **Luminosity Scan** | Beam separation effects | Van der Meer calibration |
| **Coherent Instabilities** | Coupled bunch motion | Mode spectrum |

### Phase 4: Collider Optimization (Month 3)

| Benchmark | Description | Success Metric |
|-----------|-------------|----------------|
| **Dynamic Aperture** | Long-term stability | 10⁶ turn survival |
| **Crab Cavity Crossing** | Angle compensation | Luminosity recovery |
| **Integrated Luminosity** | Full run simulation | fb⁻¹ prediction |

## Technical Approach

### Coordinate System
```
Accelerator coordinates:
  (x, y, z): Transverse positions, longitudinal offset from reference
  (px, py, δ): Normalized momenta, energy deviation
  
Or action-angle:
  (Jx, Jy, Jz): Action variables (invariants)
  (φx, φy, φz): Phase angles
```

### Beam Parameters
```python
# LHC-like proton beam
E_beam = 7000        # GeV
n_particles = 1.15e11  # per bunch
n_bunches = 2808
sigma_x = 16.6e-6    # m (IP)
sigma_y = 16.6e-6    # m (IP)
sigma_z = 7.5e-2     # m (bunch length)
epsilon_n = 3.75e-6  # m·rad (normalized emittance)
```

### Beam-Beam Interaction
```
Beam-beam force:
F(x,y) = -∂U/∂(x,y)

where U is the potential from the opposing beam
(Gaussian → Bassetti-Erskine formula)

Full kinetic treatment required for:
- Beam-beam resonances
- Long-range interactions
- Coherent modes
```

### QTT Representation

The key insight: beam distributions are **separable** to good approximation.

```
f(x, y, z, px, py, δ) ≈ f_x(x, px) × f_y(y, py) × f_z(z, δ) + corrections
```

QTT exploits this structure while capturing correlations through low-rank coupling.

## Deliverables

### Code
- `beam_beam_demo.py`: Two-beam collision simulation
- `instability_scan.py`: Threshold determination
- `luminosity_optimizer.py`: Collision parameter tuning
- `dynamic_aperture.py`: Long-term tracking
- `lattice_import.py`: MAD-X / elegant interface

### Validation
- Comparison with GUINEA-PIG++ (weak-strong)
- Comparison with BEAMBEAM3D (strong-strong)
- Experimental data from LHC Van der Meer scans

### Interface
```python
# Accelerator physicist API
result = simulate_collision(
    beam1=Beam(energy=7000, particles=1.15e11, ...),
    beam2=Beam(energy=7000, particles=1.15e11, ...),
    crossing_angle=285e-6,  # rad
    turns=1000,
    grid_6d=(32, 32, 32, 32, 32, 32),
)

# Returns
result.luminosity       # Instantaneous and integrated
result.tune_shift       # Beam-beam induced
result.emittance_growth # Rate per hour
result.stability_map    # Tune footprint
```

## Business Model

| Offering | Price | Target Customer |
|----------|-------|-----------------|
| **Research License** | Free/Academic | Universities |
| **Lab License** | $100K/year | National labs |
| **Commercial License** | $500K/year | Medical accelerator OEMs |
| **Consulting** | $3K/day | Accelerator projects |
| **Custom Development** | $1M+ project | New facility design |

## Competitive Analysis

| Code | Method | Beam-Beam | Speed |
|------|--------|-----------|-------|
| MAD-X | Transfer maps | ❌ | Fast |
| elegant | Particle tracking | Weak-strong | Medium |
| GUINEA-PIG | Monte Carlo | Strong-strong | Slow |
| BEAMBEAM3D | PIC | Strong-strong | Very slow |
| **QTT (ours)** | **6D kinetic** | **Strong-strong** | **Fast** |

## Key References

1. Chao, A. W. & Tigner, M. (2013). "Handbook of Accelerator Physics and Engineering"
2. Herr, W. & Pieloni, T. (2014). "Beam-Beam Effects" (CERN Yellow Reports)
3. Ohmi, K. et al. (2004). "Beam-beam limit for e+e- circular colliders"
4. Qiang, J. et al. (2006). "BEAMBEAM3D: A parallel simulation code"

## Success Criteria

- [ ] Single-beam emittance preservation validated
- [ ] Beam-beam tune shift matches theory
- [ ] TMCI threshold within 10% of measurement
- [ ] Demo to accelerator lab (SLAC, FNAL, CERN)
- [ ] Integration with MAD-X workflow
- [ ] First consulting contract ($50K+)

---

*ELITE Engineering — Simulating 10¹¹ × 10¹¹ particle collisions in seconds*
