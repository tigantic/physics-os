# HVAC CFD Validation Guide: Nielsen/IEA Annex 20 Benchmark

## BENCHMARK SOURCES (DOWNLOAD THESE FIRST)

**Official CFD Benchmarks Website:**
- Main site: https://www.en.build.aau.dk/web/cfd-benchmarks
- 2D Benchmark page: https://www.en.build.aau.dk/web/cfd-benchmarks/two-dimensional-benchmark-test

**Direct Downloads:**
- Benchmark Test Report (PDF): https://www.aaudxp-cms.aau.dk/media/v0xnnqtw/592301_2d_benchmark_test.pdf
- Experimental Measurements (Excel): https://www.aaudxp-cms.aau.dk/media/udffy5eg/592302_2d_-benchmark_test_measurements_new.xls
- Literature List: https://www.aaudxp-cms.aau.dk/media/ybamfbkg/literature-list-2d-benchmark-test.pdf

---

## IEA ANNEX 20 2D BENCHMARK GEOMETRY

### Dimensionless Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| L/H | 3.0 | Room length to height ratio |
| h/H | 0.056 | Inlet slot height to room height ratio |
| W/H | 1.0 | Room width to height ratio (for 3D) |
| Re | 5000 | Reynolds number based on inlet height and velocity |

### Physical Dimensions (Standard Configuration)
| Dimension | Value | Notes |
|-----------|-------|-------|
| H | 3.0 m | Room height |
| L | 9.0 m | Room length (L = 3H) |
| W | 3.0 m | Room width (W = H) |
| h_inlet | 0.168 m | Inlet slot height (h = 0.056H) |
| h_outlet | 0.48 m | Outlet slot height |
| U_inlet | 0.455 m/s | Inlet velocity (uniform profile) |

### Geometry Layout
```
+--h--+------------------------------------------+
|inlet|                                          |
+-----+                                          |
|                                                |
|              ROOM (L x H)                      |
|                                                |
|                                          +-----+
|                                          |outlt|
+------------------------------------------+--h--+
```

- Inlet: Upper left corner, ceiling-attached slot
- Outlet: Lower right corner, floor-adjacent slot
- Flow: Horizontal jet along ceiling, recirculation in room

---

## BOUNDARY CONDITIONS

### Inlet
- Type: Velocity inlet
- Velocity: 0.455 m/s (uniform profile)
- Turbulence intensity: < 1% (low turbulence)
- Hydraulic diameter: 0.168 m

### Outlet
- Type: Pressure outlet or outflow
- Gauge pressure: 0 Pa

### Walls
- Type: No-slip wall
- Thermal: Adiabatic (isothermal case)

### Reynolds Number Calculation
```
Re = (U × h) / ν = (0.455 × 0.168) / (1.5e-5) ≈ 5000
```
Where ν = 1.5×10⁻⁵ m²/s (air at ~20°C)

---

## MEASUREMENT LOCATIONS

### Vertical Profiles (measure U-velocity vs y)
| Location | x/H | x (meters) |
|----------|-----|------------|
| Profile 1 | 1.0 | 3.0 m |
| Profile 2 | 2.0 | 6.0 m |

### Horizontal Profiles (measure U-velocity vs x)
| Location | y/H | y (meters) | Description |
|----------|-----|------------|-------------|
| Profile 3 | 0.028 | 0.084 m | Near ceiling (jet region) |
| Profile 4 | 0.972 | 2.916 m | Near floor (reverse flow) |

---

## EXECUTION STEPS

### Step 1: Environment Setup
```bash
# Install required packages (pick your solver)
pip install numpy scipy matplotlib pandas

# For OpenFOAM
# apt install openfoam

# For Python-based: FEniCS, PyFR, or your QTT solver
```

### Step 2: Create Geometry
```python
# Domain dimensions
H = 3.0      # height (m)
L = 9.0      # length (m)  
W = 3.0      # width (m) - for 3D, use 1 cell for 2D

# Inlet/outlet
h_inlet = 0.168   # inlet height
h_outlet = 0.48   # outlet height

# Inlet position: x=0, y=[H-h_inlet, H]
# Outlet position: x=L, y=[0, h_outlet]
```

### Step 3: Mesh Requirements
```
Minimum cells (2D): 50×50 = 2,500
Recommended (2D): 100×100 = 10,000
Fine (2D): 200×200 = 40,000

For 3D: multiply by ~50 cells in z-direction

Wall y+ target: < 1 (if using wall-resolved)
             : 30-300 (if using wall functions)

Grid refinement: Near inlet slot, ceiling, and walls
```

### Step 4: Solver Settings
```
Turbulence model (RANS): 
  - Standard k-ε (baseline)
  - RNG k-ε (recommended)
  - k-ω SST (alternative)

Numerical schemes:
  - Second-order upwind for momentum
  - Second-order for pressure
  - SIMPLE or PISO algorithm

Convergence criteria:
  - Residuals < 1e-5
  - Monitor velocity at key points
```

### Step 5: Run Simulation
```
1. Initialize with uniform velocity = 0
2. Run to steady state (or pseudo-transient)
3. Monitor:
   - Inlet mass flow rate
   - Outlet mass flow rate (should balance)
   - Velocity at x/H=1, y/H=0.5
4. Continue until residuals converged
```

### Step 6: Extract Data
```python
# Vertical profiles at x/H = 1 and x/H = 2
for x_H in [1.0, 2.0]:
    x = x_H * H
    y_points = np.linspace(0, H, 50)
    U_profile = extract_velocity(x, y_points)
    
# Horizontal profiles at y/H = 0.028 and y/H = 0.972
for y_H in [0.028, 0.972]:
    y = y_H * H
    x_points = np.linspace(0, L, 100)
    U_profile = extract_velocity(x_points, y)
```

---

## VALIDATION CRITERIA

### Quantitative Targets
| Metric | Acceptable | Good |
|--------|------------|------|
| Max velocity location at x/H=1 | Within 5% of H | Within 2% |
| Max velocity magnitude | Within 15% | Within 10% |
| Jet spreading rate | Visible agreement | Quantitative match |
| Recirculation zone size | Qualitative match | Within 10% |

### What "Passing" Looks Like
1. **Jet attachment**: Flow stays attached to ceiling
2. **Velocity decay**: Peak velocity decreases with distance from inlet
3. **Recirculation**: Single large clockwise vortex fills room
4. **Jet profile shape**: Gaussian-like at x/H=1 and x/H=2
5. **Near-floor reversal**: Negative U-velocity near floor

### Error Calculation
```python
# Normalize velocities by inlet velocity
U_norm_cfd = U_cfd / U_inlet
U_norm_exp = U_exp / U_inlet

# Calculate RMS error
error = np.sqrt(np.mean((U_norm_cfd - U_norm_exp)**2))

# Target: error < 0.10 (10% of inlet velocity)
```

---

## EXPERIMENTAL DATA FORMAT

The Excel file contains columns:
- `x/H` or `y/H`: Normalized position
- `U/U0`: Normalized velocity (U0 = inlet velocity)
- `u'/U0`: Turbulence intensity (optional)

Data is provided for:
- Isothermal case (primary)
- Non-isothermal case (with temperature difference)

---

## COMMON ISSUES & FIXES

| Problem | Cause | Fix |
|---------|-------|-----|
| Jet detaches from ceiling | Insufficient mesh at inlet | Refine mesh near slot |
| Wrong recirculation pattern | Numerical diffusion | Use higher-order schemes |
| Velocity too high/low | Wrong inlet BC | Check mass flow rate |
| Asymmetric flow | 3D effects | Use 3D mesh if needed |
| Slow convergence | Under-relaxation too high | Reduce URF to 0.3-0.5 |

---

## PROGRESSION TO HVAC APPLICATIONS

### After Passing Nielsen Benchmark:

**Level 2: Add Thermal**
- Add temperature difference (inlet 5-10°C below room)
- Validate against non-isothermal Nielsen data
- Check buoyancy effects (Archimedes number)

**Level 3: Realistic Geometry**
- Add furniture/obstacles
- Multiple supply/return vents
- Complex room shapes

**Level 4: HVAC-Specific**
- Diffuser modeling (box method or momentum source)
- People as heat sources (75-100W each)
- Equipment loads
- Transient analysis

---

## REFERENCE PAPERS

1. Nielsen, P.V. (1990). "Specification of a Two-Dimensional Test Case (IEA)"
2. Nielsen, P.V., Rong, L., Olmedo, I. (2010). "The IEA Annex 20 Two-Dimensional Benchmark Test for CFD Predictions"
3. Nielsen, P.V. (2015). "Fifty years of CFD for room air distribution" - Building and Environment, Vol 91
4. Sørensen, D.N., Nielsen, P.V. (2003). "Quality control of CFD in indoor environments"

---

## QUICK START CHECKLIST

- [ ] Download benchmark PDF and Excel from AAU website
- [ ] Create 2D geometry: 9m × 3m room
- [ ] Define inlet slot: 0.168m height, U = 0.455 m/s
- [ ] Define outlet slot: 0.48m height, pressure outlet
- [ ] Generate mesh: minimum 100×100 cells
- [ ] Set turbulence model: RNG k-ε
- [ ] Run to convergence: residuals < 1e-5
- [ ] Extract velocity profiles at x/H = 1 and x/H = 2
- [ ] Compare with experimental data from Excel file
- [ ] Calculate normalized RMS error
- [ ] Target: < 10% error at measurement locations
