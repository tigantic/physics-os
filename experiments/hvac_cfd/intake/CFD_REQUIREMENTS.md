# CFD Simulation Requirements for HVAC Thermal Comfort

## Minimum Data Required for Reliable Results

This document defines the bare minimum inputs required for accurate HVAC CFD simulation.
Missing any of these will cause the simulation to either **crash** or produce **junk data**.

---

## 1. Geometry (The "Domain")

You cannot simulate what you cannot define. At minimum, you need a **watertight 3D volume**.

| Input | Required | Notes |
|-------|----------|-------|
| Room Length | ✅ CRITICAL | Meters (internally) |
| Room Width | ✅ CRITICAL | Meters (internally) |
| Room Height | ✅ CRITICAL | Meters (internally) |
| Supply Diffuser Location | ✅ CRITICAL | Position in room |
| **Diffuser Effective Area** | ⚠️ IMPORTANT | Open area where air exits (NOT face size) |
| Return Grille Location | ✅ CRITICAL | Position in room |
| Major Obstructions | Optional | Only if they block airflow significantly |

> **Critical Note on Effective Area:**
> If you model a 12"×12" diffuser as a fully open hole (0.093 m²), your inlet velocity 
> will be too low. The **effective area** is typically 40-60% of face area due to vanes,
> dampers, and frame. A 12" diffuser typically has ~0.05 m² effective area.

---

## 2. Boundary Conditions (The "Driver")

This is the most critical physics section.

| Boundary | Required Data | Why It's Needed |
|----------|--------------|-----------------|
| **Inlet (Supply)** | Velocity (m/s) OR Flow Rate (m³/s) | Drives momentum |
| | + Supply Temperature (°C) | Drives cooling/heating capacity |
| **Outlet (Return)** | 0 Pa Gauge Pressure | Allows air to exit without backpressure |
| **Walls** | Temperature OR Heat Flux OR Adiabatic | Defines envelope heat transfer |
| **Heat Sources** | Heat Generation (W) | People, equipment, lighting |

### Wall Thermal Boundary Options:

1. **Adiabatic** (Easiest, Lowest Accuracy)
   - Walls are perfect insulators
   - Good for checking airflow patterns only

2. **Fixed Temperature** (Medium Accuracy)
   - Assign surface temperatures (e.g., windows 30°C, walls 24°C)
   - Good for worst-case scenarios

3. **U-Value / Heat Flux** (Best Accuracy)
   - q = U × (T_outdoor - T_indoor)
   - Calculate from wall R-value and ΔT

---

## 3. Fluid Properties

For standard HVAC (non-industrial), use **Incompressible Ideal Gas** or **Boussinesq approximation**.

| Property | Value | Notes |
|----------|-------|-------|
| Fluid | Air | |
| Density (ρ) | 1.2 kg/m³ | At sea level, 20°C |
| Dynamic Viscosity (μ) | 1.8×10⁻⁵ Pa·s | |
| Kinematic Viscosity (ν) | 1.5×10⁻⁵ m²/s | μ/ρ |
| Specific Heat (cₚ) | 1005 J/(kg·K) | |
| Thermal Conductivity (k) | 0.026 W/(m·K) | |
| Prandtl Number (Pr) | 0.71 | |

---

## 4. Turbulence Model

Industry standard minimum for HVAC is **RANS** with **k-ε**.

| Option | Use Case |
|--------|----------|
| k-ε Standard | General HVAC, computationally cheap |
| k-ε Realizable | Better for jets and recirculation |
| k-ω SST | Better near-wall treatment |
| Laminar | Very low Re flows only |

---

## 5. Gravity / Buoyancy

**CRITICAL** - Without gravity, hot air won't rise!

```
Gravity: g = -9.81 m/s² (in Y-axis, assuming Y is up)
Buoyancy: Boussinesq approximation enabled
```

---

## Implementation Checklist

Before running simulation, verify:

- [ ] **3D Manifold Geometry** - Watertight volume
- [ ] **Inlet Velocity** - Calculated from CFM ÷ Effective Area
- [ ] **Inlet Temperature** - Supply air temperature in °C
- [ ] **Outlet Pressure** - Set to 0 Pa gauge
- [ ] **Heat Sources** - Watts per person/equipment defined
- [ ] **Gravity** - Enabled at -9.81 m/s² in vertical axis

---

## Unit System (The "Sandwich Method")

```
INPUT (Imperial/User) → SOLVER (SI) → OUTPUT (Imperial/User)
```

All solver calculations use SI units internally:
- Length: meters (m)
- Temperature: Kelvin (K) for solver, Celsius (°C) for display
- Velocity: m/s
- Airflow: m³/s
- Pressure: Pascals (Pa)

Conversion constants (NIST SP 811):
- 1 ft = 0.3048 m (exact)
- 1 CFM = 4.71947×10⁻⁴ m³/s
- K = °C + 273.15
- °F = °C × 9/5 + 32

---

## References

- ASHRAE Fundamentals Handbook
- ASHRAE Standard 55 (Thermal Comfort)
- ASHRAE Standard 62.1 (Ventilation)
- NIST SP 811 (Guide for SI Units)
