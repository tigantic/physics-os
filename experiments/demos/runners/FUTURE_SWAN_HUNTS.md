# 🦢 Future Black Swan Hunts

**Goal:** Find ONE initial condition that causes rank explosion → potential Navier-Stokes singularity

**Current Status:** Taylor-Green vortex confirmed as "white swan" (smooth, no blowup)

---

## Priority Queue

### 1. Colliding Vortex Rings (HIGH PRIORITY)
```
u, v, w = two_vortex_rings(
    ring1_center=(π, π, π/2), ring1_radius=π/2, ring1_strength=1.0,
    ring2_center=(π, π, 3π/2), ring2_radius=π/2, ring2_strength=1.0,
    collision_axis='z'
)
```
- **Why:** Violent vortex reconnection events
- **Literature:** Kida & Takaoka (1994), Kerr (2013)
- **Expected:** Higher rank, possible explosion during reconnection
- **Command:** `python demos/trap_the_swan.py --ic vortex_rings`

### 2. Kida Vortex
```
u = sin(x)(cos(3y)cos(z) - cos(y)cos(3z))
v = sin(y)(cos(3z)cos(x) - cos(z)cos(3x))
w = sin(z)(cos(3x)cos(y) - cos(x)cos(3y))
```
- **Why:** Known to develop intense vortex stretching
- **Literature:** Kida (1985)
- **Expected:** More aggressive dynamics than Taylor-Green
- **Command:** `python demos/trap_the_swan.py --ic kida`

### 3. Anti-Parallel Vortex Tubes
```
Two parallel vortex tubes with opposite circulation
Separation distance: d = π/4
Core radius: a = π/8
```
- **Why:** Simplified 2D reconnection geometry, well-studied
- **Literature:** Pumir & Siggia (1990)
- **Expected:** Localized intense strain
- **Command:** `python demos/trap_the_swan.py --ic antiparallel`

### 4. Trefoil Vortex Knot
```
Vortex line following trefoil knot topology
Parametric: (sin(t) + 2sin(2t), cos(t) - 2cos(2t), -sin(3t))
```
- **Why:** Topological changes during unknotting
- **Literature:** Kleckner & Irvine (2013)
- **Expected:** Complex reconnection cascade
- **Command:** `python demos/trap_the_swan.py --ic trefoil`

### 5. Random High-Wavenumber Turbulence
```
u = Σ A_k * exp(i*k·x) with |k| > k_threshold
Random phases, prescribed energy spectrum E(k) ~ k^(-5/3)
```
- **Why:** Brute force chaos — if singularity exists, might trigger it
- **Expected:** High initial rank, unpredictable dynamics
- **Command:** `python demos/trap_the_swan.py --ic random_turb`

---

## Run Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Grid | 512³ | Balance of resolution and compute |
| Rank Cap | 1024 | Let physics breathe |
| Blowup Threshold | 400 | Trigger evidence capture |
| t_max | 10.0 (or until trigger) | Singularity zone |

---

## Running on Remote Machine

1. Clone repo: `git clone https://github.com/tigantic/The Physics OS.git`
2. Install deps: `pip install -r requirements-lock.txt && pip install dilithium-py`
3. Run trap: `python demos/trap_the_swan.py --ic <IC_NAME>`
4. Check `logs/` for BLACK_SWAN_*.json or HUNT_COMPLETE_*.json

---

## Completed Runs

| IC | Grid | t_reached | Final Rank | Result |
|----|------|-----------|------------|--------|
| Taylor-Green | 512³ | 10.0 | 34 | ✅ White swan (rank cap=128) |
| Taylor-Green | 512³ | 2.59 | 27 | ✅ White swan (rank cap=1024, aborted) |
| Taylor-Green | 128³ | 10.0 | 36 | ✅ White swan |
| Taylor-Green | 64³ | 7.0 | 37 | ✅ White swan |
| Taylor-Green | 32³ | 10.0 | 39 | ✅ White swan |

---

## The Asymmetry

- **Proving smoothness:** Must show for ALL initial conditions (impossible)
- **Proving blowup:** Need only ONE counterexample (achievable)

We're hunting the ONE black swan. Every white swan just means "try different bait."
