# HyperTensor Universal Physics Solver — Extraction Map

## Target: Standalone Package `hypertensor-physics`

Extract the core physics engine into a minimal, dependency-light universal solver 
that can be pip-installed and used anywhere.

---

## Package Structure

```
hypertensor-physics/
├── hypertensor/
│   ├── __init__.py              # Public API
│   ├── core/
│   │   ├── __init__.py
│   │   ├── tensor_train.py      # TTTensor, tt_round, tt_to_full, tt_add, tt_dot
│   │   ├── decompositions.py    # svd_truncated, qr_stabilized
│   │   └── constants.py         # Physical constants (k_B, e, m_p, μ₀, ε₀, c, ℏ)
│   │
│   ├── integrators/
│   │   ├── __init__.py
│   │   ├── symplectic.py        # SymplecticIntegrator (Verlet, Leapfrog)
│   │   ├── langevin.py          # LangevinDynamics (BAOAB)
│   │   └── runge_kutta.py       # RK4, RK45 (adaptive)
│   │
│   ├── pde/
│   │   ├── __init__.py
│   │   ├── mhd.py               # ResistiveMHD, IdealMHD
│   │   ├── fokker_planck.py     # FokkerPlanck (probability evolution)
│   │   ├── euler.py             # CompressibleEuler (CFD)
│   │   └── diffusion.py         # HeatEquation, FickDiffusion
│   │
│   ├── quantum/
│   │   ├── __init__.py
│   │   ├── schrodinger.py       # TimeEvolution, GroundState
│   │   └── hamiltonians.py      # IsingModel, HeisenbergModel
│   │
│   ├── materials/
│   │   ├── __init__.py
│   │   ├── superconductor.py    # AllenDynes Tc, Gap equation
│   │   ├── thermal.py           # HeatConduction, CompositeWall
│   │   └── ionic.py             # NernstEinstein, ArrheniusMobility
│   │
│   └── utils/
│       ├── __init__.py
│       ├── units.py             # Unit conversion helpers
│       └── attestation.py       # SHA-256 proof generation
│
├── tests/
├── examples/
├── pyproject.toml
└── README.md
```

---

## Extraction Sources

### 1. CORE: Tensor-Train Compression

| Target File | Source | What to Extract |
|-------------|--------|-----------------|
| `core/tensor_train.py` | `hypertensor_dynamics.py` L31-93 | `TTTensor`, `tt_round`, `tt_to_full` |
| `core/tensor_train.py` | `tensornet/core/decompositions.py` | `svd_truncated` (torch version) |
| `core/constants.py` | `starheart_fusion_solver.py` L39-53 | Physical constants |

**Key Innovation (The Patent):**
```python
def step(self, state, dt):
    state_new = physics_update(state, dt)
    return tt_round(state_new, max_rank=12)  # RE-COMPRESS EVERY STEP
```

---

### 2. INTEGRATORS: Time Evolution

| Target File | Source | What to Extract |
|-------------|--------|-----------------|
| `integrators/symplectic.py` | `hypertensor_dynamics.py` L102-138 | `SymplecticIntegrator` |
| `integrators/langevin.py` | `hypertensor_dynamics.py` L145-222 | `LangevinDynamics` |
| `integrators/runge_kutta.py` | NEW (standard RK4/RK45) | Adaptive stepping |

---

### 3. PDE SOLVERS: Continuous Physics

| Target File | Source | What to Extract |
|-------------|--------|-----------------|
| `pde/mhd.py` | `hypertensor_dynamics.py` L229-316 | `ResistiveMHD` |
| `pde/fokker_planck.py` | `hypertensor_dynamics.py` L323-401 | `FokkerPlanck` |
| `pde/euler.py` | `tensornet/physics/hypersonic.py` | Dynamic pressure, shocks |
| `pde/diffusion.py` | `hellskin_thermal_solver.py` | Heat equation |

---

### 4. QUANTUM: Many-Body Physics

| Target File | Source | What to Extract |
|-------------|--------|-----------------|
| `quantum/schrodinger.py` | `experiments/benchmarks/experiments/benchmarks/benchmarks/tfim_ground_state.py` | DMRG concepts |
| `quantum/hamiltonians.py` | `tensornet/core/mpo.py` | MPO builders |

---

### 5. MATERIALS: Domain Solvers

| Target File | Source | What to Extract |
|-------------|--------|-----------------|
| `materials/superconductor.py` | `odin_superconductor_solver.py` | Allen-Dynes, gap equation |
| `materials/thermal.py` | `hellskin_thermal_solver.py` | Composite wall analysis |
| `materials/ionic.py` | `ssb_superionic_solver.py` | Nernst-Einstein conductivity |

---

## Minimal Dependencies

```toml
[project]
name = "hypertensor-physics"
version = "0.1.0"
dependencies = [
    "numpy>=1.20",
]

[project.optional-dependencies]
gpu = ["torch>=2.0"]
quantum = ["scipy>=1.7"]
full = ["torch>=2.0", "scipy>=1.7"]
```

**Design Goal:** NumPy-only core, optional torch/scipy for advanced features.

---

## Public API (What Users Import)

```python
from hypertensor import TTTensor, tt_round, tt_to_full

# Integrators
from hypertensor.integrators import SymplecticIntegrator, LangevinDynamics

# PDE Solvers
from hypertensor.pde import ResistiveMHD, FokkerPlanck, HeatEquation

# Materials
from hypertensor.materials import allen_dynes_tc, nernst_einstein

# Convenience: All-in-one physics step
from hypertensor import HyperStep

def my_simulation():
    state = initial_condition()
    for t in time_steps:
        state = HyperStep(state, dt, physics="mhd", max_rank=12)
```

---

## What Gets LEFT BEHIND

These stay in the main HyperTensor-VM repository:

| Category | Files | Why Left Behind |
|----------|-------|-----------------|
| **Apps** | `apps/glass_cockpit/`, `apps/global_eye/` | GUI-specific |
| **Demos** | `demos/*.py` | Visualization, domain-specific |
| **Benchmarks** | `experiments/benchmarks/benchmarks/*.py` | Testing/validation |
| **Proofs** | `proofs/*.py` | Attestation scripts |
| **Domain Modules** | `tensornet/defense/`, `tensornet/medical/`, etc. | Vertical integrations |
| **Discovery Solvers** | `tig011a_*.py`, `euv_*.py`, `starheart_*.py` | Application-specific |
| **Rust Bindings** | `crates/`, `crates/tci_core_rust/` | Separate package |

---

## Extraction Commands

```bash
# 1. Create new package
mkdir -p hypertensor-physics/hypertensor/{core,integrators,pde,quantum,materials,utils}

# 2. Extract core (from this repo)
cp hypertensor_dynamics.py hypertensor-physics/hypertensor/

# 3. Split into modules
# (See detailed extraction below)

# 4. Create pyproject.toml
# 5. Run tests
# 6. Publish to PyPI
```

---

## File-by-File Extraction

### `hypertensor/core/tensor_train.py`

```python
# Extract from: hypertensor_dynamics.py lines 28-93
# Add: tt_add, tt_dot, tt_matvec operations
# Add: torch backend option

@dataclass
class TTTensor:
    cores: list
    shape: tuple
    ranks: tuple
    
def tt_round(tensor, max_rank=12) -> TTTensor: ...
def tt_to_full(tt) -> np.ndarray: ...
def tt_add(a: TTTensor, b: TTTensor) -> TTTensor: ...
def tt_dot(a: TTTensor, b: TTTensor) -> float: ...
```

### `hypertensor/integrators/symplectic.py`

```python
# Extract from: hypertensor_dynamics.py lines 102-138

class SymplecticIntegrator:
    def __init__(self, force_fn, mass=1.0, max_rank=12): ...
    def step(self, x, v, dt) -> Tuple[ndarray, ndarray]: ...
    def run(self, x0, v0, n_steps, dt) -> Dict: ...
```

### `hypertensor/pde/mhd.py`

```python
# Extract from: hypertensor_dynamics.py lines 229-316

class ResistiveMHD:
    def __init__(self, nx, L, eta): ...
    def step(self, rho, v, B, dt): ...
    def run(self, n_steps, dt) -> Dict: ...

class IdealMHD(ResistiveMHD):
    def __init__(self, nx, L):
        super().__init__(nx, L, eta=0)
```

---

## Version Roadmap

| Version | Features |
|---------|----------|
| 0.1.0 | Core TT, Symplectic, Langevin, MHD, Fokker-Planck |
| 0.2.0 | + Heat equation, Euler CFD, adaptive stepping |
| 0.3.0 | + Quantum (DMRG-lite), GPU backend |
| 0.4.0 | + Materials (superconductor, ionic) |
| 1.0.0 | Stable API, full test coverage |

---

## The Patent (Reiterated)

**Everyone else:** State grows → O(N^d) → RAM explosion → crash

**HyperTensor:** State grows → tt_round(state, rank=12) → O(N·d·r²) → bounded memory

```python
# The core loop that makes this work:
for t in time_steps:
    state = physics_step(state, dt)      # State may grow in rank
    state = tt_round(state, max_rank=12) # COMPRESS BACK DOWN
    # Memory never exceeds O(N·d·144)
```

This is why we can simulate 10¹⁰⁰ dimensional phase spaces on a laptop.

---

*Extraction Map — HyperTensor Physics Engine*
*January 5, 2026*
