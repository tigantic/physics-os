# HyperTensor Physics Engine

**Universal physics solver with TT-compressed state evolution.**

> *The Patent: Everyone else runs out of RAM. We compress the universe every millisecond.*

## Installation

```bash
pip install hypertensor-physics
```

Or from source:
```bash
git clone https://github.com/hypertensor/hypertensor-physics
cd hypertensor-physics
pip install -e .
```

## Quick Start

```python
from hypertensor import TTTensor, tt_round, tt_to_full
from hypertensor.integrators import LangevinDynamics
from hypertensor.pde import ResistiveMHD, FokkerPlanck
import numpy as np

# 1. Tensor-Train Compression
tensor = np.random.randn(10, 10, 10, 10)  # 10,000 elements
tt = tt_round(tensor, max_rank=5)
print(f"Compression: {tt.compression_ratio:.1f}×")

# 2. Langevin Dynamics (Drug Binding Test)
def binding_potential(x):
    return (x[0]**2 - 1)**2 + 0.5 * np.sum(x[1:]**2)

langevin = LangevinDynamics(
    potential_fn=binding_potential,
    temperature=310,  # Body temperature (K)
    friction=10.0
)
result = langevin.run(np.array([1.0, 0.0, 0.0]), n_steps=1000, dt=1e-14)
print(f"Binding stable: {result['stable']}")

# 3. Resistive MHD (Plasma Physics)
mhd = ResistiveMHD(nx=64, L=1.0, eta=0.01)
result = mhd.run(n_steps=100, dt=1e-5)
print(f"Reconnection rate: {result['reconnection_rate']:.4f}")

# 4. Fokker-Planck (Probability Evolution)
fp = FokkerPlanck(nx=128, x_range=(-5, 5), diffusion=0.5)
P0 = fp.initialize_gaussian(mean=2.0, std=0.5)
result = fp.run(P0, n_steps=500, dt=0.01)
print(f"Final mean: {result['mean']:.3f}")
```

## Core Concept: TT Re-Compression

The key innovation is re-compressing state after every physics step:

```python
for t in time_steps:
    state = physics_update(state, dt)    # State may grow in rank
    state = tt_round(state, max_rank=12) # COMPRESS BACK DOWN
    # Memory never exceeds O(N·d·r²)
```

This enables simulation of high-dimensional systems that would otherwise explode in memory.

## Modules

### `hypertensor.core`
- `TTTensor` - Tensor-Train data structure
- `tt_round` - Compress to TT format
- `tt_to_full` - Reconstruct full tensor
- `tt_add`, `tt_dot` - TT arithmetic
- Physical constants (`k_B`, `e`, `c`, `hbar`, etc.)

### `hypertensor.integrators`
- `SymplecticIntegrator` - Velocity Verlet (energy-conserving)
- `LeapfrogIntegrator` - Störmer-Verlet
- `LangevinDynamics` - BAOAB integrator for finite-temperature MD

### `hypertensor.pde`
- `ResistiveMHD` - Plasma magnetohydrodynamics
- `IdealMHD` - Zero-resistivity limit
- `FokkerPlanck` - Probability distribution evolution
- `HeatEquation1D` - Thermal diffusion
- `CompositeWall` - Multi-layer thermal analysis

## Applications

| Domain | Solver | Example |
|--------|--------|---------|
| Drug Design | `LangevinDynamics` | Binding stability at 310K |
| Fusion | `ResistiveMHD` | Tokamak plasma control |
| Finance | `FokkerPlanck` | Option pricing, risk modeling |
| Materials | `CompositeWall` | Thermal protection design |

## Memory Scaling

| Method | Memory | With TT (rank 12) |
|--------|--------|-------------------|
| Full tensor N^d | O(N^d) | O(N·d·144) |
| 10^6 elements | 8 MB | ~100 KB |
| 10^12 elements | 8 TB | ~1 MB |

## License

MIT

## Citation

```bibtex
@software{hypertensor_physics,
  title = {HyperTensor Physics Engine},
  year = {2026},
  url = {https://github.com/hypertensor/hypertensor-physics}
}
```
