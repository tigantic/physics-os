# Quick Start

## Your First DMRG Calculation

```python
from tensornet import MPS, heisenberg_mpo, dmrg

# Build a Heisenberg XXZ Hamiltonian for a 20-site chain
H = heisenberg_mpo(L=20, J=1.0, Jz=1.0)

# Random initial MPS with bond dimension 64
psi = MPS.random(L=20, d=2, chi=64)

# Run DMRG
psi, E, info = dmrg(psi, H, num_sweeps=20, chi_max=128)
print(f"Ground state energy: {E:.10f}")
print(f"Final bond dimension: {max(psi.bond_dims)}")
```

## CFD: Sod Shock Tube

```python
from tensornet.cfd import Euler1D, sod_shock_tube_ic, euler_to_mps

# Initialize Sod shock tube problem
state = sod_shock_tube_ic(nx=256)
mps = euler_to_mps(state, chi_max=64)

# Create solver and evolve
solver = Euler1D(gamma=1.4)
for step in range(100):
    mps = solver.step(mps, dt=1e-3)
```

## Time Evolution: TEBD

```python
from tensornet import MPS, tebd
from tensornet.algorithms.tebd import transverse_ising_gates

L, d, chi = 20, 2, 32
psi = MPS.random(L=L, d=d, chi=chi)
gates = transverse_ising_gates(L=L, J=1.0, h=0.5, dt=0.01)
psi = tebd(psi, gates, steps=100, chi_max=64)
```
