# Tutorial: Ground State Physics with Tensor Networks

This tutorial demonstrates how to use The Physics OS to compute ground states of quantum spin chains using the Density Matrix Renormalization Group (DMRG) algorithm.

## Overview

Tensor networks provide an efficient representation of quantum many-body states. The Matrix Product State (MPS) ansatz is particularly effective for 1D systems, where the entanglement follows an area law.

## Installation

```bash
pip install ontic-engine
```

## Quick Start: Heisenberg Spin Chain

The Heisenberg XXZ model describes interacting spin-1/2 particles:

$$H = J \sum_i \left( S^x_i S^x_{i+1} + S^y_i S^y_{i+1} + \Delta S^z_i S^z_{i+1} \right)$$

### Finding the Ground State

```python
import torch
from ontic import dmrg, heisenberg_mpo

# System parameters
L = 20          # Chain length
chi_max = 64    # Bond dimension
J = 1.0         # Exchange coupling
Delta = 1.0     # Anisotropy (Delta=1 is isotropic)

# Create Hamiltonian as Matrix Product Operator
H = heisenberg_mpo(L, J=J, Jz=Delta)

# Run DMRG to find ground state
result = dmrg(H, chi_max=chi_max, num_sweeps=10, verbose=True)

print(f"Ground state energy: {result.energy:.8f}")
print(f"Energy per site: {result.energy / L:.8f}")
print(f"Converged: {result.converged}")
```

### Expected Results

For the isotropic Heisenberg chain (Delta=1), the exact ground state energy per site in the thermodynamic limit is:

$$e_0 = \frac{1}{4} - \ln(2) \approx -0.4431$$

With L=20 and chi=64, DMRG typically achieves accuracy within 0.1% of this value.

## Understanding the Output

The `DMRGResult` object contains:

- `energy`: Ground state energy
- `psi`: The optimized MPS representing |psi_0>
- `energies`: Energy at each sweep
- `entropies`: Entanglement entropy at each bond
- `converged`: Whether the algorithm converged

### Entanglement Entropy

The entanglement entropy reveals the quantum correlations:

```python
import matplotlib.pyplot as plt

# Plot entanglement entropy across the chain
plt.figure(figsize=(8, 4))
plt.plot(range(1, L), result.entropies[-1], 'bo-')
plt.xlabel('Bond position')
plt.ylabel('Entanglement entropy')
plt.title('Entanglement Profile')
plt.show()
```

For the Heisenberg chain, the entropy is maximal in the center and decreases toward the edges (for open boundary conditions).

## Advanced: Transverse Field Ising Model

The TFIM exhibits a quantum phase transition:

$$H = -J \sum_i S^z_i S^z_{i+1} - h \sum_i S^x_i$$

At the critical point h/J = 1, the system transitions from ferromagnetic (h < J) to paramagnetic (h > J).

```python
from ontic import tfim_mpo

# Critical point
h_c = 1.0
H_critical = tfim_mpo(L=20, J=1.0, h=h_c)

result = dmrg(H_critical, chi_max=64, num_sweeps=10)
print(f"Critical ground state energy: {result.energy:.6f}")
```

## Time Evolution with TEBD

For dynamics, use the Time-Evolving Block Decimation algorithm:

```python
from ontic.algorithms.tebd import tebd_step
from ontic.core import MPS

# Create initial state (e.g., Neel state)
psi = MPS.product_state(L, [0, 1] * (L // 2))

# Evolution parameters
dt = 0.01
chi_max = 64

# Create time evolution gates
# (see documentation for gate construction)

# Evolve in time
for step in range(100):
    tebd_step(psi, gates_odd, gates_even, chi_max=chi_max)
```

## Performance Tips

1. **Bond dimension**: Start with chi=32, increase until energy converges
2. **Sweeps**: 5-10 sweeps usually sufficient for simple models
3. **GPU acceleration**: Use `device='cuda'` for large systems
4. **Precision**: Use `dtype=torch.float64` for better accuracy

## References

1. White, S.R. "Density matrix formulation for quantum renormalization groups" (1992)
2. Schollwoeck, U. "The density-matrix renormalization group in the age of matrix product states" (2011)
3. Orus, R. "A practical introduction to tensor networks" (2014)

## Next Steps

- Explore the [API Reference](../api/algorithms.dmrg.md) for detailed function documentation
- See [examples/](../../examples/) for more use cases
- Check [benchmarks/](../../Physics/benchmarks/) for performance comparisons with TeNPy
