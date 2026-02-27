# Tensor Network Engine

The core engine implements three fundamental data structures and their
associated algorithms.

## Data Structures

### Matrix Product States (MPS)

An MPS represents a quantum/classical state as a chain of rank-3 tensors:

$$|\psi\rangle = \sum_{s_1 \ldots s_L} A^{s_1} A^{s_2} \cdots A^{s_L} |s_1 s_2 \ldots s_L\rangle$$

::: tensornet.core.mps.MPS
    options:
      show_source: false
      members: false

### Matrix Product Operators (MPO)

::: tensornet.mpo
    options:
      show_source: false
      members: false

### Quantized Tensor Train (QTT)

::: tensornet.qtt
    options:
      show_source: false
      members: false

## Algorithms

| Algorithm | Module | Purpose |
|-----------|--------|---------|
| DMRG | `tensornet.algorithms.dmrg` | Variational ground state |
| TEBD | `tensornet.algorithms.tebd` | Real/imaginary time evolution |
| TDVP | `tensornet.algorithms.tdvp` | Time-dependent variational |
| Lanczos | `tensornet.algorithms.lanczos` | Eigenvalue computation |
