# Tensor Network Engine

The core engine implements three fundamental data structures and their
associated algorithms.

## Data Structures

### Matrix Product States (MPS)

An MPS represents a quantum/classical state as a chain of rank-3 tensors:

$$|\psi\rangle = \sum_{s_1 \ldots s_L} A^{s_1} A^{s_2} \cdots A^{s_L} |s_1 s_2 \ldots s_L\rangle$$

::: ontic.core.mps.MPS
    options:
      show_source: false
      members: false

### Matrix Product Operators (MPO)

::: tensornet.mpo
    options:
      show_source: false
      members: false

### Quantized Tensor Train (QTT)

::: ontic.qtt
    options:
      show_source: false
      members: false

## Algorithms

| Algorithm | Module | Purpose |
|-----------|--------|---------|
| DMRG | `ontic.algorithms.dmrg` | Variational ground state |
| TEBD | `ontic.algorithms.tebd` | Real/imaginary time evolution |
| TDVP | `ontic.algorithms.tdvp` | Time-dependent variational |
| Lanczos | `ontic.algorithms.lanczos` | Eigenvalue computation |
