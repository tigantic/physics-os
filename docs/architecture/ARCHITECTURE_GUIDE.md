# The Physics OS — Architecture Guide

This document provides detailed architecture documentation for The Physics OS, including dependency diagrams and data flow for key operations.

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Module Dependencies](#module-dependencies)
3. [Core Data Flow](#core-data-flow)
4. [Algorithm Workflows](#algorithm-workflows)
5. [Extension Points](#extension-points)

---

## High-Level Architecture

```mermaid
graph TB
    subgraph "User Layer"
        CLI[CLI Tools]
        API[REST API]
        SDK[Python SDK]
        NB[Jupyter Notebooks]
    end
    
    subgraph "Application Layer"
        DEMOS[demos/]
        PROOFS[proofs/]
        BENCH[benchmarks/]
    end
    
    subgraph "Library Layer - ontic"
        CORE[core/]
        ALGO[algorithms/]
        CFD[cfd/]
        MPS[mps/]
        ML[ml_surrogates/]
        AUTO[autonomy/]
    end
    
    subgraph "Extension Layer"
        RUST[tci_core_rust]
        SERVER[sdk/server]
        QTT_SDK[sdk/qtt-sdk]
    end
    
    subgraph "External Dependencies"
        TORCH[PyTorch]
        NUMPY[NumPy]
        SCIPY[SciPy]
    end
    
    CLI --> SDK
    API --> SERVER
    SDK --> CORE
    NB --> SDK
    
    DEMOS --> CORE
    DEMOS --> CFD
    PROOFS --> CORE
    BENCH --> CORE
    
    CORE --> TORCH
    CORE --> NUMPY
    ALGO --> CORE
    CFD --> ALGO
    MPS --> CORE
    ML --> CORE
    AUTO --> ALGO
    
    ALGO --> RUST
    CFD --> RUST
    
    SERVER --> CORE
    QTT_SDK --> CORE
```

---

## Module Dependencies

### ontic/ Internal Dependencies

```mermaid
graph LR
    subgraph "ontic/"
        core[core/]
        algorithms[algorithms/]
        cfd[cfd/]
        mps[mps/]
        ml[ml_surrogates/]
        autonomy[autonomy/]
        operators[operators/]
    end
    
    algorithms --> core
    cfd --> algorithms
    cfd --> core
    mps --> core
    ml --> core
    autonomy --> algorithms
    autonomy --> ml
    operators --> core
```

### Core Submodule Structure

```mermaid
graph TD
    subgraph "ontic/core/"
        mps_mod[mps.py]
        mpo[mpo.py]
        qtt[qtt.py]
        tci[tci.py]
        decomp[decomposition.py]
        contract[contraction.py]
        utils[utils.py]
    end
    
    mps_mod --> decomp
    mps_mod --> contract
    mpo --> contract
    qtt --> decomp
    tci --> qtt
    tci --> decomp
    contract --> utils
```

### External Dependency Map

| Module | PyTorch | NumPy | SciPy | Rust |
|--------|---------|-------|-------|------|
| ontic/core/ | ✅ | ✅ | ✅ | ⚪ |
| ontic/algorithms/ | ✅ | ✅ | ✅ | ✅ |
| ontic/cfd/ | ✅ | ✅ | ✅ | ✅ |
| ontic/mps/ | ✅ | ✅ | ⚪ | ⚪ |
| ontic/ml_surrogates/ | ✅ | ✅ | ⚪ | ⚪ |
| sdk/server/ | ⚪ | ✅ | ⚪ | ⚪ |

Legend: ✅ Required | ⚪ Optional

---

## Core Data Flow

### MPS Creation and Manipulation

```mermaid
sequenceDiagram
    participant User
    participant MPS
    participant Decomp as decomposition
    participant Torch as PyTorch
    
    User->>MPS: MPS.random(L=10, d=2, chi=32)
    MPS->>Torch: Create random tensors
    Torch-->>MPS: tensors list
    MPS-->>User: MPS object
    
    User->>MPS: mps.canonicalize("right")
    MPS->>Decomp: QR decomposition
    Decomp->>Torch: torch.linalg.qr()
    Torch-->>Decomp: Q, R matrices
    Decomp-->>MPS: Canonical form
    MPS-->>User: MPS (right-canonical)
```

### DMRG Algorithm Flow

```mermaid
sequenceDiagram
    participant User
    participant DMRG
    participant MPS
    participant MPO
    participant Lanczos
    
    User->>DMRG: dmrg(psi, H, num_sweeps=10)
    
    loop For each sweep
        loop For each bond
            DMRG->>MPS: Get two-site tensor
            DMRG->>MPO: Build effective Hamiltonian
            DMRG->>Lanczos: Find ground state
            Lanczos-->>DMRG: Optimal tensor
            DMRG->>MPS: SVD truncate and update
        end
        DMRG->>DMRG: Check convergence
    end
    
    DMRG-->>User: (psi, energy, info)
```

### QTT Compression Flow

```mermaid
sequenceDiagram
    participant User
    participant QTT
    participant TCI
    participant Rust as tci_core
    
    User->>QTT: compress_qtt(data, max_rank=32)
    QTT->>TCI: Initialize TCI sampler
    TCI->>Rust: Create TCISampler
    
    loop Adaptive refinement
        Rust->>TCI: MaxVol pivot selection
        TCI->>QTT: Sample function at pivots
        QTT->>TCI: Function values
        TCI->>Rust: Update skeleton
        Rust->>TCI: Error estimate
    end
    
    TCI-->>QTT: TT-cores
    QTT-->>User: CompressedQTT
```

---

## Algorithm Workflows

### CFD Solver Pipeline

```mermaid
flowchart TB
    subgraph "Initialization"
        IC[Initial Conditions]
        MESH[Mesh Generation]
        QTT_INIT[QTT Compression]
    end
    
    subgraph "Time Stepping"
        FLUX[Compute Fluxes]
        TCI_ADAPT[TCI Adaptation]
        RECOMP[Recompression]
        UPDATE[Update State]
    end
    
    subgraph "Output"
        DECOMP[Decompression]
        VIZ[Visualization]
        EXPORT[Export Results]
    end
    
    IC --> MESH
    MESH --> QTT_INIT
    QTT_INIT --> FLUX
    
    FLUX --> TCI_ADAPT
    TCI_ADAPT --> RECOMP
    RECOMP --> UPDATE
    UPDATE --> FLUX
    
    UPDATE -->|Final time| DECOMP
    DECOMP --> VIZ
    DECOMP --> EXPORT
```

### ML Surrogate Training

```mermaid
flowchart LR
    subgraph "Data Preparation"
        PHYS[Physics Data]
        SAMPLE[Sampling]
        SPLIT[Train/Val Split]
    end
    
    subgraph "Training Loop"
        FWD[Forward Pass]
        LOSS[Physics Loss]
        BACK[Backward Pass]
        OPT[Optimizer Step]
    end
    
    subgraph "Evaluation"
        VAL[Validation]
        EXPORT[Export Model]
    end
    
    PHYS --> SAMPLE
    SAMPLE --> SPLIT
    SPLIT --> FWD
    
    FWD --> LOSS
    LOSS --> BACK
    BACK --> OPT
    OPT --> FWD
    
    OPT -->|Converged| VAL
    VAL --> EXPORT
```

---

## Extension Points

### Adding New Hamiltonians

```python
# In ontic/mps/hamiltonians.py
from ontic.core import MPO

def my_hamiltonian_mpo(L: int, **params) -> MPO:
    """Create MPO for my Hamiltonian."""
    # 1. Define local operators
    # 2. Build MPO tensors
    # 3. Return MPO
    ...
```

### Adding New CFD Flux Schemes

```python
# In ontic/cfd/fluxes.py
from ontic.cfd.base import FluxScheme

class MyFlux(FluxScheme):
    """Custom flux scheme."""
    
    def compute(self, UL, UR, gamma):
        # Implement flux computation
        ...
```

### Adding Rust Extensions

```rust
// In crates/tci_core_rust/src/lib.rs
#[pyclass]
pub struct MyExtension {
    // ...
}

#[pymethods]
impl MyExtension {
    #[new]
    fn new() -> Self {
        // ...
    }
}
```

---

## See Also

- [ONBOARDING.md](ONBOARDING.md) - Getting started guide
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines
- [README.md](../README.md) - Project overview
