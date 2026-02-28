# Module `algorithms.tebd`

Time-Evolving Block Decimation (TEBD) ======================================

Real and imaginary time evolution for Matrix Product States.

Theory
------
TEBD applies the Suzuki-Trotter decomposition to approximate e^{-iHt}
(real time) or e^{-βH} (imaginary time) for nearest-neighbor Hamiltonians.

H = Σᵢ hᵢ,ᵢ₊₁

First-order Trotter: e^{-iHdt} ≈ Πᵢ e^{-ihᵢ,ᵢ₊₁ dt}
Second-order Trotter: 
    e^{-iHdt} ≈ Π_odd e^{-ihdt/2} · Π_even e^{-ihdt} · Π_odd e^{-ihdt/2}

Each e^{-ih_{i,i+1}dt} is a two-site gate applied via SVD truncation.

Error Scaling:
- First-order: O(dt²)
- Second-order: O(dt³)
- Fourth-order: O(dt⁵)

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `TEBDResult`

Result container for TEBD time evolution.

#### Attributes

- **psi** (`<class 'ontic.core.mps.MPS'>`): 
- **times** (`typing.List[float]`): 
- **energies** (`typing.Optional[typing.List[float]]`): 
- **entropies** (`typing.List[typing.List[float]]`): 
- **truncation_errors** (`typing.List[float]`): 

#### Methods

##### `__init__`

```python
def __init__(self, psi: ontic.core.mps.MPS, times: List[float], energies: Optional[List[float]], entropies: List[List[float]], truncation_errors: List[float]) -> None
```

## Functions

### `build_gates_from_mpo`

```python
def build_gates_from_mpo(H: 'MPO', dt: complex) -> List[torch.Tensor]
```

Build TEBD gates from an MPO Hamiltonian.

Extracts nearest-neighbor terms from the MPO.

**Parameters:**

- **H**: Hamiltonian MPO
- **dt**: Time step

**Returns**: value - List of two-site gates

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\algorithms\tebd.py:132](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\algorithms\tebd.py#L132)*

### `build_heisenberg_gates`

```python
def build_heisenberg_gates(L: int, dt: complex, Jx: float = 1.0, Jy: float = 1.0, Jz: float = 1.0, h: float = 0.0, order: int = 2, dtype: torch.dtype = torch.complex128, device: Optional[torch.device] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]
```

Build TEBD gates for Heisenberg XXZ model.

H = Σᵢ [Jx SˣᵢSˣᵢ₊₁ + Jy SʸᵢSʸᵢ₊₁ + Jz SᶻᵢSᶻᵢ₊₁] - h Σᵢ Sᶻᵢ

**Parameters:**

- **L** (`<class 'int'>`): Number of sites
- **dt** (`<class 'complex'>`): Time step Jx, Jy, Jz: Exchange couplings
- **h** (`<class 'float'>`): Magnetic field
- **order** (`<class 'int'>`): Trotter order
- **dtype** (`<class 'torch.dtype'>`): Data type
- **device** (`typing.Optional[torch.device]`): Device

**Returns**: `typing.Tuple[typing.List[torch.Tensor], typing.List[torch.Tensor]]` - (gates_odd, gates_even) for TEBD

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\algorithms\tebd.py:236](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\algorithms\tebd.py#L236)*

### `build_tfim_gates`

```python
def build_tfim_gates(L: int, dt: complex, J: float = 1.0, g: float = 1.0, order: int = 2, dtype: torch.dtype = torch.complex128, device: Optional[torch.device] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]
```

Build TEBD gates for Transverse Field Ising Model.

H = -J Σᵢ SᶻᵢSᶻᵢ₊₁ - g Σᵢ Sˣᵢ

**Parameters:**

- **L** (`<class 'int'>`): Number of sites
- **dt** (`<class 'complex'>`): Time step
- **J** (`<class 'float'>`): Ising coupling
- **g** (`<class 'float'>`): Transverse field
- **order** (`<class 'int'>`): Trotter order
- **dtype** (`<class 'torch.dtype'>`): Data type
- **device** (`typing.Optional[torch.device]`): Device

**Returns**: `typing.Tuple[typing.List[torch.Tensor], typing.List[torch.Tensor]]` - (gates_odd, gates_even) for TEBD

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\algorithms\tebd.py:321](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\algorithms\tebd.py#L321)*

### `imaginary_time_evolution`

```python
def imaginary_time_evolution(psi: ontic.core.mps.MPS, gates_odd: List[torch.Tensor], gates_even: List[torch.Tensor], num_steps: int, chi_max: int, order: int = 2, cutoff: float = 1e-14, normalize_every: int = 1, verbose: bool = False) -> ontic.core.mps.MPS
```

Imaginary time evolution for ground state preparation.

|ψ(β)⟩ = e^{-βH} |ψ(0)⟩ / ||...||

As β → ∞, |ψ⟩ → ground state (if overlap is non-zero).

**Parameters:**

- **psi** (`<class 'ontic.core.mps.MPS'>`): Initial MPS (modified in-place) gates_odd, gates_even: Imaginary time gates
- **num_steps** (`<class 'int'>`): Number of steps
- **chi_max** (`<class 'int'>`): Maximum bond dimension
- **order** (`<class 'int'>`): Trotter order
- **cutoff** (`<class 'float'>`): SVD cutoff
- **normalize_every** (`<class 'int'>`): Normalize every N steps
- **verbose** (`<class 'bool'>`): Print progress

**Returns**: `<class 'ontic.core.mps.MPS'>` - Ground state MPS

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\algorithms\tebd.py:461](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\algorithms\tebd.py#L461)*

### `tebd`

```python
def tebd(psi: ontic.core.mps.MPS, gates_odd: List[torch.Tensor], gates_even: List[torch.Tensor], num_steps: int, dt: float, chi_max: int, order: int = 2, cutoff: float = 1e-14, compute_energy: Optional[Callable[[ontic.core.mps.MPS], float]] = None, compute_every: int = 1, normalize_every: int = 1, verbose: bool = False) -> tebd.TEBDResult
```

Run TEBD time evolution.

**Parameters:**

- **psi** (`<class 'ontic.core.mps.MPS'>`): Initial MPS (modified in-place)
- **gates_odd** (`typing.List[torch.Tensor]`): Gates for odd bonds
- **gates_even** (`typing.List[torch.Tensor]`): Gates for even bonds
- **num_steps** (`<class 'int'>`): Number of time steps
- **dt** (`<class 'float'>`): Time step (for logging, gates should already encode dt)
- **chi_max** (`<class 'int'>`): Maximum bond dimension
- **order** (`<class 'int'>`): Trotter order
- **cutoff** (`<class 'float'>`): SVD cutoff
- **compute_energy** (`typing.Optional[typing.Callable[[ontic.core.mps.MPS], float]]`): Optional function to compute energy
- **compute_every** (`<class 'int'>`): Compute observables every N steps
- **normalize_every** (`<class 'int'>`): Normalize every N steps
- **verbose** (`<class 'bool'>`): Print progress

**Returns**: `<class 'tebd.TEBDResult'>` - TEBDResult with time evolution data

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\algorithms\tebd.py:385](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\algorithms\tebd.py#L385)*

### `tebd_step`

```python
def tebd_step(psi: ontic.core.mps.MPS, gates_odd: List[torch.Tensor], gates_even: List[torch.Tensor], chi_max: int, cutoff: float = 1e-14, order: int = 2) -> float
```

Perform one TEBD step (second-order Trotter).

For second-order:
    e^{-iHdt} ≈ U_odd(dt/2) · U_even(dt) · U_odd(dt/2)

**Parameters:**

- **psi** (`<class 'ontic.core.mps.MPS'>`): MPS (modified in-place)
- **gates_odd** (`typing.List[torch.Tensor]`): Gates for odd bonds (0-1, 2-3, ...)
- **gates_even** (`typing.List[torch.Tensor]`): Gates for even bonds (1-2, 3-4, ...)
- **chi_max** (`<class 'int'>`): Maximum bond dimension
- **cutoff** (`<class 'float'>`): SVD cutoff
- **order** (`<class 'int'>`): Trotter order (1 or 2)

**Returns**: `<class 'float'>` - Maximum truncation error in this step

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\algorithms\tebd.py:160](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\algorithms\tebd.py#L160)*
