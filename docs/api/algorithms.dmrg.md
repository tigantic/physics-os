# Module `algorithms.dmrg`

Density Matrix Renormalization Group (DMRG) ============================================

Variational ground state algorithm for Matrix Product States.

Theory
------
DMRG minimizes E = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩ by sweeping through the chain
and optimizing one (1-site DMRG) or two (2-site DMRG) tensors at a time.

The key insight is that the optimization at each site is a local
eigenvalue problem in the effective Hamiltonian H_eff.

2-Site Algorithm:
1. Sweep left → right:
   - Contract left environment L, two-site tensor Θ, right environment R
   - Diagonalize H_eff to get ground state Θ'
   - SVD: Θ' = U S V^†, truncate to χ
   - Update A[i] = U, A[i+1] = S V^†
   - Grow left environment

2. Sweep right → left (similarly)

Convergence when ΔE < tol.

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `DMRGResult`

Result container for DMRG.

#### Attributes

- **psi** (`<class 'tensornet.core.mps.MPS'>`): 
- **energy** (`<class 'float'>`): 
- **energies** (`typing.List[float]`): 
- **entropies** (`typing.List[float]`): 
- **truncation_errors** (`typing.List[float]`): 
- **converged** (`<class 'bool'>`): 
- **sweeps** (`<class 'int'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, psi: tensornet.core.mps.MPS, energy: float, energies: List[float], entropies: List[float], truncation_errors: List[float], converged: bool, sweeps: int) -> None
```

## Functions

### `dmrg`

```python
def dmrg(H: tensornet.core.mpo.MPO, chi_max: int, num_sweeps: int = 10, tol: float = 1e-10, psi0: Optional[tensornet.core.mps.MPS] = None, svd_cutoff: float = 1e-14, verbose: bool = False) -> dmrg.DMRGResult
```

Run 2-site DMRG to find the ground state.

**Parameters:**

- **H** (`<class 'tensornet.core.mpo.MPO'>`): Hamiltonian as MPO
- **chi_max** (`<class 'int'>`): Maximum bond dimension
- **num_sweeps** (`<class 'int'>`): Maximum number of sweeps
- **tol** (`<class 'float'>`): Energy convergence tolerance
- **psi0** (`typing.Optional[tensornet.core.mps.MPS]`): Initial MPS (random if None)
- **svd_cutoff** (`<class 'float'>`): SVD singular value cutoff
- **verbose** (`<class 'bool'>`): Print progress

**Returns**: `<class 'dmrg.DMRGResult'>` - DMRGResult with ground state MPS and diagnostics

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\algorithms\dmrg.py:470](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\algorithms\dmrg.py#L470)*

### `dmrg_sweep`

```python
def dmrg_sweep(psi: tensornet.core.mps.MPS, H: tensornet.core.mpo.MPO, chi_max: int, L_envs: List[torch.Tensor], R_envs: List[torch.Tensor], direction: str = 'right', svd_cutoff: float = 1e-14) -> Tuple[float, float, float]
```

Perform one DMRG sweep.

**Parameters:**

- **psi** (`<class 'tensornet.core.mps.MPS'>`): MPS to optimize (modified in-place)
- **H** (`<class 'tensornet.core.mpo.MPO'>`): Hamiltonian MPO
- **chi_max** (`<class 'int'>`): Maximum bond dimension
- **L_envs** (`typing.List[torch.Tensor]`): Left environments (modified in-place)
- **R_envs** (`typing.List[torch.Tensor]`): Right environments
- **direction** (`<class 'str'>`): 'right' or 'left'
- **svd_cutoff** (`<class 'float'>`): SVD singular value cutoff

**Returns**: `typing.Tuple[float, float, float]` - (energy, max_entropy, max_truncation_error)

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\algorithms\dmrg.py:347](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\algorithms\dmrg.py#L347)*
