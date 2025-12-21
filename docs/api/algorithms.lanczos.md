# Module `algorithms.lanczos`

Lanczos Algorithm =================

Krylov subspace methods for eigenvalue problems and matrix exponentials.

Theory
------
The Lanczos algorithm builds an orthonormal basis for the Krylov subspace:
    K_m(A, v) = span{v, Av, A²v, ..., A^{m-1}v}

In this basis, A is represented by a tridiagonal matrix T_m, which can
be efficiently diagonalized to approximate eigenvalues and eigenvectors.

Key property: Extremal eigenvalues converge fastest (Kaniel-Paige theory).

For matrix exponential: exp(A)v ≈ V_m exp(T_m) e_1 ||v||

Degenerate Eigenvalues (Article V.5.3)
--------------------------------------
When the ground state is degenerate (multiple eigenvalues with same value),
the Lanczos algorithm will converge to ONE eigenvector in the degenerate
subspace. The specific eigenvector depends on the initial vector v0.

Behavior with degenerate spectra:
- Convergence rate may be slower if gap to first excited state is small
- The returned eigenvector is a valid ground state but not unique
- For computing full degenerate subspace, use block Lanczos or run
  multiple times with orthogonalized initial vectors
- The residual ||Av - λv|| remains a valid convergence criterion

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `LanczosResult`

Result container for Lanczos algorithm.

#### Attributes

- **eigenvalue** (`<class 'float'>`): 
- **eigenvector** (`<class 'torch.Tensor'>`): 
- **converged** (`<class 'bool'>`): 
- **iterations** (`<class 'int'>`): 
- **residual** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, eigenvalue: float, eigenvector: torch.Tensor, converged: bool, iterations: int, residual: float) -> None
```

## Functions

### `lanczos_eigenvalues`

```python
def lanczos_eigenvalues(matvec: Callable[[torch.Tensor], torch.Tensor], v0: torch.Tensor, num_eigenvalues: int = 1, num_iter: int = 100, tol: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]
```

Find multiple lowest eigenvalues using Lanczos.

**Parameters:**

- **matvec** (`typing.Callable[[torch.Tensor], torch.Tensor]`): Matrix-vector product function
- **v0** (`<class 'torch.Tensor'>`): Initial vector
- **num_eigenvalues** (`<class 'int'>`): Number of eigenvalues to find
- **num_iter** (`<class 'int'>`): Maximum iterations
- **tol** (`<class 'float'>`): Convergence tolerance

**Returns**: `typing.Tuple[torch.Tensor, torch.Tensor]` - (eigenvalues, eigenvectors) - sorted by eigenvalue

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\algorithms\lanczos.py:284](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\algorithms\lanczos.py#L284)*

### `lanczos_expm`

```python
def lanczos_expm(matvec: Callable[[torch.Tensor], torch.Tensor], v: torch.Tensor, t: complex, num_iter: int = 30, tol: float = 1e-12) -> torch.Tensor
```

Compute exp(t*A) @ v using Lanczos.

Uses the Krylov approximation:
    exp(tA)v ≈ ||v|| * V_m * exp(t*T_m) * e_1

where T_m is the tridiagonal representation in the Krylov basis.

**Parameters:**

- **matvec** (`typing.Callable[[torch.Tensor], torch.Tensor]`): Function that computes A @ v
- **v** (`<class 'torch.Tensor'>`): Vector to apply exponential to
- **t** (`<class 'complex'>`): Time parameter (can be complex for e^{-iHt})
- **num_iter** (`<class 'int'>`): Number of Lanczos iterations
- **tol** (`<class 'float'>`): Tolerance for convergence

**Returns**: `<class 'torch.Tensor'>` - exp(t*A) @ v

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\algorithms\lanczos.py:189](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\algorithms\lanczos.py#L189)*

### `lanczos_ground_state`

```python
def lanczos_ground_state(matvec: Callable[[torch.Tensor], torch.Tensor], v0: torch.Tensor, num_iter: int = 100, tol: float = 1e-12, reorthogonalize: bool = True) -> lanczos.LanczosResult
```

Find the ground state (lowest eigenvalue) using Lanczos.

**Parameters:**

- **matvec** (`typing.Callable[[torch.Tensor], torch.Tensor]`): Function that computes A @ v
- **v0** (`<class 'torch.Tensor'>`): Initial vector
- **num_iter** (`<class 'int'>`): Maximum Lanczos iterations
- **tol** (`<class 'float'>`): Convergence tolerance for eigenvalue
- **reorthogonalize** (`<class 'bool'>`): Full reorthogonalization (slower but more stable)

**Returns**: `<class 'lanczos.LanczosResult'>` - LanczosResult with ground state eigenvalue and eigenvector

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\algorithms\lanczos.py:58](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\algorithms\lanczos.py#L58)*
