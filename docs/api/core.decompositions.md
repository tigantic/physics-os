# Module `core.decompositions`

Tensor Decompositions =====================

SVD and QR decompositions with truncation for tensor networks.

Constitutional Compliance:
    - Article V.5.1: Condition number warnings when κ > 10¹⁰
    - Article V.5.2: SVD truncation with return_info
    - Article VIII.8.2: Memory profiling decorator

**Contents:**

- [Functions](#functions)

## Functions

### `polar_decomposition`

```python
def polar_decomposition(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```

Polar decomposition A = U @ P where U is unitary and P is positive semidefinite.

**Parameters:**

- **A** (`<class 'torch.Tensor'>`): Input matrix (m, n) with m >= n

**Returns**: `typing.Tuple[torch.Tensor, torch.Tensor]` - Unitary matrix (m, n)
    P: Positive semidefinite (n, n)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\decompositions.py:167](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\decompositions.py#L167)*

### `qr_positive`

```python
def qr_positive(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```

QR decomposition with positive diagonal R.

Standard QR can have arbitrary signs on the diagonal of R.
This function ensures R has positive diagonal, which is
important for canonical MPS forms.

**Parameters:**

- **A** (`<class 'torch.Tensor'>`): Input matrix of shape (m, n)

**Returns**: `typing.Tuple[torch.Tensor, torch.Tensor]` - Orthogonal matrix (m, min(m,n))
    R: Upper triangular with positive diagonal (min(m,n), n)

**Examples:**

```python
A = torch.randn(50, 30, dtype=torch.float64)
Q, R = qr_positive(A)
assert torch.allclose(A, Q @ R)
assert (torch.diag(R) >= 0).all()
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\decompositions.py:118](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\decompositions.py#L118)*

### `svd_truncated`

```python
def svd_truncated(A: torch.Tensor, chi_max: Optional[int] = None, cutoff: float = 1e-14, return_info: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]]
```

Truncated SVD with bond dimension control.

Computes A ≈ U @ diag(S) @ Vh where rank is limited by chi_max
and singular values below cutoff are discarded.

This is the optimal rank-k approximation by the Eckart-Young-Mirsky theorem.

**Parameters:**

- **A** (`<class 'torch.Tensor'>`): Input matrix of shape (m, n)
- **chi_max** (`typing.Optional[int]`): Maximum number of singular values to keep
- **cutoff** (`<class 'float'>`): Discard singular values below this threshold
- **return_info** (`<class 'bool'>`): If True, return dictionary with truncation info

**Returns**: `typing.Union[typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor], typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]]` - Left singular vectors (m, k)
    S: Singular values (k,)
    Vh: Right singular vectors (k, n)
    info: (optional) Dictionary with truncation_error, rank, etc.

**Examples:**

```python
A = torch.randn(100, 100, dtype=torch.float64)
U, S, Vh = svd_truncated(A, chi_max=20)
A_approx = U @ torch.diag(S) @ Vh
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\decompositions.py:28](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\decompositions.py#L28)*

### `thin_svd`

```python
def thin_svd(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

Thin (economy) SVD without truncation.

**Parameters:**

- **A** (`<class 'torch.Tensor'>`): Input matrix (m, n)

**Returns**: `typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]` - (m, k) where k = min(m, n)
    S: (k,)
    Vh: (k, n)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\decompositions.py:152](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\decompositions.py#L152)*
