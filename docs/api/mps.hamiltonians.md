# Module `mps.hamiltonians`

Standard Hamiltonians as MPOs.

Provides analytical MPO constructions for common models.

**Contents:**

- [Functions](#functions)

## Functions

### `bose_hubbard_mpo`

```python
def bose_hubbard_mpo(L: int, n_max: int = 3, t: float = 1.0, U: float = 1.0, mu: float = 0.0, dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None) -> tensornet.core.mpo.MPO
```

Bose-Hubbard model as MPO.

H = -t * sum_i (b_i^dag b_{i+1} + h.c.) + (U/2) * sum_i n_i(n_i-1) - mu * sum_i n_i

**Parameters:**

- **L** (`<class 'int'>`): Number of sites
- **n_max** (`<class 'int'>`): Maximum occupation per site (Fock space truncation)
- **t** (`<class 'float'>`): Hopping strength
- **U** (`<class 'float'>`): On-site interaction
- **mu** (`<class 'float'>`): Chemical potential
- **dtype** (`<class 'torch.dtype'>`): Data type
- **device** (`typing.Optional[torch.device]`): Device

**Returns**: `<class 'tensornet.core.mpo.MPO'>` - MPO representation

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\mps\hamiltonians.py:336](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\mps\hamiltonians.py#L336)*

### `heisenberg_mpo`

```python
def heisenberg_mpo(L: int, J: float = 1.0, Jz: Optional[float] = None, h: float = 0.0, dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None) -> tensornet.core.mpo.MPO
```

Heisenberg XXZ chain Hamiltonian as MPO.

H = J * sum_i (S^x_i S^x_{i+1} + S^y_i S^y_{i+1}) + Jz * sum_i S^z_i S^z_{i+1} + h * sum_i S^z_i

For XXX model, set Jz = J (or leave as None, which defaults to J).

**Parameters:**

- **L** (`<class 'int'>`): Number of sites
- **J** (`<class 'float'>`): XY coupling strength
- **Jz** (`typing.Optional[float]`): Z coupling strength (defaults to J) Default: `to J)`.
- **h** (`<class 'float'>`): Magnetic field
- **dtype** (`<class 'torch.dtype'>`): Data type (real dtypes use real representation)
- **device** (`typing.Optional[torch.device]`): Device

**Returns**: `<class 'tensornet.core.mpo.MPO'>` - MPO representation of Hamiltonian

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\mps\hamiltonians.py:73](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\mps\hamiltonians.py#L73)*

### `pauli_matrices`

```python
def pauli_matrices(dtype: torch.dtype = torch.complex128, device: Optional[torch.device] = None) -> tuple
```

Return Pauli matrices sigma_x, sigma_y, sigma_z.

**Returns**: `<class 'tuple'>` - (sigma_x, sigma_y, sigma_z) each of shape (2, 2)

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\mps\hamiltonians.py:13](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\mps\hamiltonians.py#L13)*

### `spin_operators`

```python
def spin_operators(S: float = 0.5, dtype: torch.dtype = torch.complex128, device: Optional[torch.device] = None) -> tuple
```

Return spin operators S_x, S_y, S_z for spin S.

**Parameters:**

- **S** (`<class 'float'>`): Spin value (0.5, 1, 1.5, ...)

**Returns**: `<class 'tuple'>` - (S_x, S_y, S_z, S_p, S_m) where S_p = S_x + i*S_y, S_m = S_x - i*S_y

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\mps\hamiltonians.py:33](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\mps\hamiltonians.py#L33)*

### `tfim_mpo`

```python
def tfim_mpo(L: int, J: float = 1.0, g: float = 1.0, dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None) -> tensornet.core.mpo.MPO
```

Transverse-field Ising model as MPO.

H = -J * sum_i Z_i Z_{i+1} - g * sum_i X_i

Critical point at g = 1 (for J = 1).

**Parameters:**

- **L** (`<class 'int'>`): Number of sites
- **J** (`<class 'float'>`): Ising coupling
- **g** (`<class 'float'>`): Transverse field strength
- **dtype** (`<class 'torch.dtype'>`): Data type
- **device** (`typing.Optional[torch.device]`): Device

**Returns**: `<class 'tensornet.core.mpo.MPO'>` - MPO representation

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\mps\hamiltonians.py:168](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\mps\hamiltonians.py#L168)*

### `xx_mpo`

```python
def xx_mpo(L: int, J: float = 1.0, h: float = 0.0, dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None) -> tensornet.core.mpo.MPO
```

XX model as MPO.

H = J * sum_i (X_i X_{i+1} + Y_i Y_{i+1}) + h * sum_i Z_i

This is equivalent to free fermions and exactly solvable.

**Parameters:**

- **L** (`<class 'int'>`): Number of sites
- **J** (`<class 'float'>`): Coupling strength
- **h** (`<class 'float'>`): Magnetic field
- **dtype** (`<class 'torch.dtype'>`): Data type
- **device** (`typing.Optional[torch.device]`): Device

**Returns**: `<class 'tensornet.core.mpo.MPO'>` - MPO representation

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\mps\hamiltonians.py:230](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\mps\hamiltonians.py#L230)*

### `xyz_mpo`

```python
def xyz_mpo(L: int, Jx: float = 1.0, Jy: float = 1.0, Jz: float = 1.0, h: float = 0.0, dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None) -> tensornet.core.mpo.MPO
```

XYZ model as MPO.

H = sum_i (Jx * X_i X_{i+1} + Jy * Y_i Y_{i+1} + Jz * Z_i Z_{i+1}) + h * sum_i Z_i

**Parameters:**

- **L** (`<class 'int'>`): Number of sites
- **Jx** (`<class 'float'>`): X coupling strength
- **Jy** (`<class 'float'>`): Y coupling strength
- **Jz** (`<class 'float'>`): Z coupling strength
- **h** (`<class 'float'>`): Magnetic field
- **dtype** (`<class 'torch.dtype'>`): Data type
- **device** (`typing.Optional[torch.device]`): Device

**Returns**: `<class 'tensornet.core.mpo.MPO'>` - MPO representation

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\mps\hamiltonians.py:258](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\mps\hamiltonians.py#L258)*
