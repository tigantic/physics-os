# Module `algorithms.fermionic`

Fermionic MPS =============

Tensor networks for fermionic systems using Jordan-Wigner transformation.

Physics:
    Fermions have anticommutation relations: {c_i, c_j†} = δ_ij

    The Jordan-Wigner transformation maps fermions to spins:
    c_i = (∏_{j<i} σ^z_j) σ^-_i
    c_i† = (∏_{j<i} σ^z_j) σ^+_i

    This allows us to use standard MPS for fermionic systems,
    with the string operators (∏ σ^z) handled implicitly in the MPO.

Models:
    - Spinless fermion chain: H = -t Σᵢ (c†_i c_{i+1} + h.c.) + V Σᵢ n_i n_{i+1}
    - Hubbard model: H = -t Σᵢσ (c†_iσ c_{i+1,σ} + h.c.) + U Σᵢ n_i↑ n_i↓

The key insight is that in 1D with nearest-neighbor hopping,
the Jordan-Wigner strings cancel, making the MPO local.

**Contents:**

- [Functions](#functions)

## Functions

### `compute_density`

```python
def compute_density(mps: ontic.core.mps.MPS) -> torch.Tensor
```

Compute local density ⟨n_i⟩ for each site.

**Parameters:**

- **mps** (`<class 'ontic.core.mps.MPS'>`): MPS state (for spinless fermions)

**Returns**: `<class 'torch.Tensor'>` - Tensor of local densities

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\algorithms\fermionic.py:318](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\algorithms\fermionic.py#L318)*

### `fermi_sea_mps`

```python
def fermi_sea_mps(L: int, n_particles: int, dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None) -> ontic.core.mps.MPS
```

Create MPS for a Fermi sea (filled lowest modes).

For spinless fermions, this is a product state with
the first n_particles sites occupied.

**Parameters:**

- **L** (`<class 'int'>`): Number of sites
- **n_particles** (`<class 'int'>`): Number of fermions
- **dtype** (`<class 'torch.dtype'>`): Data type
- **device** (`typing.Optional[torch.device]`): Device

**Returns**: `<class 'ontic.core.mps.MPS'>` - MPS representing the Fermi sea

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\algorithms\fermionic.py:248](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\algorithms\fermionic.py#L248)*

### `half_filled_mps`

```python
def half_filled_mps(L: int, dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None) -> ontic.core.mps.MPS
```

Create half-filled MPS (alternating occupied/empty).

This is a good initial state for repulsive interactions
(CDW-like order).

**Parameters:**

- **L** (`<class 'int'>`): Number of sites
- **dtype** (`<class 'torch.dtype'>`): Data type
- **device** (`typing.Optional[torch.device]`): Device

**Returns**: `<class 'ontic.core.mps.MPS'>` - MPS with alternating occupation

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\algorithms\fermionic.py:284](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\algorithms\fermionic.py#L284)*

### `hubbard_mpo`

```python
def hubbard_mpo(L: int, t: float = 1.0, U: float = 4.0, mu: float = 0.0, dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None) -> ontic.core.mpo.MPO
```

Hubbard model Hamiltonian as MPO.

H = -t Σᵢσ (c†_iσ c_{i+1,σ} + h.c.) + U Σᵢ n_i↑ n_i↓ - μ Σᵢ (n_i↑ + n_i↓)

Uses a 4-dimensional local Hilbert space:
|0⟩ = empty, |↑⟩ = spin up, |↓⟩ = spin down, |↑↓⟩ = doubly occupied

**Parameters:**

- **L** (`<class 'int'>`): Number of sites
- **t** (`<class 'float'>`): Hopping amplitude
- **U** (`<class 'float'>`): On-site Coulomb repulsion
- **mu** (`<class 'float'>`): Chemical potential
- **dtype** (`<class 'torch.dtype'>`): Data type
- **device** (`typing.Optional[torch.device]`): Device

**Returns**: `<class 'ontic.core.mpo.MPO'>` - MPO representation

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\algorithms\fermionic.py:128](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\algorithms\fermionic.py#L128)*

### `spinless_fermion_mpo`

```python
def spinless_fermion_mpo(L: int, t: float = 1.0, V: float = 0.0, mu: float = 0.0, dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None) -> ontic.core.mpo.MPO
```

Spinless fermion chain Hamiltonian as MPO.

H = -t Σᵢ (c†_i c_{i+1} + c†_{i+1} c_i) + V Σᵢ n_i n_{i+1} - μ Σᵢ n_i

After Jordan-Wigner transformation to spins:
c†_i c_{i+1} = σ^+_i σ^-_{i+1}  (for nearest neighbors, no string!)
n_i = (1 + σ^z_i) / 2

**Parameters:**

- **L** (`<class 'int'>`): Number of sites
- **t** (`<class 'float'>`): Hopping amplitude
- **V** (`<class 'float'>`): Nearest-neighbor interaction
- **mu** (`<class 'float'>`): Chemical potential
- **dtype** (`<class 'torch.dtype'>`): Data type
- **device** (`typing.Optional[torch.device]`): Device

**Returns**: `<class 'ontic.core.mpo.MPO'>` - MPO representation

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\algorithms\fermionic.py:31](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\algorithms\fermionic.py#L31)*
