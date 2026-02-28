# Module `core.mps`

Matrix Product State (MPS) ==========================

Core MPS class for representing 1D quantum states and classical fields.

Tensor convention:
    A[i] : (χ_left, d, χ_right)

    For a chain of L sites with physical dimension d and bond dimension χ:
    |ψ⟩ = Σ A[0]_{1,σ₀,α₀} A[1]_{α₀,σ₁,α₁} ... A[L-1]_{α_{L-2},σ_{L-1},1} |σ₀σ₁...σ_{L-1}⟩

**Contents:**

- [Classes](#classes)

## Classes

### class `MPS`

Matrix Product State representation.

#### Properties

##### `L`

```python
def L(self) -> 'int'
```

Number of sites.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py:49](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py#L49)*

##### `chi`

```python
def chi(self) -> 'int'
```

Maximum bond dimension.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py:59](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py#L59)*

##### `d`

```python
def d(self) -> 'int'
```

Physical dimension (from first site).

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py:54](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py#L54)*

##### `device`

```python
def device(self) -> 'torch.device'
```

Device of tensors.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py:71](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py#L71)*

##### `dtype`

```python
def dtype(self) -> 'torch.dtype'
```

Data type of tensors.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py:66](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py#L66)*

#### Methods

##### `__init__`

```python
def __init__(self, tensors: 'List[Tensor]')
```

Initialize MPS from list of tensors.

**Parameters:**

- **tensors** (`typing.List[torch.Tensor]`): List of tensors with shape (χ_left, d, χ_right)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py:39](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py#L39)*

##### `bond_dims`

```python
def bond_dims(self) -> 'List[int]'
```

Return list of bond dimensions [χ₀, χ₁, ..., χ_{L-1}].

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py:76](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py#L76)*

##### `canonicalize_left_`

```python
def canonicalize_left_(self) -> 'MPS'
```

Left-canonicalize MPS in-place.

After this, A[i]^† @ A[i] = I for all i < L-1.
The norm is absorbed into the last tensor.

**Returns**: `<class 'mps.MPS'>` - self

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py:218](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py#L218)*

##### `canonicalize_right_`

```python
def canonicalize_right_(self) -> 'MPS'
```

Right-canonicalize MPS in-place.

After this, A[i] @ A[i]^† = I for all i > 0.
The norm is absorbed into the first tensor.

**Returns**: `<class 'mps.MPS'>` - self

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py:243](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py#L243)*

##### `canonicalize_to_`

```python
def canonicalize_to_(self, site: 'int') -> 'MPS'
```

Mixed-canonical form with orthogonality center at site.

**Parameters:**

- **site** (`<class 'int'>`): Orthogonality center (0 to L-1)

**Returns**: `<class 'mps.MPS'>` - self

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py:268](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py#L268)*

##### `copy`

```python
def copy(self) -> 'MPS'
```

Return deep copy of MPS.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py:188](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py#L188)*

##### `entropy`

```python
def entropy(self, bond: 'int') -> 'Tensor'
```

Compute von Neumann entanglement entropy at bond.

S = -Tr(ρ log ρ) where ρ is the reduced density matrix.

**Parameters:**

- **bond** (`<class 'int'>`): Bond index (0 to L-2)

**Returns**: `<class 'torch.Tensor'>` - Entanglement entropy

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py:299](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py#L299)*

##### `expectation_local`

```python
def expectation_local(self, op: 'Tensor', site: 'int') -> 'Tensor'
```

Compute ⟨ψ|O_site|ψ⟩ for local operator O.

**Parameters:**

- **op** (`<class 'torch.Tensor'>`): Local operator of shape (d, d)
- **site** (`<class 'int'>`): Site index

**Returns**: `<class 'torch.Tensor'>` - Expectation value

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py:332](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py#L332)*

##### `from_tensor`

```python
def from_tensor(tensor: 'Tensor', chi_max: 'Optional[int]' = None, cutoff: 'float' = 1e-14) -> 'MPS'
```

Convert dense tensor to MPS via successive SVD.

**Parameters:**

- **tensor** (`<class 'torch.Tensor'>`): Dense tensor of shape (d, d, ..., d)
- **chi_max** (`typing.Optional[int]`): Maximum bond dimension
- **cutoff** (`<class 'float'>`): SVD cutoff

**Returns**: `<class 'mps.MPS'>` - MPS approximation

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py:120](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py#L120)*

##### `norm`

```python
def norm(self) -> 'Tensor'
```

Compute norm ⟨ψ|ψ⟩^{1/2}.

**Returns**: `<class 'torch.Tensor'>` - Scalar tensor with norm

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py:192](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py#L192)*

##### `normalize_`

```python
def normalize_(self) -> 'MPS'
```

Normalize MPS in-place. Returns self.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py:208](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py#L208)*

##### `random`

```python
def random(L: 'int', d: 'int', chi: 'int', dtype: 'torch.dtype' = torch.float64, device: 'Optional[torch.device]' = None, normalize: 'bool' = True) -> 'MPS'
```

Create random MPS with given dimensions.

**Parameters:**

- **L** (`<class 'int'>`): Number of sites
- **d** (`<class 'int'>`): Physical dimension
- **chi** (`<class 'int'>`): Bond dimension
- **dtype** (`<class 'torch.dtype'>`): Data type
- **device** (`typing.Optional[torch.device]`): Device
- **normalize** (`<class 'bool'>`): If True, normalize the state

**Returns**: `<class 'mps.MPS'>` - Random MPS

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py:80](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py#L80)*

##### `to_tensor`

```python
def to_tensor(self) -> 'Tensor'
```

Contract MPS to dense tensor.

**Returns**: `<class 'torch.Tensor'>` - Dense tensor of shape (d, d, ..., d)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py:169](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py#L169)*

##### `truncate_`

```python
def truncate_(self, chi_max: 'int', cutoff: 'float' = 1e-14) -> 'MPS'
```

Truncate bond dimension via SVD.

**Parameters:**

- **chi_max** (`<class 'int'>`): Maximum bond dimension
- **cutoff** (`<class 'float'>`): SVD cutoff

**Returns**: `<class 'mps.MPS'>` - self

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py:366](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\mps.py#L366)*
