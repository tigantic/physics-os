# Module `core.states`

Standard MPS States ===================

Factory functions for common quantum states.

**Contents:**

- [Functions](#functions)

## Functions

### `all_down_mps`

```python
def all_down_mps(L: int, dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None) -> tensornet.core.mps.MPS
```

Create |↓↓...↓⟩ state.

**Parameters:**

- **L** (`<class 'int'>`): Number of sites
- **dtype** (`<class 'torch.dtype'>`): Data type
- **device** (`typing.Optional[torch.device]`): Device

**Returns**: `<class 'tensornet.core.mps.MPS'>` - All-down product state

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\states.py:138](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\states.py#L138)*

### `all_up_mps`

```python
def all_up_mps(L: int, dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None) -> tensornet.core.mps.MPS
```

Create |↑↑...↑⟩ state.

**Parameters:**

- **L** (`<class 'int'>`): Number of sites
- **dtype** (`<class 'torch.dtype'>`): Data type
- **device** (`typing.Optional[torch.device]`): Device

**Returns**: `<class 'tensornet.core.mps.MPS'>` - All-up product state

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\states.py:118](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\states.py#L118)*

### `domain_wall_mps`

```python
def domain_wall_mps(L: int, position: int, dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None) -> tensornet.core.mps.MPS
```

Create domain wall state: |↑↑...↑↓↓...↓⟩

**Parameters:**

- **L** (`<class 'int'>`): Number of sites
- **position** (`<class 'int'>`): Position of domain wall (sites 0..position-1 are up)
- **dtype** (`<class 'torch.dtype'>`): Data type
- **device** (`typing.Optional[torch.device]`): Device

**Returns**: `<class 'tensornet.core.mps.MPS'>` - Domain wall product state

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\states.py:180](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\states.py#L180)*

### `ghz_mps`

```python
def ghz_mps(L: int, dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None) -> tensornet.core.mps.MPS
```

Create GHZ state: |GHZ⟩ = (|00...0⟩ + |11...1⟩) / √2

This is a maximally entangled state with S = ln(2) at every bond.

**Parameters:**

- **L** (`<class 'int'>`): Number of sites
- **dtype** (`<class 'torch.dtype'>`): Data type
- **device** (`typing.Optional[torch.device]`): Device

**Returns**: `<class 'tensornet.core.mps.MPS'>` - MPS representation of GHZ state

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\states.py:16](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\states.py#L16)*

### `neel_mps`

```python
def neel_mps(L: int, dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None) -> tensornet.core.mps.MPS
```

Create Néel state: |↑↓↑↓...⟩

**Parameters:**

- **L** (`<class 'int'>`): Number of sites
- **dtype** (`<class 'torch.dtype'>`): Data type
- **device** (`typing.Optional[torch.device]`): Device

**Returns**: `<class 'tensornet.core.mps.MPS'>` - Néel product state

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\states.py:158](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\states.py#L158)*

### `product_mps`

```python
def product_mps(states: List[torch.Tensor], dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None) -> tensornet.core.mps.MPS
```

Create product state MPS from local states.

|ψ⟩ = |ψ₀⟩ ⊗ |ψ₁⟩ ⊗ ... ⊗ |ψ_{L-1}⟩

**Parameters:**

- **states** (`typing.List[torch.Tensor]`): List of local state vectors, each of shape (d,)
- **dtype** (`<class 'torch.dtype'>`): Data type
- **device** (`typing.Optional[torch.device]`): Device

**Returns**: `<class 'tensornet.core.mps.MPS'>` - MPS with bond dimension 1

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\states.py:61](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\states.py#L61)*

### `random_mps`

```python
def random_mps(L: int, d: int, chi: int, dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None, normalize: bool = True) -> tensornet.core.mps.MPS
```

Create random MPS.

Alias for MPS.random() for convenience.

**Parameters:**

- **L** (`<class 'int'>`): Number of sites
- **d** (`<class 'int'>`): Physical dimension
- **chi** (`<class 'int'>`): Bond dimension
- **dtype** (`<class 'torch.dtype'>`): Data type
- **device** (`typing.Optional[torch.device]`): Device
- **normalize** (`<class 'bool'>`): Normalize the state

**Returns**: `<class 'tensornet.core.mps.MPS'>` - Random MPS

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\states.py:91](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\states.py#L91)*
