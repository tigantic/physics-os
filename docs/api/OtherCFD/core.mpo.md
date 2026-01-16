# Module `core.mpo`

Matrix Product Operator (MPO) =============================

MPO representation for operators on 1D systems (Hamiltonians, time evolution).

Tensor convention:
    W[i] : (D_left, d_out, d_in, D_right)

    Operator O = Σ W[0]_{1,σ₀,σ'₀,α₀} W[1]_{α₀,σ₁,σ'₁,α₁} ... |σ⟩⟨σ'|

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `MPO`

Matrix Product Operator representation.

Used for Hamiltonians, time evolution operators, and observables.

#### Properties

##### `D`

```python
def D(self) -> 'int'
```

Maximum MPO bond dimension.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\core\mpo.py:56](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\core\mpo.py#L56)*

##### `L`

```python
def L(self) -> 'int'
```

Number of sites.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\core\mpo.py:46](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\core\mpo.py#L46)*

##### `d`

```python
def d(self) -> 'int'
```

Physical dimension.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\core\mpo.py:51](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\core\mpo.py#L51)*

##### `device`

```python
def device(self) -> 'torch.device'
```

Device.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\core\mpo.py:68](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\core\mpo.py#L68)*

##### `dtype`

```python
def dtype(self) -> 'torch.dtype'
```

Data type.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\core\mpo.py:63](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\core\mpo.py#L63)*

#### Methods

##### `__init__`

```python
def __init__(self, tensors: 'List[Tensor]')
```

Initialize MPO from list of tensors.

**Parameters:**

- **tensors** (`typing.List[torch.Tensor]`): List of tensors with shape (D_left, d_out, d_in, D_right)

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\core\mpo.py:37](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\core\mpo.py#L37)*

##### `apply`

```python
def apply(self, mps) -> "'MPS'"
```

Apply MPO to MPS: |ψ'⟩ = O|ψ⟩

The result has bond dimension χ * D.

**Parameters:**

- **mps**: Input MPS

**Returns**: value - New MPS with O applied

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\core\mpo.py:101](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\core\mpo.py#L101)*

##### `copy`

```python
def copy(self) -> 'MPO'
```

Return deep copy.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\core\mpo.py:183](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\core\mpo.py#L183)*

##### `expectation`

```python
def expectation(self, mps) -> 'Tensor'
```

Compute ⟨ψ|O|ψ⟩.

**Parameters:**

- **mps**: MPS state

**Returns**: `<class 'torch.Tensor'>` - Expectation value

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\core\mpo.py:135](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\core\mpo.py#L135)*

##### `is_hermitian`

```python
def is_hermitian(self, tol: 'float' = 1e-10) -> 'bool'
```

Check if MPO is Hermitian.

**Parameters:**

- **tol** (`<class 'float'>`): Tolerance for comparison

**Returns**: `<class 'bool'>` - True if H = H†

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\core\mpo.py:170](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\core\mpo.py#L170)*

##### `to_matrix`

```python
def to_matrix(self) -> 'Tensor'
```

Contract MPO to dense matrix.

**Returns**: `<class 'torch.Tensor'>` - Dense matrix of shape (d^L, d^L)

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\core\mpo.py:73](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\core\mpo.py#L73)*

## Functions

### `mpo_sum`

```python
def mpo_sum(mpo1: 'MPO', mpo2: 'MPO') -> 'MPO'
```

Sum two MPOs: O = O1 + O2.

Result has bond dimension D1 + D2.

**Parameters:**

- **mpo1** (`<class 'mpo.MPO'>`): First MPO
- **mpo2** (`<class 'mpo.MPO'>`): Second MPO

**Returns**: `<class 'mpo.MPO'>` - Sum MPO

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\core\mpo.py:191](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\core\mpo.py#L191)*
