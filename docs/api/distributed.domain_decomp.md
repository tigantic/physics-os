# Module `distributed.domain_decomp`

Domain decomposition for parallel CFD simulations.

This module implements domain decomposition strategies for
distributing CFD grids across multiple processors/GPUs.

Author: Tigantic Holdings LLC

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `DecompType`(Enum)

Domain decomposition type.

### class `DomainConfig`

Configuration for domain decomposition.

#### Attributes

- **nx** (`<class 'int'>`): 
- **ny** (`<class 'int'>`): 
- **nz** (`<class 'int'>`): 
- **decomp_type** (`<enum 'DecompType'>`): 
- **n_procs** (`<class 'int'>`): 
- **n_ghost** (`<class 'int'>`): 
- **periodic_x** (`<class 'bool'>`): 
- **periodic_y** (`<class 'bool'>`): 
- **periodic_z** (`<class 'bool'>`): 
- **load_balance** (`<class 'bool'>`): 
- **x_min** (`<class 'float'>`): 
- **x_max** (`<class 'float'>`): 
- **y_min** (`<class 'float'>`): 
- **y_max** (`<class 'float'>`): 
- **z_min** (`<class 'float'>`): 
- **z_max** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, nx: int = 64, ny: int = 64, nz: int = 1, decomp_type: domain_decomp.DecompType = <DecompType.BLOCK: 3>, n_procs: int = 4, n_ghost: int = 2, periodic_x: bool = False, periodic_y: bool = False, periodic_z: bool = False, load_balance: bool = True, x_min: float = 0.0, x_max: float = 1.0, y_min: float = 0.0, y_max: float = 1.0, z_min: float = 0.0, z_max: float = 1.0) -> None
```

### class `DomainDecomposition`

Domain decomposition manager.

Handles partitioning of CFD grids across multiple
processors and management of ghost zones.

#### Methods

##### `__init__`

```python
def __init__(self, config: domain_decomp.DomainConfig)
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\domain_decomp.py:103](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\domain_decomp.py#L103)*

##### `get_neighbors`

```python
def get_neighbors(self, rank: int) -> Dict[str, Optional[int]]
```

Get neighbor ranks for a processor.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\domain_decomp.py:264](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\domain_decomp.py#L264)*

##### `get_subdomain`

```python
def get_subdomain(self, rank: int) -> domain_decomp.SubdomainInfo
```

Get subdomain information for a processor.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\domain_decomp.py:260](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\domain_decomp.py#L260)*

##### `global_to_local`

```python
def global_to_local(self, rank: int, i: int, j: int, k: int = 0) -> Tuple[int, int, int]
```

Convert global index to local index.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\domain_decomp.py:276](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\domain_decomp.py#L276)*

##### `local_to_global`

```python
def local_to_global(self, rank: int, i: int, j: int, k: int = 0) -> Tuple[int, int, int]
```

Convert local index to global index.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\domain_decomp.py:288](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\domain_decomp.py#L288)*

### class `SubdomainInfo`

Information about a subdomain.

#### Attributes

- **rank** (`<class 'int'>`): 
- **i_start** (`<class 'int'>`): 
- **i_end** (`<class 'int'>`): 
- **j_start** (`<class 'int'>`): 
- **j_end** (`<class 'int'>`): 
- **k_start** (`<class 'int'>`): 
- **k_end** (`<class 'int'>`): 
- **local_nx** (`<class 'int'>`): 
- **local_ny** (`<class 'int'>`): 
- **local_nz** (`<class 'int'>`): 
- **neighbor_left** (`typing.Optional[int]`): 
- **neighbor_right** (`typing.Optional[int]`): 
- **neighbor_bottom** (`typing.Optional[int]`): 
- **neighbor_top** (`typing.Optional[int]`): 
- **neighbor_back** (`typing.Optional[int]`): 
- **neighbor_front** (`typing.Optional[int]`): 
- **x_local** (`typing.Optional[torch.Tensor]`): 
- **y_local** (`typing.Optional[torch.Tensor]`): 
- **z_local** (`typing.Optional[torch.Tensor]`): 

#### Methods

##### `__init__`

```python
def __init__(self, rank: int, i_start: int, i_end: int, j_start: int, j_end: int, k_start: int, k_end: int, local_nx: int, local_ny: int, local_nz: int, neighbor_left: Optional[int] = None, neighbor_right: Optional[int] = None, neighbor_bottom: Optional[int] = None, neighbor_top: Optional[int] = None, neighbor_back: Optional[int] = None, neighbor_front: Optional[int] = None, x_local: Optional[torch.Tensor] = None, y_local: Optional[torch.Tensor] = None, z_local: Optional[torch.Tensor] = None) -> None
```

## Functions

### `compute_ghost_zones`

```python
def compute_ghost_zones(data: torch.Tensor, subdomain: domain_decomp.SubdomainInfo, n_ghost: int) -> Dict[str, torch.Tensor]
```

Extract ghost zone data to send to neighbors.

**Parameters:**

- **data** (`<class 'torch.Tensor'>`): Local field data [ny, nx] or [nz, ny, nx]
- **subdomain** (`<class 'domain_decomp.SubdomainInfo'>`): Subdomain information
- **n_ghost** (`<class 'int'>`): Number of ghost cells

**Returns**: `typing.Dict[str, torch.Tensor]` - Dictionary of ghost zone data per direction

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\domain_decomp.py:314](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\domain_decomp.py#L314)*

### `decompose_domain`

```python
def decompose_domain(config: domain_decomp.DomainConfig) -> domain_decomp.DomainDecomposition
```

Convenience function to create domain decomposition.

**Parameters:**

- **config** (`<class 'domain_decomp.DomainConfig'>`): Domain configuration

**Returns**: `<class 'domain_decomp.DomainDecomposition'>` - Domain decomposition object

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\domain_decomp.py:301](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\domain_decomp.py#L301)*

### `exchange_ghost_data`

```python
def exchange_ghost_data(data: torch.Tensor, subdomain: domain_decomp.SubdomainInfo, received: Dict[str, torch.Tensor], n_ghost: int) -> torch.Tensor
```

Fill ghost zones with received data.

**Parameters:**

- **data** (`<class 'torch.Tensor'>`): Local field data
- **subdomain** (`<class 'domain_decomp.SubdomainInfo'>`): Subdomain information
- **received** (`typing.Dict[str, torch.Tensor]`): Dictionary of received ghost data
- **n_ghost** (`<class 'int'>`): Number of ghost cells

**Returns**: `<class 'torch.Tensor'>` - Updated data with filled ghost zones

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\domain_decomp.py:372](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\domain_decomp.py#L372)*

### `test_domain_decomposition`

```python
def test_domain_decomposition()
```

Test domain decomposition.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\domain_decomp.py:428](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\domain_decomp.py#L428)*
