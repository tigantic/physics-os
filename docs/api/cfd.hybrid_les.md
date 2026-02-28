# Module `cfd.hybrid_les`

Hybrid RANS-LES Models ======================

Bridge between RANS and LES approaches for efficient high-fidelity turbulence
modeling. Key hybrid methodologies:

    DES (Detached Eddy Simulation):
        - RANS in attached boundary layers
        - LES in separated regions and wakes
        - Length scale switching: l_hybrid = min(l_RANS, C_DES * Δ)

    DDES (Delayed DES):
        - Prevents premature LES in boundary layers
        - Shielding function delays transition to LES
        - Based on local flow/turbulence ratios

    IDDES (Improved Delayed DES):
        - Wall-modeled LES (WMLES) branch
        - Seamless RANS-LES interface
        - Optimized for wall-bounded flows

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `HybridLESState`

State variables for hybrid RANS-LES models.

#### Attributes

- **nu_sgs** (`<class 'torch.Tensor'>`): 
- **blending** (`<class 'torch.Tensor'>`): 
- **length_scale** (`<class 'torch.Tensor'>`): 
- **mode** (`<class 'torch.Tensor'>`): 
- **f_d** (`typing.Optional[torch.Tensor]`): 
- **f_e** (`typing.Optional[torch.Tensor]`): 
- **f_b** (`typing.Optional[torch.Tensor]`): 

#### Methods

##### `__init__`

```python
def __init__(self, nu_sgs: torch.Tensor, blending: torch.Tensor, length_scale: torch.Tensor, mode: torch.Tensor, f_d: Optional[torch.Tensor] = None, f_e: Optional[torch.Tensor] = None, f_b: Optional[torch.Tensor] = None) -> None
```

### class `HybridModel`(Enum)

Available hybrid RANS-LES models.

## Functions

### `compute_grid_scale`

```python
def compute_grid_scale(dx: torch.Tensor, dy: torch.Tensor, dz: Optional[torch.Tensor] = None, method: str = 'max') -> torch.Tensor
```

Compute LES grid/filter scale from mesh spacing.

**Parameters:**

- **method** (`<class 'str'>`): "max" (largest), "cube" (volume^1/3), or "sum" (sum/3)

**Returns**: `<class 'torch.Tensor'>` - Grid scale Δ

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\hybrid_les.py:78](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\hybrid_les.py#L78)*

### `compute_hybrid_viscosity`

```python
def compute_hybrid_viscosity(nu_rans: torch.Tensor, length_scale: torch.Tensor, strain_rate: torch.Tensor, model: hybrid_les.HybridModel = <HybridModel.DDES: 'ddes'>, c_s: float = 0.17) -> torch.Tensor
```

Compute turbulent viscosity for hybrid model.

In RANS regions: use RANS viscosity
In LES regions: use Smagorinsky-like model

**Parameters:**

- **nu_rans** (`<class 'torch.Tensor'>`): RANS turbulent viscosity
- **length_scale** (`<class 'torch.Tensor'>`): Hybrid length scale
- **strain_rate** (`<class 'torch.Tensor'>`): Strain rate magnitude
- **model** (`<enum 'HybridModel'>`): Hybrid model type
- **c_s** (`<class 'float'>`): Smagorinsky constant for LES regions

**Returns**: `<class 'torch.Tensor'>` - Turbulent/SGS viscosity

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\hybrid_les.py:344](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\hybrid_les.py#L344)*

### `compute_r_d`

```python
def compute_r_d(nu_t: torch.Tensor, nu: float, velocity_gradient: torch.Tensor, d_wall: torch.Tensor, kappa: float = 0.41) -> torch.Tensor
```

Compute DDES delay function parameter r_d.

r_d = (ν_t + ν) / (|∇u| κ² d²)

This measures whether the flow is "RANS-like" (r_d ≈ 1) or
resolved turbulence (r_d << 1).

**Parameters:**

- **nu_t** (`<class 'torch.Tensor'>`): Turbulent/SGS viscosity
- **nu** (`<class 'float'>`): Molecular viscosity
- **velocity_gradient** (`<class 'torch.Tensor'>`): Velocity gradient magnitude |∇u|
- **d_wall** (`<class 'torch.Tensor'>`): Wall distance
- **kappa** (`<class 'float'>`): Von Kármán constant

**Returns**: `<class 'torch.Tensor'>` - r_d parameter field

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\hybrid_les.py:155](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\hybrid_les.py#L155)*

### `compute_wall_distance_scale`

```python
def compute_wall_distance_scale(d_wall: torch.Tensor, kappa: float = 0.41) -> torch.Tensor
```

Compute RANS mixing length scale from wall distance.

l_RANS = κ * d_wall (for SA-type models)

**Parameters:**

- **d_wall** (`<class 'torch.Tensor'>`): Distance to nearest wall
- **kappa** (`<class 'float'>`): Von Kármán constant

**Returns**: `<class 'torch.Tensor'>` - RANS length scale

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\hybrid_les.py:112](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\hybrid_les.py#L112)*

### `ddes_delay_function`

```python
def ddes_delay_function(r_d: torch.Tensor, c_d1: float = 8.0, c_d2: float = 3.0) -> torch.Tensor
```

DDES delay/shielding function f_d.

f_d = 1 - tanh([C_d1 * r_d]^C_d2)

When f_d ≈ 0 (r_d large), RANS mode is preserved.
When f_d ≈ 1 (r_d small), LES mode can activate.

**Parameters:**

- **r_d** (`<class 'torch.Tensor'>`): Delay function parameter c_d1, c_d2: Model constants

**Returns**: `<class 'torch.Tensor'>` - Delay function f_d

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\hybrid_les.py:186](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\hybrid_les.py#L186)*

### `ddes_length_scale`

```python
def ddes_length_scale(l_rans: torch.Tensor, delta: torch.Tensor, f_d: torch.Tensor, c_des: float = 0.65) -> torch.Tensor
```

DDES length scale with delay function.

l_DDES = l_RANS - f_d * max(0, l_RANS - C_DES * Δ)

**Parameters:**

- **l_rans** (`<class 'torch.Tensor'>`): RANS length scale
- **delta** (`<class 'torch.Tensor'>`): LES grid scale
- **f_d** (`<class 'torch.Tensor'>`): Delay function
- **c_des** (`<class 'float'>`): DES coefficient

**Returns**: `<class 'torch.Tensor'>` - DDES hybrid length scale

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\hybrid_les.py:209](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\hybrid_les.py#L209)*

### `des_length_scale`

```python
def des_length_scale(l_rans: torch.Tensor, delta: torch.Tensor, c_des: float = 0.65) -> torch.Tensor
```

Original DES length scale.

l_DES = min(l_RANS, C_DES * Δ)

When Δ is smaller (in separated regions), LES mode activates.

**Parameters:**

- **l_rans** (`<class 'torch.Tensor'>`): RANS length scale (κ * d_wall)
- **delta** (`<class 'torch.Tensor'>`): LES grid scale
- **c_des** (`<class 'float'>`): DES coefficient

**Returns**: `<class 'torch.Tensor'>` - Hybrid length scale

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\hybrid_les.py:131](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\hybrid_les.py#L131)*

### `estimate_rans_les_ratio`

```python
def estimate_rans_les_ratio(state: hybrid_les.HybridLESState) -> Dict[str, float]
```

Compute statistics on RANS vs LES content.

**Parameters:**

- **state** (`<class 'hybrid_les.HybridLESState'>`): Hybrid LES state

**Returns**: `typing.Dict[str, float]` - Dict with RANS/LES percentages

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\hybrid_les.py:460](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\hybrid_les.py#L460)*

### `iddes_blending_function`

```python
def iddes_blending_function(d_wall: torch.Tensor, delta: torch.Tensor, r_d: torch.Tensor, c_t: float = 1.87, c_l: float = 3.55) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

IDDES blending functions for RANS-LES interface.

**Parameters:**

- **d_wall** (`<class 'torch.Tensor'>`): Wall distance
- **delta** (`<class 'torch.Tensor'>`): Grid scale
- **r_d** (`<class 'torch.Tensor'>`): Delay function parameter c_t, c_l: Model constants

**Returns**: `typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]` - (f_e, f_b, alpha) - Elevation, blending, and alpha functions

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\hybrid_les.py:233](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\hybrid_les.py#L233)*

### `iddes_length_scale`

```python
def iddes_length_scale(l_rans: torch.Tensor, delta: torch.Tensor, d_wall: torch.Tensor, alpha: torch.Tensor, c_des: float = 0.65) -> torch.Tensor
```

IDDES hybrid length scale.

l_IDDES = f_d * (1 + f_e) * l_RANS + (1 - f_d) * C_DES * Δ

Simplified version using alpha parameter:
l_IDDES = (1 - alpha) * l_RANS + alpha * C_DES * Δ

**Parameters:**

- **l_rans** (`<class 'torch.Tensor'>`): RANS length scale
- **delta** (`<class 'torch.Tensor'>`): LES grid scale
- **d_wall** (`<class 'torch.Tensor'>`): Wall distance
- **alpha** (`<class 'torch.Tensor'>`): Blending function
- **c_des** (`<class 'float'>`): DES coefficient

**Returns**: `<class 'torch.Tensor'>` - IDDES hybrid length scale

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\hybrid_les.py:270](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\hybrid_les.py#L270)*

### `run_hybrid_les`

```python
def run_hybrid_les(rho: torch.Tensor, u: torch.Tensor, d_wall: torch.Tensor, grid_spacing: Tuple[torch.Tensor, ...], nu: float, nu_rans: torch.Tensor, model: hybrid_les.HybridModel = <HybridModel.DDES: 'ddes'>) -> hybrid_les.HybridLESState
```

Main driver for hybrid RANS-LES computation.

**Parameters:**

- **rho** (`<class 'torch.Tensor'>`): Density field
- **u** (`<class 'torch.Tensor'>`): Velocity field (Nd, Nx, Ny, [Nz])
- **d_wall** (`<class 'torch.Tensor'>`): Wall distance field
- **grid_spacing** (`typing.Tuple[torch.Tensor, ...]`): Tuple of (dx, dy, [dz]) tensors
- **nu** (`<class 'float'>`): Molecular viscosity
- **nu_rans** (`<class 'torch.Tensor'>`): RANS turbulent viscosity field
- **model** (`<enum 'HybridModel'>`): Hybrid model to use

**Returns**: `<class 'hybrid_les.HybridLESState'>` - HybridLESState with computed quantities

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\hybrid_les.py:374](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\hybrid_les.py#L374)*

### `sas_length_scale`

```python
def sas_length_scale(k: torch.Tensor, omega: torch.Tensor, velocity_gradient: torch.Tensor, velocity_laplacian: torch.Tensor, c_mu: float = 0.09, kappa: float = 0.41) -> torch.Tensor
```

Scale-Adaptive Simulation (SAS) length scale.

The SAS length scale responds to resolved turbulence
without explicit grid-based switching.

L_vK = κ |S| / |∇²u|  (von Kármán length scale)
L_t = √k / (c_μ^0.25 ω)  (turbulent length scale)

**Parameters:**

- **k** (`<class 'torch.Tensor'>`): Turbulent kinetic energy
- **omega** (`<class 'torch.Tensor'>`): Specific dissipation rate
- **velocity_gradient** (`<class 'torch.Tensor'>`): |S| strain rate magnitude
- **velocity_laplacian** (`<class 'torch.Tensor'>`): |∇²u| Laplacian of velocity c_mu, kappa: Model constants

**Returns**: `<class 'torch.Tensor'>` - SAS length scale

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\hybrid_les.py:307](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\hybrid_les.py#L307)*

### `validate_hybrid_les`

```python
def validate_hybrid_les()
```

Run validation tests for hybrid RANS-LES models.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\hybrid_les.py:481](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\hybrid_les.py#L481)*
