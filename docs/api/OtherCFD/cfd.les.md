# Module `cfd.les`

Large Eddy Simulation (LES) Subgrid-Scale Models =================================================

Implements subgrid-scale (SGS) models for Large Eddy Simulation
of turbulent flows at hypersonic conditions.

LES Philosophy:
    - Resolve large energy-containing eddies explicitly
    - Model small-scale (subgrid) turbulence effects
    - Filter width Δ ~ grid spacing determines separation

Filtered Navier-Stokes:
    ∂ρ̄/∂t + ∇·(ρ̄ũ) = 0
    ∂(ρ̄ũ)/∂t + ∇·(ρ̄ũ⊗ũ) = -∇p̄ + ∇·(τ̄ - τ_sgs)

    where τ_sgs = ρ̄(ũ⊗u - ũ⊗ũ) is the subgrid stress tensor

Models Implemented:
    1. Smagorinsky (1963) - Algebraic eddy viscosity
    2. Dynamic Smagorinsky (Germano, 1991) - Self-adjusting coefficient
    3. WALE (Nicoud & Ducros, 1999) - Wall-Adapting Local Eddy-viscosity
    4. Vreman (2004) - Minimal model for anisotropic grids
    5. Sigma (Nicoud et al., 2011) - Based on singular values of ∇u

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `LESModel`(Enum)

Available LES subgrid-scale models.

### class `LESState`

State for LES subgrid quantities.

#### Attributes

- **nu_sgs** (`<class 'torch.Tensor'>`): 
- **tau_sgs** (`typing.Optional[torch.Tensor]`): 
- **q_sgs** (`typing.Optional[torch.Tensor]`): 
- **delta** (`typing.Optional[torch.Tensor]`): 

#### Properties

##### `shape`

```python
def shape(self) -> torch.Size
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\les.py:83](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\les.py#L83)*

#### Methods

##### `__init__`

```python
def __init__(self, nu_sgs: torch.Tensor, tau_sgs: Optional[torch.Tensor] = None, q_sgs: Optional[torch.Tensor] = None, delta: Optional[torch.Tensor] = None) -> None
```

##### `zeros`

```python
def zeros(shape: Tuple[int, ...], dtype=torch.float64) -> 'LESState'
```

Create zero-initialized LES state.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\les.py:87](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\les.py#L87)*

## Functions

### `compute_sgs_viscosity`

```python
def compute_sgs_viscosity(model: les.LESModel, du_dx: torch.Tensor, du_dy: torch.Tensor, dv_dx: torch.Tensor, dv_dy: torch.Tensor, delta: float, rho: torch.Tensor, u: Optional[torch.Tensor] = None, v: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor
```

Unified interface for computing SGS viscosity.

**Parameters:**

- **model** (`<enum 'LESModel'>`): LES model type Velocity gradients
- **delta** (`<class 'float'>`): Filter width
- **rho** (`<class 'torch.Tensor'>`): Density u, v: Velocities (needed for dynamic model) **kwargs: Model-specific parameters

**Returns**: `<class 'torch.Tensor'>` - SGS eddy viscosity μ_sgs

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\les.py:766](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\les.py#L766)*

### `dynamic_smagorinsky_coefficient`

```python
def dynamic_smagorinsky_coefficient(u: torch.Tensor, v: torch.Tensor, du_dx: torch.Tensor, du_dy: torch.Tensor, dv_dx: torch.Tensor, dv_dy: torch.Tensor, delta: float, w: Optional[torch.Tensor] = None, du_dz: Optional[torch.Tensor] = None, dv_dz: Optional[torch.Tensor] = None, dw_dx: Optional[torch.Tensor] = None, dw_dy: Optional[torch.Tensor] = None, dw_dz: Optional[torch.Tensor] = None) -> torch.Tensor
```

Compute dynamic Smagorinsky coefficient C_s² using Germano identity.

L_ij = <u_i u_j> - <u_i><u_j>  (Leonard stress)
M_ij = α² Δ² |<S>| <S_ij> - Δ² <|S| S_ij>

C_s² = (L_ij M_ij) / (M_ij M_ij)

with averaging (here: local with clipping)

**Parameters:**

- **delta** (`<class 'float'>`): Grid filter width w, 3D gradients: Optional for 3D

**Returns**: `<class 'torch.Tensor'>` - Dynamic coefficient C_s² (clipped to positive values)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\les.py:311](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\les.py#L311)*

### `dynamic_smagorinsky_viscosity`

```python
def dynamic_smagorinsky_viscosity(C_s_squared: torch.Tensor, S: torch.Tensor, delta: float, rho: torch.Tensor) -> torch.Tensor
```

Dynamic Smagorinsky SGS viscosity.

μ_sgs = ρ C_s² Δ² |S|

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\les.py:401](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\les.py#L401)*

### `filter_width`

```python
def filter_width(dx: float, dy: float, dz: Optional[float] = None) -> torch.Tensor
```

Compute LES filter width Δ.

Common choices:
    - Δ = (dx·dy·dz)^(1/3) for 3D
    - Δ = (dx·dy)^(1/2) for 2D
    - Δ = max(dx, dy, dz) for anisotropic grids

**Parameters:**

- **dz** (`typing.Optional[float]`): Optional z spacing for 3D

**Returns**: `<class 'torch.Tensor'>` - Filter width scalar

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\les.py:98](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\les.py#L98)*

### `sgs_heat_flux`

```python
def sgs_heat_flux(mu_sgs: torch.Tensor, dT_dx: torch.Tensor, dT_dy: torch.Tensor, cp: float = 1005.0, Pr_t: float = 0.9, dT_dz: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]
```

Compute subgrid-scale heat flux using gradient-diffusion hypothesis.

q_sgs = -k_sgs ∇T = -(μ_sgs c_p / Pr_t) ∇T

**Parameters:**

- **mu_sgs** (`<class 'torch.Tensor'>`): SGS eddy viscosity [Pa·s] dT_dx, dT_dy: Temperature gradients [K/m]
- **cp** (`<class 'float'>`): Specific heat at constant pressure [J/(kg·K)]
- **Pr_t** (`<class 'float'>`): Turbulent Prandtl number
- **dT_dz** (`typing.Optional[torch.Tensor]`): Optional z-gradient for 3D

**Returns**: `typing.Tuple[torch.Tensor, ...]` - Tuple of heat flux components (q_x, q_y) or (q_x, q_y, q_z)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\les.py:727](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\les.py#L727)*

### `sigma_viscosity`

```python
def sigma_viscosity(du_dx: torch.Tensor, du_dy: torch.Tensor, dv_dx: torch.Tensor, dv_dy: torch.Tensor, delta: float, rho: torch.Tensor, C_sigma: float = 1.35, du_dz: Optional[torch.Tensor] = None, dv_dz: Optional[torch.Tensor] = None, dw_dx: Optional[torch.Tensor] = None, dw_dy: Optional[torch.Tensor] = None, dw_dz: Optional[torch.Tensor] = None) -> torch.Tensor
```

Sigma subgrid-scale model (Nicoud et al., 2011).

Based on singular values σ₁ ≥ σ₂ ≥ σ₃ of velocity gradient:

ν_sgs = (C_σ Δ)² σ₃(σ₁ - σ₂)(σ₂ - σ₃) / σ₁²

Properties:
    - Vanishes for pure rotation, pure shear, 2D/axisymmetric flows
    - Correct near-wall scaling
    - Sensitive to 3D turbulent structures

**Parameters:**

- **delta** (`<class 'float'>`): Filter width [m]
- **rho** (`<class 'torch.Tensor'>`): Density [kg/m³]
- **C_sigma** (`<class 'float'>`): Sigma constant

**Returns**: `<class 'torch.Tensor'>` - SGS eddy viscosity μ_sgs [Pa·s]

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\les.py:636](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\les.py#L636)*

### `smagorinsky_viscosity`

```python
def smagorinsky_viscosity(S: torch.Tensor, delta: float, rho: torch.Tensor, C_s: float = 0.17) -> torch.Tensor
```

Classic Smagorinsky subgrid-scale viscosity.

ν_sgs = (C_s Δ)² |S|

**Parameters:**

- **S** (`<class 'torch.Tensor'>`): Strain rate magnitude [1/s]
- **delta** (`<class 'float'>`): Filter width [m]
- **rho** (`<class 'torch.Tensor'>`): Density [kg/m³]
- **C_s** (`<class 'float'>`): Smagorinsky constant

**Returns**: `<class 'torch.Tensor'>` - SGS eddy viscosity μ_sgs [Pa·s]

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\les.py:206](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\les.py#L206)*

### `smagorinsky_with_damping`

```python
def smagorinsky_with_damping(S: torch.Tensor, delta: float, rho: torch.Tensor, y_plus: torch.Tensor, C_s: float = 0.17, A_plus: float = 25.0) -> torch.Tensor
```

Smagorinsky model with Van Driest near-wall damping.

ν_sgs = (C_s Δ D)² |S|

where D = 1 - exp(-y⁺/A⁺)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\les.py:254](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\les.py#L254)*

### `strain_rate_magnitude`

```python
def strain_rate_magnitude(du_dx: torch.Tensor, du_dy: torch.Tensor, dv_dx: torch.Tensor, dv_dy: torch.Tensor, du_dz: Optional[torch.Tensor] = None, dv_dz: Optional[torch.Tensor] = None, dw_dx: Optional[torch.Tensor] = None, dw_dy: Optional[torch.Tensor] = None, dw_dz: Optional[torch.Tensor] = None) -> torch.Tensor
```

Compute strain rate magnitude |S| = √(2 S_ij S_ij).

S_ij = (1/2)(∂u_i/∂x_j + ∂u_j/∂x_i)

**Returns**: `<class 'torch.Tensor'>` - |S| strain rate magnitude [1/s]

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\les.py:128](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\les.py#L128)*

### `test_filter`

```python
def test_filter(field: torch.Tensor, filter_ratio: float = 2.0) -> torch.Tensor
```

Apply test filter for dynamic procedure.

Uses simple box filter at scale Δ̂ = filter_ratio × Δ

**Parameters:**

- **field** (`<class 'torch.Tensor'>`): Field to filter
- **filter_ratio** (`<class 'float'>`): Test-to-grid filter ratio

**Returns**: `<class 'torch.Tensor'>` - Test-filtered field

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\les.py:280](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\les.py#L280)*

### `validate_les`

```python
def validate_les()
```

Run validation tests for LES module.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\les.py:832](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\les.py#L832)*

### `van_driest_damping`

```python
def van_driest_damping(y_plus: torch.Tensor, A_plus: float = 25.0) -> torch.Tensor
```

Van Driest damping function for near-wall correction.

D = 1 - exp(-y⁺/A⁺)

Reduces SGS viscosity near walls where grid resolves
viscous sublayer.

**Parameters:**

- **y_plus** (`<class 'torch.Tensor'>`): Wall distance in wall units
- **A_plus** (`<class 'float'>`): Damping constant (default 25) Default: `25)`.

**Returns**: `<class 'torch.Tensor'>` - Damping factor D ∈ [0, 1]

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\les.py:232](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\les.py#L232)*

### `vorticity_magnitude`

```python
def vorticity_magnitude(du_dx: torch.Tensor, du_dy: torch.Tensor, dv_dx: torch.Tensor, dv_dy: torch.Tensor, du_dz: Optional[torch.Tensor] = None, dv_dz: Optional[torch.Tensor] = None, dw_dx: Optional[torch.Tensor] = None, dw_dy: Optional[torch.Tensor] = None, dw_dz: Optional[torch.Tensor] = None) -> torch.Tensor
```

Compute vorticity magnitude |Ω| = √(2 Ω_ij Ω_ij).

Ω_ij = (1/2)(∂u_i/∂x_j - ∂u_j/∂x_i)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\les.py:170](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\les.py#L170)*

### `vreman_viscosity`

```python
def vreman_viscosity(du_dx: torch.Tensor, du_dy: torch.Tensor, dv_dx: torch.Tensor, dv_dy: torch.Tensor, delta: float, rho: torch.Tensor, C_v: float = 0.07, du_dz: Optional[torch.Tensor] = None, dv_dz: Optional[torch.Tensor] = None, dw_dx: Optional[torch.Tensor] = None, dw_dy: Optional[torch.Tensor] = None, dw_dz: Optional[torch.Tensor] = None) -> torch.Tensor
```

Vreman subgrid-scale model (2004).

Based on the first invariant of the velocity gradient tensor:

ν_sgs = C_v √(B_β / (α_ij α_ij))

where α_ij = ∂u_j/∂x_i, β_ij = Δ²_m α_mi α_mj
and B_β = β_11 β_22 - β_12² + β_11 β_33 - β_13² + β_22 β_33 - β_23²

Advantages:
    - Vanishes in many laminar flows
    - Works well on anisotropic grids
    - Simple and efficient

**Parameters:**

- **delta** (`<class 'float'>`): Filter width [m]
- **rho** (`<class 'torch.Tensor'>`): Density [kg/m³]
- **C_v** (`<class 'float'>`): Vreman constant

**Returns**: `<class 'torch.Tensor'>` - SGS eddy viscosity μ_sgs [Pa·s]

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\les.py:544](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\les.py#L544)*

### `wale_viscosity`

```python
def wale_viscosity(du_dx: torch.Tensor, du_dy: torch.Tensor, dv_dx: torch.Tensor, dv_dy: torch.Tensor, delta: float, rho: torch.Tensor, C_w: float = 0.5, du_dz: Optional[torch.Tensor] = None, dv_dz: Optional[torch.Tensor] = None, dw_dx: Optional[torch.Tensor] = None, dw_dy: Optional[torch.Tensor] = None, dw_dz: Optional[torch.Tensor] = None) -> torch.Tensor
```

WALE (Wall-Adapting Local Eddy-viscosity) model.

Based on the traceless symmetric part of the squared
velocity gradient tensor:

S^d_ij = (1/2)(g²_ij + g²_ji) - (1/3) δ_ij g²_kk

where g_ij = ∂u_i/∂x_j and g²_ij = g_ik g_kj

Advantages:
    - Proper near-wall behavior (ν_sgs ~ y³)
    - No ad-hoc damping functions required
    - Zero in laminar shear flow

**Parameters:**

- **delta** (`<class 'float'>`): Filter width [m]
- **rho** (`<class 'torch.Tensor'>`): Density [kg/m³]
- **C_w** (`<class 'float'>`): WALE constant

**Returns**: `<class 'torch.Tensor'>` - SGS eddy viscosity μ_sgs [Pa·s]

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\les.py:422](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\les.py#L422)*
