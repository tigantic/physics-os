# Module `cfd.turbulence`

RANS Turbulence Modeling ========================

Implements Reynolds-Averaged Navier-Stokes turbulence models
for high-Reynolds number hypersonic flows.

Models:
    1. k-ε (Standard, Realizable)
    2. k-ω SST (Menter's Shear Stress Transport)
    3. Spalart-Allmaras (one-equation)

Key Features:
    - Eddy viscosity hypothesis: τ_t = μ_t (∇u + ∇uᵀ - 2/3 k I)
    - Wall functions for near-wall treatment
    - Compressibility corrections for hypersonic flows
    - Low-Reynolds number damping functions

The RANS equations:
    ∂ρ/∂t + ∇·(ρũ) = 0
    ∂(ρũ)/∂t + ∇·(ρũ⊗ũ) = -∇p̄ + ∇·(τ + τ_t)
    ∂(ρẼ)/∂t + ∇·((ρẼ + p̄)ũ) = ∇·((τ + τ_t)·ũ - q̄ - q_t)

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `TurbulenceModel`(Enum)

Available turbulence models.

### class `TurbulentState`

State for turbulent flow quantities.

For k-ε: k (TKE), epsilon (dissipation)
For k-ω: k (TKE), omega (specific dissipation)
For SA: nu_tilde (modified viscosity)

#### Attributes

- **k** (`typing.Optional[torch.Tensor]`): 
- **epsilon** (`typing.Optional[torch.Tensor]`): 
- **omega** (`typing.Optional[torch.Tensor]`): 
- **nu_tilde** (`typing.Optional[torch.Tensor]`): 
- **mu_t** (`typing.Optional[torch.Tensor]`): 

#### Properties

##### `shape`

```python
def shape(self) -> torch.Size
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py:98](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py#L98)*

#### Methods

##### `__init__`

```python
def __init__(self, k: Optional[torch.Tensor] = None, epsilon: Optional[torch.Tensor] = None, omega: Optional[torch.Tensor] = None, nu_tilde: Optional[torch.Tensor] = None, mu_t: Optional[torch.Tensor] = None) -> None
```

##### `zeros`

```python
def zeros(shape: Tuple[int, int], dtype=torch.float64) -> 'TurbulentState'
```

Create zero-initialized turbulent state with all fields.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py:106](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py#L106)*

## Functions

### `friction_velocity`

```python
def friction_velocity(tau_w: torch.Tensor, rho_w: torch.Tensor) -> torch.Tensor
```

Compute friction velocity u_τ = √(τ_w / ρ).

**Parameters:**

- **tau_w** (`<class 'torch.Tensor'>`): Wall shear stress [Pa]
- **rho_w** (`<class 'torch.Tensor'>`): Wall density [kg/m³]

**Returns**: `<class 'torch.Tensor'>` - Friction velocity [m/s]

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py:161](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py#L161)*

### `initialize_turbulence`

```python
def initialize_turbulence(model: turbulence.TurbulenceModel, rho: torch.Tensor, u: torch.Tensor, mu: torch.Tensor, turbulence_intensity: float = 0.01, viscosity_ratio: float = 10.0) -> turbulence.TurbulentState
```

Initialize turbulent quantities for a given model.

**Parameters:**

- **model** (`<enum 'TurbulenceModel'>`): Turbulence model type
- **rho** (`<class 'torch.Tensor'>`): Density field [kg/m³]
- **u** (`<class 'torch.Tensor'>`): Velocity field [m/s]
- **mu** (`<class 'torch.Tensor'>`): Molecular viscosity [Pa·s]
- **turbulence_intensity** (`<class 'float'>`): Tu = u'/U
- **viscosity_ratio** (`<class 'float'>`): μ_t / μ

**Returns**: `<class 'turbulence.TurbulentState'>` - TurbulentState with initialized fields

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py:593](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py#L593)*

### `k_epsilon_eddy_viscosity`

```python
def k_epsilon_eddy_viscosity(rho: torch.Tensor, k: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor
```

Compute eddy viscosity for k-ε model.

μ_t = ρ C_μ k² / ε

**Parameters:**

- **rho** (`<class 'torch.Tensor'>`): Density [kg/m³]
- **k** (`<class 'torch.Tensor'>`): Turbulent kinetic energy [m²/s²]
- **epsilon** (`<class 'torch.Tensor'>`): Dissipation rate [m²/s³]

**Returns**: `<class 'torch.Tensor'>` - Eddy viscosity [Pa·s]

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py:182](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py#L182)*

### `k_epsilon_production`

```python
def k_epsilon_production(mu_t: torch.Tensor, du_dx: torch.Tensor, du_dy: torch.Tensor, dv_dx: torch.Tensor, dv_dy: torch.Tensor) -> torch.Tensor
```

Compute turbulence production term P_k.

P_k = μ_t * S² where S² = 2(S_ij S_ij)

For 2D: S² = 2[(∂u/∂x)² + (∂v/∂y)² + 0.5(∂u/∂y + ∂v/∂x)²]

**Parameters:**

- **mu_t** (`<class 'torch.Tensor'>`): Eddy viscosity [Pa·s] du_dx, du_dy, dv_dx, dv_dy: Velocity gradients [1/s]

**Returns**: `<class 'torch.Tensor'>` - Production rate [kg/(m·s³)]

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py:203](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py#L203)*

### `k_epsilon_source`

```python
def k_epsilon_source(rho: torch.Tensor, k: torch.Tensor, epsilon: torch.Tensor, P_k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```

Compute source terms for k and ε equations.

S_k = P_k - ρε
S_ε = C_ε1 (ε/k) P_k - C_ε2 ρ ε²/k

**Parameters:**

- **rho** (`<class 'torch.Tensor'>`): Density [kg/m³]
- **k** (`<class 'torch.Tensor'>`): TKE [m²/s²]
- **epsilon** (`<class 'torch.Tensor'>`): Dissipation [m²/s³]
- **P_k** (`<class 'torch.Tensor'>`): Production [kg/(m·s³)]

**Returns**: `typing.Tuple[torch.Tensor, torch.Tensor]` - Tuple of (S_k, S_epsilon) source terms

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py:233](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py#L233)*

### `k_omega_blending`

```python
def k_omega_blending(d: torch.Tensor, k: torch.Tensor, omega: torch.Tensor, rho: torch.Tensor, mu: torch.Tensor, dk_dx: torch.Tensor, dk_dy: torch.Tensor, domega_dx: torch.Tensor, domega_dy: torch.Tensor) -> torch.Tensor
```

Compute SST blending function F1.

Blends between k-ω (near wall, F1→1) and k-ε (freestream, F1→0).

**Parameters:**

- **d** (`<class 'torch.Tensor'>`): Wall distance [m] k, omega: Turbulent quantities
- **rho** (`<class 'torch.Tensor'>`): Density [kg/m³]
- **mu** (`<class 'torch.Tensor'>`): Molecular viscosity [Pa·s] dk_dx, dk_dy: k gradients domega_dx, domega_dy: ω gradients

**Returns**: `<class 'torch.Tensor'>` - Blending function F1 ∈ [0, 1]

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py:268](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py#L268)*

### `k_omega_sst_eddy_viscosity`

```python
def k_omega_sst_eddy_viscosity(rho: torch.Tensor, k: torch.Tensor, omega: torch.Tensor, F2: Optional[torch.Tensor] = None, S: Optional[torch.Tensor] = None) -> torch.Tensor
```

Compute SST eddy viscosity with vorticity limiter.

μ_t = ρ a₁ k / max(a₁ ω, S F₂)

If F2 and S are not provided, uses simplified formula:
μ_t = ρ k / ω

**Parameters:**

- **rho** (`<class 'torch.Tensor'>`): Density [kg/m³]
- **k** (`<class 'torch.Tensor'>`): TKE [m²/s²]
- **omega** (`<class 'torch.Tensor'>`): Specific dissipation [1/s]
- **F2** (`typing.Optional[torch.Tensor]`): Second blending function (optional)
- **S** (`typing.Optional[torch.Tensor]`): Strain rate magnitude [1/s] (optional)

**Returns**: `<class 'torch.Tensor'>` - Eddy viscosity [Pa·s]

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py:315](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py#L315)*

### `k_omega_sst_source`

```python
def k_omega_sst_source(rho: torch.Tensor, k: torch.Tensor, omega: torch.Tensor, P_k: torch.Tensor, F1: torch.Tensor, dk_dx: torch.Tensor, dk_dy: torch.Tensor, domega_dx: torch.Tensor, domega_dy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```

Compute source terms for k-ω SST model.

**Parameters:**

- **rho** (`<class 'torch.Tensor'>`): Density [kg/m³] k, omega: Turbulent quantities
- **P_k** (`<class 'torch.Tensor'>`): Production term
- **F1** (`<class 'torch.Tensor'>`): Blending function
- **Gradients**: For cross-diffusion term

**Returns**: `typing.Tuple[torch.Tensor, torch.Tensor]` - Tuple of (S_k, S_omega) source terms

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py:394](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py#L394)*

### `log_law_velocity`

```python
def log_law_velocity(y_plus: torch.Tensor, kappa: float = 0.41, B: float = 5.2) -> torch.Tensor
```

Compute u⁺ from log-law.

u⁺ = (1/κ) ln(y⁺) + B    for y⁺ > 11.6
u⁺ = y⁺                  for y⁺ ≤ 11.6 (viscous sublayer)

**Parameters:**

- **y_plus** (`<class 'torch.Tensor'>`): Dimensionless wall distance
- **kappa** (`<class 'float'>`): Von Karman constant
- **B** (`<class 'float'>`): Log-law intercept

**Returns**: `<class 'torch.Tensor'>` - u⁺ dimensionless velocity

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py:517](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py#L517)*

### `sarkar_correction`

```python
def sarkar_correction(k: torch.Tensor, T: torch.Tensor, epsilon: Optional[torch.Tensor] = None, gamma: float = 1.4, R: float = 287.0) -> torch.Tensor
```

Sarkar compressibility correction for high-Mach flows.

Adds dilatation dissipation: ε_d = α_1 M_t² ε
where M_t = √(2k) / a is turbulent Mach number.

**Parameters:**

- **k** (`<class 'torch.Tensor'>`): TKE [m²/s²]
- **T** (`<class 'torch.Tensor'>`): Temperature [K]
- **epsilon** (`typing.Optional[torch.Tensor]`): Dissipation [m²/s³] (optional, returns M_t^2 if None)
- **gamma** (`<class 'float'>`): Specific heat ratio
- **R** (`<class 'float'>`): Gas constant [J/(kg·K)]

**Returns**: `<class 'torch.Tensor'>` - Additional dissipation term (or M_t^2 if epsilon is None)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py:659](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py#L659)*

### `spalart_allmaras_eddy_viscosity`

```python
def spalart_allmaras_eddy_viscosity(rho: torch.Tensor, nu_tilde: torch.Tensor, nu: torch.Tensor) -> torch.Tensor
```

Compute eddy viscosity for SA model.

μ_t = ρ ν̃ f_v1
f_v1 = χ³ / (χ³ + c_v1³)
χ = ν̃ / ν

**Parameters:**

- **rho** (`<class 'torch.Tensor'>`): Density [kg/m³]
- **nu_tilde** (`<class 'torch.Tensor'>`): Modified viscosity [m²/s]
- **nu** (`<class 'torch.Tensor'>`): Molecular viscosity [m²/s]

**Returns**: `<class 'torch.Tensor'>` - Eddy viscosity [Pa·s]

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py:441](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py#L441)*

### `spalart_allmaras_source`

```python
def spalart_allmaras_source(rho: torch.Tensor, nu_tilde: torch.Tensor, nu: torch.Tensor, d: torch.Tensor, S: torch.Tensor) -> torch.Tensor
```

Compute source term for SA model.

Source = c_b1 S̃ ν̃ - c_w1 f_w (ν̃/d)²

**Parameters:**

- **rho** (`<class 'torch.Tensor'>`): Density [kg/m³]
- **nu_tilde** (`<class 'torch.Tensor'>`): Modified viscosity [m²/s]
- **nu** (`<class 'torch.Tensor'>`): Molecular viscosity [m²/s]
- **d** (`<class 'torch.Tensor'>`): Wall distance [m]
- **S** (`<class 'torch.Tensor'>`): Strain rate magnitude [1/s]

**Returns**: `<class 'torch.Tensor'>` - Source term for ν̃ equation

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py:467](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py#L467)*

### `sst_blending_functions`

```python
def sst_blending_functions(k: torch.Tensor, omega: torch.Tensor, y: torch.Tensor, rho: torch.Tensor, mu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```

Compute SST blending functions F1 and F2.

Simplified version without gradient terms.

**Parameters:**

- **k** (`<class 'torch.Tensor'>`): TKE [m²/s²]
- **omega** (`<class 'torch.Tensor'>`): Specific dissipation [1/s]
- **y** (`<class 'torch.Tensor'>`): Wall distance [m]
- **rho** (`<class 'torch.Tensor'>`): Density [kg/m³]
- **mu** (`<class 'torch.Tensor'>`): Molecular viscosity [Pa·s]

**Returns**: `typing.Tuple[torch.Tensor, torch.Tensor]` - Tuple of (F1, F2) blending functions

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py:352](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py#L352)*

### `validate_turbulence`

```python
def validate_turbulence()
```

Run validation tests for turbulence module.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py:733](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py#L733)*

### `wall_distance`

```python
def wall_distance(Ny: int, Nx: int, dy: float, wall_j: int = 0) -> torch.Tensor
```

Compute distance from wall for each grid point.

**Parameters:**

- **dy** (`<class 'float'>`): Grid spacing in y
- **wall_j** (`<class 'int'>`): Wall location index (default 0 = bottom) Default: `0 = bottom)`.

**Returns**: `<class 'torch.Tensor'>` - Wall distance field [m]

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py:118](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py#L118)*

### `wall_function_tau`

```python
def wall_function_tau(rho: torch.Tensor, u_parallel: torch.Tensor, y: torch.Tensor, mu: torch.Tensor, kappa: float = 0.41, B: float = 5.2, max_iter: int = 10) -> torch.Tensor
```

Compute wall shear stress using wall functions.

Iteratively solves for u_τ using log-law.

**Parameters:**

- **rho** (`<class 'torch.Tensor'>`): Near-wall density [kg/m³]
- **u_parallel** (`<class 'torch.Tensor'>`): Near-wall velocity magnitude [m/s]
- **y** (`<class 'torch.Tensor'>`): Wall distance [m]
- **mu** (`<class 'torch.Tensor'>`): Dynamic viscosity [Pa·s]
- **kappa** (`<class 'float'>`): Von Karman constant
- **B** (`<class 'float'>`): Log-law intercept
- **max_iter** (`<class 'int'>`): Newton iterations

**Returns**: `<class 'torch.Tensor'>` - Wall shear stress [Pa]

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py:545](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py#L545)*

### `wilcox_compressibility`

```python
def wilcox_compressibility(k: torch.Tensor, T: torch.Tensor, omega: Optional[torch.Tensor] = None, gamma: float = 1.4, R: float = 287.0) -> Tuple[torch.Tensor, torch.Tensor]
```

Wilcox compressibility correction for k-ω.

Modifies β* based on turbulent Mach number.

**Parameters:**

- **k** (`<class 'torch.Tensor'>`): TKE [m²/s²]
- **T** (`<class 'torch.Tensor'>`): Temperature [K]
- **omega** (`typing.Optional[torch.Tensor]`): Specific dissipation [1/s] (optional)
- **gamma** (`<class 'float'>`): Specific heat ratio
- **R** (`<class 'float'>`): Gas constant [J/(kg·K)]

**Returns**: `typing.Tuple[torch.Tensor, torch.Tensor]` - Tuple of (beta_star_modified, F_Mt)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py:696](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py#L696)*

### `y_plus`

```python
def y_plus(rho: torch.Tensor, u_tau: torch.Tensor, y: torch.Tensor, mu: torch.Tensor) -> torch.Tensor
```

Compute dimensionless wall distance y⁺ = ρ u_τ y / μ.

**Parameters:**

- **rho** (`<class 'torch.Tensor'>`): Density [kg/m³]
- **u_tau** (`<class 'torch.Tensor'>`): Friction velocity [m/s]
- **y** (`<class 'torch.Tensor'>`): Wall distance [m]
- **mu** (`<class 'torch.Tensor'>`): Dynamic viscosity [Pa·s]

**Returns**: `<class 'torch.Tensor'>` - y⁺ dimensionless wall distance

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py:140](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\turbulence.py#L140)*
