# Module `cfd.euler_1d`

1D Euler Equations for Compressible Flow =========================================

The 1D Euler equations are the fundamental conservation laws for
compressible inviscid flow:

    ∂U/∂t + ∂F(U)/∂x = 0

where U = [ρ, ρu, E]ᵀ is the vector of conserved variables:
    - ρ: density
    - ρu: momentum
    - E: total energy

and F(U) is the flux vector:
    F = [ρu, ρu² + p, (E + p)u]ᵀ

The pressure p is determined by the equation of state:
    p = (γ - 1)(E - ½ρu²)

where γ is the ratio of specific heats (γ = 1.4 for air).

Tensor Network Approach
-----------------------
We discretize the solution on a grid and represent the state as an MPS.
Each site corresponds to a spatial grid point, with physical dimension
d = n_vars (number of conserved variables, typically 3).

For hypersonic flows with shocks, we need:
1. Shock-capturing schemes (Godunov-type)
2. Adaptive bond dimension for discontinuity resolution
3. Characteristic-based MPO operators

This module implements the foundation for tensor network CFD.

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `BCType1D`(Enum)

Boundary condition types for 1D Euler equations.

### class `Euler1D`

1D Euler equation solver using tensor networks.

Discretization: Finite volume with N cells on domain [x_min, x_max].

The solution is stored as an MPS where each site represents a cell,
and the physical dimension is 3 (for ρ, ρu, E).

For now, we use a simple product state (χ=1) representation,
which is equivalent to classical FVM. The tensor network structure
enables future extensions:
- Entangled representations for multi-scale features
- MPO-based flux operators
- Automatic adaptivity via bond dimension

#### Methods

##### `__init__`

```python
def __init__(self, N: int, x_min: float = 0.0, x_max: float = 1.0, gamma: float = 1.4, cfl: float = 0.5, dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None)
```

Initialize Euler solver.

**Parameters:**

- **N** (`<class 'int'>`): Number of grid cells x_min, x_max: Domain bounds
- **gamma** (`<class 'float'>`): Ratio of specific heats
- **cfl** (`<class 'float'>`): CFL number for time stepping
- **dtype** (`<class 'torch.dtype'>`): Data type
- **device** (`typing.Optional[torch.device]`): Device

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py:167](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py#L167)*

##### `compute_dt`

```python
def compute_dt(self) -> float
```

Compute time step from CFL condition.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py:299](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py#L299)*

##### `compute_flux`

```python
def compute_flux(self, U_L: torch.Tensor, U_R: torch.Tensor) -> torch.Tensor
```

Compute numerical flux at cell interface using Rusanov (local Lax-Friedrichs).

F_{i+1/2} = ½(F_L + F_R) - ½ λ_max (U_R - U_L)

**Parameters:**

- **U_L** (`<class 'torch.Tensor'>`): Left state (batch, 3)
- **U_R** (`<class 'torch.Tensor'>`): Right state (batch, 3)

**Returns**: `<class 'torch.Tensor'>` - Flux (batch, 3)

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py:244](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py#L244)*

##### `from_mps`

```python
def from_mps(mps: tensornet.core.mps.MPS, x_min: float = 0.0, x_max: float = 1.0, gamma: float = 1.4, cfl: float = 0.5) -> 'Euler1D'
```

Create solver from MPS state.

**Parameters:**

- **mps** (`<class 'tensornet.core.mps.MPS'>`): MPS with physical dimension 3 x_min, x_max: Domain bounds
- **gamma** (`<class 'float'>`): Ratio of specific heats
- **cfl** (`<class 'float'>`): CFL number

**Returns**: `<class 'euler_1d.Euler1D'>` - Euler1D solver with state from MPS

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py:471](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py#L471)*

##### `set_boundary_conditions`

```python
def set_boundary_conditions(self, left: euler_1d.BCType1D = <BCType1D.TRANSMISSIVE: 'transmissive'>, right: euler_1d.BCType1D = <BCType1D.TRANSMISSIVE: 'transmissive'>) -> 'Euler1D'
```

Set boundary conditions.

**Parameters:**

- **left** (`<enum 'BCType1D'>`): Left boundary condition type
- **right** (`<enum 'BCType1D'>`): Right boundary condition type

**Returns**: `<class 'euler_1d.Euler1D'>` - self for method chaining

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py:219](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py#L219)*

##### `set_initial_condition`

```python
def set_initial_condition(self, state: euler_1d.EulerState)
```

Set initial condition.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py:238](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py#L238)*

##### `solve`

```python
def solve(self, t_final: float, callback: Optional[Callable[[ForwardRef('Euler1D'), float], NoneType]] = None, callback_interval: float = 0.0) -> List[Tuple[float, euler_1d.EulerState]]
```

Solve to final time.

**Parameters:**

- **t_final** (`<class 'float'>`): Final time
- **callback** (`typing.Optional[typing.Callable[[euler_1d.Euler1D, float], NoneType]]`): Optional callback(solver, t) called periodically
- **callback_interval** (`<class 'float'>`): Interval for callback (0 = every step)

**Returns**: `typing.List[typing.Tuple[float, euler_1d.EulerState]]` - List of (time, state) snapshots

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py:394](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py#L394)*

##### `step`

```python
def step(self, dt: Optional[float] = None) -> float
```

Advance solution by one time step.

Uses first-order forward Euler in time with
Rusanov flux in space.

**Parameters:**

- **dt** (`typing.Optional[float]`): Time step (computed from CFL if None)

**Returns**: `<class 'float'>` - Actual dt used

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py:316](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py#L316)*

##### `to_mps`

```python
def to_mps(self, chi_max: int = 1) -> tensornet.core.mps.MPS
```

Convert current state to MPS representation.

Each site is a grid cell with physical dimension 3 (ρ, ρu, E).

For chi_max=1 (product state), this is equivalent to classical FVM.
Larger chi enables entanglement for multi-scale representation.

**Parameters:**

- **chi_max** (`<class 'int'>`): Maximum bond dimension

**Returns**: `<class 'tensornet.core.mps.MPS'>` - MPS representation of state

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py:443](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py#L443)*

### class `EulerState`

Container for Euler equation state variables.

#### Attributes

- **rho** (`<class 'torch.Tensor'>`): 
- **rho_u** (`<class 'torch.Tensor'>`): 
- **E** (`<class 'torch.Tensor'>`): 
- **gamma** (`<class 'float'>`): 

#### Properties

##### `M`

```python
def M(self) -> torch.Tensor
```

Mach number.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py:88](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py#L88)*

##### `N`

```python
def N(self) -> int
```

Number of grid points.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py:62](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py#L62)*

##### `T`

```python
def T(self) -> torch.Tensor
```

Temperature (assuming ideal gas, R = 1).

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py:78](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py#L78)*

##### `a`

```python
def a(self) -> torch.Tensor
```

Speed of sound.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py:83](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py#L83)*

##### `p`

```python
def p(self) -> torch.Tensor
```

Pressure from equation of state.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py:72](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py#L72)*

##### `u`

```python
def u(self) -> torch.Tensor
```

Velocity.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py:67](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py#L67)*

#### Methods

##### `__init__`

```python
def __init__(self, rho: torch.Tensor, rho_u: torch.Tensor, E: torch.Tensor, gamma: float) -> None
```

##### `from_conserved`

```python
def from_conserved(U: torch.Tensor, gamma: float = 1.4) -> 'EulerState'
```

Create from conserved variable tensor (N, 3).

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py:97](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py#L97)*

##### `from_primitive`

```python
def from_primitive(rho: torch.Tensor, u: torch.Tensor, p: torch.Tensor, gamma: float = 1.4) -> 'EulerState'
```

Create from primitive variables (ρ, u, p).

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py:107](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py#L107)*

##### `to_conserved`

```python
def to_conserved(self) -> torch.Tensor
```

Stack conserved variables: (N, 3).

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py:93](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py#L93)*

## Functions

### `euler_to_mps`

```python
def euler_to_mps(state: euler_1d.EulerState) -> tensornet.core.mps.MPS
```

Convert EulerState to MPS representation.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py:637](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py#L637)*

### `lax_shock_tube_ic`

```python
def lax_shock_tube_ic(N: int, x_min: float = 0.0, x_max: float = 1.0, x_discontinuity: float = 0.5, gamma: float = 1.4, dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None) -> euler_1d.EulerState
```

Lax shock tube initial condition.

More challenging than Sod with higher pressure ratio.

Left state: ρ = 0.445, u = 0.698, p = 3.528
Right state: ρ = 0.5, u = 0, p = 0.571

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py:560](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py#L560)*

### `mps_to_euler`

```python
def mps_to_euler(mps: tensornet.core.mps.MPS, gamma: float = 1.4) -> euler_1d.EulerState
```

Convert MPS to EulerState.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py:644](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py#L644)*

### `shu_osher_ic`

```python
def shu_osher_ic(N: int, x_min: float = -5.0, x_max: float = 5.0, gamma: float = 1.4, dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None) -> euler_1d.EulerState
```

Shu-Osher problem initial condition.

Shock interacting with a sine wave in density.
Tests shock-capturing + oscillatory feature resolution.

Left of x=-4: Post-shock state (ρ=3.857, u=2.629, p=10.333)
Right of x=-4: Pre-shock with density perturbation

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py:596](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py#L596)*

### `sod_shock_tube_ic`

```python
def sod_shock_tube_ic(N: int, x_min: float = 0.0, x_max: float = 1.0, x_discontinuity: float = 0.5, gamma: float = 1.4, dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None) -> euler_1d.EulerState
```

Sod shock tube initial condition.

Classic test problem with exact solution available.

Left state (x < 0.5): ρ = 1, u = 0, p = 1
Right state (x > 0.5): ρ = 0.125, u = 0, p = 0.1

Features: rarefaction, contact, shock

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py:523](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\euler_1d.py#L523)*
