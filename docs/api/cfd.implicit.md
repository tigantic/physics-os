# Module `cfd.implicit`

Implicit Time Integration for Stiff Systems ============================================

Provides backward Euler and BDF-2 schemes for integrating
stiff ODEs arising from finite-rate chemistry.

Key Features:
    - Newton iteration for nonlinear solve
    - Analytical and numerical Jacobian options
    - Line search for robustness
    - Adaptive substepping for very stiff problems

Problem:
    dY/dt = ω(Y, T) / ρ

The chemistry source term ω can vary by 10+ orders of magnitude,
making explicit methods require tiny timesteps. Implicit methods
allow larger timesteps at the cost of solving nonlinear systems.

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `AdaptiveImplicit`

Adaptive implicit integrator with error control.

Uses embedded BDF-1/BDF-2 for error estimation.

#### Methods

##### `__init__`

```python
def __init__(self, config: implicit.ImplicitConfig = None, rtol: float = 0.0001, atol: float = 1e-10)
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\implicit.py:343](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\implicit.py#L343)*

##### `integrate`

```python
def integrate(self, y0: torch.Tensor, f: Callable[[torch.Tensor], torch.Tensor], dt: float) -> Tuple[torch.Tensor, float, int]
```

Integrate with adaptive substepping.

**Parameters:**

- **y0** (`<class 'torch.Tensor'>`): Initial state
- **f** (`typing.Callable[[torch.Tensor], torch.Tensor]`): RHS function f(y) -> dy/dt
- **dt** (`<class 'float'>`): Target timestep

**Returns**: `typing.Tuple[torch.Tensor, float, int]` - Tuple of (y_new, actual_dt, n_substeps)

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\implicit.py:353](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\implicit.py#L353)*

### class `ChemistryIntegrator`

Implicit integrator for chemistry source terms.

Solves: dY/dt = ω(Y, T) / ρ
using backward Euler.

#### Attributes

- **config** (`<class 'implicit.ImplicitConfig'>`): 
- **rho** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, config: implicit.ImplicitConfig, rho: float = 1.0) -> None
```

##### `integrate`

```python
def integrate(self, Y0: torch.Tensor, T: float, omega_fn: Callable[[torch.Tensor, float], torch.Tensor], dt: float) -> Tuple[torch.Tensor, implicit.SolverStatus]
```

Integrate chemistry one timestep using backward Euler.

Y^{n+1} = Y^n + dt * ω(Y^{n+1}, T) / ρ

Residual: F(Y) = Y - Y^n - dt * ω(Y, T) / ρ = 0

**Parameters:**

- **Y0** (`<class 'torch.Tensor'>`): Initial mass fractions [n_species]
- **T** (`<class 'float'>`): Temperature [K] (assumed constant over dt)
- **omega_fn** (`typing.Callable[[torch.Tensor, float], torch.Tensor]`): Function (Y, T) -> production rates
- **dt** (`<class 'float'>`): Timestep [s]

**Returns**: `typing.Tuple[torch.Tensor, implicit.SolverStatus]` - Tuple of (Y_new, status)

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\implicit.py:193](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\implicit.py#L193)*

### class `ImplicitConfig`

Configuration for implicit integrator.

#### Attributes

- **max_newton_iters** (`<class 'int'>`): 
- **newton_tol** (`<class 'float'>`): 
- **line_search** (`<class 'bool'>`): 
- **line_search_max_iters** (`<class 'int'>`): 
- **line_search_alpha** (`<class 'float'>`): 
- **jacobian_numerical** (`<class 'bool'>`): 
- **jacobian_eps** (`<class 'float'>`): 
- **adaptive_substep** (`<class 'bool'>`): 
- **min_substeps** (`<class 'int'>`): 
- **max_substeps** (`<class 'int'>`): 
- **substep_safety** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, max_newton_iters: int = 20, newton_tol: float = 1e-08, line_search: bool = True, line_search_max_iters: int = 10, line_search_alpha: float = 0.0001, jacobian_numerical: bool = True, jacobian_eps: float = 1e-06, adaptive_substep: bool = True, min_substeps: int = 1, max_substeps: int = 100, substep_safety: float = 0.5) -> None
```

### class `NewtonResult`

Result from Newton iteration.

#### Attributes

- **x** (`<class 'torch.Tensor'>`): 
- **status** (`<enum 'SolverStatus'>`): 
- **iterations** (`<class 'int'>`): 
- **residual_norm** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, x: torch.Tensor, status: implicit.SolverStatus, iterations: int, residual_norm: float) -> None
```

### class `SolverStatus`(Enum)

Status of nonlinear solver.

## Functions

### `backward_euler_scalar`

```python
def backward_euler_scalar(y0: float, f: Callable[[float, float], float], t0: float, dt: float, tol: float = 1e-10, max_iter: int = 20) -> float
```

Backward Euler for scalar ODE: dy/dt = f(t, y)

y^{n+1} = y^n + dt * f(t^{n+1}, y^{n+1})

**Parameters:**

- **y0** (`<class 'float'>`): Initial value
- **f** (`typing.Callable[[float, float], float]`): RHS function f(t, y)
- **t0** (`<class 'float'>`): Initial time
- **dt** (`<class 'float'>`): Timestep
- **tol** (`<class 'float'>`): Newton convergence tolerance
- **max_iter** (`<class 'int'>`): Maximum Newton iterations

**Returns**: `<class 'float'>` - y at t0 + dt

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\implicit.py:246](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\implicit.py#L246)*

### `bdf2_scalar`

```python
def bdf2_scalar(y0: float, y1: float, f: Callable[[float, float], float], t1: float, dt: float, tol: float = 1e-10, max_iter: int = 20) -> float
```

BDF-2 for scalar ODE: dy/dt = f(t, y)

y^{n+2} = (4/3) y^{n+1} - (1/3) y^n + (2/3) dt * f(t^{n+2}, y^{n+2})

**Parameters:**

- **y0** (`<class 'float'>`): Value at t - dt
- **y1** (`<class 'float'>`): Value at t
- **f** (`typing.Callable[[float, float], float]`): RHS function f(t, y)
- **t1** (`<class 'float'>`): Time at y1
- **dt** (`<class 'float'>`): Timestep
- **tol** (`<class 'float'>`): Newton convergence tolerance
- **max_iter** (`<class 'int'>`): Maximum Newton iterations

**Returns**: `<class 'float'>` - y at t1 + dt

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\implicit.py:290](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\implicit.py#L290)*

### `newton_solve`

```python
def newton_solve(residual_fn: Callable[[torch.Tensor], torch.Tensor], jacobian_fn: Callable[[torch.Tensor], torch.Tensor], x0: torch.Tensor, config: implicit.ImplicitConfig) -> implicit.NewtonResult
```

Solve F(x) = 0 using Newton's method with optional line search.

Newton iteration: x_{n+1} = x_n - J^{-1} F(x_n)

**Parameters:**

- **residual_fn** (`typing.Callable[[torch.Tensor], torch.Tensor]`): Function computing F(x)
- **jacobian_fn** (`typing.Callable[[torch.Tensor], torch.Tensor]`): Function computing Jacobian J = dF/dx
- **x0** (`<class 'torch.Tensor'>`): Initial guess
- **config** (`<class 'implicit.ImplicitConfig'>`): Solver configuration

**Returns**: `<class 'implicit.NewtonResult'>` - NewtonResult with solution and status

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\implicit.py:67](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\implicit.py#L67)*

### `numerical_jacobian`

```python
def numerical_jacobian(func: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, eps: float = 1e-06) -> torch.Tensor
```

Compute Jacobian using finite differences.

J_ij = d F_i / d x_j ≈ (F_i(x + eps*e_j) - F_i(x)) / eps

**Parameters:**

- **func** (`typing.Callable[[torch.Tensor], torch.Tensor]`): Function F: R^n -> R^m
- **x** (`<class 'torch.Tensor'>`): Point to evaluate Jacobian
- **eps** (`<class 'float'>`): Finite difference step

**Returns**: `<class 'torch.Tensor'>` - Jacobian matrix [m, n]

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\implicit.py:149](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\implicit.py#L149)*

### `validate_implicit`

```python
def validate_implicit()
```

Run validation tests for implicit integrator.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\implicit.py:428](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\implicit.py#L428)*
