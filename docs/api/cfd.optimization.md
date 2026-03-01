# Module `cfd.optimization`

Shape Optimization for Hypersonic Vehicles ===========================================

Gradient-based optimization using adjoint sensitivities to design
optimal aerodynamic shapes for hypersonic flight conditions.

Key Features:
    - Parameterized geometry representation (B-splines, FFD)
    - Gradient computation via adjoint method
    - Multiple optimization algorithms (steepest descent, L-BFGS, SQP)
    - Constraint handling (volume, thickness, manufacturability)
    - Multi-objective capabilities

Design Variables:
    - Surface node positions
    - B-spline control points
    - Free-Form Deformation (FFD) box vertices
    - Parametric shape descriptors

Objective Functions:
    - Minimize drag: C_D
    - Minimize heating: ∫q_w dS
    - Maximize L/D
    - Minimize heating subject to L/D constraint

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `BSplineParameterization`(GeometryParameterization)

B-spline curve/surface parameterization.

X(u) = Σ N_i(u) P_i

where N_i are B-spline basis functions and P_i are control points.

#### Methods

##### `__init__`

```python
def __init__(self, n_control_points: int, degree: int = 3, n_eval_points: int = 100)
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\optimization.py:126](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\optimization.py#L126)*

##### `evaluate`

```python
def evaluate(self, alpha: torch.Tensor) -> torch.Tensor
```

Evaluate B-spline curve from control points.

**Parameters:**

- **alpha** (`<class 'torch.Tensor'>`): Control points (n_control * 2,) [x0, y0, x1, y1, ...]

**Returns**: `<class 'torch.Tensor'>` - Curve coordinates (n_eval, 2)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\optimization.py:186](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\optimization.py#L186)*

##### `gradient`

```python
def gradient(self, alpha: torch.Tensor) -> torch.Tensor
```

Compute dX/dα.

**Returns**: `<class 'torch.Tensor'>` - Jacobian (n_eval * 2, n_control * 2)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\optimization.py:204](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\optimization.py#L204)*

### class `ConstraintSpec`

Specification for a design constraint.

#### Attributes

- **name** (`<class 'str'>`): 
- **function** (`typing.Callable[[torch.Tensor], torch.Tensor]`): 
- **gradient** (`typing.Callable[[torch.Tensor], torch.Tensor]`): 
- **type** (`<class 'str'>`): 
- **value** (`<class 'float'>`): 
- **penalty_weight** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, name: str, function: Callable[[torch.Tensor], torch.Tensor], gradient: Callable[[torch.Tensor], torch.Tensor], type: str = 'inequality', value: float = 0.0, penalty_weight: float = 100.0) -> None
```

### class `ConstraintType`(Enum)

Constraint handling method.

### class `FFDParameterization`(GeometryParameterization)

Free-Form Deformation (FFD) box parameterization.

Embeds the geometry in a parametric volume and deforms
by moving control vertices.

stu_coords = local coordinates in FFD box
P_deformed = Σ B(s,t,u) ΔP_ijk

#### Methods

##### `__init__`

```python
def __init__(self, box_origin: Tuple[float, float], box_size: Tuple[float, float], n_control: Tuple[int, int], surface_coords: torch.Tensor)
```

Args:

box_origin: (x0, y0) of FFD box
    box_size: (Lx, Ly) of FFD box
    n_control: (ni, nj) number of control points
    surface_coords: (N, 2) original surface coordinates

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\optimization.py:239](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\optimization.py#L239)*

##### `evaluate`

```python
def evaluate(self, alpha: torch.Tensor) -> torch.Tensor
```

Apply FFD deformation.

**Parameters:**

- **alpha** (`<class 'torch.Tensor'>`): Control point displacements (ni * nj * 2,)

**Returns**: `<class 'torch.Tensor'>` - Deformed surface (N, 2)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\optimization.py:283](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\optimization.py#L283)*

##### `gradient`

```python
def gradient(self, alpha: torch.Tensor) -> torch.Tensor
```

Compute dX/dα (sensitivity to control point movements).

**Returns**: `<class 'torch.Tensor'>` - Jacobian (N * 2, ni * nj * 2)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\optimization.py:313](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\optimization.py#L313)*

### class `GeometryParameterization`

Base class for geometry parameterization.

Maps design variables α to surface mesh coordinates X.

#### Methods

##### `__init__`

```python
def __init__(self, n_design_vars: int)
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\optimization.py:105](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\optimization.py#L105)*

##### `evaluate`

```python
def evaluate(self, alpha: torch.Tensor) -> torch.Tensor
```

Map design variables to surface coordinates.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\optimization.py:108](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\optimization.py#L108)*

##### `gradient`

```python
def gradient(self, alpha: torch.Tensor) -> torch.Tensor
```

Compute dX/dα (Jacobian).

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\optimization.py:112](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\optimization.py#L112)*

### class `OptimizationConfig`

Configuration for shape optimization.

#### Attributes

- **optimizer** (`<enum 'OptimizerType'>`): 
- **max_iterations** (`<class 'int'>`): 
- **tolerance** (`<class 'float'>`): 
- **step_size** (`<class 'float'>`): 
- **line_search** (`<class 'bool'>`): 
- **ls_max_iter** (`<class 'int'>`): 
- **ls_c1** (`<class 'float'>`): 
- **ls_c2** (`<class 'float'>`): 
- **gradient_smoothing** (`<class 'bool'>`): 
- **smoothing_iterations** (`<class 'int'>`): 
- **smoothing_weight** (`<class 'float'>`): 
- **history_size** (`<class 'int'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, optimizer: optimization.OptimizerType = <OptimizerType.LBFGS: 'lbfgs'>, max_iterations: int = 100, tolerance: float = 1e-06, step_size: float = 0.01, line_search: bool = True, ls_max_iter: int = 10, ls_c1: float = 0.0001, ls_c2: float = 0.9, gradient_smoothing: bool = True, smoothing_iterations: int = 5, smoothing_weight: float = 0.5, history_size: int = 10) -> None
```

### class `OptimizationResult`

Result from optimization run.

#### Attributes

- **design_variables** (`<class 'torch.Tensor'>`): 
- **objective_value** (`<class 'float'>`): 
- **constraint_values** (`typing.Dict[str, float]`): 
- **gradient_norm** (`<class 'float'>`): 
- **converged** (`<class 'bool'>`): 
- **iterations** (`<class 'int'>`): 
- **history** (`typing.List[typing.Dict]`): 

#### Methods

##### `__init__`

```python
def __init__(self, design_variables: torch.Tensor, objective_value: float, constraint_values: Dict[str, float], gradient_norm: float, converged: bool, iterations: int, history: List[Dict]) -> None
```

### class `OptimizerType`(Enum)

Optimization algorithm selection.

### class `ShapeOptimizer`

Main shape optimization driver.

Combines:
    - Geometry parameterization
    - Flow solver (Euler/NS)
    - Adjoint solver for gradients
    - Optimization algorithm

#### Methods

##### `__init__`

```python
def __init__(self, parameterization: optimization.GeometryParameterization, flow_solver: Callable[[torch.Tensor], Dict], adjoint_solver: Callable[[torch.Tensor, Dict], torch.Tensor], objective: Callable[[Dict], float], config: optimization.OptimizationConfig = None)
```

Args:

parameterization: Geometry parameterization object
    flow_solver: Function(geometry) -> flow_state dict
    adjoint_solver: Function(geometry, flow_state) -> sensitivity
    objective: Function(flow_state) -> scalar objective
    config: Optimization configuration

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\optimization.py:357](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\optimization.py#L357)*

##### `add_constraint`

```python
def add_constraint(self, constraint: optimization.ConstraintSpec)
```

Add a design constraint.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\optimization.py:382](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\optimization.py#L382)*

##### `evaluate_objective`

```python
def evaluate_objective(self, alpha: torch.Tensor) -> Tuple[float, torch.Tensor]
```

Evaluate objective and gradient.

**Returns**: `typing.Tuple[float, torch.Tensor]` - (J, dJ/dα)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\optimization.py:386](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\optimization.py#L386)*

##### `optimize`

```python
def optimize(self, alpha0: torch.Tensor) -> optimization.OptimizationResult
```

Run optimization with configured algorithm.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\optimization.py:576](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\optimization.py#L576)*

##### `optimize_lbfgs`

```python
def optimize_lbfgs(self, alpha0: torch.Tensor) -> optimization.OptimizationResult
```

Run L-BFGS optimization.

Uses PyTorch's built-in LBFGS optimizer.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\optimization.py:511](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\optimization.py#L511)*

##### `optimize_steepest_descent`

```python
def optimize_steepest_descent(self, alpha0: torch.Tensor) -> optimization.OptimizationResult
```

Run steepest descent optimization.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\optimization.py:460](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\optimization.py#L460)*

## Functions

### `create_wedge_design_problem`

```python
def create_wedge_design_problem(n_control: int = 10, Mach: float = 5.0, theta_initial: float = 10.0) -> Tuple[optimization.BSplineParameterization, torch.Tensor]
```

Create a wedge design problem for validation.

Design a 2D compression ramp shape to minimize
pressure drag at hypersonic conditions.

**Parameters:**

- **n_control** (`<class 'int'>`): Number of B-spline control points
- **Mach** (`<class 'float'>`): Freestream Mach number
- **theta_initial** (`<class 'float'>`): Initial wedge half-angle (degrees)

**Returns**: `typing.Tuple[optimization.BSplineParameterization, torch.Tensor]` - (parameterization, initial_design)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\optimization.py:588](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\optimization.py#L588)*

### `validate_optimization`

```python
def validate_optimization()
```

Run validation tests for optimization module.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\optimization.py:623](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\optimization.py#L623)*
