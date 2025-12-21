# Module `cfd.adjoint`

Adjoint Solver for Sensitivity Analysis ========================================

Implements continuous and discrete adjoint methods for computing
sensitivities of objective functions with respect to design variables.

Key Applications:
    - Shape optimization (minimize drag, control heating)
    - Flow control (actuation placement)
    - Uncertainty quantification
    - Inverse problems

The Adjoint Method:
    For objective J(U, α) where U solves R(U, α) = 0:

    dJ/dα = ∂J/∂α + ψᵀ ∂R/∂α

    where adjoint variable ψ solves:

    (∂R/∂U)ᵀ ψ = -(∂J/∂U)ᵀ

Advantages:
    - Cost independent of number of design variables
    - Single adjoint solve gives all sensitivities
    - Enables gradient-based optimization

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `AdjointConfig`

Configuration for adjoint solver.

#### Attributes

- **method** (`<enum 'AdjointMethod'>`): 
- **max_iterations** (`<class 'int'>`): 
- **tolerance** (`<class 'float'>`): 
- **cfl** (`<class 'float'>`): 
- **smoothing_iterations** (`<class 'int'>`): 
- **dissipation_coeff** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, method: adjoint.AdjointMethod = <AdjointMethod.DISCRETE: 'discrete'>, max_iterations: int = 1000, tolerance: float = 1e-10, cfl: float = 0.5, smoothing_iterations: int = 0, dissipation_coeff: float = 0.5) -> None
```

### class `AdjointEuler2D`

Adjoint solver for 2D Euler equations.

Solves the adjoint equations backward in time (for unsteady)
or to steady state (for steady primal).

#### Methods

##### `__init__`

```python
def __init__(self, Nx: int, Ny: int, dx: float, dy: float, gamma: float = 1.4, config: adjoint.AdjointConfig = None)
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py:307](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py#L307)*

##### `adjoint_rhs`

```python
def adjoint_rhs(self, psi: adjoint.AdjointState, rho: torch.Tensor, u: torch.Tensor, v: torch.Tensor, p: torch.Tensor, source_term: torch.Tensor) -> adjoint.AdjointState
```

Compute RHS for adjoint equations.

∂ψ/∂t = Aᵀ ∂ψ/∂x + Bᵀ ∂ψ/∂y - (∂J/∂U)ᵀ

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py:408](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py#L408)*

##### `flux_jacobian_x`

```python
def flux_jacobian_x(self, rho: torch.Tensor, u: torch.Tensor, v: torch.Tensor, p: torch.Tensor) -> torch.Tensor
```

Compute Jacobian ∂F/∂U for x-flux.

**Returns**: `<class 'torch.Tensor'>` - (4, 4, Ny, Nx) tensor of Jacobians at each point

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py:323](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py#L323)*

##### `flux_jacobian_y`

```python
def flux_jacobian_y(self, rho: torch.Tensor, u: torch.Tensor, v: torch.Tensor, p: torch.Tensor) -> torch.Tensor
```

Compute Jacobian ∂G/∂U for y-flux.

**Returns**: `<class 'torch.Tensor'>` - (4, 4, Ny, Nx) tensor of Jacobians at each point

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py:366](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py#L366)*

##### `solve_steady`

```python
def solve_steady(self, rho: torch.Tensor, u: torch.Tensor, v: torch.Tensor, p: torch.Tensor, objective: adjoint.ObjectiveFunction) -> adjoint.SensitivityResult
```

Solve steady adjoint equations.

Iterates until adjoint residual converges.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py:461](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py#L461)*

### class `AdjointMethod`(Enum)

Adjoint formulation type.

### class `AdjointState`

Adjoint variable state.

For Euler equations, adjoint variables are:
    ψ = [ψ_ρ, ψ_ρu, ψ_ρv, ψ_E]ᵀ

#### Attributes

- **psi_rho** (`<class 'torch.Tensor'>`): 
- **psi_rhou** (`<class 'torch.Tensor'>`): 
- **psi_rhov** (`<class 'torch.Tensor'>`): 
- **psi_E** (`<class 'torch.Tensor'>`): 

#### Properties

##### `shape`

```python
def shape(self) -> torch.Size
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py:71](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py#L71)*

#### Methods

##### `__init__`

```python
def __init__(self, psi_rho: torch.Tensor, psi_rhou: torch.Tensor, psi_rhov: torch.Tensor, psi_E: torch.Tensor) -> None
```

##### `from_tensor`

```python
def from_tensor(psi: torch.Tensor) -> 'AdjointState'
```

Create from (4, Ny, Nx) tensor.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py:81](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py#L81)*

##### `to_tensor`

```python
def to_tensor(self) -> torch.Tensor
```

Stack adjoint variables into (4, Ny, Nx) tensor.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py:75](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py#L75)*

##### `zeros`

```python
def zeros(shape: Tuple[int, int], dtype=torch.float64) -> 'AdjointState'
```

Create zero-initialized adjoint state.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py:91](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py#L91)*

### class `DragObjective`(ObjectiveFunction)

Drag coefficient objective.

C_D = (1/q_∞ S) ∫ p n_x dS

where n_x is the x-component of surface normal.

#### Methods

##### `__init__`

```python
def __init__(self, surface_mask: torch.Tensor, normal_x: torch.Tensor, normal_y: torch.Tensor, q_inf: float, S_ref: float)
```

Args:

surface_mask: Boolean mask for surface cells
    normal_x, normal_y: Surface normal components
    q_inf: Freestream dynamic pressure
    S_ref: Reference area

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py:169](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py#L169)*

##### `evaluate`

```python
def evaluate(self, rho: torch.Tensor, u: torch.Tensor, v: torch.Tensor, p: torch.Tensor, **kwargs) -> torch.Tensor
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py:190](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py#L190)*

##### `gradient`

```python
def gradient(self, rho: torch.Tensor, u: torch.Tensor, v: torch.Tensor, p: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py:204](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py#L204)*

### class `HeatFluxObjective`(ObjectiveFunction)

Integrated wall heat flux objective.

J = ∫ q_w dS

For hypersonic vehicles, minimizing peak or integrated
heat flux is critical for TPS design.

#### Methods

##### `__init__`

```python
def __init__(self, wall_mask: torch.Tensor, dy: float, k: torch.Tensor)
```

Args:

wall_mask: Boolean mask for wall-adjacent cells
    dy: Grid spacing normal to wall
    k: Thermal conductivity field

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py:233](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py#L233)*

##### `evaluate`

```python
def evaluate(self, rho: torch.Tensor, u: torch.Tensor, v: torch.Tensor, p: torch.Tensor, T: Optional[torch.Tensor] = None, T_wall: float = 300.0, gamma: float = 1.4, R: float = 287.0, **kwargs) -> torch.Tensor
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py:249](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py#L249)*

##### `gradient`

```python
def gradient(self, rho: torch.Tensor, u: torch.Tensor, v: torch.Tensor, p: torch.Tensor, T_wall: float = 300.0, gamma: float = 1.4, R: float = 287.0, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py:271](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py#L271)*

### class `ObjectiveFunction`

Base class for objective functions.

Subclasses implement:
    - evaluate: Compute J(U)
    - gradient: Compute ∂J/∂U

#### Methods

##### `evaluate`

```python
def evaluate(self, rho: torch.Tensor, u: torch.Tensor, v: torch.Tensor, p: torch.Tensor, **kwargs) -> torch.Tensor
```

Evaluate objective function.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py:132](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py#L132)*

##### `gradient`

```python
def gradient(self, rho: torch.Tensor, u: torch.Tensor, v: torch.Tensor, p: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

Compute gradient ∂J/∂U in primitive variables.

**Returns**: `typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]` - Tuple of (∂J/∂ρ, ∂J/∂u, ∂J/∂v, ∂J/∂p)

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py:143](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py#L143)*

### class `ObjectiveType`(Enum)

Common objective function types.

### class `SensitivityResult`

Result from sensitivity computation.

#### Attributes

- **dJ_dalpha** (`<class 'torch.Tensor'>`): 
- **objective_value** (`<class 'float'>`): 
- **adjoint_state** (`<class 'adjoint.AdjointState'>`): 
- **converged** (`<class 'bool'>`): 
- **iterations** (`<class 'int'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, dJ_dalpha: torch.Tensor, objective_value: float, adjoint_state: adjoint.AdjointState, converged: bool, iterations: int) -> None
```

## Functions

### `compute_shape_sensitivity`

```python
def compute_shape_sensitivity(adjoint_state: adjoint.AdjointState, rho: torch.Tensor, u: torch.Tensor, v: torch.Tensor, p: torch.Tensor, surface_mask: torch.Tensor, normal_x: torch.Tensor, normal_y: torch.Tensor, gamma: float = 1.4) -> torch.Tensor
```

Compute shape sensitivity dJ/d(surface).

For surface points, sensitivity is:
    dJ/dn = ψᵀ (∂R/∂n)

**Parameters:**

- **adjoint_state** (`<class 'adjoint.AdjointState'>`): Converged adjoint solution rho, u, v, p: Flow state
- **surface_mask** (`<class 'torch.Tensor'>`): Boolean mask for surface normal_x, normal_y: Surface normals

**Returns**: `<class 'torch.Tensor'>` - Sensitivity on surface

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py:522](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py#L522)*

### `validate_adjoint`

```python
def validate_adjoint()
```

Run validation tests for adjoint module.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py:570](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\adjoint.py#L570)*
