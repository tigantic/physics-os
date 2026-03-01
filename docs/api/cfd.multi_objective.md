# Module `cfd.multi_objective`

Multi-Objective Optimization for Aerodynamic Design ====================================================

Extends the single-objective optimization framework to handle
multiple competing objectives common in hypersonic design:

    - Minimize drag coefficient C_D
    - Minimize peak heat flux q_max
    - Maximize lift-to-drag ratio L/D
    - Minimize weight (volume constraints)

Key Concepts:
    - Pareto optimality: No improvement in one objective without
      degrading another
    - Pareto front: Set of all Pareto-optimal solutions
    - Dominance: Solution A dominates B if A is better in at least
      one objective and no worse in all others

Algorithms:
    1. Weighted Sum Method - Simple but uneven Pareto sampling
    2. ε-Constraint Method - Systematic Pareto exploration
    3. NSGA-II - Evolutionary multi-objective optimization
    4. Reference Point Method - User-specified aspiration levels

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `MOOAlgorithm`(Enum)

Multi-objective optimization algorithms.

### class `MOOConfig`

Configuration for multi-objective optimization.

#### Attributes

- **algorithm** (`<enum 'MOOAlgorithm'>`): 
- **population_size** (`<class 'int'>`): 
- **n_generations** (`<class 'int'>`): 
- **crossover_prob** (`<class 'float'>`): 
- **mutation_prob** (`<class 'float'>`): 
- **mutation_strength** (`<class 'float'>`): 
- **n_weight_samples** (`<class 'int'>`): 
- **epsilon_steps** (`<class 'int'>`): 
- **reference_point** (`typing.Optional[typing.Dict[str, float]]`): 

#### Methods

##### `__init__`

```python
def __init__(self, algorithm: multi_objective.MOOAlgorithm = <MOOAlgorithm.NSGA_II: 'nsga-ii'>, population_size: int = 100, n_generations: int = 50, crossover_prob: float = 0.9, mutation_prob: float = 0.1, mutation_strength: float = 0.1, n_weight_samples: int = 21, epsilon_steps: int = 10, reference_point: Optional[Dict[str, float]] = None) -> None
```

### class `MOOResult`

Result from multi-objective optimization.

#### Attributes

- **pareto_front** (`typing.List[multi_objective.ParetoSolution]`): 
- **hypervolume** (`<class 'float'>`): 
- **n_generations** (`<class 'int'>`): 
- **n_evaluations** (`<class 'int'>`): 
- **utopia_point** (`typing.Dict[str, float]`): 
- **nadir_point** (`typing.Dict[str, float]`): 

#### Methods

##### `__init__`

```python
def __init__(self, pareto_front: List[multi_objective.ParetoSolution], hypervolume: float, n_generations: int, n_evaluations: int, utopia_point: Dict[str, float], nadir_point: Dict[str, float]) -> None
```

### class `MultiObjectiveOptimizer`

Multi-objective optimization driver.

Finds Pareto-optimal designs trading off multiple objectives.

#### Methods

##### `__init__`

```python
def __init__(self, objectives: List[multi_objective.ObjectiveSpec], bounds: Tuple[torch.Tensor, torch.Tensor], config: multi_objective.MOOConfig = None)
```

Args:

objectives: List of objective specifications
    bounds: (lower, upper) bounds for design variables
    config: Optimization configuration

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\multi_objective.py:294](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\multi_objective.py#L294)*

##### `crossover`

```python
def crossover(self, parent1: torch.Tensor, parent2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```

Simulated Binary Crossover (SBX).

Creates two offspring from two parents with distribution
controlled by η_c parameter.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\multi_objective.py:333](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\multi_objective.py#L333)*

##### `evaluate`

```python
def evaluate(self, design: torch.Tensor) -> Dict[str, float]
```

Evaluate all objectives for a design.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\multi_objective.py:315](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\multi_objective.py#L315)*

##### `initialize_population`

```python
def initialize_population(self) -> List[multi_objective.ParetoSolution]
```

Create initial random population.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\multi_objective.py:324](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\multi_objective.py#L324)*

##### `mutate`

```python
def mutate(self, design: torch.Tensor) -> torch.Tensor
```

Polynomial mutation.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\multi_objective.py:392](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\multi_objective.py#L392)*

##### `optimize`

```python
def optimize(self) -> multi_objective.MOOResult
```

Run optimization with configured algorithm.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\multi_objective.py:619](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\multi_objective.py#L619)*

##### `random_design`

```python
def random_design(self) -> torch.Tensor
```

Generate a random design within bounds.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\multi_objective.py:320](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\multi_objective.py#L320)*

##### `run_nsga2`

```python
def run_nsga2(self) -> multi_objective.MOOResult
```

Run NSGA-II algorithm.

**Returns**: `<class 'multi_objective.MOOResult'>` - MOOResult with Pareto front and metrics

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\multi_objective.py:450](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\multi_objective.py#L450)*

##### `run_weighted_sum`

```python
def run_weighted_sum(self) -> multi_objective.MOOResult
```

Run weighted sum scalarization.

Samples different weight combinations to approximate Pareto front.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\multi_objective.py:549](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\multi_objective.py#L549)*

##### `tournament_selection`

```python
def tournament_selection(self, population: List[multi_objective.ParetoSolution]) -> multi_objective.ParetoSolution
```

Binary tournament selection based on rank and crowding distance.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\multi_objective.py:426](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\multi_objective.py#L426)*

### class `ObjectiveSpec`

Specification for an objective function.

#### Attributes

- **name** (`<class 'str'>`): 
- **function** (`typing.Callable[[torch.Tensor], float]`): 
- **gradient** (`typing.Callable[[torch.Tensor], torch.Tensor]`): 
- **minimize** (`<class 'bool'>`): 
- **weight** (`<class 'float'>`): 
- **reference** (`typing.Optional[float]`): 
- **utopia** (`typing.Optional[float]`): 
- **nadir** (`typing.Optional[float]`): 

#### Methods

##### `__init__`

```python
def __init__(self, name: str, function: Callable[[torch.Tensor], float], gradient: Callable[[torch.Tensor], torch.Tensor], minimize: bool = True, weight: float = 1.0, reference: Optional[float] = None, utopia: Optional[float] = None, nadir: Optional[float] = None) -> None
```

### class `ParetoSolution`

A single solution on the Pareto front.

#### Attributes

- **design** (`<class 'torch.Tensor'>`): 
- **objectives** (`typing.Dict[str, float]`): 
- **rank** (`<class 'int'>`): 
- **crowding_distance** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, design: torch.Tensor, objectives: Dict[str, float], rank: int = 0, crowding_distance: float = 0.0) -> None
```

## Functions

### `create_drag_heating_problem`

```python
def create_drag_heating_problem(n_vars: int = 10) -> Tuple[List[multi_objective.ObjectiveSpec], Tuple[torch.Tensor, torch.Tensor]]
```

Create a test bi-objective problem: minimize drag and heating.

Uses simplified analytical objectives for testing.

**Returns**: `typing.Tuple[typing.List[multi_objective.ObjectiveSpec], typing.Tuple[torch.Tensor, torch.Tensor]]` - (objectives, bounds)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\multi_objective.py:629](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\multi_objective.py#L629)*

### `crowding_distance`

```python
def crowding_distance(population: List[multi_objective.ParetoSolution], front_indices: List[int], minimize: Dict[str, bool]) -> None
```

Compute crowding distance for solutions in a front.

Measures the density of solutions around each point.
Higher distance = more isolated = better diversity.

**Parameters:**

- **population** (`typing.List[multi_objective.ParetoSolution]`): Full population
- **front_indices** (`typing.List[int]`): Indices of solutions in this front
- **minimize** (`typing.Dict[str, bool]`): Minimization flags (not used directly)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\multi_objective.py:197](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\multi_objective.py#L197)*

### `dominates`

```python
def dominates(obj_a: Dict[str, float], obj_b: Dict[str, float], minimize: Dict[str, bool]) -> bool
```

Check if solution A dominates solution B.

A dominates B if:
    - A is no worse than B in all objectives
    - A is strictly better than B in at least one objective

**Parameters:**

- **minimize** (`typing.Dict[str, bool]`): Dict indicating which objectives to minimize

**Returns**: `<class 'bool'>` - True if A dominates B

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\multi_objective.py:97](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\multi_objective.py#L97)*

### `fast_non_dominated_sort`

```python
def fast_non_dominated_sort(population: List[multi_objective.ParetoSolution], minimize: Dict[str, bool]) -> List[List[int]]
```

Fast non-dominated sorting (NSGA-II).

Assigns each solution to a Pareto front (rank).
Rank 0 = non-dominated, Rank 1 = dominated by rank 0, etc.

**Parameters:**

- **population** (`typing.List[multi_objective.ParetoSolution]`): List of solutions
- **minimize** (`typing.Dict[str, bool]`): Minimization flags per objective

**Returns**: `typing.List[typing.List[int]]` - List of fronts, each containing indices of solutions

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\multi_objective.py:134](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\multi_objective.py#L134)*

### `hypervolume_2d`

```python
def hypervolume_2d(pareto_front: List[multi_objective.ParetoSolution], reference: Dict[str, float], obj_names: Tuple[str, str]) -> float
```

Compute hypervolume indicator for 2D Pareto front.

The hypervolume is the area dominated by the Pareto front
and bounded by the reference point.

**Parameters:**

- **pareto_front** (`typing.List[multi_objective.ParetoSolution]`): List of Pareto-optimal solutions
- **reference** (`typing.Dict[str, float]`): Reference point (worst acceptable values)
- **obj_names** (`typing.Tuple[str, str]`): Tuple of (obj1_name, obj2_name)

**Returns**: `<class 'float'>` - Hypervolume value

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\multi_objective.py:249](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\multi_objective.py#L249)*

### `validate_moo`

```python
def validate_moo()
```

Run validation tests for multi-objective optimization.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\multi_objective.py:676](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\multi_objective.py#L676)*
