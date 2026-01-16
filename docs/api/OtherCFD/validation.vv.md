# Module `validation.vv`

Verification and Validation (V&V) Module for Project HyperTensor.

Provides formal V&V framework including:
- Verification: Solving the equations right (code correctness)
- Validation: Solving the right equations (physical accuracy)
- Uncertainty quantification for validation
- V&V plan management and reporting

These tools support rigorous scientific software quality assurance.

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `AnalyticalValidation`

Validation against analytical solutions.

Compares simulation results to exact analytical solutions
for convergence rate verification.

#### Methods

##### `__init__`

```python
def __init__(self, case: vv.ValidationCase, grid_sizes: List[int])
```

Initialize analytical validation.

**Parameters:**

- **case** (`<class 'vv.ValidationCase'>`): Validation case with analytical solution
- **grid_sizes** (`typing.List[int]`): Grid sizes for convergence study

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py:576](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py#L576)*

##### `validate`

```python
def validate(self) -> Dict[str, Any]
```

Run convergence study against analytical solution.

**Returns**: `typing.Dict[str, typing.Any]` - Dictionary with convergence results

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py:591](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py#L591)*

### class `CodeVerification`(ABC)

Abstract base class for code verification tests.

Code verification answers: "Did we solve the equations correctly?"

#### Methods

##### `verify`

```python
def verify(self) -> Dict[str, Any]
```

Run verification test.

**Returns**: `typing.Dict[str, typing.Any]` - Dictionary of verification metrics

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py:367](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py#L367)*

### class `ExperimentalValidation`

Validation against experimental data.

Compares simulation results to experimental measurements.

#### Methods

##### `__init__`

```python
def __init__(self, case: vv.ValidationCase, metrics: Optional[List[str]] = None)
```

Initialize experimental validation.

**Parameters:**

- **case** (`<class 'vv.ValidationCase'>`): Validation case definition
- **metrics** (`typing.Optional[typing.List[str]]`): Specific metrics to compare

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py:512](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py#L512)*

##### `validate`

```python
def validate(self) -> Dict[str, Any]
```

Run validation against experimental data.

**Returns**: `typing.Dict[str, typing.Any]` - Dictionary of validation results

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py:527](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py#L527)*

### class `IntegrationVerification`(CodeVerification)

Integration-level code verification.

Tests interactions between components.

#### Methods

##### `__init__`

```python
def __init__(self, workflow: Callable[[], Any], expected_properties: Dict[str, Callable[[Any], bool]])
```

Initialize integration verification.

**Parameters:**

- **workflow** (`typing.Callable[[], typing.Any]`): Function that runs the integration test
- **expected_properties** (`typing.Dict[str, typing.Callable[[typing.Any], bool]]`): Dict of property_name -> checker function

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py:443](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py#L443)*

##### `verify`

```python
def verify(self) -> Dict[str, Any]
```

Run integration test.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py:458](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py#L458)*

### class `UncertaintyBand`

Uncertainty band for validation comparisons.

Represents experimental or numerical uncertainty.

#### Attributes

- **lower** (`<class 'numpy.ndarray'>`): 
- **upper** (`<class 'numpy.ndarray'>`): 
- **confidence** (`<class 'float'>`): 

#### Properties

##### `width`

```python
def width(self) -> numpy.ndarray
```

Get band width at each point.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py:675](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py#L675)*

#### Methods

##### `__init__`

```python
def __init__(self, lower: numpy.ndarray, upper: numpy.ndarray, confidence: float = 0.95) -> None
```

##### `contains`

```python
def contains(self, values: numpy.ndarray) -> numpy.ndarray
```

Check which values fall within the band.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py:671](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py#L671)*

##### `from_mean_std`

```python
def from_mean_std(mean: numpy.ndarray, std: numpy.ndarray, n_sigma: float = 2.0) -> 'UncertaintyBand'
```

Create uncertainty band from mean and standard deviation.

**Parameters:**

- **mean** (`<class 'numpy.ndarray'>`): Mean values
- **std** (`<class 'numpy.ndarray'>`): Standard deviations
- **n_sigma** (`<class 'float'>`): Number of standard deviations

**Returns**: `<class 'vv.UncertaintyBand'>` - UncertaintyBand

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py:647](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py#L647)*

### class `UnitVerification`(CodeVerification)

Unit-level code verification.

Tests individual functions/classes for correctness.

#### Methods

##### `__init__`

```python
def __init__(self, function: Callable, test_cases: List[Tuple[tuple, Any]], comparator: Optional[Callable[[Any, Any], bool]] = None)
```

Initialize unit verification.

**Parameters:**

- **function** (`typing.Callable`): Function to test
- **test_cases** (`typing.List[typing.Tuple[tuple, typing.Any]]`): List of (inputs, expected_output) tuples
- **comparator** (`typing.Optional[typing.Callable[[typing.Any, typing.Any], bool]]`): Custom comparison function

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py:385](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py#L385)*

##### `verify`

```python
def verify(self) -> Dict[str, Any]
```

Run all test cases.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py:403](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py#L403)*

### class `VVCategory`(Enum)

Category of V&V activity.

### class `VVLevel`(Enum)

Level of V&V rigor.

### class `VVPlan`

V&V Plan containing all tests to execute.

Organizes tests by category and level, with dependency tracking.

#### Attributes

- **name** (`<class 'str'>`): 
- **version** (`<class 'str'>`): 
- **description** (`<class 'str'>`): 
- **tests** (`typing.List[vv.VVTest]`): 
- **results** (`typing.List[vv.VVTestResult]`): 

#### Properties

##### `summary`

```python
def summary(self) -> Dict
```

Get summary statistics.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py:239](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py#L239)*

#### Methods

##### `__init__`

```python
def __init__(self, name: str, version: str = '1.0', description: str = '', tests: List[vv.VVTest] = <factory>, results: List[vv.VVTestResult] = <factory>) -> None
```

##### `add_test`

```python
def add_test(self, test: vv.VVTest)
```

Add a test to the plan.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py:161](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py#L161)*

##### `get_tests_by_category`

```python
def get_tests_by_category(self, category: vv.VVCategory) -> List[vv.VVTest]
```

Get all tests in a category.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py:165](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py#L165)*

##### `get_tests_by_level`

```python
def get_tests_by_level(self, level: vv.VVLevel) -> List[vv.VVTest]
```

Get all tests at a given level.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py:169](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py#L169)*

##### `run`

```python
def run(self, categories: Optional[List[vv.VVCategory]] = None, levels: Optional[List[vv.VVLevel]] = None, verbose: bool = True) -> List[vv.VVTestResult]
```

Execute the V&V plan.

**Parameters:**

- **categories** (`typing.Optional[typing.List[vv.VVCategory]]`): Filter by categories (None = all)
- **levels** (`typing.Optional[typing.List[vv.VVLevel]]`): Filter by levels (None = all)
- **verbose** (`<class 'bool'>`): Print progress

**Returns**: `typing.List[vv.VVTestResult]` - List of test results

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py:196](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py#L196)*

### class `VVReport`

Comprehensive V&V report.

Generates formatted reports of V&V activities and outcomes.

#### Attributes

- **plan** (`<class 'vv.VVPlan'>`): 
- **generated_at** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, plan: vv.VVPlan, generated_at: float = <factory>) -> None
```

##### `save`

```python
def save(self, filepath: Union[str, pathlib.Path], format: str = 'markdown')
```

Save report to file.

**Parameters:**

- **filepath** (`typing.Union[str, pathlib.Path]`): Output file path
- **format** (`<class 'str'>`): "markdown" or "json"

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py:345](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py#L345)*

##### `to_dict`

```python
def to_dict(self) -> Dict
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py:335](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py#L335)*

##### `to_markdown`

```python
def to_markdown(self) -> str
```

Generate markdown report.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py:266](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py#L266)*

### class `VVTest`

Definition of a V&V test.

#### Attributes

- **name** (`<class 'str'>`): 
- **category** (`<enum 'VVCategory'>`): 
- **level** (`<enum 'VVLevel'>`): 
- **description** (`<class 'str'>`): 
- **executor** (`typing.Callable[[], typing.Dict[str, typing.Any]]`): 
- **acceptance_criteria** (`typing.Dict[str, float]`): 
- **priority** (`<class 'int'>`): 
- **dependencies** (`typing.List[str]`): 

#### Methods

##### `__init__`

```python
def __init__(self, name: str, category: vv.VVCategory, level: vv.VVLevel, description: str, executor: Callable[[], Dict[str, Any]], acceptance_criteria: Dict[str, float] = <factory>, priority: int = 2, dependencies: List[str] = <factory>) -> None
```

##### `run`

```python
def run(self) -> 'VVTestResult'
```

Execute the V&V test.

**Returns**: `<class 'vv.VVTestResult'>` - VVTestResult with outcomes

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py:63](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py#L63)*

### class `VVTestResult`

Result of a V&V test execution.

#### Attributes

- **test_name** (`<class 'str'>`): 
- **category** (`<enum 'VVCategory'>`): 
- **passed** (`<class 'bool'>`): 
- **duration** (`<class 'float'>`): 
- **outputs** (`typing.Dict[str, typing.Any]`): 
- **criteria_results** (`typing.Dict[str, typing.Dict]`): 
- **error** (`typing.Optional[str]`): 

#### Methods

##### `__init__`

```python
def __init__(self, test_name: str, category: vv.VVCategory, passed: bool, duration: float, outputs: Dict[str, Any] = <factory>, criteria_results: Dict[str, Dict] = <factory>, error: Optional[str] = None) -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py:134](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py#L134)*

### class `ValidationCase`

Definition of a validation case.

Validation answers: "Are we solving the right equations?"

#### Attributes

- **name** (`<class 'str'>`): 
- **description** (`<class 'str'>`): 
- **experimental_data** (`typing.Optional[typing.Any]`): 
- **analytical_solution** (`typing.Optional[typing.Callable]`): 
- **simulation_runner** (`typing.Optional[typing.Callable]`): 
- **metrics** (`typing.List[str]`): 

#### Methods

##### `__init__`

```python
def __init__(self, name: str, description: str, experimental_data: Optional[Any] = None, analytical_solution: Optional[Callable] = None, simulation_runner: Optional[Callable] = None, metrics: List[str] = <factory>) -> None
```

### class `ValidationUncertainty`

Uncertainty quantification for validation.

Propagates uncertainties through validation comparisons.

#### Methods

##### `__init__`

```python
def __init__(self, experimental_uncertainty: vv.UncertaintyBand, numerical_uncertainty: Optional[vv.UncertaintyBand] = None)
```

Initialize validation uncertainty.

**Parameters:**

- **experimental_uncertainty** (`<class 'vv.UncertaintyBand'>`): Experimental data uncertainty
- **numerical_uncertainty** (`typing.Optional[vv.UncertaintyBand]`): Numerical solution uncertainty

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py:688](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py#L688)*

##### `compute_validation_metric`

```python
def compute_validation_metric(self, simulation: numpy.ndarray, experimental_mean: numpy.ndarray) -> Dict[str, float]
```

Compute validation metric accounting for uncertainties.

Uses ASME V&V 20-2009 approach.

**Parameters:**

- **simulation** (`<class 'numpy.ndarray'>`): Simulation results
- **experimental_mean** (`<class 'numpy.ndarray'>`): Experimental mean values

**Returns**: `typing.Dict[str, float]` - Dictionary of validation metrics

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py:703](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py#L703)*

## Functions

### `generate_vv_report`

```python
def generate_vv_report(plan: vv.VVPlan, format: str = 'markdown') -> str
```

Generate V&V report from completed plan.

**Parameters:**

- **plan** (`<class 'vv.VVPlan'>`): Completed V&V plan
- **format** (`<class 'str'>`): Output format ("markdown" or "json")

**Returns**: `<class 'str'>` - Formatted report string

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py:779](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py#L779)*

### `run_vv_plan`

```python
def run_vv_plan(plan: vv.VVPlan, output_path: Union[str, pathlib.Path, NoneType] = None, verbose: bool = True) -> Tuple[bool, vv.VVReport]
```

Execute a V&V plan and generate report.

**Parameters:**

- **plan** (`<class 'vv.VVPlan'>`): V&V plan to execute
- **output_path** (`typing.Union[str, pathlib.Path, NoneType]`): Optional path to save report
- **verbose** (`<class 'bool'>`): Print progress

**Returns**: `typing.Tuple[bool, vv.VVReport]` - Tuple of (all_passed, report)

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py:751](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\vv.py#L751)*
