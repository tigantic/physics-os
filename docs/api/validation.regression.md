# Module `validation.regression`

Regression Testing Module for Project The Physics OS.

Provides regression testing framework including:
- Golden value comparison and management
- Array and tensor comparison utilities
- State comparison for CFD simulations
- Automated regression test suite execution

These tools ensure code changes don't introduce regressions.

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `ArrayComparator`

Compare arrays with configurable tolerance.

Supports numpy arrays and provides detailed mismatch information.

#### Methods

##### `__init__`

```python
def __init__(self, rtol: float = 1e-05, atol: float = 1e-08, comparison_type: regression.ComparisonType = <ComparisonType.HYBRID: 4>)
```

Initialize array comparator.

**Parameters:**

- **rtol** (`<class 'float'>`): Relative tolerance
- **atol** (`<class 'float'>`): Absolute tolerance
- **comparison_type** (`<enum 'ComparisonType'>`): Type of comparison

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py:291](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py#L291)*

##### `compare`

```python
def compare(self, actual: numpy.ndarray, expected: numpy.ndarray, name: str = 'array') -> regression.RegressionResult
```

Compare two arrays.

**Parameters:**

- **actual** (`<class 'numpy.ndarray'>`): Actual/computed array
- **expected** (`<class 'numpy.ndarray'>`): Expected/reference array
- **name** (`<class 'str'>`): Name for the comparison

**Returns**: `<class 'regression.RegressionResult'>` - RegressionResult with comparison details

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py:309](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py#L309)*

### class `ComparisonType`(Enum)

Type of comparison for regression testing.

### class `GoldenValue`

Reference value for regression testing.

Stores a value along with metadata for version tracking
and reproducibility.

#### Attributes

- **name** (`<class 'str'>`): 
- **value** (`typing.Any`): 
- **value_hash** (`<class 'str'>`): 
- **created_at** (`<class 'float'>`): 
- **version** (`<class 'str'>`): 
- **metadata** (`typing.Dict[str, typing.Any]`): 

#### Methods

##### `__init__`

```python
def __init__(self, name: str, value: Any, value_hash: str = '', created_at: float = <factory>, version: str = '1.0', metadata: Dict[str, Any] = <factory>) -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict
```

Convert metadata to dictionary (excludes large value).

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py:128](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py#L128)*

##### `verify_hash`

```python
def verify_hash(self) -> bool
```

Verify the stored hash matches current value.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py:124](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py#L124)*

### class `GoldenValueStore`

Persistent storage for golden values.

Manages a collection of golden values with versioning
and efficient lookup.

#### Methods

##### `__init__`

```python
def __init__(self, directory: Union[str, pathlib.Path])
```

Initialize golden value store.

**Parameters:**

- **directory** (`typing.Union[str, pathlib.Path]`): Directory for storing golden values

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py:154](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py#L154)*

##### `delete`

```python
def delete(self, name: str)
```

Delete a golden value.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py:240](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py#L240)*

##### `exists`

```python
def exists(self, name: str) -> bool
```

Check if a golden value exists.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py:232](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py#L232)*

##### `get_info`

```python
def get_info(self, name: str) -> Optional[Dict]
```

Get metadata about a golden value without loading it.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py:249](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py#L249)*

##### `list_all`

```python
def list_all(self) -> List[str]
```

List all golden value names.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py:236](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py#L236)*

##### `load`

```python
def load(self, name: str) -> Optional[regression.GoldenValue]
```

Load a golden value.

**Parameters:**

- **name** (`<class 'str'>`): Identifier of the golden value

**Returns**: `typing.Optional[regression.GoldenValue]` - The GoldenValue or None if not found

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py:215](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py#L215)*

##### `save`

```python
def save(self, name: str, value: Any, version: str = '1.0', **metadata) -> regression.GoldenValue
```

Save a golden value.

**Parameters:**

- **name** (`<class 'str'>`): Unique identifier
- **value** (`typing.Any`): Value to save
- **version** (`<class 'str'>`): Version string **metadata: Additional metadata

**Returns**: `<class 'regression.GoldenValue'>` - The created GoldenValue

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py:178](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py#L178)*

### class `RegressionResult`

Result of a regression test.

#### Attributes

- **test_name** (`<class 'str'>`): 
- **passed** (`<class 'bool'>`): 
- **comparison_type** (`<enum 'ComparisonType'>`): 
- **max_difference** (`<class 'float'>`): 
- **mean_difference** (`<class 'float'>`): 
- **tolerance_used** (`<class 'float'>`): 
- **n_mismatches** (`<class 'int'>`): 
- **n_elements** (`<class 'int'>`): 
- **message** (`<class 'str'>`): 
- **details** (`typing.Dict[str, typing.Any]`): 

#### Properties

##### `mismatch_rate`

```python
def mismatch_rate(self) -> float
```

Fraction of elements that mismatch.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py:61](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py#L61)*

#### Methods

##### `__init__`

```python
def __init__(self, test_name: str, passed: bool, comparison_type: regression.ComparisonType, max_difference: float, mean_difference: float, tolerance_used: float, n_mismatches: int = 0, n_elements: int = 0, message: str = '', details: Dict[str, Any] = <factory>) -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py:68](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py#L68)*

### class `RegressionSuite`

Collection of regression tests.

Manages multiple regression tests and provides
summary reporting.

#### Attributes

- **name** (`<class 'str'>`): 
- **store** (`<class 'regression.GoldenValueStore'>`): 
- **tests** (`typing.List[regression.RegressionTest]`): 
- **results** (`typing.List[regression.RegressionResult]`): 

#### Properties

##### `all_passed`

```python
def all_passed(self) -> bool
```

Check if all tests passed.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py:620](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py#L620)*

##### `fail_count`

```python
def fail_count(self) -> int
```

Count of failed tests.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py:630](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py#L630)*

##### `pass_count`

```python
def pass_count(self) -> int
```

Count of passed tests.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py:625](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py#L625)*

#### Methods

##### `__init__`

```python
def __init__(self, name: str, store: regression.GoldenValueStore, tests: List[regression.RegressionTest] = <factory>, results: List[regression.RegressionResult] = <factory>) -> None
```

##### `add_test`

```python
def add_test(self, test: regression.RegressionTest)
```

Add a test to the suite.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py:591](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py#L591)*

##### `report`

```python
def report(self, format: str = 'text') -> str
```

Generate regression test report.

**Parameters:**

- **format** (`<class 'str'>`): Output format ("text", "markdown")

**Returns**: `<class 'str'>` - Formatted report

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py:635](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py#L635)*

##### `run_all`

```python
def run_all(self, verbose: bool = True) -> List[regression.RegressionResult]
```

Run all regression tests.

**Parameters:**

- **verbose** (`<class 'bool'>`): Print progress

**Returns**: `typing.List[regression.RegressionResult]` - List of RegressionResult

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py:595](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py#L595)*

### class `RegressionTest`

Definition of a regression test.

#### Attributes

- **name** (`<class 'str'>`): 
- **generator** (`typing.Callable[[], typing.Any]`): 
- **golden_name** (`<class 'str'>`): 
- **comparator** (`typing.Union[regression.ArrayComparator, regression.TensorComparator, regression.StateComparator]`): 
- **description** (`<class 'str'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, name: str, generator: Callable[[], Any], golden_name: str, comparator: Union[regression.ArrayComparator, regression.TensorComparator, regression.StateComparator] = <factory>, description: str = '') -> None
```

##### `run`

```python
def run(self, store: regression.GoldenValueStore) -> regression.RegressionResult
```

Run the regression test.

**Parameters:**

- **store** (`<class 'regression.GoldenValueStore'>`): Golden value store

**Returns**: `<class 'regression.RegressionResult'>` - RegressionResult

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py:521](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py#L521)*

### class `StateComparator`

Compare CFD states with field-specific tolerances.

Handles multi-field states where different variables
may have different accuracy requirements.

#### Methods

##### `__init__`

```python
def __init__(self, field_tolerances: Optional[Dict[str, float]] = None, default_rtol: float = 1e-05, default_atol: float = 1e-08)
```

Initialize state comparator.

**Parameters:**

- **field_tolerances** (`typing.Optional[typing.Dict[str, float]]`): Per-field relative tolerances
- **default_rtol** (`<class 'float'>`): Default relative tolerance Default: `relative tolerance`.
- **default_atol** (`<class 'float'>`): Default absolute tolerance Default: `absolute tolerance`.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py:421](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py#L421)*

##### `compare`

```python
def compare(self, actual: Dict[str, torch.Tensor], expected: Dict[str, torch.Tensor], name: str = 'state') -> List[regression.RegressionResult]
```

Compare two states (dictionaries of fields).

**Parameters:**

- **actual** (`typing.Dict[str, torch.Tensor]`): Actual state dictionary
- **expected** (`typing.Dict[str, torch.Tensor]`): Expected state dictionary
- **name** (`<class 'str'>`): Base name for comparisons

**Returns**: `typing.List[regression.RegressionResult]` - List of RegressionResult for each field

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py:439](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py#L439)*

### class `TensorComparator`(ArrayComparator)

Compare PyTorch tensors with configurable tolerance.

Extends ArrayComparator with tensor-specific handling.

#### Methods

##### `compare`

```python
def compare(self, actual: Union[torch.Tensor, numpy.ndarray], expected: Union[torch.Tensor, numpy.ndarray], name: str = 'tensor') -> regression.RegressionResult
```

Compare two tensors.

**Parameters:**

- **actual** (`typing.Union[torch.Tensor, numpy.ndarray]`): Actual/computed tensor
- **expected** (`typing.Union[torch.Tensor, numpy.ndarray]`): Expected/reference tensor
- **name** (`<class 'str'>`): Name for the comparison

**Returns**: `<class 'regression.RegressionResult'>` - RegressionResult with comparison details

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py:382](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py#L382)*

## Functions

### `run_full_regression`

```python
def run_full_regression(suite: regression.RegressionSuite, output_path: Union[str, pathlib.Path, NoneType] = None, verbose: bool = True) -> Tuple[bool, List[regression.RegressionResult]]
```

Run full regression suite and optionally save report.

**Parameters:**

- **suite** (`<class 'regression.RegressionSuite'>`): Regression suite to run
- **output_path** (`typing.Union[str, pathlib.Path, NoneType]`): Optional path for report
- **verbose** (`<class 'bool'>`): Print progress

**Returns**: `typing.Tuple[bool, typing.List[regression.RegressionResult]]` - Tuple of (all_passed, results)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py:720](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py#L720)*

### `run_regression_tests`

```python
def run_regression_tests(tests: List[regression.RegressionTest], store: regression.GoldenValueStore, verbose: bool = True) -> List[regression.RegressionResult]
```

Run a list of regression tests.

**Parameters:**

- **tests** (`typing.List[regression.RegressionTest]`): List of regression tests
- **store** (`<class 'regression.GoldenValueStore'>`): Golden value store
- **verbose** (`<class 'bool'>`): Print progress

**Returns**: `typing.List[regression.RegressionResult]` - List of results

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py:688](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py#L688)*

### `update_golden_values`

```python
def update_golden_values(store: regression.GoldenValueStore, values: Dict[str, Any], version: str = '1.0', force: bool = False) -> Dict[str, bool]
```

Update multiple golden values.

**Parameters:**

- **store** (`<class 'regression.GoldenValueStore'>`): Golden value store
- **values** (`typing.Dict[str, typing.Any]`): Dictionary of name -> value
- **version** (`<class 'str'>`): Version string for all values
- **force** (`<class 'bool'>`): Overwrite existing values

**Returns**: `typing.Dict[str, bool]` - Dictionary of name -> whether updated

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py:254](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\validation\regression.py#L254)*
