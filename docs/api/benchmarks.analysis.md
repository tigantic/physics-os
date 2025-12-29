# Module `benchmarks.analysis`

Performance analysis and optimization recommendations.

This module provides analysis tools for identifying performance
bottlenecks and generating optimization recommendations.

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `BottleneckAnalysis`

Analysis of performance bottleneck.

#### Attributes

- **name** (`<class 'str'>`): 
- **bottleneck_type** (`<class 'str'>`): 
- **severity** (`<enum 'ImpactLevel'>`): 
- **time_percentage** (`<class 'float'>`): 
- **memory_percentage** (`<class 'float'>`): 
- **layers** (`typing.List[str]`): 
- **operations** (`typing.List[str]`): 
- **root_cause** (`<class 'str'>`): 
- **details** (`typing.Dict[str, typing.Any]`): 
- **recommendations** (`typing.List[analysis.OptimizationRecommendation]`): 

#### Methods

##### `__init__`

```python
def __init__(self, name: str, bottleneck_type: str, severity: analysis.ImpactLevel, time_percentage: float = 0.0, memory_percentage: float = 0.0, layers: List[str] = <factory>, operations: List[str] = <factory>, root_cause: str = '', details: Dict[str, Any] = <factory>, recommendations: List[analysis.OptimizationRecommendation] = <factory>) -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\analysis.py:104](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\analysis.py#L104)*

### class `EffortLevel`(Enum)

Implementation effort level.

### class `ImpactLevel`(Enum)

Impact level of optimization.

### class `OptimizationCategory`(Enum)

Categories of optimization recommendations.

### class `OptimizationRecommendation`

Single optimization recommendation.

#### Attributes

- **title** (`<class 'str'>`): 
- **description** (`<class 'str'>`): 
- **category** (`<enum 'OptimizationCategory'>`): 
- **impact** (`<enum 'ImpactLevel'>`): 
- **effort** (`<enum 'EffortLevel'>`): 
- **expected_speedup** (`typing.Optional[float]`): 
- **expected_memory_reduction** (`typing.Optional[float]`): 
- **implementation_steps** (`typing.List[str]`): 
- **code_example** (`<class 'str'>`): 
- **references** (`typing.List[str]`): 
- **priority_score** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, title: str, description: str, category: analysis.OptimizationCategory, impact: analysis.ImpactLevel, effort: analysis.EffortLevel, expected_speedup: Optional[float] = None, expected_memory_reduction: Optional[float] = None, implementation_steps: List[str] = <factory>, code_example: str = '', references: List[str] = <factory>, priority_score: float = 0.0) -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\analysis.py:67](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\analysis.py#L67)*

### class `PerformanceAnalysis`

Complete performance analysis.

#### Attributes

- **model_name** (`<class 'str'>`): 
- **total_time_ms** (`<class 'float'>`): 
- **bottlenecks** (`typing.List[analysis.BottleneckAnalysis]`): 
- **recommendations** (`typing.List[analysis.OptimizationRecommendation]`): 
- **compute_bound** (`<class 'bool'>`): 
- **memory_bound** (`<class 'bool'>`): 
- **compute_utilization** (`<class 'float'>`): 
- **memory_bandwidth_utilization** (`<class 'float'>`): 
- **primary_bottleneck** (`<class 'str'>`): 
- **optimization_potential** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, model_name: str, total_time_ms: float, bottlenecks: List[analysis.BottleneckAnalysis] = <factory>, recommendations: List[analysis.OptimizationRecommendation] = <factory>, compute_bound: bool = False, memory_bound: bool = False, compute_utilization: float = 0.0, memory_bandwidth_utilization: float = 0.0, primary_bottleneck: str = '', optimization_potential: float = 0.0) -> None
```

##### `get_top_recommendations`

```python
def get_top_recommendations(self, n: int = 5) -> List[analysis.OptimizationRecommendation]
```

Get top N recommendations by priority.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\analysis.py:140](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\analysis.py#L140)*

##### `summary`

```python
def summary(self) -> str
```

Generate text summary.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\analysis.py:164](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\analysis.py#L164)*

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\analysis.py:149](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\analysis.py#L149)*

### class `PerformanceAnalyzer`

Analyzes performance profiles and generates recommendations.

#### Methods

##### `__init__`

```python
def __init__(self)
```

Initialize analyzer.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\analysis.py:190](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\analysis.py#L190)*

##### `analyze`

```python
def analyze(self, profile_result: Any) -> analysis.PerformanceAnalysis
```

Analyze profile results.

**Parameters:**

- **profile_result** (`typing.Any`): ProfileResult from profiler

**Returns**: `<class 'analysis.PerformanceAnalysis'>` - PerformanceAnalysis with bottlenecks and recommendations

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\analysis.py:199](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\analysis.py#L199)*

## Functions

### `analyze_performance`

```python
def analyze_performance(profile_result: Any) -> analysis.PerformanceAnalysis
```

Analyze performance from profile results.

**Parameters:**

- **profile_result** (`typing.Any`): ProfileResult from profiler

**Returns**: `<class 'analysis.PerformanceAnalysis'>` - PerformanceAnalysis with recommendations

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\analysis.py:430](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\analysis.py#L430)*

### `identify_bottlenecks`

```python
def identify_bottlenecks(profile_result: Any) -> List[analysis.BottleneckAnalysis]
```

Identify bottlenecks from profile results.

**Parameters:**

- **profile_result** (`typing.Any`): ProfileResult from profiler

**Returns**: `typing.List[analysis.BottleneckAnalysis]` - List of identified bottlenecks

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\analysis.py:444](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\analysis.py#L444)*

### `recommend_optimizations`

```python
def recommend_optimizations(profile_result: Any, max_recommendations: int = 5) -> List[analysis.OptimizationRecommendation]
```

Generate optimization recommendations.

**Parameters:**

- **profile_result** (`typing.Any`): ProfileResult from profiler
- **max_recommendations** (`<class 'int'>`): Maximum number of recommendations

**Returns**: `typing.List[analysis.OptimizationRecommendation]` - List of optimization recommendations

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\analysis.py:459](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\analysis.py#L459)*
