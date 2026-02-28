# Module `benchmarks.reports`

Benchmark reporting utilities.

This module provides report generation for benchmark results
in various formats (Markdown, JSON, CSV, HTML).

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `BenchmarkReport`

Benchmark report container.

Aggregates benchmark results and generates formatted reports.

#### Attributes

- **title** (`<class 'str'>`): 
- **description** (`<class 'str'>`): 
- **timestamp** (`<class 'float'>`): 
- **results** (`typing.List[typing.Dict[str, typing.Any]]`): 
- **summary** (`typing.Dict[str, typing.Any]`): 
- **environment** (`typing.Dict[str, typing.Any]`): 
- **configuration** (`typing.Dict[str, typing.Any]`): 

#### Methods

##### `__init__`

```python
def __init__(self, title: str = 'TensorRT Benchmark Report', description: str = '', timestamp: float = <factory>, results: List[Dict[str, Any]] = <factory>, summary: Dict[str, Any] = <factory>, environment: Dict[str, Any] = <factory>, configuration: Dict[str, Any] = <factory>) -> None
```

##### `add_result`

```python
def add_result(self, result: Any)
```

Add benchmark result.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\benchmarks\reports.py:44](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\benchmarks\reports.py#L44)*

##### `add_results`

```python
def add_results(self, results: List[Any])
```

Add multiple benchmark results.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\benchmarks\reports.py:51](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\benchmarks\reports.py#L51)*

##### `export`

```python
def export(self, path: Union[str, pathlib.Path], format: reports.ReportFormat = <ReportFormat.MARKDOWN: 'md'>)
```

Export report to file.

**Parameters:**

- **path** (`typing.Union[str, pathlib.Path]`): Output file path
- **format** (`<enum 'ReportFormat'>`): Report format

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\benchmarks\reports.py:324](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\benchmarks\reports.py#L324)*

##### `set_environment`

```python
def set_environment(self, env: Dict[str, Any])
```

Set environment info.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\benchmarks\reports.py:60](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\benchmarks\reports.py#L60)*

##### `set_summary`

```python
def set_summary(self, summary: Dict[str, Any])
```

Set report summary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\benchmarks\reports.py:56](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\benchmarks\reports.py#L56)*

##### `to_csv`

```python
def to_csv(self) -> str
```

Generate CSV report.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\benchmarks\reports.py:219](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\benchmarks\reports.py#L219)*

##### `to_html`

```python
def to_html(self) -> str
```

Generate HTML report.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\benchmarks\reports.py:240](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\benchmarks\reports.py#L240)*

##### `to_json`

```python
def to_json(self, indent: int = 2) -> str
```

Generate JSON report.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\benchmarks\reports.py:203](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\benchmarks\reports.py#L203)*

##### `to_markdown`

```python
def to_markdown(self) -> str
```

Generate Markdown report.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\benchmarks\reports.py:64](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\benchmarks\reports.py#L64)*

##### `to_text`

```python
def to_text(self) -> str
```

Generate plain text report.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\benchmarks\reports.py:273](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\benchmarks\reports.py#L273)*

### class `ReportFormat`(Enum)

Report output formats.

## Functions

### `export_to_csv`

```python
def export_to_csv(results: List[Any], path: Union[str, pathlib.Path])
```

Export results to CSV file.

**Parameters:**

- **results** (`typing.List[typing.Any]`): Benchmark results
- **path** (`typing.Union[str, pathlib.Path]`): Output file path

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\benchmarks\reports.py:400](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\benchmarks\reports.py#L400)*

### `export_to_json`

```python
def export_to_json(results: List[Any], path: Union[str, pathlib.Path])
```

Export results to JSON file.

**Parameters:**

- **results** (`typing.List[typing.Any]`): Benchmark results
- **path** (`typing.Union[str, pathlib.Path]`): Output file path

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\benchmarks\reports.py:416](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\benchmarks\reports.py#L416)*

### `export_to_markdown`

```python
def export_to_markdown(results: List[Any], path: Union[str, pathlib.Path], title: str = 'TensorRT Benchmark Report')
```

Export results to Markdown file.

**Parameters:**

- **results** (`typing.List[typing.Any]`): Benchmark results
- **path** (`typing.Union[str, pathlib.Path]`): Output file path
- **title** (`<class 'str'>`): Report title

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\benchmarks\reports.py:432](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\benchmarks\reports.py#L432)*

### `generate_report`

```python
def generate_report(results: List[Any], title: str = 'TensorRT Benchmark Report', format: reports.ReportFormat = <ReportFormat.MARKDOWN: 'md'>) -> str
```

Generate benchmark report from results.

**Parameters:**

- **results** (`typing.List[typing.Any]`): List of benchmark results
- **title** (`<class 'str'>`): Report title
- **format** (`<enum 'ReportFormat'>`): Output format

**Returns**: `<class 'str'>` - Formatted report string

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\benchmarks\reports.py:352](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\benchmarks\reports.py#L352)*
