# Module `integration.config`

Configuration Management for Project HyperTensor.

Provides hierarchical configuration with:
- Multiple sources (files, environment, defaults)
- Type validation
- Environment-specific overrides
- Secure credential handling

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `ConfigManager`

Central configuration manager.

Handles loading, merging, and accessing configuration from
multiple sources with proper precedence.

#### Methods

##### `__init__`

```python
def __init__(self)
```

Initialize config manager.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py:299](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py#L299)*

##### `get`

```python
def get(self, path: str, default: Any = None) -> Any
```

Get configuration value by path.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py:403](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py#L403)*

##### `get_instance`

```python
def get_instance() -> 'ConfigManager'
```

Get singleton instance.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py:308](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py#L308)*

##### `load_environment`

```python
def load_environment(self, prefix: str = 'HYPERTENSOR')
```

Load configuration from environment variables.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py:398](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py#L398)*

##### `load_file`

```python
def load_file(self, path: Union[str, pathlib.Path], format: str = 'auto') -> bool
```

Load configuration from file.

**Parameters:**

- **path** (`typing.Union[str, pathlib.Path]`): Path to config file
- **format** (`<class 'str'>`): File format (json, yaml, auto)

**Returns**: `<class 'bool'>` - Whether load was successful

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py:354](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py#L354)*

##### `save`

```python
def save(self, path: Union[str, pathlib.Path], format: str = 'json')
```

Save configuration to file.

**Parameters:**

- **path** (`typing.Union[str, pathlib.Path]`): Output path
- **format** (`<class 'str'>`): File format

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py:427](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py#L427)*

##### `set`

```python
def set(self, path: str, value: Any)
```

Set configuration value.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py:407](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py#L407)*

##### `to_dict`

```python
def to_dict(self) -> Dict
```

Export configuration to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py:423](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py#L423)*

##### `watch`

```python
def watch(self, callback: Callable[[str, Any], NoneType])
```

Register a configuration change watcher.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py:419](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py#L419)*

### class `ConfigSection`

Section of related configuration values.

#### Attributes

- **name** (`<class 'str'>`): 
- **values** (`typing.Dict[str, config.ConfigValue]`): 
- **description** (`<class 'str'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, name: str, values: Dict[str, config.ConfigValue] = <factory>, description: str = '') -> None
```

##### `define`

```python
def define(self, key: str, default: Any = None, dtype: Optional[Type] = None, description: str = '', required: bool = False, sensitive: bool = False)
```

Define a configuration value with metadata.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py:104](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py#L104)*

##### `get`

```python
def get(self, key: str, default: Any = None) -> Any
```

Get a configuration value.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py:98](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py#L98)*

##### `set`

```python
def set(self, key: str, value: Any, source: config.ConfigSource = <ConfigSource.OVERRIDE: 4>)
```

Set a configuration value.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py:86](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py#L86)*

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert section to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py:124](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py#L124)*

##### `validate`

```python
def validate(self) -> List[str]
```

Validate all values, returning list of errors.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py:128](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py#L128)*

### class `ConfigSource`(Enum)

Configuration value source.

### class `ConfigValidator`

Configuration validation with custom rules.

#### Methods

##### `__init__`

```python
def __init__(self)
```

Initialize validator.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py:448](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py#L448)*

##### `add_rule`

```python
def add_rule(self, rule: Callable[[config.Configuration], List[str]])
```

Add a validation rule.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py:452](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py#L452)*

##### `validate`

```python
def validate(self, config: config.Configuration) -> List[str]
```

Validate configuration.

**Returns**: `typing.List[str]` - List of error messages

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py:456](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py#L456)*

### class `ConfigValue`

Single configuration value with metadata.

#### Attributes

- **key** (`<class 'str'>`): 
- **value** (`typing.Any`): 
- **default** (`typing.Any`): 
- **source** (`<enum 'ConfigSource'>`): 
- **dtype** (`typing.Optional[typing.Type]`): 
- **description** (`<class 'str'>`): 
- **required** (`<class 'bool'>`): 
- **sensitive** (`<class 'bool'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, key: str, value: Any, default: Any = None, source: config.ConfigSource = <ConfigSource.DEFAULT: 1>, dtype: Optional[Type] = None, description: str = '', required: bool = False, sensitive: bool = False) -> None
```

##### `get`

```python
def get(self) -> Any
```

Get the value, returning default if None.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py:62](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py#L62)*

##### `validate`

```python
def validate(self) -> bool
```

Validate the value against its type.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py:52](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py#L52)*

### class `Configuration`

Complete configuration with multiple sections.

#### Attributes

- **name** (`<class 'str'>`): 
- **version** (`<class 'str'>`): 
- **sections** (`typing.Dict[str, config.ConfigSection]`): 

#### Methods

##### `__init__`

```python
def __init__(self, name: str = 'hypertensor', version: str = '1.0', sections: Dict[str, config.ConfigSection] = <factory>) -> None
```

##### `add_section`

```python
def add_section(self, section: config.ConfigSection)
```

Add a configuration section.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py:153](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py#L153)*

##### `get`

```python
def get(self, path: str, default: Any = None) -> Any
```

Get a value by dot-separated path.

**Parameters:**

- **path** (`<class 'str'>`): Path like "section.key"
- **default** (`typing.Any`): Default if not found Default: `if not found`.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py:161](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py#L161)*

##### `get_section`

```python
def get_section(self, name: str) -> Optional[config.ConfigSection]
```

Get a configuration section.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py:157](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py#L157)*

##### `merge`

```python
def merge(self, other: 'Configuration', source: config.ConfigSource = <ConfigSource.OVERRIDE: 4>)
```

Merge another configuration into this one.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py:209](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py#L209)*

##### `set`

```python
def set(self, path: str, value: Any, source: config.ConfigSource = <ConfigSource.OVERRIDE: 4>)
```

Set a value by dot-separated path.

**Parameters:**

- **path** (`<class 'str'>`): Path like "section.key"
- **value** (`typing.Any`): Value to set
- **source** (`<enum 'ConfigSource'>`): Source of the value

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py:179](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py#L179)*

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Dict[str, Any]]
```

Convert to nested dictionary.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py:198](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py#L198)*

##### `validate`

```python
def validate(self) -> List[str]
```

Validate all sections, returning list of errors.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py:202](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py#L202)*

### class `EnvironmentConfig`

Configuration from environment variables.

Follows convention: HYPERTENSOR_SECTION_KEY = value

#### Methods

##### `load`

```python
def load(prefix: Optional[str] = None) -> config.Configuration
```

Load configuration from environment.

**Parameters:**

- **prefix** (`typing.Optional[str]`): Environment variable prefix

**Returns**: `<class 'config.Configuration'>` - Configuration from environment

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py:228](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py#L228)*

## Functions

### `get_config`

```python
def get_config() -> config.ConfigManager
```

Get the global configuration manager.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py:479](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py#L479)*

### `load_config`

```python
def load_config(path: Union[str, pathlib.Path]) -> bool
```

Load configuration from file.

**Parameters:**

- **path** (`typing.Union[str, pathlib.Path]`): Config file path

**Returns**: `<class 'bool'>` - Whether load was successful

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py:484](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py#L484)*

### `merge_configs`

```python
def merge_configs(base: config.Configuration, override: config.Configuration) -> config.Configuration
```

Merge two configurations.

**Parameters:**

- **base** (`<class 'config.Configuration'>`): Base configuration
- **override** (`<class 'config.Configuration'>`): Override configuration

**Returns**: `<class 'config.Configuration'>` - Merged configuration

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py:507](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py#L507)*

### `save_config`

```python
def save_config(path: Union[str, pathlib.Path])
```

Save configuration to file.

**Parameters:**

- **path** (`typing.Union[str, pathlib.Path]`): Output path

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py:497](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py#L497)*

### `validate_config`

```python
def validate_config(config: Optional[config.Configuration] = None) -> List[str]
```

Validate configuration.

**Parameters:**

- **config** (`typing.Optional[config.Configuration]`): Configuration to validate (uses global if None)

**Returns**: `typing.List[str]` - List of validation errors

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py:526](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\integration\config.py#L526)*
