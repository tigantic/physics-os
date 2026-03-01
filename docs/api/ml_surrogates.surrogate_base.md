# Module `ml_surrogates.surrogate_base`

Base classes for CFD surrogate models.

This module provides the foundational abstractions for neural network
surrogate models used to accelerate CFD simulations. All specific
architectures (PINN, DeepONet, FNO) inherit from these base classes.

Author: Tigantic Holdings LLC

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `CFDSurrogate`(Module, ABC)

Abstract base class for CFD surrogate models.

Provides common interface and utilities for all neural
network surrogates used to approximate CFD solutions.

#### Methods

##### `__init__`

```python
def __init__(self, config: surrogate_base.SurrogateConfig)
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py:91](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py#L91)*

##### `build_network`

```python
def build_network(self)
```

Build the neural network architecture.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py:108](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py#L108)*

##### `count_parameters`

```python
def count_parameters(self) -> int
```

Count trainable parameters.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py:173](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py#L173)*

##### `denormalize_input`

```python
def denormalize_input(self, x: torch.Tensor) -> torch.Tensor
```

Denormalize inputs.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py:132](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py#L132)*

##### `denormalize_output`

```python
def denormalize_output(self, y: torch.Tensor) -> torch.Tensor
```

Denormalize outputs.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py:144](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py#L144)*

##### `forward`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Forward pass through the network.

**Parameters:**

- **x** (`<class 'torch.Tensor'>`): Input tensor of shape (batch, input_dim)

**Returns**: `<class 'torch.Tensor'>` - Output tensor of shape (batch, output_dim)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py:113](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py#L113)*

##### `get_activation`

```python
def get_activation(self) -> torch.nn.modules.module.Module
```

Get activation function based on config.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py:177](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py#L177)*

##### `normalize_input`

```python
def normalize_input(self, x: torch.Tensor) -> torch.Tensor
```

Normalize inputs using stored statistics.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py:126](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py#L126)*

##### `normalize_output`

```python
def normalize_output(self, y: torch.Tensor) -> torch.Tensor
```

Normalize outputs using stored statistics.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py:138](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py#L138)*

##### `predict`

```python
def predict(self, x: torch.Tensor) -> torch.Tensor
```

Make predictions with automatic normalization.

**Parameters:**

- **x** (`<class 'torch.Tensor'>`): Input coordinates/parameters

**Returns**: `<class 'torch.Tensor'>` - Predicted CFD solution

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py:157](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py#L157)*

##### `set_normalization`

```python
def set_normalization(self, x_data: torch.Tensor, y_data: torch.Tensor)
```

Compute and store normalization statistics from data.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py:150](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py#L150)*

### class `MLPSurrogate`(CFDSurrogate)

Multi-layer perceptron surrogate model.

Simple but effective baseline for learning input-output mappings.

#### Methods

##### `__init__`

```python
def __init__(self, config: surrogate_base.SurrogateConfig)
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py:197](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py#L197)*

##### `build_network`

```python
def build_network(self)
```

Build MLP architecture.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py:201](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py#L201)*

##### `forward`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Forward pass through MLP.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py:223](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py#L223)*

### class `ResNetSurrogate`(CFDSurrogate)

ResNet-style surrogate with skip connections.

Better for deeper networks and complex mappings.

#### Methods

##### `__init__`

```python
def __init__(self, config: surrogate_base.SurrogateConfig, n_blocks: int = 4)
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py:254](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py#L254)*

##### `build_network`

```python
def build_network(self)
```

Build ResNet architecture.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py:259](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py#L259)*

##### `forward`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Forward pass through ResNet.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py:276](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py#L276)*

### class `ResidualBlock`(Module)

Residual block for deeper networks.

#### Methods

##### `__init__`

```python
def __init__(self, dim: int, activation: torch.nn.modules.module.Module, dropout: float = 0.0)
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py:231](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py#L231)*

##### `forward`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py:243](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py#L243)*

### class `SurrogateConfig`

Base configuration for surrogate models.

#### Attributes

- **input_dim** (`<class 'int'>`): 
- **output_dim** (`<class 'int'>`): 
- **hidden_dims** (`typing.List[int]`): 
- **activation** (`<class 'str'>`): 
- **learning_rate** (`<class 'float'>`): 
- **batch_size** (`<class 'int'>`): 
- **n_epochs** (`<class 'int'>`): 
- **weight_decay** (`<class 'float'>`): 
- **normalize_inputs** (`<class 'bool'>`): 
- **normalize_outputs** (`<class 'bool'>`): 
- **dropout** (`<class 'float'>`): 
- **layer_norm** (`<class 'bool'>`): 
- **device** (`<class 'str'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, input_dim: int = 4, output_dim: int = 5, hidden_dims: List[int] = <factory>, activation: str = 'gelu', learning_rate: float = 0.001, batch_size: int = 256, n_epochs: int = 1000, weight_decay: float = 1e-05, normalize_inputs: bool = True, normalize_outputs: bool = True, dropout: float = 0.0, layer_norm: bool = False, device: str = 'cpu') -> None
```

### class `SurrogateMetrics`

Quality metrics for surrogate model.

#### Attributes

- **mse** (`<class 'float'>`): 
- **rmse** (`<class 'float'>`): 
- **mae** (`<class 'float'>`): 
- **r2** (`<class 'float'>`): 
- **max_error** (`<class 'float'>`): 
- **relative_error** (`<class 'float'>`): 
- **inference_time** (`<class 'float'>`): 
- **n_parameters** (`<class 'int'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, mse: float, rmse: float, mae: float, r2: float, max_error: float, relative_error: float, inference_time: float, n_parameters: int) -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict[str, float]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py:69](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py#L69)*

### class `SurrogateType`(Enum)

Type of surrogate model.

## Functions

### `create_surrogate`

```python
def create_surrogate(surrogate_type: surrogate_base.SurrogateType, config: surrogate_base.SurrogateConfig) -> surrogate_base.CFDSurrogate
```

Factory function to create surrogate models.

**Parameters:**

- **surrogate_type** (`<enum 'SurrogateType'>`): Type of surrogate to create
- **config** (`<class 'surrogate_base.SurrogateConfig'>`): Model configuration

**Returns**: `<class 'surrogate_base.CFDSurrogate'>` - Initialized surrogate model

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py:338](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py#L338)*

### `evaluate_surrogate`

```python
def evaluate_surrogate(model: surrogate_base.CFDSurrogate, x_test: torch.Tensor, y_test: torch.Tensor) -> surrogate_base.SurrogateMetrics
```

Evaluate surrogate model performance.

**Parameters:**

- **model** (`<class 'surrogate_base.CFDSurrogate'>`): Trained surrogate model
- **x_test** (`<class 'torch.Tensor'>`): Test inputs
- **y_test** (`<class 'torch.Tensor'>`): Test targets

**Returns**: `<class 'surrogate_base.SurrogateMetrics'>` - Quality metrics

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py:284](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py#L284)*

### `test_surrogate_base`

```python
def test_surrogate_base()
```

Test surrogate base classes.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py:365](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\ml_surrogates\surrogate_base.py#L365)*
