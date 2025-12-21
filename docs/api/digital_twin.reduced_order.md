# Module `digital_twin.reduced_order`

Reduced-Order Models for real-time digital twin simulation.

This module provides fast physics-based surrogates derived from high-fidelity
CFD simulations. These models enable real-time prediction while maintaining
physical fidelity for digital twin applications.

Key methods:
    - Proper Orthogonal Decomposition (POD): Data-driven basis reduction
    - Dynamic Mode Decomposition (DMD): Linear dynamics extraction
    - Autoencoders: Nonlinear manifold learning with neural networks

Author: HyperTensor Team

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `AutoencoderROM`(ReducedOrderModel)

Autoencoder-based reduced-order model.

Uses neural networks to learn nonlinear mappings between
high-dimensional and latent spaces.

#### Properties

##### `input_dim`

```python
def input_dim(self) -> int
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py:85](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py#L85)*

##### `latent_dim`

```python
def latent_dim(self) -> int
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py:91](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py#L91)*

#### Methods

##### `__init__`

```python
def __init__(self, config: reduced_order.ROMConfig)
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py:374](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py#L374)*

##### `decode`

```python
def decode(self, z: torch.Tensor) -> torch.Tensor
```

Decode from latent space.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py:525](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py#L525)*

##### `encode`

```python
def encode(self, x: torch.Tensor) -> torch.Tensor
```

Encode to latent space.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py:509](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py#L509)*

##### `train_from_snapshots`

```python
def train_from_snapshots(self, snapshots: torch.Tensor, n_epochs: int = 100, batch_size: int = 32, lr: float = 0.001)
```

Train autoencoder on snapshot data.

**Parameters:**

- **snapshots** (`<class 'torch.Tensor'>`): Training data of shape (n_snapshots, n_dof)
- **n_epochs** (`<class 'int'>`): Number of training epochs
- **batch_size** (`<class 'int'>`): Batch size for training
- **lr** (`<class 'float'>`): Learning rate

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py:423](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py#L423)*

### class `DMDModel`(ReducedOrderModel)

Dynamic Mode Decomposition (DMD) model.

Extracts dominant spatiotemporal modes from time-series data,
enabling prediction of future states based on linear dynamics.

#### Properties

##### `input_dim`

```python
def input_dim(self) -> int
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py:85](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py#L85)*

##### `latent_dim`

```python
def latent_dim(self) -> int
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py:91](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py#L91)*

#### Methods

##### `__init__`

```python
def __init__(self, config: reduced_order.ROMConfig)
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py:246](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py#L246)*

##### `decode`

```python
def decode(self, z: torch.Tensor) -> torch.Tensor
```

Reconstruct from DMD amplitudes.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py:322](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py#L322)*

##### `encode`

```python
def encode(self, x: torch.Tensor) -> torch.Tensor
```

Project to DMD mode amplitudes.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py:308](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py#L308)*

##### `predict`

```python
def predict(self, x0: torch.Tensor, n_steps: int) -> torch.Tensor
```

Predict future states using DMD dynamics.

**Parameters:**

- **x0** (`<class 'torch.Tensor'>`): Initial state (n_dof,)
- **n_steps** (`<class 'int'>`): Number of time steps to predict

**Returns**: `<class 'torch.Tensor'>` - Predicted states of shape (n_steps, n_dof)

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py:333](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py#L333)*

##### `train_from_snapshots`

```python
def train_from_snapshots(self, snapshots: torch.Tensor)
```

Compute DMD from time-series snapshots.

**Parameters:**

- **snapshots** (`<class 'torch.Tensor'>`): Tensor of shape (n_snapshots, n_dof)  where snapshots are sequential in time

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py:256](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py#L256)*

### class `PODModel`(ReducedOrderModel)

Proper Orthogonal Decomposition (POD) reduced-order model.

Uses SVD to extract optimal linear basis functions from snapshot data.
Also known as Principal Component Analysis (PCA) in other contexts.

#### Properties

##### `input_dim`

```python
def input_dim(self) -> int
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py:85](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py#L85)*

##### `latent_dim`

```python
def latent_dim(self) -> int
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py:91](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py#L91)*

#### Methods

##### `__init__`

```python
def __init__(self, config: reduced_order.ROMConfig)
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py:172](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py#L172)*

##### `decode`

```python
def decode(self, z: torch.Tensor) -> torch.Tensor
```

Reconstruct from POD coefficients.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py:228](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py#L228)*

##### `encode`

```python
def encode(self, x: torch.Tensor) -> torch.Tensor
```

Project to POD coefficients.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py:219](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py#L219)*

##### `train_from_snapshots`

```python
def train_from_snapshots(self, snapshots: torch.Tensor)
```

Compute POD basis from snapshot matrix.

**Parameters:**

- **snapshots** (`<class 'torch.Tensor'>`): Tensor of shape (n_snapshots, n_dof)

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py:179](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py#L179)*

### class `ROMConfig`

Configuration for reduced-order model.

#### Attributes

- **n_modes** (`<class 'int'>`): 
- **energy_threshold** (`<class 'float'>`): 
- **n_snapshots** (`<class 'int'>`): 
- **validation_split** (`<class 'float'>`): 
- **hidden_dims** (`typing.List[int]`): 
- **activation** (`<class 'str'>`): 
- **dropout** (`<class 'float'>`): 
- **l2_weight** (`<class 'float'>`): 
- **dmd_rank** (`typing.Optional[int]`): 
- **dmd_dt** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, n_modes: int = 50, energy_threshold: float = 0.99, n_snapshots: int = 1000, validation_split: float = 0.2, hidden_dims: List[int] = <factory>, activation: str = 'relu', dropout: float = 0.1, l2_weight: float = 1e-05, dmd_rank: Optional[int] = None, dmd_dt: float = 0.001) -> None
```

### class `ROMMetrics`

Quality metrics for reduced-order model.

#### Attributes

- **projection_error** (`<class 'float'>`): 
- **reconstruction_error** (`<class 'float'>`): 
- **energy_captured** (`<class 'float'>`): 
- **n_modes** (`<class 'int'>`): 
- **compression_ratio** (`<class 'float'>`): 
- **prediction_error** (`typing.Optional[float]`): 

#### Methods

##### `__init__`

```python
def __init__(self, projection_error: float, reconstruction_error: float, energy_captured: float, n_modes: int, compression_ratio: float, prediction_error: Optional[float] = None) -> None
```

### class `ROMType`(Enum)

Type of reduced-order model.

### class `ReducedOrderModel`(Module)

Base class for reduced-order models.

Provides common interface for all ROM types including
encoding, decoding, and prediction capabilities.

#### Properties

##### `input_dim`

```python
def input_dim(self) -> int
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py:85](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py#L85)*

##### `latent_dim`

```python
def latent_dim(self) -> int
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py:91](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py#L91)*

#### Methods

##### `__init__`

```python
def __init__(self, config: reduced_order.ROMConfig)
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py:76](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py#L76)*

##### `compute_metrics`

```python
def compute_metrics(self, snapshots: torch.Tensor) -> reduced_order.ROMMetrics
```

Compute quality metrics on test data.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py:131](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py#L131)*

##### `decode`

```python
def decode(self, z: torch.Tensor) -> torch.Tensor
```

Reconstruct high-dimensional state from latent representation.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py:110](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py#L110)*

##### `denormalize`

```python
def denormalize(self, x: torch.Tensor) -> torch.Tensor
```

Denormalize output using stored statistics.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py:125](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py#L125)*

##### `encode`

```python
def encode(self, x: torch.Tensor) -> torch.Tensor
```

Project high-dimensional state to latent space.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py:106](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py#L106)*

##### `forward`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Encode then decode (reconstruction).

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py:114](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py#L114)*

##### `normalize`

```python
def normalize(self, x: torch.Tensor) -> torch.Tensor
```

Normalize input using stored statistics.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py:119](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py#L119)*

##### `train_from_snapshots`

```python
def train_from_snapshots(self, snapshots: torch.Tensor)
```

Train ROM from snapshot matrix.

**Parameters:**

- **snapshots** (`<class 'torch.Tensor'>`): Tensor of shape (n_snapshots, n_dof)

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py:97](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py#L97)*

## Functions

### `compute_projection_error`

```python
def compute_projection_error(snapshots: torch.Tensor, n_modes: int) -> float
```

Compute projection error for given number of POD modes.

Useful for determining optimal number of modes.

**Parameters:**

- **snapshots** (`<class 'torch.Tensor'>`): Snapshot matrix
- **n_modes** (`<class 'int'>`): Number of modes to use

**Returns**: `<class 'float'>` - Relative projection error

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py:587](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py#L587)*

### `create_rom_from_snapshots`

```python
def create_rom_from_snapshots(snapshots: torch.Tensor, rom_type: reduced_order.ROMType = <ROMType.POD: 1>, config: Optional[reduced_order.ROMConfig] = None) -> reduced_order.ReducedOrderModel
```

Factory function to create and train a ROM.

**Parameters:**

- **snapshots** (`<class 'torch.Tensor'>`): Snapshot matrix of shape (n_snapshots, n_dof)
- **rom_type** (`<enum 'ROMType'>`): Type of ROM to create
- **config** (`typing.Optional[reduced_order.ROMConfig]`): Configuration (uses defaults if None) Default: `if None)`.

**Returns**: `<class 'reduced_order.ReducedOrderModel'>` - Trained reduced-order model

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py:542](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py#L542)*

### `test_reduced_order`

```python
def test_reduced_order()
```

Test reduced-order model implementations.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py:620](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py#L620)*

### `validate_rom_accuracy`

```python
def validate_rom_accuracy(model: reduced_order.ReducedOrderModel, test_snapshots: torch.Tensor) -> reduced_order.ROMMetrics
```

Validate ROM accuracy on test data.

**Parameters:**

- **model** (`<class 'reduced_order.ReducedOrderModel'>`): Trained ROM
- **test_snapshots** (`<class 'torch.Tensor'>`): Test data of shape (n_test, n_dof)

**Returns**: `<class 'reduced_order.ROMMetrics'>` - Quality metrics

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py:572](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\reduced_order.py#L572)*
