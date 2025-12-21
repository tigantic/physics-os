# Module `quantum.error_mitigation`

Error Mitigation and Correction for Quantum-Classical Hybrid Algorithms ========================================================================

Implements techniques to mitigate and correct errors in near-term quantum
devices and quantum-inspired simulations.

Key Components:
    - Zero-Noise Extrapolation (ZNE)
    - Probabilistic Error Cancellation (PEC)
    - Clifford Data Regression (CDR)
    - Quantum Error Correction codes
    - Noise-aware variational optimization

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `BitFlipCode`(QECCode)

3-qubit bit-flip code.

|0_L⟩ = |000⟩
|1_L⟩ = |111⟩

Corrects single bit-flip (X) errors.

#### Properties

##### `distance`

```python
def distance(self) -> int
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:651](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L651)*

##### `n_logical`

```python
def n_logical(self) -> int
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:647](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L647)*

##### `n_physical`

```python
def n_physical(self) -> int
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:643](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L643)*

#### Methods

##### `correct_error`

```python
def correct_error(self, state: torch.Tensor, syndrome: int) -> torch.Tensor
```

Apply correction based on syndrome.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:712](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L712)*

##### `decode`

```python
def decode(self, physical_state: torch.Tensor) -> torch.Tensor
```

Decode by majority voting.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:672](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L672)*

##### `encode`

```python
def encode(self, logical_state: torch.Tensor) -> torch.Tensor
```

Encode single logical qubit.

|0⟩ → |000⟩
|1⟩ → |111⟩

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:655](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L655)*

##### `syndrome_measure`

```python
def syndrome_measure(self, state: torch.Tensor) -> torch.Tensor
```

Measure syndrome bits.

Syndrome = (Z₀Z₁, Z₁Z₂)

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:686](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L686)*

### class `CDRConfig`

Configuration for Clifford Data Regression.

#### Attributes

- **n_training_circuits** (`<class 'int'>`): 
- **regression_method** (`<class 'str'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, n_training_circuits: int = 50, regression_method: str = 'linear') -> None
```

### class `CliffordDataRegression`

Clifford Data Regression for error mitigation.

Uses Clifford circuits (efficiently simulable) to learn error model,
then applies correction to non-Clifford circuits.

#### Methods

##### `__init__`

```python
def __init__(self, config: Optional[error_mitigation.CDRConfig] = None)
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:503](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L503)*

##### `generate_training_circuits`

```python
def generate_training_circuits(self, template_circuit: List, n_circuits: int) -> List[Tuple[List, float]]
```

Generate Clifford training circuits near the target.

**Returns**: `typing.List[typing.Tuple[typing.List, float]]` - List of (circuit, ideal_value) pairs

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:507](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L507)*

##### `mitigate`

```python
def mitigate(self, noisy_value: float) -> float
```

Apply learned correction.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:585](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L585)*

##### `train`

```python
def train(self, training_data: List[Tuple[List, float]], noisy_executor: Callable[[List], float])
```

Train regression model on Clifford data.

**Parameters:**

- **training_data** (`typing.List[typing.Tuple[typing.List, float]]`): List of (circuit, ideal_value) pairs
- **noisy_executor** (`typing.Callable[[typing.List], float]`): Function to run circuits with noise

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:551](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L551)*

### class `ExtrapolationMethod`(Enum)

Extrapolation methods for ZNE.

### class `KrausChannel`

Kraus representation of a quantum channel.

ρ → Σ_k K_k ρ K_k†

#### Methods

##### `__init__`

```python
def __init__(self, kraus_ops: List[torch.Tensor])
```

Args:

kraus_ops: List of Kraus operators

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:136](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L136)*

##### `amplitude_damping`

```python
def amplitude_damping(gamma: float) -> 'KrausChannel'
```

Create amplitude damping channel with decay rate gamma.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:176](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L176)*

##### `apply`

```python
def apply(self, rho: torch.Tensor) -> torch.Tensor
```

Apply channel to density matrix.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:154](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L154)*

##### `bit_flip`

```python
def bit_flip(p: float) -> 'KrausChannel'
```

Create bit-flip channel.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:190](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L190)*

##### `depolarizing`

```python
def depolarizing(p: float) -> 'KrausChannel'
```

Create depolarizing channel with probability p.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:161](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L161)*

##### `phase_damping`

```python
def phase_damping(gamma: float) -> 'KrausChannel'
```

Create phase damping channel.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:183](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L183)*

##### `phase_flip`

```python
def phase_flip(p: float) -> 'KrausChannel'
```

Create phase-flip channel.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:197](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L197)*

### class `NoiseAwareOptimizer`

Variational optimizer that accounts for noise effects.

Incorporates:
- Error mitigation in objective evaluation
- Noise-robustness as optimization objective
- Variance reduction techniques

#### Methods

##### `__init__`

```python
def __init__(self, objective: Callable[[torch.Tensor], float], n_params: int, config: Optional[error_mitigation.NoiseAwareVQEConfig] = None)
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:935](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L935)*

##### `mitigated_objective`

```python
def mitigated_objective(self, params: torch.Tensor, n_samples: int = 10) -> Tuple[float, float]
```

Evaluate objective with error mitigation.

**Returns**: `typing.Tuple[float, float]` - (mitigated_value, uncertainty)

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:955](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L955)*

##### `optimize`

```python
def optimize(self, initial_params: torch.Tensor, max_iterations: int = 100, learning_rate: float = 0.01, verbose: bool = True) -> Dict
```

Run noise-aware optimization.

**Returns**: `typing.Dict` - Optimization results

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:988](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L988)*

### class `NoiseAwareVQEConfig`

Configuration for noise-aware VQE.

#### Attributes

- **noise_model** (`typing.Optional[error_mitigation.NoiseModel]`): 
- **mitigation_method** (`<class 'str'>`): 
- **sample_variance_penalty** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, noise_model: Optional[error_mitigation.NoiseModel] = None, mitigation_method: str = 'zne', sample_variance_penalty: float = 0.1) -> None
```

### class `NoiseChannel`

Single noise channel specification.

#### Attributes

- **noise_type** (`<enum 'NoiseType'>`): 
- **probability** (`<class 'float'>`): 
- **target_qubits** (`typing.Optional[typing.List[int]]`): 

#### Methods

##### `__init__`

```python
def __init__(self, noise_type: error_mitigation.NoiseType, probability: float, target_qubits: Optional[List[int]] = None) -> None
```

### class `NoiseModel`

Complete noise model for a quantum device.

Combines multiple noise channels affecting different operations.

#### Attributes

- **channels** (`typing.List[error_mitigation.NoiseChannel]`): 
- **gate_errors** (`typing.Dict[str, float]`): 
- **readout_errors** (`typing.Dict[int, typing.Tuple[float, float]]`): 
- **t1_times** (`typing.Optional[typing.Dict[int, float]]`): 
- **t2_times** (`typing.Optional[typing.Dict[int, float]]`): 

#### Methods

##### `__init__`

```python
def __init__(self, channels: List[error_mitigation.NoiseChannel] = <factory>, gate_errors: Dict[str, float] = <factory>, readout_errors: Dict[int, Tuple[float, float]] = <factory>, t1_times: Optional[Dict[int, float]] = None, t2_times: Optional[Dict[int, float]] = None) -> None
```

##### `add_amplitude_damping`

```python
def add_amplitude_damping(self, gamma: float, qubits: Optional[List[int]] = None)
```

Add amplitude damping (T1 decay).

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:78](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L78)*

##### `add_depolarizing`

```python
def add_depolarizing(self, p: float, qubits: Optional[List[int]] = None)
```

Add depolarizing noise channel.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:73](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L73)*

##### `add_phase_damping`

```python
def add_phase_damping(self, gamma: float, qubits: Optional[List[int]] = None)
```

Add phase damping (T2 decay).

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:83](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L83)*

##### `add_readout_error`

```python
def add_readout_error(self, qubit: int, p0_to_1: float, p1_to_0: float)
```

Add readout error for a qubit.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:88](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L88)*

##### `from_device_params`

```python
def from_device_params(n_qubits: int, single_qubit_error: float = 0.001, two_qubit_error: float = 0.01, readout_error: float = 0.02, t1_us: float = 50.0, t2_us: float = 70.0) -> 'NoiseModel'
```

Create noise model from typical device parameters.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:98](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L98)*

##### `set_gate_error`

```python
def set_gate_error(self, gate_name: str, error_rate: float)
```

Set error rate for a specific gate type.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:93](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L93)*

### class `NoiseType`(Enum)

Types of quantum noise channels.

### class `PECConfig`

Configuration for Probabilistic Error Cancellation.

#### Attributes

- **n_samples** (`<class 'int'>`): 
- **noise_model** (`typing.Optional[error_mitigation.NoiseModel]`): 

#### Methods

##### `__init__`

```python
def __init__(self, n_samples: int = 1000, noise_model: Optional[error_mitigation.NoiseModel] = None) -> None
```

### class `PhaseFlipCode`(QECCode)

3-qubit phase-flip code.

|0_L⟩ = |+++⟩
|1_L⟩ = |---⟩

Corrects single phase-flip (Z) errors.

#### Properties

##### `distance`

```python
def distance(self) -> int
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:758](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L758)*

##### `n_logical`

```python
def n_logical(self) -> int
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:754](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L754)*

##### `n_physical`

```python
def n_physical(self) -> int
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:750](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L750)*

#### Methods

##### `decode`

```python
def decode(self, physical_state: torch.Tensor) -> torch.Tensor
```

Decode phase-flip code.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:781](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L781)*

##### `encode`

```python
def encode(self, logical_state: torch.Tensor) -> torch.Tensor
```

Encode into phase-flip code.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:762](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L762)*

##### `syndrome_measure`

```python
def syndrome_measure(self, state: torch.Tensor) -> torch.Tensor
```

Measure phase-flip syndrome in X basis.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:803](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L803)*

### class `ProbabilisticErrorCancellation`

Probabilistic Error Cancellation.

Represents noisy gates as quasi-probability distributions over
ideal operations and samples to reconstruct ideal expectation.

#### Methods

##### `__init__`

```python
def __init__(self, noise_model: error_mitigation.NoiseModel, config: Optional[error_mitigation.PECConfig] = None)
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:383](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L383)*

##### `decompose_noisy_gate`

```python
def decompose_noisy_gate(self, gate_name: str) -> List[Tuple[float, Callable]]
```

Decompose noisy gate into quasi-probability sum of ideal gates.

For depolarizing noise: N(ρ) = (1-p)I(ρ) + (p/3)[X(ρ) + Y(ρ) + Z(ρ)]

Inverse: I(ρ) = (1/(1-p))[N(ρ) - (p/3)(X + Y + Z)]

**Returns**: `typing.List[typing.Tuple[float, typing.Callable]]` - List of (quasi_probability, gate_operation) tuples

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:390](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L390)*

##### `mitigate`

```python
def mitigate(self, circuit_executor: Callable[[List[str]], float], n_gates: int) -> Tuple[float, float]
```

Apply PEC mitigation via Monte Carlo sampling.

**Parameters:**

- **circuit_executor** (`typing.Callable[[typing.List[str]], float]`): Function(corrections) -> expectation
- **n_gates** (`<class 'int'>`): Number of gates in circuit

**Returns**: `typing.Tuple[float, float]` - (mitigated_value, statistical_error)

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:437](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L437)*

##### `sampling_overhead`

```python
def sampling_overhead(self) -> float
```

Compute sampling overhead (cost factor).

For PEC, the variance increases by γ² where γ = Σ|c_i|.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:423](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L423)*

### class `QECCode`(ABC)

Abstract base class for quantum error correction codes.

#### Properties

##### `distance`

```python
def distance(self) -> int
```

Code distance (number of errors that can be detected).

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:611](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L611)*

##### `n_logical`

```python
def n_logical(self) -> int
```

Number of logical qubits.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:605](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L605)*

##### `n_physical`

```python
def n_physical(self) -> int
```

Number of physical qubits.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:599](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L599)*

#### Methods

##### `decode`

```python
def decode(self, physical_state: torch.Tensor) -> torch.Tensor
```

Decode physical state to logical qubits.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:622](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L622)*

##### `encode`

```python
def encode(self, logical_state: torch.Tensor) -> torch.Tensor
```

Encode logical state into physical qubits.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:617](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L617)*

##### `syndrome_measure`

```python
def syndrome_measure(self, state: torch.Tensor) -> torch.Tensor
```

Measure error syndrome.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:627](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L627)*

### class `ShorCode`(QECCode)

9-qubit Shor code.

Corrects arbitrary single-qubit errors by concatenating
bit-flip and phase-flip codes.

#### Properties

##### `distance`

```python
def distance(self) -> int
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:828](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L828)*

##### `n_logical`

```python
def n_logical(self) -> int
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:824](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L824)*

##### `n_physical`

```python
def n_physical(self) -> int
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:820](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L820)*

#### Methods

##### `decode`

```python
def decode(self, physical_state: torch.Tensor) -> torch.Tensor
```

Decode Shor code.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:878](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L878)*

##### `encode`

```python
def encode(self, logical_state: torch.Tensor) -> torch.Tensor
```

Encode using Shor's 9-qubit code.

|0_L⟩ = (|000⟩ + |111⟩)(|000⟩ + |111⟩)(|000⟩ + |111⟩) / 2√2
|1_L⟩ = (|000⟩ - |111⟩)(|000⟩ - |111⟩)(|000⟩ - |111⟩) / 2√2

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:832](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L832)*

##### `syndrome_measure`

```python
def syndrome_measure(self, state: torch.Tensor) -> torch.Tensor
```

Measure Shor code syndrome (8 syndrome bits).

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:906](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L906)*

### class `ZNEConfig`

Configuration for Zero-Noise Extrapolation.

#### Attributes

- **scale_factors** (`typing.List[float]`): 
- **extrapolation** (`<enum 'ExtrapolationMethod'>`): 
- **folding_method** (`<class 'str'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, scale_factors: List[float] = <factory>, extrapolation: error_mitigation.ExtrapolationMethod = <ExtrapolationMethod.RICHARDSON: 'richardson'>, folding_method: str = 'global') -> None
```

### class `ZeroNoiseExtrapolator`

Zero-Noise Extrapolation error mitigation.

Runs circuit at multiple noise levels and extrapolates to zero noise.

#### Methods

##### `__init__`

```python
def __init__(self, config: Optional[error_mitigation.ZNEConfig] = None)
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:232](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L232)*

##### `extrapolate`

```python
def extrapolate(self, scale_factors: List[float], values: List[float]) -> float
```

Extrapolate to zero noise.

**Parameters:**

- **scale_factors** (`typing.List[float]`): Noise scale factors used
- **values** (`typing.List[float]`): Measured values at each scale

**Returns**: `<class 'float'>` - Extrapolated zero-noise value

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:267](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L267)*

##### `fold_circuit`

```python
def fold_circuit(self, circuit_executor: Callable[[], float], scale_factor: float) -> Callable[[], float]
```

Create noise-scaled version of circuit.

For digital noise scaling, we use unitary folding:
U → U (U† U)^n for n repetitions

**Parameters:**

- **circuit_executor** (`typing.Callable[[], float]`): Function that runs circuit and returns expectation
- **scale_factor** (`<class 'float'>`): Noise scaling factor (must be odd integer for exact folding)

**Returns**: `typing.Callable[[], float]` - Scaled circuit executor

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:235](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L235)*

##### `mitigate`

```python
def mitigate(self, circuit_executor: Callable[[float], float], observable: Optional[str] = None) -> float
```

Apply ZNE mitigation.

**Parameters:**

- **circuit_executor** (`typing.Callable[[float], float]`): Function(noise_scale) -> expectation_value
- **observable** (`typing.Optional[str]`): Optional observable name for logging

**Returns**: `<class 'float'>` - Mitigated expectation value

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:335](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L335)*

## Functions

### `apply_error_mitigation`

```python
def apply_error_mitigation(circuit_executor: Callable[[], float], method: str = 'zne', noise_model: Optional[error_mitigation.NoiseModel] = None, **kwargs) -> float
```

Apply error mitigation to a circuit execution.

**Parameters:**

- **circuit_executor** (`typing.Callable[[], float]`): Function that runs circuit and returns value
- **method** (`<class 'str'>`): Mitigation method ("zne", "pec", or "cdr")
- **noise_model** (`typing.Optional[error_mitigation.NoiseModel]`): Optional noise model for PEC

**Returns**: `<class 'float'>` - Mitigated expectation value

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:1054](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L1054)*

### `create_device_noise_model`

```python
def create_device_noise_model(device_name: str = 'ibm_perth') -> error_mitigation.NoiseModel
```

Create noise model from device calibration data.

**Parameters:**

- **device_name** (`<class 'str'>`): Name of the target device

**Returns**: `<class 'error_mitigation.NoiseModel'>` - NoiseModel configured for the device

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py:1091](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\quantum\error_mitigation.py#L1091)*
