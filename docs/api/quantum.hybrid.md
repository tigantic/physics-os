# Module `quantum.hybrid`

Quantum-Classical Hybrid Algorithms for Tensor Networks ========================================================

Implements quantum-inspired and hybrid quantum-classical algorithms
that leverage both tensor network methods and quantum computing primitives.

Key Components:
    - Variational Quantum Eigensolver (VQE) integration
    - Quantum Approximate Optimization Algorithm (QAOA)
    - Tensor Network Born Machines
    - Quantum-inspired classical algorithms
    - Noise-aware simulation protocols

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `AnsatzType`(Enum)

Standard variational ansatz types.

### class `GateMatrices`

Standard quantum gate matrices.

#### Methods

##### `cnot`

```python
def cnot() -> torch.Tensor
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:186](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L186)*

##### `cz`

```python
def cz() -> torch.Tensor
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:195](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L195)*

##### `hadamard`

```python
def hadamard() -> torch.Tensor
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:163](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L163)*

##### `pauli_x`

```python
def pauli_x() -> torch.Tensor
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:151](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L151)*

##### `pauli_y`

```python
def pauli_y() -> torch.Tensor
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:155](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L155)*

##### `pauli_z`

```python
def pauli_z() -> torch.Tensor
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:159](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L159)*

##### `rx`

```python
def rx(theta: float) -> torch.Tensor
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:167](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L167)*

##### `ry`

```python
def ry(theta: float) -> torch.Tensor
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:173](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L173)*

##### `rz`

```python
def rz(theta: float) -> torch.Tensor
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:179](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L179)*

### class `GateType`(Enum)

Standard quantum gate types.

### class `QAOA`

Quantum Approximate Optimization Algorithm.

Solves combinatorial optimization problems using quantum-classical hybrid.

#### Methods

##### `__init__`

```python
def __init__(self, cost_hamiltonian: List[Tuple[str, float]], n_qubits: int, config: Optional[hybrid.QAOAConfig] = None)
```

Args:

cost_hamiltonian: Problem Hamiltonian as Pauli terms
    n_qubits: Number of qubits
    config: QAOA configuration

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:627](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L627)*

##### `cost_expectation`

```python
def cost_expectation(self, sim: hybrid.TNQuantumSimulator) -> float
```

Compute expectation of cost Hamiltonian.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:646](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L646)*

##### `optimize`

```python
def optimize(self, verbose: bool = True) -> Dict
```

Optimize QAOA parameters.

**Returns**: `typing.Dict` - Optimization results

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:695](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L695)*

##### `run_circuit`

```python
def run_circuit(self, gammas: List[float], betas: List[float]) -> float
```

Run QAOA circuit and return cost expectation.

**Parameters:**

- **gammas** (`typing.List[float]`): Problem unitary angles
- **betas** (`typing.List[float]`): Mixer unitary angles

**Returns**: `<class 'float'>` - Cost expectation value

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:654](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L654)*

### class `QAOAConfig`

Configuration for QAOA.

#### Attributes

- **n_layers** (`<class 'int'>`): 
- **optimizer** (`<class 'str'>`): 
- **max_iterations** (`<class 'int'>`): 
- **chi_max** (`<class 'int'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, n_layers: int = 3, optimizer: str = 'cobyla', max_iterations: int = 100, chi_max: int = 64) -> None
```

### class `QuantumCircuit`

Quantum circuit representation.

Stores sequence of gates and provides tensor network simulation.

#### Attributes

- **n_qubits** (`<class 'int'>`): 
- **gates** (`typing.List[hybrid.QuantumGate]`): 

#### Properties

##### `depth`

```python
def depth(self) -> int
```

Circuit depth (number of time steps).

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:125](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L125)*

##### `n_parameters`

```python
def n_parameters(self) -> int
```

Total number of variational parameters.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:138](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L138)*

#### Methods

##### `__init__`

```python
def __init__(self, n_qubits: int, gates: List[hybrid.QuantumGate] = <factory>) -> None
```

##### `add_gate`

```python
def add_gate(self, gate: hybrid.QuantumGate)
```

Add a gate to the circuit.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:88](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L88)*

##### `cnot`

```python
def cnot(self, control: int, target: int)
```

Add CNOT gate.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:115](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L115)*

##### `cz`

```python
def cz(self, q1: int, q2: int)
```

Add CZ gate.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:120](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L120)*

##### `h`

```python
def h(self, qubit: int)
```

Add Hadamard gate.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:110](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L110)*

##### `rx`

```python
def rx(self, qubit: int, theta: float)
```

Add RX rotation gate.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:95](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L95)*

##### `ry`

```python
def ry(self, qubit: int, theta: float)
```

Add RY rotation gate.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:100](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L100)*

##### `rz`

```python
def rz(self, qubit: int, theta: float)
```

Add RZ rotation gate.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:105](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L105)*

### class `QuantumGate`

Representation of a quantum gate.

#### Attributes

- **gate_type** (`<enum 'GateType'>`): 
- **qubits** (`typing.Tuple[int, ...]`): 
- **parameters** (`typing.Optional[typing.Tuple[float, ...]]`): 

#### Methods

##### `__init__`

```python
def __init__(self, gate_type: hybrid.GateType, qubits: Tuple[int, ...], parameters: Optional[Tuple[float, ...]] = None) -> None
```

### class `QuantumInspiredOptimizer`

Quantum-inspired classical optimization using tensor network techniques.

Combines:
- Imaginary time evolution for ground state search
- Tensor cross interpolation for function approximation
- MPS-based variational optimization

#### Methods

##### `__init__`

```python
def __init__(self, objective: Callable[[torch.Tensor], float], n_dims: int, bounds: Tuple[float, float] = (-1.0, 1.0), resolution: int = 32)
```

Args:

objective: Function to minimize
    n_dims: Number of dimensions
    bounds: Variable bounds
    resolution: Grid resolution per dimension

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:934](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L934)*

##### `optimize_bruteforce`

```python
def optimize_bruteforce(self) -> Tuple[torch.Tensor, float]
```

Brute-force grid search (for validation).

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:956](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L956)*

##### `optimize_mps`

```python
def optimize_mps(self, chi_max: int = 16, n_sweeps: int = 10, verbose: bool = True) -> Tuple[torch.Tensor, float]
```

Optimize using MPS representation of objective landscape.

Uses alternating least squares to fit objective as TT-decomposition.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:975](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L975)*

### class `TNQuantumSimulator`

Tensor network based quantum circuit simulator.

Uses MPS representation for efficient simulation of low-entanglement circuits.

#### Methods

##### `__init__`

```python
def __init__(self, n_qubits: int, chi_max: int = 64)
```

Args:

n_qubits: Number of qubits
    chi_max: Maximum bond dimension for MPS

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:216](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L216)*

##### `apply_circuit`

```python
def apply_circuit(self, circuit: hybrid.QuantumCircuit)
```

Apply a quantum circuit to the state.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:308](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L308)*

##### `apply_single_qubit_gate`

```python
def apply_single_qubit_gate(self, gate_matrix: torch.Tensor, qubit: int)
```

Apply a single-qubit gate to the MPS.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:248](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L248)*

##### `apply_two_qubit_gate`

```python
def apply_two_qubit_gate(self, gate_matrix: torch.Tensor, qubit1: int, qubit2: int)
```

Apply a two-qubit gate using SVD compression.

For adjacent qubits, contract and re-decompose.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:257](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L257)*

##### `expectation_value`

```python
def expectation_value(self, observable_mpo: List[torch.Tensor]) -> complex
```

Compute ⟨ψ|O|ψ⟩ using MPS-MPO-MPS contraction.

**Parameters:**

- **observable_mpo** (`typing.List[torch.Tensor]`): MPO representation of observable

**Returns**: `<class 'complex'>` - Expectation value

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:342](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L342)*

##### `measure_pauli_string`

```python
def measure_pauli_string(self, paulis: str) -> complex
```

Measure expectation of Pauli string.

**Parameters:**

- **paulis** (`<class 'str'>`): String like "XZIY" (I=identity)

**Returns**: `<class 'complex'>` - Expectation value

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:379](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L379)*

### class `TensorNetworkBornMachine`

Generative model using tensor network as quantum-inspired ansatz.

Uses MPS to represent probability distribution:
P(x) = |⟨x|ψ⟩|² where |ψ⟩ is MPS

#### Methods

##### `__init__`

```python
def __init__(self, n_sites: int, local_dim: int = 2, bond_dim: int = 16)
```

Args:

n_sites: Number of visible units
    local_dim: Local Hilbert space dimension
    bond_dim: MPS bond dimension

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:763](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L763)*

##### `amplitude`

```python
def amplitude(self, config: List[int]) -> torch.Tensor
```

Compute amplitude ⟨config|ψ⟩.

**Parameters:**

- **config** (`typing.List[int]`): Configuration as list of local indices

**Returns**: `<class 'torch.Tensor'>` - Amplitude (scalar tensor)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:798](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L798)*

##### `probability`

```python
def probability(self, config: List[int]) -> torch.Tensor
```

Compute probability P(config) = |amplitude|².

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:813](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L813)*

##### `sample`

```python
def sample(self, n_samples: int = 1000) -> torch.Tensor
```

Generate samples from the distribution.

Uses sequential sampling from conditional distributions.

**Returns**: `<class 'torch.Tensor'>` - Tensor of shape (n_samples, n_sites)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:818](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L818)*

##### `train`

```python
def train(self, data: torch.Tensor, n_epochs: int = 100, learning_rate: float = 0.01, verbose: bool = True) -> List[float]
```

Train the Born machine on data.

**Parameters:**

- **data** (`<class 'torch.Tensor'>`): Training data of shape (n_samples, n_sites)
- **n_epochs** (`<class 'int'>`): Training epochs
- **learning_rate** (`<class 'float'>`): Optimizer learning rate
- **verbose** (`<class 'bool'>`): Print progress

**Returns**: `typing.List[float]` - Training loss history

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:870](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L870)*

### class `VQE`

Variational Quantum Eigensolver.

Uses tensor network simulation for efficient classical emulation.

#### Methods

##### `__init__`

```python
def __init__(self, hamiltonian: Callable[[hybrid.TNQuantumSimulator], float], n_qubits: int, config: Optional[hybrid.VQEConfig] = None)
```

Args:

hamiltonian: Function that computes ⟨H⟩ given simulator
    n_qubits: Number of qubits
    config: VQE configuration

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:443](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L443)*

##### `energy`

```python
def energy(self, params: torch.Tensor) -> float
```

Compute energy for given parameters.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:521](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L521)*

##### `optimize`

```python
def optimize(self, verbose: bool = True) -> Dict
```

Run VQE optimization.

**Returns**: `typing.Dict` - Dictionary with optimization results

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:528](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L528)*

### class `VQEConfig`

Configuration for VQE.

#### Attributes

- **ansatz_type** (`<enum 'AnsatzType'>`): 
- **n_layers** (`<class 'int'>`): 
- **optimizer** (`<class 'str'>`): 
- **learning_rate** (`<class 'float'>`): 
- **max_iterations** (`<class 'int'>`): 
- **tolerance** (`<class 'float'>`): 
- **chi_max** (`<class 'int'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, ansatz_type: hybrid.AnsatzType = <AnsatzType.HARDWARE_EFFICIENT: 'hardware_efficient'>, n_layers: int = 2, optimizer: str = 'adam', learning_rate: float = 0.01, max_iterations: int = 100, tolerance: float = 1e-06, chi_max: int = 64) -> None
```

## Functions

### `create_ising_hamiltonian`

```python
def create_ising_hamiltonian(n_qubits: int, J: float = 1.0, h: float = 0.5) -> Callable
```

Create Ising Hamiltonian for VQE.

H = -J Σ Z_i Z_{i+1} - h Σ X_i

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:1029](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L1029)*

### `create_maxcut_hamiltonian`

```python
def create_maxcut_hamiltonian(edges: List[Tuple[int, int]], n_qubits: int) -> List[Tuple[str, float]]
```

Create MaxCut cost Hamiltonian for QAOA.

H = Σ_{(i,j) ∈ E} (1 - Z_i Z_j) / 2

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py:1053](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\quantum\hybrid.py#L1053)*
