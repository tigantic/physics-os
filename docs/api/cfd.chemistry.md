# Module `cfd.chemistry`

Multi-Species Chemistry for High-Temperature Air =================================================

Implements finite-rate chemistry for 5-species air model
used in hypersonic reentry simulations.

Species:
    N₂, O₂, N, O, NO

Reactions (Park Two-Temperature Model):
    1. O₂ + M ⇌ 2O + M       (O₂ dissociation)
    2. N₂ + M ⇌ 2N + M       (N₂ dissociation)
    3. NO + M ⇌ N + O + M    (NO dissociation)
    4. N₂ + O ⇌ NO + N       (Zeldovich NO)
    5. NO + O ⇌ O₂ + N       (Zeldovich NO)

Key Features:
    - Arrhenius rate coefficients
    - Third-body efficiencies
    - Equilibrium constants from curve fits
    - Species mass production rates

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `ArrheniusCoeffs`

Arrhenius rate coefficient parameters: k = A * T^n * exp(-E_a / (R*T))

#### Attributes

- **A** (`<class 'float'>`): 
- **n** (`<class 'float'>`): 
- **E_a** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, A: float, n: float, E_a: float) -> None
```

##### `compute`

```python
def compute(self, T: torch.Tensor) -> torch.Tensor
```

Compute forward rate coefficient at temperature T.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\chemistry.py:88](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\chemistry.py#L88)*

### class `ChemistryState`

State for multi-species chemistry solver.

#### Attributes

- **rho** (`<class 'torch.Tensor'>`): 
- **Y** (`typing.Dict[chemistry.Species, torch.Tensor]`): 
- **T** (`<class 'torch.Tensor'>`): 
- **p** (`<class 'torch.Tensor'>`): 

#### Properties

##### `shape`

```python
def shape(self) -> torch.Size
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\chemistry.py:304](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\chemistry.py#L304)*

#### Methods

##### `__init__`

```python
def __init__(self, rho: torch.Tensor, Y: Dict[chemistry.Species, torch.Tensor], T: torch.Tensor, p: torch.Tensor) -> None
```

##### `concentrations`

```python
def concentrations(self) -> Dict[chemistry.Species, torch.Tensor]
```

Compute molar concentrations [mol/m³].

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\chemistry.py:308](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\chemistry.py#L308)*

##### `mixture_R`

```python
def mixture_R(self) -> torch.Tensor
```

Compute mixture gas constant [J/(kg·K)].

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\chemistry.py:322](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\chemistry.py#L322)*

##### `mixture_molecular_weight`

```python
def mixture_molecular_weight(self) -> torch.Tensor
```

Compute mixture molecular weight [kg/mol].

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\chemistry.py:315](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\chemistry.py#L315)*

##### `validate`

```python
def validate(self) -> bool
```

Check physical constraints.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\chemistry.py:327](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\chemistry.py#L327)*

### class `Reaction`

Chemical reaction with forward/backward rates.

#### Attributes

- **name** (`<class 'str'>`): 
- **forward** (`<class 'chemistry.ArrheniusCoeffs'>`): 
- **reactants** (`typing.Dict[chemistry.Species, int]`): 
- **products** (`typing.Dict[chemistry.Species, int]`): 
- **third_body** (`<class 'bool'>`): 
- **third_body_efficiencies** (`typing.Optional[typing.Dict[chemistry.Species, float]]`): 

#### Methods

##### `__init__`

```python
def __init__(self, name: str, forward: chemistry.ArrheniusCoeffs, reactants: Dict[chemistry.Species, int], products: Dict[chemistry.Species, int], third_body: bool = False, third_body_efficiencies: Optional[Dict[chemistry.Species, float]] = None) -> None
```

### class `Species`(IntEnum)

Species indices for 5-species air.

## Functions

### `advance_chemistry_explicit`

```python
def advance_chemistry_explicit(state: chemistry.ChemistryState, dt: float) -> chemistry.ChemistryState
```

Advance chemistry with explicit Euler (for testing only).

WARNING: Chemistry is typically stiff and requires implicit methods.
This is provided for simple cases only.

**Parameters:**

- **state** (`<class 'chemistry.ChemistryState'>`): Current state
- **dt** (`<class 'float'>`): Timestep [s]

**Returns**: `<class 'chemistry.ChemistryState'>` - Updated state

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\chemistry.py:420](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\chemistry.py#L420)*

### `air_5species_ic`

```python
def air_5species_ic(shape: torch.Size, T: float = 300.0, p: float = 101325.0, Y_N2: float = 0.767, Y_O2: float = 0.233) -> chemistry.ChemistryState
```

Create initial condition for 5-species air.

Default is standard atmospheric composition:
    N₂: 76.7% by mass
    O₂: 23.3% by mass
    N, O, NO: 0%

**Parameters:**

- **shape** (`<class 'torch.Size'>`): Tensor shape (Ny, Nx)
- **T** (`<class 'float'>`): Temperature [K]
- **p** (`<class 'float'>`): Pressure [Pa]
- **Y_N2** (`<class 'float'>`): Mass fraction of N₂
- **Y_O2** (`<class 'float'>`): Mass fraction of O₂

**Returns**: `<class 'chemistry.ChemistryState'>` - ChemistryState initial condition

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\chemistry.py:344](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\chemistry.py#L344)*

### `chemistry_timestep`

```python
def chemistry_timestep(state: chemistry.ChemistryState, safety_factor: float = 0.1) -> torch.Tensor
```

Compute chemistry timestep based on stiffness.

dt_chem = safety * min(ρYᵢ / |ω̇ᵢ|)

**Parameters:**

- **state** (`<class 'chemistry.ChemistryState'>`): Current chemistry state
- **safety_factor** (`<class 'float'>`): CFL-like factor for stability

**Returns**: `<class 'torch.Tensor'>` - Chemistry timestep [s]

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\chemistry.py:390](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\chemistry.py#L390)*

### `compute_reaction_rates`

```python
def compute_reaction_rates(T: torch.Tensor, concentrations: Dict[chemistry.Species, torch.Tensor]) -> Tuple[Dict[chemistry.Species, torch.Tensor], torch.Tensor]
```

Compute species production rates from finite-rate chemistry.

ω̇ᵢ = Mᵢ Σⱼ νᵢⱼ (kf,j ∏ₖ [Xₖ]^νₖⱼ' - kb,j ∏ₖ [Xₖ]^νₖⱼ'')

**Parameters:**

- **T** (`<class 'torch.Tensor'>`): Temperature field [K]
- **concentrations** (`typing.Dict[chemistry.Species, torch.Tensor]`): Species molar concentrations [mol/m³]

**Returns**: `typing.Tuple[typing.Dict[chemistry.Species, torch.Tensor], torch.Tensor]` - - Dict of mass production rates [kg/(m³·s)]
        - Total heat release rate [W/m³]

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\chemistry.py:229](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\chemistry.py#L229)*

### `equilibrium_constant`

```python
def equilibrium_constant(reaction: chemistry.Reaction, T: torch.Tensor) -> torch.Tensor
```

Compute equilibrium constant K_eq from Gibbs free energy.

K_eq = exp(-ΔG°/RT) = exp(-ΔH°/RT + ΔS°/R)

Simplified curve fit approach for 5-species air.

**Parameters:**

- **reaction** (`<class 'chemistry.Reaction'>`): Reaction object
- **T** (`<class 'torch.Tensor'>`): Temperature [K]

**Returns**: `<class 'torch.Tensor'>` - Equilibrium constant K_eq

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\chemistry.py:161](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\chemistry.py#L161)*

### `post_shock_composition`

```python
def post_shock_composition(M: float, T1: float = 300.0, p1: float = 101325.0, gamma: float = 1.4) -> Dict[chemistry.Species, float]
```

Estimate post-shock composition for strong shocks.

Uses equilibrium chemistry to estimate dissociation
behind a normal shock.

**Parameters:**

- **M** (`<class 'float'>`): Freestream Mach number
- **T1** (`<class 'float'>`): Pre-shock temperature [K]
- **p1** (`<class 'float'>`): Pre-shock pressure [Pa]
- **gamma** (`<class 'float'>`): Ratio of specific heats

**Returns**: `typing.Dict[chemistry.Species, float]` - Dictionary of mass fractions

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\chemistry.py:474](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\chemistry.py#L474)*

### `third_body_concentration`

```python
def third_body_concentration(reaction: chemistry.Reaction, concentrations: Dict[chemistry.Species, torch.Tensor]) -> torch.Tensor
```

Compute effective third-body concentration [M].

[M] = Σᵢ αᵢ [Xᵢ]

where αᵢ is the third-body efficiency of species i.

**Parameters:**

- **reaction** (`<class 'chemistry.Reaction'>`): Reaction with third_body=True
- **concentrations** (`typing.Dict[chemistry.Species, torch.Tensor]`): Species molar concentrations [mol/m³]

**Returns**: `<class 'torch.Tensor'>` - Effective third-body concentration [mol/m³]

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\chemistry.py:198](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\chemistry.py#L198)*

### `validate_chemistry`

```python
def validate_chemistry()
```

Run validation tests for chemistry module.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\chemistry.py:541](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\cfd\chemistry.py#L541)*
