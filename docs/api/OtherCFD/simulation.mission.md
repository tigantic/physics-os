# Module `simulation.mission`

Mission Simulation Module =========================

End-to-end mission simulation with Monte Carlo capability for
uncertainty quantification and dispersion analysis.

Capabilities:
    - Full mission profile simulation (boost → glide → terminal)
    - Monte Carlo analysis with configurable dispersions
    - Statistical CEP/LEP computation
    - Sensitivity analysis
    - Performance envelope mapping

Mission Phases:
    ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌──────────┐   ┌──────────┐
    │  BOOST  │──►│ PULLOUT │──►│  GLIDE  │──►│   TAEM   │──►│ TERMINAL │
    │         │   │         │   │         │   │          │   │          │
    └─────────┘   └─────────┘   └─────────┘   └──────────┘   └──────────┘
         │             │             │             │              │
         ▼             ▼             ▼             ▼              ▼
     Thrust        Guidance      Aero-Guidance  Energy Mgmt   Final Aim
     Profile       Init          Banking        Altitude Adj   Point Corr

Monte Carlo Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                     MissionSimulator                         │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
    │  │ Uncertainty │  │ Trajectory  │  │ Performance         │  │
    │  │ Model       │──│ Solver      │──│ Metrics             │  │
    │  └─────────────┘  └─────────────┘  └─────────────────────┘  │
    └─────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
             ┌──────────────┐       ┌──────────────┐
             │ Single Run   │  ...  │ Single Run   │
             │ (Sample 1)   │       │ (Sample N)   │
             └──────────────┘       └──────────────┘
                    │                       │
                    └───────────┬───────────┘
                                ▼
                    ┌───────────────────────┐
                    │  Dispersion Analysis  │
                    │  - CEP/LEP            │
                    │  - Sensitivity        │
                    │  - Failure modes      │
                    └───────────────────────┘

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `FailureMode`(Enum)

Mission failure modes.

### class `MissionConfig`

Configuration for mission simulation.

#### Attributes

- **launch_lat** (`<class 'float'>`): 
- **launch_lon** (`<class 'float'>`): 
- **launch_alt** (`<class 'float'>`): 
- **launch_heading** (`<class 'float'>`): 
- **target_lat** (`<class 'float'>`): 
- **target_lon** (`<class 'float'>`): 
- **target_alt** (`<class 'float'>`): 
- **vehicle_mass_kg** (`<class 'float'>`): 
- **S_ref** (`<class 'float'>`): 
- **boost_thrust_N** (`<class 'float'>`): 
- **boost_duration_s** (`<class 'float'>`): 
- **boost_pitch_profile** (`<class 'str'>`): 
- **target_L_D** (`<class 'float'>`): 
- **max_bank_angle_deg** (`<class 'float'>`): 
- **max_g_load** (`<class 'float'>`): 
- **max_q_bar_Pa** (`<class 'float'>`): 
- **max_heating_rate_W_cm2** (`<class 'float'>`): 
- **dt_s** (`<class 'float'>`): 
- **max_time_s** (`<class 'float'>`): 
- **pullout_complete_alt** (`<class 'float'>`): 
- **taem_start_range_km** (`<class 'float'>`): 
- **terminal_start_range_km** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, launch_lat: float = 34.0, launch_lon: float = -118.0, launch_alt: float = 0.0, launch_heading: float = 90.0, target_lat: float = 35.0, target_lon: float = -100.0, target_alt: float = 0.0, vehicle_mass_kg: float = 1000.0, S_ref: float = 10.0, boost_thrust_N: float = 100000.0, boost_duration_s: float = 60.0, boost_pitch_profile: str = 'optimal', target_L_D: float = 3.0, max_bank_angle_deg: float = 70.0, max_g_load: float = 8.0, max_q_bar_Pa: float = 100000.0, max_heating_rate_W_cm2: float = 200.0, dt_s: float = 0.1, max_time_s: float = 1800.0, pullout_complete_alt: float = 40000.0, taem_start_range_km: float = 100.0, terminal_start_range_km: float = 20.0) -> None
```

### class `MissionPhase`(Enum)

Mission phase enumeration.

### class `MissionResult`

Result of a single mission simulation.

#### Attributes

- **success** (`<class 'bool'>`): 
- **phase_history** (`typing.List[mission.MissionPhase]`): 
- **failure_mode** (`<enum 'FailureMode'>`): 
- **impact_lat** (`<class 'float'>`): 
- **impact_lon** (`<class 'float'>`): 
- **impact_time** (`<class 'float'>`): 
- **miss_distance_m** (`<class 'float'>`): 
- **downrange_error_m** (`<class 'float'>`): 
- **crossrange_error_m** (`<class 'float'>`): 
- **max_mach** (`<class 'float'>`): 
- **max_altitude_m** (`<class 'float'>`): 
- **max_g_load** (`<class 'float'>`): 
- **max_heating_rate** (`<class 'float'>`): 
- **max_q_bar** (`<class 'float'>`): 
- **total_range_km** (`<class 'float'>`): 
- **trajectory** (`typing.Optional[numpy.ndarray]`): 
- **times** (`typing.Optional[numpy.ndarray]`): 
- **uncertainty_sample** (`typing.Optional[typing.Dict]`): 

#### Methods

##### `__init__`

```python
def __init__(self, success: bool = False, phase_history: List[mission.MissionPhase] = <factory>, failure_mode: mission.FailureMode = <FailureMode.NONE: 'none'>, impact_lat: float = 0.0, impact_lon: float = 0.0, impact_time: float = 0.0, miss_distance_m: float = 0.0, downrange_error_m: float = 0.0, crossrange_error_m: float = 0.0, max_mach: float = 0.0, max_altitude_m: float = 0.0, max_g_load: float = 0.0, max_heating_rate: float = 0.0, max_q_bar: float = 0.0, total_range_km: float = 0.0, trajectory: Optional[numpy.ndarray] = None, times: Optional[numpy.ndarray] = None, uncertainty_sample: Optional[Dict] = None) -> None
```

### class `MissionSimulator`

End-to-end mission simulator.

Integrates trajectory solver, guidance controller, and
aerodynamic models for complete mission simulation.

#### Methods

##### `__init__`

```python
def __init__(self, config: mission.MissionConfig = None, uncertainty: mission.UncertaintyModel = None)
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\mission.py:237](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\mission.py#L237)*

##### `run`

```python
def run(self, uncertainty_sample: Dict = None) -> mission.MissionResult
```

Run a single mission simulation.

**Parameters:**

- **uncertainty_sample** (`typing.Dict`): Uncertainty factors to apply

**Returns**: `<class 'mission.MissionResult'>` - MissionResult with outcome and metrics

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\mission.py:350](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\mission.py#L350)*

### class `MonteCarloConfig`

Configuration for Monte Carlo analysis.

#### Attributes

- **n_runs** (`<class 'int'>`): 
- **seed** (`typing.Optional[int]`): 
- **parallel** (`<class 'bool'>`): 
- **n_workers** (`<class 'int'>`): 
- **compute_cep** (`<class 'bool'>`): 
- **compute_lep** (`<class 'bool'>`): 
- **compute_sensitivity** (`<class 'bool'>`): 
- **save_all_trajectories** (`<class 'bool'>`): 
- **save_impact_points** (`<class 'bool'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, n_runs: int = 100, seed: Optional[int] = None, parallel: bool = True, n_workers: int = 4, compute_cep: bool = True, compute_lep: bool = True, compute_sensitivity: bool = True, save_all_trajectories: bool = False, save_impact_points: bool = True) -> None
```

### class `UncertaintyModel`

Uncertainty model for Monte Carlo analysis.

#### Attributes

- **density_sigma_pct** (`<class 'float'>`): 
- **wind_sigma_ms** (`<class 'float'>`): 
- **CL_sigma_pct** (`<class 'float'>`): 
- **CD_sigma_pct** (`<class 'float'>`): 
- **Cm_sigma_pct** (`<class 'float'>`): 
- **thrust_sigma_pct** (`<class 'float'>`): 
- **Isp_sigma_pct** (`<class 'float'>`): 
- **position_sigma_m** (`<class 'float'>`): 
- **velocity_sigma_ms** (`<class 'float'>`): 
- **mass_sigma_pct** (`<class 'float'>`): 
- **actuator_bias_deg** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, density_sigma_pct: float = 5.0, wind_sigma_ms: float = 20.0, CL_sigma_pct: float = 5.0, CD_sigma_pct: float = 10.0, Cm_sigma_pct: float = 10.0, thrust_sigma_pct: float = 3.0, Isp_sigma_pct: float = 2.0, position_sigma_m: float = 100.0, velocity_sigma_ms: float = 1.0, mass_sigma_pct: float = 1.0, actuator_bias_deg: float = 0.5) -> None
```

##### `sample`

```python
def sample(self) -> Dict[str, float]
```

Generate a sample of uncertainty factors.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\mission.py:158](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\mission.py#L158)*

## Functions

### `analyze_dispersion`

```python
def analyze_dispersion(results: List[mission.MissionResult]) -> Dict[str, Any]
```

Analyze dispersion from Monte Carlo results.

**Parameters:**

- **results** (`typing.List[mission.MissionResult]`): List of mission results

**Returns**: `typing.Dict[str, typing.Any]` - Dispersion analysis metrics

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\mission.py:575](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\mission.py#L575)*

### `compute_sensitivity`

```python
def compute_sensitivity(config: mission.MissionConfig, uncertainty: mission.UncertaintyModel, n_samples: int = 50) -> Dict[str, float]
```

Compute sensitivity indices for uncertainty factors.

Uses one-at-a-time (OAT) sensitivity analysis.

**Parameters:**

- **config** (`<class 'mission.MissionConfig'>`): Mission configuration
- **uncertainty** (`<class 'mission.UncertaintyModel'>`): Baseline uncertainty model
- **n_samples** (`<class 'int'>`): Samples per factor

**Returns**: `typing.Dict[str, float]` - Dict of factor -> sensitivity index

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\mission.py:650](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\mission.py#L650)*

### `run_monte_carlo`

```python
def run_monte_carlo(config: mission.MissionConfig, uncertainty: mission.UncertaintyModel, mc_config: mission.MonteCarloConfig) -> List[mission.MissionResult]
```

Run Monte Carlo simulation.

**Parameters:**

- **config** (`<class 'mission.MissionConfig'>`): Mission configuration
- **uncertainty** (`<class 'mission.UncertaintyModel'>`): Uncertainty model
- **mc_config** (`<class 'mission.MonteCarloConfig'>`): Monte Carlo settings

**Returns**: `typing.List[mission.MissionResult]` - List of MissionResult for all runs

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\mission.py:535](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\mission.py#L535)*

### `validate_mission_module`

```python
def validate_mission_module()
```

Validate mission simulation module.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\mission.py:701](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\mission.py#L701)*
