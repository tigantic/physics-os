# Module `integration.workflows`

Workflow Orchestration for Project The Physics OS.

Provides end-to-end workflow definitions and execution for:
- CFD simulations
- Guidance computations
- Digital twin synchronization
- Validation campaigns

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `CFDSimulationWorkflow`(Workflow)

Standard CFD simulation workflow.

Stages:
    1. Initialization - Load mesh, set ICs/BCs
    2. Preprocessing - Compute metrics, initialize solver
    3. Solving - Time stepping loop
    4. Postprocessing - Extract results, visualize

#### Methods

##### `__init__`

```python
def __init__(self, config: Optional[Dict[str, Any]] = None)
```

Initialize CFD workflow.

**Parameters:**

- **config** (`typing.Optional[typing.Dict[str, typing.Any]]`): Simulation configuration

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py:370](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py#L370)*

### class `DigitalTwinWorkflow`(Workflow)

Digital twin synchronization workflow.

Stages:
    1. Data ingestion - Receive telemetry
    2. State sync - Update digital twin
    3. Health check - Monitor anomalies
    4. Prediction - Forecast future states

#### Methods

##### `__init__`

```python
def __init__(self, config: Optional[Dict[str, Any]] = None)
```

Initialize digital twin workflow.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py:696](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py#L696)*

### class `GuidanceWorkflow`(Workflow)

Trajectory guidance workflow.

Stages:
    1. State estimation - Current vehicle state
    2. CFD query - Aerodynamic coefficients
    3. Trajectory optimization - Optimal path
    4. Control command - Actuator commands

#### Methods

##### `__init__`

```python
def __init__(self, config: Optional[Dict[str, Any]] = None)
```

Initialize guidance workflow.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py:574](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py#L574)*

### class `ValidationWorkflow`(Workflow)

Validation campaign workflow.

Stages:
    1. Setup - Configure validation suite
    2. Execution - Run validation tests
    3. Analysis - Analyze results
    4. Reporting - Generate reports

#### Methods

##### `__init__`

```python
def __init__(self, config: Optional[Dict[str, Any]] = None)
```

Initialize validation workflow.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py:814](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py#L814)*

### class `Workflow`

Complete workflow definition.

#### Attributes

- **name** (`<class 'str'>`): 
- **description** (`<class 'str'>`): 
- **stages** (`typing.List[workflows.WorkflowStage]`): 
- **initial_context** (`typing.Dict[str, typing.Any]`): 
- **version** (`<class 'str'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, name: str, description: str = '', stages: List[workflows.WorkflowStage] = <factory>, initial_context: Dict[str, Any] = <factory>, version: str = '1.0') -> None
```

##### `add_stage`

```python
def add_stage(self, stage: workflows.WorkflowStage)
```

Add a stage to the workflow.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py:158](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py#L158)*

##### `add_step`

```python
def add_step(self, step: workflows.WorkflowStep, stage_name: Optional[str] = None)
```

Add a step to a stage.

If stage_name is None, creates a new stage for the step.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py:162](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py#L162)*

### class `WorkflowEngine`

Engine for executing workflows.

Handles step execution, context management, retries, and error handling.

#### Methods

##### `__init__`

```python
def __init__(self, max_parallel: int = 4, default_timeout: float = 300.0, verbose: bool = True)
```

Initialize workflow engine.

**Parameters:**

- **max_parallel** (`<class 'int'>`): Maximum parallel steps
- **default_timeout** (`<class 'float'>`): Default step timeout Default: `step timeout`.
- **verbose** (`<class 'bool'>`): Whether to print progress

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py:190](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py#L190)*

##### `add_hook`

```python
def add_hook(self, event: str, callback: Callable)
```

Add a hook callback for an event.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py:213](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py#L213)*

##### `run`

```python
def run(self, workflow: workflows.Workflow, context: Optional[Dict[str, Any]] = None) -> workflows.WorkflowResult
```

Execute a workflow.

**Parameters:**

- **workflow** (`<class 'workflows.Workflow'>`): Workflow to execute
- **context** (`typing.Optional[typing.Dict[str, typing.Any]]`): Initial context overrides

**Returns**: `<class 'workflows.WorkflowResult'>` - WorkflowResult with outcomes

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py:269](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py#L269)*

### class `WorkflowResult`

Result of workflow execution.

#### Attributes

- **workflow_name** (`<class 'str'>`): 
- **status** (`<enum 'WorkflowStatus'>`): 
- **context** (`typing.Dict[str, typing.Any]`): 
- **step_results** (`typing.Dict[str, typing.Dict]`): 
- **duration** (`<class 'float'>`): 
- **error** (`typing.Optional[str]`): 

#### Properties

##### `success`

```python
def success(self) -> bool
```

Whether workflow completed successfully.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py:112](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py#L112)*

#### Methods

##### `__init__`

```python
def __init__(self, workflow_name: str, status: workflows.WorkflowStatus, context: Dict[str, Any], step_results: Dict[str, Dict] = <factory>, duration: float = 0.0, error: Optional[str] = None) -> None
```

##### `get_output`

```python
def get_output(self, key: str, default: Any = None) -> Any
```

Get an output from the context.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py:117](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py#L117)*

##### `summary`

```python
def summary(self) -> str
```

Generate a summary of the workflow result.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py:121](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py#L121)*

### class `WorkflowStage`

Collection of steps that can run in parallel.

#### Attributes

- **name** (`<class 'str'>`): 
- **steps** (`typing.List[workflows.WorkflowStep]`): 
- **description** (`<class 'str'>`): 
- **parallel** (`<class 'bool'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, name: str, steps: List[workflows.WorkflowStep] = <factory>, description: str = '', parallel: bool = False) -> None
```

##### `add_step`

```python
def add_step(self, step: workflows.WorkflowStep)
```

Add a step to this stage.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py:87](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py#L87)*

### class `WorkflowStatus`(Enum)

Status of a workflow or step.

### class `WorkflowStep`

Single step in a workflow.

#### Attributes

- **name** (`<class 'str'>`): 
- **executor** (`typing.Callable[[typing.Dict[str, typing.Any]], typing.Dict[str, typing.Any]]`): 
- **description** (`<class 'str'>`): 
- **required_inputs** (`typing.List[str]`): 
- **outputs** (`typing.List[str]`): 
- **timeout** (`typing.Optional[float]`): 
- **retries** (`<class 'int'>`): 
- **skip_on_failure** (`<class 'bool'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, name: str, executor: Callable[[Dict[str, Any]], Dict[str, Any]], description: str = '', required_inputs: List[str] = <factory>, outputs: List[str] = <factory>, timeout: Optional[float] = None, retries: int = 0, skip_on_failure: bool = False) -> None
```

##### `execute`

```python
def execute(self, context: Dict[str, Any]) -> Dict[str, Any]
```

Execute the step.

**Parameters:**

- **context** (`typing.Dict[str, typing.Any]`): Workflow context with inputs

**Returns**: `typing.Dict[str, typing.Any]` - Step outputs to merge into context

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py:58](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py#L58)*

##### `validate_inputs`

```python
def validate_inputs(self, context: Dict[str, Any]) -> bool
```

Check if all required inputs are available.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py:54](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py#L54)*

## Functions

### `create_cfd_workflow`

```python
def create_cfd_workflow(nx: int = 100, ny: int = 50, n_steps: int = 100, cfl: float = 0.5, **kwargs) -> workflows.CFDSimulationWorkflow
```

Create a CFD simulation workflow.

**Parameters:**

- **nx** (`<class 'int'>`): Grid points in x
- **ny** (`<class 'int'>`): Grid points in y
- **n_steps** (`<class 'int'>`): Number of time steps
- **cfl** (`<class 'float'>`): CFL number **kwargs: Additional configuration

**Returns**: `<class 'workflows.CFDSimulationWorkflow'>` - Configured CFDSimulationWorkflow

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py:857](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py#L857)*

### `create_guidance_workflow`

```python
def create_guidance_workflow(target: Optional[List[float]] = None, **kwargs) -> workflows.GuidanceWorkflow
```

Create a guidance workflow.

**Parameters:**

- **target** (`typing.Optional[typing.List[float]]`): Target position [x, y, z] **kwargs: Additional configuration

**Returns**: `<class 'workflows.GuidanceWorkflow'>` - Configured GuidanceWorkflow

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py:887](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py#L887)*

### `run_workflow`

```python
def run_workflow(workflow: workflows.Workflow, context: Optional[Dict[str, Any]] = None, verbose: bool = True) -> workflows.WorkflowResult
```

Execute a workflow.

**Parameters:**

- **workflow** (`<class 'workflows.Workflow'>`): Workflow to run
- **context** (`typing.Optional[typing.Dict[str, typing.Any]]`): Initial context
- **verbose** (`<class 'bool'>`): Print progress

**Returns**: `<class 'workflows.WorkflowResult'>` - WorkflowResult with outcomes

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py:837](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\workflows.py#L837)*
