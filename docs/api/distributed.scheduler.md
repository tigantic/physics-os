# Module `distributed.scheduler`

Distributed task scheduling for CFD simulations.

This module provides task scheduling and dependency management
for parallel CFD computations.

Author: Tigantic Holdings LLC

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `DistributedScheduler`

Distributed task scheduler.

Executes tasks from a task graph across multiple
workers with dependency management.

#### Methods

##### `__init__`

```python
def __init__(self, config: scheduler.TaskConfig)
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py:273](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py#L273)*

##### `cancel`

```python
def cancel(self, graph_id: int)
```

Cancel all pending tasks in a graph.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py:417](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py#L417)*

##### `get_status`

```python
def get_status(self, graph_id: int) -> Dict[int, scheduler.TaskStatus]
```

Get status of all tasks in a graph.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py:408](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py#L408)*

##### `shutdown`

```python
def shutdown(self, wait: bool = True)
```

Shutdown the scheduler.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py:428](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py#L428)*

##### `submit`

```python
def submit(self, graph: scheduler.TaskGraph) -> int
```

Submit a task graph for execution.

**Parameters:**

- **graph** (`<class 'scheduler.TaskGraph'>`): Task graph to execute

**Returns**: `<class 'int'>` - Graph ID for tracking

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py:288](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py#L288)*

##### `wait_graph`

```python
def wait_graph(self, graph_id: int, timeout: Optional[float] = None) -> Dict[int, Any]
```

Wait for all tasks in a graph to complete.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py:380](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py#L380)*

##### `wait_task`

```python
def wait_task(self, task_id: int, timeout: Optional[float] = None) -> Any
```

Wait for a specific task to complete.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py:370](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py#L370)*

### class `Task`

Represents a computational task.

#### Attributes

- **task_id** (`<class 'int'>`): 
- **name** (`<class 'str'>`): 
- **func** (`typing.Callable`): 
- **args** (`typing.Tuple`): 
- **kwargs** (`typing.Dict[str, typing.Any]`): 
- **dependencies** (`typing.Set[int]`): 
- **priority** (`<enum 'TaskPriority'>`): 
- **status** (`<enum 'TaskStatus'>`): 
- **result** (`typing.Any`): 
- **error** (`typing.Optional[Exception]`): 
- **start_time** (`typing.Optional[float]`): 
- **end_time** (`typing.Optional[float]`): 

#### Properties

##### `duration`

```python
def duration(self) -> Optional[float]
```

Get task duration in seconds.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py:81](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py#L81)*

#### Methods

##### `__init__`

```python
def __init__(self, task_id: int, name: str, func: Callable, args: Tuple = <factory>, kwargs: Dict[str, Any] = <factory>, dependencies: Set[int] = <factory>, priority: scheduler.TaskPriority = <TaskPriority.NORMAL: 1>, status: scheduler.TaskStatus = <TaskStatus.PENDING: 1>, result: Any = None, error: Optional[Exception] = None, start_time: Optional[float] = None, end_time: Optional[float] = None) -> None
```

### class `TaskConfig`

Configuration for task execution.

#### Attributes

- **max_workers** (`<class 'int'>`): 
- **timeout** (`<class 'float'>`): 
- **retry_count** (`<class 'int'>`): 
- **checkpoint_enabled** (`<class 'bool'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, max_workers: int = 4, timeout: float = 300.0, retry_count: int = 3, checkpoint_enabled: bool = True) -> None
```

### class `TaskGraph`

Directed acyclic graph of tasks.

Manages task dependencies and provides topological
ordering for execution.

#### Methods

##### `__init__`

```python
def __init__(self)
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py:103](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py#L103)*

##### `add_task`

```python
def add_task(self, name: str, func: Callable, args: Tuple = (), kwargs: Dict[str, Any] = None, deps: List[int] = None, priority: scheduler.TaskPriority = <TaskPriority.NORMAL: 1>) -> int
```

Add a task to the graph.

**Parameters:**

- **name** (`<class 'str'>`): Task name
- **func** (`typing.Callable`): Function to execute
- **args** (`typing.Tuple`): Function arguments
- **kwargs** (`typing.Dict[str, typing.Any]`): Function keyword arguments
- **deps** (`typing.List[int]`): Task IDs this depends on
- **priority** (`<enum 'TaskPriority'>`): Task priority

**Returns**: `<class 'int'>` - Task ID

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py:108](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py#L108)*

##### `get_critical_path`

```python
def get_critical_path(self) -> List[int]
```

Find the critical path (longest execution chain).

**Returns**: `typing.List[int]` - List of task IDs on the critical path

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py:215](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py#L215)*

##### `get_ready_tasks`

```python
def get_ready_tasks(self) -> List[scheduler.Task]
```

Get all tasks ready for execution.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py:148](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py#L148)*

##### `get_task`

```python
def get_task(self, task_id: int) -> Optional[scheduler.Task]
```

Get task by ID.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py:144](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py#L144)*

##### `has_cycle`

```python
def has_cycle(self) -> bool
```

Check if graph has a cycle.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py:207](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py#L207)*

##### `reset`

```python
def reset(self)
```

Reset all tasks to pending state.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py:250](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py#L250)*

##### `topological_order`

```python
def topological_order(self) -> List[int]
```

Get tasks in topological order.

**Returns**: `typing.List[int]` - List of task IDs in execution order

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py:172](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py#L172)*

### class `TaskPriority`(Enum)

Task priority levels.

### class `TaskStatus`(Enum)

Task execution status.

## Functions

### `execute_parallel`

```python
def execute_parallel(funcs: List[Callable], config: Optional[scheduler.TaskConfig] = None) -> List[Any]
```

Execute independent functions in parallel.

**Parameters:**

- **funcs** (`typing.List[typing.Callable]`): List of functions to execute
- **config** (`typing.Optional[scheduler.TaskConfig]`): Scheduler configuration

**Returns**: `typing.List[typing.Any]` - List of results in same order as funcs

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py:480](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py#L480)*

### `schedule_dependency_graph`

```python
def schedule_dependency_graph(tasks: Dict[str, Callable], dependencies: Dict[str, List[str]], config: Optional[scheduler.TaskConfig] = None) -> Dict[str, Any]
```

Convenience function to schedule tasks with dependencies.

**Parameters:**

- **tasks** (`typing.Dict[str, typing.Callable]`): Dictionary mapping task names to functions
- **dependencies** (`typing.Dict[str, typing.List[str]]`): Dictionary mapping task names to their dependencies
- **config** (`typing.Optional[scheduler.TaskConfig]`): Scheduler configuration

**Returns**: `typing.Dict[str, typing.Any]` - Dictionary of task results

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py:433](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py#L433)*

### `test_scheduler`

```python
def test_scheduler()
```

Test distributed scheduler.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py:510](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\distributed\scheduler.py#L510)*
