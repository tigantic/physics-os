# Module `distributed.communication`

MPI-style communication patterns for distributed CFD.

This module provides communication primitives for data exchange
between processors in distributed simulations.

Author: HyperTensor Team

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `AllReduceOp`(Enum)

All-reduce operation types.

### class `CommPattern`(Enum)

Communication pattern types.

### class `Communicator`

MPI-like communicator for distributed operations.

Provides collective operations and point-to-point
communication in a multi-GPU or simulated distributed
environment.

#### Methods

##### `__init__`

```python
def __init__(self, n_procs: int = 1, rank: int = 0)
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py:88](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py#L88)*

##### `barrier`

```python
def barrier(self)
```

Synchronize all processes.

Blocks until all processes reach this point.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py:244](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py#L244)*

##### `finalize`

```python
def finalize(self)
```

Finalize communicator.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py:114](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py#L114)*

##### `initialize`

```python
def initialize(self)
```

Initialize communicator.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py:109](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py#L109)*

##### `irecv`

```python
def irecv(self, source: int, tag: int = 0) -> int
```

Non-blocking receive.

**Parameters:**

- **source** (`<class 'int'>`): Source rank
- **tag** (`<class 'int'>`): Message tag

**Returns**: `<class 'int'>` - Request ID

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py:189](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py#L189)*

##### `isend`

```python
def isend(self, data: torch.Tensor, dest: int, tag: int = 0) -> int
```

Non-blocking send.

**Parameters:**

- **data** (`<class 'torch.Tensor'>`): Data to send
- **dest** (`<class 'int'>`): Destination rank
- **tag** (`<class 'int'>`): Message tag

**Returns**: `<class 'int'>` - Request ID

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py:164](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py#L164)*

##### `recv`

```python
def recv(self, source: int, tag: int = 0, timeout: Optional[float] = None) -> torch.Tensor
```

Blocking receive.

**Parameters:**

- **source** (`<class 'int'>`): Source rank
- **tag** (`<class 'int'>`): Message tag
- **timeout** (`typing.Optional[float]`): Timeout in seconds

**Returns**: `<class 'torch.Tensor'>` - Received data

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py:137](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py#L137)*

##### `send`

```python
def send(self, data: torch.Tensor, dest: int, tag: int = 0)
```

Blocking send.

**Parameters:**

- **data** (`<class 'torch.Tensor'>`): Data to send
- **dest** (`<class 'int'>`): Destination rank
- **tag** (`<class 'int'>`): Message tag

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py:119](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py#L119)*

##### `test`

```python
def test(self, request_id: int) -> bool
```

Test if request is complete.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py:239](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py#L239)*

##### `wait`

```python
def wait(self, request_id: int) -> Optional[torch.Tensor]
```

Wait for non-blocking operation to complete.

**Parameters:**

- **request_id** (`<class 'int'>`): Request ID from isend/irecv

**Returns**: `typing.Optional[torch.Tensor]` - Received data for irecv, None for isend

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py:213](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py#L213)*

### class `DistributedTensor`

Tensor distributed across multiple ranks.

Provides operations that automatically handle
communication for distributed computations.

#### Properties

##### `global_shape`

```python
def global_shape(self) -> Tuple[int, ...]
```

Get global tensor shape.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py:455](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py#L455)*

##### `local_shape`

```python
def local_shape(self) -> Tuple[int, ...]
```

Get local tensor shape.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py:460](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py#L460)*

#### Methods

##### `__init__`

```python
def __init__(self, local_data: torch.Tensor, comm: communication.Communicator, distribution: str = 'block')
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py:428](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py#L428)*

##### `max`

```python
def max(self) -> torch.Tensor
```

Global max reduction.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py:481](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py#L481)*

##### `mean`

```python
def mean(self) -> torch.Tensor
```

Global mean reduction.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py:470](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py#L470)*

##### `min`

```python
def min(self) -> torch.Tensor
```

Global min reduction.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py:486](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py#L486)*

##### `norm`

```python
def norm(self, p: int = 2) -> torch.Tensor
```

Global norm computation.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py:491](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py#L491)*

##### `sum`

```python
def sum(self) -> torch.Tensor
```

Global sum reduction.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py:465](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py#L465)*

##### `to_global`

```python
def to_global(self) -> Optional[torch.Tensor]
```

Gather to global tensor on rank 0.

**Returns**: `typing.Optional[torch.Tensor]` - Full tensor on rank 0, None on other ranks

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py:497](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py#L497)*

### class `Message`

Message for inter-process communication.

#### Attributes

- **source** (`<class 'int'>`): 
- **destination** (`<class 'int'>`): 
- **tag** (`<class 'int'>`): 
- **data** (`<class 'torch.Tensor'>`): 
- **timestamp** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, source: int, destination: int, tag: int, data: torch.Tensor, timestamp: float = <factory>) -> None
```

### class `MessageQueue`

Thread-safe message queue.

#### Methods

##### `__init__`

```python
def __init__(self)
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py:53](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py#L53)*

##### `empty`

```python
def empty(self) -> bool
```

Check if queue is empty.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py:69](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py#L69)*

##### `get`

```python
def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[communication.Message]
```

Get message from queue.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py:61](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py#L61)*

##### `put`

```python
def put(self, message: communication.Message)
```

Add message to queue.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py:57](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py#L57)*

## Functions

### `all_reduce`

```python
def all_reduce(comm: communication.Communicator, data: torch.Tensor, op: communication.AllReduceOp = <AllReduceOp.SUM: 1>) -> torch.Tensor
```

All-reduce operation across all processes.

**Parameters:**

- **comm** (`<class 'communication.Communicator'>`): Communicator
- **data** (`<class 'torch.Tensor'>`): Local data
- **op** (`<enum 'AllReduceOp'>`): Reduction operation

**Returns**: `<class 'torch.Tensor'>` - Reduced result (same on all processes)

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py:303](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py#L303)*

### `async_recv`

```python
def async_recv(comm: communication.Communicator, source: int, tag: int = 0) -> int
```

Convenience function for non-blocking receive.

**Parameters:**

- **comm** (`<class 'communication.Communicator'>`): Communicator
- **source** (`<class 'int'>`): Source rank
- **tag** (`<class 'int'>`): Message tag

**Returns**: `<class 'int'>` - Request ID

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py:278](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py#L278)*

### `async_send`

```python
def async_send(comm: communication.Communicator, data: torch.Tensor, dest: int, tag: int = 0) -> int
```

Convenience function for non-blocking send.

**Parameters:**

- **comm** (`<class 'communication.Communicator'>`): Communicator
- **data** (`<class 'torch.Tensor'>`): Data to send
- **dest** (`<class 'int'>`): Destination rank
- **tag** (`<class 'int'>`): Message tag

**Returns**: `<class 'int'>` - Request ID

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py:261](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py#L261)*

### `barrier`

```python
def barrier(comm: communication.Communicator)
```

Synchronization barrier.

**Parameters:**

- **comm** (`<class 'communication.Communicator'>`): Communicator

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py:293](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py#L293)*

### `broadcast`

```python
def broadcast(comm: communication.Communicator, data: torch.Tensor, root: int = 0) -> torch.Tensor
```

Broadcast data from root to all processes.

**Parameters:**

- **comm** (`<class 'communication.Communicator'>`): Communicator
- **data** (`<class 'torch.Tensor'>`): Data to broadcast (only used on root)
- **root** (`<class 'int'>`): Root rank

**Returns**: `<class 'torch.Tensor'>` - Broadcast data on all ranks

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py:341](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py#L341)*

### `gather`

```python
def gather(comm: communication.Communicator, data: torch.Tensor, root: int = 0) -> Optional[List[torch.Tensor]]
```

Gather data from all processes to root.

**Parameters:**

- **comm** (`<class 'communication.Communicator'>`): Communicator
- **data** (`<class 'torch.Tensor'>`): Local data to send
- **root** (`<class 'int'>`): Root rank

**Returns**: `typing.Optional[typing.List[torch.Tensor]]` - List of all data on root, None on other ranks

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py:392](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py#L392)*

### `scatter`

```python
def scatter(comm: communication.Communicator, data: Optional[List[torch.Tensor]], root: int = 0) -> torch.Tensor
```

Scatter data from root to all processes.

**Parameters:**

- **comm** (`<class 'communication.Communicator'>`): Communicator
- **data** (`typing.Optional[typing.List[torch.Tensor]]`): List of tensors to scatter (only used on root)
- **root** (`<class 'int'>`): Root rank

**Returns**: `<class 'torch.Tensor'>` - Local portion of scattered data

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py:365](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py#L365)*

### `test_communication`

```python
def test_communication()
```

Test communication patterns.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py:511](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\distributed\communication.py#L511)*
