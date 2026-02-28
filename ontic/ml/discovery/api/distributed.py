"""
Distributed Execution Module for Discovery Engine

Provides multi-GPU and multi-node distributed execution capabilities.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum
import time
import uuid
import logging
import threading
import queue
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, Future

logger = logging.getLogger(__name__)


class NodeStatus(str, Enum):
    """Worker node status."""
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"


class TaskStatus(str, Enum):
    """Distributed task status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DistributedConfig:
    """
    Configuration for distributed execution.
    
    Attributes:
        coordinator_host: Coordinator node hostname
        coordinator_port: Coordinator node port
        worker_threads: Number of worker threads per node
        heartbeat_interval: Heartbeat interval in seconds
        task_timeout: Task timeout in seconds
        retry_count: Number of retries for failed tasks
        load_balancing: Load balancing strategy
    """
    coordinator_host: str = "localhost"
    coordinator_port: int = 50051
    worker_threads: int = 4
    heartbeat_interval: float = 5.0
    task_timeout: float = 300.0
    retry_count: int = 3
    load_balancing: str = "round_robin"  # round_robin, least_loaded, random
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "coordinator_host": self.coordinator_host,
            "coordinator_port": self.coordinator_port,
            "worker_threads": self.worker_threads,
            "heartbeat_interval": self.heartbeat_interval,
            "task_timeout": self.task_timeout,
            "retry_count": self.retry_count,
            "load_balancing": self.load_balancing,
        }


@dataclass
class TaskResult:
    """Result from a distributed task."""
    task_id: str
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    worker_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "worker_id": self.worker_id,
        }


# Global counter for task ordering
_task_counter = 0


@dataclass
class DistributedTask:
    """A task to be executed in distributed mode."""
    id: str
    function_name: str
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    timeout: float = 300.0
    retry_count: int = 3
    sequence: int = field(default=0)
    
    def __post_init__(self):
        """Generate ID if not provided."""
        global _task_counter
        if not self.id:
            self.id = str(uuid.uuid4())
        _task_counter += 1
        self.sequence = _task_counter
    
    def __lt__(self, other: 'DistributedTask') -> bool:
        """Compare tasks for priority queue ordering."""
        if self.priority != other.priority:
            return self.priority > other.priority  # Higher priority first
        return self.sequence < other.sequence  # Earlier tasks first
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "function_name": self.function_name,
            "args": list(self.args),
            "kwargs": self.kwargs,
            "priority": self.priority,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
        }


@dataclass
class WorkerInfo:
    """Information about a worker node."""
    id: str
    host: str
    port: int
    gpu_count: int = 0
    gpu_memory_mb: int = 0
    status: NodeStatus = NodeStatus.IDLE
    current_tasks: int = 0
    max_tasks: int = 4
    last_heartbeat: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "host": self.host,
            "port": self.port,
            "gpu_count": self.gpu_count,
            "gpu_memory_mb": self.gpu_memory_mb,
            "status": self.status.value,
            "current_tasks": self.current_tasks,
            "max_tasks": self.max_tasks,
            "last_heartbeat": self.last_heartbeat,
        }


class WorkerNode:
    """
    Worker node for distributed execution.
    
    Executes tasks assigned by the coordinator.
    """
    
    def __init__(
        self,
        config: Optional[DistributedConfig] = None,
        worker_id: Optional[str] = None
    ):
        """
        Initialize worker node.
        
        Args:
            config: Distributed configuration
            worker_id: Unique worker identifier
        """
        self.config = config or DistributedConfig()
        self.id = worker_id or str(uuid.uuid4())[:8]
        self._running = False
        self._task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._results: Dict[str, TaskResult] = {}
        self._executor: Optional[ThreadPoolExecutor] = None
        self._functions: Dict[str, Callable] = {}
        self._lock = threading.Lock()
        
        # GPU info
        self._gpu_count = 0
        self._gpu_memory_mb = 0
        self._detect_gpu()
    
    def _detect_gpu(self) -> None:
        """Detect available GPUs."""
        try:
            import torch
            if torch.cuda.is_available():
                self._gpu_count = torch.cuda.device_count()
                for i in range(self._gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    self._gpu_memory_mb += props.total_memory // (1024 * 1024)
        except ImportError:
            pass
    
    def register_function(self, name: str, func: Callable) -> None:
        """
        Register a function for remote execution.
        
        Args:
            name: Function name
            func: Callable function
        """
        self._functions[name] = func
        logger.info(f"Registered function: {name}")
    
    def start(self) -> None:
        """Start the worker node."""
        if self._running:
            return
        
        self._running = True
        self._executor = ThreadPoolExecutor(max_workers=self.config.worker_threads)
        
        # Start task processing thread
        threading.Thread(target=self._process_tasks, daemon=True).start()
        
        logger.info(f"Worker {self.id} started with {self.config.worker_threads} threads")
    
    def stop(self) -> None:
        """Stop the worker node."""
        self._running = False
        if self._executor:
            self._executor.shutdown(wait=True)
        logger.info(f"Worker {self.id} stopped")
    
    def submit_task(self, task: DistributedTask) -> str:
        """
        Submit a task for execution.
        
        Args:
            task: Task to execute
            
        Returns:
            Task ID
        """
        # Priority queue now uses task's __lt__ for ordering
        self._task_queue.put(task)
        return task.id
    
    def get_result(self, task_id: str) -> Optional[TaskResult]:
        """
        Get result for a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task result if available
        """
        return self._results.get(task_id)
    
    def _process_tasks(self) -> None:
        """Process tasks from the queue."""
        while self._running:
            try:
                # Get task with timeout
                task = self._task_queue.get(timeout=1.0)
                
                # Execute task
                future = self._executor.submit(self._execute_task, task)
                future.add_done_callback(
                    lambda f, t=task: self._task_completed(t, f)
                )
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing task: {e}")
    
    def _execute_task(self, task: DistributedTask) -> Any:
        """
        Execute a single task.
        
        Args:
            task: Task to execute
            
        Returns:
            Task result
        """
        func = self._functions.get(task.function_name)
        if not func:
            raise ValueError(f"Unknown function: {task.function_name}")
        
        return func(*task.args, **task.kwargs)
    
    def _task_completed(self, task: DistributedTask, future: Future) -> None:
        """
        Handle task completion.
        
        Args:
            task: Completed task
            future: Future with result
        """
        try:
            result = future.result()
            self._results[task.id] = TaskResult(
                task_id=task.id,
                status=TaskStatus.COMPLETED,
                result=result,
                worker_id=self.id,
            )
        except Exception as e:
            self._results[task.id] = TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                error=str(e),
                worker_id=self.id,
            )
    
    def get_info(self) -> WorkerInfo:
        """Get worker node information."""
        return WorkerInfo(
            id=self.id,
            host=self.config.coordinator_host,
            port=self.config.coordinator_port,
            gpu_count=self._gpu_count,
            gpu_memory_mb=self._gpu_memory_mb,
            status=NodeStatus.BUSY if self._task_queue.qsize() > 0 else NodeStatus.IDLE,
            current_tasks=self._task_queue.qsize(),
            max_tasks=self.config.worker_threads,
            last_heartbeat=time.time(),
        )


class DistributedCoordinator:
    """
    Coordinator for distributed execution.
    
    Manages worker nodes and distributes tasks.
    """
    
    def __init__(self, config: Optional[DistributedConfig] = None):
        """
        Initialize coordinator.
        
        Args:
            config: Distributed configuration
        """
        self.config = config or DistributedConfig()
        self._workers: Dict[str, WorkerInfo] = {}
        self._tasks: Dict[str, DistributedTask] = {}
        self._results: Dict[str, TaskResult] = {}
        self._pending_queue: queue.Queue = queue.Queue()
        self._running = False
        self._lock = threading.Lock()
        self._round_robin_index = 0
        
        # Local worker for single-node operation
        self._local_worker: Optional[WorkerNode] = None
    
    def start(self, local_mode: bool = True) -> None:
        """
        Start the coordinator.
        
        Args:
            local_mode: If True, start a local worker
        """
        self._running = True
        
        if local_mode:
            self._local_worker = WorkerNode(self.config, worker_id="local")
            self._local_worker.start()
            self._register_worker(self._local_worker.get_info())
        
        # Start scheduler thread
        threading.Thread(target=self._scheduler_loop, daemon=True).start()
        
        logger.info("Coordinator started")
    
    def stop(self) -> None:
        """Stop the coordinator."""
        self._running = False
        if self._local_worker:
            self._local_worker.stop()
        logger.info("Coordinator stopped")
    
    def register_function(self, name: str, func: Callable) -> None:
        """
        Register a function for distributed execution.
        
        Args:
            name: Function name
            func: Callable function
        """
        if self._local_worker:
            self._local_worker.register_function(name, func)
    
    def _register_worker(self, info: WorkerInfo) -> None:
        """
        Register a worker node.
        
        Args:
            info: Worker information
        """
        with self._lock:
            self._workers[info.id] = info
            logger.info(f"Registered worker: {info.id}")
    
    def _unregister_worker(self, worker_id: str) -> None:
        """
        Unregister a worker node.
        
        Args:
            worker_id: Worker ID
        """
        with self._lock:
            if worker_id in self._workers:
                del self._workers[worker_id]
                logger.info(f"Unregistered worker: {worker_id}")
    
    def submit(
        self,
        function_name: str,
        *args,
        priority: int = 0,
        **kwargs
    ) -> str:
        """
        Submit a task for distributed execution.
        
        Args:
            function_name: Name of registered function
            *args: Function arguments
            priority: Task priority (higher = sooner)
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID
        """
        task = DistributedTask(
            id=str(uuid.uuid4()),
            function_name=function_name,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=self.config.task_timeout,
            retry_count=self.config.retry_count,
        )
        
        with self._lock:
            self._tasks[task.id] = task
        
        self._pending_queue.put(task)
        return task.id
    
    def submit_batch(
        self,
        function_name: str,
        items: List[Tuple[Tuple, Dict]],
        priority: int = 0
    ) -> List[str]:
        """
        Submit a batch of tasks.
        
        Args:
            function_name: Name of registered function
            items: List of (args, kwargs) tuples
            priority: Task priority
            
        Returns:
            List of task IDs
        """
        task_ids = []
        for args, kwargs in items:
            task_id = self.submit(function_name, *args, priority=priority, **kwargs)
            task_ids.append(task_id)
        return task_ids
    
    def get_result(
        self,
        task_id: str,
        timeout: Optional[float] = None
    ) -> Optional[TaskResult]:
        """
        Get result for a task.
        
        Args:
            task_id: Task ID
            timeout: Wait timeout in seconds
            
        Returns:
            Task result if available
        """
        start = time.time()
        timeout = timeout or self.config.task_timeout
        
        while True:
            # Check local worker first
            if self._local_worker:
                result = self._local_worker.get_result(task_id)
                if result:
                    return result
            
            # Check stored results
            with self._lock:
                if task_id in self._results:
                    return self._results[task_id]
            
            # Check timeout
            if time.time() - start > timeout:
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    error="Timeout waiting for result",
                )
            
            time.sleep(0.1)
    
    def get_results(
        self,
        task_ids: List[str],
        timeout: Optional[float] = None
    ) -> List[TaskResult]:
        """
        Get results for multiple tasks.
        
        Args:
            task_ids: List of task IDs
            timeout: Wait timeout in seconds
            
        Returns:
            List of task results
        """
        results = []
        for task_id in task_ids:
            result = self.get_result(task_id, timeout)
            results.append(result)
        return results
    
    def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                # Get pending task
                task = self._pending_queue.get(timeout=1.0)
                
                # Find available worker
                worker = self._select_worker()
                if worker and self._local_worker:
                    # Submit to local worker
                    self._local_worker.submit_task(task)
                else:
                    # Re-queue if no workers available
                    self._pending_queue.put(task)
                    time.sleep(0.5)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
    
    def _select_worker(self) -> Optional[WorkerInfo]:
        """
        Select a worker for task execution.
        
        Returns:
            Selected worker info or None
        """
        with self._lock:
            available = [
                w for w in self._workers.values()
                if w.status != NodeStatus.OFFLINE and w.current_tasks < w.max_tasks
            ]
            
            if not available:
                return None
            
            if self.config.load_balancing == "round_robin":
                self._round_robin_index = (self._round_robin_index + 1) % len(available)
                return available[self._round_robin_index]
            
            elif self.config.load_balancing == "least_loaded":
                return min(available, key=lambda w: w.current_tasks)
            
            else:  # random
                import random
                return random.choice(available)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get coordinator status.
        
        Returns:
            Status dictionary
        """
        with self._lock:
            return {
                "running": self._running,
                "workers": len(self._workers),
                "pending_tasks": self._pending_queue.qsize(),
                "completed_tasks": len(self._results),
                "worker_info": [w.to_dict() for w in self._workers.values()],
            }
    
    def map(
        self,
        function_name: str,
        items: List[Any],
        timeout: Optional[float] = None
    ) -> List[Any]:
        """
        Distributed map operation.
        
        Args:
            function_name: Name of registered function
            items: Items to process
            timeout: Wait timeout
            
        Returns:
            List of results
        """
        # Submit all tasks
        task_ids = []
        for item in items:
            task_id = self.submit(function_name, item)
            task_ids.append(task_id)
        
        # Collect results
        results = self.get_results(task_ids, timeout)
        return [r.result if r and r.status == TaskStatus.COMPLETED else None for r in results]
    
    def reduce(
        self,
        function_name: str,
        items: List[Any],
        reduce_func: Callable[[Any, Any], Any],
        initial: Any = None,
        timeout: Optional[float] = None
    ) -> Any:
        """
        Distributed map-reduce operation.
        
        Args:
            function_name: Name of registered function
            items: Items to process
            reduce_func: Reduction function
            initial: Initial value
            timeout: Wait timeout
            
        Returns:
            Reduced result
        """
        # Map phase
        mapped = self.map(function_name, items, timeout)
        
        # Reduce phase (local)
        result = initial
        for item in mapped:
            if item is not None:
                if result is None:
                    result = item
                else:
                    result = reduce_func(result, item)
        
        return result


class DistributedPipeline:
    """
    Distributed discovery pipeline.
    
    Distributes discovery analysis across multiple nodes.
    """
    
    def __init__(
        self,
        coordinator: Optional[DistributedCoordinator] = None,
        config: Optional[DistributedConfig] = None
    ):
        """
        Initialize distributed pipeline.
        
        Args:
            coordinator: Distributed coordinator
            config: Distributed configuration
        """
        self.config = config or DistributedConfig()
        self.coordinator = coordinator
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the distributed pipeline."""
        if self._initialized:
            return
        
        if not self.coordinator:
            self.coordinator = DistributedCoordinator(self.config)
            self.coordinator.start(local_mode=True)
        
        # Register discovery functions
        self._register_functions()
        self._initialized = True
    
    def _register_functions(self) -> None:
        """Register discovery functions for distributed execution."""
        # Import discovery pipelines
        try:
            from ..pipelines.defi_pipeline import DeFiDiscoveryPipeline
            from ..pipelines.plasma_pipeline import PlasmaDiscoveryPipeline
            from ..pipelines.molecular_pipeline import MolecularDiscoveryPipeline
            from ..pipelines.markets_pipeline import MarketsDiscoveryPipeline
            
            # Register pipeline execution functions
            self.coordinator.register_function(
                "defi_discover",
                lambda data: DeFiDiscoveryPipeline().discover(data)
            )
            self.coordinator.register_function(
                "plasma_discover",
                lambda data: PlasmaDiscoveryPipeline().discover(data)
            )
            self.coordinator.register_function(
                "molecular_discover",
                lambda data: MolecularDiscoveryPipeline().discover(data)
            )
            self.coordinator.register_function(
                "markets_discover",
                lambda data: MarketsDiscoveryPipeline().discover(data)
            )
            
        except ImportError as e:
            logger.warning(f"Could not import pipelines: {e}")
    
    def discover_batch(
        self,
        domain: str,
        items: List[Dict[str, Any]],
        timeout: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Run discovery on a batch of items.
        
        Args:
            domain: Discovery domain
            items: List of input data items
            timeout: Wait timeout
            
        Returns:
            List of discovery results
        """
        if not self._initialized:
            self.initialize()
        
        function_name = f"{domain}_discover"
        return self.coordinator.map(function_name, items, timeout)
    
    def shutdown(self) -> None:
        """Shutdown the distributed pipeline."""
        if self.coordinator:
            self.coordinator.stop()
        self._initialized = False
