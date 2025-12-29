"""
Distributed task scheduling for CFD simulations.

This module provides task scheduling and dependency management
for parallel CFD computations.

Author: HyperTensor Team
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Tuple, Any, Callable
from enum import Enum, auto
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, Future


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = auto()
    READY = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class TaskConfig:
    """Configuration for task execution."""
    max_workers: int = 4
    timeout: float = 300.0  # seconds
    retry_count: int = 3
    checkpoint_enabled: bool = True


@dataclass
class Task:
    """
    Represents a computational task.
    
    Attributes:
        task_id: Unique identifier
        name: Human-readable name
        func: Function to execute
        args: Positional arguments
        kwargs: Keyword arguments
        dependencies: Task IDs this depends on
        priority: Execution priority
    """
    task_id: int
    name: str
    func: Callable
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    dependencies: Set[int] = field(default_factory=set)
    priority: TaskPriority = TaskPriority.NORMAL
    
    # Runtime state
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    def __hash__(self):
        return hash(self.task_id)
    
    @property
    def duration(self) -> Optional[float]:
        """Get task duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class TaskGraph:
    """
    Directed acyclic graph of tasks.
    
    Manages task dependencies and provides topological
    ordering for execution.
    
    Example:
        >>> graph = TaskGraph()
        >>> t1 = graph.add_task("init", init_fn)
        >>> t2 = graph.add_task("compute", compute_fn, deps=[t1])
        >>> t3 = graph.add_task("finalize", final_fn, deps=[t2])
    """
    
    def __init__(self):
        self.tasks: Dict[int, Task] = {}
        self._task_counter = 0
        self._lock = threading.Lock()
    
    def add_task(self, name: str, func: Callable,
                args: Tuple = (), kwargs: Dict[str, Any] = None,
                deps: List[int] = None,
                priority: TaskPriority = TaskPriority.NORMAL) -> int:
        """
        Add a task to the graph.
        
        Args:
            name: Task name
            func: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            deps: Task IDs this depends on
            priority: Task priority
            
        Returns:
            Task ID
        """
        with self._lock:
            task_id = self._task_counter
            self._task_counter += 1
            
            task = Task(
                task_id=task_id,
                name=name,
                func=func,
                args=args,
                kwargs=kwargs or {},
                dependencies=set(deps) if deps else set(),
                priority=priority,
            )
            
            self.tasks[task_id] = task
            
            return task_id
    
    def get_task(self, task_id: int) -> Optional[Task]:
        """Get task by ID."""
        return self.tasks.get(task_id)
    
    def get_ready_tasks(self) -> List[Task]:
        """Get all tasks ready for execution."""
        ready = []
        
        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            
            # Check if all dependencies are completed
            deps_complete = all(
                self.tasks[dep_id].status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
                if dep_id in self.tasks
            )
            
            if deps_complete:
                task.status = TaskStatus.READY
                ready.append(task)
        
        # Sort by priority (higher first)
        ready.sort(key=lambda t: t.priority.value, reverse=True)
        
        return ready
    
    def topological_order(self) -> List[int]:
        """
        Get tasks in topological order.
        
        Returns:
            List of task IDs in execution order
        """
        # Kahn's algorithm
        in_degree = {task_id: len(task.dependencies)
                    for task_id, task in self.tasks.items()}
        
        # Start with tasks that have no dependencies
        queue = [task_id for task_id, degree in in_degree.items()
                if degree == 0]
        
        order = []
        
        while queue:
            # Sort by priority
            queue.sort(key=lambda tid: self.tasks[tid].priority.value, reverse=True)
            task_id = queue.pop(0)
            order.append(task_id)
            
            # Update in-degrees
            for other_id, other_task in self.tasks.items():
                if task_id in other_task.dependencies:
                    in_degree[other_id] -= 1
                    if in_degree[other_id] == 0:
                        queue.append(other_id)
        
        if len(order) != len(self.tasks):
            raise ValueError("Cycle detected in task graph")
        
        return order
    
    def has_cycle(self) -> bool:
        """Check if graph has a cycle."""
        try:
            self.topological_order()
            return False
        except ValueError:
            return True
    
    def get_critical_path(self) -> List[int]:
        """
        Find the critical path (longest execution chain).
        
        Returns:
            List of task IDs on the critical path
        """
        # Use dynamic programming
        order = self.topological_order()
        
        # Distance to each node
        dist: Dict[int, float] = {tid: 0.0 for tid in order}
        parent: Dict[int, Optional[int]] = {tid: None for tid in order}
        
        for task_id in order:
            task = self.tasks[task_id]
            
            for dep_id in task.dependencies:
                new_dist = dist[dep_id] + 1  # Unit weight
                if new_dist > dist[task_id]:
                    dist[task_id] = new_dist
                    parent[task_id] = dep_id
        
        # Find endpoint of critical path
        end_id = max(dist.keys(), key=lambda tid: dist[tid])
        
        # Trace back
        path = []
        current = end_id
        while current is not None:
            path.append(current)
            current = parent[current]
        
        return list(reversed(path))
    
    def reset(self):
        """Reset all tasks to pending state."""
        for task in self.tasks.values():
            task.status = TaskStatus.PENDING
            task.result = None
            task.error = None
            task.start_time = None
            task.end_time = None


class DistributedScheduler:
    """
    Distributed task scheduler.
    
    Executes tasks from a task graph across multiple
    workers with dependency management.
    
    Example:
        >>> scheduler = DistributedScheduler(config)
        >>> scheduler.submit(task_graph)
        >>> results = scheduler.wait_all()
    """
    
    def __init__(self, config: TaskConfig):
        self.config = config
        
        # Thread pool for execution
        self._executor = ThreadPoolExecutor(max_workers=config.max_workers)
        
        # Tracking
        self._graphs: Dict[int, TaskGraph] = {}
        self._futures: Dict[int, Future] = {}
        self._graph_counter = 0
        
        # Synchronization
        self._lock = threading.Lock()
        self._running = False
    
    def submit(self, graph: TaskGraph) -> int:
        """
        Submit a task graph for execution.
        
        Args:
            graph: Task graph to execute
            
        Returns:
            Graph ID for tracking
        """
        with self._lock:
            graph_id = self._graph_counter
            self._graph_counter += 1
            self._graphs[graph_id] = graph
        
        # Start execution
        self._execute_graph(graph_id)
        
        return graph_id
    
    def _execute_graph(self, graph_id: int):
        """Execute all tasks in a graph."""
        graph = self._graphs[graph_id]
        
        def execute_ready():
            while True:
                ready = graph.get_ready_tasks()
                
                if not ready:
                    # Check if all complete
                    all_done = all(
                        t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
                        for t in graph.tasks.values()
                    )
                    if all_done:
                        break
                    
                    # Wait for some tasks to complete
                    time.sleep(0.01)
                    continue
                
                # Submit ready tasks
                for task in ready:
                    task.status = TaskStatus.RUNNING
                    future = self._executor.submit(self._run_task, task)
                    self._futures[task.task_id] = future
        
        # Execute in separate thread
        threading.Thread(target=execute_ready, daemon=True).start()
    
    def _run_task(self, task: Task) -> Any:
        """Execute a single task."""
        task.start_time = time.time()
        
        try:
            result = task.func(*task.args, **task.kwargs)
            task.result = result
            task.status = TaskStatus.COMPLETED
            return result
        
        except Exception as e:
            task.error = e
            task.status = TaskStatus.FAILED
            
            # Retry if configured
            retry_count = 0
            while retry_count < self.config.retry_count:
                retry_count += 1
                try:
                    result = task.func(*task.args, **task.kwargs)
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    task.error = None
                    return result
                except Exception:
                    continue
            
            raise
        
        finally:
            task.end_time = time.time()
    
    def wait_task(self, task_id: int, timeout: Optional[float] = None
                 ) -> Any:
        """Wait for a specific task to complete."""
        future = self._futures.get(task_id)
        
        if future is None:
            raise ValueError(f"Unknown task: {task_id}")
        
        return future.result(timeout=timeout or self.config.timeout)
    
    def wait_graph(self, graph_id: int, timeout: Optional[float] = None
                  ) -> Dict[int, Any]:
        """Wait for all tasks in a graph to complete."""
        graph = self._graphs.get(graph_id)
        
        if graph is None:
            raise ValueError(f"Unknown graph: {graph_id}")
        
        start_time = time.time()
        timeout = timeout or self.config.timeout
        
        while True:
            all_done = all(
                t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
                for t in graph.tasks.values()
            )
            
            if all_done:
                break
            
            if time.time() - start_time > timeout:
                raise TimeoutError("Graph execution timed out")
            
            time.sleep(0.01)
        
        return {tid: t.result for tid, t in graph.tasks.items()
                if t.status == TaskStatus.COMPLETED}
    
    def get_status(self, graph_id: int) -> Dict[int, TaskStatus]:
        """Get status of all tasks in a graph."""
        graph = self._graphs.get(graph_id)
        
        if graph is None:
            return {}
        
        return {tid: t.status for tid, t in graph.tasks.items()}
    
    def cancel(self, graph_id: int):
        """Cancel all pending tasks in a graph."""
        graph = self._graphs.get(graph_id)
        
        if graph is None:
            return
        
        for task in graph.tasks.values():
            if task.status in (TaskStatus.PENDING, TaskStatus.READY):
                task.status = TaskStatus.CANCELLED
    
    def shutdown(self, wait: bool = True):
        """Shutdown the scheduler."""
        self._executor.shutdown(wait=wait)


def schedule_dependency_graph(tasks: Dict[str, Callable],
                             dependencies: Dict[str, List[str]],
                             config: Optional[TaskConfig] = None
                             ) -> Dict[str, Any]:
    """
    Convenience function to schedule tasks with dependencies.
    
    Args:
        tasks: Dictionary mapping task names to functions
        dependencies: Dictionary mapping task names to their dependencies
        config: Scheduler configuration
        
    Returns:
        Dictionary of task results
    """
    if config is None:
        config = TaskConfig()
    
    graph = TaskGraph()
    
    # Add tasks
    name_to_id: Dict[str, int] = {}
    
    for name, func in tasks.items():
        name_to_id[name] = graph.add_task(name, func)
    
    # Add dependencies
    for name, deps in dependencies.items():
        task_id = name_to_id.get(name)
        if task_id is not None:
            task = graph.get_task(task_id)
            for dep_name in deps:
                dep_id = name_to_id.get(dep_name)
                if dep_id is not None:
                    task.dependencies.add(dep_id)
    
    # Execute
    scheduler = DistributedScheduler(config)
    graph_id = scheduler.submit(graph)
    results = scheduler.wait_graph(graph_id)
    scheduler.shutdown()
    
    # Map back to names
    id_to_name = {v: k for k, v in name_to_id.items()}
    return {id_to_name[tid]: result for tid, result in results.items()}


def execute_parallel(funcs: List[Callable], config: Optional[TaskConfig] = None
                    ) -> List[Any]:
    """
    Execute independent functions in parallel.
    
    Args:
        funcs: List of functions to execute
        config: Scheduler configuration
        
    Returns:
        List of results in same order as funcs
    """
    if config is None:
        config = TaskConfig()
    
    graph = TaskGraph()
    task_ids = []
    
    for i, func in enumerate(funcs):
        task_id = graph.add_task(f"task_{i}", func)
        task_ids.append(task_id)
    
    scheduler = DistributedScheduler(config)
    graph_id = scheduler.submit(graph)
    results = scheduler.wait_graph(graph_id)
    scheduler.shutdown()
    
    return [results.get(tid) for tid in task_ids]


def test_scheduler():
    """Test distributed scheduler."""
    print("Testing Distributed Scheduler...")
    
    # Test TaskGraph
    print("\n  Testing TaskGraph...")
    graph = TaskGraph()
    
    # Create tasks
    t1 = graph.add_task("init", lambda: "init_result")
    t2 = graph.add_task("load", lambda: "load_result", deps=[t1])
    t3 = graph.add_task("process_a", lambda: "process_a_result", deps=[t2])
    t4 = graph.add_task("process_b", lambda: "process_b_result", deps=[t2])
    t5 = graph.add_task("combine", lambda: "combine_result", deps=[t3, t4])
    
    print(f"    Created {len(graph.tasks)} tasks")
    
    # Test topological order
    order = graph.topological_order()
    print(f"    Topological order: {order}")
    
    assert order.index(t1) < order.index(t2)
    assert order.index(t2) < order.index(t3)
    assert order.index(t2) < order.index(t4)
    assert order.index(t3) < order.index(t5)
    assert order.index(t4) < order.index(t5)
    
    # Test critical path
    critical_path = graph.get_critical_path()
    print(f"    Critical path: {critical_path}")
    
    assert len(critical_path) == 4  # init -> load -> process_a/b -> combine
    
    # Test ready tasks
    ready = graph.get_ready_tasks()
    assert len(ready) == 1
    assert ready[0].task_id == t1
    print(f"    Initial ready tasks: {[t.name for t in ready]}")
    
    # Test DistributedScheduler
    print("\n  Testing DistributedScheduler...")
    
    config = TaskConfig(max_workers=4, timeout=30.0)
    scheduler = DistributedScheduler(config)
    
    # Create test graph with actual computation
    test_graph = TaskGraph()
    
    results = []
    
    def task_a():
        time.sleep(0.05)
        results.append('a')
        return 1
    
    def task_b():
        time.sleep(0.05)
        results.append('b')
        return 2
    
    def task_c():
        time.sleep(0.05)
        results.append('c')
        return 3
    
    tid_a = test_graph.add_task("A", task_a)
    tid_b = test_graph.add_task("B", task_b)
    tid_c = test_graph.add_task("C", task_c, deps=[tid_a, tid_b])
    
    # Execute
    graph_id = scheduler.submit(test_graph)
    task_results = scheduler.wait_graph(graph_id, timeout=10.0)
    
    print(f"    Results: {task_results}")
    print(f"    Execution order: {results}")
    
    assert 'c' == results[-1]  # C should be last
    assert task_results[tid_a] == 1
    assert task_results[tid_b] == 2
    assert task_results[tid_c] == 3
    
    scheduler.shutdown()
    
    # Test schedule_dependency_graph
    print("\n  Testing schedule_dependency_graph...")
    
    tasks = {
        'load_data': lambda: "data",
        'preprocess': lambda: "preprocessed",
        'train': lambda: "model",
        'evaluate': lambda: "metrics",
    }
    
    deps = {
        'preprocess': ['load_data'],
        'train': ['preprocess'],
        'evaluate': ['train'],
    }
    
    results = schedule_dependency_graph(tasks, deps)
    
    print(f"    Results: {list(results.keys())}")
    assert len(results) == 4
    
    # Test execute_parallel
    print("\n  Testing execute_parallel...")
    
    funcs = [
        lambda: 1 ** 2,
        lambda: 2 ** 2,
        lambda: 3 ** 2,
        lambda: 4 ** 2,
    ]
    
    results = execute_parallel(funcs)
    print(f"    Parallel results: {results}")
    assert results == [1, 4, 9, 16]
    
    print("\nDistributed Scheduler: All tests passed!")


if __name__ == "__main__":
    test_scheduler()
