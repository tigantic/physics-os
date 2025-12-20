"""
Load Balancer Module
====================

Dynamic load balancing for distributed tensor network
computations. Monitors worker loads and redistributes
work to maintain efficiency.

Features:
- Work stealing
- Adaptive partitioning
- Performance monitoring
- Dynamic rebalancing
"""

from __future__ import annotations

import math
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Deque,
)

import numpy as np


class BalancingStrategy(Enum):
    """Load balancing strategies."""
    
    STATIC = auto()          # No rebalancing
    WORK_STEALING = auto()   # Steal from busy workers
    DYNAMIC = auto()         # Dynamic redistribution
    ADAPTIVE = auto()        # ML-based adaptive


class WorkerState(Enum):
    """State of a worker."""
    
    IDLE = auto()
    BUSY = auto()
    WAITING = auto()
    OVERLOADED = auto()


@dataclass
class WorkerStatus:
    """Status of a worker node.
    
    Attributes:
        worker_id: Unique identifier
        state: Current state
        current_load: Current work units
        capacity: Maximum work units
        completed_tasks: Tasks completed
        average_time: Average task time
        last_update: Time of last update
    """
    
    worker_id: int
    state: WorkerState = WorkerState.IDLE
    current_load: int = 0
    capacity: int = 100
    completed_tasks: int = 0
    average_time: float = 0.0
    last_update: float = field(default_factory=time.perf_counter)
    
    @property
    def utilization(self) -> float:
        """Current utilization percentage."""
        return self.current_load / max(1, self.capacity)
    
    @property
    def is_overloaded(self) -> bool:
        """Check if overloaded."""
        return self.utilization > 0.9
    
    @property
    def is_idle(self) -> bool:
        """Check if idle."""
        return self.current_load == 0


@dataclass
class WorkUnit:
    """A unit of work to be balanced.
    
    Attributes:
        work_id: Unique identifier
        partition_id: Associated partition
        estimated_cost: Estimated compute cost
        priority: Priority level
        created_time: Creation timestamp
        assigned_worker: Assigned worker ID
    """
    
    work_id: int
    partition_id: int
    estimated_cost: float = 1.0
    priority: int = 0
    created_time: float = field(default_factory=time.perf_counter)
    assigned_worker: Optional[int] = None
    
    @property
    def age(self) -> float:
        """Age in seconds."""
        return time.perf_counter() - self.created_time


@dataclass
class BalancingResult:
    """Result of load balancing operation.
    
    Attributes:
        num_redistributed: Work units moved
        source_workers: Workers that gave up work
        target_workers: Workers that received work
        time_taken: Time for rebalancing
        imbalance_before: Imbalance metric before
        imbalance_after: Imbalance metric after
    """
    
    num_redistributed: int
    source_workers: List[int]
    target_workers: List[int]
    time_taken: float
    imbalance_before: float
    imbalance_after: float


@dataclass
class LoadBalancerConfig:
    """Configuration for load balancer.
    
    Attributes:
        strategy: Balancing strategy
        threshold_low: Low utilization threshold
        threshold_high: High utilization threshold
        check_interval: Seconds between checks
        enable_work_stealing: Enable work stealing
        max_steal_fraction: Max fraction to steal
    """
    
    strategy: BalancingStrategy = BalancingStrategy.DYNAMIC
    threshold_low: float = 0.3
    threshold_high: float = 0.8
    check_interval: float = 1.0
    enable_work_stealing: bool = True
    max_steal_fraction: float = 0.5


class LoadBalancer:
    """Dynamic load balancer for distributed computations.
    
    Monitors worker loads and redistributes work
    to maintain balanced utilization.
    """
    
    def __init__(
        self,
        num_workers: int,
        config: Optional[LoadBalancerConfig] = None,
    ) -> None:
        """Initialize load balancer.
        
        Args:
            num_workers: Number of workers
            config: Configuration
        """
        self.num_workers = num_workers
        self.config = config or LoadBalancerConfig()
        
        self.workers: Dict[int, WorkerStatus] = {}
        self.work_queues: Dict[int, Deque[WorkUnit]] = {}
        self.work_id_counter = 0
        
        self._lock = threading.Lock()
        self._history: List[BalancingResult] = []
        
        # Initialize workers
        for i in range(num_workers):
            self.workers[i] = WorkerStatus(worker_id=i)
            self.work_queues[i] = deque()
    
    def add_work(
        self,
        partition_id: int,
        estimated_cost: float = 1.0,
        priority: int = 0,
        target_worker: Optional[int] = None,
    ) -> int:
        """Add work unit to be processed.
        
        Args:
            partition_id: Associated partition
            estimated_cost: Estimated cost
            priority: Priority (higher = more urgent)
            target_worker: Specific worker to assign
            
        Returns:
            Work unit ID
        """
        with self._lock:
            work_id = self.work_id_counter
            self.work_id_counter += 1
            
            work = WorkUnit(
                work_id=work_id,
                partition_id=partition_id,
                estimated_cost=estimated_cost,
                priority=priority,
            )
            
            if target_worker is not None:
                worker_id = target_worker
            else:
                worker_id = self._find_best_worker(estimated_cost)
            
            work.assigned_worker = worker_id
            self.work_queues[worker_id].append(work)
            self.workers[worker_id].current_load += 1
            
            self._update_worker_state(worker_id)
            
            return work_id
    
    def _find_best_worker(self, cost: float) -> int:
        """Find best worker for new work.
        
        Args:
            cost: Work cost
            
        Returns:
            Best worker ID
        """
        # Find worker with lowest utilization
        best_id = 0
        best_util = float('inf')
        
        for worker_id, status in self.workers.items():
            if status.utilization < best_util:
                best_util = status.utilization
                best_id = worker_id
        
        return best_id
    
    def _update_worker_state(self, worker_id: int) -> None:
        """Update worker state based on load.
        
        Args:
            worker_id: Worker to update
        """
        status = self.workers[worker_id]
        util = status.utilization
        
        if util == 0:
            status.state = WorkerState.IDLE
        elif util > 0.9:
            status.state = WorkerState.OVERLOADED
        elif util > 0.5:
            status.state = WorkerState.BUSY
        else:
            status.state = WorkerState.WAITING
        
        status.last_update = time.perf_counter()
    
    def complete_work(
        self,
        worker_id: int,
        work_id: int,
        time_taken: float = 0.0,
    ) -> None:
        """Mark work as completed.
        
        Args:
            worker_id: Worker that completed
            work_id: Work ID
            time_taken: Time taken
        """
        with self._lock:
            status = self.workers[worker_id]
            status.current_load = max(0, status.current_load - 1)
            status.completed_tasks += 1
            
            # Update average time
            n = status.completed_tasks
            status.average_time = (
                (status.average_time * (n - 1) + time_taken) / n
            )
            
            self._update_worker_state(worker_id)
    
    def get_next_work(self, worker_id: int) -> Optional[WorkUnit]:
        """Get next work unit for worker.
        
        Args:
            worker_id: Worker requesting work
            
        Returns:
            Next work unit or None
        """
        with self._lock:
            queue = self.work_queues[worker_id]
            
            if queue:
                return queue.popleft()
            
            # Try work stealing
            if self.config.enable_work_stealing:
                return self._steal_work(worker_id)
            
            return None
    
    def _steal_work(self, worker_id: int) -> Optional[WorkUnit]:
        """Steal work from another worker.
        
        Args:
            worker_id: Worker that wants to steal
            
        Returns:
            Stolen work or None
        """
        # Find most loaded worker
        max_load = 0
        source_id = None
        
        for wid, status in self.workers.items():
            if wid != worker_id and status.current_load > max_load:
                max_load = status.current_load
                source_id = wid
        
        if source_id is None or max_load < 2:
            return None
        
        # Steal from back of queue
        source_queue = self.work_queues[source_id]
        if source_queue:
            work = source_queue.pop()
            work.assigned_worker = worker_id
            self.workers[source_id].current_load -= 1
            return work
        
        return None
    
    def compute_imbalance(self) -> float:
        """Compute load imbalance metric.
        
        Returns:
            Imbalance (0 = perfect, 1 = worst)
        """
        loads = [s.current_load for s in self.workers.values()]
        if not loads:
            return 0.0
        
        mean_load = sum(loads) / len(loads)
        if mean_load == 0:
            return 0.0
        
        variance = sum((l - mean_load) ** 2 for l in loads) / len(loads)
        std_dev = math.sqrt(variance)
        
        # Coefficient of variation
        return std_dev / mean_load
    
    def rebalance(self) -> BalancingResult:
        """Perform load rebalancing.
        
        Returns:
            BalancingResult
        """
        start_time = time.perf_counter()
        imbalance_before = self.compute_imbalance()
        
        with self._lock:
            num_redistributed = 0
            source_workers = []
            target_workers = []
            
            if self.config.strategy == BalancingStrategy.STATIC:
                pass  # No rebalancing
            
            elif self.config.strategy in (
                BalancingStrategy.DYNAMIC,
                BalancingStrategy.ADAPTIVE,
            ):
                # Find overloaded and underloaded workers
                overloaded = []
                underloaded = []
                
                for wid, status in self.workers.items():
                    if status.utilization > self.config.threshold_high:
                        overloaded.append(wid)
                    elif status.utilization < self.config.threshold_low:
                        underloaded.append(wid)
                
                # Redistribute
                for source_id in overloaded:
                    if not underloaded:
                        break
                    
                    source_queue = self.work_queues[source_id]
                    source_status = self.workers[source_id]
                    
                    # Calculate how much to move
                    to_move = int(
                        source_status.current_load * 
                        self.config.max_steal_fraction
                    )
                    
                    for _ in range(to_move):
                        if not source_queue or not underloaded:
                            break
                        
                        target_id = underloaded[0]
                        work = source_queue.pop()
                        work.assigned_worker = target_id
                        self.work_queues[target_id].append(work)
                        
                        source_status.current_load -= 1
                        self.workers[target_id].current_load += 1
                        
                        num_redistributed += 1
                        
                        if source_id not in source_workers:
                            source_workers.append(source_id)
                        if target_id not in target_workers:
                            target_workers.append(target_id)
                        
                        # Check if target is now balanced
                        if self.workers[target_id].utilization > self.config.threshold_low:
                            underloaded.remove(target_id)
                    
                    self._update_worker_state(source_id)
                
                for target_id in target_workers:
                    self._update_worker_state(target_id)
        
        imbalance_after = self.compute_imbalance()
        time_taken = time.perf_counter() - start_time
        
        result = BalancingResult(
            num_redistributed=num_redistributed,
            source_workers=source_workers,
            target_workers=target_workers,
            time_taken=time_taken,
            imbalance_before=imbalance_before,
            imbalance_after=imbalance_after,
        )
        
        self._history.append(result)
        return result
    
    def get_worker_status(self, worker_id: int) -> WorkerStatus:
        """Get status of specific worker.
        
        Args:
            worker_id: Worker ID
            
        Returns:
            WorkerStatus
        """
        return self.workers[worker_id]
    
    def get_all_status(self) -> Dict[int, WorkerStatus]:
        """Get status of all workers.
        
        Returns:
            Dict of worker statuses
        """
        return dict(self.workers)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get load balancer statistics.
        
        Returns:
            Statistics dictionary
        """
        total_work = sum(s.current_load for s in self.workers.values())
        total_completed = sum(s.completed_tasks for s in self.workers.values())
        
        return {
            "num_workers": self.num_workers,
            "total_pending_work": total_work,
            "total_completed": total_completed,
            "current_imbalance": self.compute_imbalance(),
            "num_rebalances": len(self._history),
            "worker_utilizations": {
                wid: s.utilization for wid, s in self.workers.items()
            },
        }
    
    def reset(self) -> None:
        """Reset all workers and queues."""
        with self._lock:
            for worker_id in self.workers:
                self.workers[worker_id] = WorkerStatus(worker_id=worker_id)
                self.work_queues[worker_id] = deque()
            
            self.work_id_counter = 0
            self._history = []


def rebalance_workload(
    worker_loads: List[int],
    strategy: BalancingStrategy = BalancingStrategy.DYNAMIC,
) -> List[Tuple[int, int, int]]:
    """Compute workload redistribution.
    
    Args:
        worker_loads: Current load per worker
        strategy: Balancing strategy
        
    Returns:
        List of (source, target, amount) moves
    """
    if strategy == BalancingStrategy.STATIC:
        return []
    
    moves = []
    loads = list(worker_loads)
    n = len(loads)
    
    if n == 0:
        return []
    
    target_load = sum(loads) // n
    remainder = sum(loads) % n
    
    # Calculate target for each worker
    targets = [target_load + (1 if i < remainder else 0) for i in range(n)]
    
    # Find sources and sinks
    excess = [(i, loads[i] - targets[i]) for i in range(n) if loads[i] > targets[i]]
    deficit = [(i, targets[i] - loads[i]) for i in range(n) if loads[i] < targets[i]]
    
    # Match sources to sinks
    for src_id, src_excess in excess:
        for dst_idx, (dst_id, dst_deficit) in enumerate(deficit):
            if dst_deficit == 0:
                continue
            
            amount = min(src_excess, dst_deficit)
            if amount > 0:
                moves.append((src_id, dst_id, amount))
                src_excess -= amount
                deficit[dst_idx] = (dst_id, dst_deficit - amount)
            
            if src_excess == 0:
                break
    
    return moves
