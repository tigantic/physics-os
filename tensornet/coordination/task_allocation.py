# Copyright (c) 2025 Tigantic
# Phase 18: Task Allocation
"""
Distributed task allocation for multi-vehicle systems.

Provides task definitions, allocation algorithms, and auction-based
protocols for optimal task assignment in autonomous vehicle swarms.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

import numpy as np

from tensornet.coordination.swarm import VehicleState


class TaskPriority(Enum):
    """Task priority levels."""
    
    CRITICAL = 1    # Must be executed immediately
    HIGH = 2        # High priority
    NORMAL = 3      # Normal priority
    LOW = 4         # Low priority
    DEFERRED = 5    # Can be delayed


class TaskStatus(Enum):
    """Task execution status."""
    
    PENDING = auto()       # Awaiting assignment
    ASSIGNED = auto()      # Assigned to vehicle
    IN_PROGRESS = auto()   # Being executed
    COMPLETED = auto()     # Successfully completed
    FAILED = auto()        # Execution failed
    CANCELLED = auto()     # Cancelled


@dataclass
class Task:
    """Task definition.
    
    Attributes:
        task_id: Unique task identifier
        task_type: Type of task (e.g., "survey", "deliver", "monitor")
        position: Task location [x, y, z]
        priority: Task priority
        status: Current status
        deadline: Optional deadline timestamp
        duration: Estimated duration (seconds)
        requirements: Required vehicle capabilities
        reward: Task completion reward
        metadata: Additional task data
    """
    
    task_id: str
    task_type: str
    position: np.ndarray
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    deadline: Optional[float] = None
    duration: float = 60.0
    requirements: Set[str] = field(default_factory=set)
    reward: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate and convert fields."""
        self.position = np.asarray(self.position, dtype=np.float64)
        if self.position.shape != (3,):
            raise ValueError("position must be 3D")
        if isinstance(self.requirements, list):
            self.requirements = set(self.requirements)
    
    def __lt__(self, other: "Task") -> bool:
        """Compare by priority for heap operations."""
        return self.priority.value < other.priority.value
    
    def is_expired(self) -> bool:
        """Check if task has expired.
        
        Returns:
            True if past deadline
        """
        if self.deadline is None:
            return False
        return time.time() > self.deadline
    
    def distance_from(self, position: np.ndarray) -> float:
        """Compute distance from a position.
        
        Args:
            position: Reference position
            
        Returns:
            Euclidean distance
        """
        return float(np.linalg.norm(self.position - position))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "position": self.position.tolist(),
            "priority": self.priority.name,
            "status": self.status.name,
            "deadline": self.deadline,
            "duration": self.duration,
            "requirements": list(self.requirements),
            "reward": self.reward,
            "metadata": self.metadata,
        }


@dataclass
class Assignment:
    """Task assignment to a vehicle.
    
    Attributes:
        task: Assigned task
        vehicle_id: Assigned vehicle ID
        cost: Assignment cost (e.g., travel time)
        start_time: Assignment start time
        estimated_completion: Estimated completion time
        utility: Assignment utility value
    """
    
    task: Task
    vehicle_id: str
    cost: float = 0.0
    start_time: float = field(default_factory=time.time)
    estimated_completion: float = 0.0
    utility: float = 0.0
    
    def __post_init__(self) -> None:
        """Compute estimated completion."""
        if self.estimated_completion <= 0:
            self.estimated_completion = self.start_time + self.task.duration + self.cost


class TaskAllocator:
    """Centralized task allocator.
    
    Assigns tasks to vehicles using various allocation strategies
    including greedy, Hungarian, and auction-based methods.
    
    Attributes:
        tasks: List of pending tasks
        assignments: Current task assignments
    """
    
    def __init__(self) -> None:
        """Initialize task allocator."""
        self._tasks: Dict[str, Task] = {}
        self._assignments: Dict[str, Assignment] = {}
        self._vehicle_assignments: Dict[str, List[str]] = {}
    
    @property
    def num_pending_tasks(self) -> int:
        """Get number of pending tasks."""
        return sum(1 for t in self._tasks.values() if t.status == TaskStatus.PENDING)
    
    @property
    def num_assigned_tasks(self) -> int:
        """Get number of assigned tasks."""
        return sum(1 for t in self._tasks.values() if t.status == TaskStatus.ASSIGNED)
    
    def add_task(self, task: Task) -> None:
        """Add a task to the queue.
        
        Args:
            task: Task to add
        """
        self._tasks[task.task_id] = task
    
    def remove_task(self, task_id: str) -> Optional[Task]:
        """Remove a task.
        
        Args:
            task_id: Task ID to remove
            
        Returns:
            Removed task or None
        """
        return self._tasks.pop(task_id, None)
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task or None
        """
        return self._tasks.get(task_id)
    
    def get_pending_tasks(self) -> List[Task]:
        """Get all pending tasks.
        
        Returns:
            List of pending tasks, sorted by priority
        """
        pending = [t for t in self._tasks.values() if t.status == TaskStatus.PENDING]
        return sorted(pending, key=lambda t: t.priority.value)
    
    def compute_cost(
        self,
        task: Task,
        state: VehicleState,
        speed: float = 10.0,
    ) -> float:
        """Compute cost of assigning task to vehicle.
        
        Args:
            task: Task to assign
            state: Vehicle state
            speed: Vehicle speed (m/s)
            
        Returns:
            Assignment cost (travel time + task duration)
        """
        distance = task.distance_from(state.position)
        travel_time = distance / speed
        return travel_time + task.duration
    
    def compute_utility(
        self,
        task: Task,
        state: VehicleState,
        cost: float,
    ) -> float:
        """Compute utility of assignment.
        
        Args:
            task: Task to assign
            state: Vehicle state
            cost: Assignment cost
            
        Returns:
            Utility value (higher is better)
        """
        # Base utility from reward
        utility = task.reward
        
        # Priority bonus
        priority_bonus = (6 - task.priority.value) * 0.2
        utility += priority_bonus
        
        # Deadline urgency
        if task.deadline is not None:
            time_remaining = task.deadline - time.time()
            if time_remaining < cost:
                # Can't complete before deadline
                utility *= 0.1
            elif time_remaining < cost * 2:
                # Urgent
                utility *= 1.5
        
        # Cost penalty
        if cost > 0:
            utility = utility / (1 + cost / 100)
        
        return utility
    
    def allocate_greedy(
        self,
        states: Dict[str, VehicleState],
        max_tasks_per_vehicle: int = 3,
    ) -> List[Assignment]:
        """Greedy task allocation.
        
        Assigns highest utility tasks to vehicles greedily.
        
        Args:
            states: Vehicle states
            max_tasks_per_vehicle: Maximum tasks per vehicle
            
        Returns:
            List of assignments
        """
        pending = self.get_pending_tasks()
        if not pending or not states:
            return []
        
        assignments = []
        vehicle_task_count = {vid: 0 for vid in states}
        assigned_tasks = set()
        
        # Create all (task, vehicle) pairs with utilities
        candidates = []
        for task in pending:
            for vid, state in states.items():
                cost = self.compute_cost(task, state)
                utility = self.compute_utility(task, state, cost)
                candidates.append((utility, task, vid, state, cost))
        
        # Sort by utility (descending)
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Greedy assignment
        for utility, task, vid, state, cost in candidates:
            if task.task_id in assigned_tasks:
                continue
            if vehicle_task_count[vid] >= max_tasks_per_vehicle:
                continue
            
            # Create assignment
            assignment = Assignment(
                task=task,
                vehicle_id=vid,
                cost=cost,
                utility=utility,
            )
            assignments.append(assignment)
            
            task.status = TaskStatus.ASSIGNED
            assigned_tasks.add(task.task_id)
            vehicle_task_count[vid] += 1
            
            self._assignments[task.task_id] = assignment
            
            if vid not in self._vehicle_assignments:
                self._vehicle_assignments[vid] = []
            self._vehicle_assignments[vid].append(task.task_id)
        
        return assignments
    
    def allocate_nearest(
        self,
        states: Dict[str, VehicleState],
        max_tasks_per_vehicle: int = 1,
    ) -> List[Assignment]:
        """Nearest-task allocation.
        
        Assigns each vehicle to its nearest pending task.
        
        Args:
            states: Vehicle states
            max_tasks_per_vehicle: Maximum tasks per vehicle
            
        Returns:
            List of assignments
        """
        pending = self.get_pending_tasks()
        if not pending or not states:
            return []
        
        assignments = []
        assigned_tasks = set()
        
        for vid, state in states.items():
            if len([a for a in assignments if a.vehicle_id == vid]) >= max_tasks_per_vehicle:
                continue
            
            # Find nearest unassigned task
            best_task = None
            best_dist = float('inf')
            
            for task in pending:
                if task.task_id in assigned_tasks:
                    continue
                dist = task.distance_from(state.position)
                if dist < best_dist:
                    best_dist = dist
                    best_task = task
            
            if best_task is not None:
                cost = self.compute_cost(best_task, state)
                assignment = Assignment(
                    task=best_task,
                    vehicle_id=vid,
                    cost=cost,
                    utility=self.compute_utility(best_task, state, cost),
                )
                assignments.append(assignment)
                best_task.status = TaskStatus.ASSIGNED
                assigned_tasks.add(best_task.task_id)
                self._assignments[best_task.task_id] = assignment
        
        return assignments
    
    def complete_task(self, task_id: str) -> bool:
        """Mark a task as completed.
        
        Args:
            task_id: Task ID
            
        Returns:
            True if task was found and completed
        """
        task = self._tasks.get(task_id)
        if task is None:
            return False
        
        task.status = TaskStatus.COMPLETED
        
        # Remove from vehicle assignments
        if task_id in self._assignments:
            assignment = self._assignments[task_id]
            vid = assignment.vehicle_id
            if vid in self._vehicle_assignments:
                if task_id in self._vehicle_assignments[vid]:
                    self._vehicle_assignments[vid].remove(task_id)
        
        return True
    
    def get_vehicle_assignments(self, vehicle_id: str) -> List[Assignment]:
        """Get tasks assigned to a vehicle.
        
        Args:
            vehicle_id: Vehicle ID
            
        Returns:
            List of assignments
        """
        task_ids = self._vehicle_assignments.get(vehicle_id, [])
        return [self._assignments[tid] for tid in task_ids if tid in self._assignments]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get allocation metrics.
        
        Returns:
            Dict of metrics
        """
        total_tasks = len(self._tasks)
        pending = sum(1 for t in self._tasks.values() if t.status == TaskStatus.PENDING)
        assigned = sum(1 for t in self._tasks.values() if t.status == TaskStatus.ASSIGNED)
        completed = sum(1 for t in self._tasks.values() if t.status == TaskStatus.COMPLETED)
        
        total_utility = sum(a.utility for a in self._assignments.values())
        
        return {
            "total_tasks": total_tasks,
            "pending_tasks": pending,
            "assigned_tasks": assigned,
            "completed_tasks": completed,
            "total_assignments": len(self._assignments),
            "total_utility": total_utility,
        }


class AuctionProtocol:
    """Auction-based task allocation.
    
    Implements distributed auction protocols for decentralized
    task allocation in multi-vehicle systems.
    
    Attributes:
        allocator: Underlying task allocator
    """
    
    def __init__(self) -> None:
        """Initialize auction protocol."""
        self._bids: Dict[str, Dict[str, float]] = {}  # task_id -> {vehicle_id -> bid}
        self._allocator = TaskAllocator()
    
    def add_task(self, task: Task) -> None:
        """Add task to auction.
        
        Args:
            task: Task to auction
        """
        self._allocator.add_task(task)
        self._bids[task.task_id] = {}
    
    def submit_bid(
        self,
        task_id: str,
        vehicle_id: str,
        bid: float,
    ) -> bool:
        """Submit a bid for a task.
        
        Args:
            task_id: Task to bid on
            vehicle_id: Bidding vehicle
            bid: Bid amount (utility value)
            
        Returns:
            True if bid was accepted
        """
        if task_id not in self._bids:
            return False
        
        self._bids[task_id][vehicle_id] = bid
        return True
    
    def compute_bid(
        self,
        task: Task,
        state: VehicleState,
    ) -> float:
        """Compute bid for a task.
        
        Args:
            task: Task to bid on
            state: Vehicle state
            
        Returns:
            Bid value
        """
        cost = self._allocator.compute_cost(task, state)
        utility = self._allocator.compute_utility(task, state, cost)
        return utility
    
    def run_auction(
        self,
        tasks: List[Task],
        states: Dict[str, VehicleState],
    ) -> List[Assignment]:
        """Run complete auction round.
        
        Args:
            tasks: Tasks to auction
            states: Vehicle states
            
        Returns:
            Winning assignments
        """
        # Add tasks
        for task in tasks:
            self.add_task(task)
        
        # Collect bids from all vehicles
        for task in tasks:
            for vid, state in states.items():
                bid = self.compute_bid(task, state)
                self.submit_bid(task.task_id, vid, bid)
        
        # Resolve winners
        return self.resolve_winners(states)
    
    def resolve_winners(
        self,
        states: Dict[str, VehicleState],
    ) -> List[Assignment]:
        """Resolve auction winners.
        
        Assigns each task to highest bidder using sequential
        single-item auctions.
        
        Args:
            states: Vehicle states
            
        Returns:
            List of winning assignments
        """
        assignments = []
        assigned_vehicles = set()
        
        # Sort tasks by total bid value (more competed = higher priority)
        task_ids = sorted(
            self._bids.keys(),
            key=lambda tid: sum(self._bids[tid].values()),
            reverse=True,
        )
        
        for task_id in task_ids:
            task = self._allocator.get_task(task_id)
            if task is None or task.status != TaskStatus.PENDING:
                continue
            
            bids = self._bids[task_id]
            
            # Find highest bidder not already assigned
            winner = None
            highest_bid = -float('inf')
            
            for vid, bid in bids.items():
                if vid in assigned_vehicles:
                    continue
                if bid > highest_bid:
                    highest_bid = bid
                    winner = vid
            
            if winner is not None:
                state = states[winner]
                cost = self._allocator.compute_cost(task, state)
                
                assignment = Assignment(
                    task=task,
                    vehicle_id=winner,
                    cost=cost,
                    utility=highest_bid,
                )
                assignments.append(assignment)
                
                task.status = TaskStatus.ASSIGNED
                assigned_vehicles.add(winner)
        
        return assignments
    
    def clear(self) -> None:
        """Clear all bids and reset auction state."""
        self._bids.clear()


def allocate_tasks(
    tasks: List[Task],
    states: Dict[str, VehicleState],
    method: str = "greedy",
) -> List[Assignment]:
    """Allocate tasks to vehicles.
    
    Args:
        tasks: Tasks to allocate
        states: Vehicle states
        method: Allocation method ("greedy", "nearest", "auction")
        
    Returns:
        List of assignments
    """
    if method == "auction":
        auction = AuctionProtocol()
        return auction.run_auction(tasks, states)
    
    allocator = TaskAllocator()
    for task in tasks:
        allocator.add_task(task)
    
    if method == "nearest":
        return allocator.allocate_nearest(states)
    else:
        return allocator.allocate_greedy(states)
