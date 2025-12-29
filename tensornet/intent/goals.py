"""
Goal System
=============

High-level goal specification and planning.

Supports:
- Goal definition (what to achieve)
- Action planning (how to achieve it)
- Multi-step execution
- Progress tracking
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Union
from enum import Enum, auto
import time


# =============================================================================
# GOAL TYPES
# =============================================================================

class GoalType(Enum):
    """Type of goal."""
    OPTIMIZE = auto()        # Minimize/maximize objective
    REACH = auto()           # Reach target state
    MAINTAIN = auto()        # Keep field in bounds
    TRACK = auto()           # Follow trajectory
    EXPLORE = auto()         # Explore parameter space


class GoalStatus(Enum):
    """Status of goal execution."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# =============================================================================
# GOAL
# =============================================================================

@dataclass
class Goal:
    """
    High-level goal specification.
    
    Example:
        # Minimize drag
        goal = Goal(
            name="minimize_drag",
            type=GoalType.OPTIMIZE,
            objective=lambda field: compute_drag(field),
            target_value=0.01,
        )
        
        # Reach steady state
        goal = Goal(
            name="steady_state",
            type=GoalType.REACH,
            target_field=steady_field,
            tolerance=1e-6,
        )
    """
    name: str
    type: GoalType
    
    # For OPTIMIZE goals
    objective: Optional[Callable[[np.ndarray], float]] = None
    minimize: bool = True
    target_value: Optional[float] = None
    
    # For REACH goals
    target_field: Optional[np.ndarray] = None
    tolerance: float = 1e-6
    
    # For MAINTAIN goals
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    
    # For TRACK goals
    trajectory: Optional[List[np.ndarray]] = None
    
    # Execution settings
    max_steps: int = 1000
    timeout: Optional[float] = None  # seconds
    
    # Metadata
    priority: int = 0
    description: str = ""
    
    def is_satisfied(
        self,
        field_data: np.ndarray,
        **context,
    ) -> bool:
        """Check if goal is satisfied."""
        if self.type == GoalType.OPTIMIZE:
            if self.objective is None or self.target_value is None:
                return False
            value = self.objective(field_data)
            return value <= self.target_value if self.minimize else value >= self.target_value
        
        elif self.type == GoalType.REACH:
            if self.target_field is None:
                return False
            error = np.max(np.abs(field_data - self.target_field))
            return error <= self.tolerance
        
        elif self.type == GoalType.MAINTAIN:
            in_bounds = True
            if self.lower_bound is not None:
                in_bounds = in_bounds and np.all(field_data >= self.lower_bound)
            if self.upper_bound is not None:
                in_bounds = in_bounds and np.all(field_data <= self.upper_bound)
            return in_bounds
        
        elif self.type == GoalType.TRACK:
            # Check if at final trajectory point
            if self.trajectory is None or len(self.trajectory) == 0:
                return False
            step = context.get("step", len(self.trajectory) - 1)
            target = self.trajectory[min(step, len(self.trajectory) - 1)]
            error = np.max(np.abs(field_data - target))
            return error <= self.tolerance
        
        return False
    
    def progress(
        self,
        field_data: np.ndarray,
        **context,
    ) -> float:
        """
        Get progress toward goal (0.0 to 1.0).
        """
        if self.type == GoalType.OPTIMIZE:
            if self.objective is None or self.target_value is None:
                return 0.0
            
            initial = context.get("initial_value", float('inf') if self.minimize else float('-inf'))
            current = self.objective(field_data)
            
            if self.minimize:
                total = initial - self.target_value
                if total <= 0:
                    return 1.0
                achieved = initial - current
                return min(1.0, max(0.0, achieved / total))
            else:
                total = self.target_value - initial
                if total <= 0:
                    return 1.0
                achieved = current - initial
                return min(1.0, max(0.0, achieved / total))
        
        elif self.type == GoalType.REACH:
            if self.target_field is None:
                return 0.0
            
            initial_error = context.get("initial_error", np.max(np.abs(self.target_field)))
            current_error = np.max(np.abs(field_data - self.target_field))
            
            if initial_error <= self.tolerance:
                return 1.0
            
            return 1.0 - min(1.0, current_error / initial_error)
        
        elif self.type == GoalType.TRACK:
            if self.trajectory is None:
                return 0.0
            step = context.get("step", 0)
            return step / max(1, len(self.trajectory) - 1)
        
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type.name,
            "priority": self.priority,
            "description": self.description,
        }


# =============================================================================
# PLAN STEP
# =============================================================================

@dataclass
class PlanStep:
    """
    Single step in an execution plan.
    """
    action: str  # Action identifier
    params: Dict[str, Any] = field(default_factory=dict)
    expected_effect: str = ""
    
    # Execution state
    status: GoalStatus = GoalStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    duration: float = 0.0
    
    def execute(
        self,
        field_data: np.ndarray,
        actions: Dict[str, Callable],
        **context,
    ) -> np.ndarray:
        """
        Execute this step.
        
        Args:
            field_data: Current field state
            actions: Dict of action_name -> callable
            
        Returns:
            Updated field
        """
        if self.action not in actions:
            self.status = GoalStatus.FAILED
            self.error = f"Unknown action: {self.action}"
            return field_data
        
        self.status = GoalStatus.IN_PROGRESS
        start = time.time()
        
        try:
            action_fn = actions[self.action]
            result = action_fn(field_data, **self.params, **context)
            
            self.status = GoalStatus.COMPLETED
            self.result = result
            self.duration = time.time() - start
            
            return result
        except Exception as e:
            self.status = GoalStatus.FAILED
            self.error = str(e)
            self.duration = time.time() - start
            return field_data


# =============================================================================
# ACTION PLAN
# =============================================================================

@dataclass
class ActionPlan:
    """
    Sequence of steps to achieve a goal.
    """
    goal: Goal
    steps: List[PlanStep] = field(default_factory=list)
    
    # Execution state
    current_step: int = 0
    status: GoalStatus = GoalStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    def add_step(
        self,
        action: str,
        params: Optional[Dict[str, Any]] = None,
        expected_effect: str = "",
    ):
        """Add step to plan."""
        step = PlanStep(
            action=action,
            params=params or {},
            expected_effect=expected_effect,
        )
        self.steps.append(step)
    
    def execute(
        self,
        field_data: np.ndarray,
        actions: Dict[str, Callable],
        **context,
    ) -> np.ndarray:
        """
        Execute all steps in plan.
        """
        self.status = GoalStatus.IN_PROGRESS
        self.start_time = time.time()
        
        result = field_data
        
        for i, step in enumerate(self.steps):
            self.current_step = i
            result = step.execute(result, actions, **context)
            
            if step.status == GoalStatus.FAILED:
                self.status = GoalStatus.FAILED
                self.end_time = time.time()
                return result
        
        self.status = GoalStatus.COMPLETED
        self.end_time = time.time()
        return result
    
    def progress(self) -> float:
        """Get execution progress (0.0 to 1.0)."""
        if len(self.steps) == 0:
            return 1.0
        
        completed = sum(1 for s in self.steps if s.status == GoalStatus.COMPLETED)
        return completed / len(self.steps)
    
    def summary(self) -> Dict[str, Any]:
        """Get plan execution summary."""
        return {
            "goal": self.goal.name,
            "total_steps": len(self.steps),
            "completed_steps": sum(1 for s in self.steps if s.status == GoalStatus.COMPLETED),
            "failed_steps": sum(1 for s in self.steps if s.status == GoalStatus.FAILED),
            "status": self.status.value,
            "duration": (self.end_time - self.start_time) if self.end_time else None,
        }


# =============================================================================
# GOAL DIRECTOR
# =============================================================================

class GoalDirector:
    """
    Plans and directs goal execution.
    
    Responsible for:
    - Breaking goals into action plans
    - Selecting actions based on current state
    - Monitoring progress
    - Replanning when needed
    
    Example:
        director = GoalDirector()
        
        # Register available actions
        director.register_action("step", simulation_step)
        director.register_action("optimize", gradient_descent)
        
        # Create and execute plan
        plan = director.plan(goal, current_field)
        result = director.execute(plan, current_field)
    """
    
    def __init__(self):
        self._actions: Dict[str, Callable] = {}
        self._planners: Dict[GoalType, Callable] = {}
        
        # Register default planners
        self._planners[GoalType.OPTIMIZE] = self._plan_optimization
        self._planners[GoalType.REACH] = self._plan_reach
        self._planners[GoalType.MAINTAIN] = self._plan_maintain
        self._planners[GoalType.TRACK] = self._plan_track
    
    def register_action(
        self,
        name: str,
        action: Callable[[np.ndarray], np.ndarray],
    ):
        """Register an action that can be used in plans."""
        self._actions[name] = action
    
    def register_planner(
        self,
        goal_type: GoalType,
        planner: Callable[[Goal, np.ndarray], ActionPlan],
    ):
        """Register a custom planner for a goal type."""
        self._planners[goal_type] = planner
    
    def plan(
        self,
        goal: Goal,
        field_data: np.ndarray,
        **context,
    ) -> ActionPlan:
        """
        Create action plan for goal.
        """
        if goal.type in self._planners:
            return self._planners[goal.type](goal, field_data, **context)
        
        # Default: single step plan
        plan = ActionPlan(goal=goal)
        plan.add_step("identity", {}, "No-op")
        return plan
    
    def execute(
        self,
        plan: ActionPlan,
        field_data: np.ndarray,
        **context,
    ) -> np.ndarray:
        """Execute a plan."""
        return plan.execute(field_data, self._actions, **context)
    
    def achieve(
        self,
        goal: Goal,
        field_data: np.ndarray,
        **context,
    ) -> Tuple[np.ndarray, ActionPlan]:
        """
        Plan and execute to achieve goal.
        
        Returns:
            (final_field, executed_plan)
        """
        plan = self.plan(goal, field_data, **context)
        result = self.execute(plan, field_data, **context)
        return result, plan
    
    # -------------------------------------------------------------------------
    # Default planners
    # -------------------------------------------------------------------------
    
    def _plan_optimization(
        self,
        goal: Goal,
        field_data: np.ndarray,
        **context,
    ) -> ActionPlan:
        """Plan for optimization goals."""
        plan = ActionPlan(goal=goal)
        
        # Simple gradient descent plan
        n_steps = min(goal.max_steps, 100)
        
        for i in range(n_steps):
            plan.add_step(
                "gradient_step",
                {"learning_rate": 0.01},
                f"Optimization step {i+1}",
            )
        
        return plan
    
    def _plan_reach(
        self,
        goal: Goal,
        field_data: np.ndarray,
        **context,
    ) -> ActionPlan:
        """Plan for reach goals."""
        plan = ActionPlan(goal=goal)
        
        # Iterative refinement
        for i in range(goal.max_steps):
            plan.add_step(
                "step_toward_target",
                {"target": goal.target_field, "alpha": 0.1},
                f"Move toward target step {i+1}",
            )
        
        return plan
    
    def _plan_maintain(
        self,
        goal: Goal,
        field_data: np.ndarray,
        **context,
    ) -> ActionPlan:
        """Plan for maintain goals."""
        plan = ActionPlan(goal=goal)
        
        # Just clamp to bounds
        plan.add_step(
            "clamp",
            {"lower": goal.lower_bound, "upper": goal.upper_bound},
            "Clamp field to bounds",
        )
        
        return plan
    
    def _plan_track(
        self,
        goal: Goal,
        field_data: np.ndarray,
        **context,
    ) -> ActionPlan:
        """Plan for tracking goals."""
        plan = ActionPlan(goal=goal)
        
        if goal.trajectory is None:
            return plan
        
        # Step through trajectory
        for i, target in enumerate(goal.trajectory):
            plan.add_step(
                "set_target",
                {"target": target, "step": i},
                f"Track point {i+1}/{len(goal.trajectory)}",
            )
        
        return plan


# =============================================================================
# MULTI-GOAL COORDINATOR
# =============================================================================

class GoalCoordinator:
    """
    Coordinates multiple goals.
    
    Handles:
    - Goal prioritization
    - Conflict resolution
    - Parallel/sequential execution
    """
    
    def __init__(self, director: Optional[GoalDirector] = None):
        self.director = director or GoalDirector()
        self._goals: List[Goal] = []
        self._active_plans: Dict[str, ActionPlan] = {}
    
    def add_goal(self, goal: Goal):
        """Add goal to coordinator."""
        self._goals.append(goal)
        self._goals.sort(key=lambda g: g.priority, reverse=True)
    
    def remove_goal(self, name: str):
        """Remove goal by name."""
        self._goals = [g for g in self._goals if g.name != name]
        if name in self._active_plans:
            del self._active_plans[name]
    
    def execute_all(
        self,
        field_data: np.ndarray,
        **context,
    ) -> Tuple[np.ndarray, Dict[str, ActionPlan]]:
        """
        Execute all goals in priority order.
        
        Returns:
            (final_field, dict of plans by goal name)
        """
        result = field_data
        
        for goal in self._goals:
            result, plan = self.director.achieve(goal, result, **context)
            self._active_plans[goal.name] = plan
        
        return result, self._active_plans
    
    def status(self) -> Dict[str, Any]:
        """Get status of all goals."""
        return {
            goal.name: {
                "priority": goal.priority,
                "type": goal.type.name,
                "plan_status": self._active_plans.get(goal.name, ActionPlan(goal=goal)).status.value,
            }
            for goal in self._goals
        }
