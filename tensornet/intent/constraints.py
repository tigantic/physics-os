"""
Constraint System
==================

Define and solve constraints on simulation fields.

Supports:
- Bound constraints (field <= max)
- Relation constraints (fieldA == fieldB)
- Conservation constraints (integral stays constant)
- Custom constraints
"""

from __future__ import annotations

import numpy as np
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from enum import Enum


# =============================================================================
# CONSTRAINT TYPES
# =============================================================================

class ConstraintType(Enum):
    """Type of constraint."""
    BOUND = "bound"           # Upper/lower bound
    EQUALITY = "equality"     # Must equal value
    RELATION = "relation"     # Relation between fields
    INTEGRAL = "integral"     # Conservation/integral
    CUSTOM = "custom"


class ConstraintViolation(Enum):
    """How constraint is violated."""
    NONE = "none"             # Not violated
    BELOW = "below"           # Below lower bound
    ABOVE = "above"           # Above upper bound
    MISMATCH = "mismatch"     # Values don't match


# =============================================================================
# CONSTRAINT BASE
# =============================================================================

@dataclass
class Constraint(ABC):
    """
    Abstract base for field constraints.
    """
    name: str = ""
    priority: int = 0  # Higher = more important
    weight: float = 1.0  # For soft constraints
    is_hard: bool = True  # Hard vs soft constraint
    
    @abstractmethod
    def check(
        self,
        field_data: np.ndarray,
        **context,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Check if constraint is satisfied.
        
        Args:
            field_data: Field data to check
            **context: Additional context (other fields, etc.)
            
        Returns:
            (is_satisfied, violation_amount, details)
        """
        pass
    
    @abstractmethod
    def project(
        self,
        field_data: np.ndarray,
        **context,
    ) -> np.ndarray:
        """
        Project field onto constraint set.
        
        Returns closest field that satisfies constraint.
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "priority": self.priority,
            "weight": self.weight,
            "is_hard": self.is_hard,
        }


# =============================================================================
# BOUND CONSTRAINT
# =============================================================================

@dataclass
class BoundConstraint(Constraint):
    """
    Bound constraint on field values.
    
    Enforces: lower <= field <= upper
    
    Example:
        # Pressure must be positive
        constraint = BoundConstraint(lower=0.0, name="positive_pressure")
        
        # Temperature in range
        constraint = BoundConstraint(lower=200, upper=400, name="temp_range")
    """
    lower: Optional[float] = None
    upper: Optional[float] = None
    
    # Optional region mask
    mask: Optional[np.ndarray] = None
    
    def check(
        self,
        field_data: np.ndarray,
        **context,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Check bound constraint."""
        data = field_data
        if self.mask is not None:
            data = field_data[self.mask]
        
        violations = 0.0
        details = {"violation_type": ConstraintViolation.NONE}
        
        if self.lower is not None:
            below = data < self.lower
            if np.any(below):
                violations += np.sum(self.lower - data[below])
                details["violation_type"] = ConstraintViolation.BELOW
                details["min_value"] = float(np.min(data))
        
        if self.upper is not None:
            above = data > self.upper
            if np.any(above):
                violations += np.sum(data[above] - self.upper)
                details["violation_type"] = ConstraintViolation.ABOVE
                details["max_value"] = float(np.max(data))
        
        return violations == 0, violations, details
    
    def project(
        self,
        field_data: np.ndarray,
        **context,
    ) -> np.ndarray:
        """Project onto bounds."""
        result = field_data.copy()
        
        if self.mask is not None:
            if self.lower is not None:
                result[self.mask] = np.maximum(result[self.mask], self.lower)
            if self.upper is not None:
                result[self.mask] = np.minimum(result[self.mask], self.upper)
        else:
            if self.lower is not None:
                result = np.maximum(result, self.lower)
            if self.upper is not None:
                result = np.minimum(result, self.upper)
        
        return result


# =============================================================================
# RELATION CONSTRAINT
# =============================================================================

@dataclass
class RelationConstraint(Constraint):
    """
    Constraint relating multiple fields.
    
    Example:
        # Pressure gradient equals force
        constraint = RelationConstraint(
            func=lambda p, f: np.gradient(p) - f,
            field_names=["pressure", "force"],
            name="pressure_force_relation"
        )
    """
    func: Callable = None  # Takes fields, returns residual
    field_names: List[str] = field(default_factory=list)
    tolerance: float = 1e-6
    
    def check(
        self,
        field_data: np.ndarray,
        **context,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Check relation constraint."""
        if self.func is None:
            return True, 0.0, {}
        
        # Get all required fields
        fields = [field_data]
        for name in self.field_names[1:]:
            if name in context:
                fields.append(context[name])
            else:
                return False, float('inf'), {"error": f"Missing field: {name}"}
        
        # Compute residual
        residual = self.func(*fields)
        violation = float(np.max(np.abs(residual)))
        
        is_satisfied = violation <= self.tolerance
        
        return is_satisfied, violation, {
            "residual_max": violation,
            "residual_mean": float(np.mean(np.abs(residual))),
        }
    
    def project(
        self,
        field_data: np.ndarray,
        **context,
    ) -> np.ndarray:
        """
        Project onto constraint.
        
        Note: General relation constraints may not have unique projection.
        Returns field as-is by default.
        """
        return field_data


# =============================================================================
# INTEGRAL CONSTRAINT
# =============================================================================

@dataclass
class IntegralConstraint(Constraint):
    """
    Constraint on field integral (conservation).
    
    Example:
        # Total mass conserved
        constraint = IntegralConstraint(
            target_integral=100.0,
            name="mass_conservation"
        )
    """
    target_integral: float = 0.0
    tolerance: float = 1e-6
    weights: Optional[np.ndarray] = None  # Volume weights for integration
    
    def check(
        self,
        field_data: np.ndarray,
        **context,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Check integral constraint."""
        if self.weights is not None:
            integral = np.sum(field_data * self.weights)
        else:
            integral = np.sum(field_data)
        
        error = abs(integral - self.target_integral)
        is_satisfied = error <= self.tolerance
        
        return is_satisfied, error, {
            "current_integral": float(integral),
            "target_integral": self.target_integral,
            "error": float(error),
        }
    
    def project(
        self,
        field_data: np.ndarray,
        **context,
    ) -> np.ndarray:
        """Project to satisfy integral constraint."""
        if self.weights is not None:
            current = np.sum(field_data * self.weights)
            total_weight = np.sum(self.weights)
        else:
            current = np.sum(field_data)
            total_weight = field_data.size
        
        # Add uniform correction
        correction = (self.target_integral - current) / total_weight
        return field_data + correction


# =============================================================================
# CONSTRAINT SET
# =============================================================================

class ConstraintSet:
    """
    Collection of constraints to be satisfied.
    
    Example:
        constraints = ConstraintSet()
        constraints.add(BoundConstraint(lower=0, name="positive"))
        constraints.add(IntegralConstraint(target=100, name="conservation"))
        
        # Check all constraints
        satisfied, violations = constraints.check_all(field_data)
        
        # Project onto feasible set
        projected = constraints.project(field_data)
    """
    
    def __init__(self):
        self._constraints: List[Constraint] = []
    
    def add(self, constraint: Constraint):
        """Add constraint to set."""
        self._constraints.append(constraint)
        # Sort by priority (highest first)
        self._constraints.sort(key=lambda c: c.priority, reverse=True)
    
    def remove(self, name: str):
        """Remove constraint by name."""
        self._constraints = [c for c in self._constraints if c.name != name]
    
    def get(self, name: str) -> Optional[Constraint]:
        """Get constraint by name."""
        for c in self._constraints:
            if c.name == name:
                return c
        return None
    
    def __len__(self) -> int:
        return len(self._constraints)
    
    def __iter__(self):
        return iter(self._constraints)
    
    def check_all(
        self,
        field_data: np.ndarray,
        **context,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check all constraints.
        
        Returns:
            (all_satisfied, dict of violations by constraint name)
        """
        all_satisfied = True
        violations = {}
        
        for constraint in self._constraints:
            satisfied, amount, details = constraint.check(field_data, **context)
            
            if not satisfied:
                all_satisfied = False
            
            violations[constraint.name] = {
                "satisfied": satisfied,
                "violation": amount,
                "details": details,
            }
        
        return all_satisfied, violations
    
    def project(
        self,
        field_data: np.ndarray,
        max_iterations: int = 10,
        **context,
    ) -> np.ndarray:
        """
        Project field onto feasible set (Dykstra's algorithm).
        
        Iteratively projects onto each constraint.
        """
        result = field_data.copy()
        
        # Dykstra's algorithm increments
        increments = [np.zeros_like(field_data) for _ in self._constraints]
        
        for _ in range(max_iterations):
            old = result.copy()
            
            for i, constraint in enumerate(self._constraints):
                if not constraint.is_hard:
                    continue
                
                # Project with increment
                projected = constraint.project(result + increments[i], **context)
                increments[i] = result + increments[i] - projected
                result = projected
            
            # Check convergence
            if np.allclose(result, old):
                break
        
        return result
    
    def soft_penalty(
        self,
        field_data: np.ndarray,
        **context,
    ) -> float:
        """
        Compute total soft constraint penalty.
        
        For use in optimization objectives.
        """
        total = 0.0
        
        for constraint in self._constraints:
            if constraint.is_hard:
                continue
            
            satisfied, amount, _ = constraint.check(field_data, **context)
            total += constraint.weight * amount
        
        return total


# =============================================================================
# CONSTRAINT SOLVER
# =============================================================================

class ConstraintSolver:
    """
    Solver for constraint satisfaction/optimization.
    
    Methods:
    - Project: Find nearest feasible point
    - Optimize: Minimize objective subject to constraints
    - Satisfy: Find any feasible point
    """
    
    def __init__(
        self,
        constraints: Optional[ConstraintSet] = None,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ):
        self.constraints = constraints or ConstraintSet()
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def project(
        self,
        field_data: np.ndarray,
        **context,
    ) -> np.ndarray:
        """Project field onto constraint set."""
        return self.constraints.project(
            field_data,
            max_iterations=self.max_iterations,
            **context,
        )
    
    def optimize(
        self,
        field_data: np.ndarray,
        objective: Callable[[np.ndarray], float],
        gradient: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        learning_rate: float = 0.01,
        **context,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Minimize objective subject to constraints.
        
        Uses projected gradient descent.
        
        Args:
            field_data: Initial field
            objective: Function to minimize
            gradient: Gradient of objective (optional, uses finite diff)
            learning_rate: Step size
            
        Returns:
            (optimized_field, optimization_info)
        """
        result = field_data.copy()
        
        history = {
            "objective": [],
            "constraint_violations": [],
        }
        
        for i in range(self.max_iterations):
            # Compute gradient
            if gradient is not None:
                grad = gradient(result)
            else:
                grad = self._finite_diff_gradient(result, objective)
            
            # Gradient step
            result = result - learning_rate * grad
            
            # Project onto constraints
            result = self.constraints.project(result, **context)
            
            # Record progress
            obj_value = objective(result)
            history["objective"].append(obj_value)
            
            satisfied, violations = self.constraints.check_all(result, **context)
            history["constraint_violations"].append(violations)
            
            # Check convergence
            if i > 0 and abs(history["objective"][-1] - history["objective"][-2]) < self.tolerance:
                break
        
        return result, history
    
    def _finite_diff_gradient(
        self,
        field_data: np.ndarray,
        objective: Callable,
        eps: float = 1e-5,
    ) -> np.ndarray:
        """Compute gradient via finite differences."""
        grad = np.zeros_like(field_data)
        f0 = objective(field_data)
        
        # Sample gradient at random subset for efficiency
        n_samples = min(1000, field_data.size)
        indices = np.random.choice(field_data.size, n_samples, replace=False)
        
        flat = field_data.flatten()
        flat_grad = np.zeros_like(flat)
        
        for idx in indices:
            flat[idx] += eps
            f1 = objective(flat.reshape(field_data.shape))
            flat_grad[idx] = (f1 - f0) / eps
            flat[idx] -= eps
        
        return flat_grad.reshape(field_data.shape)
    
    def is_feasible(
        self,
        field_data: np.ndarray,
        **context,
    ) -> bool:
        """Check if field satisfies all hard constraints."""
        for constraint in self.constraints:
            if constraint.is_hard:
                satisfied, _, _ = constraint.check(field_data, **context)
                if not satisfied:
                    return False
        return True
    
    def violation_summary(
        self,
        field_data: np.ndarray,
        **context,
    ) -> Dict[str, Any]:
        """Get summary of all constraint violations."""
        satisfied, violations = self.constraints.check_all(field_data, **context)
        
        hard_violations = []
        soft_penalty = 0.0
        
        for name, info in violations.items():
            constraint = self.constraints.get(name)
            if constraint and not info["satisfied"]:
                if constraint.is_hard:
                    hard_violations.append(name)
                else:
                    soft_penalty += constraint.weight * info["violation"]
        
        return {
            "feasible": len(hard_violations) == 0,
            "hard_violations": hard_violations,
            "soft_penalty": soft_penalty,
            "details": violations,
        }
