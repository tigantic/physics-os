"""
Intent Engine
==============

Central orchestrator for intent-based field control.

Integrates:
- Natural language parsing
- Query execution
- Constraint satisfaction
- Goal-directed planning
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Union
from enum import Enum, auto
import time

from .parser import IntentParser, ParseResult, IntentType
from .query import FieldQuery, QueryResult
from .constraints import ConstraintSet, ConstraintSolver
from .goals import Goal, GoalDirector, ActionPlan, GoalStatus


# =============================================================================
# EXECUTION RESULT
# =============================================================================

class ResultStatus(Enum):
    """Status of intent execution."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    PENDING = "pending"


@dataclass
class IntentResult:
    """
    Result of intent execution.
    """
    status: ResultStatus
    message: str
    
    # Results
    value: Optional[Any] = None
    field: Optional[np.ndarray] = None
    
    # Execution details
    parse_result: Optional[ParseResult] = None
    query_result: Optional[QueryResult] = None
    plan: Optional[ActionPlan] = None
    
    # Timing
    parse_time: float = 0.0
    execution_time: float = 0.0
    total_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "message": self.message,
            "value": self.value if not isinstance(self.value, np.ndarray) else self.value.tolist(),
            "parse_time": self.parse_time,
            "execution_time": self.execution_time,
            "total_time": self.total_time,
        }


# =============================================================================
# EXECUTION CONTEXT
# =============================================================================

@dataclass
class ExecutionContext:
    """
    Context for intent execution.
    
    Contains:
    - Available fields
    - Actions
    - Constraints
    - History
    """
    # Field state
    fields: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Available actions
    actions: Dict[str, Callable] = field(default_factory=dict)
    
    # Constraints
    constraints: Optional[ConstraintSet] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Execution history
    history: List[IntentResult] = field(default_factory=list)
    
    def get_field(self, name: str) -> Optional[np.ndarray]:
        """Get field by name."""
        return self.fields.get(name)
    
    def set_field(self, name: str, data: np.ndarray):
        """Set field."""
        self.fields[name] = data
    
    def register_action(
        self,
        name: str,
        action: Callable,
    ):
        """Register an action."""
        self.actions[name] = action
    
    def add_result(self, result: IntentResult):
        """Add result to history."""
        self.history.append(result)
    
    @property
    def last_result(self) -> Optional[IntentResult]:
        """Get last result."""
        return self.history[-1] if self.history else None


# =============================================================================
# INTENT ENGINE
# =============================================================================

class IntentEngine:
    """
    Central engine for intent-based field control.
    
    Processes natural language or structured intents and
    executes appropriate actions on fields.
    
    Example:
        engine = IntentEngine()
        
        # Register fields
        engine.register_field("velocity", velocity_field)
        engine.register_field("pressure", pressure_field)
        
        # Register actions
        engine.register_action("step", simulation_step)
        
        # Process intents
        result = engine.execute("show me maximum velocity")
        result = engine.execute("set pressure to 100 at inlet")
        result = engine.execute("minimize drag while maintaining lift > 1000")
    """
    
    def __init__(
        self,
        context: Optional[ExecutionContext] = None,
    ):
        self.context = context or ExecutionContext()
        self.parser = IntentParser()
        self.solver = ConstraintSolver()
        self.director = GoalDirector()
        
        # Query handlers
        self._query_handlers: Dict[IntentType, Callable] = {
            IntentType.QUERY_VALUE: self._handle_query_value,
            IntentType.QUERY_LOCATION: self._handle_query_location,
            IntentType.QUERY_COMPARE: self._handle_query_comparison,
            IntentType.QUERY_TREND: self._handle_query_statistics,
        }
        
        # Action handlers
        self._action_handlers: Dict[IntentType, Callable] = {
            IntentType.ACTION_SET: self._handle_action_set,
            IntentType.ACTION_INCREASE: self._handle_action_modify,
            IntentType.ACTION_DECREASE: self._handle_action_modify,
            IntentType.ACTION_OPTIMIZE: self._handle_action_optimize,
        }
        
        # Control handlers
        self._control_handlers: Dict[IntentType, Callable] = {
            IntentType.CONTROL_RUN: self._handle_control_start,
            IntentType.CONTROL_STOP: self._handle_control_stop,
            IntentType.CONTROL_RESET: self._handle_control_reset,
        }
    
    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------
    
    def register_field(
        self,
        name: str,
        data: np.ndarray,
    ):
        """Register a field for querying/modification."""
        self.context.set_field(name, data)
    
    def register_action(
        self,
        name: str,
        action: Callable,
    ):
        """Register an action that can be invoked."""
        self.context.register_action(name, action)
        self.director.register_action(name, action)
    
    def register_constraint(
        self,
        constraint,
    ):
        """Register a constraint."""
        if self.context.constraints is None:
            self.context.constraints = ConstraintSet()
        self.context.constraints.add(constraint)
        self.solver.constraints = self.context.constraints
    
    # -------------------------------------------------------------------------
    # Main execution
    # -------------------------------------------------------------------------
    
    def execute(
        self,
        intent: Union[str, ParseResult, FieldQuery, Goal],
        **kwargs,
    ) -> IntentResult:
        """
        Execute an intent.
        
        Args:
            intent: Natural language string, parsed intent, query, or goal
            **kwargs: Additional context
            
        Returns:
            IntentResult with execution details
        """
        start_time = time.time()
        
        try:
            # Parse if string
            if isinstance(intent, str):
                parse_start = time.time()
                parsed = self.parser.parse(intent)
                parse_time = time.time() - parse_start
            elif isinstance(intent, ParseResult):
                parsed = intent
                parse_time = 0.0
            elif isinstance(intent, FieldQuery):
                # Execute query directly
                return self._execute_query(intent, start_time)
            elif isinstance(intent, Goal):
                # Execute goal directly
                return self._execute_goal(intent, start_time)
            else:
                return IntentResult(
                    status=ResultStatus.FAILED,
                    message=f"Unknown intent type: {type(intent)}",
                    total_time=time.time() - start_time,
                )
            
            # Route to appropriate handler
            exec_start = time.time()
            
            if parsed.intent_type in self._query_handlers:
                result = self._query_handlers[parsed.intent_type](parsed, **kwargs)
            elif parsed.intent_type in self._action_handlers:
                result = self._action_handlers[parsed.intent_type](parsed, **kwargs)
            elif parsed.intent_type in self._control_handlers:
                result = self._control_handlers[parsed.intent_type](parsed, **kwargs)
            elif parsed.intent_type == IntentType.INFO_HELP:
                result = self._handle_help(parsed, **kwargs)
            elif parsed.intent_type == IntentType.INFO_STATUS:
                result = self._handle_status(parsed, **kwargs)
            else:
                result = IntentResult(
                    status=ResultStatus.FAILED,
                    message=f"Unhandled intent type: {parsed.intent_type}",
                )
            
            # Add timing
            result.parse_result = parsed
            result.parse_time = parse_time
            result.execution_time = time.time() - exec_start
            result.total_time = time.time() - start_time
            
            # Record in history
            self.context.add_result(result)
            
            return result
            
        except Exception as e:
            return IntentResult(
                status=ResultStatus.FAILED,
                message=f"Execution error: {str(e)}",
                total_time=time.time() - start_time,
            )
    
    # -------------------------------------------------------------------------
    # Query handlers
    # -------------------------------------------------------------------------
    
    def _execute_query(
        self,
        query: FieldQuery,
        start_time: float,
    ) -> IntentResult:
        """Execute a FieldQuery directly."""
        # Get field data
        field_name = query.field_name or "default"
        data = self.context.get_field(field_name)
        
        if data is None:
            return IntentResult(
                status=ResultStatus.FAILED,
                message=f"Field not found: {field_name}",
                total_time=time.time() - start_time,
            )
        
        # Execute query
        query_result = query.execute(data)
        
        return IntentResult(
            status=ResultStatus.SUCCESS,
            message=f"Query executed on {field_name}",
            value=query_result.value,
            query_result=query_result,
            execution_time=time.time() - start_time,
            total_time=time.time() - start_time,
        )
    
    def _handle_query_value(
        self,
        parsed: ParseResult,
        **kwargs,
    ) -> IntentResult:
        """Handle value query."""
        if parsed.query is None:
            return IntentResult(
                status=ResultStatus.FAILED,
                message="No query constructed from intent",
            )
        
        # Find field
        field_name = parsed.field_name or "default"
        
        data = self.context.get_field(field_name)
        if data is None:
            return IntentResult(
                status=ResultStatus.FAILED,
                message=f"Field not found: {field_name}",
            )
        
        # Execute query
        query_result = parsed.query.execute(data)
        
        return IntentResult(
            status=ResultStatus.SUCCESS,
            message=f"Value from {field_name}: {query_result.value}",
            value=query_result.value,
            query_result=query_result,
        )
    
    def _handle_query_location(
        self,
        parsed: ParseResult,
        **kwargs,
    ) -> IntentResult:
        """Handle location query."""
        # Find field
        field_name = parsed.field_name or "default"
        
        data = self.context.get_field(field_name)
        if data is None:
            return IntentResult(
                status=ResultStatus.FAILED,
                message=f"Field not found: {field_name}",
            )
        
        # Find location of max/min
        if "maximum" in parsed.text.lower() or "max" in parsed.text.lower():
            idx = np.unravel_index(np.argmax(data), data.shape)
            value = data[idx]
            location = {"index": idx, "value": float(value)}
        else:
            idx = np.unravel_index(np.argmin(data), data.shape)
            value = data[idx]
            location = {"index": idx, "value": float(value)}
        
        return IntentResult(
            status=ResultStatus.SUCCESS,
            message=f"Location found at index {idx}",
            value=location,
        )
    
    def _handle_query_statistics(
        self,
        parsed: ParseResult,
        **kwargs,
    ) -> IntentResult:
        """Handle statistics query."""
        field_name = parsed.field_name or "default"
        
        data = self.context.get_field(field_name)
        if data is None:
            return IntentResult(
                status=ResultStatus.FAILED,
                message=f"Field not found: {field_name}",
            )
        
        stats = {
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "shape": data.shape,
        }
        
        return IntentResult(
            status=ResultStatus.SUCCESS,
            message=f"Statistics for {field_name}",
            value=stats,
        )
    
    def _handle_query_comparison(
        self,
        parsed: ParseResult,
        **kwargs,
    ) -> IntentResult:
        """Handle comparison query."""
        fields = [e.value for e in parsed.entities if e.entity_type == "field"]
        
        if len(fields) < 2:
            return IntentResult(
                status=ResultStatus.PARTIAL,
                message="Need two fields to compare",
            )
        
        data1 = self.context.get_field(fields[0])
        data2 = self.context.get_field(fields[1])
        
        if data1 is None or data2 is None:
            return IntentResult(
                status=ResultStatus.FAILED,
                message="One or both fields not found",
            )
        
        comparison = {
            "field1": fields[0],
            "field2": fields[1],
            "diff_max": float(np.max(np.abs(data1 - data2))),
            "diff_mean": float(np.mean(np.abs(data1 - data2))),
            "correlation": float(np.corrcoef(data1.flatten(), data2.flatten())[0, 1]) if data1.shape == data2.shape else None,
        }
        
        return IntentResult(
            status=ResultStatus.SUCCESS,
            message=f"Comparison between {fields[0]} and {fields[1]}",
            value=comparison,
        )
    
    # -------------------------------------------------------------------------
    # Action handlers
    # -------------------------------------------------------------------------
    
    def _handle_action_set(
        self,
        parsed: ParseResult,
        **kwargs,
    ) -> IntentResult:
        """Handle set action."""
        field_name = parsed.field_name or "default"
        value = parsed.value
        
        if value is None:
            return IntentResult(
                status=ResultStatus.FAILED,
                message="No value specified for set action",
            )
        
        data = self.context.get_field(field_name)
        if data is None:
            # Create new field
            data = np.full((10, 10), value)
        else:
            data = np.full_like(data, value)
        
        self.context.set_field(field_name, data)
        
        return IntentResult(
            status=ResultStatus.SUCCESS,
            message=f"Set {field_name} to {value}",
            field=data,
        )
    
    def _handle_action_modify(
        self,
        parsed: ParseResult,
        **kwargs,
    ) -> IntentResult:
        """Handle modify action."""
        field_name = parsed.field_name or "default"
        
        data = self.context.get_field(field_name)
        if data is None:
            return IntentResult(
                status=ResultStatus.FAILED,
                message=f"Field not found: {field_name}",
            )
        
        # Parse modification
        text = parsed.text.lower()
        amount = parsed.value if parsed.value is not None else 1.0
        
        if "increase" in text or "add" in text:
            data = data + amount
        elif "decrease" in text or "subtract" in text:
            data = data - amount
        elif "scale" in text or "multiply" in text:
            data = data * amount
        
        self.context.set_field(field_name, data)
        
        return IntentResult(
            status=ResultStatus.SUCCESS,
            message=f"Modified {field_name}",
            field=data,
        )
    
    def _handle_action_optimize(
        self,
        parsed: ParseResult,
        **kwargs,
    ) -> IntentResult:
        """Handle optimization action."""
        # Find field and objective
        field_name = parsed.field_name or "default"
        minimize = True
        
        text = parsed.text.lower()
        if "maximize" in text:
            minimize = False
        
        data = self.context.get_field(field_name)
        if data is None:
            return IntentResult(
                status=ResultStatus.FAILED,
                message=f"Field not found: {field_name}",
            )
        
        # Simple optimization: project onto constraints
        if self.solver.constraints and len(self.solver.constraints) > 0:
            optimized = self.solver.project(data)
            self.context.set_field(field_name, optimized)
            
            return IntentResult(
                status=ResultStatus.SUCCESS,
                message=f"Optimized {field_name} subject to constraints",
                field=optimized,
            )
        
        return IntentResult(
            status=ResultStatus.PARTIAL,
            message="No constraints defined for optimization",
        )
    
    # -------------------------------------------------------------------------
    # Control handlers
    # -------------------------------------------------------------------------
    
    def _handle_control_start(
        self,
        parsed: ParseResult,
        **kwargs,
    ) -> IntentResult:
        """Handle start control."""
        action_name = parsed.operator
        
        if action_name and action_name in self.context.actions:
            # Execute action
            field = list(self.context.fields.values())[0] if self.context.fields else None
            if field is not None:
                result = self.context.actions[action_name](field)
                return IntentResult(
                    status=ResultStatus.SUCCESS,
                    message=f"Started {action_name}",
                    field=result,
                )
        
        return IntentResult(
            status=ResultStatus.SUCCESS,
            message="Simulation started",
        )
    
    def _handle_control_stop(
        self,
        parsed: ParseResult,
        **kwargs,
    ) -> IntentResult:
        """Handle stop control."""
        return IntentResult(
            status=ResultStatus.SUCCESS,
            message="Simulation stopped",
        )
    
    def _handle_control_reset(
        self,
        parsed: ParseResult,
        **kwargs,
    ) -> IntentResult:
        """Handle reset control."""
        # Clear all fields
        self.context.fields.clear()
        self.context.history.clear()
        
        return IntentResult(
            status=ResultStatus.SUCCESS,
            message="System reset",
        )
    
    # -------------------------------------------------------------------------
    # Info handlers
    # -------------------------------------------------------------------------
    
    def _handle_help(
        self,
        parsed: ParseResult,
        **kwargs,
    ) -> IntentResult:
        """Handle help request."""
        help_text = """
Available commands:
- Queries: "show maximum velocity", "what is the mean pressure"
- Actions: "set pressure to 100", "increase velocity by 10"
- Control: "start simulation", "stop", "reset"
- Optimization: "minimize drag", "optimize pressure"
        """
        
        return IntentResult(
            status=ResultStatus.SUCCESS,
            message=help_text.strip(),
            value={
                "fields": list(self.context.fields.keys()),
                "actions": list(self.context.actions.keys()),
            },
        )
    
    def _handle_status(
        self,
        parsed: ParseResult,
        **kwargs,
    ) -> IntentResult:
        """Handle status request."""
        status = {
            "fields": list(self.context.fields.keys()),
            "num_actions": len(self.context.actions),
            "num_constraints": len(self.context.constraints) if self.context.constraints else 0,
            "history_length": len(self.context.history),
        }
        
        return IntentResult(
            status=ResultStatus.SUCCESS,
            message="System status",
            value=status,
        )
    
    # -------------------------------------------------------------------------
    # Goal execution
    # -------------------------------------------------------------------------
    
    def _execute_goal(
        self,
        goal: Goal,
        start_time: float,
    ) -> IntentResult:
        """Execute a Goal directly."""
        # Get primary field
        if not self.context.fields:
            return IntentResult(
                status=ResultStatus.FAILED,
                message="No fields registered",
                total_time=time.time() - start_time,
            )
        
        field_name, data = list(self.context.fields.items())[0]
        
        # Plan and execute
        result, plan = self.director.achieve(goal, data)
        
        # Update field
        self.context.set_field(field_name, result)
        
        status = ResultStatus.SUCCESS if plan.status == GoalStatus.COMPLETED else ResultStatus.PARTIAL
        
        return IntentResult(
            status=status,
            message=f"Goal '{goal.name}' executed",
            field=result,
            plan=plan,
            execution_time=time.time() - start_time,
            total_time=time.time() - start_time,
        )
    
    # -------------------------------------------------------------------------
    # Conversational interface
    # -------------------------------------------------------------------------
    
    def chat(
        self,
        message: str,
    ) -> str:
        """
        Simple conversational interface.
        
        Returns natural language response.
        """
        result = self.execute(message)
        
        if result.status == ResultStatus.SUCCESS:
            if result.value is not None:
                if isinstance(result.value, dict):
                    # Format dict nicely
                    lines = [result.message]
                    for k, v in result.value.items():
                        lines.append(f"  {k}: {v}")
                    return "\n".join(lines)
                return f"{result.message}: {result.value}"
            return result.message
        elif result.status == ResultStatus.PARTIAL:
            return f"Partial result: {result.message}"
        else:
            return f"Error: {result.message}"
