"""
Intent - Natural Language Field Steering
==========================================

Layer 6 of the HyperTensor platform.

Provides high-level intent-based control over simulations:
- Natural language queries parsed to field operations
- FieldQuery DSL for structured queries
- Constraint-based optimization
- Goal-directed simulation steering

Components:
    FieldQuery      - Structured query DSL
    IntentParser    - NL to query translation
    ConstraintSolver- Field constraint optimization
    GoalDirector    - Multi-step goal planning

Example:
    from tensornet.intent import FieldQuery, IntentEngine

    engine = IntentEngine(simulation)

    # Natural language steering
    result = engine.execute("increase vorticity near the inlet")

    # Structured query
    query = FieldQuery.where("region == 'inlet'").maximize("vorticity")
    result = engine.execute(query)
"""

from __future__ import annotations

from .constraints import (
                          BoundConstraint,
                          Constraint,
                          ConstraintSet,
                          ConstraintSolver,
                          ConstraintType,
                          IntegralConstraint,
                          RelationConstraint,
)
from .engine import ExecutionContext, IntentEngine, IntentResult, ResultStatus
from .goals import ActionPlan, Goal, GoalCoordinator, GoalDirector, GoalStatus, GoalType, PlanStep
from .parser import EntityExtractor, IntentParser, IntentType, ParseResult

# Query DSL
from .query import Aggregator, FieldQuery, Predicate, QueryBuilder, QueryResult, Selector

# §5 — LLM solver pipeline
from .llm_pipeline import (
    IntentClassifier,
    LLMSolverPipeline,
    MockLLMBackend,
    ParsedQuery,
    SolverDispatcher,
    SolverIntent,
)

__all__ = [
    # Query
    "FieldQuery",
    "QueryBuilder",
    "QueryResult",
    "Predicate",
    "Selector",
    "Aggregator",
    # Parser
    "IntentParser",
    "ParseResult",
    "IntentType",
    "EntityExtractor",
    # Constraints
    "Constraint",
    "ConstraintType",
    "ConstraintSet",
    "ConstraintSolver",
    "BoundConstraint",
    "RelationConstraint",
    "IntegralConstraint",
    # Goals
    "Goal",
    "GoalType",
    "GoalStatus",
    "GoalDirector",
    "GoalCoordinator",
    "ActionPlan",
    "PlanStep",
    # Engine
    "IntentEngine",
    "IntentResult",
    "ExecutionContext",
    "ResultStatus",
    # §5 — LLM solver pipeline
    "SolverIntent",
    "ParsedQuery",
    "MockLLMBackend",
    "IntentClassifier",
    "SolverDispatcher",
    "LLMSolverPipeline",
]

__version__ = "0.1.0"
