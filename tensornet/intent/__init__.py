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

# Query DSL
from .query import (
    FieldQuery,
    QueryBuilder,
    QueryResult,
    Predicate,
    Selector,
    Aggregator,
)

from .parser import (
    IntentParser,
    ParseResult,
    IntentType,
    EntityExtractor,
)

from .constraints import (
    Constraint,
    ConstraintType,
    ConstraintSet,
    ConstraintSolver,
    BoundConstraint,
    RelationConstraint,
    IntegralConstraint,
)

from .goals import (
    Goal,
    GoalType,
    GoalStatus,
    GoalDirector,
    GoalCoordinator,
    ActionPlan,
    PlanStep,
)

from .engine import (
    IntentEngine,
    IntentResult,
    ExecutionContext,
    ResultStatus,
)

__all__ = [
    # Query
    'FieldQuery',
    'QueryBuilder',
    'QueryResult',
    'Predicate',
    'Selector',
    'Aggregator',
    
    # Parser
    'IntentParser',
    'ParseResult',
    'IntentType',
    'EntityExtractor',
    
    # Constraints
    'Constraint',
    'ConstraintType',
    'ConstraintSet',
    'ConstraintSolver',
    'BoundConstraint',
    'RelationConstraint',
    'IntegralConstraint',
    
    # Goals
    'Goal',
    'GoalType',
    'GoalStatus',
    'GoalDirector',
    'GoalCoordinator',
    'ActionPlan',
    'PlanStep',
    
    # Engine
    'IntentEngine',
    'IntentResult',
    'ExecutionContext',
    'ResultStatus',
]

__version__ = '0.1.0'
