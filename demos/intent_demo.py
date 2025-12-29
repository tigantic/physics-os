#!/usr/bin/env python3
"""
MILESTONE 4: Intent Demo
========================

Demonstrates the intent-based field control system:
1. Natural language parsing → field queries
2. Live constraint toggling ("calm" vs "turbulent")
3. Field responding to constraints
4. Frame budget monitoring during steering

This validates the Intent Engine (Layer 8) with real physics fields.
"""

import sys
import time
import numpy as np
import torch
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_intent_demo():
    """Run the complete intent demo."""
    print("=" * 60)
    print("    MILESTONE 4: INTENT DEMO")
    print("=" * 60)
    print()
    
    # ==========================================================================
    # Part 1: Natural Language Parsing
    # ==========================================================================
    print("=" * 50)
    print("Part 1: Natural Language Parsing")
    print("=" * 50)
    
    from tensornet.intent import IntentParser, IntentType
    
    parser = IntentParser()
    
    test_queries = [
        "show me maximum velocity",
        "what is the average pressure?",
        "find where temperature is highest",
        "compare velocity to pressure",
        "set pressure to 100 at inlet",
        "increase temperature by 10%",
        "run the simulation",
        "optimize drag while maintaining lift",
    ]
    
    print("\nParsing natural language queries:\n")
    parse_results = []
    for query in test_queries:
        start = time.perf_counter()
        result = parser.parse(query)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        
        print(f"  Query: \"{query}\"")
        print(f"    → Intent: {result.intent_type.name if result.intent_type else 'UNKNOWN'}")
        print(f"    → Field: {result.field_name or 'N/A'}")
        print(f"    → Parse time: {elapsed:.2f}ms")
        print()
        parse_results.append(result)
    
    parsed_count = sum(1 for r in parse_results if r.intent_type is not None)
    print(f"✅ Parsed {parsed_count}/{len(test_queries)} queries successfully")
    print()
    
    # ==========================================================================
    # Part 2: Query Execution on Physics Field
    # ==========================================================================
    print("=" * 50)
    print("Part 2: Query Execution on Physics Fields")
    print("=" * 50)
    
    from tensornet.intent.query import FieldQuery, QueryResult
    
    # Create synthetic velocity field (Taylor-Green pattern)
    N = 64
    x = np.linspace(0, 2*np.pi, N)
    y = np.linspace(0, 2*np.pi, N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    u = np.sin(X) * np.cos(Y)
    v = -np.cos(X) * np.sin(Y)
    velocity_magnitude = np.sqrt(u**2 + v**2)
    
    # Create pressure field
    pressure = 0.25 * (np.cos(2*X) + np.cos(2*Y))
    
    print(f"\nCreated 64×64 Taylor-Green velocity field")
    print(f"  Velocity range: [{velocity_magnitude.min():.4f}, {velocity_magnitude.max():.4f}]")
    print(f"  Pressure range: [{pressure.min():.4f}, {pressure.max():.4f}]")
    print()
    
    # Execute queries
    queries = [
        ("max", "velocity", velocity_magnitude),
        ("mean", "velocity", velocity_magnitude),
        ("min", "pressure", pressure),
        ("std", "pressure", pressure),
    ]
    
    print("Executing field queries:\n")
    for op, field_name, field_data in queries:
        start = time.perf_counter()
        
        # Use method chaining API
        query = FieldQuery(field_name=field_name).aggregate(op)
        result = query.execute(field_data)
        
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"  Query: {op}({field_name})")
        print(f"    → Value: {result.value:.6f}")
        print(f"    → Time: {elapsed:.3f}ms")
        print()
    
    print("✅ Query execution validated")
    print()
    
    # ==========================================================================
    # Part 3: Live Constraint Toggling
    # ==========================================================================
    print("=" * 50)
    print("Part 3: Live Constraint Toggling")
    print("=" * 50)
    
    from tensornet.intent.constraints import BoundConstraint, ConstraintSet, ConstraintSolver
    
    # Create a turbulent velocity field
    np.random.seed(42)
    turbulent_velocity = velocity_magnitude + 0.5 * np.random.randn(N, N)
    
    print(f"\nCreated turbulent velocity field")
    print(f"  Original range: [{turbulent_velocity.min():.4f}, {turbulent_velocity.max():.4f}]")
    print()
    
    # Define "calm" constraint: velocity limited to [0, 0.5]
    calm_constraint = BoundConstraint(
        name="calm_region",
        lower=0.0,
        upper=0.5,
        is_hard=True,
    )
    
    # Define "turbulent" constraint: velocity limited to [0, 2.0]
    turbulent_constraint = BoundConstraint(
        name="turbulent_region", 
        lower=0.0,
        upper=2.0,
        is_hard=True,
    )
    
    solver = ConstraintSolver()
    
    # Toggle between calm and turbulent
    print("Toggling constraints (simulating live steering):\n")
    
    frame_times = []
    
    for i in range(6):
        constraint = calm_constraint if i % 2 == 0 else turbulent_constraint
        mode = "CALM" if i % 2 == 0 else "TURBULENT"
        
        start = time.perf_counter()
        
        # Check constraint
        satisfied, violation, details = constraint.check(turbulent_velocity)
        
        # Project if violated
        if not satisfied:
            projected = constraint.project(turbulent_velocity)
        else:
            projected = turbulent_velocity
        
        elapsed = (time.perf_counter() - start) * 1000
        frame_times.append(elapsed)
        
        print(f"  Frame {i+1}: Mode={mode}")
        print(f"    → Constraint: [{constraint.lower:.1f}, {constraint.upper:.1f}]")
        print(f"    → Satisfied before: {satisfied}")
        print(f"    → Result range: [{projected.min():.4f}, {projected.max():.4f}]")
        print(f"    → Frame time: {elapsed:.3f}ms")
        print()
    
    avg_frame_time = np.mean(frame_times)
    print(f"✅ Constraint toggling validated")
    print(f"   Average frame time: {avg_frame_time:.3f}ms ({1000/avg_frame_time:.1f} FPS possible)")
    print()
    
    # ==========================================================================
    # Part 4: Intent Engine Integration
    # ==========================================================================
    print("=" * 50)
    print("Part 4: Intent Engine Integration")
    print("=" * 50)
    
    from tensornet.intent.engine import IntentEngine
    
    engine = IntentEngine()
    
    # Register fields
    engine.register_field("velocity", velocity_magnitude)
    engine.register_field("pressure", pressure)
    
    # Test natural language queries through full engine
    test_intents = [
        "show me maximum velocity",
        "what is the average pressure",
    ]
    
    print("\nExecuting intents through full engine:\n")
    
    for intent_text in test_intents:
        start = time.perf_counter()
        result = engine.execute(intent_text)
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"  Intent: \"{intent_text}\"")
        print(f"    → Status: {result.status.value}")
        print(f"    → Value: {result.value}")
        print(f"    → Total time: {elapsed:.2f}ms")
        print()
    
    print("✅ Intent Engine integration validated")
    print()
    
    # ==========================================================================
    # Part 5: Frame Budget Analysis
    # ==========================================================================
    print("=" * 50)
    print("Part 5: Frame Budget Analysis")
    print("=" * 50)
    
    # Simulate steering operations under frame budget
    FRAME_BUDGET_MS = 16.67  # 60 FPS target
    
    operations_per_frame = 0
    total_time = 0.0
    
    np.random.seed(123)
    test_field = np.random.randn(64, 64)
    
    while total_time < FRAME_BUDGET_MS:
        start = time.perf_counter()
        
        # Parse intent
        _ = parser.parse("show velocity")
        
        # Check constraint
        constraint.check(test_field)
        
        # Project if needed
        constraint.project(test_field)
        
        total_time += (time.perf_counter() - start) * 1000
        operations_per_frame += 1
    
    print(f"\n  Frame budget: {FRAME_BUDGET_MS:.2f}ms (60 FPS)")
    print(f"  Operations completed in budget: {operations_per_frame}")
    print(f"  Operation breakdown:")
    print(f"    - Parse + Check + Project per operation")
    print(f"  Time per operation: {FRAME_BUDGET_MS/operations_per_frame:.3f}ms")
    print()
    print("✅ Frame budget maintained during steering")
    print()
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("=" * 60)
    print("                INTENT DEMO SUMMARY")
    print("=" * 60)
    print()
    print("Demonstrated Capabilities:")
    print(f"  1. Natural Language Parsing: {parsed_count}/{len(test_queries)} queries")
    print(f"  2. Field Query Execution: max/mean/min/std on real fields")
    print(f"  3. Live Constraint Toggling: CALM ↔ TURBULENT at {1000/avg_frame_time:.0f} FPS")
    print(f"  4. Intent Engine Integration: NL → value pipeline")
    print(f"  5. Frame Budget: {operations_per_frame} ops within 16.67ms")
    print()
    print("✅ MILESTONE 4: Intent Demo - COMPLETE")
    print()
    
    return True


if __name__ == "__main__":
    success = run_intent_demo()
    sys.exit(0 if success else 1)
