"""
Layer 8 Audit: Intent/FDL Validation
=====================================

Validates that the Intent Engine can:
1. Parse natural language commands
2. Execute queries on real field data
3. Apply constraints to physics
"""

import numpy as np


def test_intent_parser():
    """Test natural language parsing."""
    from ontic.applied.intent import IntentParser, IntentType

    parser = IntentParser()

    # Parse various intents (use exact phrases from test_intent.py)
    result = parser.parse("show me the maximum velocity")
    assert (
        result.intent_type == IntentType.QUERY_VALUE
    ), f"Expected QUERY_VALUE, got {result.intent_type}"

    result = parser.parse("set pressure to 100")
    assert (
        result.intent_type == IntentType.ACTION_SET
    ), f"Expected ACTION_SET, got {result.intent_type}"

    result = parser.parse("minimize drag")
    assert (
        result.intent_type == IntentType.ACTION_OPTIMIZE
    ), f"Expected ACTION_OPTIMIZE, got {result.intent_type}"

    return True, "3/3 intents parsed correctly"


def test_query_on_physics():
    """Test query execution on real physics data."""
    from ontic.applied.intent import Aggregator, FieldQuery, Predicate

    # Create real physics-like field: Gaussian temperature
    x = np.linspace(-1, 1, 64)
    y = np.linspace(-1, 1, 64)
    X, Y = np.meshgrid(x, y)
    temperature = 300 + 50 * np.exp(-(X**2 + Y**2))

    # Query: max temperature
    query = FieldQuery("temperature").aggregate(Aggregator.max())
    result = query.execute(temperature)

    expected_max = 350.0  # 300 + 50 at center
    assert (
        abs(result.value - expected_max) < 0.1
    ), f"Max: {result.value} != {expected_max}"

    # Query: mean of hot region (>340)
    query = (
        FieldQuery("temperature").where(Predicate.gt(340)).aggregate(Aggregator.mean())
    )
    result = query.execute(temperature)

    assert result.value > 340, f"Mean of hot region: {result.value}"

    return True, f"max={350.0:.1f}, hot_region_mean={result.value:.1f}"


def test_constraint_solver():
    """Test constraint satisfaction on fields."""
    from ontic.applied.intent import (BoundConstraint, ConstraintSet,
                                  ConstraintSolver)

    # Create constraint set
    cs = ConstraintSet()
    cs.add(BoundConstraint(lower=-1.0, upper=1.0, name="bounds"))

    # Create solver with constraints
    solver = ConstraintSolver(constraints=cs)

    # Create field with out-of-bound values
    field = np.array([0.5, -0.5, 1.5, -1.5, 0.0])

    # Project onto constraint set
    result = solver.project(field)

    # All values should be in bounds
    assert np.all(result >= -1.0), "Lower bound violated"
    assert np.all(result <= 1.0), "Upper bound violated"

    return True, f"projected: {result}"


def test_intent_engine_integration():
    """Test full intent engine with physics field."""
    from ontic.applied.intent import (Aggregator, FieldQuery, IntentEngine,
                                  ResultStatus)

    # Create physics field
    x = np.linspace(0, 1, 32)
    y = np.linspace(0, 1, 32)
    X, Y = np.meshgrid(x, y)
    velocity = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)

    # Create engine and register field
    engine = IntentEngine()
    engine.register_field("velocity", velocity)

    # Execute structured query (more reliable)
    query = FieldQuery("velocity").aggregate(Aggregator.max())
    result = engine.execute(query)

    expected = np.max(velocity)
    assert result.status == ResultStatus.SUCCESS, f"Status: {result.status}"
    assert (
        abs(result.value - expected) < 1e-10
    ), f"Expected {expected}, got {result.value}"

    return True, f"max(velocity) -> {result.value:.4f}"


def run_audit():
    print()
    print("=" * 66)
    print("           LAYER 8 AUDIT: Intent / FDL")
    print("=" * 66)
    print()

    results = []

    tests = [
        ("Intent Parser", test_intent_parser),
        ("Query on Physics", test_query_on_physics),
        ("Constraint Solver", test_constraint_solver),
        ("Engine Integration", test_intent_engine_integration),
    ]

    for name, test_fn in tests:
        print(f"Test: {name}...")
        try:
            passed, detail = test_fn()
            results.append((name, passed, detail))
            print(f"  {'PASS' if passed else 'FAIL'} | {detail}")
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"  FAIL | {e}")

    print()
    all_passed = all(r[1] for r in results)
    n_passed = sum(1 for r in results if r[1])

    print("=" * 66)
    if all_passed:
        print("  ALL TESTS PASSED")
        print("  Intent steering validated on real physics data")
    else:
        print(f"  {n_passed}/{len(results)} TESTS PASSED")
    print("=" * 66)

    return all_passed


if __name__ == "__main__":
    success = run_audit()
    sys.exit(0 if success else 1)
