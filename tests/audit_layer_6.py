"""
Layer 6 Audit: Benchmark Suite Validation
==========================================

Validates which benchmarks in the benchmarks/ directory actually run.
"""

import os
import subprocess
from pathlib import Path

# Get project root for benchmark paths
PROJECT_ROOT = Path(__file__).parent.parent

BENCHMARKS = [
    ("Sod Shock Tube", "sod_shock_tube_inline"),
    ("QTT Compression", "qtt_compression"),
    ("Blasius Validation", "blasius_validation"),
]


def test_sod_shock_tube():
    """Inline test of Sod shock tube."""
    import torch

    from tensornet.cfd import Euler1D, exact_riemann, sod_shock_tube_ic

    N = 200
    solver = Euler1D(N=N, x_min=0.0, x_max=1.0, gamma=1.4, cfl=0.5)
    ic = sod_shock_tube_ic(N, x_min=0.0, x_max=1.0)
    solver.set_initial_condition(ic)

    t_final = 0.2
    snapshots = solver.solve(t_final)

    final_state = solver.state
    x = solver.x_cell
    rho_exact, u_exact, p_exact = exact_riemann(
        rho_L=1.0,
        u_L=0.0,
        p_L=1.0,
        rho_R=0.125,
        u_R=0.0,
        p_R=0.1,
        gamma=1.4,
        x=x,
        t=t_final,
        x0=0.5,
    )

    rho_err = torch.abs(final_state.rho - rho_exact).mean().item()
    return rho_err < 0.1, f"L1(rho)={rho_err:.4e}"


def test_qtt_compression():
    """Run QTT compression benchmark inline (no plots)."""
    try:
        # Import and run directly instead of subprocess
        from benchmarks.qtt_compression import benchmark_uniform_flow

        # Test: Uniform flow
        r1 = benchmark_uniform_flow(Nx=64, Ny=32, verbose=False)

        passed = r1["pass"]
        return passed, f"uniform_flow: error={r1['rho_error']:.2e}"
    except Exception as e:
        return False, str(e)


def test_blasius():
    """Run Blasius validation."""
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = PROJECT_ROOT
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [sys.executable, "experiments/benchmarks/benchmarks/blasius_validation.py"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=PROJECT_ROOT,
            env=env,
            encoding="utf-8",
            errors="replace",
        )
        passed = "ALL VALIDATIONS PASSED" in result.stdout
        return passed, "5/5 validations passed" if passed else "Partial pass"
    except Exception as e:
        return False, str(e)


def run_audit():
    print()
    print("=" * 66)
    print("           LAYER 6 AUDIT: Benchmark Suite")
    print("=" * 66)
    print()

    results = []

    # Test 1: Sod Shock Tube
    print("Test 1: Sod Shock Tube (1D Euler)...")
    try:
        passed, detail = test_sod_shock_tube()
        results.append(("Sod Shock Tube", passed, detail))
        print(f"  {'PASS' if passed else 'FAIL'} | {detail}")
    except Exception as e:
        results.append(("Sod Shock Tube", False, str(e)))
        print(f"  FAIL | {e}")

    # Test 2: QTT Compression
    print("Test 2: QTT Compression...")
    passed, detail = test_qtt_compression()
    results.append(("QTT Compression", passed, detail))
    print(f"  {'PASS' if passed else 'FAIL'} | {detail}")

    # Test 3: Blasius Validation
    print("Test 3: Blasius Viscous Terms...")
    passed, detail = test_blasius()
    results.append(("Blasius", passed, detail))
    print(f"  {'PASS' if passed else 'FAIL'} | {detail}")

    print()
    all_passed = all(r[1] for r in results)
    n_passed = sum(1 for r in results if r[1])

    print("=" * 66)
    if all_passed:
        print("  ALL BENCHMARKS PASSED")
    else:
        print(f"  {n_passed}/{len(results)} BENCHMARKS PASSED")
    print("=" * 66)

    return all_passed


if __name__ == "__main__":
    success = run_audit()
    sys.exit(0 if success else 1)
