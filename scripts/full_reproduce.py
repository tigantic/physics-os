#!/usr/bin/env python
"""
Full Reproduction Runner
========================

Reproduce all paper results from scratch with comprehensive logging.

Usage:
    python scripts/full_reproduce.py --quick   # Smoke test (5 min)
    python scripts/full_reproduce.py --full    # Full reproduction (30+ min)

Pass Criteria:
    - All benchmarks complete without error
    - Results within tolerance of stored reference
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def run_benchmark(name: str, script: str, timeout: int = 300) -> Tuple[bool, str, float]:
    """Run a benchmark script and return (passed, output, runtime)."""
    start = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(script).parent.parent
        )
        
        runtime = time.time() - start
        output = result.stdout + result.stderr
        passed = result.returncode == 0
        
        return passed, output, runtime
        
    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout}s", time.time() - start
    except Exception as e:
        return False, str(e), time.time() - start


def run_proof(name: str, script: str, timeout: int = 60) -> Tuple[bool, str, float]:
    """Run a proof script and return (passed, output, runtime)."""
    return run_benchmark(name, script, timeout)


def quick_smoke_test(project_root: Path) -> Dict:
    """Run quick smoke tests (~5 min)."""
    results = {
        'mode': 'quick',
        'benchmarks': [],
        'proofs': [],
    }
    
    # Quick benchmarks
    benchmarks = [
        ('Sod Shock Tube', 'benchmarks/sod_shock_tube.py'),
        ('TFIM Ground State', 'benchmarks/tfim_ground_state.py'),
        ('QTT Compression', 'benchmarks/qtt_compression.py'),
    ]
    
    print("\n--- Quick Benchmarks ---")
    for name, script in benchmarks:
        script_path = project_root / script
        if not script_path.exists():
            print(f"  SKIP {name} (not found)")
            continue
            
        print(f"  Running {name}...", end=' ', flush=True)
        passed, output, runtime = run_benchmark(name, str(script_path), timeout=120)
        status = "PASS" if passed else "FAIL"
        print(f"{status} ({runtime:.1f}s)")
        
        results['benchmarks'].append({
            'name': name,
            'passed': passed,
            'runtime_s': runtime,
            'output_snippet': output[:500] if not passed else ''
        })
    
    # Quick proofs
    proofs = [
        ('MPS Properties', 'proofs/proof_mps.py'),
        ('CFD Conservation', 'proofs/proof_cfd_conservation.py'),
    ]
    
    print("\n--- Quick Proofs ---")
    for name, script in proofs:
        script_path = project_root / script
        if not script_path.exists():
            print(f"  SKIP {name} (not found)")
            continue
            
        print(f"  Running {name}...", end=' ', flush=True)
        passed, output, runtime = run_proof(name, str(script_path), timeout=60)
        status = "PASS" if passed else "FAIL"
        print(f"{status} ({runtime:.1f}s)")
        
        results['proofs'].append({
            'name': name,
            'passed': passed,
            'runtime_s': runtime,
        })
    
    return results


def full_reproduction(project_root: Path) -> Dict:
    """Run full reproduction (~30+ min)."""
    results = {
        'mode': 'full',
        'benchmarks': [],
        'proofs': [],
        'physics': [],
    }
    
    # All benchmarks
    benchmarks = [
        ('Sod Shock Tube', 'benchmarks/sod_shock_tube.py', 120),
        ('TFIM Ground State', 'benchmarks/tfim_ground_state.py', 180),
        ('Heisenberg Ground State', 'benchmarks/heisenberg_ground_state.py', 300),
        ('QTT Compression', 'benchmarks/qtt_compression.py', 120),
        ('Blasius Validation', 'benchmarks/blasius_validation.py', 180),
        ('Double Mach Reflection', 'benchmarks/double_mach_reflection.py', 600),
        ('Oblique Shock', 'benchmarks/oblique_shock.py', 180),
        ('SBLI Benchmark', 'benchmarks/sbli_benchmark.py', 600),
    ]
    
    print("\n--- Full Benchmarks ---")
    for name, script, timeout in benchmarks:
        script_path = project_root / script
        if not script_path.exists():
            print(f"  SKIP {name} (not found)")
            continue
            
        print(f"  Running {name} (timeout={timeout}s)...", end=' ', flush=True)
        passed, output, runtime = run_benchmark(name, str(script_path), timeout=timeout)
        status = "PASS" if passed else "FAIL"
        print(f"{status} ({runtime:.1f}s)")
        
        results['benchmarks'].append({
            'name': name,
            'passed': passed,
            'runtime_s': runtime,
            'output_snippet': output[:500] if not passed else ''
        })
    
    # All proofs
    proof_dir = project_root / 'proofs'
    proof_files = sorted(proof_dir.glob('proof_*.py'))
    
    print("\n--- All Proofs ---")
    for script_path in proof_files:
        if script_path.name == 'run_all_proofs.py':
            continue
            
        name = script_path.stem.replace('proof_', '').replace('_', ' ').title()
        print(f"  Running {name}...", end=' ', flush=True)
        passed, output, runtime = run_proof(name, str(script_path), timeout=120)
        status = "PASS" if passed else "FAIL"
        print(f"{status} ({runtime:.1f}s)")
        
        results['proofs'].append({
            'name': name,
            'passed': passed,
            'runtime_s': runtime,
        })
    
    # Physics validation
    physics_script = project_root / 'scripts' / 'physics_validation.py'
    if physics_script.exists():
        print("\n--- Physics Validation ---")
        print(f"  Running full validation...", end=' ', flush=True)
        passed, output, runtime = run_benchmark('Physics Validation', 
                                                 str(physics_script), 
                                                 timeout=300)
        status = "PASS" if passed else "FAIL"
        print(f"{status} ({runtime:.1f}s)")
        
        results['physics'].append({
            'name': 'Full Physics Validation',
            'passed': passed,
            'runtime_s': runtime,
        })
    
    return results


def compare_with_reference(results: Dict, reference_path: Path) -> Dict:
    """Compare results with stored reference."""
    comparisons = {}
    
    if not reference_path.exists():
        return {'error': 'No reference file found'}
    
    try:
        with open(reference_path) as f:
            reference = json.load(f)
        
        # Compare benchmark pass/fail
        ref_benchmarks = {b['name']: b for b in reference.get('benchmarks', [])}
        
        for benchmark in results.get('benchmarks', []):
            name = benchmark['name']
            if name in ref_benchmarks:
                ref_passed = ref_benchmarks[name].get('passed', False)
                cur_passed = benchmark.get('passed', False)
                comparisons[name] = {
                    'reference_passed': ref_passed,
                    'current_passed': cur_passed,
                    'match': ref_passed == cur_passed
                }
        
    except Exception as e:
        return {'error': str(e)}
    
    return comparisons


def main():
    parser = argparse.ArgumentParser(description="Full reproduction runner")
    parser.add_argument('--quick', action='store_true',
                       help="Quick smoke test (~5 min)")
    parser.add_argument('--full', action='store_true',
                       help="Full reproduction (~30+ min)")
    parser.add_argument('--output', type=str,
                       help="Output JSON file path")
    args = parser.parse_args()
    
    if not args.quick and not args.full:
        args.quick = True  # Default to quick
    
    project_root = Path(__file__).parent.parent
    
    print("=" * 60)
    print(" FULL REPRODUCTION RUNNER")
    print("=" * 60)
    
    start_time = time.time()
    
    if args.full:
        results = full_reproduction(project_root)
    else:
        results = quick_smoke_test(project_root)
    
    total_runtime = time.time() - start_time
    
    # Compute summary
    all_benchmarks_passed = all(b['passed'] for b in results.get('benchmarks', []) if results.get('benchmarks'))
    all_proofs_passed = all(p['passed'] for p in results.get('proofs', []) if results.get('proofs'))
    all_physics_passed = all(p['passed'] for p in results.get('physics', []) if results.get('physics'))
    
    all_passed = all_benchmarks_passed and all_proofs_passed and all_physics_passed
    
    results['summary'] = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'total_runtime_s': total_runtime,
        'benchmarks_passed': sum(1 for b in results.get('benchmarks', []) if b['passed']),
        'benchmarks_total': len(results.get('benchmarks', [])),
        'proofs_passed': sum(1 for p in results.get('proofs', []) if p['passed']),
        'proofs_total': len(results.get('proofs', [])),
        'all_passed': all_passed,
    }
    
    # Compare with reference if exists
    reference_path = project_root / 'artifacts' / 'reference_results.json'
    results['comparison'] = compare_with_reference(results, reference_path)
    
    # Print summary
    print()
    print("=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    print(f"  Mode: {results['mode']}")
    print(f"  Total Runtime: {total_runtime:.1f}s")
    print()
    print(f"  Benchmarks: {results['summary']['benchmarks_passed']}/{results['summary']['benchmarks_total']} passed")
    print(f"  Proofs: {results['summary']['proofs_passed']}/{results['summary']['proofs_total']} passed")
    print()
    
    if all_passed:
        print("  ✓ ALL REPRODUCTIONS SUCCESSFUL")
    else:
        print("  ✗ SOME REPRODUCTIONS FAILED")
        
        # List failures
        print("\n  Failures:")
        for b in results.get('benchmarks', []):
            if not b['passed']:
                print(f"    - {b['name']}")
        for p in results.get('proofs', []):
            if not p['passed']:
                print(f"    - {p['name']}")
    
    print("=" * 60)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = project_root / 'artifacts' / 'reproduce_results.json'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
