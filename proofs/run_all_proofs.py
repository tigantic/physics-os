#!/usr/bin/env python
"""
G) Run All Proofs Script
========================

Executes all proof files and generates consolidated report.

Usage:
    python proofs/run_all_proofs.py

Pass Criteria:
    - All proofs pass
    - Proof report regenerated
    - Proof JSON artifact emitted
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


# Proof files to run (in order)
PROOF_FILES = [
    'proof_mps.py',
    'proof_decompositions.py',
    'proof_algorithms.py',
    'proof_cfd_conservation.py',
    'proof_21_weno_order.py',
    'proof_21_weno_shock.py',
    'proof_21_tdvp_euler_conservation.py',
    'proof_21_tdvp_euler_sod.py',
    'proof_phase_22.py',
    'proof_phase_23.py',
    'proof_phase_24.py',
]


def run_proof(proof_path: Path) -> Dict[str, Any]:
    """Run a single proof file and capture results."""
    import os
    start_time = time.time()
    
    result = {
        'file': proof_path.name,
        'passed': False,
        'duration_sec': 0,
        'output': '',
        'error': None,
    }
    
    try:
        # Set UTF-8 encoding to handle unicode characters on Windows
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        proc = subprocess.run(
            [sys.executable, str(proof_path)],
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout per proof
            cwd=proof_path.parent.parent,  # Run from project root
            env=env,
        )
        
        result['output'] = proc.stdout + proc.stderr
        result['return_code'] = proc.returncode
        result['passed'] = proc.returncode == 0
        
    except subprocess.TimeoutExpired:
        result['error'] = 'Timeout after 300 seconds'
    except Exception as e:
        result['error'] = str(e)
    
    result['duration_sec'] = time.time() - start_time
    return result


def load_result_json(proof_path: Path) -> Dict[str, Any]:
    """Try to load the result JSON for a proof."""
    # Try common naming patterns
    patterns = [
        proof_path.stem + '_result.json',
        proof_path.stem.replace('proof_', 'proof_') + '_result.json',
        proof_path.stem.replace('proof_', '') + '_results.json',
    ]
    
    # Also check for numbered patterns
    if 'phase_' in proof_path.stem:
        num = proof_path.stem.split('_')[-1]
        patterns.append(f'proof_{num}_results.json')
    
    for pattern in patterns:
        result_path = proof_path.parent / pattern
        if result_path.exists():
            try:
                with open(result_path) as f:
                    return json.load(f)
            except Exception:
                pass
    
    return {}


def generate_report(results: List[Dict[str, Any]], proofs_dir: Path) -> Dict[str, Any]:
    """Generate consolidated proof report."""
    total = len(results)
    passed = sum(1 for r in results if r['passed'])
    failed = total - passed
    
    report = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'summary': {
            'total': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / total if total > 0 else 0,
        },
        'proofs': [],
    }
    
    for result in results:
        proof_entry = {
            'file': result['file'],
            'passed': result['passed'],
            'duration_sec': round(result['duration_sec'], 2),
        }
        
        if result['error']:
            proof_entry['error'] = result['error']
        
        # Try to load detailed results
        proof_path = proofs_dir / result['file']
        detailed = load_result_json(proof_path)
        if detailed:
            proof_entry['details'] = detailed.get('proofs', detailed.get('summary', {}))
        
        report['proofs'].append(proof_entry)
    
    return report


def main():
    print("=" * 70)
    print(" PROOF SUITE RUNNER")
    print("=" * 70)
    print()
    
    proofs_dir = Path(__file__).parent
    results = []
    
    # Run each proof
    for proof_file in PROOF_FILES:
        proof_path = proofs_dir / proof_file
        
        if not proof_path.exists():
            print(f"⚠️  SKIP: {proof_file} (not found)")
            results.append({
                'file': proof_file,
                'passed': False,
                'duration_sec': 0,
                'error': 'File not found',
                'output': '',
            })
            continue
        
        print(f"Running: {proof_file}...", end=' ', flush=True)
        result = run_proof(proof_path)
        results.append(result)
        
        if result['passed']:
            print(f"✓ PASSED ({result['duration_sec']:.1f}s)")
        else:
            print(f"✗ FAILED ({result['duration_sec']:.1f}s)")
            if result['error']:
                print(f"    Error: {result['error']}")
    
    # Generate report
    print()
    print("-" * 70)
    report = generate_report(results, proofs_dir)
    
    # Save report
    report_path = proofs_dir / 'proof_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    summary = report['summary']
    print()
    print(f"SUMMARY: {summary['passed']}/{summary['total']} proofs passed")
    print(f"Pass rate: {summary['pass_rate']*100:.1f}%")
    print(f"Report: {report_path}")
    
    # List failures
    if summary['failed'] > 0:
        print()
        print("FAILURES:")
        for result in results:
            if not result['passed']:
                print(f"  - {result['file']}")
                if result['error']:
                    print(f"      {result['error']}")
    
    print()
    print("=" * 70)
    if summary['failed'] == 0:
        print(" ✓ ALL PROOFS PASSED")
    else:
        print(f" ✗ {summary['failed']} PROOF(S) FAILED")
    print("=" * 70)
    
    return 0 if summary['failed'] == 0 else 1


if __name__ == '__main__':
    exit(main())
