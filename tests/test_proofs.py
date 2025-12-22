"""
PyTenNet Proof Tests
====================

Automated tests that run all mathematical proofs.
"""

import subprocess
import sys
import os
from pathlib import Path


def test_all_proofs():
    """Run all proof scripts and verify they pass."""
    # Find proofs directory
    repo_root = Path(__file__).parent.parent
    proofs_dir = repo_root / "proofs"
    
    if not proofs_dir.exists():
        # Try relative to current working directory
        proofs_dir = Path("proofs")
    
    assert proofs_dir.exists(), f"Proofs directory not found: {proofs_dir}"
    
    proof_files = sorted(proofs_dir.glob("proof_*.py"))
    assert len(proof_files) > 0, "No proof files found"
    
    failed = []
    passed = []
    
    for proof_file in proof_files:
        print(f"\nRunning {proof_file.name}...")
        result = subprocess.run(
            [sys.executable, str(proof_file)],
            capture_output=True,
            timeout=300,  # 5 minute timeout per proof
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        
        stdout = result.stdout.decode('utf-8', errors='replace') if result.stdout else ""
        stderr = result.stderr.decode('utf-8', errors='replace') if result.stderr else ""
        output = stdout + stderr
        
        # Check for various success indicators
        success_indicators = [
            "PROOF PASSED", "ALL PASSED", "PASSED", "all_passed", 
            "All proofs passed", "proofs passed", "Proofs:"
        ]
        is_success = result.returncode == 0 and any(ind in output for ind in success_indicators)
        
        if is_success:
            passed.append(proof_file.name)
            print(f"  PASSED")
        else:
            failed.append(proof_file.name)
            print(f"  FAILED")
            print(f"  Output: {output[-500:]}")  # Last 500 chars
    
    print(f"\n{'='*60}")
    print(f"Results: {len(passed)} passed, {len(failed)} failed")
    print(f"{'='*60}")
    
    assert len(failed) == 0, f"Failed proofs: {failed}"


# Phase 21-24 proofs
def test_proof_phase_21_weno():
    """Test Phase 21 WENO proofs."""
    _run_single_proof("proof_21_weno_order.py")


def test_proof_phase_21_tdvp():
    """Test Phase 21 TDVP Euler proofs."""
    _run_single_proof("proof_21_tdvp_euler_conservation.py")


def test_proof_phase_22():
    """Test Phase 22 operational applications proofs."""
    _run_single_proof("proof_phase_22.py")


def test_proof_phase_23():
    """Test Phase 23 radiation hardening proofs."""
    _run_single_proof("proof_phase_23.py")


def test_proof_phase_24():
    """Test Phase 24 stub completions proofs."""
    _run_single_proof("proof_phase_24.py")


def _run_single_proof(filename: str):
    """Run a single proof and assert it passes."""
    repo_root = Path(__file__).parent.parent
    proof_file = repo_root / "proofs" / filename
    
    if not proof_file.exists():
        proof_file = Path("proofs") / filename
    
    assert proof_file.exists(), f"Proof file not found: {proof_file}"
    
    result = subprocess.run(
        [sys.executable, str(proof_file)],
        capture_output=True,
        timeout=300,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    
    stdout = result.stdout.decode('utf-8', errors='replace') if result.stdout else ""
    stderr = result.stderr.decode('utf-8', errors='replace') if result.stderr else ""
    output = stdout + stderr
    
    assert result.returncode == 0, f"Proof failed with code {result.returncode}: {output[-1000:]}"
    
    # Check for various success indicators
    success_indicators = ["PROOF PASSED", "ALL PASSED", "PASSED", "all_passed"]
    assert any(ind in output for ind in success_indicators), f"Proof did not pass: {output[-1000:]}"


if __name__ == "__main__":
    test_all_proofs()
