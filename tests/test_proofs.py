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
        
        if result.returncode == 0 and "PROOF PASSED" in output:
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


def test_proof_17_tebd():
    """Test TEBD unitarity proof."""
    _run_single_proof("proof_17_tebd_unitarity.py")


def test_proof_18_idmrg():
    """Test iDMRG thermodynamic proof."""
    _run_single_proof("proof_18_idmrg_thermodynamic.py")


def test_proof_19_tdvp():
    """Test TDVP ground state proof."""
    _run_single_proof("proof_19_tdvp_ground_state.py")


def test_proof_20_accuracy():
    """Test ground state accuracy proof."""
    _run_single_proof("proof_20_ground_state_accuracy.py")


def test_proof_21_fermionic():
    """Test fermionic MPS proof."""
    _run_single_proof("proof_21_fermionic_mps.py")


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
    assert "PROOF PASSED" in output, f"Proof did not pass: {output[-1000:]}"


if __name__ == "__main__":
    test_all_proofs()
