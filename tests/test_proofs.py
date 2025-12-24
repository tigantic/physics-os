"""
PyTenNet Proof Tests
====================

Lightweight tests that validate proof infrastructure exists.
Heavy proof execution is done separately in CI via run_all_proofs.py.
"""

import sys
from pathlib import Path
import ast


def test_proof_files_exist():
    """Verify all expected proof files exist."""
    repo_root = Path(__file__).parent.parent
    proofs_dir = repo_root / "proofs"
    
    assert proofs_dir.exists(), "proofs directory not found"
    
    # Check key proof files exist
    expected_files = [
        "proof_mps.py",
        "proof_decompositions.py",
        "proof_algorithms.py",
        "proof_cfd_conservation.py",
    ]
    
    for fname in expected_files:
        fpath = proofs_dir / fname
        assert fpath.exists(), f"Missing proof file: {fname}"


def test_proof_files_valid_python():
    """Verify proof files are valid Python syntax."""
    repo_root = Path(__file__).parent.parent
    proofs_dir = repo_root / "proofs"
    
    for proof_file in proofs_dir.glob("proof_*.py"):
        try:
            ast.parse(proof_file.read_text(encoding='utf-8'))
        except SyntaxError as e:
            raise AssertionError(f"Syntax error in {proof_file.name}: {e}")


def test_run_all_proofs_exists():
    """Verify the main proof runner exists."""
    repo_root = Path(__file__).parent.parent
    runner = repo_root / "proofs" / "run_all_proofs.py"
    assert runner.exists(), "run_all_proofs.py not found"


if __name__ == "__main__":
    test_proof_files_exist()
    test_proof_files_valid_python()
    test_run_all_proofs_exists()
    print("All proof infrastructure tests passed!")
