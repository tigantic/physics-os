#!/usr/bin/env python3
"""
Proof: Tensor Decompositions
============================

Executable mathematical proofs for SVD and QR decompositions.

Proofs:
    1.1 - SVD Truncation Optimality (Eckart-Young-Mirsky)
    1.2 - SVD Orthogonality
    1.3 - QR Reconstruction
    1.4 - QR Orthogonality

References:
    [1] Eckart & Young, "The approximation of one matrix by another 
        of lower rank", Psychometrika 1, 211-218 (1936)
    [2] Golub & Van Loan, "Matrix Computations", 4th ed. (2013)

Usage:
    python proofs/proof_decompositions.py
"""

import sys
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tensornet.core.decompositions import svd_truncated, qr_positive


# Constitutional tolerances (Article I, Section 1.2)
MACHINE_PRECISION = 1e-14
NUMERICAL_STABILITY = 1e-10


@dataclass
class ProofResult:
    """Container for proof result."""
    id: str
    name: str
    category: str
    claim: str
    status: str  # PASS or FAIL
    measurements: List[Dict[str, Any]]
    tolerance: float
    max_error: float


def proof_1_1_svd_truncation_optimality() -> ProofResult:
    """
    Proof 1.1: SVD Truncation Optimality
    
    Claim: svd_truncated(A, k) produces the optimal rank-k approximation
           under the Frobenius norm (Eckart-Young-Mirsky theorem).
    
    Method: Compare ||A - U @ diag(S) @ Vh||_F to torch.linalg.svd 
            truncated to same rank.
    """
    torch.manual_seed(42)
    
    # Create test matrix
    A = torch.randn(100, 80, dtype=torch.float64)
    
    measurements = []
    max_diff = 0.0
    
    for k in [5, 10, 20, 40]:
        # Our implementation
        U, S, Vh = svd_truncated(A, chi_max=k)
        A_approx = U @ torch.diag(S) @ Vh
        our_error = torch.linalg.norm(A - A_approx, ord='fro').item()
        
        # Reference: torch.linalg.svd
        U_ref, S_ref, Vh_ref = torch.linalg.svd(A, full_matrices=False)
        A_ref = U_ref[:, :k] @ torch.diag(S_ref[:k]) @ Vh_ref[:k, :]
        ref_error = torch.linalg.norm(A - A_ref, ord='fro').item()
        
        diff = abs(our_error - ref_error)
        max_diff = max(max_diff, diff)
        
        measurements.append({
            "k": k,
            "our_error": our_error,
            "ref_error": ref_error,
            "diff": diff
        })
    
    return ProofResult(
        id="1.1",
        name="SVD Truncation Optimality",
        category="Tensor Decompositions",
        claim="svd_truncated(A, k) produces optimal rank-k approximation (Eckart-Young-Mirsky)",
        status="PASS" if max_diff < NUMERICAL_STABILITY else "FAIL",
        measurements=measurements,
        tolerance=NUMERICAL_STABILITY,
        max_error=max_diff
    )


def proof_1_2_svd_orthogonality() -> ProofResult:
    """
    Proof 1.2: SVD Orthogonality
    
    Claim: U and V from SVD are orthogonal matrices.
    """
    torch.manual_seed(42)
    
    A = torch.randn(50, 50, dtype=torch.float64)
    U, S, Vh = svd_truncated(A)
    
    # Check U^T @ U = I
    U_ortho_error = torch.linalg.norm(U.T @ U - torch.eye(U.shape[1], dtype=torch.float64)).item()
    
    # Check V @ V^T = I (Vh^T @ Vh)
    V_ortho_error = torch.linalg.norm(Vh @ Vh.T - torch.eye(Vh.shape[0], dtype=torch.float64)).item()
    
    max_error = max(U_ortho_error, V_ortho_error)
    
    return ProofResult(
        id="1.2",
        name="SVD Orthogonality",
        category="Tensor Decompositions",
        claim="U and V from SVD are orthogonal matrices",
        status="PASS" if max_error < NUMERICAL_STABILITY else "FAIL",
        measurements=[
            {"property": "||U^T @ U - I||_F", "value": U_ortho_error},
            {"property": "||V @ V^T - I||_F", "value": V_ortho_error}
        ],
        tolerance=NUMERICAL_STABILITY,
        max_error=max_error
    )


def proof_1_3_qr_reconstruction() -> ProofResult:
    """
    Proof 1.3: QR Reconstruction
    
    Claim: QR decomposition satisfies A = Q @ R.
    """
    torch.manual_seed(42)
    
    A = torch.randn(50, 30, dtype=torch.float64)
    Q, R = qr_positive(A)
    
    reconstruction_error = torch.linalg.norm(A - Q @ R, ord='fro').item()
    
    return ProofResult(
        id="1.3",
        name="QR Reconstruction",
        category="Tensor Decompositions",
        claim="QR decomposition satisfies A = Q @ R",
        status="PASS" if reconstruction_error < NUMERICAL_STABILITY else "FAIL",
        measurements=[
            {"property": "||A - Q @ R||_F", "value": reconstruction_error}
        ],
        tolerance=NUMERICAL_STABILITY,
        max_error=reconstruction_error
    )


def proof_1_4_qr_orthogonality() -> ProofResult:
    """
    Proof 1.4: QR Orthogonality
    
    Claim: Q from QR is orthogonal.
    """
    torch.manual_seed(42)
    
    A = torch.randn(50, 30, dtype=torch.float64)
    Q, R = qr_positive(A)
    
    ortho_error = torch.linalg.norm(Q.T @ Q - torch.eye(Q.shape[1], dtype=torch.float64)).item()
    
    return ProofResult(
        id="1.4",
        name="QR Orthogonality",
        category="Tensor Decompositions",
        claim="Q from QR is orthogonal",
        status="PASS" if ortho_error < NUMERICAL_STABILITY else "FAIL",
        measurements=[
            {"property": "||Q^T @ Q - I||_F", "value": ortho_error}
        ],
        tolerance=NUMERICAL_STABILITY,
        max_error=ortho_error
    )


def run_all_proofs() -> Dict[str, Any]:
    """Execute all decomposition proofs and return results."""
    proofs = [
        proof_1_1_svd_truncation_optimality,
        proof_1_2_svd_orthogonality,
        proof_1_3_qr_reconstruction,
        proof_1_4_qr_orthogonality,
    ]
    
    results = []
    all_passed = True
    
    print("=" * 60)
    print("PROOF SUITE: Tensor Decompositions")
    print("=" * 60)
    print()
    
    for proof_func in proofs:
        result = proof_func()
        results.append(asdict(result))
        
        status_icon = "[PASS]" if result.status == "PASS" else "[FAIL]"
        print(f"{status_icon} Proof {result.id}: {result.name}")
        print(f"   Claim: {result.claim}")
        print(f"   Max Error: {result.max_error:.2e} (tol: {result.tolerance:.0e})")
        print(f"   Status: {result.status}")
        print()
        
        if result.status != "PASS":
            all_passed = False
    
    print("=" * 60)
    passed_count = sum(1 for r in results if r["status"] == "PASS")
    print(f"SUMMARY: {passed_count}/{len(results)} proofs passed")
    print("=" * 60)
    
    return {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "category": "Tensor Decompositions",
            "torch_version": torch.__version__,
            "seed": 42
        },
        "proofs": results,
        "summary": {
            "total": len(results),
            "passed": passed_count,
            "failed": len(results) - passed_count,
            "all_passed": all_passed
        }
    }


if __name__ == "__main__":
    results = run_all_proofs()
    
    # Save JSON artifact
    output_path = Path(__file__).parent / "proof_decompositions_result.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nArtifact saved: {output_path}")
    
    # Exit with error code if any proof failed
    sys.exit(0 if results["summary"]["all_passed"] else 1)
