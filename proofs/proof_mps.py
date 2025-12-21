#!/usr/bin/env python3
"""
Proof: MPS Operations
=====================

Executable mathematical proofs for Matrix Product State operations.

Proofs:
    2.1 - MPS Round-Trip Fidelity
    2.2 - GHZ Entanglement Entropy
    2.3 - Product State Zero Entropy
    2.4 - Norm Preservation Under Canonicalization
    2.5 - Left-Canonical Orthogonality

References:
    [1] Schollwöck, "The density-matrix renormalization group in the age 
        of matrix product states", Ann. Phys. 326, 96-192 (2011)
    [2] Vidal, "Efficient classical simulation of slightly entangled 
        quantum computations", Phys. Rev. Lett. 91, 147902 (2003)

Usage:
    python proofs/proof_mps.py
"""

import sys
import json
import math
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tensornet.core.mps import MPS
from tensornet.core.states import ghz_mps, product_mps


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
    status: str
    measurements: List[Dict[str, Any]]
    tolerance: float
    max_error: float


def proof_2_1_mps_round_trip() -> ProofResult:
    """
    Proof 2.1: MPS Round-Trip Fidelity
    
    Claim: tensor -> MPS -> tensor preserves information within bond dimension.
    """
    torch.manual_seed(42)
    
    # Create small tensor that fits in MPS exactly
    L, d, chi = 5, 2, 4
    shape = [d] * L
    T_original = torch.randn(*shape, dtype=torch.float64)
    
    # Convert to MPS
    mps = MPS.from_tensor(T_original, chi_max=chi)
    
    # Convert back
    T_reconstructed = mps.to_tensor()
    
    error = torch.linalg.norm(T_original - T_reconstructed).item()
    
    return ProofResult(
        id="2.1",
        name="MPS Round-Trip Fidelity",
        category="MPS Operations",
        claim="tensor -> MPS -> tensor preserves information within bond dimension",
        status="PASS" if error < NUMERICAL_STABILITY else "FAIL",
        measurements=[
            {"property": "input_shape", "value": shape},
            {"property": "bond_dimension", "value": chi},
            {"property": "||T - T_reconstructed||_F", "value": error}
        ],
        tolerance=NUMERICAL_STABILITY,
        max_error=error
    )


def proof_2_2_ghz_entropy() -> ProofResult:
    """
    Proof 2.2: GHZ Entanglement Entropy
    
    Claim: GHZ state |00...0⟩ + |11...1⟩ has von Neumann entropy S = ln(2)
           at every bipartition.
    """
    torch.manual_seed(42)
    
    L = 6
    ln2 = math.log(2)
    
    # Create GHZ state
    mps = ghz_mps(L=L)
    
    # Compute entropy at each bond using entropy(bond) method
    measurements = []
    max_error = 0.0
    
    for i in range(L - 1):
        S = mps.entropy(i).item()
        error = abs(S - ln2)
        max_error = max(max_error, error)
        measurements.append({
            "bond": i,
            "computed_S": S,
            "theoretical_S": ln2,
            "error": error
        })
    
    return ProofResult(
        id="2.2",
        name="GHZ Entanglement Entropy",
        category="MPS Operations",
        claim="GHZ state has S = ln(2) = 0.693... at every bipartition",
        status="PASS" if max_error < NUMERICAL_STABILITY else "FAIL",
        measurements=measurements,
        tolerance=NUMERICAL_STABILITY,
        max_error=max_error
    )


def proof_2_3_product_state_entropy() -> ProofResult:
    """
    Proof 2.3: Product State Zero Entropy
    
    Claim: Product state has zero entanglement entropy at all bonds.
    """
    torch.manual_seed(42)
    
    L = 6
    
    # Create product state |+⟩^L using list of |+⟩ states
    plus_state = torch.tensor([1.0, 1.0], dtype=torch.float64) / math.sqrt(2)
    states = [plus_state.clone() for _ in range(L)]
    mps = product_mps(states)
    
    # Compute entropy at each bond using entropy(bond) method
    measurements = []
    max_error = 0.0
    
    for i in range(L - 1):
        S = mps.entropy(i).item()
        max_error = max(max_error, abs(S))
        measurements.append({
            "bond": i,
            "computed_S": S,
            "expected": 0.0
        })
    
    return ProofResult(
        id="2.3",
        name="Product State Zero Entropy",
        category="MPS Operations",
        claim="Product state has zero entanglement entropy",
        status="PASS" if max_error < MACHINE_PRECISION else "FAIL",
        measurements=measurements,
        tolerance=MACHINE_PRECISION,
        max_error=max_error
    )


def proof_2_4_norm_preservation() -> ProofResult:
    """
    Proof 2.4: Norm Preservation Under Canonicalization
    
    Claim: Canonicalization preserves MPS norm.
    """
    torch.manual_seed(42)
    
    L, d, chi = 8, 2, 8
    mps = MPS.random(L=L, d=d, chi=chi)
    
    norm_before = mps.norm().item()
    mps.canonicalize_to_(L // 2)
    norm_after = mps.norm().item()
    
    diff = abs(norm_before - norm_after)
    
    return ProofResult(
        id="2.4",
        name="Norm Preservation Under Canonicalization",
        category="MPS Operations",
        claim="Canonicalization preserves MPS norm",
        status="PASS" if diff < NUMERICAL_STABILITY else "FAIL",
        measurements=[
            {"property": "norm_before", "value": norm_before},
            {"property": "norm_after", "value": norm_after},
            {"property": "difference", "value": diff}
        ],
        tolerance=NUMERICAL_STABILITY,
        max_error=diff
    )


def proof_2_5_left_canonical_orthogonality() -> ProofResult:
    """
    Proof 2.5: Left-Canonical Orthogonality
    
    Claim: Left-canonical tensors satisfy A^† @ A = I when reshaped.
    """
    torch.manual_seed(42)
    
    L, d, chi = 7, 2, 6
    mps = MPS.random(L=L, d=d, chi=chi)
    
    # Left-canonicalize to last site
    mps.canonicalize_to_(L - 1)
    
    measurements = []
    max_error = 0.0
    
    for i in range(L - 1):  # Check all but last site
        A = mps.tensors[i]  # Shape: (chi_left, d, chi_right)
        chi_left, d_phys, chi_right = A.shape
        
        # Reshape to (chi_left * d, chi_right)
        A_matrix = A.reshape(chi_left * d_phys, chi_right)
        
        # Check A^† @ A = I
        product = A_matrix.T.conj() @ A_matrix
        identity = torch.eye(chi_right, dtype=torch.float64)
        error = torch.linalg.norm(product - identity).item()
        
        max_error = max(max_error, error)
        measurements.append({
            "site": i,
            "||A^†A - I||_F": error
        })
    
    return ProofResult(
        id="2.5",
        name="Left-Canonical Orthogonality",
        category="MPS Operations",
        claim="Left-canonical tensors satisfy A^† @ A = I",
        status="PASS" if max_error < NUMERICAL_STABILITY else "FAIL",
        measurements=measurements,
        tolerance=NUMERICAL_STABILITY,
        max_error=max_error
    )


def run_all_proofs() -> Dict[str, Any]:
    """Execute all MPS proofs and return results."""
    proofs = [
        proof_2_1_mps_round_trip,
        proof_2_2_ghz_entropy,
        proof_2_3_product_state_entropy,
        proof_2_4_norm_preservation,
        proof_2_5_left_canonical_orthogonality,
    ]
    
    results = []
    all_passed = True
    
    print("=" * 60)
    print("PROOF SUITE: MPS Operations")
    print("=" * 60)
    print()
    
    for proof_func in proofs:
        result = proof_func()
        results.append(asdict(result))
        
        status_icon = "✅" if result.status == "PASS" else "❌"
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
            "category": "MPS Operations",
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
    output_path = Path(__file__).parent / "proof_mps_result.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nArtifact saved: {output_path}")
    
    # Exit with error code if any proof failed
    sys.exit(0 if results["summary"]["all_passed"] else 1)
