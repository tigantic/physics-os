#!/usr/bin/env python
"""
K) Evidence Pack Builder
========================

Generates cryptographically signed evidence packs for validation cases.

Usage:
    python tools/scripts/build_evidence_pack.py --case sod --out artifacts/evidence/sod_pack/

Pass Criteria:
    - manifest.json with hashes of all files
    - manifest.sig (HMAC signature)
    - verify.py script that succeeds on fresh machine
"""

import argparse
import hashlib
import hmac
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# HMAC key for signing (in production, use env var or secure storage)
SIGNING_KEY = b"hypertensor-evidence-pack-2025"


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sign_manifest(manifest: Dict[str, Any]) -> str:
    """Create HMAC signature of manifest."""
    # Deterministic JSON serialization
    manifest_str = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    signature = hmac.new(SIGNING_KEY, manifest_str.encode(), hashlib.sha256)
    return signature.hexdigest()


def run_validation_case(case: str, output_dir: Path) -> Dict[str, Any]:
    """Run a validation case and collect outputs."""
    project_root = Path(__file__).parent.parent

    case_info = {
        "case": case,
        "run_at": datetime.utcnow().isoformat() + "Z",
        "files": {},
    }

    if case == "sod":
        # Run Sod shock tube validation
        import numpy as np
        import torch

        from tensornet.cfd.euler_1d import Euler1D, EulerState

        N = 200
        solver = Euler1D(N=N, x_min=0.0, x_max=1.0, gamma=1.4)

        x = torch.linspace(0, 1, N, dtype=torch.float64)
        rho = torch.where(x < 0.5, torch.ones_like(x), 0.125 * torch.ones_like(x))
        u = torch.zeros_like(x)
        p = torch.where(x < 0.5, torch.ones_like(x), 0.1 * torch.ones_like(x))

        rho_u = rho * u
        E = p / (1.4 - 1) + 0.5 * rho * u**2

        state = EulerState(rho=rho, rho_u=rho_u, E=E, gamma=1.4)
        solver.set_initial_condition(state)

        # Run to t=0.2
        t_final = 0.2
        dt = 0.0001
        n_steps = int(t_final / dt)

        for _ in range(n_steps):
            solver.step(dt)

        # Save results
        results = {
            "case": "sod",
            "N": N,
            "t_final": t_final,
            "x": x.numpy().tolist(),
            "rho": solver.state.rho.numpy().tolist(),
            "u": solver.state.u.numpy().tolist(),
            "p": solver.state.p.numpy().tolist(),
        }

        results_file = output_dir / "sod_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        case_info["files"]["sod_results.json"] = {
            "sha256": compute_file_hash(results_file),
            "size": results_file.stat().st_size,
        }

        case_info["metrics"] = {
            "n_cells": N,
            "t_final": t_final,
            "n_steps": n_steps,
        }

    elif case == "weno":
        # Run WENO order verification
        import numpy as np
        import torch

        from tensornet.cfd.weno import weno5_js_reconstruct

        errors = []
        Ns = [32, 64, 128, 256, 512]

        for N in Ns:
            x = torch.linspace(0, 2 * np.pi, N, dtype=torch.float64)
            dx = x[1] - x[0]
            u = torch.sin(x)
            u_exact = torch.cos(x)  # derivative

            uL, uR = weno5_js_reconstruct(u)
            # Approximate derivative
            u_approx = (uR[:-1] - uL[1:]) / dx

            # L2 error (skip boundaries)
            error = torch.sqrt(torch.mean((u_approx[2:-2] - u_exact[3:-3]) ** 2)).item()
            errors.append({"N": N, "error": error})

        results = {
            "case": "weno_order",
            "errors": errors,
        }

        # Compute order of convergence
        for i in range(1, len(errors)):
            ratio = np.log(errors[i - 1]["error"] / errors[i]["error"]) / np.log(2)
            errors[i]["order"] = round(ratio, 2)

        results_file = output_dir / "weno_order_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        case_info["files"]["weno_order_results.json"] = {
            "sha256": compute_file_hash(results_file),
            "size": results_file.stat().st_size,
        }

    else:
        raise ValueError(f"Unknown case: {case}")

    return case_info


def create_verify_script(output_dir: Path):
    """Create verification script."""
    verify_script = '''#!/usr/bin/env python
"""
Verification script for evidence pack.

Usage:
    python verify.py

This script verifies the integrity of the evidence pack by checking
file hashes against the manifest.
"""

import hashlib
import hmac
import json
import sys
from pathlib import Path


SIGNING_KEY = b'hypertensor-evidence-pack-2025'


def compute_file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_signature(manifest: dict, signature: str) -> bool:
    manifest_str = json.dumps(manifest, sort_keys=True, separators=(',', ':'))
    expected = hmac.new(SIGNING_KEY, manifest_str.encode(), hashlib.sha256)
    return hmac.compare_digest(expected.hexdigest(), signature)


def main():
    pack_dir = Path(__file__).parent
    
    # Load manifest
    manifest_path = pack_dir / 'manifest.json'
    if not manifest_path.exists():
        print("ERROR: manifest.json not found")
        return 1
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # Load signature
    sig_path = pack_dir / 'manifest.sig'
    if not sig_path.exists():
        print("ERROR: manifest.sig not found")
        return 1
    
    with open(sig_path) as f:
        signature = f.read().strip()
    
    # Verify signature
    if not verify_signature(manifest, signature):
        print("ERROR: Signature verification failed")
        return 1
    
    print("Signature: VALID")
    
    # Verify file hashes
    all_valid = True
    for filename, info in manifest.get('files', {}).items():
        file_path = pack_dir / filename
        
        if not file_path.exists():
            print(f"  MISSING: {filename}")
            all_valid = False
            continue
        
        actual_hash = compute_file_hash(file_path)
        expected_hash = info.get('sha256', '')
        
        if actual_hash == expected_hash:
            print(f"  VALID: {filename}")
        else:
            print(f"  CORRUPTED: {filename}")
            all_valid = False
    
    if all_valid:
        print("\\nVERIFICATION PASSED")
        return 0
    else:
        print("\\nVERIFICATION FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
'''

    verify_path = output_dir / "verify.py"
    with open(verify_path, "w") as f:
        f.write(verify_script)


def main():
    parser = argparse.ArgumentParser(
        description="Build evidence pack for validation case"
    )
    parser.add_argument(
        "--case",
        "-c",
        required=True,
        choices=["sod", "weno", "oblique", "dmr", "sbli"],
        help="Validation case to run",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        required=True,
        help="Output directory for evidence pack",
    )

    args = parser.parse_args()
    output_dir = args.out.resolve()

    print("=" * 60)
    print(" EVIDENCE PACK BUILDER")
    print("=" * 60)
    print(f"Case: {args.case}")
    print(f"Output: {output_dir}")
    print()

    # Add project to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Create output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Run validation case
    print("Running validation case...")
    case_info = run_validation_case(args.case, output_dir)

    # Create manifest
    manifest = {
        "version": "1.0",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "case": args.case,
        "files": case_info["files"],
        "metrics": case_info.get("metrics", {}),
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Sign manifest
    signature = sign_manifest(manifest)
    sig_path = output_dir / "manifest.sig"
    with open(sig_path, "w") as f:
        f.write(signature)

    print(f"Manifest signed: {signature[:16]}...")

    # Create verify script
    create_verify_script(output_dir)
    print("Created verify.py")

    # Verify pack
    print()
    print("Verifying pack...")
    result = subprocess.run(
        [sys.executable, str(output_dir / "verify.py")], capture_output=True, text=True
    )

    if result.returncode == 0:
        print(result.stdout)
        print()
        print("=" * 60)
        print(" ✓ EVIDENCE PACK BUILT SUCCESSFULLY")
        print("=" * 60)
        return 0
    else:
        print(result.stdout)
        print(result.stderr)
        print()
        print("=" * 60)
        print(" ✗ EVIDENCE PACK VERIFICATION FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    exit(main())
