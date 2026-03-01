#!/usr/bin/env python
"""
Evidence Pack Verification Script
==================================

Verifies the integrity and correctness of the flagship pipeline evidence pack.

Usage:
    python verify.py

Expected Output: PASS
"""

import hashlib
import hmac
import json
import sys
from pathlib import Path

EXPECTED_SIGNATURE_KEY = b'physics-os-flagship-2024'

def main():
    print("=" * 50)
    print(" EVIDENCE PACK VERIFICATION")
    print("=" * 50)
    
    pack_dir = Path(__file__).parent
    manifest_path = pack_dir / 'manifest.json'
    data_dir = pack_dir / 'data'
    
    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    stored_signature = manifest.pop('signature', None)
    
    # Verify signature
    manifest_json = json.dumps(manifest, indent=2, sort_keys=True)
    computed_signature = hmac.new(
        EXPECTED_SIGNATURE_KEY, 
        manifest_json.encode(), 
        hashlib.sha256
    ).hexdigest()
    
    print("\n[1] Signature Verification")
    if computed_signature == stored_signature:
        print("    ✓ Manifest signature valid")
    else:
        print("    ✗ Manifest signature INVALID")
        print("    FAIL")
        return 1
    
    # Verify file hashes
    print("\n[2] File Hash Verification")
    file_hashes = manifest.get('file_hashes', {})
    all_hashes_valid = True
    
    for filename, expected_hash in file_hashes.items():
        file_path = data_dir / filename
        if not file_path.exists():
            print(f"    ✗ Missing: {filename}")
            all_hashes_valid = False
            continue
        
        with open(file_path, 'rb') as f:
            computed_hash = hashlib.sha256(f.read()).hexdigest()
        
        if computed_hash == expected_hash:
            print(f"    ✓ {filename}")
        else:
            print(f"    ✗ {filename} hash mismatch")
            all_hashes_valid = False
    
    if not all_hashes_valid:
        print("    FAIL")
        return 1
    
    # Verify pass status
    print("\n[3] Validation Status")
    validations = manifest.get('results', {}).get('validations', {})
    all_passed = validations.get('all_passed', False)
    
    if all_passed:
        print("    ✓ All physics validations passed")
    else:
        print("    ✗ Some validations failed")
        for key, val in validations.items():
            if isinstance(val, dict) and not val.get('passed', True):
                print(f"      - {key}: FAILED")
    
    # Final verdict
    print()
    print("=" * 50)
    if stored_signature == computed_signature and all_hashes_valid and all_passed:
        print(" ✓ PASS")
        print("=" * 50)
        return 0
    else:
        print(" ✗ FAIL")
        print("=" * 50)
        return 1


if __name__ == '__main__':
    sys.exit(main())
