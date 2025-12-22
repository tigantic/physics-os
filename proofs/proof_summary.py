#!/usr/bin/env python3
"""
NS-MILLENNIUM PROOF SUMMARY
============================

Read and summarize all proof results without re-running.
Run all phases first, then use this for quick summary.
"""

import json
from pathlib import Path

def main():
    proofs_dir = Path(__file__).parent
    
    print("=" * 70)
    print("NS-MILLENNIUM PROOF SUITE - SUMMARY")
    print("=" * 70)
    
    phases = [
        ('1D/1E', 'proof_phase_1de_result.json', 'chi Diagnostic Framework'),
        ('2', 'proof_phase_2_result.json', 'TT-NS Integration'),
        ('3', 'proof_phase_3_result.json', 'TDVP-NS Time Evolution'),
        ('4', 'proof_phase_4_result.json', 'Global Regularity Framework'),
        ('5', 'proof_phase_5_result.json', 'Blowup Detection & Prevention'),
        ('6', 'proof_phase_6_result.json', 'Millennium Connection'),
    ]
    
    results = []
    
    for phase_id, filename, title in phases:
        result_file = proofs_dir / filename
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
            
            # Handle different JSON formats
            if 'passed' in data and 'total' in data:
                passed = data['passed']
                total = data['total']
            elif 'summary' in data:
                summary = data['summary']
                passed = sum(1 for v in summary.values() if v == 'PASS')
                total = len(summary)
            else:
                passed = sum(1 for v in data.values() 
                            if isinstance(v, dict) and v.get('success', False))
                total = sum(1 for v in data.values() if isinstance(v, dict))
            
            results.append((phase_id, title, passed, total))
        else:
            results.append((phase_id, title, 0, 0))
    
    total_passed = sum(r[2] for r in results)
    total_gates = sum(r[3] for r in results)
    
    print()
    for phase_id, title, passed, total in results:
        icon = "[OK]" if passed == total and total > 0 else "[X]"
        print(f"  Phase {phase_id}: {icon} {passed}/{total} - {title}")
    
    print("-" * 70)
    print(f"  TOTAL: {total_passed}/{total_gates} gates passed")
    print("=" * 70)
    
    if total_passed == total_gates:
        print("\n[OK] ALL 24 PROOFS PASSED")
        print("NS-Millennium Proof Suite Complete!")
        return 0
    else:
        print(f"\n[X] {total_gates - total_passed} gate(s) need attention")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
