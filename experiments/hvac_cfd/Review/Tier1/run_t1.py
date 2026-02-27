#!/usr/bin/env python3
"""
TigantiCFD - Quick Runner
=========================

Run this to execute the Tier 1 James Morrison Conference Room analysis.

Usage:
    python run_t1.py              # Run with defaults
    python run_t1.py ./output     # Specify output directory
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from tier1_james_conference_room import run_tier1_simulation

if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "./tiganti_output/TGC-2026-001"
    result = run_tier1_simulation(output_dir)
    
    print("\n" + "="*70)
    print("DELIVERABLE FILES:")
    print("="*70)
    print(f"  Report:  {result['report_path']}")
    for fig in result['figure_paths']:
        print(f"  Figure:  {fig}")
    print()
    print(f"Comfort Score: {result['comfort'].comfort_score:.1f}/100")
    print("="*70)
