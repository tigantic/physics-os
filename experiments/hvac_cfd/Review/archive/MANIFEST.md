# Archive Manifest

**Created:** 2026-01-10  
**Purpose:** Track files archived during comprehensive audit

## archive/demo_code/

| File | Original Location | Reason |
|------|------------------|--------|
| bridge_standalone.py | hyperfoam/ | Uses synthetic animated data instead of real physics |

## archive/duplicate_code/

| File | Original Location | Reason |
|------|------------------|--------|
| hyperfoam_solver.py | Tier1/ | Duplicate of hyperfoam/core/solver.py |

## archive/tier1_old_versions/

| File | Original Location | Reason |
|------|------------------|--------|
| qtt_ns_3d_v3.py | Tier1/ | Older QTT version (ceiling jet fix) - superseded by qtt_ns_3d_fixed.py |
| qtt_ns_3d_v4.py | Tier1/ | Older QTT version (multi-cell inlet) - superseded by qtt_ns_3d_fixed.py |
| qtt_nielsen_runner.py | Tier1/ | v1 benchmark runner - superseded by qtt_nielsen_runner_v2.py |
| tier1_james_conference_room.py | Tier1/ | v1 James simulation - superseded by tier1_james_v2.py |

## archive/orphaned_code/

| File | Original Location | Reason |
|------|------------------|--------|
| advanced_optimizer.py | hyperfoam/ | Adjoint/NSGA-II optimizer never integrated into solver |
| rom.py | hyperfoam/ | POD/ROM standalone module never integrated |
| low_mach.py | hyperfoam/ | Low-Mach compressible solver never integrated |

## Deleted (not archived)

| File | Original Location | Reason |
|------|------------------|--------|
| dominion-gui/ | root | Entire Rust GUI - fake timer, not connected to physics |
| trust_fabric.py | hyperfoam/ | Fake PQC using random bytes, not real crypto |
| hyperfoam/pyproject.toml | hyperfoam/ | Duplicate of root pyproject.toml |
| hyperfoam-bridge.spec | hyperfoam/ | Duplicate of hyperfoam_bridge.spec |
| hyperfoam.egg-info/ | hyperfoam/ | Auto-generated build artifacts |

## Notes

These files were archived (not deleted) because:
1. They may have historical value
2. They could be restored if needed
3. They serve as examples of what NOT to do

The live codebase now contains only production-ready code.

## Restoration

To restore any file:
```bash
mv archive/demo_code/bridge_standalone.py hyperfoam/
```
