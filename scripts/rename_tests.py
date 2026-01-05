#!/usr/bin/env python3
"""
Test Renaming Script for Constitutional Compliance (Article III.3.2)
====================================================================

Renames tests from `test_<component>` to `test_<component>_<behavior>_<condition>`
where behavior and condition can be inferred from context.

Usage: python scripts/rename_tests.py [--dry-run]
"""

import re
import sys
from pathlib import Path

# Mapping of old names to new names following test_<component>_<behavior>_<condition>
RENAMES = {
    # TestCoreImports
    "test_import_tensornet": "test_tensornet_import_succeeds_when_installed",
    "test_import_mps": "test_mps_import_succeeds_when_installed",
    "test_import_mpo": "test_mpo_import_succeeds_when_installed",
    "test_import_algorithms": "test_algorithms_import_succeeds_when_installed",
    "test_import_hamiltonians": "test_hamiltonians_import_succeeds_when_installed",
    "test_import_cfd": "test_cfd_import_succeeds_when_installed",
    # TestMPS
    "test_random_mps": "test_mps_random_creates_valid_state",
    "test_mps_norm": "test_mps_norm_equals_one_when_normalized",
    "test_mps_canonicalization": "test_mps_canonicalize_produces_orthogonal_tensors",
    "test_ghz_state": "test_mps_ghz_has_log2_entropy_at_center",
    # TestMPO
    "test_heisenberg_mpo": "test_mpo_heisenberg_creates_valid_structure",
    "test_tfim_mpo": "test_mpo_tfim_creates_valid_structure",
    "test_mpo_hermiticity": "test_mpo_hermiticity_satisfied_for_heisenberg",
    # TestDMRG
    "test_dmrg_runs": "test_dmrg_runs_without_error_on_small_chain",
    # TestTEBD
    "test_tebd_gates": "test_tebd_gates_have_correct_shape_for_heisenberg",
    # TestCFD
    "test_euler_1d_creation": "test_euler1d_creates_correct_grid_spacing",
    "test_sod_initial_condition": "test_euler1d_sod_ic_has_correct_states",
    "test_euler_step": "test_euler1d_step_advances_time_positively",
    "test_euler_to_mps": "test_euler1d_to_mps_preserves_dimensions",
    "test_exact_riemann_solver": "test_riemann_exact_produces_positive_density",
    # TestLimiters
    "test_minmod": "test_limiter_minmod_clips_to_valid_range",
    "test_superbee": "test_limiter_superbee_is_compressive",
    # TestEuler2D
    "test_euler_2d_creation": "test_euler2d_creates_correct_grid",
    "test_euler_2d_state": "test_euler2d_state_computes_mach_number",
    "test_euler_2d_conservative_conversion": "test_euler2d_conservative_roundtrip_preserves_state",
    "test_supersonic_wedge_ic": "test_euler2d_wedge_ic_is_uniform_supersonic",
    # Add more mappings as needed...
}


def rename_tests(file_path: Path, dry_run: bool = True) -> int:
    """Rename test functions in a file."""
    content = file_path.read_text(encoding="utf-8")
    original = content
    count = 0

    for old_name, new_name in RENAMES.items():
        # Match function definition
        pattern = rf"\bdef {old_name}\("
        replacement = f"def {new_name}("

        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            count += 1
            print(f"  {old_name} -> {new_name}")

    if not dry_run and content != original:
        file_path.write_text(content, encoding="utf-8")
        print(f"\nWrote {count} renames to {file_path}")
    elif dry_run:
        print(f"\n[DRY RUN] Would rename {count} tests")

    return count


def main():
    dry_run = "--dry-run" in sys.argv or len(sys.argv) == 1

    if dry_run:
        print("Running in DRY RUN mode. Use --apply to make changes.\n")

    test_file = Path(__file__).parent.parent / "tests" / "test_integration.py"

    if not test_file.exists():
        print(f"Error: {test_file} not found")
        return 1

    print(f"Processing: {test_file}\n")
    count = rename_tests(test_file, dry_run=dry_run)

    if count == 0:
        print("No renames needed or all already applied.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
