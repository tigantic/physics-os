#!/usr/bin/env python3
"""Phase 5 — ontic/ domain decomposition migration script.

Moves 89 flat submodules into 13 logical groups (4 expanded existing + 9 new),
reducing ontic/ from 107 top-level packages to ~27.

Mechanism:
  1. Create group directories with __init__.py
  2. Move modules into their group directories
  3. Create backward-compatibility shim __init__.py at old locations
  4. Rewrite all imports across the codebase to use new paths

The shims ensure zero breakage during transition; import rewrites ensure
code eventually references canonical paths.

Usage:
  python tools/migrate_tensornet_phase5.py          # dry-run (default)
  python tools/migrate_tensornet_phase5.py --execute # perform actual migration
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Configuration: module → group mapping
# ──────────────────────────────────────────────────────────────

# Modules that STAY flat at ontic/ level (no move)
FLAT_MODULES = frozenset({
    "core", "types", "mps", "mpo", "qtt", "numerics", "algorithms",
    "cfd", "em", "genesis", "platform", "packs", "cuda", "docs",
})

# Existing modules that stay flat AND become group hosts
# (other modules get moved INTO them)
EXPAND_HOSTS = frozenset({"quantum", "fluids", "astro", "materials"})

# Module → group mapping (module_name → group_directory_name)
# For EXPAND_HOSTS, the group name IS the existing module name.
# For new groups, the group directory is created fresh.
MODULE_TO_GROUP: dict[str, str] = {
    # engine/ (11 modules)
    "vm": "engine",
    "gpu": "engine",
    "distributed": "engine",
    "distributed_tn": "engine",
    "adaptive": "engine",
    "substrate": "engine",
    "gateway": "engine",
    "realtime": "engine",
    "hardware": "engine",
    "hw": "engine",
    "fuel": "engine",
    # quantum/ — expand existing (7 modules absorbed)
    "quantum_mechanics": "quantum",
    "qm": "quantum",
    "qft": "quantum",
    "condensed_matter": "quantum",
    "statmech": "quantum",
    "electronic_structure": "quantum",
    "semiconductor": "quantum",
    # fluids/ — expand existing (10 modules absorbed)
    "free_surface": "fluids",
    "multiphase": "fluids",
    "porous_media": "fluids",
    "heat_transfer": "fluids",
    "coupled": "fluids",
    "fsi": "fluids",
    "multiscale": "fluids",
    "mesh_amr": "fluids",
    "computational_methods": "fluids",
    "phase_field": "fluids",
    # astro/ — expand existing (2 modules absorbed)
    "geophysics": "astro",
    "relativity": "astro",
    # materials/ — expand existing (2 modules absorbed)
    "mechanics": "materials",
    "manufacturing": "materials",
    # plasma_nuclear/ (3 modules)
    "plasma": "plasma_nuclear",
    "nuclear": "plasma_nuclear",
    "fusion": "plasma_nuclear",
    # life_sci/ (6 modules)
    "chemistry": "life_sci",
    "biology": "life_sci",
    "biomedical": "life_sci",
    "biophysics": "life_sci",
    "membrane_bio": "life_sci",
    "md": "life_sci",
    # energy_env/ (4 modules)
    "energy": "energy_env",
    "environmental": "energy_env",
    "urban": "energy_env",
    "agri": "energy_env",
    # ml/ (5 modules)
    "ml_surrogates": "ml",
    "ml_physics": "ml",
    "neural": "ml",
    "discovery": "ml",
    "data": "ml",
    # sim/ (6 modules)
    "simulation": "sim",
    "validation": "sim",
    "benchmarks": "sim",
    "certification": "sim",
    "flight_validation": "sim",
    "visualization": "sim",
    # aerospace/ (5 modules)
    "guidance": "aerospace",
    "autonomy": "aerospace",
    "defense": "aerospace",
    "exploit": "aerospace",
    "racing": "aerospace",
    # infra/ (15 modules)
    "oracle": "infra",
    "coordination": "infra",
    "hyperenv": "infra",
    "hypersim": "infra",
    "integration": "infra",
    "site": "infra",
    "sdk": "infra",
    "sovereign": "infra",
    "zk": "infra",
    "provenance": "infra",
    "deployment": "infra",
    "digital_twin": "infra",
    "fieldops": "infra",
    "fieldos": "infra",
    "hypervisual": "infra",
    # applied/ (13 modules)
    "medical": "applied",
    "financial": "applied",
    "cyber": "applied",
    "emergency": "applied",
    "special_applied": "applied",
    "robotics_physics": "applied",
    "physics": "applied",
    "shaders": "applied",
    "intent": "applied",
    "particle": "applied",
    "acoustics": "applied",
    "radiation": "applied",
    "optics": "applied",
}

# New group directories (not expanding an existing module)
NEW_GROUPS = frozenset({
    "engine", "plasma_nuclear", "life_sci", "energy_env",
    "ml", "sim", "aerospace", "infra", "applied",
})

# ──────────────────────────────────────────────────────────────
# Shim template
# ──────────────────────────────────────────────────────────────

SHIM_TEMPLATE = '''\
"""Backward-compatibility shim — real module at ontic.{group}.{module}.

This shim exists so that legacy imports like::

    from ontic.{module} import X
    from ontic.{module}.sub import Y

continue to work after the Phase 5 domain decomposition.
The canonical import path is now ``ontic.{group}.{module}``.
"""
import importlib as _il
import sys as _sys

_real = _il.import_module("ontic.{group}.{module}")
_sys.modules[__name__] = _real
'''

# ──────────────────────────────────────────────────────────────
# Group __init__.py template
# ──────────────────────────────────────────────────────────────

GROUP_INIT_TEMPLATE = '''\
"""ontic.{group} — {description}.

Submodules
----------
{submodule_list}
"""
from __future__ import annotations
'''

GROUP_DESCRIPTIONS: dict[str, str] = {
    "engine": "execution engine, GPU, distributed runtime, and hardware abstraction",
    "plasma_nuclear": "plasma physics, nuclear engineering, and fusion energy",
    "life_sci": "chemistry, biology, biomedical engineering, and molecular dynamics",
    "energy_env": "energy systems, environmental modeling, urban physics, and agriculture",
    "ml": "machine-learning surrogates, neural operators, and scientific discovery",
    "sim": "simulation orchestration, validation, benchmarking, and certification",
    "aerospace": "guidance, autonomy, defense, exploit analysis, and racing",
    "infra": "platform infrastructure, coordination, deployment, and integrations",
    "applied": "applied physics domains (medical, financial, cyber, acoustics, optics, …)",
}


def repo_root() -> Path:
    """Return the repository root (parent of ontic/)."""
    here = Path(__file__).resolve().parent
    # tools/ is one level below root
    root = here.parent
    if not (root / "ontic" / "__init__.py").exists():
        raise RuntimeError(f"Cannot find ontic/ at {root}")
    return root


def all_python_files(root: Path) -> list[Path]:
    """Collect every .py file in the repo, excluding __pycache__ and .git."""
    result: list[Path] = []
    skip = {".git", "__pycache__", "target", ".cache_data", "node_modules", ".venv"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip]
        for f in filenames:
            if f.endswith(".py"):
                result.append(Path(dirpath) / f)
    return result


# ──────────────────────────────────────────────────────────────
# Step 1: Create group directories
# ──────────────────────────────────────────────────────────────

def create_group_dirs(root: Path, *, dry_run: bool = True) -> list[str]:
    """Create new group directories under ontic/ and write __init__.py."""
    actions: list[str] = []
    tn = root / "ontic"

    for group in NEW_GROUPS:
        group_dir = tn / group
        init_file = group_dir / "__init__.py"
        if group_dir.exists():
            actions.append(f"SKIP group dir (exists): {group_dir.relative_to(root)}")
            continue
        actions.append(f"CREATE group dir: {group_dir.relative_to(root)}")

        # Build submodule list for docstring
        members = sorted(m for m, g in MODULE_TO_GROUP.items() if g == group)
        submod_lines = "\n".join(f"- ``{m}``" for m in members)
        desc = GROUP_DESCRIPTIONS.get(group, group)
        content = GROUP_INIT_TEMPLATE.format(
            group=group, description=desc, submodule_list=submod_lines,
        )

        if not dry_run:
            group_dir.mkdir(parents=True, exist_ok=True)
            init_file.write_text(content)

    return actions


# ──────────────────────────────────────────────────────────────
# Step 2: Move modules into group directories
# ──────────────────────────────────────────────────────────────

def move_modules(root: Path, *, dry_run: bool = True) -> list[str]:
    """Move each module directory into its group directory."""
    actions: list[str] = []
    tn = root / "ontic"

    for module, group in sorted(MODULE_TO_GROUP.items()):
        src = tn / module
        dst = tn / group / module

        if not src.exists():
            actions.append(f"WARN source missing: {src.relative_to(root)}")
            continue
        if dst.exists():
            actions.append(f"SKIP (already moved): {module} → {group}/{module}")
            continue

        actions.append(f"MOVE: ontic/{module}/ → ontic/{group}/{module}/")

        if not dry_run:
            # Ensure group dir exists (for expand_hosts case)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))

    return actions


# ──────────────────────────────────────────────────────────────
# Step 3: Create backward-compatibility shim packages
# ──────────────────────────────────────────────────────────────

def create_shims(root: Path, *, dry_run: bool = True) -> list[str]:
    """Create shim __init__.py at old module locations."""
    actions: list[str] = []
    tn = root / "ontic"

    for module, group in sorted(MODULE_TO_GROUP.items()):
        # For expand_hosts, the old directory IS the group — no shim needed
        if group in EXPAND_HOSTS and module in EXPAND_HOSTS:
            continue

        shim_dir = tn / module
        shim_init = shim_dir / "__init__.py"

        if shim_dir.exists() and shim_init.exists():
            # Check if it's already a shim
            content = shim_init.read_text()
            if "Backward-compatibility shim" in content:
                actions.append(f"SKIP shim (exists): ontic/{module}/")
                continue

        actions.append(f"CREATE shim: ontic/{module}/__init__.py → ontic.{group}.{module}")

        if not dry_run:
            shim_dir.mkdir(parents=True, exist_ok=True)
            shim_init.write_text(
                SHIM_TEMPLATE.format(group=group, module=module)
            )

    return actions


# ──────────────────────────────────────────────────────────────
# Step 4: Rewrite imports across the codebase
# ──────────────────────────────────────────────────────────────

def build_import_patterns() -> list[tuple[re.Pattern[str], str]]:
    """Build regex patterns for rewriting imports.

    Returns (compiled_pattern, replacement_template) pairs.
    Patterns are ordered longest-module-name first to avoid partial matches.
    """
    patterns: list[tuple[re.Pattern[str], str]] = []

    # Sort by module name length (descending) to match longest first
    # e.g. "quantum_mechanics" before "quantum"
    for module in sorted(MODULE_TO_GROUP, key=len, reverse=True):
        group = MODULE_TO_GROUP[module]

        # Skip expand_hosts — their OWN imports don't change
        # (quantum stays ontic.quantum, but quantum_mechanics → ontic.quantum.quantum_mechanics)
        if module in EXPAND_HOSTS:
            continue

        # Pattern: from ontic.MODULE -> from ontic.GROUP.MODULE
        # Matches: from ontic.MODULE.sub import X
        #          from ontic.MODULE import X
        #          import ontic.MODULE
        # Must NOT match: from ontic.MODULE_something (partial match)
        # Uses word boundary after module name

        # "from ontic.MODULE." → "from ontic.GROUP.MODULE."
        patterns.append((
            re.compile(rf"from ontic\.{re.escape(module)}\."),
            f"from ontic.{group}.{module}.",
        ))

        # "from ontic.MODULE import" → "from ontic.GROUP.MODULE import"
        patterns.append((
            re.compile(rf"from ontic\.{re.escape(module)} import"),
            f"from ontic.{group}.{module} import",
        ))

        # "import ontic.MODULE" → "import ontic.GROUP.MODULE"
        patterns.append((
            re.compile(rf"(?<![.\w])import ontic\.{re.escape(module)}\b"),
            f"import ontic.{group}.{module}",
        ))

        # "ontic.MODULE." in string literals / comments (for completeness)
        # Skip this — too aggressive and might break docs/strings

    return patterns


def rewrite_imports(
    root: Path,
    *,
    dry_run: bool = True,
) -> tuple[list[str], int]:
    """Rewrite all imports across the codebase to use new paths.

    Returns (actions_log, total_rewrites_count).
    """
    actions: list[str] = []
    patterns = build_import_patterns()
    total_rewrites = 0

    py_files = all_python_files(root)

    # Files to SKIP rewriting (the shim __init__.py files themselves)
    shim_files: set[Path] = set()
    tn = root / "ontic"
    for module, group in MODULE_TO_GROUP.items():
        if module not in EXPAND_HOSTS:
            shim_files.add(tn / module / "__init__.py")

    for fpath in py_files:
        if fpath in shim_files:
            continue

        try:
            original = fpath.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue

        modified = original
        file_rewrites = 0

        for pattern, replacement in patterns:
            new_text, count = pattern.subn(replacement, modified)
            if count > 0:
                file_rewrites += count
                modified = new_text

        if file_rewrites > 0:
            rel = fpath.relative_to(root)
            actions.append(f"REWRITE ({file_rewrites:3d} changes): {rel}")
            total_rewrites += file_rewrites

            if not dry_run:
                fpath.write_text(modified, encoding="utf-8")

    return actions, total_rewrites


# ──────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────

def validate_structure(root: Path) -> list[str]:
    """Check that the migration produced a valid structure."""
    issues: list[str] = []
    tn = root / "ontic"

    # Check all group dirs exist
    for group in NEW_GROUPS:
        if not (tn / group / "__init__.py").exists():
            issues.append(f"MISSING group __init__.py: ontic/{group}/__init__.py")

    # Check all modules are in their group
    for module, group in MODULE_TO_GROUP.items():
        target = tn / group / module
        if not target.exists():
            issues.append(f"MISSING moved module: ontic/{group}/{module}/")

    # Check all shims exist
    for module, group in MODULE_TO_GROUP.items():
        if module in EXPAND_HOSTS:
            continue
        shim = tn / module / "__init__.py"
        if not shim.exists():
            issues.append(f"MISSING shim: ontic/{module}/__init__.py")

    # Count top-level dirs
    top_level = [d for d in tn.iterdir() if d.is_dir() and d.name != "__pycache__"]
    n = len(top_level)
    if n > 35:
        issues.append(f"WARNING: {n} top-level dirs in ontic/ (target ≤30)")

    return issues


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5: ontic/ domain decomposition")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the migration (default is dry-run)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation checks on current state",
    )
    args = parser.parse_args()

    dry_run = not args.execute
    root = repo_root()

    if args.validate_only:
        print("=" * 60)
        print("  VALIDATION ONLY")
        print("=" * 60)
        issues = validate_structure(root)
        if issues:
            for i in issues:
                print(f"  ❌ {i}")
            sys.exit(1)
        else:
            print("  ✅ All checks passed")
            # Count final structure
            tn = root / "ontic"
            top_dirs = sorted(
                d.name for d in tn.iterdir()
                if d.is_dir() and d.name != "__pycache__"
            )
            print(f"  📁 {len(top_dirs)} top-level directories in ontic/")
        return

    mode = "DRY RUN" if dry_run else "EXECUTING"
    print("=" * 60)
    print(f"  Phase 5: ontic/ Domain Decomposition — {mode}")
    print("=" * 60)
    print(f"  Modules to move:  {len(MODULE_TO_GROUP)}")
    print(f"  New group dirs:   {len(NEW_GROUPS)}")
    print(f"  Expand hosts:     {len(EXPAND_HOSTS)}")
    print(f"  Flat (no change): {len(FLAT_MODULES)}")
    print()

    # Step 1
    print("─── Step 1: Create group directories ───")
    for action in create_group_dirs(root, dry_run=dry_run):
        print(f"  {action}")
    print()

    # Step 2
    print("─── Step 2: Move modules into groups ───")
    for action in move_modules(root, dry_run=dry_run):
        print(f"  {action}")
    print()

    # Step 3
    print("─── Step 3: Create backward-compatibility shims ───")
    for action in create_shims(root, dry_run=dry_run):
        print(f"  {action}")
    print()

    # Step 4
    print("─── Step 4: Rewrite imports ───")
    actions, total = rewrite_imports(root, dry_run=dry_run)
    for action in actions:
        print(f"  {action}")
    print(f"\n  Total import rewrites: {total}")
    print()

    if not dry_run:
        print("─── Validation ───")
        issues = validate_structure(root)
        if issues:
            for i in issues:
                print(f"  ❌ {i}")
            print("\n  ⚠️  Migration completed with issues")
        else:
            tn = root / "ontic"
            top_dirs = sorted(
                d.name for d in tn.iterdir()
                if d.is_dir() and d.name != "__pycache__"
            )
            print(f"  ✅ Migration complete — {len(top_dirs)} top-level dirs in ontic/")
    else:
        print("  ℹ️  Dry run complete. Run with --execute to apply changes.")


if __name__ == "__main__":
    main()
