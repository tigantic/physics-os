#!/usr/bin/env python
"""
L) Truth Boundary Report Generator
===================================

Auto-generates a table showing implementation status of all features.

Usage:
    python tools/scripts/generate_truth_boundary.py

Pass Criteria:
    - Committed artifact matches actual code scan
    - No manual drift between docs and implementation
"""

import ast
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class FeatureStatus:
    """Status of a feature."""

    name: str
    module: str
    category: str
    status: str  # 'implemented', 'stub', 'planned', 'tested'
    has_tests: bool
    has_docs: bool
    line_count: int
    functions: List[str]


def count_lines(path: Path) -> int:
    """Count non-empty, non-comment lines."""
    count = 0
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    count += 1
    except Exception:
        pass
    return count


def extract_functions(path: Path) -> List[str]:
    """Extract function and class names from a Python file."""
    functions = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(f"def {node.name}")
            elif isinstance(node, ast.ClassDef):
                functions.append(f"class {node.name}")
    except Exception:
        pass
    return functions


def has_not_implemented(path: Path) -> bool:
    """Check if file contains NotImplementedError."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return "NotImplementedError" in content or "raise NotImplemented" in content
    except Exception:
        return False


def find_test_file(module_path: Path, tests_dir: Path) -> Optional[Path]:
    """Find corresponding test file for a module."""
    module_name = module_path.stem

    # Try various patterns
    patterns = [
        f"test_{module_name}.py",
        f"test_{module_name.replace('_', '')}.py",
    ]

    for pattern in patterns:
        test_path = tests_dir / pattern
        if test_path.exists():
            return test_path

    # Search recursively
    for test_file in tests_dir.rglob(f"test_{module_name}*.py"):
        return test_file

    return None


def find_doc_file(
    module_path: Path, docs_dir: Path, project_root: Path
) -> Optional[Path]:
    """Find corresponding documentation file for a module."""
    rel_path = module_path.relative_to(project_root / "ontic")

    # Try docs/api/{path}.md
    doc_name = str(rel_path).replace("/", ".").replace("\\", ".").replace(".py", ".md")
    doc_path = docs_dir / "api" / doc_name

    if doc_path.exists():
        return doc_path

    return None


def analyze_module(
    path: Path, project_root: Path, tests_dir: Path, docs_dir: Path
) -> FeatureStatus:
    """Analyze a single module."""
    rel_path = path.relative_to(project_root)
    module_name = path.stem
    category = path.parent.name

    line_count = count_lines(path)
    functions = extract_functions(path)
    has_stubs = has_not_implemented(path)
    test_file = find_test_file(path, tests_dir)
    doc_file = find_doc_file(path, docs_dir, project_root)

    # Determine status
    if line_count < 50:
        status = "stub"
    elif has_stubs:
        status = "partial"
    else:
        status = "implemented"

    if test_file and status == "implemented":
        status = "tested"

    return FeatureStatus(
        name=module_name,
        module=str(rel_path),
        category=category,
        status=status,
        has_tests=test_file is not None,
        has_docs=doc_file is not None,
        line_count=line_count,
        functions=functions[:10],  # Limit to first 10
    )


def generate_report(project_root: Path) -> Dict[str, Any]:
    """Generate the truth boundary report."""
    tensornet_dir = project_root / "ontic"
    tests_dir = project_root / "tests"
    docs_dir = project_root / "docs"

    modules = []

    # Scan all Python files in ontic
    for py_file in tensornet_dir.rglob("*.py"):
        if py_file.name.startswith("_") and py_file.name != "__init__.py":
            continue
        if "__pycache__" in str(py_file):
            continue

        feature = analyze_module(py_file, project_root, tests_dir, docs_dir)
        modules.append(feature)

    # Group by category and status
    by_category: Dict[str, List[FeatureStatus]] = {}
    by_status: Dict[str, int] = {"tested": 0, "implemented": 0, "partial": 0, "stub": 0}

    for module in modules:
        if module.category not in by_category:
            by_category[module.category] = []
        by_category[module.category].append(module)
        by_status[module.status] = by_status.get(module.status, 0) + 1

    # Build report
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "summary": {
            "total_modules": len(modules),
            "by_status": by_status,
            "total_lines": sum(m.line_count for m in modules),
            "with_tests": sum(1 for m in modules if m.has_tests),
            "with_docs": sum(1 for m in modules if m.has_docs),
        },
        "categories": {},
        "modules": [],
    }

    for category, mods in sorted(by_category.items()):
        report["categories"][category] = {
            "count": len(mods),
            "lines": sum(m.line_count for m in mods),
            "tested": sum(1 for m in mods if m.status == "tested"),
        }

    for module in sorted(modules, key=lambda m: (m.category, m.name)):
        report["modules"].append(
            {
                "name": module.name,
                "module": module.module,
                "category": module.category,
                "status": module.status,
                "has_tests": module.has_tests,
                "has_docs": module.has_docs,
                "lines": module.line_count,
            }
        )

    return report


def generate_markdown(report: Dict[str, Any]) -> str:
    """Generate markdown table from report."""
    lines = [
        "# Truth Boundary Report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Summary",
        "",
        f"- **Total Modules**: {report['summary']['total_modules']}",
        f"- **Total Lines**: {report['summary']['total_lines']:,}",
        f"- **Tested**: {report['summary']['by_status'].get('tested', 0)}",
        f"- **Implemented**: {report['summary']['by_status'].get('implemented', 0)}",
        f"- **Partial**: {report['summary']['by_status'].get('partial', 0)}",
        f"- **Stub**: {report['summary']['by_status'].get('stub', 0)}",
        "",
        "## By Category",
        "",
        "| Category | Modules | Lines | Tested |",
        "|----------|---------|-------|--------|",
    ]

    for category, info in sorted(report["categories"].items()):
        lines.append(
            f"| {category} | {info['count']} | {info['lines']:,} | {info['tested']} |"
        )

    lines.extend(
        [
            "",
            "## All Modules",
            "",
            "| Module | Category | Status | Lines | Tests | Docs |",
            "|--------|----------|--------|-------|-------|------|",
        ]
    )

    status_emoji = {
        "tested": "✅",
        "implemented": "🟢",
        "partial": "🟡",
        "stub": "⚪",
    }

    for module in report["modules"]:
        status = status_emoji.get(module["status"], "❓")
        tests = "✓" if module["has_tests"] else ""
        docs = "✓" if module["has_docs"] else ""
        lines.append(
            f"| {module['name']} | {module['category']} | {status} | "
            f"{module['lines']} | {tests} | {docs} |"
        )

    return "\n".join(lines)


def main():
    print("=" * 60)
    print(" TRUTH BOUNDARY REPORT GENERATOR")
    print("=" * 60)
    print()

    project_root = Path(__file__).parent.parent

    # Generate report
    print("Scanning modules...")
    report = generate_report(project_root)

    # Print summary
    summary = report["summary"]
    print(f"\nTotal modules: {summary['total_modules']}")
    print(f"Total lines: {summary['total_lines']:,}")
    print(f"With tests: {summary['with_tests']}")
    print(f"With docs: {summary['with_docs']}")
    print()
    print("By status:")
    for status, count in sorted(summary["by_status"].items()):
        print(f"  {status}: {count}")

    # Save JSON
    json_path = project_root / "artifacts" / "truth_boundary.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON: {json_path}")

    # Save markdown
    md_path = project_root / "docs" / "TRUTH_BOUNDARY.md"
    md_content = generate_markdown(report)
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"Markdown: {md_path}")

    print()
    print("=" * 60)
    print(" ✓ TRUTH BOUNDARY REPORT GENERATED")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
