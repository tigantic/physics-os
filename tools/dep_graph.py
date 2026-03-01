#!/usr/bin/env python3
"""Generate dependency graph of ontic's internal module structure.

Walks ontic/ source tree, parses imports, and outputs a graph in
SVG (via graphviz), Mermaid markdown, or DOT format.

Usage:
    python tools/dep_graph.py                             # Print Mermaid to stdout
    python tools/dep_graph.py --format svg -o graph.svg   # SVG via graphviz
    python tools/dep_graph.py --format dot -o graph.dot   # DOT file
    python tools/dep_graph.py --format mermaid -o graph.md # Mermaid file
    python tools/dep_graph.py --group-only                # Show only group-level edges

Requirements (SVG only):
    pip install graphviz   # Python bindings
    # Plus: graphviz system package (apt install graphviz / brew install graphviz)
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ONTIC_ROOT = REPO_ROOT / "ontic"

# ── Domain group classification ──────────────────────────────────────────
# Mirrors the Phase 5 migration groups.
MODULE_TO_GROUP: dict[str, str] = {}

GROUP_MEMBERS: dict[str, list[str]] = {
    "core": [
        "core", "types", "mps", "mpo", "qtt", "numerics", "algorithms",
    ],
    "engine": [
        "engine", "platform", "sdk", "genesis", "gateway", "sovereign",
        "oracle", "vm", "hyperenv", "hypersim", "fieldos", "fieldops",
        "site", "realtime", "intent", "coordination", "integration",
        "logging_config", "benchmark_runner", "benchmarks",
    ],
    "cfd": [
        "cfd", "fluids", "multiphase", "free_surface", "fsi", "heat_transfer",
    ],
    "quantum": [
        "quantum", "qm", "qft", "condensed_matter", "statmech",
        "electronic_structure", "quantum_mechanics",
    ],
    "materials": [
        "materials", "mechanics", "phase_field",
    ],
    "aerospace": [
        "aerospace", "flight_validation", "guidance", "racing",
    ],
    "astro": [
        "astro", "relativity", "geophysics",
    ],
    "plasma_nuclear": [
        "plasma", "plasma_nuclear", "nuclear", "fusion", "radiation",
    ],
    "life_sci": [
        "life_sci", "biology", "biomedical", "biophysics", "medical",
        "membrane_bio",
    ],
    "energy_env": [
        "energy_env", "energy", "environmental", "fuel",
    ],
    "ml": [
        "ml", "ml_physics", "ml_surrogates", "neural", "discovery",
    ],
    "sim": [
        "sim", "simulation", "digital_twin", "coupled", "multiscale",
        "mesh_amr", "porous_media", "particle", "md", "computational_methods",
    ],
    "infra": [
        "infra", "distributed", "distributed_tn", "cuda", "gpu", "hw",
        "hardware", "shaders", "deployment", "hypervisual", "visualization",
        "data",
    ],
    "applied": [
        "applied", "special_applied", "defense", "cyber", "financial",
        "urban", "manufacturing", "autonomy", "robotics_physics",
        "emergency", "agri",
    ],
    "em": [
        "em", "optics", "semiconductor", "acoustics", "physics",
    ],
    "security": [
        "zk", "exploit", "certification", "provenance", "validation",
    ],
    "packs": [
        "packs", "substrate", "adaptive", "docs",
    ],
}

for group, members in GROUP_MEMBERS.items():
    for mod in members:
        MODULE_TO_GROUP[mod] = group

# Colors for groups
GROUP_COLORS: dict[str, str] = {
    "core": "#6366f1",
    "engine": "#8b5cf6",
    "cfd": "#ef4444",
    "quantum": "#3b82f6",
    "materials": "#f59e0b",
    "aerospace": "#10b981",
    "astro": "#6366f1",
    "plasma_nuclear": "#f97316",
    "life_sci": "#ec4899",
    "energy_env": "#84cc16",
    "ml": "#14b8a6",
    "sim": "#a855f7",
    "infra": "#64748b",
    "applied": "#78716c",
    "em": "#0ea5e9",
    "security": "#dc2626",
    "packs": "#9ca3af",
}


def get_group(module_name: str) -> str:
    """Return the domain group for a top-level ontic submodule."""
    return MODULE_TO_GROUP.get(module_name, "other")


def parse_imports(filepath: Path) -> set[str]:
    """Extract ontic imports from a Python file."""
    imports: set[str] = set()
    try:
        tree = ast.parse(filepath.read_text(errors="replace"))
    except SyntaxError:
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("ontic."):
                    parts = alias.name.split(".")
                    if len(parts) >= 2:
                        imports.add(parts[1])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("ontic."):
                parts = node.module.split(".")
                if len(parts) >= 2:
                    imports.add(parts[1])
    return imports


def build_dependency_graph(
    group_only: bool = False,
) -> tuple[set[str], set[tuple[str, str]]]:
    """Walk ontic/ and build (nodes, edges) dependency graph.

    If group_only=True, nodes are group names and edges are group→group.
    Otherwise, nodes are top-level submodule names.
    """
    # module → set of modules it imports
    deps: dict[str, set[str]] = defaultdict(set)
    all_modules: set[str] = set()

    for py_file in ONTIC_ROOT.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        # Determine the top-level submodule
        rel = py_file.relative_to(ONTIC_ROOT)
        parts = rel.parts
        if len(parts) == 1:
            # Root-level file (e.g. __init__.py, benchmark_runner.py)
            mod_name = parts[0].replace(".py", "")
            if mod_name == "__init__":
                continue
        else:
            mod_name = parts[0]

        all_modules.add(mod_name)
        imported = parse_imports(py_file)
        for imp in imported:
            if imp != mod_name:  # skip self-imports
                deps[mod_name].add(imp)
                all_modules.add(imp)

    if group_only:
        group_nodes: set[str] = set()
        group_edges: set[tuple[str, str]] = set()
        for mod, targets in deps.items():
            src_group = get_group(mod)
            group_nodes.add(src_group)
            for tgt in targets:
                tgt_group = get_group(tgt)
                if src_group != tgt_group:
                    group_nodes.add(tgt_group)
                    group_edges.add((src_group, tgt_group))
        return group_nodes, group_edges

    nodes = all_modules
    edges: set[tuple[str, str]] = set()
    for mod, targets in deps.items():
        for tgt in targets:
            edges.add((mod, tgt))
    return nodes, edges


def render_mermaid(
    nodes: set[str],
    edges: set[tuple[str, str]],
    group_only: bool = False,
) -> str:
    """Render dependency graph as Mermaid flowchart."""
    lines = ["```mermaid", "flowchart LR"]

    if group_only:
        for node in sorted(nodes):
            color = GROUP_COLORS.get(node, "#9ca3af")
            lines.append(f"    {node}[{node}]")
            lines.append(f"    style {node} fill:{color},color:#fff")
    else:
        # Subgraphs by group
        groups: dict[str, list[str]] = defaultdict(list)
        for node in sorted(nodes):
            groups[get_group(node)].append(node)
        for group_name in sorted(groups):
            color = GROUP_COLORS.get(group_name, "#9ca3af")
            lines.append(f"    subgraph {group_name}")
            for mod in sorted(groups[group_name]):
                lines.append(f"        {mod}[{mod}]")
                lines.append(f"        style {mod} fill:{color},color:#fff")
            lines.append("    end")

    for src, tgt in sorted(edges):
        lines.append(f"    {src} --> {tgt}")

    lines.append("```")
    return "\n".join(lines)


def render_dot(
    nodes: set[str],
    edges: set[tuple[str, str]],
    group_only: bool = False,
) -> str:
    """Render as Graphviz DOT."""
    lines = ['digraph ontic {', '    rankdir=LR;', '    node [shape=box, style=filled];']

    if group_only:
        for node in sorted(nodes):
            color = GROUP_COLORS.get(node, "#9ca3af")
            lines.append(f'    "{node}" [fillcolor="{color}", fontcolor=white];')
    else:
        groups: dict[str, list[str]] = defaultdict(list)
        for node in sorted(nodes):
            groups[get_group(node)].append(node)
        for group_name in sorted(groups):
            color = GROUP_COLORS.get(group_name, "#9ca3af")
            lines.append(f'    subgraph cluster_{group_name} {{')
            lines.append(f'        label="{group_name}";')
            lines.append(f'        style=filled; color="{color}20";')
            for mod in sorted(groups[group_name]):
                lines.append(f'        "{mod}" [fillcolor="{color}", fontcolor=white];')
            lines.append("    }")

    for src, tgt in sorted(edges):
        lines.append(f'    "{src}" -> "{tgt}";')

    lines.append("}")
    return "\n".join(lines)


def render_svg(dot_source: str) -> bytes:
    """Render DOT to SVG via graphviz Python package."""
    try:
        import graphviz  # type: ignore[import-untyped]
    except ImportError:
        print(
            "ERROR: graphviz package required for SVG output.\n"
            "  pip install graphviz\n"
            "  Also install system graphviz: apt install graphviz",
            file=sys.stderr,
        )
        sys.exit(1)

    src = graphviz.Source(dot_source, format="svg")
    return src.pipe()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ontic dependency graph")
    parser.add_argument(
        "--format", choices=["mermaid", "dot", "svg"], default="mermaid",
        help="Output format (default: mermaid)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output file path (default: stdout for text formats)",
    )
    parser.add_argument(
        "--group-only", action="store_true",
        help="Show only group-level dependencies (collapsed view)",
    )
    args = parser.parse_args()

    nodes, edges = build_dependency_graph(group_only=args.group_only)
    print(f"Graph: {len(nodes)} nodes, {len(edges)} edges", file=sys.stderr)

    if args.format == "mermaid":
        result = render_mermaid(nodes, edges, group_only=args.group_only)
        if args.output:
            args.output.write_text(result)
            print(f"Written: {args.output}", file=sys.stderr)
        else:
            print(result)

    elif args.format == "dot":
        result = render_dot(nodes, edges, group_only=args.group_only)
        if args.output:
            args.output.write_text(result)
            print(f"Written: {args.output}", file=sys.stderr)
        else:
            print(result)

    elif args.format == "svg":
        dot = render_dot(nodes, edges, group_only=args.group_only)
        svg = render_svg(dot)
        if args.output:
            args.output.write_bytes(svg)
            print(f"Written: {args.output}", file=sys.stderr)
        else:
            sys.stdout.buffer.write(svg)


if __name__ == "__main__":
    main()
