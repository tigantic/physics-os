#!/usr/bin/env python3
"""
Generate a Markdown dashboard from the capability ledger.

Output: ledger/DASHBOARD.md

Usage:
  python3 ledger/generate_dashboard.py
"""

from __future__ import annotations

import pathlib
from datetime import datetime
from typing import Any

try:
    import yaml
except ImportError:
    raise SystemExit("PyYAML required: pip install pyyaml")

ROOT = pathlib.Path(__file__).resolve().parent.parent
LEDGER = ROOT / "ledger"
NODES_DIR = LEDGER / "nodes"
OUTPUT = LEDGER / "DASHBOARD.md"

STATE_ORDER = ["V0.0", "V0.1", "V0.2", "V0.3", "V0.4", "V0.5", "V0.6", "V1.0"]
PACK_ORDER = [
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
    "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX",
]


def load_nodes() -> list[dict[str, Any]]:
    nodes = []
    for f in sorted(NODES_DIR.glob("*.yaml")):
        data = yaml.safe_load(f.read_text())
        if isinstance(data, dict):
            nodes.append(data)
    return nodes


def state_badge(state: str) -> str:
    """Return a visual badge for a V-state."""
    badges = {
        "V0.0": "⬜",
        "V0.1": "🟫",
        "V0.2": "🟨",
        "V0.3": "🟧",
        "V0.4": "🟩",
        "V0.5": "🟦",
        "V0.6": "🟪",
        "V1.0": "✅",
    }
    return badges.get(state, "❓")


def tier_label(tier: str) -> str:
    labels = {"A": "**A**", "B": "B", "C": "C"}
    return labels.get(tier, tier)


def generate_dashboard(nodes: list[dict[str, Any]]) -> str:
    lines: list[str] = []

    # Header
    lines.append("# Capability Ledger Dashboard")
    lines.append("")
    lines.append(f"*Auto-generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC*")
    lines.append("")

    # Overall summary
    total = len(nodes)
    by_state: dict[str, int] = {}
    by_tier: dict[str, int] = {}
    by_pack: dict[str, list[dict]] = {}

    for n in nodes:
        s = n.get("state", "V0.0")
        t = n.get("tier", "B")
        p = n.get("pack", "?")
        by_state[s] = by_state.get(s, 0) + 1
        by_tier[t] = by_tier.get(t, 0) + 1
        by_pack.setdefault(p, []).append(n)

    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total nodes | **{total}** |")
    lines.append(f"| Tier A | {by_tier.get('A', 0)} |")
    lines.append(f"| Tier B | {by_tier.get('B', 0)} |")
    lines.append(f"| Tier C | {by_tier.get('C', 0)} |")
    lines.append("")

    # State distribution bar chart
    lines.append("## State Distribution")
    lines.append("")
    lines.append("| State | Count | Bar |")
    lines.append("|-------|------:|-----|")
    for s in STATE_ORDER:
        count = by_state.get(s, 0)
        bar = "█" * (count // 2) if count > 0 else ""
        badge = state_badge(s)
        lines.append(f"| {badge} {s} | {count} | {bar} |")
    lines.append("")

    # Maturity progress
    at_v02_plus = sum(v for k, v in by_state.items() if k >= "V0.2")
    at_v04_plus = sum(v for k, v in by_state.items() if k >= "V0.4")
    at_v06_plus = sum(v for k, v in by_state.items() if k >= "V0.6")
    at_v10 = by_state.get("V1.0", 0)

    lines.append("## Maturity Progress")
    lines.append("")
    lines.append(f"| Milestone | Count | Percent |")
    lines.append(f"|-----------|------:|--------:|")
    lines.append(f"| ≥ V0.2 Correctness | {at_v02_plus} | {at_v02_plus * 100 // total}% |")
    lines.append(f"| ≥ V0.4 Validated | {at_v04_plus} | {at_v04_plus * 100 // total}% |")
    lines.append(f"| ≥ V0.6 Accelerated | {at_v06_plus} | {at_v06_plus * 100 // total}% |")
    lines.append(f"| = V1.0 Stable | {at_v10} | {at_v10 * 100 // total}% |")
    lines.append("")

    # Tier A focus
    tier_a = [n for n in nodes if n.get("tier") == "A"]
    lines.append("## Tier A Nodes (Priority)")
    lines.append("")
    lines.append("| ID | Name | State | Benchmarks | QTT |")
    lines.append("|----|------|-------|------------|-----|")
    for n in sorted(tier_a, key=lambda x: x["id"]):
        bench = len(n.get("benchmarks", []) or [])
        qtt = "✓" if n.get("qtt_hooks") else "—"
        badge = state_badge(n["state"])
        lines.append(f"| {n['id']} | {n['name']} | {badge} {n['state']} | {bench} | {qtt} |")
    lines.append("")

    # Per-pack tables
    lines.append("## Per-Pack Detail")
    lines.append("")

    pack_names: dict[str, str] = {}
    for n in nodes:
        pack_names[n["pack"]] = n.get("pack_name", n["pack"])

    for pk in PACK_ORDER:
        if pk not in by_pack:
            continue
        pack_nodes = by_pack[pk]
        pname = pack_names.get(pk, pk)
        lines.append(f"### Pack {pk}: {pname}")
        lines.append("")
        lines.append("| ID | Name | Tier | State | Tests | Benchmarks |")
        lines.append("|----|------|------|-------|-------|------------|")
        for n in sorted(pack_nodes, key=lambda x: x["id"]):
            tier = tier_label(n.get("tier", "B"))
            badge = state_badge(n["state"])
            tests = n.get("tests", {}) or {}
            test_count = sum(len(v) if isinstance(v, list) else 0 for v in tests.values())
            bench = len(n.get("benchmarks", []) or [])
            lines.append(
                f"| {n['id']} | {n['name']} | {tier} | {badge} {n['state']} "
                f"| {test_count} | {bench} |"
            )
        lines.append("")

    # Legend
    lines.append("## Legend")
    lines.append("")
    lines.append("| Badge | State | Meaning |")
    lines.append("|-------|-------|---------|")
    meanings = {
        "V0.0": "Draft — spec only",
        "V0.1": "Scaffolded — runs on toy case",
        "V0.2": "Correctness — reproduces reference solutions",
        "V0.3": "Verified — deterministic, CI green",
        "V0.4": "Validated — benchmark match quantified",
        "V0.5": "Optimized — performance engineering done",
        "V0.6": "Accelerated — QTT/TN integrated",
        "V1.0": "Stable — API frozen, docs complete",
    }
    for s in STATE_ORDER:
        lines.append(f"| {state_badge(s)} | {s} | {meanings.get(s, '')} |")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    nodes = load_nodes()
    if not nodes:
        raise SystemExit(f"No node files found in {NODES_DIR}")

    dashboard = generate_dashboard(nodes)
    OUTPUT.write_text(dashboard)
    print(f"Dashboard generated: {OUTPUT} ({len(nodes)} nodes)")


if __name__ == "__main__":
    main()
