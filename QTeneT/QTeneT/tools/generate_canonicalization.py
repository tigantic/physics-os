#!/usr/bin/env python3
"""Generate canonicalization maps and enterprise specs for QTenet.

Inputs
- inventory/qtt_repo_index.json (generated from HyperTensor-VM-main)

Outputs
- docs/09_CANONICALIZATION_MAP.md
- docs/specs/*
- docs/10_CLI_CONTRACT.md

Heuristics
- Prefer non-archive over archive
- Prefer core/genesis/cfd/fluidelite/sdk categories over demos/benchmarks/qtt-misc/other
- Prefer tensornet/ and fluidelite/ over root scripts
- Within matches, prefer shorter path depth (often indicates library module vs demo)

This is a *productization* artifact: it does not claim semantic equivalence.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass

ROOT = os.environ.get("QTENET_ROOT", "/home/brad/clawd/QTeneT")
INP = os.path.join(ROOT, "inventory/qtt_repo_index.json")


def load():
    with open(INP, "r", encoding="utf-8") as f:
        return json.load(f)


def score_record(rec: dict) -> int:
    """Higher is better."""
    p = rec["path"].replace('\\','/').lower()
    cat = rec.get("category", "other")

    score = 0

    # Category preference
    cat_weight = {
        "core": 100,
        "genesis": 95,
        "tci": 90,
        "cfd": 85,
        "fluidelite": 80,
        "sdk": 75,
        "gpu": 70,
        "compressor": 60,  # separate product; still legit
        "oracle": 55,
        "benchmarks": 30,
        "demos": 20,
        "qtt-misc": 10,
        "other": 0,
        "archive": -50,
    }
    score += cat_weight.get(cat, 0)

    # Penalize archived
    if p.startswith("archive/") or p.startswith("_archived"):
        score -= 200

    # Prefer package-like locations
    if p.startswith("tensornet/"):
        score += 50
    if p.startswith("fluidelite/"):
        score += 35
    if p.startswith("sdk/"):
        score += 25

    # Prefer not root scripts
    if "/" not in p:
        score -= 10

    # Prefer shallower paths (library vs deeply nested tools)
    score -= max(0, p.count("/") - 2) * 2

    # Prefer python source over docs
    if rec.get("lang") == "py":
        score += 10

    return score


def build_symbol_index(records: list[dict]):
    """Return symbol -> list of (rec, sym)."""
    sym_index: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for rec in records:
        if rec.get("lang") != "py":
            continue
        for sym in rec.get("symbols") or []:
            name = sym.get("name")
            if not name:
                continue
            sym_index[name].append((rec, sym))
    return sym_index


def choose_canonical(entries: list[tuple[dict, dict]]):
    """Pick best entry for a symbol."""
    best = None
    best_score = -10**9
    for rec, sym in entries:
        s = score_record(rec)
        # slight preference for classes (public API anchoring)
        if sym.get("kind") == "class":
            s += 2
        if s > best_score:
            best_score = s
            best = (rec, sym, s)
    return best


def write(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def md(s: str) -> str:
    return s.replace("`", "\\`")


def main():
    idx = load()
    records = idx["records"]

    sym_index = build_symbol_index(records)

    # Filter to "QTT-ish" symbol names to avoid gigantic noise.
    # Keep anything with qtt/tt/tci/mps/mpo/rank/round/shift/laplace/euler/navier/sinkhorn/wasserstein/etc.
    keep_rx = re.compile(
        r"qtt|tt|tci|mps|mpo|rank|round|truncate|shift|roll|laplac|grad|div|hadam|walsh|euler|navier|poisson|imex|tdvp|sinkhorn|wasser|bary|rkhs|kernel|homolog|clifford|multivector|tropical|random|resolvent",
        re.IGNORECASE,
    )

    kept = {k: v for k, v in sym_index.items() if keep_rx.search(k)}

    canon_rows = []
    for name, entries in kept.items():
        if len(entries) < 2:
            continue
        best = choose_canonical(entries)
        if not best:
            continue
        rec, sym, sc = best
        canon_rows.append((name, rec["path"], sym.get("kind"), sym.get("lineno"), sc, entries))

    canon_rows.sort(key=lambda t: (-len(t[5]), t[0].lower()))

    out = []
    out.append("# Canonicalization Map (Draft)\n")
    out.append("This document resolves **multiple implementations** of similarly-named QTT/TT/TCI primitives across the upstream monorepo into a **single preferred canonical location** for product packaging.\n")
    out.append("Rules are heuristic and *meant to be edited*. The goal is a clean, enterprise API boundary, not perfect semantic proof.\n")

    out.append("## Scoring heuristics (current)\n")
    out.append("Preference order (high → low): `tensornet/core` / `tensornet/genesis` / `tci` / `tensornet/cfd` / `fluidelite` / `sdk` / `compressor` (separate product) / demos & benchmarks / archived.\n")

    out.append("## Symbol-level canonical picks\n")
    out.append("Each entry lists: canonical path + all known alternates.\n")

    for name, path, kind, lineno, sc, entries in canon_rows[:600]:
        out.append(f"### `{name}` ({len(entries)} implementations)\n")
        out.append(f"**Canonical:** `{path}` — `{kind}` at L{lineno} (score {sc})\n")
        out.append("**Alternates:**\n")
        # sort alternates by score
        alts = sorted(entries, key=lambda e: -score_record(e[0]))
        for rec, sym in alts:
            out.append(f"- `{rec['path']}` — {sym.get('kind')} L{sym.get('lineno')} (cat={rec.get('category')})")
        out.append("\n")

    if len(canon_rows) > 600:
        out.append(f"*(Truncated: showing 600 of {len(canon_rows)} multi-impl symbols. See inventory for full detail.)*\n")

    write(os.path.join(ROOT, "docs/09_CANONICALIZATION_MAP.md"), "\n".join(out))

    # Specs: rank control, operator versioning, provenance
    write(
        os.path.join(ROOT, "docs/specs/01_RANK_CONTROL.md"),
        """# Spec: Rank Control (Product Invariant)

## Objective
Rank control is the core product invariant that makes QTT/TT operational at scale.

## Definitions
- **TT-ranks**: internal bond dimensions between cores.
- **Rank explosion**: uncontrolled rank growth after operations (apply/contract/shift/etc.).

## Requirements
1. Every rank-increasing operation MUST have an immediate, explicit **rounding/truncation step** available.
2. Any public API that can grow rank MUST accept:
   - `eps` (relative tolerance) and/or `max_rank` (hard cap)
3. Implementations SHOULD expose diagnostics:
   - pre/post ranks, truncation error, time cost
4. Never Go Dense is the default: dense reconstruction is an escape hatch.

## Recommended API knobs
- `eps: float` (default e.g. `1e-6`)
- `max_rank: int | None`
- `min_rank: int | None` (rare)
- `rounding: Literal['svd','qr_svd','randomized']`

## Test obligations
- Property: `round(round(x)) ≈ round(x)` (idempotence within tolerance)
- Property: `||x - round(x)|| <= eps * ||x||` (within method limits)
- Golden: known tensors/functions where expected ranks are known
""",
    )

    write(
        os.path.join(ROOT, "docs/specs/02_OPERATOR_VERSIONING.md"),
        """# Spec: Operator Versioning

## Objective
Operators (MPO builders) must be **versioned** so solver results remain reproducible and debuggable.

## Operator identity
An operator is identified by:
- `name` (e.g., `laplacian`, `shift`, `gradient`)
- `scheme` (e.g., `cd2`, `upwind1`, `weno5`)
- `version` (semantic, e.g., `v1`, `v2`)
- `grid_layout` (binary quantics layout assumptions)

## Requirements
1. Each operator builder must return metadata capturing the identity.
2. Solver runs must emit operator identities into a run manifest.

## Suggested structure
- `qtenet.operators.laplacian(..., scheme='cd2', version='v1')`
- returns `(mpo, meta)` where `meta` is JSON-serializable.

## Backward compatibility
- `v1` operators must remain available once shipped.
- breaking changes require new `version`.
""",
    )

    write(
        os.path.join(ROOT, "docs/specs/03_PROVENANCE_ATTESTATION.md"),
        """# Spec: Provenance & Attestation

## Objective
Enterprise-grade reproducibility and audit trails for QTT computations.

## Minimum run manifest
Every non-trivial execution SHOULD produce a manifest:
- code version identifiers (git sha when available)
- environment: Python, torch, CUDA availability
- operator identities (see operator versioning)
- rank control parameters (eps/max_rank)
- input dataset identifiers (hash/path)
- output identifiers (hash/path)

## Attestations
Where upstream uses attestation JSONs, QTeneT treats them as first-class artifacts.

## Determinism
- Global seed controls must be centralized.
- Non-deterministic GPU ops must be disclosed in manifests.
""",
    )

    write(
        os.path.join(ROOT, "docs/specs/04_API_STABILITY.md"),
        """# Spec: API Stability Contract

## Objective
Define what enterprise users can rely on.

## Stability tiers
- **Stable**: `qtenet.sdk.*`
- **Experimental**: everything else (until promoted)

## Promotion criteria (experimental → stable)
- docstring + docs page
- deterministic behavior defined (or explicitly non-deterministic)
- tests: golden + property
- performance envelope documented

## Deprecation policy
- Deprecations require warnings for at least one minor release.
- Removals only in major releases.
""",
    )

    # CLI contract
    write(
        os.path.join(ROOT, "docs/10_CLI_CONTRACT.md"),
        """# CLI Contract: `qtenet`

This document defines a stable, enterprise-grade CLI contract. Commands may remain thin wrappers initially.

## Principles
- Machine-readable outputs by default (`--json`).
- Deterministic exit codes.
- Explicit escape hatches for densification.

## Commands

### `qtenet inventory`
Print the location of the repo index inventory.

- Output: path string
- Exit codes: 0 success

### `qtenet inspect <artifact>`
Inspect a QTT artifact/container.

- `--json` emits metadata.

### `qtenet compress ...`
Facade entrypoint for compression.

Notes:
- The_Compressor remains a separate product; this command may delegate to it or to other codecs.

### `qtenet query <artifact> <index>`
Point query.

### `qtenet reconstruct <artifact>`
Dense escape hatch.

- MUST require explicit confirmation flag: `--allow-dense`.

### `qtenet benchmark`
Run benchmark suites.

## Output schema (`--json`)
All commands that emit JSON should follow:

```json
{
  "tool": "qtenet",
  "command": "query",
  "status": "ok",
  "meta": {},
  "result": {}
}
```
""",
    )

    print("Generated canonicalization + specs + CLI contract")


if __name__ == "__main__":
    main()
