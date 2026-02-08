#!/usr/bin/env python3
"""Capability-level canonicalization for QTeneT.

Consumes inventory/qtt_repo_index.json and produces:
- docs/11_CAPABILITY_MAP.md

A "capability" is an enterprise packaging unit (what a product team ships):
- Decomposition & rounding
- TCI / TT-Cross construction
- Operators (shift/derivative/laplacian/spectral)
- Solvers (Euler/NS/time integrators)
- Genesis primitives
- GPU acceleration
- SDK facade
- Provenance/attestation

For each capability, we:
- find candidate upstream modules/files
- rank them with a heuristic score
- pick a small canonical set (top N)
- propose a target module namespace inside qtenet

This document is meant to be edited; it is a production packaging map.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass

ROOT = os.environ.get("QTENET_ROOT", "/home/brad/clawd/QTeneT")
INP = os.path.join(ROOT, "inventory/qtt_repo_index.json")
OUT = os.path.join(ROOT, "docs/11_CAPABILITY_MAP.md")


def load_records():
    with open(INP, "r", encoding="utf-8") as f:
        return json.load(f)["records"]


def score(rec: dict) -> int:
    """Higher is better."""
    p = rec["path"].replace('\\','/').lower()
    cat = rec.get("category", "other")

    s = 0
    cat_weight = {
        "core": 100,
        "genesis": 95,
        "tci": 90,
        "cfd": 85,
        "fluidelite": 80,
        "sdk": 75,
        "gpu": 70,
        "oracle": 55,
        "compressor": 40,  # separate product; valid but not canonical library
        "benchmarks": 20,
        "demos": 10,
        "qtt-misc": 5,
        "other": 0,
        "archive": -100,
    }
    s += cat_weight.get(cat, 0)

    if p.startswith("archive/") or p.startswith("_archived"):
        s -= 250

    if p.startswith("tensornet/"):
        s += 40
    if p.startswith("fluidelite/"):
        s += 25
    if p.startswith("sdk/"):
        s += 20

    # prefer library-ish depth
    s -= max(0, p.count("/") - 2) * 2

    # prefer py over md/json
    if rec.get("lang") == "py":
        s += 10

    return s


def matches_any(path: str, pats: list[re.Pattern[str]]) -> bool:
    p = path.replace('\\','/').lower()
    return any(r.search(p) for r in pats)


def record_text_signature(rec: dict) -> str:
    # For capability matching, use path + symbol names + keywords.
    syms = rec.get("symbols") or []
    sym_names = " ".join(s.get("name", "") for s in syms)
    kws = " ".join(rec.get("keywords") or [])
    return f"{rec['path']} {rec.get('category','')} {kws} {sym_names}".lower()


@dataclass(frozen=True)
class Capability:
    id: str
    title: str
    target_namespace: str
    include: list[re.Pattern[str]]
    include_text: list[re.Pattern[str]]
    exclude: list[re.Pattern[str]]
    notes: str
    top_n: int = 8


def cap_defs() -> list[Capability]:
    R = lambda s: re.compile(s, re.IGNORECASE)

    return [
        Capability(
            id="core-types",
            title="Core Types: QTTTensor / TT cores / MPS / MPO",
            target_namespace="qtenet.core",
            include=[R(r"^tensornet/(core|mps|mpo)/"), R(r"^fluidelite/core/")],
            include_text=[R(r"mps|mpo|tt|qtt")],
            exclude=[R(r"test"), R(r"bench"), R(r"demo")],
            notes="Canonical tensor network types and containers. This is the foundation layer.",
        ),
        Capability(
            id="decomposition-rounding",
            title="Decomposition & Rounding: TT-SVD / truncation / rank control",
            target_namespace="qtenet.core.decomposition",
            include=[R(r"decomp|truncate|round|svd"), R(r"^tensornet/core/"), R(r"^The_Compressor/")],
            include_text=[R(r"tt-?svd|round|truncate|max_rank|eps")],
            exclude=[R(r"archive"), R(r"_archived")],
            notes="Choose one canonical decomposition/rounding stack, then wrap everything else behind it.",
        ),
        Capability(
            id="tci-ttcross",
            title="Construction: TCI / TT-Cross",
            target_namespace="qtenet.tci",
            include=[R(r"^tci_core_rust/"), R(r"^tci_llm/"), R(r"tci"), R(r"tt\s*-?cross")],
            include_text=[R(r"\btci\b|tt\s*-?cross|maxvol|sweep")],
            exclude=[R(r"deprecated"), R(r"_archived")],
            notes="Black-box function → QTT construction. Includes sampling policies and maxvol/skeleton selection.",
        ),
        Capability(
            id="operators-shift-spectral",
            title="Operators: shift/roll, Hadamard/Walsh-Hadamard, spectral",
            target_namespace="qtenet.operators",
            include=[R(r"^tensornet/cfd/"), R(r"shift|roll|hadamard|walsh|spectral")],
            include_text=[R(r"shift|roll|hadamard|walsh|spectral|fourier")],
            exclude=[R(r"demo"), R(r"benchmark")],
            notes="Pure operator library: return MPOs (or operator-like objects) + metadata (scheme/version).",
        ),
        Capability(
            id="operators-derivative-laplacian",
            title="Operators: gradient/divergence/Laplacian/advection/diffusion",
            target_namespace="qtenet.operators.pde",
            include=[R(r"^tensornet/cfd/"), R(r"laplac|grad|div|advec|diffus|poisson")],
            include_text=[R(r"laplac|gradient|diverg|advec|diffus|poisson")],
            exclude=[R(r"demo"), R(r"benchmark")],
            notes="MPO builders + application kernels used by PDE solvers.",
        ),
        Capability(
            id="solvers-pde-cfd",
            title="Solvers: Euler/NS/PDE pipelines (IMEX/TDVP)",
            target_namespace="qtenet.solvers",
            include=[R(r"^tensornet/cfd/"), R(r"euler|navier|imex|tdvp|riemann")],
            include_text=[R(r"euler|navier|stokes|imex|tdvp|weno|riemann")],
            exclude=[R(r"test"), R(r"benchmark")],
            notes="End-to-end solver layer. The enterprise contract here is reproducibility + rank control + diagnostics.",
            top_n=12,
        ),
        Capability(
            id="genesis-ot",
            title="Genesis: QTT Optimal Transport (OT)",
            target_namespace="qtenet.genesis.ot",
            include=[R(r"^tensornet/genesis/ot/")],
            include_text=[R(r"sinkhorn|wasserstein|barycenter|transport")],
            exclude=[],
            notes="OT primitives: distributions, cost matrices, Sinkhorn, Wasserstein distance.",
        ),
        Capability(
            id="genesis-sgw",
            title="Genesis: QTT Spectral Graph Wavelets (SGW)",
            target_namespace="qtenet.genesis.sgw",
            include=[R(r"^tensornet/genesis/sgw/")],
            include_text=[R(r"wavelet|laplacian|chebyshev")],
            exclude=[],
            notes="Graph Laplacian, wavelet kernels, Chebyshev approximations.",
        ),
        Capability(
            id="genesis-rmt",
            title="Genesis: QTT Random Matrix Theory (RMT)",
            target_namespace="qtenet.genesis.rmt",
            include=[R(r"^tensornet/genesis/rmt/")],
            include_text=[R(r"wigner|wishart|resolvent|spectral")],
            exclude=[],
            notes="Ensembles, resolvents, spectral density estimation.",
        ),
        Capability(
            id="genesis-tropical",
            title="Genesis: QTT Tropical Geometry",
            target_namespace="qtenet.genesis.tropical",
            include=[R(r"^tensornet/genesis/tropical/")],
            include_text=[R(r"tropical|min-plus|max-plus|floyd")],
            exclude=[],
            notes="Tropical semiring, shortest paths, tropical eigen computations.",
        ),
        Capability(
            id="genesis-rkhs",
            title="Genesis: QTT RKHS / Kernel Methods",
            target_namespace="qtenet.genesis.rkhs",
            include=[R(r"^tensornet/genesis/rkhs/")],
            include_text=[R(r"kernel|rkhs|mmd|gaussian\s+process")],
            exclude=[],
            notes="Kernel ridge, MMD, GP regression primitives.",
        ),
        Capability(
            id="genesis-ph",
            title="Genesis: QTT Persistent Homology",
            target_namespace="qtenet.genesis.topology",
            include=[R(r"^tensornet/genesis/topology/")],
            include_text=[R(r"homology|persistence|vietoris|rips|betti")],
            exclude=[],
            notes="Boundary operators, persistence diagrams, Betti numbers.",
        ),
        Capability(
            id="genesis-ga",
            title="Genesis: QTT Geometric Algebra (GA)",
            target_namespace="qtenet.genesis.ga",
            include=[R(r"^tensornet/genesis/ga/")],
            include_text=[R(r"clifford|multivector|geometric\s+product")],
            exclude=[],
            notes="Clifford algebra, multivectors stored in QTT.",
        ),
        Capability(
            id="gpu-accel",
            title="GPU Acceleration: CUDA/Triton kernels and dispatch",
            target_namespace="qtenet.gpu",
            include=[R(r"^tensornet/cuda/"), R(r"^tensornet/gpu/"), R(r"triton"), R(r"kernel")],
            include_text=[R(r"cuda|triton|kernel|gpu")],
            exclude=[R(r"test")],
            notes="Compute kernels, autotuning, GPU point-eval and apply operations.",
        ),
        Capability(
            id="sdk-facade",
            title="Enterprise SDK: stable facade (what users import)",
            target_namespace="qtenet.sdk",
            include=[R(r"^sdk/qtt-sdk/"), R(r"qtenet/sdk")],
            include_text=[R(r"sdk")],
            exclude=[],
            notes="The SDK layer should define the stable public surface.",
        ),
        Capability(
            id="oracle-slicing-encoding",
            title="Oracle utilities: QTT slicing / encoding",
            target_namespace="qtenet.integrations.oracle",
            include=[R(r"^oracle/"), R(r"oracle_qtt|qtt_encoder|slicer")],
            include_text=[R(r"encoder|slicer")],
            exclude=[R(r"deprecated")],
            notes="QTT encoders/slicers used for oracle-related workflows.",
        ),
        Capability(
            id="provenance-attestation",
            title="Provenance & Attestation artifacts",
            target_namespace="qtenet.provenance",
            include=[R(r"attestation"), R(r"determinism")],
            include_text=[R(r"attestation|determinism|seed")],
            exclude=[R(r"archive")],
            notes="Run manifests, deterministic controls, attestations.",
        ),
    ]


def select_candidates(records: list[dict], cap: Capability) -> list[dict]:
    out = []
    for r in records:
        sig = record_text_signature(r)
        p = r["path"]
        if cap.exclude and matches_any(p, cap.exclude):
            continue
        if cap.include and not matches_any(p, cap.include):
            # allow inclusion by text signature too
            if not any(rx.search(sig) for rx in cap.include_text):
                continue
        else:
            if cap.include_text and not any(rx.search(sig) for rx in cap.include_text):
                # if path matched, keep anyway
                pass
        # Avoid dragging in The_Compressor for canonical library, unless cap is explicit
        if r.get("category") == "compressor" and cap.id not in ("decomposition-rounding",):
            continue
        out.append(r)
    return out


def main():
    records = load_records()

    caps = cap_defs()

    # Map each record to at most one primary capability (highest score among matching caps)
    cap_to_recs: dict[str, list[dict]] = {c.id: [] for c in caps}
    for cap in caps:
        cand = select_candidates(records, cap)
        cap_to_recs[cap.id] = cand

    # Render
    lines: list[str] = []
    lines.append("# Capability Map (Draft)\n")
    lines.append("This document is the **enterprise packaging map** for QTeneT.\n")
    lines.append("It groups upstream implementations into **capabilities** and proposes:\n")
    lines.append("- canonical module targets inside `qtenet.*`\n")
    lines.append("- recommended upstream sources to wire first\n")
    lines.append("\nIt is intentionally opinionated, and should be edited as you unify implementations.\n")

    lines.append("## Capability index\n")
    for cap in caps:
        n = len(cap_to_recs.get(cap.id, []))
        lines.append(f"- [{cap.title}](#{cap.id}) — {n} candidate artifacts")

    for cap in caps:
        recs = cap_to_recs.get(cap.id, [])
        recs_sorted = sorted(recs, key=lambda r: (-score(r), r.get('category',''), r['path']))
        top = recs_sorted[: cap.top_n]

        lines.append("\n---\n")
        lines.append(f"\n## {cap.title}\n")
        lines.append(f"<a id=\"{cap.id}\"></a>\n")
        lines.append(f"**Target namespace:** `{cap.target_namespace}`\n")
        lines.append(f"**Notes:** {cap.notes}\n")

        if not recs_sorted:
            lines.append("No candidates found (check matching heuristics).\n")
            continue

        lines.append("### Recommended canonical upstream sources (top picks)\n")
        for r in top:
            sc = score(r)
            lines.append(f"- `{r['path']}` (cat={r.get('category')}, lang={r.get('lang')}, score={sc})")

        lines.append("\n### Candidate set (ranked)\n")
        for r in recs_sorted[: min(len(recs_sorted), 40)]:
            sc = score(r)
            lines.append(f"- `{r['path']}` (cat={r.get('category')}, score={sc})")
        if len(recs_sorted) > 40:
            lines.append(f"- *(+{len(recs_sorted)-40} more candidates)*")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
