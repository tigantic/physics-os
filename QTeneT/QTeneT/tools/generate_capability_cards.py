#!/usr/bin/env python3
"""Generate capability cards for QTeneT.

Outputs:
- docs/12_CAPABILITY_CARDS_INDEX.md
- docs/capabilities/<capability-id>.md

Cards are enterprise-facing packaging units: contract, invariants, metrics,
canonical upstream sources, and proposed qtenet namespaces.

This is a docs generator; implementations are wired later.
"""

from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass

ROOT = os.environ.get("QTENET_ROOT", "/home/brad/clawd/QTeneT")
INP = os.path.join(ROOT, "inventory/qtt_repo_index.json")
OUT_DIR = os.path.join(ROOT, "docs/capabilities")
INDEX_OUT = os.path.join(ROOT, "docs/12_CAPABILITY_CARDS_INDEX.md")


def load_records():
    with open(INP, "r", encoding="utf-8") as f:
        return json.load(f)["records"]


def score(rec: dict) -> int:
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
        "compressor": 40,
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

    s -= max(0, p.count("/") - 2) * 2

    if rec.get("lang") == "py":
        s += 10

    return s


def record_text_signature(rec: dict) -> str:
    syms = rec.get("symbols") or []
    sym_names = " ".join(s.get("name", "") for s in syms)
    kws = " ".join(rec.get("keywords") or [])
    return f"{rec['path']} {rec.get('category','')} {kws} {sym_names}".lower()


def matches_any(path: str, pats: list[re.Pattern[str]]) -> bool:
    p = path.replace('\\','/').lower()
    return any(r.search(p) for r in pats)


@dataclass(frozen=True)
class Capability:
    id: str
    title: str
    target_namespace: str
    include: list[re.Pattern[str]]
    include_text: list[re.Pattern[str]]
    exclude: list[re.Pattern[str]]
    notes: str
    contract: str
    invariants: list[str]
    observability: list[str]
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
            contract="Defines the canonical runtime objects representing QTT/TT tensors and linear operators.",
            invariants=[
                "Never Go Dense by default",
                "Tensor metadata must be carried (dims/ranks/layout/dtype)",
                "All ops must be rank-controlled (eps/max_rank)",
            ],
            observability=[
                "Expose ranks before/after ops",
                "Emit truncation error and time cost",
            ],
        ),
        Capability(
            id="decomposition-rounding",
            title="Decomposition & Rounding: TT-SVD / truncation / rank control",
            target_namespace="qtenet.core.decomposition",
            include=[R(r"decomp|truncate|round|svd"), R(r"^tensornet/core/")],
            include_text=[R(r"tt-?svd|round|truncate|max_rank|eps")],
            exclude=[R(r"archive"), R(r"_archived")],
            notes="Choose one canonical decomposition/rounding stack, then wrap everything else behind it.",
            contract="Provides deterministic decomposition and rounding APIs used by everything else.",
            invariants=[
                "Deterministic where possible (or disclosed non-determinism)",
                "Idempotence under repeated rounding (within tolerance)",
                "No silent densification",
            ],
            observability=[
                "Log SVD mode (full/randomized)",
                "Record truncation error + ranks",
            ],
        ),
        Capability(
            id="tci-ttcross",
            title="Construction: TCI / TT-Cross",
            target_namespace="qtenet.tci",
            include=[R(r"^tci_core_rust/"), R(r"^tci_llm/"), R(r"tci")],
            include_text=[R(r"\btci\b|tt\s*-?cross|maxvol|sweep")],
            exclude=[R(r"deprecated"), R(r"_archived")],
            notes="Black-box function → QTT construction. Includes sampling policies and maxvol/skeleton selection.",
            contract="Build QTT representations from black-box functions or samples with evaluation budgets.",
            invariants=[
                "Bound evaluations (n_evals tracked)",
                "Rank selection is explicit and auditable",
            ],
            observability=[
                "Emit n_evals, convergence/sweep stats",
                "Emit skeleton indices / maxvol diagnostics",
            ],
        ),
        Capability(
            id="operators",
            title="Operators Library (MPO builders)",
            target_namespace="qtenet.operators",
            include=[R(r"^tensornet/cfd/"), R(r"shift|roll|hadamard|walsh|spectral|laplac|grad|div|advec|diffus|poisson")],
            include_text=[R(r"operator|mpo|shift|roll|laplac|gradient|diverg|hadamard|walsh")],
            exclude=[R(r"demo"), R(r"benchmark")],
            notes="Versioned operator builders returning MPO + metadata.",
            contract="Expose a stable set of operator builders with scheme/version metadata.",
            invariants=[
                "Operators are versioned identities (name/scheme/version)",
                "Operators must be composable without densification",
            ],
            observability=[
                "Operator meta is JSON-serializable",
                "Record operator identities in run manifests",
            ],
            top_n=12,
        ),
        Capability(
            id="solvers",
            title="Solvers: Euler/NS/PDE pipelines (IMEX/TDVP)",
            target_namespace="qtenet.solvers",
            include=[R(r"^tensornet/cfd/"), R(r"euler|navier|imex|tdvp|riemann")],
            include_text=[R(r"euler|navier|stokes|imex|tdvp|weno|riemann")],
            exclude=[R(r"test"), R(r"benchmark")],
            notes="End-to-end solver layer. Enterprise contract: reproducibility + rank control + diagnostics.",
            contract="Provide solver entrypoints with strict rank-control and provenance outputs.",
            invariants=[
                "Every timestep must apply rank control",
                "Conservation/diagnostics computed without densification when feasible",
            ],
            observability=[
                "Emit per-step ranks + truncation error",
                "Emit conservation deltas",
            ],
            top_n=14,
        ),
        Capability(
            id="genesis",
            title="Genesis Primitives (OT/SGW/RMT/Tropical/RKHS/PH/GA)",
            target_namespace="qtenet.genesis",
            include=[R(r"^tensornet/genesis/")],
            include_text=[R(r"sinkhorn|wasserstein|wavelet|wigner|tropical|rkhs|kernel|homology|clifford|multivector")],
            exclude=[],
            notes="Seven meta-primitives as product modules.",
            contract="Expose genesis modules with stable high-level APIs and explicit compute/memory envelopes.",
            invariants=[
                "No densification across pipeline stages",
                "Rank control explicitly applied between stages",
            ],
            observability=[
                "Per-stage compression ratio + ranks",
                "Emit pipeline manifest",
            ],
            top_n=14,
        ),
        Capability(
            id="gpu",
            title="GPU Acceleration: CUDA/Triton kernels and dispatch",
            target_namespace="qtenet.gpu",
            include=[R(r"^tensornet/cuda/"), R(r"^tensornet/gpu/"), R(r"triton")],
            include_text=[R(r"cuda|triton|kernel|gpu")],
            exclude=[R(r"test")],
            notes="Compute kernels, autotuning, GPU point-eval and apply operations.",
            contract="Provide a dispatch layer that selects CPU/GPU implementations deterministically.",
            invariants=[
                "Graceful CPU fallback",
                "GPU non-determinism disclosed",
            ],
            observability=[
                "Record backend selection (cpu/cuda/triton)",
                "Record kernel versions and autotune params",
            ],
            top_n=12,
        ),
        Capability(
            id="sdk",
            title="Enterprise SDK: stable facade (what users import)",
            target_namespace="qtenet.sdk",
            include=[R(r"^sdk/qtt-sdk/")],
            include_text=[R(r"qtt_sdk|sdk")],
            exclude=[],
            notes="The SDK layer defines the stable public surface.",
            contract="Stabilize naming, types, and error semantics for external users.",
            invariants=[
                "API stability contract enforced",
                "Errors are typed and actionable",
            ],
            observability=[
                "Emit JSON-friendly error payloads in CLI",
            ],
        ),
        Capability(
            id="integrations-oracle",
            title="Integration: Oracle slicing/encoding",
            target_namespace="qtenet.integrations.oracle",
            include=[R(r"^oracle/"), R(r"oracle_qtt|qtt_encoder|slicer")],
            include_text=[R(r"encoder|slicer|oracle")],
            exclude=[R(r"deprecated")],
            notes="QTT encoders/slicers used for oracle workflows.",
            contract="Provide stable adapters for oracle systems without contaminating core APIs.",
            invariants=[
                "Adapters do not own core tensor semantics",
            ],
            observability=[
                "Emit schema/version for encoded artifacts",
            ],
        ),
        Capability(
            id="provenance",
            title="Provenance & Attestation",
            target_namespace="qtenet.provenance",
            include=[R(r"attestation"), R(r"determinism")],
            include_text=[R(r"attestation|determinism|seed|manifest")],
            exclude=[R(r"archive")],
            notes="Run manifests, deterministic controls, attestations.",
            contract="Define and emit run manifests and attach attestation artifacts where available.",
            invariants=[
                "Every run has an identity",
                "Manifests are JSON-serializable",
            ],
            observability=[
                "Capture seeds, versions, operator identities",
            ],
        ),
    ]


def select(records: list[dict], cap: Capability) -> list[dict]:
    out = []
    for r in records:
        sig = record_text_signature(r)
        p = r["path"]
        if cap.exclude and matches_any(p, cap.exclude):
            continue
        if cap.include and not matches_any(p, cap.include):
            if cap.include_text and not any(rx.search(sig) for rx in cap.include_text):
                continue
        # avoid pulling compressor unless explicitly needed
        if r.get("category") == "compressor" and cap.id not in ("decomposition-rounding",):
            continue
        out.append(r)
    return sorted(out, key=lambda r: (-score(r), r.get('category',''), r['path']))


def write(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def render_card(cap: Capability, recs: list[dict]) -> str:
    top = recs[: cap.top_n]

    lines: list[str] = []
    lines.append(f"# {cap.title}\n")
    lines.append(f"**Capability ID:** `{cap.id}`\n")
    lines.append(f"**Target namespace:** `{cap.target_namespace}`\n")

    lines.append("## Summary\n")
    lines.append(cap.contract + "\n")

    lines.append("## Notes\n")
    lines.append(cap.notes + "\n")

    lines.append("## Product invariants\n")
    for inv in cap.invariants:
        lines.append(f"- {inv}")

    lines.append("\n## Observability (enterprise requirements)\n")
    for ob in cap.observability:
        lines.append(f"- {ob}")

    lines.append("\n## Canonical upstream sources (recommended top picks)\n")
    for r in top:
        lines.append(f"- `{r['path']}` (cat={r.get('category')}, lang={r.get('lang')}, score={score(r)})")

    lines.append("\n## Candidate set (ranked, truncated)\n")
    for r in recs[:50]:
        lines.append(f"- `{r['path']}` (cat={r.get('category')}, score={score(r)})")
    if len(recs) > 50:
        lines.append(f"- *(+{len(recs)-50} more candidates in inventory)*")

    lines.append("\n## Required tests (draft)\n")
    lines.append("- Golden tests (known tensors/known ranks)\n- Property tests (idempotence, tolerance bounds)\n- No-dense guards where applicable\n")

    lines.append("## Promotion checklist (experimental → stable)\n")
    lines.append("- [ ] Stable API defined in `qtenet.sdk`\n- [ ] Doc page exists\n- [ ] Determinism documented\n- [ ] Tests exist\n- [ ] Benchmark envelope documented\n")

    return "\n".join(lines) + "\n"


def main():
    records = load_records()
    caps = cap_defs()

    os.makedirs(OUT_DIR, exist_ok=True)

    index_lines = ["# Capability Cards\n", "These are enterprise-facing capability definitions for QTeneT.\n"]
    index_lines.append("## Index\n")

    for cap in caps:
        recs = select(records, cap)
        card_path = os.path.join(OUT_DIR, f"{cap.id}.md")
        write(card_path, render_card(cap, recs))
        index_lines.append(f"- [{cap.title}](capabilities/{cap.id}.md)")

    write(INDEX_OUT, "\n".join(index_lines) + "\n")
    print(f"Wrote {INDEX_OUT} and {len(caps)} capability cards")


if __name__ == "__main__":
    main()
