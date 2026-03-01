#!/usr/bin/env python3
"""Triage the 'other' bucket from inventory/qtt_repo_index.json.

Outputs:
- docs/14_TAXONOMY_TRIAGE_OTHER.md
- inventory/other_triage.json

Goal:
- rank 'other' artifacts by QTT relevance signal
- propose reclassification targets (core/cfd/tci/genesis/gpu/fluidelite/sdk/oracle)
- propose explicit exclusions (docs-only, vendor, irrelevant keywords)

This is a Phase 2 due-diligence hygiene artifact.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict

ROOT = os.environ.get("QTENET_ROOT", "/home/brad/clawd/QTeneT")
INP = os.path.join(ROOT, "inventory/qtt_repo_index.json")
OUT_MD = os.path.join(ROOT, "docs/14_TAXONOMY_TRIAGE_OTHER.md")
OUT_JSON = os.path.join(ROOT, "inventory/other_triage.json")


def load_records():
    with open(INP, "r", encoding="utf-8") as f:
        return json.load(f)["records"]


def path_norm(p: str) -> str:
    return p.replace('\\', '/').lower()


def propose_category(path: str, text_sig: str) -> str | None:
    p = path_norm(path)

    # strong path-based rules
    if p.startswith("ontic/core/") or p.startswith("ontic/mps/") or p.startswith("ontic/mpo/"):
        return "core"
    if p.startswith("ontic/cfd/"):
        return "cfd"
    if p.startswith("ontic/genesis/"):
        return "genesis"
    if p.startswith("ontic/cuda/") or p.startswith("ontic/gpu/"):
        return "gpu"
    if p.startswith("fluidelite/"):
        return "fluidelite"
    if p.startswith("sdk/"):
        return "sdk"
    if p.startswith("tci_core_rust/") or p.startswith("tci_llm/"):
        return "tci"
    if p.startswith("oracle/"):
        return "oracle"

    # heuristic keyword-based
    if re.search(r"\btci\b|tt\s*-?cross|maxvol", text_sig):
        return "tci"
    if re.search(r"sinkhorn|wasserstein|homology|clifford|multivector|tropical|wavelet|rkhs|mmd", text_sig):
        return "genesis"
    if re.search(r"euler|navier|stokes|riemann|weno|poisson|laplac", text_sig):
        return "cfd"
    if re.search(r"cuda|triton|kernel|autotun", text_sig):
        return "gpu"
    if re.search(r"mps|mpo|tensor\s+train|tt-?svd|round|truncate", text_sig):
        return "core"

    return None


def relevance_score(rec: dict) -> int:
    """Rank how likely an 'other' record is actually QTT-relevant."""
    p = path_norm(rec["path"])
    lang = rec.get("lang")

    sig = text_signature(rec)

    s = 0
    # direct qtt signal
    if "qtt" in p:
        s += 60

    # keyword signals
    kw_hits = 0
    for pat, w in [
        (r"\bqtt\b", 40),
        (r"quantized\s+tensor\s+train", 40),
        (r"tt-?svd", 25),
        (r"tt\s*-?cross|maxvol|\btci\b", 25),
        (r"\bmpo\b|\bmps\b", 15),
        (r"tensor\s+train", 15),
        (r"rank|round|truncate", 10),
    ]:
        if re.search(pat, sig):
            s += w
            kw_hits += 1

    # Prefer source code
    if lang == "py":
        s += 10
    elif lang == "rs":
        s += 8
    elif lang == "md":
        s -= 5

    # Penalize obvious noise
    if p.startswith("node_modules/") or "/node_modules/" in p:
        s -= 200
    if p.startswith("clawdbot-main/"):
        s -= 100
    if p.startswith("apps/glass_cockpit/") and "qtt" not in p:
        s -= 20

    # Penalize archive
    if p.startswith("archive/") or p.startswith("_archived"):
        s -= 80

    # mild penalty if only one weak hit
    if kw_hits <= 1 and "qtt" not in p:
        s -= 10

    return s


def text_signature(rec: dict) -> str:
    syms = rec.get("symbols") or []
    sym_names = " ".join(s.get("name", "") for s in syms)
    kws = " ".join(rec.get("keywords") or [])
    return f"{rec['path']} {kws} {sym_names}".lower()


def main():
    records = load_records()
    others = [r for r in records if r.get("category") == "other"]

    triaged = []
    for r in others:
        sig = text_signature(r)
        proposed = propose_category(r["path"], sig)
        score = relevance_score(r)
        triaged.append({
            "path": r["path"],
            "lang": r.get("lang"),
            "score": score,
            "proposed_category": proposed,
            "keywords": r.get("keywords") or [],
            "n_symbols": len(r.get("symbols") or []) if r.get("lang") == "py" else None,
        })

    triaged.sort(key=lambda x: (-x["score"], x["path"]))

    # Aggregate summaries
    prop_counts = Counter(t["proposed_category"] or "unresolved" for t in triaged)
    lang_counts = Counter(t["lang"] for t in triaged)

    # Suggested actions
    # - Reclassify: score >= 50 and proposed_category not None
    # - Investigate: score between 25..49
    # - Exclude as noise: score < 0
    rec_reclass = [t for t in triaged if t["score"] >= 50 and t["proposed_category"]]
    rec_invest = [t for t in triaged if 25 <= t["score"] < 50]
    rec_exclude = [t for t in triaged if t["score"] < 0]

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "n_other": len(others),
            "language_breakdown": dict(lang_counts),
            "proposed_category_breakdown": dict(prop_counts),
            "recommendations": {
                "reclassify": rec_reclass[:200],
                "investigate": rec_invest[:200],
                "exclude": rec_exclude[:200],
            },
            "all": triaged,
        }, f, indent=2)

    # Markdown report
    md = []
    md.append("# Taxonomy Triage — `other` Bucket\n")
    md.append("**Classification:** Proprietary & Confidential\n")
    md.append("This report triages artifacts currently categorized as **`other`** in `inventory/qtt_repo_index.json`.\n")
    md.append("Goal: reduce due-diligence risk by reclassifying true QTT artifacts and explicitly excluding noise.\n")

    md.append("## Summary\n")
    md.append(f"- Total `other`: **{len(others)}**")
    md.append(f"- Language breakdown: `{dict(lang_counts)}`")
    md.append("\n### Proposed reclassification breakdown\n")
    for k, v in prop_counts.most_common():
        md.append(f"- **{k}**: {v}")

    md.append("\n## Action Plan\n")
    md.append("### 1) Auto-reclassify candidates (high confidence)\n")
    md.append(f"Criteria: score ≥ 50 and proposed_category present. Count: **{len(rec_reclass)}**\n")
    md.append("Top 50:\n")
    for t in rec_reclass[:50]:
        md.append(f"- `{t['path']}` → **{t['proposed_category']}** (score={t['score']}, lang={t['lang']})")

    md.append("\n### 2) Investigate candidates (medium confidence)\n")
    md.append(f"Criteria: 25 ≤ score < 50. Count: **{len(rec_invest)}**\n")
    md.append("Top 50:\n")
    for t in rec_invest[:50]:
        md.append(f"- `{t['path']}` (score={t['score']}, lang={t['lang']})")

    md.append("\n### 3) Exclude candidates (likely noise / non-QTT)\n")
    md.append(f"Criteria: score < 0. Count: **{len(rec_exclude)}**\n")
    md.append("Top 50:\n")
    for t in rec_exclude[:50]:
        md.append(f"- `{t['path']}` (score={t['score']}, lang={t['lang']})")

    md.append("\n## Notes\n")
    md.append("- This triage is heuristic; it should be followed by a manual review pass of the top-ranked 100–200 items.")
    md.append("- The canonicalization engine can be extended with manual overrides once the reclassification is complete.")
    md.append("- Full machine-readable output: `inventory/other_triage.json`\n")

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"Wrote {OUT_MD} and {OUT_JSON}")


if __name__ == "__main__":
    main()
