"""
Interactive Proof Dashboard
============================

Real-time monitoring and visualization of proof status across
all HyperTensor verification layers.

Provides:
- ProofStatus: per-proof state tracking
- ProofDashboard: aggregated proof registry with filtering
- DashboardReport: JSON/HTML summary generation
- CoverageMap: which modules have proofs, which need them
- Timeline: proof generation history
- Anomaly detection: regressions where proofs stop verifying
- Export to static HTML dashboard

This is item 4.10: Interactive proof dashboard.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Proof status
# ---------------------------------------------------------------------------

class Verdict(Enum):
    """Proof verification verdict."""

    VERIFIED = auto()
    FAILED = auto()
    TIMEOUT = auto()
    SKIPPED = auto()
    PENDING = auto()


class ProofLayer(Enum):
    """Which verification layer produced the proof."""

    LEAN4 = "Lean 4"
    COQ = "Coq"
    ISABELLE = "Isabelle/HOL"
    INTERVAL = "Interval Arithmetic"
    CERTIFICATE = "Certificate"
    PCC = "Proof-Carrying Code"
    RUNTIME = "Runtime Check"


@dataclass
class ProofStatus:
    """Status of a single proof artifact."""

    proof_id: str
    module: str
    claim: str
    layer: ProofLayer
    verdict: Verdict = Verdict.PENDING
    error_msg: str = ""
    wall_time_s: float = 0.0
    timestamp: float = 0.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    @property
    def is_ok(self) -> bool:
        return self.verdict == Verdict.VERIFIED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proof_id": self.proof_id,
            "module": self.module,
            "claim": self.claim,
            "layer": self.layer.value,
            "verdict": self.verdict.name,
            "error_msg": self.error_msg,
            "wall_time_s": self.wall_time_s,
            "timestamp": self.timestamp,
            "tags": self.tags,
        }


# ---------------------------------------------------------------------------
# Coverage map
# ---------------------------------------------------------------------------

@dataclass
class CoverageEntry:
    """Coverage info for a single module."""

    module: str
    total_claims: int
    verified: int
    failed: int
    pending: int

    @property
    def coverage_pct(self) -> float:
        return 100.0 * self.verified / max(self.total_claims, 1)


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

@dataclass
class Anomaly:
    """A regression where a proof stopped verifying."""

    proof_id: str
    module: str
    previous_verdict: Verdict
    current_verdict: Verdict
    detected_at: float = 0.0

    def __post_init__(self) -> None:
        if self.detected_at == 0.0:
            self.detected_at = time.time()


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

class ProofDashboard:
    """Aggregated proof verification dashboard."""

    def __init__(self) -> None:
        self._proofs: Dict[str, ProofStatus] = {}
        self._history: List[ProofStatus] = []
        self._anomalies: List[Anomaly] = []
        self._previous_verdicts: Dict[str, Verdict] = {}

    def register(self, proof: ProofStatus) -> None:
        """Register or update a proof status."""
        pid = proof.proof_id

        # Anomaly detection
        if pid in self._previous_verdicts:
            prev = self._previous_verdicts[pid]
            if prev == Verdict.VERIFIED and proof.verdict != Verdict.VERIFIED:
                self._anomalies.append(Anomaly(
                    proof_id=pid,
                    module=proof.module,
                    previous_verdict=prev,
                    current_verdict=proof.verdict,
                ))
                logger.warning(
                    "Proof regression: %s was %s, now %s",
                    pid, prev.name, proof.verdict.name,
                )

        self._previous_verdicts[pid] = proof.verdict
        self._proofs[pid] = proof
        self._history.append(proof)

    def batch_register(self, proofs: Sequence[ProofStatus]) -> None:
        for p in proofs:
            self.register(p)

    @property
    def proofs(self) -> Dict[str, ProofStatus]:
        return dict(self._proofs)

    @property
    def anomalies(self) -> List[Anomaly]:
        return list(self._anomalies)

    def filter_by_module(self, module: str) -> List[ProofStatus]:
        return [p for p in self._proofs.values() if p.module == module]

    def filter_by_layer(self, layer: ProofLayer) -> List[ProofStatus]:
        return [p for p in self._proofs.values() if p.layer == layer]

    def filter_by_verdict(self, verdict: Verdict) -> List[ProofStatus]:
        return [p for p in self._proofs.values() if p.verdict == verdict]

    def filter_by_tag(self, tag: str) -> List[ProofStatus]:
        return [p for p in self._proofs.values() if tag in p.tags]

    # -----------------------------------------------------------------------
    # Coverage
    # -----------------------------------------------------------------------

    def coverage_map(self) -> List[CoverageEntry]:
        """Compute per-module proof coverage."""
        modules: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "verified": 0, "failed": 0, "pending": 0}
        )

        for p in self._proofs.values():
            m = modules[p.module]
            m["total"] += 1
            if p.verdict == Verdict.VERIFIED:
                m["verified"] += 1
            elif p.verdict == Verdict.FAILED:
                m["failed"] += 1
            else:
                m["pending"] += 1

        return [
            CoverageEntry(
                module=mod,
                total_claims=counts["total"],
                verified=counts["verified"],
                failed=counts["failed"],
                pending=counts["pending"],
            )
            for mod, counts in sorted(modules.items())
        ]

    def overall_coverage(self) -> float:
        """Overall verification coverage percentage."""
        total = len(self._proofs)
        verified = sum(1 for p in self._proofs.values() if p.is_ok)
        return 100.0 * verified / max(total, 1)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        by_verdict: Dict[str, int] = defaultdict(int)
        by_layer: Dict[str, int] = defaultdict(int)
        by_module: Dict[str, int] = defaultdict(int)

        for p in self._proofs.values():
            by_verdict[p.verdict.name] += 1
            by_layer[p.layer.value] += 1
            by_module[p.module] += 1

        return {
            "total_proofs": len(self._proofs),
            "coverage_pct": self.overall_coverage(),
            "by_verdict": dict(by_verdict),
            "by_layer": dict(by_layer),
            "by_module": dict(by_module),
            "anomalies": len(self._anomalies),
            "history_length": len(self._history),
        }

    # -----------------------------------------------------------------------
    # Reports
    # -----------------------------------------------------------------------

    def to_json(self, indent: int = 2) -> str:
        """Export dashboard state to JSON."""
        return json.dumps({
            "summary": self.summary(),
            "proofs": [p.to_dict() for p in self._proofs.values()],
            "coverage": [
                {
                    "module": c.module,
                    "total": c.total_claims,
                    "verified": c.verified,
                    "failed": c.failed,
                    "pending": c.pending,
                    "pct": c.coverage_pct,
                }
                for c in self.coverage_map()
            ],
            "anomalies": [
                {
                    "proof_id": a.proof_id,
                    "module": a.module,
                    "previous": a.previous_verdict.name,
                    "current": a.current_verdict.name,
                }
                for a in self._anomalies
            ],
        }, indent=indent, default=str)

    def to_html(self) -> str:
        """Generate a static HTML dashboard report."""
        summary = self.summary()
        coverage = self.coverage_map()

        rows = []
        for c in coverage:
            color = "#4caf50" if c.coverage_pct >= 90 else "#ff9800" if c.coverage_pct >= 50 else "#f44336"
            rows.append(
                f'<tr><td>{c.module}</td><td>{c.verified}/{c.total_claims}</td>'
                f'<td><span style="color:{color}">{c.coverage_pct:.1f}%</span></td>'
                f'<td>{c.failed}</td><td>{c.pending}</td></tr>'
            )

        anomaly_rows = []
        for a in self._anomalies:
            anomaly_rows.append(
                f'<tr><td>{a.proof_id}</td><td>{a.module}</td>'
                f'<td>{a.previous_verdict.name}</td><td>{a.current_verdict.name}</td></tr>'
            )

        return f"""<!DOCTYPE html>
<html>
<head>
<title>HyperTensor Proof Dashboard</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 2rem; background: #1a1a2e; color: #eee; }}
h1 {{ color: #00d4ff; }}
h2 {{ color: #7c83ff; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #333; }}
th {{ background: #16213e; color: #00d4ff; }}
.stat {{ display: inline-block; margin: 1rem; padding: 1rem; background: #16213e; border-radius: 8px; min-width: 150px; text-align: center; }}
.stat .value {{ font-size: 2rem; font-weight: bold; color: #00d4ff; }}
.stat .label {{ font-size: 0.9rem; color: #aaa; }}
</style>
</head>
<body>
<h1>HyperTensor Proof Dashboard</h1>

<div>
<div class="stat"><div class="value">{summary['total_proofs']}</div><div class="label">Total Proofs</div></div>
<div class="stat"><div class="value">{summary['coverage_pct']:.1f}%</div><div class="label">Coverage</div></div>
<div class="stat"><div class="value">{summary.get('by_verdict', {}).get('VERIFIED', 0)}</div><div class="label">Verified</div></div>
<div class="stat"><div class="value">{summary.get('by_verdict', {}).get('FAILED', 0)}</div><div class="label">Failed</div></div>
<div class="stat"><div class="value">{len(self._anomalies)}</div><div class="label">Regressions</div></div>
</div>

<h2>Coverage by Module</h2>
<table>
<tr><th>Module</th><th>Verified/Total</th><th>Coverage</th><th>Failed</th><th>Pending</th></tr>
{''.join(rows)}
</table>

{"<h2>Regressions</h2><table><tr><th>Proof ID</th><th>Module</th><th>Previous</th><th>Current</th></tr>" + ''.join(anomaly_rows) + "</table>" if anomaly_rows else ""}

<h2>Layer Distribution</h2>
<table>
<tr><th>Layer</th><th>Count</th></tr>
{''.join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in summary.get('by_layer', {}).items())}
</table>

<footer style="margin-top:2rem;color:#666;font-size:0.8rem;">Generated by HyperTensor Proof Engine</footer>
</body>
</html>"""

    def export_html(self, path: Union[str, Path]) -> Path:
        """Write HTML dashboard to file."""
        p = Path(path)
        p.write_text(self.to_html())
        logger.info("Dashboard exported to %s", p)
        return p

    def export_json(self, path: Union[str, Path]) -> Path:
        """Write JSON dashboard to file."""
        p = Path(path)
        p.write_text(self.to_json())
        return p


__all__ = [
    "Verdict",
    "ProofLayer",
    "ProofStatus",
    "CoverageEntry",
    "Anomaly",
    "ProofDashboard",
]
