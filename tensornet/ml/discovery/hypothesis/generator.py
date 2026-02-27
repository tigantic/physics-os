#!/usr/bin/env python3
"""
Hypothesis Generator

Synthesizes findings from all pipeline stages into actionable hypotheses.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from ..engine_v2 import Finding, DiscoveryResult


@dataclass
class Hypothesis:
    """A synthesized hypothesis from multiple findings."""
    id: str
    title: str
    description: str
    confidence: float  # 0.0 - 1.0
    severity: str  # INFO, LOW, MEDIUM, HIGH, CRITICAL
    findings: List[Finding]
    evidence_summary: str
    recommended_action: str
    domain_specific: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def hash(self) -> str:
        content = json.dumps({
            "id": self.id,
            "title": self.title,
            "confidence": self.confidence,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class HypothesisGenerator:
    """
    Generates hypotheses by combining findings from multiple primitives.
    
    Patterns detected:
    - Correlated anomalies across stages → likely real issue
    - Single-stage anomaly → noise or edge case
    - Topological + geometric signature → structural vulnerability
    - Distribution drift + high MMD → regime change
    """
    
    SEVERITY_ORDER = ["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    
    def __init__(self, domain: str = "general"):
        self.domain = domain
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Dict]:
        """Load hypothesis templates for different patterns."""
        return {
            "distribution_shift": {
                "title": "Distribution Regime Change",
                "pattern": ["OT", "RKHS"],
                "description_template": (
                    "Combined evidence from Optimal Transport (W₂={w2:.4f}) and "
                    "Kernel Methods (MMD={mmd:.4f}) indicates a significant "
                    "distribution shift from the baseline."
                ),
                "action": "Investigate data source for structural changes.",
            },
            "topological_anomaly": {
                "title": "Topological Structure Anomaly",
                "pattern": ["PH", "GA"],
                "description_template": (
                    "Persistent homology and geometric algebra both detect "
                    "unusual structural patterns. This may indicate a fundamental "
                    "change in the underlying data topology."
                ),
                "action": "Examine data for new connectivity patterns or holes.",
            },
            "multi_scale_energy": {
                "title": "Multi-Scale Energy Concentration",
                "pattern": ["SGW", "OT"],
                "description_template": (
                    "Spectral graph wavelets show energy concentration at "
                    "unusual scales, correlated with transport cost anomalies."
                ),
                "action": "Check for localized perturbations in the system.",
            },
            "full_pipeline_alert": {
                "title": "Cross-Primitive Alert",
                "pattern": ["OT", "SGW", "RKHS", "PH", "GA"],
                "description_template": (
                    "All 5 pipeline stages detected anomalies. This is a "
                    "high-confidence alert indicating significant deviation "
                    "from baseline across all analysis dimensions."
                ),
                "action": "Immediate investigation required.",
            },
            "geometric_invariant_break": {
                "title": "Geometric Invariant Violation",
                "pattern": ["GA"],
                "description_template": (
                    "Geometric algebra detected a break in expected invariants. "
                    "Severity: {severity:.4f}. This may indicate a symmetry "
                    "breaking or structural transformation."
                ),
                "action": "Check for rotational or translational anomalies.",
            },
        }
    
    def generate(self, result: DiscoveryResult) -> List[Hypothesis]:
        """Generate hypotheses from discovery result."""
        hypotheses = []
        
        # Group findings by primitive
        by_primitive = {}
        for f in result.findings:
            if f.primitive not in by_primitive:
                by_primitive[f.primitive] = []
            by_primitive[f.primitive].append(f)
        
        primitives_with_findings = set(by_primitive.keys())
        
        # Check each template pattern
        for template_id, template in self.templates.items():
            pattern = set(template["pattern"])
            
            # Check if all pattern primitives have findings
            if pattern.issubset(primitives_with_findings):
                # Collect relevant findings
                relevant_findings = []
                for p in pattern:
                    relevant_findings.extend(by_primitive[p])
                
                # Generate hypothesis
                hypothesis = self._apply_template(
                    template_id, 
                    template, 
                    relevant_findings
                )
                if hypothesis:
                    hypotheses.append(hypothesis)
        
        # If multiple HIGH+ findings across different primitives, generate alert
        high_findings = [f for f in result.findings 
                        if f.severity in ["HIGH", "CRITICAL"]]
        high_primitives = set(f.primitive for f in high_findings)
        
        if len(high_primitives) >= 3 and not any(
            h.title == "Cross-Primitive Alert" for h in hypotheses
        ):
            hypotheses.append(Hypothesis(
                id=f"hyp-alert-{len(hypotheses)+1}",
                title="Multi-Primitive High Severity Alert",
                description=(
                    f"High severity findings detected across {len(high_primitives)} "
                    f"primitives: {', '.join(sorted(high_primitives))}. "
                    "Strong correlation suggests a real anomaly."
                ),
                confidence=min(0.95, 0.6 + 0.1 * len(high_primitives)),
                severity="HIGH" if len(high_primitives) < 4 else "CRITICAL",
                findings=high_findings,
                evidence_summary="; ".join(f.summary for f in high_findings[:3]),
                recommended_action="Immediate manual review recommended.",
            ))
        
        # Sort by confidence
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        
        return hypotheses
    
    def _apply_template(
        self, 
        template_id: str, 
        template: Dict, 
        findings: List[Finding]
    ) -> Optional[Hypothesis]:
        """Apply a template to generate a hypothesis."""
        if not findings:
            return None
        
        # Extract values for template
        values = {}
        for f in findings:
            if f.primitive == "OT":
                values["w2"] = f.evidence.get("wasserstein_distance", 0)
            elif f.primitive == "RKHS":
                values["mmd"] = f.evidence.get("mmd_score", 0)
            elif f.primitive == "GA":
                values["severity"] = f.evidence.get("geometric_severity", 0)
        
        # Calculate confidence based on findings
        severities = [f.severity for f in findings]
        severity_scores = [self.SEVERITY_ORDER.index(s) for s in severities]
        avg_severity_score = sum(severity_scores) / len(severity_scores)
        
        confidence = min(0.95, 0.3 + 0.15 * avg_severity_score + 0.05 * len(findings))
        
        # Determine overall severity
        max_severity_idx = max(severity_scores)
        overall_severity = self.SEVERITY_ORDER[max_severity_idx]
        
        try:
            description = template["description_template"].format(**values)
        except KeyError:
            description = template["description_template"]
        
        return Hypothesis(
            id=f"hyp-{template_id}-{len(findings)}",
            title=template["title"],
            description=description,
            confidence=confidence,
            severity=overall_severity,
            findings=findings,
            evidence_summary="; ".join(f.summary for f in findings[:3]),
            recommended_action=template["action"],
            domain_specific={"template_id": template_id},
        )
    
    def to_report(self, hypotheses: List[Hypothesis]) -> str:
        """Generate markdown report from hypotheses."""
        if not hypotheses:
            return "# Hypothesis Report\n\nNo hypotheses generated.\n"
        
        report = f"""# Hypothesis Report

**Generated**: {datetime.now(timezone.utc).isoformat()}
**Domain**: {self.domain}
**Total Hypotheses**: {len(hypotheses)}

---

"""
        
        for i, h in enumerate(hypotheses, 1):
            report += f"""## {i}. {h.title}

**ID**: `{h.id}`
**Confidence**: {h.confidence:.1%}
**Severity**: {h.severity}
**Hash**: `{h.hash}`

### Description

{h.description}

### Evidence

{h.evidence_summary}

### Supporting Findings

"""
            for j, f in enumerate(h.findings, 1):
                report += f"- [{f.severity}] {f.primitive}: {f.summary}\n"
            
            report += f"""
### Recommended Action

{h.recommended_action}

---

"""
        
        return report


def main():
    """
    Demonstrate the hypothesis generator.
    
    ⚠️  DEMONSTRATION ONLY - NOT FOR PRODUCTION USE
    
    This function:
    - Uses random tensor data (not real scientific data)
    - Generates sample hypotheses for testing
    - Intended for verifying generator functionality
    
    For production use, integrate with real discovery results:
        engine = DiscoveryEngineV2()
        result = engine.discover(real_data)
        generator = HypothesisGenerator(domain="your_domain")
        hypotheses = generator.generate(result)
    """
    import logging
    logging.getLogger(__name__).warning(
        "main() demo uses random data - not for production use"
    )
    
    from ..engine_v2 import DiscoveryEngineV2
    import torch
    
    print("=" * 60)
    print("Hypothesis Generator Demo")
    print("=" * 60)
    
    # Run discovery
    engine = DiscoveryEngineV2(grid_bits=10)
    torch.manual_seed(42)
    data = torch.randn(2, 1024)
    result = engine.discover(data)
    
    print(f"\nFindings: {len(result.findings)}")
    for f in result.findings:
        print(f"  [{f.severity}] {f.primitive}: {f.summary[:50]}")
    
    # Generate hypotheses
    generator = HypothesisGenerator(domain="demo")
    hypotheses = generator.generate(result)
    
    print(f"\nHypotheses: {len(hypotheses)}")
    for h in hypotheses:
        print(f"  [{h.confidence:.0%}] {h.title}")
    
    # Generate report
    report = generator.to_report(hypotheses)
    print(f"\nReport length: {len(report)} chars")
    
    print("\n" + "=" * 60)
    print("✅ Hypothesis Generator operational")
    print("=" * 60)


if __name__ == "__main__":
    main()
