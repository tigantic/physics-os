"""Surgical simulation report generation.

Produces structured, reproducible reports containing:
  - Patient demographics (anonymizable)
  - Pre-operative anatomy summary
  - Surgical plan specification
  - Simulation results (FEM, CFD, healing)
  - Aesthetic, functional, and safety metrics
  - Uncertainty quantification summary
  - Comparative analysis (pre vs post prediction)
  - Provenance chain for full reproducibility

Output formats: JSON (machine-readable), HTML (printable), Markdown.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .core.provenance import hash_dict
from .core.types import ClinicalMeasurement

logger = logging.getLogger(__name__)


@dataclass
class ReportSection:
    """A named section in the report."""
    title: str
    content: Dict[str, Any] = field(default_factory=dict)
    subsections: List[ReportSection] = field(default_factory=list)
    figures: List[str] = field(default_factory=list)  # paths to figure files
    tables: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "subsections": [s.to_dict() for s in self.subsections],
            "figures": self.figures,
            "tables": self.tables,
        }


@dataclass
class ReportMetadata:
    """Report metadata for tracking and compliance."""
    report_id: str = ""
    case_id: str = ""
    generated_at: float = 0.0
    generated_by: str = ""
    platform_version: str = ""
    plan_hash: str = ""
    sim_hash: str = ""
    report_hash: str = ""
    disclaimers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "case_id": self.case_id,
            "generated_at": self.generated_at,
            "generated_by": self.generated_by,
            "platform_version": self.platform_version,
            "plan_hash": self.plan_hash,
            "sim_hash": self.sim_hash,
            "report_hash": self.report_hash,
            "disclaimers": self.disclaimers,
        }


class ReportBuilder:
    """Build structured simulation reports.

    Assembles report sections from simulation results, metrics,
    and case data, then serializes to JSON, HTML, or Markdown.
    """

    def __init__(
        self,
        case_id: str,
        *,
        generated_by: str = "",
        platform_version: str = "0.1.0",
    ) -> None:
        self._case_id = case_id
        self._metadata = ReportMetadata(
            case_id=case_id,
            generated_at=time.time(),
            generated_by=generated_by,
            platform_version=platform_version,
        )
        self._sections: List[ReportSection] = []
        self._measurements: List[ClinicalMeasurement] = []

    def add_patient_demographics(
        self,
        demographics: Dict[str, Any],
        *,
        anonymize: bool = False,
    ) -> ReportBuilder:
        """Add patient demographics section."""
        content = dict(demographics)
        if anonymize:
            for key in ("name", "patient_id", "dob", "mrn"):
                if key in content:
                    content[key] = "[REDACTED]"

        self._sections.append(ReportSection(
            title="Patient Demographics",
            content=content,
        ))
        return self

    def add_anatomy_summary(
        self,
        preop_measurements: List[ClinicalMeasurement],
        anatomy_data: Dict[str, Any],
    ) -> ReportBuilder:
        """Add pre-operative anatomy summary."""
        content: Dict[str, Any] = {
            "anatomy": anatomy_data,
            "measurements": [
                {"name": m.name, "value": m.value, "unit": m.unit}
                for m in preop_measurements
            ],
        }
        self._sections.append(ReportSection(
            title="Pre-operative Anatomy",
            content=content,
        ))
        self._measurements.extend(preop_measurements)
        return self

    def add_surgical_plan(self, plan_data: Dict[str, Any]) -> ReportBuilder:
        """Add surgical plan specification."""
        self._metadata.plan_hash = plan_data.get("plan_hash", "")
        self._sections.append(ReportSection(
            title="Surgical Plan",
            content=plan_data,
        ))
        return self

    def add_simulation_results(
        self,
        sim_data: Dict[str, Any],
    ) -> ReportBuilder:
        """Add simulation results."""
        self._metadata.sim_hash = sim_data.get("run_hash", "")
        self._sections.append(ReportSection(
            title="Simulation Results",
            content=sim_data,
        ))
        return self

    def add_aesthetic_report(self, data: Dict[str, Any]) -> ReportBuilder:
        """Add aesthetic metrics."""
        self._sections.append(ReportSection(
            title="Aesthetic Assessment",
            content=data,
        ))
        return self

    def add_functional_report(self, data: Dict[str, Any]) -> ReportBuilder:
        """Add functional metrics."""
        self._sections.append(ReportSection(
            title="Functional Assessment",
            content=data,
        ))
        return self

    def add_safety_report(self, data: Dict[str, Any]) -> ReportBuilder:
        """Add safety assessment."""
        self._sections.append(ReportSection(
            title="Safety Assessment",
            content=data,
        ))
        return self

    def add_uncertainty_report(self, data: Dict[str, Any]) -> ReportBuilder:
        """Add uncertainty quantification."""
        self._sections.append(ReportSection(
            title="Uncertainty Quantification",
            content=data,
        ))
        return self

    def add_healing_prediction(self, data: Dict[str, Any]) -> ReportBuilder:
        """Add healing timeline prediction."""
        self._sections.append(ReportSection(
            title="Healing Prediction",
            content=data,
        ))
        return self

    def add_comparison(
        self,
        preop_data: Dict[str, Any],
        postop_prediction: Dict[str, Any],
        deltas: Dict[str, float],
    ) -> ReportBuilder:
        """Add pre-op vs post-op comparison."""
        self._sections.append(ReportSection(
            title="Pre/Post Comparison",
            content={
                "preop": preop_data,
                "postop_prediction": postop_prediction,
                "deltas": deltas,
            },
        ))
        return self

    def add_disclaimers(self, disclaimers: List[str]) -> ReportBuilder:
        """Add regulatory disclaimers."""
        self._metadata.disclaimers = disclaimers
        return self

    def build_json(self) -> Dict[str, Any]:
        """Build the complete report as a JSON-serializable dict."""
        report: Dict[str, Any] = {
            "metadata": self._metadata.to_dict(),
            "sections": [s.to_dict() for s in self._sections],
        }
        report_hash = hash_dict(report)
        self._metadata.report_hash = report_hash
        report["metadata"]["report_hash"] = report_hash
        return report

    def save_json(self, path: Path) -> Path:
        """Save the report as JSON."""
        report = self.build_json()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Report saved: %s", path)
        return path

    def build_markdown(self) -> str:
        """Build the report as Markdown text."""
        lines: List[str] = []
        lines.append(f"# Surgical Simulation Report")
        lines.append(f"")
        lines.append(f"**Case:** {self._case_id}")
        lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(self._metadata.generated_at))}")
        lines.append(f"**Platform:** v{self._metadata.platform_version}")
        lines.append(f"")

        for section in self._sections:
            lines.append(f"## {section.title}")
            lines.append("")
            self._render_content_md(section.content, lines, depth=0)
            lines.append("")
            for sub in section.subsections:
                lines.append(f"### {sub.title}")
                lines.append("")
                self._render_content_md(sub.content, lines, depth=0)
                lines.append("")

        if self._metadata.disclaimers:
            lines.append("## Disclaimers")
            lines.append("")
            for d in self._metadata.disclaimers:
                lines.append(f"- {d}")
            lines.append("")

        lines.append("---")
        lines.append(f"Report hash: `{self._metadata.report_hash[:16]}`")
        lines.append(f"Plan hash: `{self._metadata.plan_hash[:16]}`")
        lines.append(f"Simulation hash: `{self._metadata.sim_hash[:16]}`")

        return "\n".join(lines)

    def save_markdown(self, path: Path) -> Path:
        """Save the report as Markdown."""
        md = self.build_markdown()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
        logger.info("Report saved: %s", path)
        return path

    def build_html(self) -> str:
        """Build the report as standalone HTML."""
        md = self.build_markdown()
        # Simple Markdown → HTML conversion
        html_body = self._md_to_html(md)
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Simulation Report - {self._case_id}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 900px; margin: 2em auto; padding: 0 1em; color: #333; }}
h1 {{ border-bottom: 2px solid #2563eb; padding-bottom: 0.3em; }}
h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 0.2em; margin-top: 1.5em; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #f3f4f6; }}
.disclaimer {{ font-size: 0.85em; color: #666; margin-top: 2em; padding: 1em;
               background: #fef3c7; border-left: 4px solid #f59e0b; }}
code {{ background: #f3f4f6; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
.hash {{ color: #999; font-size: 0.8em; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

    def save_html(self, path: Path) -> Path:
        """Save the report as HTML."""
        html = self.build_html()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info("Report saved: %s", path)
        return path

    @staticmethod
    def _render_content_md(
        content: Dict[str, Any],
        lines: List[str],
        depth: int,
    ) -> None:
        """Render a content dict as Markdown key-value pairs."""
        indent = "  " * depth
        for key, value in content.items():
            if isinstance(value, dict):
                lines.append(f"{indent}**{key}:**")
                ReportBuilder._render_content_md(value, lines, depth + 1)
            elif isinstance(value, list):
                lines.append(f"{indent}**{key}:**")
                for item in value:
                    if isinstance(item, dict):
                        parts = [f"{k}={v}" for k, v in item.items()]
                        lines.append(f"{indent}  - {', '.join(parts)}")
                    else:
                        lines.append(f"{indent}  - {item}")
            else:
                lines.append(f"{indent}- **{key}:** {value}")

    @staticmethod
    def _md_to_html(md: str) -> str:
        """Minimal Markdown to HTML converter."""
        html_lines: List[str] = []
        in_list = False

        for line in md.split("\n"):
            stripped = line.strip()

            if stripped.startswith("# "):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append(f"<h1>{stripped[2:]}</h1>")
            elif stripped.startswith("## "):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append(f"<h2>{stripped[3:]}</h2>")
            elif stripped.startswith("### "):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append(f"<h3>{stripped[4:]}</h3>")
            elif stripped.startswith("- "):
                if not in_list:
                    html_lines.append("<ul>")
                    in_list = True
                content = stripped[2:]
                # Bold markers
                content = content.replace("**", "<b>", 1).replace("**", "</b>", 1)
                html_lines.append(f"<li>{content}</li>")
            elif stripped.startswith("---"):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append("<hr>")
            elif stripped.startswith("`") and stripped.endswith("`"):
                html_lines.append(f"<code>{stripped[1:-1]}</code>")
            elif stripped:
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                # Bold markers
                text = stripped
                while "**" in text:
                    text = text.replace("**", "<b>", 1).replace("**", "</b>", 1)
                html_lines.append(f"<p>{text}</p>")
            else:
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False

        if in_list:
            html_lines.append("</ul>")

        return "\n".join(html_lines)
