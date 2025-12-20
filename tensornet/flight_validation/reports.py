"""
Validation campaign reports and documentation.

This module provides tools for generating comprehensive
validation reports from flight data comparisons.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TextIO
from enum import Enum, auto
from pathlib import Path
import json
import time

from .comparison import ComparisonResult, ComparisonStatus, FieldComparison
from .uncertainty import UncertaintyBudget, ValidationUncertainty


class ReportFormat(Enum):
    """Report output formats."""
    MARKDOWN = "md"
    HTML = "html"
    LATEX = "tex"
    JSON = "json"
    TEXT = "txt"


class ValidationLevel(Enum):
    """Validation rigor levels (AIAA standards)."""
    UNIT = auto()          # Unit-level validation
    BENCHMARK = auto()     # Benchmark validation
    SUBSYSTEM = auto()     # Subsystem validation
    SYSTEM = auto()        # Full system validation
    COMPLETE = auto()      # Complete qualification


@dataclass
class ValidationCase:
    """Single validation case within a campaign."""
    case_id: str
    description: str
    
    # Comparison result
    comparison: Optional[ComparisonResult] = None
    
    # Uncertainty
    uncertainty_budget: Optional[UncertaintyBudget] = None
    
    # Metadata
    flight_condition: str = ""
    mach_range: str = ""
    aoa_range: str = ""
    
    # Status
    passed: bool = False
    notes: str = ""
    
    def get_summary(self) -> Dict[str, Any]:
        """Get case summary."""
        return {
            'case_id': self.case_id,
            'description': self.description,
            'passed': self.passed,
            'status': self.comparison.overall_status.name if self.comparison else "NOT_RUN",
            'flight_condition': self.flight_condition,
        }


@dataclass
class ValidationCampaign:
    """Complete validation campaign."""
    campaign_id: str
    name: str
    description: str = ""
    
    # Validation level
    level: ValidationLevel = ValidationLevel.SYSTEM
    
    # Cases
    cases: List[ValidationCase] = field(default_factory=list)
    
    # Overall results
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    
    # Metadata
    vehicle_name: str = ""
    simulation_tool: str = ""
    date: str = ""
    analyst: str = ""
    
    # Requirements
    pass_threshold: float = 0.9  # 90% of cases must pass
    
    def add_case(self, case: ValidationCase):
        """Add validation case."""
        self.cases.append(case)
        self._update_statistics()
    
    def _update_statistics(self):
        """Update campaign statistics."""
        self.total_cases = len(self.cases)
        self.passed_cases = sum(1 for c in self.cases if c.passed)
        self.failed_cases = self.total_cases - self.passed_cases
    
    def get_pass_rate(self) -> float:
        """Get overall pass rate."""
        if self.total_cases == 0:
            return 0.0
        return self.passed_cases / self.total_cases
    
    def is_successful(self) -> bool:
        """Check if campaign meets pass threshold."""
        return self.get_pass_rate() >= self.pass_threshold
    
    def get_summary(self) -> Dict[str, Any]:
        """Get campaign summary."""
        return {
            'campaign_id': self.campaign_id,
            'name': self.name,
            'level': self.level.name,
            'total_cases': self.total_cases,
            'passed_cases': self.passed_cases,
            'failed_cases': self.failed_cases,
            'pass_rate': self.get_pass_rate(),
            'successful': self.is_successful(),
        }
    
    def get_failed_cases(self) -> List[ValidationCase]:
        """Get list of failed cases."""
        return [c for c in self.cases if not c.passed]


class ValidationReport:
    """
    Generator for validation reports.
    """
    
    def __init__(self, campaign: ValidationCampaign):
        """
        Initialize report generator.
        
        Args:
            campaign: Validation campaign to report on
        """
        self.campaign = campaign
    
    def generate(
        self,
        format: ReportFormat = ReportFormat.MARKDOWN,
        output_path: Optional[Path] = None,
    ) -> str:
        """
        Generate validation report.
        
        Args:
            format: Output format
            output_path: Optional path to save report
        
        Returns:
            Report content as string
        """
        generators = {
            ReportFormat.MARKDOWN: self._generate_markdown,
            ReportFormat.HTML: self._generate_html,
            ReportFormat.JSON: self._generate_json,
            ReportFormat.TEXT: self._generate_text,
            ReportFormat.LATEX: self._generate_latex,
        }
        
        generator = generators.get(format, self._generate_markdown)
        content = generator()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(content)
        
        return content
    
    def _generate_markdown(self) -> str:
        """Generate Markdown report."""
        lines = []
        c = self.campaign
        
        # Title
        lines.append(f"# Validation Report: {c.name}")
        lines.append("")
        lines.append(f"**Campaign ID:** {c.campaign_id}")
        lines.append(f"**Validation Level:** {c.level.name}")
        lines.append(f"**Date:** {c.date or time.strftime('%Y-%m-%d')}")
        if c.analyst:
            lines.append(f"**Analyst:** {c.analyst}")
        lines.append("")
        
        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        
        status = "✅ PASSED" if c.is_successful() else "❌ FAILED"
        lines.append(f"**Overall Status:** {status}")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total Cases | {c.total_cases} |")
        lines.append(f"| Passed | {c.passed_cases} |")
        lines.append(f"| Failed | {c.failed_cases} |")
        lines.append(f"| Pass Rate | {100*c.get_pass_rate():.1f}% |")
        lines.append(f"| Required | {100*c.pass_threshold:.1f}% |")
        lines.append("")
        
        if c.description:
            lines.append("### Description")
            lines.append("")
            lines.append(c.description)
            lines.append("")
        
        # Case Results
        lines.append("## Validation Cases")
        lines.append("")
        
        for case in c.cases:
            status_icon = "✅" if case.passed else "❌"
            lines.append(f"### {status_icon} Case: {case.case_id}")
            lines.append("")
            lines.append(f"**Description:** {case.description}")
            
            if case.flight_condition:
                lines.append(f"**Flight Condition:** {case.flight_condition}")
            
            if case.comparison:
                comp = case.comparison
                lines.append("")
                lines.append("#### Comparison Results")
                lines.append("")
                lines.append(f"| Field | Mean Error (%) | RMS Error | Status |")
                lines.append(f"|-------|----------------|-----------|--------|")
                
                for name, fc in comp.field_comparisons.items():
                    status = fc.status.name
                    lines.append(f"| {name} | {fc.mean_percent_error:.2f} | {fc.rms_error:.4f} | {status} |")
                lines.append("")
            
            if case.notes:
                lines.append(f"**Notes:** {case.notes}")
                lines.append("")
        
        # Failed Cases Summary
        failed = c.get_failed_cases()
        if failed:
            lines.append("## Failed Cases Summary")
            lines.append("")
            
            for case in failed:
                lines.append(f"- **{case.case_id}**: {case.description}")
                if case.comparison:
                    worst = max(
                        case.comparison.field_comparisons.values(),
                        key=lambda x: x.mean_percent_error,
                        default=None
                    )
                    if worst:
                        lines.append(f"  - Worst field: {worst.field_name} ({worst.mean_percent_error:.1f}% error)")
            lines.append("")
        
        # Conclusions
        lines.append("## Conclusions")
        lines.append("")
        
        if c.is_successful():
            lines.append("The validation campaign has **successfully demonstrated** that the simulation "
                        "results are in acceptable agreement with flight test data across the "
                        "evaluated conditions.")
        else:
            lines.append("The validation campaign **did not meet** the required pass threshold. "
                        "Additional investigation is required for the failed cases listed above.")
        lines.append("")
        
        # Footer
        lines.append("---")
        lines.append(f"*Report generated by HyperTensor Validation Framework*")
        
        return "\n".join(lines)
    
    def _generate_html(self) -> str:
        """Generate HTML report."""
        c = self.campaign
        status_class = "success" if c.is_successful() else "failure"
        status_text = "PASSED" if c.is_successful() else "FAILED"
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Validation Report: {c.name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #1a1a2e; color: white; padding: 20px; }}
        .summary {{ margin: 20px 0; padding: 20px; background: #f5f5f5; }}
        .success {{ color: #28a745; }}
        .failure {{ color: #dc3545; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #4a4e69; color: white; }}
        .case {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
        .case.passed {{ border-left: 4px solid #28a745; }}
        .case.failed {{ border-left: 4px solid #dc3545; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Validation Report: {c.name}</h1>
        <p>Campaign ID: {c.campaign_id} | Level: {c.level.name}</p>
    </div>
    
    <div class="summary">
        <h2>Executive Summary</h2>
        <p><strong>Overall Status:</strong> <span class="{status_class}">{status_text}</span></p>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Cases</td><td>{c.total_cases}</td></tr>
            <tr><td>Passed</td><td>{c.passed_cases}</td></tr>
            <tr><td>Failed</td><td>{c.failed_cases}</td></tr>
            <tr><td>Pass Rate</td><td>{100*c.get_pass_rate():.1f}%</td></tr>
        </table>
    </div>
    
    <h2>Validation Cases</h2>
"""
        
        for case in c.cases:
            case_class = "passed" if case.passed else "failed"
            html += f"""
    <div class="case {case_class}">
        <h3>Case: {case.case_id}</h3>
        <p>{case.description}</p>
"""
            if case.comparison:
                html += """        <table>
            <tr><th>Field</th><th>Mean Error (%)</th><th>Status</th></tr>
"""
                for name, fc in case.comparison.field_comparisons.items():
                    html += f"            <tr><td>{name}</td><td>{fc.mean_percent_error:.2f}</td><td>{fc.status.name}</td></tr>\n"
                html += "        </table>\n"
            
            html += "    </div>\n"
        
        html += """
    <footer>
        <p><em>Report generated by HyperTensor Validation Framework</em></p>
    </footer>
</body>
</html>
"""
        return html
    
    def _generate_json(self) -> str:
        """Generate JSON report."""
        data = {
            'campaign': self.campaign.get_summary(),
            'cases': [c.get_summary() for c in self.campaign.cases],
            'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
        
        # Add detailed comparisons
        for case in self.campaign.cases:
            if case.comparison:
                for name, fc in case.comparison.field_comparisons.items():
                    case_data = next(
                        (c for c in data['cases'] if c['case_id'] == case.case_id),
                        None
                    )
                    if case_data:
                        if 'field_comparisons' not in case_data:
                            case_data['field_comparisons'] = {}
                        case_data['field_comparisons'][name] = fc.to_dict()
        
        return json.dumps(data, indent=2)
    
    def _generate_text(self) -> str:
        """Generate plain text report."""
        lines = []
        c = self.campaign
        
        lines.append("=" * 60)
        lines.append(f"VALIDATION REPORT: {c.name.upper()}")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Campaign ID: {c.campaign_id}")
        lines.append(f"Level: {c.level.name}")
        lines.append(f"Status: {'PASSED' if c.is_successful() else 'FAILED'}")
        lines.append("")
        lines.append("-" * 40)
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"  Total Cases:  {c.total_cases}")
        lines.append(f"  Passed:       {c.passed_cases}")
        lines.append(f"  Failed:       {c.failed_cases}")
        lines.append(f"  Pass Rate:    {100*c.get_pass_rate():.1f}%")
        lines.append("")
        
        for case in c.cases:
            lines.append("-" * 40)
            status = "[PASS]" if case.passed else "[FAIL]"
            lines.append(f"{status} Case: {case.case_id}")
            lines.append(f"  {case.description}")
            
            if case.comparison:
                for name, fc in case.comparison.field_comparisons.items():
                    lines.append(f"    {name}: {fc.mean_percent_error:.2f}% error ({fc.status.name})")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def _generate_latex(self) -> str:
        """Generate LaTeX report."""
        c = self.campaign
        
        latex = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{xcolor}
\definecolor{success}{RGB}{40, 167, 69}
\definecolor{failure}{RGB}{220, 53, 69}

\title{Validation Report: """ + c.name + r"""}
\author{HyperTensor Validation Framework}
\date{""" + time.strftime('%Y-%m-%d') + r"""}

\begin{document}
\maketitle

\section{Executive Summary}

\textbf{Campaign ID:} """ + c.campaign_id + r"""\\
\textbf{Validation Level:} """ + c.level.name + r"""\\
\textbf{Overall Status:} """ + (r"\textcolor{success}{PASSED}" if c.is_successful() else r"\textcolor{failure}{FAILED}") + r"""

\begin{table}[h]
\centering
\begin{tabular}{lr}
\toprule
Metric & Value \\
\midrule
Total Cases & """ + str(c.total_cases) + r""" \\
Passed & """ + str(c.passed_cases) + r""" \\
Failed & """ + str(c.failed_cases) + r""" \\
Pass Rate & """ + f"{100*c.get_pass_rate():.1f}" + r"""\% \\
\bottomrule
\end{tabular}
\caption{Campaign Summary}
\end{table}

\section{Validation Cases}
"""
        
        for case in c.cases:
            status = r"\textcolor{success}{PASS}" if case.passed else r"\textcolor{failure}{FAIL}"
            latex += r"""
\subsection{Case: """ + case.case_id + r""" [""" + status + r"""]}

""" + case.description + r"""

"""
            if case.comparison and case.comparison.field_comparisons:
                latex += r"""\begin{table}[h]
\centering
\begin{tabular}{lrr}
\toprule
Field & Mean Error (\%) & Status \\
\midrule
"""
                for name, fc in case.comparison.field_comparisons.items():
                    latex += f"{name} & {fc.mean_percent_error:.2f} & {fc.status.name} \\\\\n"
                
                latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
        
        latex += r"""
\end{document}
"""
        return latex


def generate_validation_report(
    campaign: ValidationCampaign,
    format: ReportFormat = ReportFormat.MARKDOWN,
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate validation report.
    
    Args:
        campaign: Validation campaign
        format: Output format
        output_path: Optional path to save report
    
    Returns:
        Report content as string
    """
    report = ValidationReport(campaign)
    return report.generate(format=format, output_path=output_path)


def create_validation_case(
    case_id: str,
    description: str,
    comparison: ComparisonResult,
    uncertainty_budget: Optional[UncertaintyBudget] = None,
    pass_threshold: float = 0.1,
) -> ValidationCase:
    """
    Create validation case from comparison result.
    
    Args:
        case_id: Case identifier
        description: Case description
        comparison: Comparison result
        uncertainty_budget: Optional uncertainty budget
        pass_threshold: Error threshold for passing (as fraction)
    
    Returns:
        ValidationCase object
    """
    # Check if passed based on comparison status
    passed = comparison.validation_passed
    
    # Also check uncertainty if available
    if uncertainty_budget:
        for param, val_unc in uncertainty_budget.validation_uncertainties.items():
            if not val_unc.is_validated():
                passed = False
                break
    
    return ValidationCase(
        case_id=case_id,
        description=description,
        comparison=comparison,
        uncertainty_budget=uncertainty_budget,
        passed=passed,
    )
