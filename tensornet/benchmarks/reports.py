"""
Benchmark reporting utilities.

This module provides report generation for benchmark results
in various formats (Markdown, JSON, CSV, HTML).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum, auto
from pathlib import Path
import json
import time


class ReportFormat(Enum):
    """Report output formats."""
    MARKDOWN = "md"
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    TEXT = "txt"


@dataclass
class BenchmarkReport:
    """
    Benchmark report container.
    
    Aggregates benchmark results and generates formatted reports.
    """
    title: str = "TensorRT Benchmark Report"
    description: str = ""
    timestamp: float = field(default_factory=time.time)
    
    # Results
    results: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    environment: Dict[str, Any] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)
    
    def add_result(self, result: Any):
        """Add benchmark result."""
        if hasattr(result, 'to_dict'):
            self.results.append(result.to_dict())
        else:
            self.results.append(result)
    
    def add_results(self, results: List[Any]):
        """Add multiple benchmark results."""
        for result in results:
            self.add_result(result)
    
    def set_summary(self, summary: Dict[str, Any]):
        """Set report summary."""
        self.summary = summary
    
    def set_environment(self, env: Dict[str, Any]):
        """Set environment info."""
        self.environment = env
    
    def to_markdown(self) -> str:
        """Generate Markdown report."""
        lines = [
            f"# {self.title}",
            "",
            f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}*",
            "",
        ]
        
        if self.description:
            lines.extend([self.description, ""])
        
        # Environment section
        if self.environment:
            lines.extend([
                "## Environment",
                "",
                "| Property | Value |",
                "|----------|-------|",
            ])
            for key, value in self.environment.items():
                lines.append(f"| {key} | {value} |")
            lines.append("")
        
        # Summary section
        if self.summary:
            lines.extend([
                "## Summary",
                "",
            ])
            for key, value in self.summary.items():
                if isinstance(value, dict):
                    lines.append(f"### {key}")
                    lines.append("")
                    for k, v in value.items():
                        lines.append(f"- **{k}**: {v}")
                    lines.append("")
                else:
                    lines.append(f"- **{key}**: {value}")
            lines.append("")
        
        # Results section
        if self.results:
            lines.extend([
                "## Detailed Results",
                "",
            ])
            
            # Create results table
            if self.results:
                first_result = self.results[0]
                
                # Latency table
                lines.extend([
                    "### Latency Results",
                    "",
                    "| Name | Precision | Batch Size | Mean (ms) | P95 (ms) | P99 (ms) |",
                    "|------|-----------|------------|-----------|----------|----------|",
                ])
                
                for result in self.results:
                    name = result.get('name', 'N/A')
                    precision = result.get('precision', 'N/A')
                    batch_size = result.get('batch_size', 'N/A')
                    
                    latency = result.get('latency', {})
                    mean = latency.get('mean_ms', 0)
                    p95 = latency.get('p95_ms', 0)
                    p99 = latency.get('p99_ms', 0)
                    
                    lines.append(
                        f"| {name} | {precision} | {batch_size} | "
                        f"{mean:.4f} | {p95:.4f} | {p99:.4f} |"
                    )
                
                lines.append("")
                
                # Throughput table
                lines.extend([
                    "### Throughput Results",
                    "",
                    "| Name | Precision | Batch Size | Samples/sec | Batches/sec |",
                    "|------|-----------|------------|-------------|-------------|",
                ])
                
                for result in self.results:
                    name = result.get('name', 'N/A')
                    precision = result.get('precision', 'N/A')
                    batch_size = result.get('batch_size', 'N/A')
                    
                    throughput = result.get('throughput', {})
                    samples = throughput.get('samples_per_second', 0)
                    batches = throughput.get('batches_per_second', 0)
                    
                    lines.append(
                        f"| {name} | {precision} | {batch_size} | "
                        f"{samples:.2f} | {batches:.2f} |"
                    )
                
                lines.append("")
                
                # Memory table
                if any('memory' in r for r in self.results):
                    lines.extend([
                        "### Memory Usage",
                        "",
                        "| Name | Precision | Peak (MB) | Allocated (MB) | Model Size (MB) |",
                        "|------|-----------|-----------|----------------|-----------------|",
                    ])
                    
                    for result in self.results:
                        name = result.get('name', 'N/A')
                        precision = result.get('precision', 'N/A')
                        
                        memory = result.get('memory', {})
                        peak = memory.get('peak_memory_mb', 0)
                        allocated = memory.get('allocated_memory_mb', 0)
                        model_size = memory.get('model_size_mb', 0)
                        
                        lines.append(
                            f"| {name} | {precision} | "
                            f"{peak:.2f} | {allocated:.2f} | {model_size:.2f} |"
                        )
                    
                    lines.append("")
        
        # Configuration section
        if self.configuration:
            lines.extend([
                "## Configuration",
                "",
                "```json",
                json.dumps(self.configuration, indent=2),
                "```",
                "",
            ])
        
        return "\n".join(lines)
    
    def to_json(self, indent: int = 2) -> str:
        """Generate JSON report."""
        return json.dumps({
            'title': self.title,
            'description': self.description,
            'timestamp': self.timestamp,
            'timestamp_formatted': time.strftime(
                '%Y-%m-%dT%H:%M:%SZ', 
                time.gmtime(self.timestamp)
            ),
            'environment': self.environment,
            'configuration': self.configuration,
            'summary': self.summary,
            'results': self.results,
        }, indent=indent)
    
    def to_csv(self) -> str:
        """Generate CSV report."""
        if not self.results:
            return ""
        
        # Extract all unique keys from results
        all_keys = set()
        for result in self.results:
            all_keys.update(self._flatten_dict(result).keys())
        
        headers = sorted(all_keys)
        
        lines = [",".join(headers)]
        
        for result in self.results:
            flat = self._flatten_dict(result)
            row = [str(flat.get(h, "")) for h in headers]
            lines.append(",".join(row))
        
        return "\n".join(lines)
    
    def to_html(self) -> str:
        """Generate HTML report."""
        md_content = self.to_markdown()
        
        # Simple MD to HTML conversion
        html_content = md_content
        html_content = html_content.replace("# ", "<h1>").replace("\n\n", "</h1>\n\n")
        html_content = html_content.replace("## ", "<h2>").replace("\n\n", "</h2>\n\n")
        html_content = html_content.replace("### ", "<h3>").replace("\n\n", "</h3>\n\n")
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; 
               max-width: 1200px; margin: 0 auto; padding: 2rem; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f5f5f5; }}
        tr:nth-child(even) {{ background: #fafafa; }}
        code {{ background: #f5f5f5; padding: 2px 6px; border-radius: 3px; }}
        pre {{ background: #1e1e1e; color: #d4d4d4; padding: 1rem; 
               border-radius: 8px; overflow-x: auto; }}
    </style>
</head>
<body>
    <pre>{md_content}</pre>
</body>
</html>"""
    
    def to_text(self) -> str:
        """Generate plain text report."""
        lines = [
            "=" * 60,
            self.title.center(60),
            "=" * 60,
            "",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}",
            "",
        ]
        
        if self.summary:
            lines.extend(["SUMMARY", "-" * 40])
            for key, value in self.summary.items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        if self.results:
            lines.extend(["RESULTS", "-" * 40])
            for i, result in enumerate(self.results, 1):
                lines.append(f"\n[{i}] {result.get('name', 'Unknown')}")
                lines.append(f"    Precision: {result.get('precision', 'N/A')}")
                lines.append(f"    Batch Size: {result.get('batch_size', 'N/A')}")
                
                if 'latency' in result:
                    lat = result['latency']
                    lines.append(f"    Latency: {lat.get('mean_ms', 0):.4f} ms (mean)")
                
                if 'throughput' in result:
                    tp = result['throughput']
                    lines.append(f"    Throughput: {tp.get('samples_per_second', 0):.2f} samples/sec")
        
        lines.extend(["", "=" * 60])
        
        return "\n".join(lines)
    
    def _flatten_dict(
        self, 
        d: Dict[str, Any], 
        prefix: str = ""
    ) -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = {}
        for k, v in d.items():
            key = f"{prefix}_{k}" if prefix else k
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, key))
            else:
                items[key] = v
        return items
    
    def export(
        self, 
        path: Union[str, Path], 
        format: ReportFormat = ReportFormat.MARKDOWN
    ):
        """
        Export report to file.
        
        Args:
            path: Output file path
            format: Report format
        """
        path = Path(path)
        
        if format == ReportFormat.MARKDOWN:
            content = self.to_markdown()
        elif format == ReportFormat.JSON:
            content = self.to_json()
        elif format == ReportFormat.CSV:
            content = self.to_csv()
        elif format == ReportFormat.HTML:
            content = self.to_html()
        else:
            content = self.to_text()
        
        path.write_text(content, encoding='utf-8')


def generate_report(
    results: List[Any],
    title: str = "TensorRT Benchmark Report",
    format: ReportFormat = ReportFormat.MARKDOWN,
) -> str:
    """
    Generate benchmark report from results.
    
    Args:
        results: List of benchmark results
        title: Report title
        format: Output format
    
    Returns:
        Formatted report string
    """
    report = BenchmarkReport(title=title)
    report.add_results(results)
    
    # Calculate summary
    if results:
        latencies = []
        throughputs = []
        
        for result in report.results:
            if 'latency' in result:
                latencies.append(result['latency'].get('mean_ms', 0))
            if 'throughput' in result:
                throughputs.append(result['throughput'].get('samples_per_second', 0))
        
        report.set_summary({
            'total_benchmarks': len(results),
            'avg_latency_ms': sum(latencies) / len(latencies) if latencies else 0,
            'max_throughput': max(throughputs) if throughputs else 0,
        })
    
    if format == ReportFormat.MARKDOWN:
        return report.to_markdown()
    elif format == ReportFormat.JSON:
        return report.to_json()
    elif format == ReportFormat.CSV:
        return report.to_csv()
    elif format == ReportFormat.HTML:
        return report.to_html()
    else:
        return report.to_text()


def export_to_csv(
    results: List[Any],
    path: Union[str, Path],
):
    """
    Export results to CSV file.
    
    Args:
        results: Benchmark results
        path: Output file path
    """
    report = BenchmarkReport()
    report.add_results(results)
    report.export(path, ReportFormat.CSV)


def export_to_json(
    results: List[Any],
    path: Union[str, Path],
):
    """
    Export results to JSON file.
    
    Args:
        results: Benchmark results
        path: Output file path
    """
    report = BenchmarkReport()
    report.add_results(results)
    report.export(path, ReportFormat.JSON)


def export_to_markdown(
    results: List[Any],
    path: Union[str, Path],
    title: str = "TensorRT Benchmark Report",
):
    """
    Export results to Markdown file.
    
    Args:
        results: Benchmark results
        path: Output file path
        title: Report title
    """
    report = BenchmarkReport(title=title)
    report.add_results(results)
    report.export(path, ReportFormat.MARKDOWN)
