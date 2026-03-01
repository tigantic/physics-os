"""Report generator for scenario output packages.

Generates:
- Standalone HTML report with embedded images (base64)
- PDF report (via WeasyPrint)

Both reports contain:
- Executive summary table
- Per-scenario detail cards:
    • Problem description and physics context
    • Compilation parameters and advisor warnings
    • Field contour plots
    • Validation checks, claims, certificate digest
- Cross-scenario comparison charts
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, BaseLoader

logger = logging.getLogger("scenario_report")


# ═══════════════════════════════════════════════════════════════════
# HTML Template (Jinja2)
# ═══════════════════════════════════════════════════════════════════

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ title }}</title>
<style>
  @page { size: A4 landscape; margin: 15mm; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
    color: #1a1a2e; background: #f7f8fc; line-height: 1.5;
    padding: 2rem;
  }
  .container { max-width: 1200px; margin: 0 auto; }
  h1 { font-size: 1.8rem; color: #16213e; margin-bottom: 0.3rem; }
  h2 { font-size: 1.4rem; color: #0f3460; margin: 2rem 0 0.8rem 0;
       border-bottom: 2px solid #e94560; padding-bottom: 0.3rem; }
  h3 { font-size: 1.1rem; color: #533483; margin: 1rem 0 0.5rem 0; }
  .meta { color: #666; font-size: 0.85rem; margin-bottom: 2rem; }
  .badge {
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 0.78rem; font-weight: 600; margin-right: 6px;
  }
  .badge-success { background: #d4edda; color: #155724; }
  .badge-fail { background: #f8d7da; color: #721c24; }
  .badge-warn { background: #fff3cd; color: #856404; }
  .badge-info { background: #d1ecf1; color: #0c5460; }

  /* Summary table */
  table {
    width: 100%; border-collapse: collapse; margin: 1rem 0;
    font-size: 0.85rem;
  }
  th { background: #16213e; color: #fff; padding: 8px 12px; text-align: left; }
  td { padding: 6px 12px; border-bottom: 1px solid #e0e0e0; }
  tr:nth-child(even) { background: #f1f2f6; }
  tr:hover { background: #e8ecfb; }

  /* Scenario cards */
  .card {
    background: #fff; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    padding: 1.5rem; margin: 1.5rem 0; page-break-inside: avoid;
  }
  .card-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 1rem;
  }
  .card-header h3 { margin: 0; }
  .desc { color: #555; font-size: 0.9rem; margin-bottom: 1rem;
          font-style: italic; }
  .params-grid {
    display: grid; grid-template-columns: 200px 200px 200px;
    gap: 0.5rem; margin: 0.8rem 0;
  }
  .param-item {
    background: #f7f8fc; border-radius: 6px; padding: 6px 10px;
    font-size: 0.82rem;
  }
  .param-label { color: #888; font-size: 0.75rem; }
  .param-value { font-weight: 600; color: #16213e; }

  /* Images */
  .field-plots {
    display: flex; flex-wrap: wrap; gap: 1rem; margin: 1rem 0;
    justify-content: center;
  }
  .field-plots img {
    max-width: 48%; height: auto; border-radius: 6px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
  }
  .summary-img {
    display: block; margin: 1rem auto; max-width: 90%;
    border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  }

  /* Checks / Claims */
  .check-list { list-style: none; padding: 0; }
  .check-list li {
    padding: 4px 0; font-size: 0.83rem;
    border-bottom: 1px dotted #e0e0e0;
  }
  .check-pass::before { content: "✓ "; color: #28a745; font-weight: bold; }
  .check-fail::before { content: "✗ "; color: #dc3545; font-weight: bold; }

  /* Certificate */
  .cert-box {
    background: #f0f4ff; border: 1px solid #c3cfe2; border-radius: 8px;
    padding: 1rem; font-family: monospace; font-size: 0.78rem;
    word-break: break-all; margin: 0.8rem 0;
  }

  /* Warnings */
  .warning-list { list-style: none; padding: 0; }
  .warning-list li::before { content: "⚠ "; color: #f39c12; }
  .warning-list li { font-size: 0.83rem; padding: 2px 0; }

  /* Videos */
  .video-section { margin: 1rem 0; text-align: center; }
  .video-section img { max-width: 60%; border-radius: 8px; }

  /* Footer */
  .footer {
    margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #ccc;
    font-size: 0.78rem; color: #888; text-align: center;
  }
</style>
</head>
<body>
<div class="container">

<!-- ══════════════ HEADER ══════════════ -->
<h1>{{ title }}</h1>
<p class="meta">
  Generated {{ generated_at }} &nbsp;|&nbsp;
  Runtime v{{ runtime_version }} &nbsp;|&nbsp;
  Platform v{{ platform_version }} &nbsp;|&nbsp;
  {{ total_scenarios }} scenario{{ 's' if total_scenarios != 1 else '' }}
</p>

<!-- ══════════════ EXECUTIVE SUMMARY ══════════════ -->
<h2>Executive Summary</h2>
<table>
<thead>
  <tr>
    <th>#</th><th>Scenario</th><th>Domain</th>
    <th>Re</th><th>Ma</th><th>Compiled n_bits</th>
    <th>Quality</th><th>Valid</th><th>Wall Time</th>
  </tr>
</thead>
<tbody>
{% for row in scenario_table %}
  <tr>
    <td>{{ row.index }}</td>
    <td>{{ row.name }}</td>
    <td><code>{{ row.domain }}</code></td>
    <td>{{ "%.2e"|format(row.reynolds) }}</td>
    <td>{{ "%.3f"|format(row.mach) }}</td>
    <td>{{ row.n_bits }}</td>
    <td><span class="badge badge-info">{{ row.quality }}</span></td>
    <td>
      {% if row.valid %}
        <span class="badge badge-success">VALID</span>
      {% else %}
        <span class="badge badge-fail">INVALID</span>
      {% endif %}
    </td>
    <td>{{ "%.2f"|format(row.wall_time_s) }}s</td>
  </tr>
{% endfor %}
</tbody>
</table>

<div style="display:flex;gap:1rem;flex-wrap:wrap;margin:1.5rem 0;">
  <div style="flex:1;min-width:300px;">
    <strong>Total wall time:</strong> {{ "%.1f"|format(total_wall_time) }}s
    &nbsp;|&nbsp; <strong>Grid points:</strong> {{ "{:,}".format(total_grid_points) }}
    &nbsp;|&nbsp; <strong>Fields:</strong> {{ total_fields }}
  </div>
  <div>
    <span class="badge {{ 'badge-success' if all_valid else 'badge-warn' }}">
      All valid: {{ all_valid }}
    </span>
    <span class="badge {{ 'badge-success' if all_claims else 'badge-warn' }}">
      All claims met: {{ all_claims }}
    </span>
  </div>
</div>

<!-- ══════════════ SUMMARY CHARTS ══════════════ -->
<h2>Cross-Scenario Analysis</h2>

{% if b64_claims_matrix %}
<h3>Claims Satisfaction Matrix</h3>
<img class="summary-img" src="data:image/png;base64,{{ b64_claims_matrix }}"
     alt="Claims Matrix">
{% endif %}

{% if b64_performance_bars %}
<h3>Performance Comparison</h3>
<img class="summary-img" src="data:image/png;base64,{{ b64_performance_bars }}"
     alt="Performance Bars">
{% endif %}

<!-- ══════════════ SCENARIO DETAILS ══════════════ -->
<h2>Scenario Details</h2>

{% for sc in scenarios %}
{% if sc.status == 'success' %}
<div class="card">
  <div class="card-header">
    <h3>[{{ sc.scenario_index }}] {{ sc.name }}</h3>
    {% if sc.validation.valid %}
      <span class="badge badge-success">VALID</span>
    {% else %}
      <span class="badge badge-fail">INVALID</span>
    {% endif %}
  </div>

  <p class="desc">{{ sc.description }}</p>

  <!-- Compilation parameters -->
  <h3>Compilation</h3>
  <div class="params-grid">
    <div class="param-item">
      <div class="param-label">Domain</div>
      <div class="param-value">{{ sc.compilation.domain }}</div>
    </div>
    <div class="param-item">
      <div class="param-label">Reynolds Number</div>
      <div class="param-value">{{ "%.2e"|format(sc.compilation.reynolds_number) }}</div>
    </div>
    <div class="param-item">
      <div class="param-label">Mach Number</div>
      <div class="param-value">{{ "%.4f"|format(sc.compilation.mach_number) }}</div>
    </div>
    <div class="param-item">
      <div class="param-label">Fluid</div>
      <div class="param-value">{{ sc.compilation.fluid_name }}</div>
    </div>
    <div class="param-item">
      <div class="param-label">Geometry</div>
      <div class="param-value">{{ sc.compilation.geometry_type }}</div>
    </div>
    <div class="param-item">
      <div class="param-label">Recommended n_bits</div>
      <div class="param-value">{{ sc.compilation.n_bits }} ({{ 2**sc.compilation.n_bits }}×{{ 2**sc.compilation.n_bits }})</div>
    </div>
    <div class="param-item">
      <div class="param-label">Executed n_bits</div>
      <div class="param-value">{{ sc.execution_overrides.n_bits }} ({{ 2**sc.execution_overrides.n_bits }}×{{ 2**sc.execution_overrides.n_bits }})</div>
    </div>
    <div class="param-item">
      <div class="param-label">Steps</div>
      <div class="param-value">{{ sc.execution_overrides.n_steps }} (of {{ sc.compilation.n_steps }})</div>
    </div>
    <div class="param-item">
      <div class="param-label">Quality Tier</div>
      <div class="param-value">{{ sc.compilation.quality_tier }}</div>
    </div>
    <div class="param-item">
      <div class="param-label">Char. Length</div>
      <div class="param-value">{{ "%.4f"|format(sc.compilation.characteristic_length) }} m</div>
    </div>
  </div>

  {% if sc.compilation.warnings %}
  <h3>Advisor Warnings</h3>
  <ul class="warning-list">
    {% for w in sc.compilation.warnings %}
    <li>{{ w }}</li>
    {% endfor %}
  </ul>
  {% endif %}

  <!-- Performance -->
  <h3>Execution Performance</h3>
  {% set perf = sc.result.performance %}
  <div class="params-grid">
    <div class="param-item">
      <div class="param-label">Wall Time</div>
      <div class="param-value">{{ "%.3f"|format(perf.wall_time_s) }}s</div>
    </div>
    <div class="param-item">
      <div class="param-label">Grid Points</div>
      <div class="param-value">{{ "{:,}".format(perf.grid_points) }}</div>
    </div>
    <div class="param-item">
      <div class="param-label">Time Steps</div>
      <div class="param-value">{{ perf.time_steps }}</div>
    </div>
    <div class="param-item">
      <div class="param-label">Throughput</div>
      <div class="param-value">{{ "{:,.0f}".format(perf.throughput_gp_per_s) }} gp/s</div>
    </div>
  </div>

  <!-- Field plots -->
  {% set sc_b64 = base64_images.get(sc.scenario_index, {}) %}
  {% if sc_b64 %}
  <h3>Field Contours</h3>
  <div class="field-plots">
    {% for fname, b64 in sc_b64.items() %}
    {% if fname not in ['diagnostics'] %}
    <img src="data:image/png;base64,{{ b64 }}" alt="{{ fname }}">
    {% endif %}
    {% endfor %}
  </div>
  {% endif %}

  {% if sc_b64.get('diagnostics') %}
  <h3>Time-Step Diagnostics</h3>
  <img class="summary-img" src="data:image/png;base64,{{ sc_b64.diagnostics }}"
       alt="Diagnostics">
  {% endif %}

  <!-- Conservation -->
  {% if sc.result.conservation %}
  <h3>Conservation</h3>
  {% set cons = sc.result.conservation %}
  <div class="params-grid">
    <div class="param-item">
      <div class="param-label">Quantity</div>
      <div class="param-value">{{ cons.quantity }}</div>
    </div>
    <div class="param-item">
      <div class="param-label">Initial → Final</div>
      <div class="param-value">{{ "%.6e"|format(cons.initial_value) }} → {{ "%.6e"|format(cons.final_value) }}</div>
    </div>
    <div class="param-item">
      <div class="param-label">Relative Error</div>
      <div class="param-value">{{ "%.2e"|format(cons.relative_error) }}</div>
    </div>
    <div class="param-item">
      <div class="param-label">Status</div>
      <div class="param-value">
        <span class="badge {{ 'badge-success' if cons.status == 'conserved' else 'badge-fail' }}">
          {{ cons.status }}
        </span>
      </div>
    </div>
  </div>
  {% endif %}

  <!-- Validation checks -->
  <h3>Validation Checks</h3>
  <ul class="check-list">
    {% for chk in sc.validation.checks %}
    <li class="{{ 'check-pass' if chk.passed else 'check-fail' }}">
      <strong>{{ chk.name }}</strong>: {{ chk.detail }}
    </li>
    {% endfor %}
  </ul>

  <!-- Claims -->
  <h3>Trust Claims</h3>
  <ul class="check-list">
    {% for cl in sc.claims %}
    <li class="{{ 'check-pass' if cl.satisfied else 'check-fail' }}">
      <strong>[{{ cl.tag }}]</strong> {{ cl.claim }}
    </li>
    {% endfor %}
  </ul>

  <!-- Certificate -->
  <h3>Certificate</h3>
  <div class="cert-box">
    <strong>Job ID:</strong> {{ sc.certificate.job_id }}<br>
    <strong>Issued:</strong> {{ sc.certificate.issued_at }}<br>
    <strong>Signature:</strong> {{ sc.certificate.signature }}<br>
    <strong>Result Hash:</strong> {{ sc.hashes.result }}<br>
    <strong>Config Hash:</strong> {{ sc.hashes.config }}
  </div>
</div>
{% endif %}
{% endfor %}

<!-- ══════════════ FOOTER ══════════════ -->
<div class="footer">
  Physics OS — The Ontic Engine &nbsp;|&nbsp;
  Tigantic Holdings LLC &nbsp;|&nbsp;
  Report generated {{ generated_at }}
</div>

</div>
</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════════
# Report generator
# ═══════════════════════════════════════════════════════════════════


def _build_context(
    data: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Build the Jinja2 template context from scenario data + manifest."""
    summary = data.get("summary", {})
    meta = data.get("meta", {})

    return {
        "title": meta.get("title", "Physics OS — Scenario Report"),
        "generated_at": meta.get("generated_at", datetime.now(timezone.utc).isoformat()),
        "runtime_version": meta.get("runtime_version", ""),
        "platform_version": meta.get("platform_version", ""),
        "total_scenarios": summary.get("total_scenarios", 0),
        "scenario_table": summary.get("scenario_table", []),
        "total_wall_time": summary.get("total_wall_time_s", 0),
        "total_grid_points": summary.get("total_grid_points_computed", 0),
        "total_fields": summary.get("total_fields_produced", 0),
        "all_valid": summary.get("all_validations_passed", False),
        "all_claims": summary.get("all_claims_satisfied", False),
        "scenarios": data.get("scenarios", []),
        "base64_images": manifest.get("base64", {}),
        "b64_claims_matrix": manifest.get("base64", {}).get("claims_matrix", ""),
        "b64_performance_bars": manifest.get("base64", {}).get("performance_bars", ""),
    }


def generate_html_report(
    data: dict[str, Any],
    manifest: dict[str, Any],
    output_path: Path,
) -> Path:
    """Generate a standalone HTML report with embedded images.

    Parameters
    ----------
    data : dict
        The combined scenario data (from scenario_results.json).
    manifest : dict
        The visualization manifest from ``generate_all_visuals()``.
    output_path : Path
        Where to write the HTML file.

    Returns
    -------
    Path
        Path to the generated HTML file.
    """
    env = Environment(loader=BaseLoader())
    template = env.from_string(_HTML_TEMPLATE)
    ctx = _build_context(data, manifest)

    html = template.render(**ctx)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Wrote HTML report: %s (%d bytes)", output_path, len(html))
    return output_path


def generate_pdf_report(
    html_path: Path,
    output_path: Path,
) -> Path:
    """Convert an HTML report to PDF via WeasyPrint.

    Parameters
    ----------
    html_path : Path
        Path to the HTML source.
    output_path : Path
        Where to write the PDF.

    Returns
    -------
    Path
        Path to the generated PDF file.
    """
    try:
        from weasyprint import HTML as WeasyprintHTML
        WeasyprintHTML(filename=str(html_path)).write_pdf(str(output_path))
        logger.info("Wrote PDF report: %s", output_path)
        return output_path
    except ImportError:
        logger.warning("WeasyPrint not available — skipping PDF generation.")
        return html_path
    except Exception as exc:
        logger.warning("PDF generation failed: %s", exc)
        return html_path

