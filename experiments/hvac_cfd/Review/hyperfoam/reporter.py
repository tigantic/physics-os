#!/usr/bin/env python3
"""DOMINION Reporter - Forensic Report Generation

Phase 8.3: The Artifact
Generates legally-defensible reports with full provenance chain.

Outputs:
- PDF: Professional report with figures, tables, compliance checks
- PNG: High-resolution renders from simulation
- JSON: Provenance record for reproducibility audit

Author: HyperTensor Physics Laboratory
Copyright (c) 2025 TiganticLabz. All Rights Reserved.
SPDX-License-Identifier: Proprietary
"""

import json
import hashlib
import datetime
import shutil
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any
import argparse
import sys

# Optional dependencies - graceful degradation
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.units import mm, inch
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, PageBreak, KeepTogether
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ProvenanceRecord:
    """Immutable record of simulation provenance for legal defensibility."""
    timestamp: str
    software_version: str
    solver_version: str
    geometry_hash: str
    grid_resolution: List[int]
    sim_time_range: List[float]
    turbulence_model: str
    buoyancy_enabled: bool
    radiation_enabled: bool
    time_step: float
    cfl_target: float
    random_seed: Optional[int] = None
    operator_id: str = "SYSTEM"
    attestation_hash: Optional[str] = None


@dataclass
class ComfortMetrics:
    """T1: Thermal Comfort results."""
    pmv_mean: float
    pmv_range: List[float]
    ppd_mean: float
    ppd_max: float
    comfort_zone_pct: float  # % of volume in -0.5 < PMV < +0.5
    ashrae_compliant: bool
    metabolic_rate: float  # met
    clothing_level: float  # clo


@dataclass
class AirflowMetrics:
    """T2/T3: HVAC and Physics results."""
    velocity_max: float  # m/s
    velocity_mean: float
    temperature_range: List[float]  # [min, max] in Celsius
    stratification: float  # delta-T floor-to-ceiling
    draft_risk_zones: int  # count of zones with v > 0.25 m/s


@dataclass  
class DataCenterMetrics:
    """T4: Data Center Thermal Management."""
    supply_temp: float
    return_temp: float
    delta_t: float
    hot_spots: int  # count of cells > threshold
    cold_spots: int
    cooling_efficiency: float  # % of airflow reaching equipment


@dataclass
class FireMetrics:
    """T5: Fire & Smoke results."""
    aset: float  # Available Safe Egress Time (seconds)
    rset: float  # Required Safe Egress Time
    aset_rset_margin: float
    visibility_min: float  # meters at 1.8m height
    temp_at_head_height: float
    tenability_maintained: bool


@dataclass
class ReportData:
    """Complete report data bundle."""
    project_name: str
    report_type: str
    generated_at: str
    provenance: ProvenanceRecord
    comfort: Optional[ComfortMetrics] = None
    airflow: Optional[AirflowMetrics] = None
    datacenter: Optional[DataCenterMetrics] = None
    fire: Optional[FireMetrics] = None
    renders: List[Path] = field(default_factory=list)
    notes: str = ""


# ============================================================================
# PROVENANCE GENERATION
# ============================================================================

def compute_file_hash(filepath: Path) -> str:
    """Compute SHA-256 hash of file for provenance."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def create_provenance(
    geometry_path: Path,
    grid_resolution: List[int],
    sim_time_range: List[float],
    params: Dict[str, Any]
) -> ProvenanceRecord:
    """Create immutable provenance record."""
    return ProvenanceRecord(
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        software_version="DOMINION 0.1.0",
        solver_version="hyperfoam 0.4.2",
        geometry_hash=compute_file_hash(geometry_path) if geometry_path.exists() else "NO_GEOMETRY",
        grid_resolution=grid_resolution,
        sim_time_range=sim_time_range,
        turbulence_model=params.get("turbulence_model", "k-epsilon"),
        buoyancy_enabled=params.get("buoyancy", True),
        radiation_enabled=params.get("radiation_p1", False),
        time_step=params.get("dt", 0.01),
        cfl_target=params.get("cfl", 0.8),
        random_seed=params.get("seed"),
        operator_id=params.get("operator", "SYSTEM"),
    )


def sign_provenance(record: ProvenanceRecord) -> str:
    """Create attestation hash of provenance record."""
    record_bytes = json.dumps(asdict(record), sort_keys=True).encode()
    return hashlib.sha256(record_bytes).hexdigest()[:16]


# ============================================================================
# PDF GENERATION (requires reportlab)
# ============================================================================

def create_pdf_report(data: ReportData, output_path: Path) -> bool:
    """Generate professional PDF report."""
    if not HAS_REPORTLAB:
        print("WARNING: reportlab not installed, skipping PDF generation")
        return False
    
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=20*mm,
        leftMargin=20*mm,
        topMargin=20*mm,
        bottomMargin=20*mm
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DominionTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=20,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#00FFAA")
    )
    
    heading_style = ParagraphStyle(
        'DominionHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=10,
        textColor=colors.HexColor("#00D9FF")
    )
    
    body_style = ParagraphStyle(
        'DominionBody',
        parent=styles['Normal'],
        fontSize=10,
        leading=14
    )
    
    elements = []
    
    # === TITLE PAGE ===
    elements.append(Spacer(1, 50*mm))
    elements.append(Paragraph("DOMINION", title_style))
    elements.append(Paragraph("Forensic Simulation Report", styles['Heading2']))
    elements.append(Spacer(1, 20*mm))
    elements.append(Paragraph(f"<b>Project:</b> {data.project_name}", body_style))
    elements.append(Paragraph(f"<b>Report Type:</b> {data.report_type}", body_style))
    elements.append(Paragraph(f"<b>Generated:</b> {data.generated_at}", body_style))
    elements.append(PageBreak())
    
    # === PROVENANCE SECTION ===
    elements.append(Paragraph("1. PROVENANCE CHAIN", heading_style))
    
    prov = data.provenance
    prov_data = [
        ["Timestamp", prov.timestamp],
        ["Software", prov.software_version],
        ["Solver", prov.solver_version],
        ["Geometry Hash", prov.geometry_hash[:32] + "..."],
        ["Grid Resolution", f"{prov.grid_resolution}"],
        ["Sim Time Range", f"{prov.sim_time_range}"],
        ["Turbulence Model", prov.turbulence_model],
        ["Buoyancy", "Enabled" if prov.buoyancy_enabled else "Disabled"],
        ["Radiation P-1", "Enabled" if prov.radiation_enabled else "Disabled"],
    ]
    
    prov_table = Table(prov_data, colWidths=[50*mm, 100*mm])
    prov_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#1a1a2e")),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#333355")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(prov_table)
    elements.append(Spacer(1, 10*mm))
    
    # === COMFORT RESULTS (if applicable) ===
    if data.comfort:
        elements.append(Paragraph("2. THERMAL COMFORT ANALYSIS", heading_style))
        c = data.comfort
        
        # Compliance status
        status = "✓ COMPLIANT" if c.ashrae_compliant else "✗ NON-COMPLIANT"
        status_color = colors.green if c.ashrae_compliant else colors.red
        elements.append(Paragraph(
            f"<b>ASHRAE 55 Status:</b> <font color='{status_color}'>{status}</font>",
            body_style
        ))
        
        comfort_data = [
            ["Metric", "Value", "Threshold"],
            ["PMV Mean", f"{c.pmv_mean:+.2f}", "-0.5 to +0.5"],
            ["PMV Range", f"{c.pmv_range[0]:+.2f} to {c.pmv_range[1]:+.2f}", "-3 to +3"],
            ["PPD Mean", f"{c.ppd_mean:.1f}%", "< 10%"],
            ["PPD Max", f"{c.ppd_max:.1f}%", "< 20%"],
            ["Comfort Zone", f"{c.comfort_zone_pct:.1f}%", "> 80%"],
            ["Metabolic Rate", f"{c.metabolic_rate:.1f} met", "Input"],
            ["Clothing Level", f"{c.clothing_level:.2f} clo", "Input"],
        ]
        
        comfort_table = Table(comfort_data, colWidths=[50*mm, 45*mm, 45*mm])
        comfort_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#16213e")),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#333355")),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ]))
        elements.append(comfort_table)
        elements.append(Spacer(1, 10*mm))
    
    # === AIRFLOW RESULTS ===
    if data.airflow:
        elements.append(Paragraph("3. AIRFLOW VALIDATION", heading_style))
        a = data.airflow
        
        airflow_data = [
            ["Metric", "Value"],
            ["Max Velocity", f"{a.velocity_max:.2f} m/s"],
            ["Mean Velocity", f"{a.velocity_mean:.2f} m/s"],
            ["Temperature Range", f"{a.temperature_range[0]:.1f}°C to {a.temperature_range[1]:.1f}°C"],
            ["Stratification (ΔT)", f"{a.stratification:.1f}°C"],
            ["Draft Risk Zones", f"{a.draft_risk_zones}"],
        ]
        
        airflow_table = Table(airflow_data, colWidths=[60*mm, 60*mm])
        airflow_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#16213e")),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#333355")),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ]))
        elements.append(airflow_table)
        elements.append(Spacer(1, 10*mm))
    
    # === FIRE RESULTS ===
    if data.fire:
        elements.append(Paragraph("4. FIRE & EGRESS ANALYSIS", heading_style))
        f = data.fire
        
        # ASET/RSET verdict
        if f.aset_rset_margin > 60:
            verdict = "✓ SAFE MARGIN (> 60s)"
            verdict_color = colors.green
        elif f.aset_rset_margin > 0:
            verdict = "⚠ MARGINAL"
            verdict_color = colors.orange
        else:
            verdict = "✗ UNSAFE"
            verdict_color = colors.red
        
        elements.append(Paragraph(
            f"<b>ASET/RSET Verdict:</b> <font color='{verdict_color}'>{verdict}</font>",
            body_style
        ))
        
        fire_data = [
            ["Metric", "Value", "Threshold"],
            ["ASET", f"{f.aset:.0f} s", "-"],
            ["RSET", f"{f.rset:.0f} s", "-"],
            ["Margin", f"{f.aset_rset_margin:.0f} s", "> 60 s recommended"],
            ["Min Visibility", f"{f.visibility_min:.1f} m", "> 10 m"],
            ["Temp at Head", f"{f.temp_at_head_height:.0f}°C", "< 60°C"],
            ["Tenability", "Maintained" if f.tenability_maintained else "LOST", "-"],
        ]
        
        fire_table = Table(fire_data, colWidths=[45*mm, 45*mm, 50*mm])
        fire_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#3d1c1c")),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#553333")),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ]))
        elements.append(fire_table)
        elements.append(Spacer(1, 10*mm))
    
    # === FIGURES ===
    if data.renders:
        elements.append(PageBreak())
        elements.append(Paragraph("RENDERED VIEWS", heading_style))
        
        for i, render_path in enumerate(data.renders):
            if render_path.exists() and HAS_PIL:
                # Scale to fit page
                img = RLImage(str(render_path), width=160*mm, height=100*mm)
                elements.append(img)
                elements.append(Paragraph(f"Figure {i+1}: {render_path.stem}", body_style))
                elements.append(Spacer(1, 5*mm))
    
    # === SIGNATURE BLOCK ===
    elements.append(PageBreak())
    elements.append(Paragraph("ATTESTATION", heading_style))
    elements.append(Paragraph(
        "This report was automatically generated by DOMINION Forensic Simulation Platform. "
        "All results are reproducible given the input geometry, parameters, and solver version "
        "recorded in the provenance chain above.",
        body_style
    ))
    elements.append(Spacer(1, 10*mm))
    
    attestation_hash = sign_provenance(data.provenance)
    elements.append(Paragraph(f"<b>Attestation Hash:</b> {attestation_hash}", body_style))
    elements.append(Paragraph(f"<b>Generated:</b> {data.generated_at}", body_style))
    
    # Build PDF
    doc.build(elements)
    return True


# ============================================================================
# JSON EXPORT
# ============================================================================

def export_json(data: ReportData, output_path: Path) -> bool:
    """Export raw data as JSON for programmatic access."""
    
    def serialize(obj):
        if isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return {k: serialize(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
        elif isinstance(obj, list):
            return [serialize(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        return obj
    
    with open(output_path, 'w') as f:
        json.dump(serialize(data), f, indent=2)
    
    return True


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def generate_report(
    project_name: str,
    report_type: str,
    output_dir: Path,
    geometry_path: Optional[Path] = None,
    results_path: Optional[Path] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Path:
    """Generate complete report package.
    
    Returns the path to the output folder containing all artifacts.
    """
    # Ensure output exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now()
    generated_at = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    # Load results if path provided
    comfort = None
    airflow = None
    datacenter = None
    fire = None
    
    if results_path and results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        
        # Parse based on report type
        if 'comfort' in results:
            c = results['comfort']
            comfort = ComfortMetrics(**c)
        if 'airflow' in results:
            a = results['airflow']
            airflow = AirflowMetrics(**a)
        if 'fire' in results:
            f_data = results['fire']
            fire = FireMetrics(**f_data)
    else:
        # ═══════════════════════════════════════════════════════════════════════
        # WARNING: PLACEHOLDER DATA FOR TESTING ONLY
        # This code generates synthetic metrics when no results file exists.
        # DO NOT use this for client deliverables.
        # ═══════════════════════════════════════════════════════════════════════
        import warnings
        warnings.warn(
            "⚠️ GENERATING PLACEHOLDER DATA - No simulation results file found. "
            "Run a real simulation before generating client reports.",
            UserWarning
        )
        print("⚠️" * 20)
        print("⚠️ WARNING: USING PLACEHOLDER DATA - NOT FROM REAL SIMULATION")
        print("⚠️" * 20)
        
        if report_type in ('Comfort Analysis', 'Full Forensic'):
            comfort = ComfortMetrics(
                pmv_mean=0.3,
                pmv_range=[-0.5, 1.2],
                ppd_mean=8.5,
                ppd_max=23.0,
                comfort_zone_pct=72.5,
                ashrae_compliant=False,
                metabolic_rate=1.2,
                clothing_level=0.65
            )
        
        if report_type in ('Airflow Validation', 'Full Forensic'):
            airflow = AirflowMetrics(
                velocity_max=1.8,
                velocity_mean=0.15,
                temperature_range=[19.5, 24.2],
                stratification=2.3,
                draft_risk_zones=3
            )
        
        if report_type in ('Fire & Egress', 'Full Forensic'):
            fire = FireMetrics(
                aset=180.0,
                rset=120.0,
                aset_rset_margin=60.0,
                visibility_min=8.5,
                temp_at_head_height=45.0,
                tenability_maintained=True
            )
    
    # Create provenance
    params = params or {}
    if geometry_path and geometry_path.exists():
        provenance = create_provenance(
            geometry_path,
            grid_resolution=params.get('grid_resolution', [128, 128, 64]),
            sim_time_range=params.get('sim_time_range', [0.0, 300.0]),
            params=params
        )
    else:
        provenance = ProvenanceRecord(
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            software_version="DOMINION 0.1.0",
            solver_version="hyperfoam 0.4.2",
            geometry_hash="NO_GEOMETRY_PROVIDED",
            grid_resolution=[128, 128, 64],
            sim_time_range=[0.0, 300.0],
            turbulence_model="k-epsilon",
            buoyancy_enabled=True,
            radiation_enabled=False,
            time_step=0.01,
            cfl_target=0.8
        )
    
    # Sign provenance
    provenance.attestation_hash = sign_provenance(provenance)
    
    # Collect renders (look for PNG files in output_dir)
    renders = list(output_dir.glob("*.png"))
    
    # Assemble report data
    data = ReportData(
        project_name=project_name,
        report_type=report_type,
        generated_at=generated_at,
        provenance=provenance,
        comfort=comfort,
        airflow=airflow,
        datacenter=datacenter,
        fire=fire,
        renders=renders
    )
    
    # Generate outputs
    pdf_path = output_dir / f"{project_name}_Report.pdf"
    json_path = output_dir / f"{project_name}_Data.json"
    prov_path = output_dir / f"{project_name}_Provenance.json"
    
    # PDF
    if HAS_REPORTLAB:
        if create_pdf_report(data, pdf_path):
            print(f"✓ PDF Report: {pdf_path}")
    else:
        print("⚠ reportlab not installed - PDF skipped")
    
    # JSON data
    export_json(data, json_path)
    print(f"✓ JSON Data: {json_path}")
    
    # Provenance record (separate for easier verification)
    with open(prov_path, 'w') as f:
        json.dump(asdict(provenance), f, indent=2)
    print(f"✓ Provenance: {prov_path}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="DOMINION Report Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate comfort report with demo data
  python reporter.py --project "Office_Building" --type "Comfort Analysis"
  
  # Full forensic with geometry hash
  python reporter.py --project "Datacenter_A" --type "Full Forensic" --geometry mesh.obj
        """
    )
    
    parser.add_argument('--project', '-p', required=True, help='Project name')
    parser.add_argument('--type', '-t', default='Comfort Analysis',
                       choices=['Comfort Analysis', 'Airflow Validation', 
                               'Data Center Audit', 'Fire & Egress', 'Full Forensic'])
    parser.add_argument('--output', '-o', default='./reports', help='Output directory')
    parser.add_argument('--geometry', '-g', help='Path to geometry file (for hash)')
    parser.add_argument('--results', '-r', help='Path to results JSON')
    
    args = parser.parse_args()
    
    # Create timestamped output folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / f"{args.project}_{timestamp}"
    
    geometry_path = Path(args.geometry) if args.geometry else None
    results_path = Path(args.results) if args.results else None
    
    output_path = generate_report(
        project_name=args.project,
        report_type=args.type,
        output_dir=output_dir,
        geometry_path=geometry_path,
        results_path=results_path
    )
    
    print(f"\n✓ Report generated: {output_path}")


if __name__ == '__main__':
    main()
