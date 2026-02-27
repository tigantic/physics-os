"""
HyperFOAM Production Pipeline

Transforms client job specifications into engineering deliverables.
No sliders. No demos. Pure business logic.

Input:  job_spec.json (Client requirements)
Output: PDF Report + Validation Data

Usage:
    python -m hyperfoam.pipeline projects/2026-001_Apex_HQ/job_spec.json
"""

import json
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

# Ensure hyperfoam is importable
_pkg_root = Path(__file__).parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))


# =============================================================================
# PROGRESS HELPER - Structured output for UI parsing
# =============================================================================
def emit_progress(phase: str, pct: int, message: str = ""):
    """
    Emit structured progress that the UI can parse reliably.
    
    Format: [PROGRESS] phase=X pct=Y | message
    
    Phases: init, baseline, optimize, validate, report, complete
    pct: 0-100 overall progress
    """
    print(f"[PROGRESS] phase={phase} pct={pct} | {message}", flush=True)


def emit_status(message: str):
    """Emit a status message."""
    print(f"[STATUS] {message}", flush=True)


@dataclass
class JobSpec:
    """Parsed job specification from JSON."""
    
    client_name: str
    project_id: str
    contact: str
    room_name: str
    room_type: str
    lx: float
    ly: float
    lz: float
    n_occupants: int
    heat_per_person: float
    equipment_load: float
    lighting_load: float
    supply_temp: float
    supply_velocity: float
    supply_angle: float
    num_diffusers: int
    diffuser_area: float
    max_velocity: float
    target_temp: float
    temp_min: float
    temp_max: float
    max_co2: float
    generate_heatmap: bool
    generate_velocity: bool
    generate_convergence: bool
    generate_pdf: bool
    notes: str
    job_dir: Path
    
    @classmethod
    def from_json(cls, filepath: Path) -> 'JobSpec':
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        client = data['client']
        room = data['room']
        load = data['load']
        hvac = data.get('hvac', {})
        constraints = data['constraints']
        deliverables = data.get('deliverables', {})
        dims = room['dimensions_m']
        temp_range = constraints.get('temp_range_c', [20.0, 24.0])
        
        # Handle both max_velocity_ms and max_draft_ms (UI uses both names)
        max_vel = constraints.get('max_velocity_ms') or constraints.get('max_draft_ms', 0.25)
        target_temp = constraints.get('target_temp_c', (temp_range[0] + temp_range[1]) / 2)
        max_co2 = constraints.get('max_co2_ppm', 1000.0)
        
        return cls(
            client_name=client['name'],
            project_id=client['project_id'],
            contact=client.get('contact', 'Unknown'),
            room_name=room.get('name', 'Conference Room'),
            room_type=room.get('type', 'conference_room'),
            lx=dims[0], ly=dims[1], lz=dims[2],
            n_occupants=load['occupants'],
            heat_per_person=load.get('heat_load_per_person_watts', 100),
            equipment_load=load.get('equipment_load_watts', 0),
            lighting_load=load.get('lighting_load_watts', 0),
            supply_temp=hvac.get('supply_temp_c', 18.0),
            supply_velocity=hvac.get('supply_velocity_ms', 1.0),
            supply_angle=hvac.get('supply_angle_deg', 45.0),
            num_diffusers=hvac.get('num_diffusers', 2),
            diffuser_area=hvac.get('diffuser_area_m2', 0.1),
            max_velocity=max_vel,
            target_temp=target_temp,
            temp_min=temp_range[0],
            temp_max=temp_range[1],
            max_co2=max_co2,
            generate_heatmap=deliverables.get('thermal_heatmap', True) or deliverables.get('heatmap', True),
            generate_velocity=deliverables.get('velocity_field', True),
            generate_convergence=deliverables.get('convergence_plot', True),
            generate_pdf=deliverables.get('pdf_report', True),
            notes=data.get('notes', ''),
            job_dir=filepath.parent
        )
    
    @property
    def total_heat_load(self) -> float:
        return self.n_occupants * self.heat_per_person + self.equipment_load + self.lighting_load
    
    @property
    def room_volume(self) -> float:
        return self.lx * self.ly * self.lz


def print_header(spec: JobSpec):
    """Print production job header."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                      HYPERFOAM PRODUCTION PIPELINE                            ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Mode: TRANSACTIONAL (No Demo - Pure Business Logic)                         ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """, flush=True)
    print(f"  Job ID:     {spec.project_id}", flush=True)
    print(f"  Client:     {spec.client_name}", flush=True)
    print(f"  Room:       {spec.room_name}", flush=True)
    print(f"  Geometry:   {spec.lx:.2f}m x {spec.ly:.2f}m x {spec.lz:.2f}m ({spec.room_volume:.0f} m³)", flush=True)
    print(f"  Occupants:  {spec.n_occupants} persons", flush=True)
    print(f"  Heat Load:  {spec.total_heat_load:.0f} W", flush=True)
    print(f"  Constraints: T={spec.temp_min:.1f}-{spec.temp_max:.1f}°C, V<{spec.max_velocity}m/s, CO2<{spec.max_co2:.0f}ppm", flush=True)
    sys.stdout.flush()


def run_production_pipeline(job_file: str, skip_optimize: bool = False, 
                            sim_duration: float = 60.0) -> Dict[str, Any]:
    """Execute the full production pipeline."""
    
    # =========================================================================
    # PHASE 1: INITIALIZATION (0-10%)
    # =========================================================================
    emit_progress("init", 0, "Starting HyperFOAM Pipeline")
    emit_status("Loading CFD engine...")
    
    from hyperfoam import Solver, ConferenceRoom
    from hyperfoam.presets import setup_conference_room
    from hyperfoam.optimizer import HVACOptimizer, OptimizationTarget, OptimizationBounds
    from hyperfoam.report import EngineeringReport
    
    emit_progress("init", 5, "CFD engine loaded")
    
    job_path = Path(job_file)
    if not job_path.exists():
        emit_status(f"ERROR: Job file not found: {job_file}")
        sys.exit(1)
    
    spec = JobSpec.from_json(job_path)
    print_header(spec)
    
    emit_progress("init", 10, "Job loaded")
    
    pipeline_start = time.time()
    results = {'spec': spec, 'phases': {}}
    output_dir = spec.job_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    # =========================================================================
    # PHASE 2: BASELINE SIMULATION (10-30%)
    # =========================================================================
    emit_progress("baseline", 10, "Starting baseline simulation")
    print("\n" + "="*70, flush=True)
    print("  PHASE 1: BASELINE (Client's Configuration)", flush=True)
    print("="*70, flush=True)
    print(f"  Settings: V={spec.supply_velocity:.2f}m/s, A={spec.supply_angle:.1f}°, T={spec.supply_temp:.1f}°C", flush=True)
    
    config = ConferenceRoom()
    config.lx, config.ly, config.lz = spec.lx, spec.ly, spec.lz
    config.supply_velocity = spec.supply_velocity
    config.supply_angle = spec.supply_angle
    config.supply_temp = spec.supply_temp
    
    baseline_solver = Solver(config)
    setup_conference_room(baseline_solver, n_occupants=spec.n_occupants)
    
    baseline_duration = 60.0
    
    def baseline_cb(t, m):
        pct = 10 + int((t / baseline_duration) * 20)
        emit_progress("baseline", pct, f"t={t:.0f}s T={m['T']:.1f}°C")
    
    baseline_solver.solve(duration=baseline_duration, callback=baseline_cb, 
                         log_interval=15.0, verbose=False)
    baseline_metrics = baseline_solver.get_comfort_metrics()
    baseline_pass = baseline_metrics['overall_pass']
    
    emit_progress("baseline", 30, f"Baseline {'PASS' if baseline_pass else 'FAIL'}")
    print(f"  Result: T={baseline_metrics['temperature']:.1f}°C "
          f"CO2={baseline_metrics['co2']:.0f}ppm V={baseline_metrics['velocity']:.3f}m/s "
          f"[{'PASS' if baseline_pass else 'FAIL'}]", flush=True)
    
    results['phases']['baseline'] = {'pass': baseline_pass, 'metrics': baseline_metrics}
    
    # =========================================================================
    # PHASE 3: OPTIMIZATION (30-60%)
    # =========================================================================
    if baseline_pass:
        emit_progress("optimize", 30, "Baseline PASSED - skipping optimization")
        optimal_velocity = spec.supply_velocity
        optimal_angle = spec.supply_angle
        optimal_supply_temp = spec.supply_temp
        emit_progress("optimize", 60, "Using client settings")
    else:
        emit_progress("optimize", 30, "Baseline FAILED - optimizing")
        print("\n" + "="*70, flush=True)
        print("  PHASE 2: OPTIMIZATION", flush=True)
        print("="*70, flush=True)
        
        targets = OptimizationTarget(
            temp_min=spec.temp_min, temp_max=spec.temp_max,
            temp_target=spec.target_temp, co2_max=spec.max_co2,
            velocity_max=spec.max_velocity
        )
        
        optimizer = HVACOptimizer(
            n_occupants=spec.n_occupants, targets=targets,
            sim_duration=60.0, validation_duration=60.0,
            supply_temp=spec.supply_temp, verbose=False
        )
        
        # Progress callback for optimizer's internal validation
        def opt_progress(pct: int, msg: str):
            emit_progress("optimize", pct, msg)
        
        emit_progress("optimize", 40, "Computing optimal settings...")
        opt_result = optimizer.optimize(method='hybrid',
                                        initial_guess=(spec.supply_velocity, spec.supply_angle),
                                        progress_callback=opt_progress)
        
        optimal_velocity = opt_result.optimal_velocity
        optimal_angle = opt_result.optimal_angle
        optimal_supply_temp = opt_result.optimal_supply_temp
        
        emit_progress("optimize", 60, f"Optimal: V={optimal_velocity:.2f}m/s A={optimal_angle:.0f}°")
        print(f"  Optimal: V={optimal_velocity:.2f}m/s, A={optimal_angle:.1f}°, T={optimal_supply_temp:.1f}°C", flush=True)
    
    # =========================================================================
    # PHASE 4: VALIDATION (60-85%)
    # =========================================================================
    emit_progress("validate", 60, "Starting validation")
    print("\n" + "="*70, flush=True)
    print("  PHASE 3: VALIDATION SIMULATION", flush=True)
    print("="*70, flush=True)
    
    config = ConferenceRoom()
    config.lx, config.ly, config.lz = spec.lx, spec.ly, spec.lz
    config.supply_velocity = optimal_velocity
    config.supply_angle = optimal_angle
    config.supply_temp = optimal_supply_temp
    
    solver = Solver(config)
    setup_conference_room(solver, n_occupants=spec.n_occupants)
    
    print(f"  Grid: {config.nx}x{config.ny}x{config.nz}, Device: {solver.device}", flush=True)
    
    from collections import deque
    history = {'time': deque(maxlen=10000), 'T': deque(maxlen=10000),
               'CO2': deque(maxlen=10000), 'V': deque(maxlen=10000)}
    
    log_interval = max(10.0, sim_duration / 10)
    
    def val_cb(t, m):
        history['time'].append(t)
        history['T'].append(m['T'])
        history['CO2'].append(m['CO2'])
        history['V'].append(m['V'])
        pct = 60 + int((t / sim_duration) * 25)
        emit_progress("validate", pct, f"t={t:.0f}s T={m['T']:.1f}°C")
    
    sim_start = time.time()
    solver.solve(duration=sim_duration, callback=val_cb, log_interval=log_interval, verbose=False)
    sim_elapsed = time.time() - sim_start
    
    metrics = solver.get_comfort_metrics()
    emit_progress("validate", 85, f"Validation {'PASS' if metrics['overall_pass'] else 'FAIL'}")
    
    print(f"  Result: T={metrics['temperature']:.2f}°C CO2={metrics['co2']:.0f}ppm "
          f"V={metrics['velocity']:.3f}m/s [{'PASS' if metrics['overall_pass'] else 'FAIL'}]", flush=True)
    
    results['phases']['validation'] = {
        'temperature': metrics['temperature'], 'co2': metrics['co2'],
        'velocity': metrics['velocity'], 'overall_pass': metrics['overall_pass'],
        'history': history
    }
    
    # =========================================================================
    # PHASE 5: REPORT GENERATION (85-95%)
    # =========================================================================
    emit_progress("report", 85, "Generating deliverables")
    print("\n" + "="*70, flush=True)
    print("  PHASE 4: DELIVERABLES", flush=True)
    print("="*70, flush=True)
    
    from hyperfoam.visuals import (
        render_thermal_heatmap, render_velocity_field,
        render_convergence_plot, render_dashboard_summary
    )
    
    assets = {}
    T_field = solver.thermal_solver.temperature.phi[:, config.ny//2, :].cpu().numpy() - 273.15
    u_field = solver.flow.u[:, config.ny//2, :].cpu().numpy()
    w_field = solver.flow.w[:, config.ny//2, :].cpu().numpy()
    
    if spec.generate_heatmap:
        emit_progress("report", 87, "Thermal heatmap...")
        path = output_dir / "thermal_heatmap.png"
        render_thermal_heatmap(T_field, spec.lx, spec.lz, spec.room_name, str(path),
                              18.0, 26.0, spec.target_temp, True, True)
        plt.close()
        assets['thermal'] = str(path)
        print(f"    -> {path.name}", flush=True)
    
    if spec.generate_velocity:
        emit_progress("report", 89, "Velocity field...")
        path = output_dir / "velocity_field.png"
        render_velocity_field(u_field, w_field, spec.lx, spec.lz, spec.room_name,
                             str(path), 1.0, True, True)
        plt.close()
        assets['velocity'] = str(path)
        print(f"    -> {path.name}", flush=True)
    
    if spec.generate_convergence:
        emit_progress("report", 91, "Convergence plot...")
        path = output_dir / "convergence_plot.png"
        render_convergence_plot(history, spec, str(path))
        plt.close()
        assets['convergence'] = str(path)
        print(f"    -> {path.name}", flush=True)
    
    emit_progress("report", 93, "Dashboard...")
    path = output_dir / "dashboard_summary.png"
    render_dashboard_summary(T_field, u_field, w_field, metrics, spec, str(path))
    plt.close()
    assets['dashboard'] = str(path)
    print(f"    -> {path.name}", flush=True)
    
    if spec.generate_pdf:
        emit_progress("report", 95, "PDF report...")
        pdf = EngineeringReport(spec.client_name, spec.project_id, "HyperFOAM Consulting")
        pdf.add_page()
        pdf.section_title(1, "EXECUTIVE SUMMARY")
        pdf.body_text(f"CFD analysis for {spec.room_name}. "
                     f"{'MEETS' if metrics['overall_pass'] else 'DOES NOT MEET'} ASHRAE 55.")
        pdf.section_title(2, "OPTIMIZED SETTINGS")
        pdf.body_text(f"V={optimal_velocity:.2f}m/s, A={optimal_angle:.1f}°, T={optimal_supply_temp:.1f}°C")
        
        if 'thermal' in assets:
            pdf.add_page()
            pdf.section_title(3, "THERMAL ANALYSIS")
            pdf.image(assets['thermal'], x=10, w=190)
        if 'velocity' in assets:
            pdf.section_title(4, "AIRFLOW ANALYSIS")
            pdf.image(assets['velocity'], x=10, w=190)
        
        pdf_path = output_dir / f"{spec.project_id}_CFD_Report.pdf"
        pdf.output(str(pdf_path))
        assets['pdf'] = str(pdf_path)
        print(f"    -> {pdf_path.name}", flush=True)
    
    results['assets'] = assets
    
    # =========================================================================
    # PHASE 6: COMPLETE (100%)
    # =========================================================================
    total_elapsed = time.time() - pipeline_start
    emit_progress("complete", 100, "Pipeline complete")
    
    print("\n" + "="*70, flush=True)
    print("  JOB COMPLETE", flush=True)
    print("="*70, flush=True)
    print(f"  Project: {spec.project_id}", flush=True)
    print(f"  Time: {total_elapsed:.1f}s", flush=True)
    print(f"  Result: {'ALL PASS' if metrics['overall_pass'] else 'FAIL'}", flush=True)
    
    # Save results
    results_json = {
        'project_id': spec.project_id,
        'client': spec.client_name,
        'timestamp': datetime.now().isoformat(),
        'optimal_settings': {
            'velocity': float(optimal_velocity),
            'angle': float(optimal_angle),
            'supply_temp': float(optimal_supply_temp)
        },
        'validation': {
            'temperature': float(metrics['temperature']),
            'co2': float(metrics['co2']),
            'velocity': float(metrics['velocity']),
            'all_pass': bool(metrics['overall_pass'])
        },
        'assets': assets,
        'elapsed_seconds': float(total_elapsed)
    }
    results_path = output_dir / "job_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="HyperFOAM Production Pipeline")
    parser.add_argument("job_file", help="Path to job_spec.json")
    parser.add_argument("--skip-optimize", action="store_true")
    parser.add_argument("--duration", "-d", type=float, default=60.0)
    args = parser.parse_args()
    run_production_pipeline(args.job_file, args.skip_optimize, args.duration)


if __name__ == "__main__":
    main()
