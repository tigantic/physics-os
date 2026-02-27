"""
HyperFOAM CLI - The Complete Consultant's Cockpit

Usage:
    python -m hyperfoam dashboard          # Launch interactive demo
    python -m hyperfoam optimize           # AI inverse design
    python -m hyperfoam report             # Generate client PDF
    python -m hyperfoam demo               # Full workflow demo
    python -m hyperfoam benchmark          # Performance test

The "Unfair Advantage": Trade GPU cycles for engineering intuition.
"""

import sys
import os
import argparse
import time
from pathlib import Path


def run_dashboard(args):
    """Launches the Streamlit Dashboard."""
    import subprocess
    
    dashboard_path = Path(__file__).parent / "dashboard.py"
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                  HYPERFOAM DASHBOARD                          ║
╠═══════════════════════════════════════════════════════════════╣
║  Launching interactive HVAC Digital Twin...                   ║
║                                                               ║
║  Open in browser: http://localhost:{:<5}                      ║
║  Press Ctrl+C to stop                                         ║
╚═══════════════════════════════════════════════════════════════╝
    """.format(args.port))
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port", str(args.port),
        "--server.headless", "true"
    ])


def run_optimizer(args):
    """Runs the Inverse Design Optimization."""
    from hyperfoam.optimizer import optimize_hvac, quick_optimize
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                  HYPERFOAM OPTIMIZER                          ║
╠═══════════════════════════════════════════════════════════════╣
║  AI-Driven Inverse Design                                     ║
║                                                               ║
║  "Trading GPU cycles for engineering intuition"               ║
║                                                               ║
║  Scenario: {} occupants ({} W heat load){}║
╚═══════════════════════════════════════════════════════════════╝
    """.format(
        args.occupants, 
        args.occupants * 100,
        " " * (21 - len(str(args.occupants * 100)))
    ))
    
    if args.quick:
        result = quick_optimize(n_occupants=args.occupants)
    else:
        result = optimize_hvac(
            n_occupants=args.occupants,
            target_temp=args.target_temp,
            max_velocity=args.max_velocity,
            method=args.method
        )
    
    # Print copy-paste recommendation
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                 COPY-PASTE RECOMMENDATION                     ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  For a room with {:<3} occupants, configure:                   ║
║                                                               ║
║    config.supply_velocity = {:.2f}   # m/s                    ║
║    config.supply_angle    = {:.1f}   # degrees                ║
║    config.supply_temp     = {:.1f}   # Celsius                ║
║                                                               ║
║  Predicted Performance:                                       ║
║    Temperature: {:.2f} C   {}                            ║
║    CO2:         {:>4.0f} ppm   {}                            ║
║    Draft:       {:.3f} m/s  {}                            ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """.format(
        args.occupants,
        result.optimal_velocity,
        result.optimal_angle,
        result.optimal_supply_temp,
        result.final_temp,
        "[PASS]" if result.temp_pass else "[FAIL]",
        result.final_co2,
        "[PASS]" if result.co2_pass else "[FAIL]",
        result.final_velocity,
        "[PASS]" if result.velocity_pass else "[FAIL]"
    ))
    
    return result


def run_report(args):
    """Generates the Client PDF."""
    from hyperfoam.report import generate_report
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                  HYPERFOAM REPORT GENERATOR                   ║
╠═══════════════════════════════════════════════════════════════╣
║  Creating engineering-grade PDF deliverable                   ║
║                                                               ║
║  Client:  {:<50} ║
║  Project: {:<50} ║
╚═══════════════════════════════════════════════════════════════╝
    """.format(args.client[:50], args.project[:50]))
    
    output_path = generate_report(
        client=args.client,
        project_id=args.project,
        author=args.author
    )
    
    print(f"\n  Report saved: {output_path}")
    print(f"  Ready to attach to ${args.invoice:,} invoice.\n")
    
    return output_path


def run_demo(args):
    """Runs the complete 3-phase demo."""
    from hyperfoam.demo import run_demo as demo_workflow
    
    demo_workflow(
        n_occupants=args.occupants,
        client_name=args.client,
        project_id=args.project,
        skip_optimize=args.skip_optimize,
        skip_report=args.skip_report
    )


def run_benchmark(args):
    """Performance benchmark."""
    from hyperfoam.presets import setup_conference_room
    from hyperfoam import Solver, ConferenceRoom
    import torch
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                  HYPERFOAM BENCHMARK                          ║
╠═══════════════════════════════════════════════════════════════╣
║  Measuring GPU-native CFD performance                         ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    config = ConferenceRoom()
    solver = Solver(config)
    setup_conference_room(solver, n_occupants=12)
    
    print(f"  Grid: {config.nx}x{config.ny}x{config.nz} = {config.nx*config.ny*config.nz:,} cells")
    print(f"  Device: {solver.device}")
    print(f"  Warming up...")
    
    # Warmup
    for _ in range(100):
        solver.step()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"  Running {args.steps} steps...")
    start = time.perf_counter()
    
    for _ in range(args.steps):
        solver.step()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    steps_per_sec = args.steps / elapsed
    realtime_factor = args.steps * config.dt / elapsed
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║                     BENCHMARK RESULTS                         ║
╠═══════════════════════════════════════════════════════════════╣
║  Steps:            {args.steps:>10,}                                   ║
║  Wall time:        {elapsed:>10.3f} s                                  ║
║  Performance:      {steps_per_sec:>10.0f} steps/s                           ║
║  Real-time factor: {realtime_factor:>10.1f}x                                  ║
╠═══════════════════════════════════════════════════════════════╣
║  Comparison:                                                  ║
║    OpenFOAM (CPU):    ~8 steps/s     [{steps_per_sec/8:>5.0f}x faster]             ║
║    ANSYS Fluent:      ~15 steps/s    [{steps_per_sec/15:>5.0f}x faster]             ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    return steps_per_sec


def run_new_job(args):
    """Interactive job spec creator."""
    import json
    from datetime import datetime
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                  CREATE NEW JOB                               ║
╠═══════════════════════════════════════════════════════════════╣
║  Answer a few questions to generate your job_spec.json        ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Client info
    print("─── CLIENT INFO ───")
    client_name = input("  Client name: ").strip() or "Acme Corp"
    project_id = input("  Project ID (e.g., 2026-002): ").strip() or f"{datetime.now().year}-{datetime.now().strftime('%m%d')}"
    contact = input("  Contact person: ").strip() or "N/A"
    
    # Room info
    print("\n─── ROOM INFO ───")
    room_name = input("  Room name: ").strip() or "Main Conference Room"
    room_type = input("  Room type [conference/office/lobby]: ").strip() or "conference"
    
    print("  Room dimensions (meters):")
    try:
        length = float(input("    Length: ").strip() or "9")
        width = float(input("    Width: ").strip() or "6")
        height = float(input("    Height: ").strip() or "3")
    except ValueError:
        length, width, height = 9.0, 6.0, 3.0
    
    # Load info
    print("\n─── HEAT LOADS ───")
    try:
        occupants = int(input("  Number of occupants: ").strip() or "12")
        equipment = int(input("  Equipment load (watts): ").strip() or "500")
        lighting = int(input("  Lighting load (watts): ").strip() or "200")
    except ValueError:
        occupants, equipment, lighting = 12, 500, 200
    
    # Constraints (use defaults)
    print("\n─── CONSTRAINTS (press Enter for ASHRAE defaults) ───")
    try:
        max_vel = float(input("  Max draft velocity [0.25 m/s]: ").strip() or "0.25")
        target_temp = float(input("  Target temperature [22°C]: ").strip() or "22")
        max_co2 = int(input("  Max CO2 [1000 ppm]: ").strip() or "1000")
    except ValueError:
        max_vel, target_temp, max_co2 = 0.25, 22.0, 1000
    
    # Notes
    notes = input("\n  Notes (optional): ").strip() or ""
    
    # Build spec
    spec = {
        "client": {
            "name": client_name,
            "project_id": project_id,
            "contact": contact
        },
        "room": {
            "name": room_name,
            "type": room_type,
            "dimensions_m": [length, width, height]
        },
        "load": {
            "occupants": occupants,
            "heat_load_per_person_watts": 100,
            "equipment_load_watts": equipment,
            "lighting_load_watts": lighting
        },
        "constraints": {
            "max_velocity_ms": max_vel,
            "target_temp_c": target_temp,
            "temp_tolerance_c": 2.0,
            "max_co2_ppm": max_co2
        },
        "deliverables": {
            "thermal_heatmap": True,
            "velocity_field": True,
            "convergence_plot": True,
            "pdf_report": True
        },
        "notes": notes
    }
    
    # Create output directory
    safe_id = project_id.replace("/", "-").replace(" ", "_")
    output_dir = Path(f"projects/{safe_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    spec_path = output_dir / "job_spec.json"
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║                  JOB SPEC CREATED                             ║
╠═══════════════════════════════════════════════════════════════╣
║  File: {str(spec_path):<53} ║
╠═══════════════════════════════════════════════════════════════╣
║  Client:     {client_name:<46} ║
║  Room:       {length}m × {width}m × {height}m ({length*width*height:.0f} m³){' '*(29-len(f'{length}m × {width}m × {height}m ({length*width*height:.0f} m³)'))}║
║  Occupants:  {occupants:<46} ║
║  Heat Load:  {occupants*100 + equipment + lighting} W total{' '*36}║
╠═══════════════════════════════════════════════════════════════╣
║  To run:  hyperfoam run {str(spec_path):<35} ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Ask if they want to run it now
    run_now = input("  Run this job now? [Y/n]: ").strip().lower()
    if run_now != 'n':
        from hyperfoam.pipeline import run_production_pipeline
        run_production_pipeline(str(spec_path), skip_optimize=False, sim_duration=300.0)


def main():
    parser = argparse.ArgumentParser(
        prog="hyperfoam",
        description="HyperFOAM: GPU-Native CFD for HVAC Digital Twins",
        epilog="The 'Unfair Advantage': Trade GPU cycles for engineering intuition."
    )
    
    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available commands:"
    )
    
    # =========================================================================
    # 1. DASHBOARD Command
    # =========================================================================
    parser_dash = subparsers.add_parser(
        "dashboard",
        help="Launch interactive Streamlit demo",
        description="Opens the live HVAC Digital Twin dashboard with sliders and visualizations."
    )
    parser_dash.add_argument(
        "--port", "-p",
        type=int,
        default=8501,
        help="Port for dashboard (default: 8501)"
    )
    
    # =========================================================================
    # 2. OPTIMIZE Command
    # =========================================================================
    parser_opt = subparsers.add_parser(
        "optimize",
        help="Run AI inverse design optimizer",
        description="Automatically finds optimal HVAC settings using gradient-free optimization."
    )
    parser_opt.add_argument(
        "--occupants", "-n",
        type=int,
        default=12,
        help="Number of room occupants (default: 12)"
    )
    parser_opt.add_argument(
        "--target-temp", "-t",
        type=float,
        default=22.0,
        help="Target temperature in Celsius (default: 22.0)"
    )
    parser_opt.add_argument(
        "--max-velocity", "-v",
        type=float,
        default=0.25,
        help="Maximum draft velocity in m/s (default: 0.25)"
    )
    parser_opt.add_argument(
        "--method", "-m",
        choices=['differential_evolution', 'nelder-mead', 'grid'],
        default='differential_evolution',
        help="Optimization method (default: differential_evolution)"
    )
    parser_opt.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick mode: faster but less precise (uses grid search)"
    )
    
    # =========================================================================
    # 3. REPORT Command
    # =========================================================================
    parser_rep = subparsers.add_parser(
        "report",
        help="Generate professional PDF deliverable",
        description="Creates an engineering-grade CFD report with thermal heatmaps and compliance data."
    )
    parser_rep.add_argument(
        "--client", "-c",
        type=str,
        default="Apex Architecture Group",
        help="Client company name"
    )
    parser_rep.add_argument(
        "--project", "-p",
        type=str,
        default="CR-2026-B",
        help="Project identifier"
    )
    parser_rep.add_argument(
        "--author", "-a",
        type=str,
        default="HyperFOAM Consulting",
        help="Author/firm name for report header"
    )
    parser_rep.add_argument(
        "--invoice",
        type=int,
        default=5000,
        help="Invoice amount for display (default: 5000)"
    )
    
    # =========================================================================
    # 4. DEMO Command
    # =========================================================================
    parser_demo = subparsers.add_parser(
        "demo",
        help="Run complete 3-phase workflow",
        description="Executes full pipeline: Optimize -> Validate -> Report"
    )
    parser_demo.add_argument(
        "--occupants", "-n",
        type=int,
        default=12,
        help="Number of room occupants (default: 12)"
    )
    parser_demo.add_argument(
        "--client", "-c",
        type=str,
        default="Apex Architecture Group",
        help="Client company name"
    )
    parser_demo.add_argument(
        "--project", "-p",
        type=str,
        default="CR-2026-B",
        help="Project identifier"
    )
    parser_demo.add_argument(
        "--skip-optimize",
        action="store_true",
        help="Skip optimization phase"
    )
    parser_demo.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip report generation"
    )
    
    # =========================================================================
    # 5. BENCHMARK Command
    # =========================================================================
    parser_bench = subparsers.add_parser(
        "benchmark",
        help="Run performance benchmark",
        description="Measures GPU CFD performance and compares to legacy solvers."
    )
    parser_bench.add_argument(
        "--steps", "-s",
        type=int,
        default=1000,
        help="Number of timesteps to benchmark (default: 1000)"
    )
    
    # =========================================================================
    # 6. RUN Command (Production Pipeline)
    # =========================================================================
    parser_run = subparsers.add_parser(
        "run",
        help="Execute production job from JSON spec",
        description="Runs the full production pipeline: Load spec -> Optimize -> Validate -> Report"
    )
    parser_run.add_argument(
        "job_file",
        help="Path to job_spec.json file"
    )
    parser_run.add_argument(
        "--skip-optimize",
        action="store_true",
        help="Use physics heuristic instead of AI optimization"
    )
    parser_run.add_argument(
        "--duration", "-d",
        type=float,
        default=300.0,
        help="Simulation duration in seconds (default: 300)"
    )
    
    # =========================================================================
    # 7. NEW Command (Interactive Job Creator)
    # =========================================================================
    parser_new = subparsers.add_parser(
        "new",
        help="Create a new job interactively",
        description="Walks you through creating a job_spec.json file step by step"
    )
    
    # =========================================================================
    # Parse and dispatch
    # =========================================================================
    args = parser.parse_args()
    
    if args.command is None:
        # Default: show help with quick start
        parser.print_help()
        print("""
  Quick start examples:
    python -m hyperfoam run projects/2026-001/job_spec.json   # Production job
    python -m hyperfoam optimize -n 20          # Find settings for 20 occupants
    python -m hyperfoam dashboard               # Launch interactive UI
    python -m hyperfoam report -c "Acme Corp"   # Generate client PDF
        """)
        return
    
    # Dispatch to appropriate handler
    handlers = {
        "dashboard": run_dashboard,
        "optimize": run_optimizer,
        "report": run_report,
        "demo": run_demo,
        "benchmark": run_benchmark,
        "run": lambda args: __import__('hyperfoam.pipeline', fromlist=['run_production_pipeline']).run_production_pipeline(args.job_file, args.skip_optimize, args.duration),
        "new": run_new_job
    }
    
    handler = handlers.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
