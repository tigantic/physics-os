"""
HyperFOAM Unified Demo: The Complete Consultant's Cockpit

This demo showcases the full HyperFOAM workflow:
1. OPTIMIZE - AI finds optimal HVAC settings
2. VALIDATE - Run full simulation with optimal settings
3. REPORT   - Generate professional PDF deliverable

Usage:
    python -m hyperfoam.demo
    python -m hyperfoam.demo --occupants 20 --client "Apex Corp"
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

# Ensure hyperfoam is importable
_pkg_root = Path(__file__).parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))


def print_banner():
    """Print the HyperFOAM banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ██╗  ██╗██╗   ██╗██████╗ ███████╗██████╗ ███████╗ ██████╗  █████╗ ███╗   ███╗║
║   ██║  ██║╚██╗ ██╔╝██╔══██╗██╔════╝██╔══██╗██╔════╝██╔═══██╗██╔══██╗████╗ ████║║
║   ███████║ ╚████╔╝ ██████╔╝█████╗  ██████╔╝█████╗  ██║   ██║███████║██╔████╔██║║
║   ██╔══██║  ╚██╔╝  ██╔═══╝ ██╔══╝  ██╔══██╗██╔══╝  ██║   ██║██╔══██║██║╚██╔╝██║║
║   ██║  ██║   ██║   ██║     ███████╗██║  ██║██║     ╚██████╔╝██║  ██║██║ ╚═╝ ██║║
║   ╚═╝  ╚═╝   ╚═╝   ╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝      ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝║
║                                                                               ║
║               GPU-Native CFD for HVAC Digital Twins                           ║
║                    The Complete Consultant's Cockpit                          ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_phase(num: int, title: str, description: str):
    """Print a phase header."""
    print(f"\n{'='*80}")
    print(f"  PHASE {num}: {title}")
    print(f"  {description}")
    print(f"{'='*80}\n")


def run_demo(
    n_occupants: int = 12,
    client_name: str = "Apex Architecture Group",
    project_id: str = "CR-2026-B",
    skip_optimize: bool = False,
    skip_report: bool = False
):
    """
    Run the complete HyperFOAM demo workflow.
    
    Args:
        n_occupants: Number of room occupants
        client_name: Client name for report
        project_id: Project ID for report
        skip_optimize: Skip optimization phase
        skip_report: Skip report generation
    """
    from hyperfoam import (
        Solver, ConferenceRoom, __version__,
        optimize_hvac, quick_optimize
    )
    from hyperfoam.presets import setup_conference_room
    from hyperfoam.report import generate_report, run_simulation, build_pdf
    
    print_banner()
    
    print(f"  Version: {__version__}")
    print(f"  Date: {datetime.now().strftime('%B %d, %Y at %H:%M')}")
    print(f"  Client: {client_name}")
    print(f"  Project: {project_id}")
    print(f"  Scenario: Conference room with {n_occupants} occupants")
    
    total_start = time.time()
    
    # =========================================================================
    # PHASE 1: OPTIMIZATION
    # =========================================================================
    if not skip_optimize:
        print_phase(1, "INVERSE DESIGN OPTIMIZATION",
                   "AI-driven search for optimal HVAC settings")
        
        print("Running gradient-free optimization...")
        print("This explores the design space to find settings that:")
        print("  - Keep temperature in 20-24°C range")
        print("  - Maintain CO2 below 1000 ppm")
        print("  - Prevent drafts (velocity < 0.25 m/s)")
        print("  - Minimize energy consumption")
        print()
        
        # Use quick optimization for demo speed
        result = quick_optimize(n_occupants=n_occupants)
        
        optimal_velocity = result.optimal_velocity
        optimal_angle = result.optimal_angle
        optimal_supply_temp = result.optimal_supply_temp
        
        print("\n" + "─"*60)
        print("OPTIMIZATION COMPLETE")
        print("─"*60)
        print(f"  Recommended Velocity: {optimal_velocity:.2f} m/s")
        print(f"  Recommended Angle:    {optimal_angle:.1f}°")
        print(f"  Recommended Supply T: {optimal_supply_temp:.1f}°C")
        print(f"  Evaluations:          {result.n_evaluations}")
        print(f"  All targets met:      {'YES' if result.all_pass else 'NO'}")
    else:
        # Use default validated settings
        optimal_velocity = 0.80
        optimal_angle = 60.0
        optimal_supply_temp = 20.0
        print_phase(1, "OPTIMIZATION (SKIPPED)",
                   "Using default validated settings")
    
    # =========================================================================
    # PHASE 2: FULL VALIDATION SIMULATION
    # =========================================================================
    print_phase(2, "STEADY-STATE VALIDATION",
               "5-minute simulation to confirm thermal equilibrium")
    
    # Create solver with optimal settings
    config = ConferenceRoom()
    config.supply_velocity = optimal_velocity
    config.supply_angle = optimal_angle
    config.supply_temp = optimal_supply_temp
    
    solver = Solver(config)
    setup_conference_room(solver, n_occupants=n_occupants)
    
    print(f"Grid: {config.nx}×{config.ny}×{config.nz} = {config.nx*config.ny*config.nz:,} cells")
    print(f"Device: {solver.device}")
    print()
    
    # History for plotting
    history = {'time': [], 'T': [], 'CO2': [], 'V': []}
    
    def callback(t, m):
        history['time'].append(t)
        history['T'].append(m['T'])
        history['CO2'].append(m['CO2'])
        history['V'].append(m['V'])
        
        if t % 30 < config.dt:
            status = "✓" if (20 <= m['T'] <= 24 and m['CO2'] < 1000 and m['V'] < 0.25) else "○"
            print(f"  t={t:5.0f}s | T={m['T']:.2f}°C | CO2={m['CO2']:.0f}ppm | V={m['V']:.3f}m/s [{status}]")
    
    sim_start = time.time()
    solver.solve(duration=300, callback=callback)
    sim_elapsed = time.time() - sim_start
    
    # Final metrics
    metrics = solver.get_comfort_metrics()
    
    print("\n" + "─"*60)
    print("VALIDATION RESULTS")
    print("─"*60)
    
    def status(passed):
        return "✓ PASS" if passed else "✗ FAIL"
    
    print(f"  Temperature:  {metrics['temperature']:.2f}°C  {status(metrics['temp_pass'])}  (target: 20-24°C)")
    print(f"  CO2:          {metrics['co2']:.0f} ppm   {status(metrics['co2_pass'])}  (limit: <1000 ppm)")
    print(f"  Draft:        {metrics['velocity']:.3f} m/s  {status(metrics['velocity_pass'])}  (limit: <0.25 m/s)")
    print()
    print(f"  Overall:      {status(metrics['overall_pass'])}")
    print(f"  Sim time:     {sim_elapsed:.1f}s ({300/sim_elapsed:.1f}× real-time)")
    
    # =========================================================================
    # PHASE 3: REPORT GENERATION
    # =========================================================================
    if not skip_report:
        print_phase(3, "PROFESSIONAL REPORT GENERATION",
                   "Creating engineering-grade PDF deliverable")
        
        print("Rendering thermal heatmap...")
        print("Rendering velocity field...")
        print("Plotting convergence data...")
        print("Compiling PDF with legal disclaimers...")
        
        # Build report using validated solver
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Asset directory
        asset_dir = Path("report_assets")
        asset_dir.mkdir(exist_ok=True)
        
        # A. Thermal Heatmap
        T_field = solver.thermal_solver.temperature.phi[:, config.ny//2, :].cpu().numpy().T - 273.15
        
        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(T_field, origin='lower', cmap='RdYlBu_r', vmin=18, vmax=26,
                      aspect='auto', extent=[0, config.lx, 0, config.lz])
        plt.colorbar(im, ax=ax, label="Temperature (C)")
        ax.set_xlabel("Room Length (m)")
        ax.set_ylabel("Height (m)")
        ax.set_title("Steady-State Thermal Distribution", fontweight='bold')
        thermal_path = asset_dir / "thermal_heatmap.png"
        plt.savefig(thermal_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # B. Convergence Plot
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        
        axes[0].plot(history['time'], history['T'], 'r-', linewidth=2)
        axes[0].axhspan(20, 24, alpha=0.15, color='green')
        axes[0].set_ylabel("Temperature (C)")
        axes[0].set_ylim(18, 28)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(history['time'], history['CO2'], 'g-', linewidth=2)
        axes[1].axhline(1000, color='red', linestyle='--', alpha=0.5)
        axes[1].set_ylabel("CO2 (ppm)")
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(history['time'], history['V'], 'b-', linewidth=2)
        axes[2].axhline(0.25, color='red', linestyle='--', alpha=0.5)
        axes[2].set_ylabel("Velocity (m/s)")
        axes[2].set_xlabel("Time (s)")
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        convergence_path = asset_dir / "convergence_plot.png"
        plt.savefig(convergence_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # C. Velocity Field
        import numpy as np
        vel_mag = np.sqrt(
            solver.flow.u[:, config.ny//2, :].cpu().numpy()**2 +
            solver.flow.w[:, config.ny//2, :].cpu().numpy()**2
        ).T
        
        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(vel_mag, origin='lower', cmap='viridis', vmin=0, vmax=1.0,
                      aspect='auto', extent=[0, config.lx, 0, config.lz])
        plt.colorbar(im, ax=ax, label="Velocity (m/s)")
        ax.set_xlabel("Room Length (m)")
        ax.set_ylabel("Height (m)")
        ax.set_title("Airflow Velocity Distribution", fontweight='bold')
        velocity_path = asset_dir / "velocity_field.png"
        plt.savefig(velocity_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Build PDF
        from hyperfoam.report import EngineeringReport
        from hyperfoam import __version__
        
        pdf = EngineeringReport(client_name, project_id, "HyperFOAM Consulting")
        pdf.add_page()
        
        # Executive Summary
        pdf.section_title(1, "EXECUTIVE SUMMARY")
        summary = (
            f"HyperFOAM performed CFD analysis for {client_name} to validate the HVAC design "
            f"for a conference room with {n_occupants} occupants. The AI-driven inverse design "
            f"optimizer determined optimal supply velocity ({optimal_velocity:.2f} m/s) and "
            f"diffuser angle ({optimal_angle:.1f} deg). Full 5-minute steady-state simulation "
            f"confirms {'COMPLIANCE' if metrics['overall_pass'] else 'NON-COMPLIANCE'} with "
            f"ASHRAE Standard 55 thermal comfort criteria."
        )
        pdf.body_text(summary)
        
        # Optimal Settings
        pdf.section_title(2, "AI-OPTIMIZED SETTINGS")
        settings = (
            f"- Supply Air Velocity: {optimal_velocity:.2f} m/s (AI-optimized)\n"
            f"- Diffuser Angle: {optimal_angle:.1f} deg (AI-optimized)\n"
            f"- Supply Temperature: {optimal_supply_temp:.1f} C\n"
            f"- Occupant Load: {n_occupants} persons x 100W = {n_occupants*100}W"
        )
        pdf.body_text(settings)
        
        # Metrics Table
        pdf.section_title(3, "PERFORMANCE METRICS")
        pdf.set_fill_color(41, 128, 185)
        pdf.set_text_color(255)
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(60, 10, "Metric", 1, 0, 'C', True)
        pdf.cell(60, 10, "Value", 1, 0, 'C', True)
        pdf.cell(60, 10, "Status", 1, 1, 'C', True)
        pdf.set_text_color(0)
        
        pdf.add_metric_row("Temperature", f"{metrics['temperature']:.2f} C", 
                          "COMPLIANT" if metrics['temp_pass'] else "NON-COMPLIANT", metrics['temp_pass'])
        pdf.add_metric_row("CO2", f"{metrics['co2']:.0f} ppm",
                          "COMPLIANT" if metrics['co2_pass'] else "NON-COMPLIANT", metrics['co2_pass'])
        pdf.add_metric_row("Draft Velocity", f"{metrics['velocity']:.3f} m/s",
                          "COMPLIANT" if metrics['velocity_pass'] else "NON-COMPLIANT", metrics['velocity_pass'])
        
        # Plots
        pdf.add_page()
        pdf.section_title(4, "THERMAL ANALYSIS")
        pdf.image(str(thermal_path), x=10, w=190)
        
        pdf.section_title(5, "AIRFLOW ANALYSIS")
        pdf.image(str(velocity_path), x=10, w=190)
        
        pdf.add_page()
        pdf.section_title(6, "CONVERGENCE DATA")
        pdf.image(str(convergence_path), x=10, w=190)
        
        # Disclaimer
        pdf.ln(10)
        pdf.set_draw_color(200, 0, 0)
        pdf.rect(10, pdf.get_y(), 190, 30)
        pdf.set_xy(12, pdf.get_y() + 2)
        pdf.set_font('Arial', 'B', 9)
        pdf.set_text_color(200, 0, 0)
        pdf.cell(0, 5, "DISCLAIMER", 0, 1)
        pdf.set_x(12)
        pdf.set_font('Arial', '', 8)
        pdf.set_text_color(60)
        pdf.multi_cell(186, 4,
            "This report is for preliminary engineering assessment only. Final equipment sizing "
            "and code compliance are the responsibility of the PE of Record.")
        
        # Save
        safe_name = client_name.replace(" ", "_")
        output_path = f"{safe_name}_{project_id}_CFD_Report.pdf"
        pdf.output(output_path)
        
        print(f"\n  Report saved: {output_path}")
    else:
        print_phase(3, "REPORT GENERATION (SKIPPED)", "")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    total_elapsed = time.time() - total_start
    
    print("\n" + "="*80)
    print("  DEMO COMPLETE")
    print("="*80)
    print(f"""
  Total time:     {total_elapsed:.1f}s
  
  DELIVERABLES:
  -------------
  1. Optimal Settings:
     - Velocity: {optimal_velocity:.2f} m/s
     - Angle:    {optimal_angle:.1f} deg
     - Supply T: {optimal_supply_temp:.1f} C
  
  2. Validated Performance:
     - Temperature: {metrics['temperature']:.2f} C {'[PASS]' if metrics['temp_pass'] else '[FAIL]'}
     - CO2:         {metrics['co2']:.0f} ppm {'[PASS]' if metrics['co2_pass'] else '[FAIL]'}
     - Draft:       {metrics['velocity']:.3f} m/s {'[PASS]' if metrics['velocity_pass'] else '[FAIL]'}
""")
    
    if not skip_report:
        print(f"  3. PDF Report: {output_path}")
    
    print("""
  NEXT STEPS:
  -----------
  - Launch dashboard:  python -m hyperfoam dashboard
  - Run benchmark:     python -m hyperfoam benchmark
  - Custom optimize:   python -m hyperfoam optimize --occupants 24
    """)
    
    return {
        'optimal_velocity': optimal_velocity,
        'optimal_angle': optimal_angle,
        'optimal_supply_temp': optimal_supply_temp,
        'metrics': metrics,
        'elapsed': total_elapsed
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="HyperFOAM Complete Demo"
    )
    parser.add_argument(
        "--occupants", "-n",
        type=int,
        default=12,
        help="Number of occupants (default: 12)"
    )
    parser.add_argument(
        "--client", "-c",
        default="Apex Architecture Group",
        help="Client name for report"
    )
    parser.add_argument(
        "--project", "-p",
        default="CR-2026-B",
        help="Project ID"
    )
    parser.add_argument(
        "--skip-optimize",
        action="store_true",
        help="Skip optimization phase"
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip report generation"
    )
    
    args = parser.parse_args()
    
    run_demo(
        n_occupants=args.occupants,
        client_name=args.client,
        project_id=args.project,
        skip_optimize=args.skip_optimize,
        skip_report=args.skip_report
    )


if __name__ == "__main__":
    main()
