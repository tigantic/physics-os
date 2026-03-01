"""
HyperFOAM Solver Runner
=======================

Bridges the UI payload to the actual ontic HVAC solver.

Article VII, Section 7.2: "Done" means working end-to-end.
"""

from __future__ import annotations

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

# Add ontic to path (lazy - don't import yet)
REPO_ROOT = Path(__file__).parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Lazy check for solver availability
SOLVER_AVAILABLE = None
SOLVER_IMPORT_ERROR = None


def _check_solver_available():
    """Lazy check for solver availability."""
    global SOLVER_AVAILABLE, SOLVER_IMPORT_ERROR
    if SOLVER_AVAILABLE is not None:
        return SOLVER_AVAILABLE
    
    try:
        import torch
        # Quick check without full ontic import
        SOLVER_AVAILABLE = True
        SOLVER_IMPORT_ERROR = None
    except ImportError as e:
        SOLVER_AVAILABLE = False
        SOLVER_IMPORT_ERROR = f"torch not available: {e}"
    
    return SOLVER_AVAILABLE


def _get_solver_classes():
    """Lazy import of solver classes."""
    from ontic.hvac.solver_3d import Solver3D, Solver3DConfig, Inlet3D, Outlet3D, HeatSource
    return Solver3D, Solver3DConfig, Inlet3D, Outlet3D, HeatSource


@dataclass
class SolverResult:
    """Result from solver execution."""
    success: bool
    converged: bool = False
    iterations: int = 0
    final_residual: float = 0.0
    max_velocity: float = 0.0
    mean_temperature: float = 0.0
    runtime_seconds: float = 0.0
    error: Optional[str] = None
    state: Any = None  # Solver3DState if successful
    config: Any = None  # Solver3DConfig
    residual_history: List[float] = None  # Full convergence history
    frames: List[Any] = None  # Visualization frames captured during solve


def payload_to_config(payload: Dict[str, Any]):
    """
    Convert UI payload (JSON) to Solver3DConfig.
    
    Payload structure (from SimulationSubmitter):
        domain: {width_x_m, height_y_m, length_z_m, grid_resolution}
        boundary_conditions: {inlet: {velocity_vector_ms, temperature_k, ...}}
        heat_sources: [{position_m, power_w}]
    """
    # Lazy import
    Solver3D, Solver3DConfig, Inlet3D, Outlet3D, HeatSource = _get_solver_classes()
    
    domain = payload.get("domain", {})
    bc = payload.get("boundary_conditions", {})
    inlet_bc = bc.get("inlet", {})
    
    # Grid from payload (or defaults)
    grid = domain.get("grid_resolution", [64, 32, 32])
    
    # Build inlet
    inlet_pos = inlet_bc.get("inlet_position_m", [0, 0, 0])
    inlet_dims = inlet_bc.get("inlet_dimensions_m", [0.6, 0.6])
    velocity_vec = inlet_bc.get("velocity_vector_ms", [0, -0.5, 0])
    inlet_temp_k = inlet_bc.get("temperature_k", 286)
    inlet_temp_c = inlet_temp_k - 273.15
    
    # Velocity magnitude (assuming downward flow, Y component)
    velocity_mag = abs(velocity_vec[1]) if len(velocity_vec) > 1 else 0.5
    
    inlets = [Inlet3D(
        x=inlet_pos[0] if len(inlet_pos) > 0 else domain.get("width_x_m", 6) / 2,
        y=inlet_pos[1] if len(inlet_pos) > 1 else domain.get("length_z_m", 4.5) / 2,
        z=inlet_pos[2] if len(inlet_pos) > 2 else domain.get("height_y_m", 2.7) - 0.1,
        width=inlet_dims[0] if len(inlet_dims) > 0 else 0.6,
        height=inlet_dims[1] if len(inlet_dims) > 1 else 0.6,
        velocity=velocity_mag,
        T=inlet_temp_c,
        direction='z-',  # Ceiling diffuser pointing down
    )]
    
    # Build outlet (floor return)
    outlets = [Outlet3D(
        x=0.1,
        y=domain.get("length_z_m", 4.5) / 2,
        z=0.15,
        width=0.4,
        height=0.3,
        bc_type='pressure',
    )]
    
    # Heat sources from payload
    heat_sources = []
    for hs in payload.get("heat_sources", []):
        pos = hs.get("position_m", [0, 0, 0])
        power = hs.get("power_w", 0)
        if power > 0:
            heat_sources.append(HeatSource(
                x=pos[0] if len(pos) > 0 else domain.get("width_x_m", 6) / 2,
                y=pos[1] if len(pos) > 1 else domain.get("length_z_m", 4.5) / 2,
                z=pos[2] if len(pos) > 2 else 1.0,
                power=power,
                type='equipment',
            ))
    
    config = Solver3DConfig(
        length=domain.get("width_x_m", 6.0),
        width=domain.get("length_z_m", 4.5),
        height=domain.get("height_y_m", 2.7),
        nx=grid[0] if len(grid) > 0 else 64,
        ny=grid[1] if len(grid) > 1 else 32,
        nz=grid[2] if len(grid) > 2 else 32,
        inlets=inlets,
        outlets=outlets,
        heat_sources=heat_sources,
        max_iterations=2000,  # Cap for interactive use
        convergence_tol=1e-4,
        verbose=True,
        diag_interval=50,
    )
    
    return config


def run_solver(payload: Dict[str, Any], progress_callback=None, 
               frame_callback=None, capture_interval: int = 50) -> SolverResult:
    """
    Run the HyperFOAM solver with the given payload.
    
    Args:
        payload: Simulation payload from SimulationSubmitter
        progress_callback: Optional callback(iteration, residual, message)
        frame_callback: Optional callback(frame) for visualization updates
        capture_interval: Capture visualization frame every N iterations
    
    Returns:
        SolverResult with success status and results
    """
    global SOLVER_AVAILABLE, SOLVER_IMPORT_ERROR
    
    # Check availability on first call
    if not _check_solver_available():
        return SolverResult(
            success=False,
            error=f"Solver not available: {SOLVER_IMPORT_ERROR}. "
                  f"Ensure ontic is installed and torch is available."
        )
    
    start_time = time.perf_counter()
    captured_frames = []
    
    try:
        # Lazy import solver
        Solver3D, Solver3DConfig, Inlet3D, Outlet3D, HeatSource = _get_solver_classes()
        
        # Import visualizer for frame extraction
        try:
            from staging.visualizer import extract_frame_from_state, VisualizationFrame
            can_capture = True
        except ImportError:
            can_capture = False
        
        # Convert payload to config
        config = payload_to_config(payload)
        
        if progress_callback:
            progress_callback(0, 0, f"Initializing solver ({config.nx}×{config.ny}×{config.nz} grid)...")
        
        # Create solver
        solver = Solver3D(config)
        
        # Define callback for progress updates AND frame capture
        def solver_callback(state, iteration):
            # Progress update
            if progress_callback and iteration % 25 == 0:
                residual = state.residual_history[-1] if state.residual_history else 0
                vel_max = state.velocity_magnitude.max().item()
                temp_mean = state.T.mean().item()
                progress_callback(
                    iteration, 
                    residual,
                    f"Iter {iteration}: res={residual:.2e}, v_max={vel_max:.3f}m/s, T={temp_mean:.1f}°C"
                )
            
            # Frame capture for visualization
            if can_capture and iteration % capture_interval == 0:
                frame = extract_frame_from_state(state, iteration, config)
                captured_frames.append(frame)
                if frame_callback:
                    frame_callback(frame, state.residual_history)
        
        if progress_callback:
            progress_callback(0, 0, "Starting CFD solve...")
        
        # Run solver
        final_state = solver.solve(callback=solver_callback)
        
        # Capture final frame
        if can_capture:
            final_frame = extract_frame_from_state(
                final_state, 
                len(final_state.residual_history), 
                config
            )
            captured_frames.append(final_frame)
        
        runtime = time.perf_counter() - start_time
        
        return SolverResult(
            success=True,
            converged=final_state.converged,
            iterations=len(final_state.residual_history),
            final_residual=final_state.residual_history[-1] if final_state.residual_history else 0,
            max_velocity=final_state.velocity_magnitude.max().item(),
            mean_temperature=final_state.T.mean().item(),
            runtime_seconds=runtime,
            state=final_state,
            config=config,
            residual_history=list(final_state.residual_history),
            frames=captured_frames,
        )
        
    except Exception as e:
        import traceback
        runtime = time.perf_counter() - start_time
        return SolverResult(
            success=False,
            runtime_seconds=runtime,
            error=f"Solver error: {str(e)}\n{traceback.format_exc()}",
            frames=captured_frames,
        )
        )


def run_solver_from_json(json_path: Path, progress_callback=None) -> SolverResult:
    """Load payload from JSON file and run solver."""
    with open(json_path) as f:
        payload = json.load(f)
    return run_solver(payload, progress_callback)


# CLI entry point
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m staging.runner <payload.json>")
        sys.exit(1)
    
    json_path = Path(sys.argv[1])
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)
    
    def cli_progress(iteration, residual, message):
        print(f"  {message}")
    
    print(f"Loading payload: {json_path}")
    result = run_solver_from_json(json_path, cli_progress)
    
    if result.success:
        print(f"\n✅ Solver completed!")
        print(f"   Converged: {result.converged}")
        print(f"   Iterations: {result.iterations}")
        print(f"   Final residual: {result.final_residual:.2e}")
        print(f"   Max velocity: {result.max_velocity:.3f} m/s")
        print(f"   Mean temperature: {result.mean_temperature:.1f}°C")
        print(f"   Runtime: {result.runtime_seconds:.1f}s")
    else:
        print(f"\n❌ Solver failed: {result.error}")
        sys.exit(1)
