"""
HyperFOAM Inverse Design Optimizer

Automatically finds optimal HVAC settings to meet comfort targets.
Instead of manually adjusting sliders, click "Optimize" and get the answer.

Usage:
    python -m hyperfoam.optimizer
    python -m hyperfoam.optimizer --occupants 20 --target-temp 22.0
    
Or programmatically:
    from hyperfoam.optimizer import optimize_hvac
    result = optimize_hvac(n_occupants=12, target_temp=22.0)
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Optional
import time
import argparse

# Ensure hyperfoam is importable
_pkg_root = Path(__file__).parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

import numpy as np
from scipy.optimize import minimize, differential_evolution

from hyperfoam import Solver, ConferenceRoom
from hyperfoam.presets import setup_conference_room


@dataclass
class OptimizationTarget:
    """ASHRAE comfort targets for optimization."""
    
    # Temperature bounds (ASHRAE 55)
    temp_min: float = 20.0   # °C
    temp_max: float = 24.0   # °C
    temp_target: float = 22.0  # Ideal center
    
    # CO2 limit (ASHRAE 62.1)
    co2_max: float = 1000.0  # ppm
    
    # Draft velocity limit (ASHRAE 55)
    velocity_max: float = 0.25  # m/s
    
    # Weights for multi-objective optimization
    weight_temp: float = 1.0
    weight_co2: float = 0.5
    weight_velocity: float = 1.5  # Penalize draft heavily
    weight_energy: float = 0.3    # Prefer lower velocities for energy


@dataclass
class OptimizationBounds:
    """Search bounds for HVAC parameters."""
    
    # Supply velocity (m/s)
    velocity_min: float = 0.3
    velocity_max: float = 2.0
    
    # Diffuser angle (degrees from vertical)
    angle_min: float = 15.0
    angle_max: float = 75.0
    
    # Supply temperature (°C) - optional optimization
    temp_min: float = 16.0
    temp_max: float = 22.0


@dataclass
class OptimizationResult:
    """Result of HVAC optimization."""
    
    # Optimal parameters
    optimal_velocity: float
    optimal_angle: float
    optimal_supply_temp: float
    
    # Achieved metrics
    final_temp: float
    final_co2: float
    final_velocity: float
    
    # Compliance
    all_pass: bool
    temp_pass: bool
    co2_pass: bool
    velocity_pass: bool
    
    # Optimization metadata
    n_evaluations: int
    total_time: float
    convergence_message: str
    
    # History for visualization
    history: list = field(default_factory=list)
    
    def __str__(self):
        status = "✓ ALL TARGETS MET" if self.all_pass else "✗ TARGETS NOT MET"
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                 HYPERFOAM OPTIMIZATION RESULT                ║
╠══════════════════════════════════════════════════════════════╣
║  {status:^58}  ║
╠══════════════════════════════════════════════════════════════╣
║  RECOMMENDED SETTINGS:                                       ║
║    Supply Velocity:  {self.optimal_velocity:6.2f} m/s                            ║
║    Diffuser Angle:   {self.optimal_angle:6.1f}°                                 ║
║    Supply Temp:      {self.optimal_supply_temp:6.1f}°C                               ║
╠══════════════════════════════════════════════════════════════╣
║  PREDICTED PERFORMANCE:                                      ║
║    Temperature:  {self.final_temp:5.2f}°C   {'[PASS]' if self.temp_pass else '[FAIL]':>8}  (target: 20-24°C)    ║
║    CO2:          {self.final_co2:5.0f} ppm  {'[PASS]' if self.co2_pass else '[FAIL]':>8}  (limit: <1000 ppm)   ║
║    Draft:        {self.final_velocity:5.3f} m/s  {'[PASS]' if self.velocity_pass else '[FAIL]':>8}  (limit: <0.25 m/s)  ║
╠══════════════════════════════════════════════════════════════╣
║  Evaluations: {self.n_evaluations:4d}  |  Time: {self.total_time:6.1f}s  |  {self.convergence_message[:20]:20}  ║
╚══════════════════════════════════════════════════════════════╝
"""


class HVACOptimizer:
    """
    Gradient-free optimizer for HVAC system design.
    
    Uses differential evolution to find optimal supply velocity and
    diffuser angle that meet ASHRAE comfort targets while minimizing
    energy consumption.
    """
    
    def __init__(
        self,
        n_occupants: int = 12,
        targets: OptimizationTarget = None,
        bounds: OptimizationBounds = None,
        sim_duration: float = 60.0,  # Fast sims during optimization
        validation_duration: float = 300.0,  # Full validation
        optimize_supply_temp: bool = False,
        supply_temp: float = 13.0,  # Client's supply air temperature (°C)
        verbose: bool = True
    ):
        self.n_occupants = n_occupants
        self.targets = targets or OptimizationTarget()
        self.bounds = bounds or OptimizationBounds()
        self.sim_duration = sim_duration
        self.validation_duration = validation_duration
        self.optimize_supply_temp = optimize_supply_temp
        self.supply_temp = supply_temp  # Store for heat balance calculation
        self.verbose = verbose
        
        # Tracking
        self.n_evaluations = 0
        self.history = []
        self.best_cost = float('inf')
        self.best_params = None
    
    def _run_simulation(self, velocity: float, angle: float, 
                        supply_temp: float, duration: float,
                        verbose_progress: bool = False,
                        progress_emitter: callable = None) -> dict:
        """Run a single CFD simulation and return metrics.
        
        Args:
            progress_emitter: Optional callback(pct, msg) for UI progress updates
        """
        
        config = ConferenceRoom()
        config.supply_velocity = velocity
        config.supply_angle = angle
        config.supply_temp = supply_temp
        
        solver = Solver(config)
        setup_conference_room(solver, n_occupants=self.n_occupants)
        
        # Progress callback for verbose mode
        callback = None
        if verbose_progress or progress_emitter:
            def progress_cb(t, metrics):
                pct = int((t / duration) * 100)
                print(f"    t={t:.0f}s ({pct}%) | T={metrics['T']:.1f}°C CO2={metrics['CO2']:.0f}ppm", flush=True)
                # Emit structured progress if emitter provided
                if progress_emitter:
                    # Map internal 0-100% to optimizer's 45-55% range
                    ui_pct = 45 + int(pct * 0.1)  # 45% -> 55%
                    progress_emitter(ui_pct, f"Optimizer validation t={t:.0f}s")
            callback = progress_cb
        
        solver.solve(duration=duration, callback=callback, log_interval=30.0, verbose=False)
        
        return solver.get_comfort_metrics()
    
    def _objective(self, params: np.ndarray) -> float:
        """
        Objective function for optimization.
        
        Lower is better. Returns weighted sum of constraint violations
        plus energy penalty.
        """
        if self.optimize_supply_temp:
            velocity, angle, supply_temp = params
        else:
            velocity, angle = params
            supply_temp = self.targets.temp_target - 2.0  # Default: 2°C below target
        
        self.n_evaluations += 1
        
        # Run fast simulation
        try:
            metrics = self._run_simulation(velocity, angle, supply_temp, self.sim_duration)
        except Exception as e:
            if self.verbose:
                print(f"    [!] Simulation failed: {e}")
            return 1000.0  # Penalty for failed sim
        
        # Extract values
        T = metrics['temperature']
        CO2 = metrics['co2']
        V = metrics['velocity']
        
        # Compute constraint violations (positive = violation)
        temp_violation_low = max(0, self.targets.temp_min - T)
        temp_violation_high = max(0, T - self.targets.temp_max)
        temp_violation = temp_violation_low + temp_violation_high
        
        co2_violation = max(0, CO2 - self.targets.co2_max) / 100.0  # Normalize
        
        velocity_violation = max(0, V - self.targets.velocity_max) * 10.0  # Scale up
        
        # Energy penalty (prefer lower supply velocity)
        energy_penalty = (velocity - self.bounds.velocity_min) / (self.bounds.velocity_max - self.bounds.velocity_min)
        
        # Temperature distance from target (prefer center of range)
        temp_distance = abs(T - self.targets.temp_target) / 2.0
        
        # Weighted sum
        cost = (
            self.targets.weight_temp * (temp_violation + temp_distance) +
            self.targets.weight_co2 * co2_violation +
            self.targets.weight_velocity * velocity_violation +
            self.targets.weight_energy * energy_penalty
        )
        
        # Track history
        record = {
            'eval': self.n_evaluations,
            'velocity': velocity,
            'angle': angle,
            'supply_temp': supply_temp,
            'T': T,
            'CO2': CO2,
            'V': V,
            'cost': cost,
            'feasible': temp_violation == 0 and co2_violation == 0 and velocity_violation == 0
        }
        self.history.append(record)
        
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_params = (velocity, angle, supply_temp)
        
        if self.verbose:
            status = "OK" if record['feasible'] else "X"
            print(f"    [{self.n_evaluations:3d}] V={velocity:.2f} A={angle:.1f}° "
                  f"| T={T:.1f}°C CO2={CO2:.0f} v={V:.3f} | cost={cost:.3f} [{status}]")
        
        return cost
    
    def optimize(self, method: str = 'hybrid', initial_guess: tuple = None,
                 progress_callback: callable = None) -> OptimizationResult:
        """
        Run optimization to find optimal HVAC settings.
        
        Args:
            method: Optimization approach
                - 'hybrid' (default): Physics-direct calculation, 0 search evals, ~3s
                    Computes velocity from heat balance (Q = ṁ·cp·ΔT)
                    Computes angle from throw physics (ceiling height, terminal velocity)
                    Runs ONE validation simulation
            progress_callback: Optional callback(pct, msg) for UI progress updates
                - 'differential_evolution': Global search (~75 evals, ~4min)
                - 'nelder-mead': Local search (~50 evals, ~2.5min)  
                - 'grid': Brute force grid search (~50 evals, ~2.5min)
            initial_guess: (velocity, angle) tuple - ignored for 'hybrid' method
                For other methods: starting point for search
            
        Returns:
            OptimizationResult with optimal settings and validation metrics
        
        Note:
            The 'hybrid' method uses physics equations directly instead of
            guess-and-check optimization. The Navier-Stokes solver is deterministic;
            we compute the answer from first principles rather than searching for it.
        """
        print("\n" + "=" * 64)
        print("       HYPERFOAM INVERSE DESIGN OPTIMIZER")
        print("=" * 64)
        print(f"\nScenario: {self.n_occupants} occupants, {self.n_occupants * 100}W heat load")
        print(f"Targets:  T=[{self.targets.temp_min}-{self.targets.temp_max}°C], "
              f"CO2<{self.targets.co2_max:.0f}ppm, V<{self.targets.velocity_max}m/s")
        print(f"Method:   {method}")
        if initial_guess:
            print(f"Start:    V={initial_guess[0]:.2f} m/s, A={initial_guess[1]:.1f}° (from equipment config)")
        print("\n" + "-" * 64)
        
        start_time = time.time()
        
        # Define bounds
        if self.optimize_supply_temp:
            bounds_list = [
                (self.bounds.velocity_min, self.bounds.velocity_max),
                (self.bounds.angle_min, self.bounds.angle_max),
                (self.bounds.temp_min, self.bounds.temp_max)
            ]
        else:
            bounds_list = [
                (self.bounds.velocity_min, self.bounds.velocity_max),
                (self.bounds.angle_min, self.bounds.angle_max)
            ]
        
        if method == 'differential_evolution':
            # Global optimizer - more robust
            result = differential_evolution(
                self._objective,
                bounds=bounds_list,
                strategy='best1bin',
                maxiter=15,
                popsize=5,
                tol=0.01,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=42,
                polish=True,
                disp=False,
                workers=1  # Can't parallelize due to GPU
            )
            convergence_msg = "Global search complete"
            
        elif method == 'nelder-mead':
            # Local optimizer - faster but may miss global optimum
            x0 = np.array([1.0, 45.0] if not self.optimize_supply_temp else [1.0, 45.0, 20.0])
            result = minimize(
                self._objective,
                x0,
                method='Nelder-Mead',
                bounds=bounds_list,
                options={'maxiter': 50, 'xatol': 0.05, 'fatol': 0.01}
            )
            convergence_msg = "Local search complete"
            
        elif method == 'grid':
            # Grid search with refinement
            best_cost = float('inf')
            best_x = None
            
            # Coarse grid
            for v in np.linspace(self.bounds.velocity_min, self.bounds.velocity_max, 5):
                for a in np.linspace(self.bounds.angle_min, self.bounds.angle_max, 5):
                    params = np.array([v, a]) if not self.optimize_supply_temp else np.array([v, a, 20.0])
                    cost = self._objective(params)
                    if cost < best_cost:
                        best_cost = cost
                        best_x = params
            
            # Fine grid around best
            v_range = np.linspace(max(best_x[0] - 0.2, self.bounds.velocity_min),
                                   min(best_x[0] + 0.2, self.bounds.velocity_max), 5)
            a_range = np.linspace(max(best_x[1] - 10, self.bounds.angle_min),
                                   min(best_x[1] + 10, self.bounds.angle_max), 5)
            
            for v in v_range:
                for a in a_range:
                    params = np.array([v, a]) if not self.optimize_supply_temp else np.array([v, a, 20.0])
                    cost = self._objective(params)
                    if cost < best_cost:
                        best_cost = cost
                        best_x = params
            
            class GridResult:
                x = best_x
                fun = best_cost
            
            result = GridResult()
            convergence_msg = "Grid search complete"
        
        elif method == 'hybrid':
            # HYBRID: Physics-direct calculation + ONE validation run
            # No optimization loop - the physics is deterministic
            
            # ================================================================
            # STEP 1: Compute velocity from ventilation requirements (ASHRAE 62.1)
            # ================================================================
            # Required outdoor air: 5 CFM/person + 0.06 CFM/ft² (conference room)
            # For 12 occupants in ~400 ft² room: 60 + 24 = 84 CFM minimum OA
            # But we need enough airflow to remove heat AND dilute CO2
            
            # Heat balance: Q = ṁ·cp·ΔT → ṁ = Q / (cp·ΔT)
            heat_load = self.n_occupants * 100  # W (sensible heat per person)
            # Use ACTUAL supply-room temperature delta (not hardcoded 4°C)
            # supply_temp is ~13°C, target room temp is ~22°C → ΔT ≈ 9°C
            dT = max(self.targets.temp_target - self.supply_temp, 3.0)  # minimum 3°C delta
            rho, cp = 1.2, 1005.0  # air density (kg/m³), specific heat (J/kg·K)
            required_mass_flow = heat_load / (cp * dT)  # kg/s
            required_volume_flow = required_mass_flow / rho  # m³/s
            
            # Ventilation area (assume 2 diffusers @ 0.1m² each = 0.2m² total)
            total_vent_area = 0.2
            opt_velocity = required_volume_flow / total_vent_area
            opt_velocity = np.clip(opt_velocity, self.bounds.velocity_min, self.bounds.velocity_max)
            
            # ================================================================
            # STEP 2: Compute angle from throw physics
            # ================================================================
            # Throw distance: how far air travels before V drops to terminal (~0.25 m/s)
            # For ceiling diffuser at height H, want air to reach occupant zone (1.2m)
            # without exceeding draft limit at occupied level
            #
            # Throw equation (ASHRAE): T_0.25 = K * sqrt(Q) where K depends on diffuser
            # For 4-way ceiling diffuser: K ≈ 5.0
            # 
            # Angle selection based on ceiling height and room dimensions:
            # - Low angle (15-30°): long throw, for large rooms
            # - Medium angle (30-50°): balanced, for typical conference rooms  
            # - High angle (50-75°): short throw, mixing, for small/high rooms
            
            ceiling_height = 3.0  # m (typical conference room)
            occupied_height = 1.2  # m (seated person head level)
            drop_distance = ceiling_height - occupied_height  # 1.8m
            
            # Terminal velocity constraint: V at occupied zone < 0.25 m/s
            # Velocity decay: V(x) = V0 * (x0/x)^n where n ≈ 0.5 for radial diffusers
            # Solve for angle such that vertical component reaches floor gently
            
            # Simplified: higher velocity needs steeper angle (more vertical) to 
            # dissipate energy before reaching occupants
            if opt_velocity > 1.5:
                opt_angle = 60.0  # steep, short throw
            elif opt_velocity > 1.0:
                opt_angle = 45.0  # balanced
            elif opt_velocity > 0.6:
                opt_angle = 35.0  # medium throw
            else:
                opt_angle = 25.0  # long throw for low velocity
            
            opt_angle = np.clip(opt_angle, self.bounds.angle_min, self.bounds.angle_max)
            
            print(f"    [PHYSICS-DIRECT] Computed from first principles:")
            print(f"        Heat load:     {heat_load:.0f} W ({self.n_occupants} × 100W)")
            print(f"        Required flow: {required_volume_flow*1000:.1f} L/s")
            print(f"        → Velocity:    {opt_velocity:.2f} m/s")
            print(f"        → Angle:       {opt_angle:.0f}° (throw-optimized)")
            
            # Create result object (no scipy optimization, just direct values)
            class DirectResult:
                x = np.array([opt_velocity, opt_angle])
                fun = 0.0
            result = DirectResult()
            
            convergence_msg = "Physics-direct (0 search evals)"
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Extract optimal parameters
        if self.optimize_supply_temp:
            opt_velocity, opt_angle, opt_supply_temp = result.x
        else:
            opt_velocity, opt_angle = result.x
            opt_supply_temp = self.targets.temp_target - 2.0
        
        print("\n" + "-" * 64)
        print("VALIDATING OPTIMAL SETTINGS (full 5-minute simulation)...")
        print("-" * 64)
        
        # Validate with full simulation - show progress
        val_metrics = self._run_simulation(
            opt_velocity, opt_angle, opt_supply_temp, 
            self.validation_duration,
            verbose_progress=True,
            progress_emitter=progress_callback
        )
        
        elapsed = time.time() - start_time
        
        # Build result
        opt_result = OptimizationResult(
            optimal_velocity=opt_velocity,
            optimal_angle=opt_angle,
            optimal_supply_temp=opt_supply_temp,
            final_temp=val_metrics['temperature'],
            final_co2=val_metrics['co2'],
            final_velocity=val_metrics['velocity'],
            all_pass=val_metrics['overall_pass'],
            temp_pass=val_metrics['temp_pass'],
            co2_pass=val_metrics['co2_pass'],
            velocity_pass=val_metrics['velocity_pass'],
            n_evaluations=self.n_evaluations,
            total_time=elapsed,
            convergence_message=convergence_msg,
            history=self.history
        )
        
        print(opt_result)
        
        return opt_result


def optimize_hvac(
    n_occupants: int = 12,
    target_temp: float = 22.0,
    temp_range: tuple = (20.0, 24.0),
    max_co2: float = 1000.0,
    max_velocity: float = 0.25,
    method: str = 'differential_evolution',
    verbose: bool = True
) -> OptimizationResult:
    """
    High-level API to optimize HVAC settings.
    
    Args:
        n_occupants: Number of people in room
        target_temp: Ideal temperature (°C)
        temp_range: (min, max) acceptable temperature
        max_co2: Maximum CO2 concentration (ppm)
        max_velocity: Maximum draft velocity (m/s)
        method: 'differential_evolution', 'nelder-mead', or 'grid'
        verbose: Print progress
        
    Returns:
        OptimizationResult with optimal settings
        
    Example:
        >>> result = optimize_hvac(n_occupants=20, target_temp=21.0)
        >>> print(f"Set velocity to {result.optimal_velocity:.2f} m/s")
    """
    targets = OptimizationTarget(
        temp_min=temp_range[0],
        temp_max=temp_range[1],
        temp_target=target_temp,
        co2_max=max_co2,
        velocity_max=max_velocity
    )
    
    optimizer = HVACOptimizer(
        n_occupants=n_occupants,
        targets=targets,
        verbose=verbose
    )
    
    return optimizer.optimize(method=method)


def quick_optimize(n_occupants: int = 12) -> OptimizationResult:
    """
    Fast optimization using grid search.
    
    Runs ~25 fast simulations to find good settings.
    Use for quick estimates; use optimize_hvac() for precision.
    """
    targets = OptimizationTarget()
    
    optimizer = HVACOptimizer(
        n_occupants=n_occupants,
        targets=targets,
        sim_duration=30.0,  # Very fast
        validation_duration=120.0,  # Quick validation
        verbose=True
    )
    
    return optimizer.optimize(method='grid')


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="HyperFOAM Inverse Design Optimizer"
    )
    parser.add_argument(
        "--occupants", "-n",
        type=int,
        default=12,
        help="Number of occupants (default: 12)"
    )
    parser.add_argument(
        "--target-temp", "-t",
        type=float,
        default=22.0,
        help="Target temperature in Celsius (default: 22.0)"
    )
    parser.add_argument(
        "--max-velocity", "-v",
        type=float,
        default=0.25,
        help="Maximum draft velocity in m/s (default: 0.25)"
    )
    parser.add_argument(
        "--method", "-m",
        choices=['differential_evolution', 'nelder-mead', 'grid'],
        default='differential_evolution',
        help="Optimization method (default: differential_evolution)"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick mode: faster but less precise"
    )
    
    args = parser.parse_args()
    
    if args.quick:
        result = quick_optimize(args.occupants)
    else:
        result = optimize_hvac(
            n_occupants=args.occupants,
            target_temp=args.target_temp,
            max_velocity=args.max_velocity,
            method=args.method
        )
    
    # Export recommendation
    print("\n" + "=" * 64)
    print("COPY-PASTE RECOMMENDATION:")
    print("=" * 64)
    print(f"""
For a conference room with {args.occupants} occupants:

  config.supply_velocity = {result.optimal_velocity:.2f}  # m/s
  config.supply_angle = {result.optimal_angle:.1f}       # degrees
  config.supply_temp = {result.optimal_supply_temp:.1f}        # Celsius

This achieves:
  - Temperature: {result.final_temp:.1f}°C (target: 20-24°C)
  - CO2: {result.final_co2:.0f} ppm (limit: <1000 ppm)  
  - Draft: {result.final_velocity:.3f} m/s (limit: <0.25 m/s)
""")


if __name__ == "__main__":
    main()
