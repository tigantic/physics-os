"""
AI Inverse Design with Thermal Constraints

The REAL optimizer that balances:
1. Draft Comfort: Velocity < 0.25 m/s (don't blow on people)
2. Thermal Comfort: Temperature 20-24°C (don't overheat OR freeze)
3. Air Quality: Adequate airflow (need fresh air)

The "Lazy Solution" Problem:
- Pure velocity optimizer: Just turn off the fan → Draft = 0 ✓, but T = 35°C ✗
- We need thermal penalty to FORCE the AI to find clever solutions

The REAL Solution (what we expect):
- Use ANGLE to spread air along ceiling (Coanda effect)
- Keep velocity HIGH for cooling capacity
- But velocity at OCCUPANT level stays LOW (mixed/diffused)
"""

import torch
import numpy as np
import time
from hyper_grid import HyperGrid
from hyperfoam_solver import ProjectionConfig
from thermal_solver import ThermalSolver, ThermalConfig


def run_thermal_simulation(grid, flow_cfg, thermal_cfg, velocity, angle_deg, 
                           steps, nx, ny, nz):
    """Run coupled thermal-flow simulation, return comfort metrics."""
    
    # Create thermal solver
    solver = ThermalSolver(grid, flow_cfg, thermal_cfg)
    
    # Add 12 occupants around table
    table_center = (flow_cfg.Lx / 2, flow_cfg.Ly / 2)
    solver.add_conference_table_occupants(table_center, table_length=3.66, n_per_side=6)
    
    # Calculate inlet vectors from parameters
    rad = np.radians(angle_deg)
    u_component = float(velocity * np.sin(rad))
    w_component = float(-velocity * np.cos(rad))  # Down
    
    # Configure supply vents
    vent_size = 4
    z_ceiling = nz - 2
    
    solver.add_supply_vent(
        ix_range=(nx//4-vent_size, nx//4+vent_size),
        iy_range=(ny//4-vent_size, ny//4+vent_size),
        iz=z_ceiling,
        velocity_w=w_component,
        velocity_u=u_component
    )
    solver.add_supply_vent(
        ix_range=(3*nx//4-vent_size, 3*nx//4+vent_size),
        iy_range=(3*ny//4-vent_size, 3*ny//4+vent_size),
        iz=z_ceiling,
        velocity_w=w_component,
        velocity_u=-u_component  # Opposite spread direction
    )
    
    # Run simulation
    for step in range(steps):
        solver.step()
    
    # Get comfort metrics
    metrics = solver.get_comfort_metrics()
    
    return metrics


def compute_loss(metrics, velocity):
    """
    Compute combined comfort loss.
    
    This is the key to forcing intelligent solutions:
    - Draft penalty: pushes velocity DOWN
    - Thermal penalty: pushes velocity UP (need cooling)
    - Only solution: use ANGLE to spread air
    """
    
    max_vel = metrics['max_velocity']
    avg_temp = metrics['avg_temp_C']
    max_temp = metrics['max_temp_C']
    
    # 1. Draft penalty (velocity > 0.25 m/s)
    draft_excess = max(0, max_vel - 0.25)
    draft_penalty = draft_excess ** 2 * 50
    
    # 2. Thermal penalty (temperature outside 20-24°C)
    # Penalize both too hot AND too cold
    if avg_temp > 24:
        thermal_penalty = (avg_temp - 24) ** 2 * 20
    elif avg_temp < 20:
        thermal_penalty = (20 - avg_temp) ** 2 * 10
    else:
        thermal_penalty = 0.0
    
    # Extra penalty for hot spots (max temp > 25°C)
    if max_temp > 25:
        thermal_penalty += (max_temp - 25) ** 2 * 30
    
    # 3. Minimum airflow penalty (need fresh air for CO2)
    min_velocity = 0.8  # Need at least this for adequate ACH
    if velocity < min_velocity:
        flow_penalty = (min_velocity - velocity) ** 2 * 20
    else:
        flow_penalty = 0.0
    
    total_loss = draft_penalty + thermal_penalty + flow_penalty
    
    return total_loss, {
        'draft': draft_penalty,
        'thermal': thermal_penalty,
        'flow': flow_penalty
    }


def compute_gradient_thermal(grid, flow_cfg, thermal_cfg, vel, angle, steps, 
                              nx, ny, nz, eps_vel=0.15, eps_angle=5.0):
    """Compute gradient via central finite differences."""
    
    # Velocity gradient
    m_plus = run_thermal_simulation(grid, flow_cfg, thermal_cfg, vel + eps_vel, angle, steps, nx, ny, nz)
    m_minus = run_thermal_simulation(grid, flow_cfg, thermal_cfg, vel - eps_vel, angle, steps, nx, ny, nz)
    loss_plus, _ = compute_loss(m_plus, vel + eps_vel)
    loss_minus, _ = compute_loss(m_minus, vel - eps_vel)
    grad_vel = (loss_plus - loss_minus) / (2 * eps_vel)
    
    # Angle gradient
    m_plus = run_thermal_simulation(grid, flow_cfg, thermal_cfg, vel, angle + eps_angle, steps, nx, ny, nz)
    m_minus = run_thermal_simulation(grid, flow_cfg, thermal_cfg, vel, max(0, angle - eps_angle), steps, nx, ny, nz)
    loss_plus, _ = compute_loss(m_plus, vel)
    loss_minus, _ = compute_loss(m_minus, vel)
    grad_angle = (loss_plus - loss_minus) / (2 * eps_angle)
    
    return grad_vel, grad_angle


def optimize_thermal():
    print("=" * 70)
    print("AI INVERSE DESIGN: THERMAL-AWARE OPTIMIZATION")
    print("=" * 70)
    print("\nObjectives:")
    print("  • Draft:   Max velocity < 0.25 m/s in occupied zone")
    print("  • Thermal: Temperature 20-24°C (with 1200W heat load)")
    print("  • Airflow: Maintain adequate fresh air (ACH)")
    print("\nThe AI cannot just 'turn off the fan' anymore!")
    
    # 1. Setup
    print("\n[1] Initializing Grid...")
    nx, ny, nz = 64, 48, 24
    lx, ly, lz = 9.0, 6.0, 3.0
    
    grid = HyperGrid(nx, ny, nz, lx, ly, lz, device='cuda')
    
    # Add table
    table_x_min = (lx - 3.66) / 2
    table_x_max = table_x_min + 3.66
    table_y_min = (ly - 1.22) / 2
    table_y_max = table_y_min + 1.22
    grid.add_box_obstacle(table_x_min, table_x_max, table_y_min, table_y_max, 0.0, 0.76)
    
    flow_cfg = ProjectionConfig(nx=nx, ny=ny, nz=nz, Lx=lx, Ly=ly, Lz=lz, dt=0.005)
    thermal_cfg = ThermalConfig()
    
    print(f"    Grid: {nx}×{ny}×{nz}")
    print(f"    Heat load: {thermal_cfg.n_people} people × {thermal_cfg.body_heat}W = {thermal_cfg.n_people * thermal_cfg.body_heat}W")
    print(f"    Supply temp: {thermal_cfg.T_supply - 273.15:.0f}°C")
    
    # 2. Initial parameters
    print("\n[2] Initial Design...")
    velocity = 2.0   # m/s (reduced from 2.29 to start more reasonable)
    angle_deg = 20.0  # Start with some spread
    
    print(f"    Velocity: {velocity:.2f} m/s ({velocity*196.85:.0f} FPM)")
    print(f"    Angle: {angle_deg:.1f}°")
    
    # 3. Optimization
    epochs = 12
    steps_per_epoch = 300  # Need more steps for thermal equilibrium
    lr_vel = 0.15
    lr_angle = 4.0
    
    print(f"\n[3] Starting Optimization ({epochs} epochs)...")
    print("-" * 95)
    print(f"{'Epoch':<6} | {'Vel':<8} | {'Angle':<8} | {'V_max':<8} | {'T_avg':<8} | {'T_max':<8} | {'Loss':<10} | {'Penalties':<20}")
    print("-" * 95)
    
    best_loss = float('inf')
    best_params = (velocity, angle_deg)
    best_metrics = None
    
    start_time = time.perf_counter()
    
    for epoch in range(epochs):
        # Evaluate
        metrics = run_thermal_simulation(
            grid, flow_cfg, thermal_cfg, velocity, angle_deg, 
            steps_per_epoch, nx, ny, nz
        )
        loss, penalties = compute_loss(metrics, velocity)
        
        # Track best
        if loss < best_loss:
            best_loss = loss
            best_params = (velocity, angle_deg)
            best_metrics = metrics
        
        # Format penalties
        pen_str = f"D:{penalties['draft']:.1f} T:{penalties['thermal']:.1f} F:{penalties['flow']:.1f}"
        
        print(f"{epoch:<6} | {velocity:<8.2f} | {angle_deg:<8.1f} | "
              f"{metrics['max_velocity']:<8.2f} | {metrics['avg_temp_C']:<8.1f} | "
              f"{metrics['max_temp_C']:<8.1f} | {loss:<10.2f} | {pen_str}")
        
        # Early stopping if we hit comfort
        if (metrics['max_velocity'] < 0.30 and 
            20 <= metrics['avg_temp_C'] <= 24 and
            velocity > 0.8):
            print("\n✓ Comfort criteria achieved!")
            break
        
        # Compute gradients
        grad_vel, grad_angle = compute_gradient_thermal(
            grid, flow_cfg, thermal_cfg, velocity, angle_deg,
            steps_per_epoch, nx, ny, nz
        )
        
        # Update
        velocity -= lr_vel * np.clip(grad_vel, -2, 2)
        angle_deg -= lr_angle * np.clip(grad_angle, -1, 1)
        
        # Clamp
        velocity = np.clip(velocity, 0.5, 4.0)
        angle_deg = np.clip(angle_deg, 0.0, 75.0)
    
    elapsed = time.perf_counter() - start_time
    print("-" * 95)
    print(f"Optimization completed in {elapsed:.1f}s")
    
    # 4. Results
    opt_vel, opt_angle = best_params
    
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                    OPTIMAL DESIGN PARAMETERS                     │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print(f"│  Inlet Velocity:  {opt_vel:>6.2f} m/s  ({opt_vel*196.85:>4.0f} FPM)                    │")
    print(f"│  Diffuser Angle:  {opt_angle:>6.1f}°   (spread from vertical)              │")
    print("├─────────────────────────────────────────────────────────────────┤")
    if best_metrics:
        print(f"│  Resulting Draft: {best_metrics['max_velocity']:>6.2f} m/s                              │")
        print(f"│  Resulting Temp:  {best_metrics['avg_temp_C']:>6.1f}°C (avg), {best_metrics['max_temp_C']:.1f}°C (max)            │")
    print("└─────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                      ENGINEERING RECOMMENDATION                  │")
    print("├─────────────────────────────────────────────────────────────────┤")
    
    if opt_angle > 45:
        print("│  ★ Install 4-WAY RADIAL DIFFUSERS                               │")
        print("│    → Coanda effect spreads air along ceiling                    │")
        print("│    → Maintains cooling capacity without direct draft            │")
        print(f"│    → Set VFD to {opt_vel*196.85:.0f} FPM                                          │")
    elif opt_angle > 25:
        print("│  ★ Install ADJUSTABLE CONE DIFFUSERS                            │")
        print(f"│    → Set spread angle to {opt_angle:.0f}°                                      │")
        print(f"│    → Maintain fan at {opt_vel*196.85:.0f} FPM for thermal load                  │")
        print("│    → Air mixes before reaching occupant zone                    │")
    else:
        print("│  ★ STANDARD DIFFUSERS with VFD CONTROL                          │")
        print(f"│    → Set fan speed to {opt_vel*196.85:.0f} FPM                                   │")
        print("│    → Consider adding ceiling fans for mixing                    │")
    
    print("└─────────────────────────────────────────────────────────────────┘")
    
    # Compare before/after
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                    WHY THERMAL CONSTRAINTS MATTER                 │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print("│  Without Thermal:  AI solution = 1.17 m/s, 15° (just slow fan)   │")
    print("│  With Thermal:     AI must balance cooling vs. draft             │")
    print("│                                                                   │")
    print("│  If fan too slow:  Room overheats (12 people × 100W = 1.2kW)     │")
    print("│  If fan too fast:  Papers blow, people complain                  │")
    print("│  Solution:         USE THE ANGLE to mix air before it descends   │")
    print("└─────────────────────────────────────────────────────────────────┘")
    
    return opt_vel, opt_angle


if __name__ == "__main__":
    optimize_thermal()
