"""
HyperTensor Physics - Example Gallery

Quick demonstrations of all major modules.
"""

import numpy as np
import matplotlib.pyplot as plt


def example_tt_compression():
    """Demonstrate TT compression on a 4D tensor."""
    from hypertensor.core import tt_round, tt_to_full
    
    print("=" * 60)
    print("TT COMPRESSION EXAMPLE")
    print("=" * 60)
    
    # Create a smooth 4D function (highly compressible)
    n = 16
    x = np.linspace(0, 1, n)
    xx, yy, zz, ww = np.meshgrid(x, x, x, x, indexing='ij')
    
    # Smooth function - rank should be low
    A = np.sin(np.pi * xx) * np.cos(np.pi * yy) * np.exp(-zz) * (1 + ww)
    
    print(f"Original shape: {A.shape}")
    print(f"Original memory: {A.nbytes:,} bytes")
    
    # Compress
    tt = tt_round(A, max_rank=5)
    
    print(f"TT ranks: {tt.ranks}")
    print(f"TT memory: {tt.memory:,} bytes")
    print(f"Compression ratio: {tt.compression_ratio:.1f}×")
    
    # Verify accuracy
    A_rec = tt_to_full(tt)
    error = np.linalg.norm(A - A_rec) / np.linalg.norm(A)
    print(f"Reconstruction error: {error:.2e}")
    
    return tt


def example_harmonic_oscillator():
    """Symplectic integration of harmonic oscillator."""
    from hypertensor.integrators import SymplecticIntegrator
    
    print("\n" + "=" * 60)
    print("HARMONIC OSCILLATOR (Symplectic)")
    print("=" * 60)
    
    def force(x):
        return -x  # F = -kx (k=1)
    
    integrator = SymplecticIntegrator(force, mass=1.0)
    
    # Initial conditions
    x = np.array([1.0])
    v = np.array([0.0])
    
    trajectory_x = [x[0]]
    trajectory_v = [v[0]]
    energies = []
    
    n_steps = 1000
    dt = 0.01
    
    for _ in range(n_steps):
        x, v = integrator.step(x, v, dt=dt)
        trajectory_x.append(x[0])
        trajectory_v.append(v[0])
        energies.append(0.5 * v[0]**2 + 0.5 * x[0]**2)
    
    # Plot phase space
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(trajectory_x, trajectory_v, 'b-', alpha=0.7)
    ax1.set_xlabel("Position x")
    ax1.set_ylabel("Velocity v")
    ax1.set_title("Phase Space (should be closed orbit)")
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(energies, 'g-')
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Energy")
    ax2.set_title(f"Energy Conservation (Δ = {100*(max(energies)-min(energies))/energies[0]:.4f}%)")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("harmonic_oscillator.png", dpi=150)
    print("Saved: harmonic_oscillator.png")
    plt.close()
    
    return trajectory_x, energies


def example_langevin_dynamics():
    """Finite-temperature molecular dynamics."""
    from hypertensor.integrators import LangevinDynamics
    
    print("\n" + "=" * 60)
    print("LANGEVIN DYNAMICS (T=300K)")
    print("=" * 60)
    
    def harmonic_potential(x):
        return 0.5 * np.sum(x**2)
    
    langevin = LangevinDynamics(
        potential_fn=harmonic_potential,
        temperature=300,
        friction=1.0
    )
    
    x0 = np.array([5.0, 5.0, 5.0])  # Start far from equilibrium
    result = langevin.run(x0, n_steps=500, dt=1e-14)
    
    print(f"Final position: {result['final_position']}")
    print(f"RMSD from origin: {result['rmsd']:.2f} Å")
    
    # Plot trajectory
    if len(result["trajectory"]) > 0:
        traj = np.array(result["trajectory"])
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.5, lw=0.5)
        ax.scatter(traj[0, 0], traj[0, 1], c='green', s=100, zorder=5, label='Start')
        ax.scatter(traj[-1, 0], traj[-1, 1], c='red', s=100, zorder=5, label='End')
        ax.set_xlabel("x (Å)")
        ax.set_ylabel("y (Å)")
        ax.set_title("Langevin Dynamics Trajectory (xy-projection)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig("langevin_trajectory.png", dpi=150)
        print("Saved: langevin_trajectory.png")
        plt.close()
    
    return result


def example_mhd_reconnection():
    """Magnetic reconnection in Harris current sheet."""
    from hypertensor.pde import ResistiveMHD
    
    print("\n" + "=" * 60)
    print("MHD MAGNETIC RECONNECTION")
    print("=" * 60)
    
    mhd = ResistiveMHD(nx=128, L=1.0, eta=0.01)
    result = mhd.run(n_steps=100, dt=1e-5)
    
    print(f"Stable: {result['stable']}")
    print(f"Final B range: [{result['final_B'].min():.3f}, {result['final_B'].max():.3f}]")
    
    # Plot final B profile
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(0, 1, len(result['final_B']))
    ax.plot(x, result['final_B'], 'b-', lw=2)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("x / L")
    ax.set_ylabel("B_z")
    ax.set_title("Harris Sheet After Resistive Diffusion")
    ax.grid(True, alpha=0.3)
    plt.savefig("mhd_harris_sheet.png", dpi=150)
    print("Saved: mhd_harris_sheet.png")
    plt.close()
    
    return result


def example_fokker_planck():
    """Probability evolution under drift-diffusion."""
    from hypertensor.pde import FokkerPlanck
    
    print("\n" + "=" * 60)
    print("FOKKER-PLANCK RELAXATION")
    print("=" * 60)
    
    fp = FokkerPlanck(nx=128, x_range=(-6, 6), diffusion=0.5)
    
    # Start with off-center Gaussian
    P0 = fp.initialize_gaussian(mean=3.0, std=0.5)
    
    # Capture snapshots
    snapshots = [P0.copy()]
    P = P0.copy()
    
    for i in range(4):
        result = fp.run(P, n_steps=100, dt=0.01)
        P = result['final_P']
        snapshots.append(P.copy())
    
    # Plot evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(-6, 6, len(P0))
    
    for i, snap in enumerate(snapshots):
        alpha = 0.3 + 0.7 * i / len(snapshots)
        ax.plot(x, snap, alpha=alpha, lw=2, label=f"t = {i * 100 * 0.01:.1f}")
    
    ax.set_xlabel("x")
    ax.set_ylabel("P(x)")
    ax.set_title("Fokker-Planck: Relaxation to Equilibrium")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig("fokker_planck_evolution.png", dpi=150)
    print("Saved: fokker_planck_evolution.png")
    plt.close()
    
    return snapshots


def example_composite_wall():
    """Heat transfer through multi-layer wall."""
    from hypertensor.pde import CompositeWall
    
    print("\n" + "=" * 60)
    print("COMPOSITE WALL HEAT TRANSFER")
    print("=" * 60)
    
    # Fusion reactor first wall example
    # Tungsten armor -> Steel structure -> Coolant channel
    wall = CompositeWall(
        layers=[
            (0.005, 170),    # 5mm tungsten (k = 170 W/m·K)
            (0.020, 30),     # 20mm steel (k = 30 W/m·K)
            (0.003, 20),     # 3mm cooling tube wall (k = 20 W/m·K)
        ],
        T_hot=1500,          # Plasma-facing surface
        T_cold=300,          # Coolant temperature
        h_conv=50000         # Forced convection to water
    )
    
    result = wall.analyze()
    
    print(f"\nHeat flux: {result['heat_flux_W_m2']/1e6:.2f} MW/m²")
    print(f"Surface temperature: {result['T_surface']:.0f} K")
    print(f"Interface temperatures: {[f'{T:.0f}' for T in result['interface_temps']]} K")
    print(f"Total thermal resistance: {result['total_resistance']:.6f} K·m²/W")
    
    # Plot temperature profile
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_positions = np.cumsum([0] + [L for L, _ in wall.layers]) * 1000  # mm
    temps = result['interface_temps']
    
    ax.plot(x_positions, temps, 'ro-', markersize=10, lw=2)
    
    # Fill layers with different colors
    layer_names = ['Tungsten', 'Steel', 'Tube Wall']
    colors = ['gray', 'steelblue', 'brown']
    
    for i, (name, color) in enumerate(zip(layer_names, colors)):
        ax.axvspan(x_positions[i], x_positions[i+1], alpha=0.3, 
                   color=color, label=name)
    
    ax.set_xlabel("Position (mm)")
    ax.set_ylabel("Temperature (K)")
    ax.set_title(f"Fusion Reactor First Wall - q = {result['heat_flux_W_m2']/1e6:.2f} MW/m²")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig("composite_wall.png", dpi=150)
    print("Saved: composite_wall.png")
    plt.close()
    
    return result


def main():
    """Run all examples."""
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " HYPERTENSOR PHYSICS ENGINE - EXAMPLE GALLERY ".center(58) + "║")
    print("╚" + "═" * 58 + "╝\n")
    
    example_tt_compression()
    example_harmonic_oscillator()
    example_langevin_dynamics()
    example_mhd_reconnection()
    example_fokker_planck()
    example_composite_wall()
    
    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
