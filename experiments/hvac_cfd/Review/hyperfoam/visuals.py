"""
HyperFOAM Professional Visualizations

High-resolution, publication-quality CFD renders.
No pixel shit. Clean engineering aesthetics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter
from pathlib import Path


# ============================================================================
# STYLE CONFIG
# ============================================================================
HYPERFOAM_STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'axes.grid.axis': 'both',
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'figure.facecolor': 'white',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'savefig.pad_inches': 0.1,
}


def apply_style():
    """Apply HyperFOAM visual style."""
    plt.rcParams.update(HYPERFOAM_STYLE)


# ============================================================================
# VISUALIZATION CONSTANTS
# ============================================================================

# Temperature colormap range for HVAC comfort visualization (°C)
TEMP_COLORMAP_MIN = 18.0
TEMP_COLORMAP_MAX = 26.0

# Velocity colormap range for typical HVAC flows (m/s)
VEL_COLORMAP_MAX = 1.0


# ============================================================================
# COLOR MAPS
# ============================================================================
def thermal_cmap():
    """Professional thermal colormap - cool to hot."""
    colors = [
        '#2166ac',  # Deep blue (cold)
        '#67a9cf',  # Light blue
        '#d1e5f0',  # Very light blue
        '#f7f7f7',  # White (neutral)
        '#fddbc7',  # Light orange
        '#ef8a62',  # Orange
        '#b2182b',  # Deep red (hot)
    ]
    return mcolors.LinearSegmentedColormap.from_list('thermal', colors, N=256)


def velocity_cmap():
    """Professional velocity colormap - dark to bright."""
    colors = [
        '#0d0887',  # Deep purple (stagnant)
        '#5302a3',  # Purple
        '#8b0aa5',  # Magenta
        '#b83289',  # Pink
        '#db5c68',  # Salmon
        '#f48849',  # Orange
        '#febc2a',  # Yellow
        '#f0f921',  # Bright yellow (fast)
    ]
    return mcolors.LinearSegmentedColormap.from_list('velocity', colors, N=256)


def comfort_cmap():
    """Comfort zone colormap - red/yellow/green."""
    colors = [
        '#d73027',  # Red (too cold)
        '#fc8d59',  # Orange
        '#fee08b',  # Yellow
        '#d9ef8b',  # Light green
        '#91cf60',  # Green (comfortable)
        '#d9ef8b',  # Light green
        '#fee08b',  # Yellow
        '#fc8d59',  # Orange
        '#d73027',  # Red (too hot)
    ]
    return mcolors.LinearSegmentedColormap.from_list('comfort', colors, N=256)


# ============================================================================
# THERMAL HEATMAP
# ============================================================================
def render_thermal_heatmap(
    temperature_field: np.ndarray,
    lx: float,
    lz: float,
    room_name: str = "Conference Room",
    output_path: str = None,
    vmin: float = TEMP_COLORMAP_MIN,
    vmax: float = TEMP_COLORMAP_MAX,
    target_temp: float = 22.0,
    show_comfort_zones: bool = True,
    smooth: bool = True
) -> plt.Figure:
    """
    Render publication-quality thermal heatmap.
    
    Args:
        temperature_field: 2D array (x, z) in Celsius
        lx, lz: Room dimensions in meters
        room_name: Title text
        output_path: Where to save (None = don't save)
        vmin, vmax: Temperature range for colorbar
        target_temp: Target temperature for annotation
        show_comfort_zones: Draw comfort band overlay
        smooth: Apply Gaussian smoothing for anti-aliasing
    """
    apply_style()
    
    # Smooth the field to remove pixelation
    if smooth:
        T_smooth = gaussian_filter(temperature_field, sigma=1.5)
    else:
        T_smooth = temperature_field
    
    # High-res interpolation
    fig, ax = plt.subplots(figsize=(14, 6))
    
    im = ax.imshow(
        T_smooth.T,
        origin='lower',
        cmap=thermal_cmap(),
        vmin=vmin,
        vmax=vmax,
        aspect='auto',
        extent=[0, lx, 0, lz],
        interpolation='bicubic'  # Smooth interpolation
    )
    
    # Professional colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Temperature (°C)", fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Add comfort zone indicator on colorbar
    if show_comfort_zones:
        cbar.ax.axhline(y=(20 - vmin) / (vmax - vmin), color='green', linewidth=2, alpha=0.8)
        cbar.ax.axhline(y=(24 - vmin) / (vmax - vmin), color='green', linewidth=2, alpha=0.8)
        cbar.ax.text(1.5, (22 - vmin) / (vmax - vmin), 'COMFORT\nZONE', 
                     fontsize=8, va='center', color='green', fontweight='bold')
    
    # Axis labels
    ax.set_xlabel("Room Length (m)", fontsize=12)
    ax.set_ylabel("Height (m)", fontsize=12)
    
    # Title with styling
    ax.set_title(
        f"Thermal Distribution — {room_name}",
        fontsize=16,
        fontweight='bold',
        pad=15
    )
    
    # Draw floor and ceiling lines
    ax.axhline(y=0, color='saddlebrown', linewidth=3, label='Floor')
    ax.axhline(y=lz, color='gray', linewidth=3, label='Ceiling')
    
    # Add average temperature annotation
    T_avg = np.mean(T_smooth)
    status = "✓" if 20 <= T_avg <= 24 else "✗"
    ax.text(
        0.02, 0.98,
        f"Avg: {T_avg:.1f}°C {status}",
        transform=ax.transAxes,
        fontsize=12,
        fontweight='bold',
        va='top',
        ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray')
    )
    
    # Scale bar
    scale_length = 1.0  # 1 meter
    ax.plot([lx - 1.5, lx - 0.5], [0.15, 0.15], 'k-', linewidth=3)
    ax.text(lx - 1.0, 0.25, "1 m", ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


# ============================================================================
# VELOCITY FIELD
# ============================================================================
def render_velocity_field(
    u_field: np.ndarray,
    w_field: np.ndarray,
    lx: float,
    lz: float,
    room_name: str = "Conference Room",
    output_path: str = None,
    vmax: float = 1.0,
    show_streamlines: bool = True,
    show_vectors: bool = False,
    smooth: bool = True
) -> plt.Figure:
    """
    Render publication-quality velocity field with optional streamlines.
    
    Args:
        u_field: Horizontal velocity (x, z)
        w_field: Vertical velocity (x, z)
        lx, lz: Room dimensions
        room_name: Title
        output_path: Where to save
        vmax: Max velocity for colorbar
        show_streamlines: Overlay streamlines
        show_vectors: Overlay velocity vectors (arrows)
        smooth: Apply smoothing
    """
    apply_style()
    
    # Compute velocity magnitude
    vel_mag = np.sqrt(u_field**2 + w_field**2)
    
    if smooth:
        vel_mag = gaussian_filter(vel_mag, sigma=1.5)
        u_smooth = gaussian_filter(u_field, sigma=1.0)
        w_smooth = gaussian_filter(w_field, sigma=1.0)
    else:
        u_smooth, w_smooth = u_field, w_field
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Velocity magnitude background
    im = ax.imshow(
        vel_mag.T,
        origin='lower',
        cmap=velocity_cmap(),
        vmin=0,
        vmax=vmax,
        aspect='auto',
        extent=[0, lx, 0, lz],
        interpolation='bicubic'
    )
    
    # Streamlines
    if show_streamlines:
        nx, nz = u_smooth.shape
        x = np.linspace(0, lx, nx)
        z = np.linspace(0, lz, nz)
        X, Z = np.meshgrid(x, z)
        
        # Streamline seed points
        seed_x = np.linspace(0.5, lx - 0.5, 8)
        seed_z = np.linspace(0.3, lz - 0.3, 5)
        seed_points = np.array([[sx, sz] for sx in seed_x for sz in seed_z])
        
        try:
            ax.streamplot(
                X, Z, u_smooth.T, w_smooth.T,
                color='white',
                linewidth=0.8,
                density=1.5,
                arrowsize=1.2,
                arrowstyle='->',
                minlength=0.3,
                start_points=seed_points[:20]
            )
        except (ValueError, IndexError):
            # Fallback if streamplot fails (e.g., bad seed points)
            ax.streamplot(
                X, Z, u_smooth.T, w_smooth.T,
                color='white',
                linewidth=0.6,
                density=1.0,
                arrowsize=1.0
            )
    
    # Velocity vectors (quiver)
    if show_vectors:
        nx, nz = u_smooth.shape
        x = np.linspace(0, lx, nx)
        z = np.linspace(0, lz, nz)
        X, Z = np.meshgrid(x, z)
        
        # Subsample for cleaner arrows
        skip = 4
        ax.quiver(
            X[::skip, ::skip], Z[::skip, ::skip],
            u_smooth.T[::skip, ::skip], w_smooth.T[::skip, ::skip],
            color='white',
            alpha=0.7,
            scale=15,
            width=0.003
        )
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Velocity (m/s)", fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Draft limit indicator
    cbar.ax.axhline(y=0.25 / vmax, color='red', linewidth=2, linestyle='--')
    cbar.ax.text(1.5, 0.25 / vmax, 'DRAFT\nLIMIT', fontsize=8, va='center', color='red', fontweight='bold')
    
    ax.set_xlabel("Room Length (m)", fontsize=12)
    ax.set_ylabel("Height (m)", fontsize=12)
    ax.set_title(
        f"Airflow Distribution — {room_name}",
        fontsize=16,
        fontweight='bold',
        pad=15
    )
    
    # Draw floor/ceiling
    ax.axhline(y=0, color='saddlebrown', linewidth=3)
    ax.axhline(y=lz, color='gray', linewidth=3)
    
    # Max velocity annotation
    V_max = np.max(vel_mag)
    V_avg = np.mean(vel_mag)
    status = "✓" if V_max < 0.25 else "✗"
    ax.text(
        0.02, 0.98,
        f"Max: {V_max:.2f} m/s {status}\nAvg: {V_avg:.2f} m/s",
        transform=ax.transAxes,
        fontsize=11,
        fontweight='bold',
        va='top',
        ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray')
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


# ============================================================================
# CONVERGENCE PLOT
# ============================================================================
def render_convergence_plot(
    history: dict,
    spec,
    output_path: str = None
) -> plt.Figure:
    """
    Render publication-quality convergence plot.
    
    Args:
        history: Dict with 'time', 'T', 'CO2', 'V' lists
        spec: JobSpec with limits
        output_path: Where to save
    """
    apply_style()
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Simulation Convergence", fontsize=18, fontweight='bold', y=0.98)
    
    time = history['time']
    
    # --- Temperature ---
    ax = axes[0]
    ax.fill_between(time, spec.temp_min, spec.temp_max, alpha=0.2, color='green', label='Comfort Zone')
    ax.plot(time, history['T'], color='#e74c3c', linewidth=2.5, label='Temperature')
    ax.axhline(spec.temp_min, color='green', linestyle='--', alpha=0.6, linewidth=1.5)
    ax.axhline(spec.temp_max, color='green', linestyle='--', alpha=0.6, linewidth=1.5)
    
    final_T = history['T'][-1] if history['T'] else 0
    status = "✓ PASS" if spec.temp_min <= final_T <= spec.temp_max else "✗ FAIL"
    ax.text(0.98, 0.95, f"{final_T:.1f}°C {status}",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax.set_ylabel("Temperature (°C)", fontsize=12)
    ax.set_ylim(16, 28)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_title("Thermal Equilibrium", fontsize=13, fontweight='bold', loc='left')
    
    # --- CO2 ---
    ax = axes[1]
    ax.fill_between(time, 0, spec.max_co2, alpha=0.15, color='green')
    ax.plot(time, history['CO2'], color='#27ae60', linewidth=2.5, label='CO₂')
    ax.axhline(spec.max_co2, color='red', linestyle='--', linewidth=1.5, label='Limit')
    
    final_CO2 = history['CO2'][-1] if history['CO2'] else 0
    status = "✓ PASS" if final_CO2 < spec.max_co2 else "✗ FAIL"
    ax.text(0.98, 0.95, f"{final_CO2:.0f} ppm {status}",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax.set_ylabel("CO₂ (ppm)", fontsize=12)
    ax.set_ylim(350, max(1200, spec.max_co2 * 1.2))
    ax.legend(loc='upper left', fontsize=10)
    ax.set_title("Indoor Air Quality", fontsize=13, fontweight='bold', loc='left')
    
    # --- Velocity ---
    ax = axes[2]
    ax.fill_between(time, 0, spec.max_velocity, alpha=0.15, color='green')
    ax.plot(time, history['V'], color='#3498db', linewidth=2.5, label='Draft Velocity')
    ax.axhline(spec.max_velocity, color='red', linestyle='--', linewidth=1.5, label='Limit')
    
    final_V = history['V'][-1] if history['V'] else 0
    status = "✓ PASS" if final_V < spec.max_velocity else "✗ FAIL"
    ax.text(0.98, 0.95, f"{final_V:.3f} m/s {status}",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax.set_ylabel("Velocity (m/s)", fontsize=12)
    ax.set_xlabel("Simulation Time (s)", fontsize=12)
    ax.set_ylim(0, max(0.5, spec.max_velocity * 2))
    ax.legend(loc='upper left', fontsize=10)
    ax.set_title("Draft Velocity at Occupant Level", fontsize=13, fontweight='bold', loc='left')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


# ============================================================================
# COMBINED DASHBOARD VIEW
# ============================================================================
def render_dashboard_summary(
    temperature_field: np.ndarray,
    u_field: np.ndarray,
    w_field: np.ndarray,
    metrics: dict,
    spec,
    output_path: str = None
) -> plt.Figure:
    """
    Render a combined 2x2 dashboard view for quick client preview.
    """
    apply_style()
    
    fig = plt.figure(figsize=(16, 12))
    
    # Layout: 2x2 grid + metrics panel
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.4], hspace=0.3, wspace=0.3)
    
    lx, lz = spec.lx, spec.lz
    
    # Thermal
    ax1 = fig.add_subplot(gs[0, 0])
    T_smooth = gaussian_filter(temperature_field, sigma=1.5)
    im1 = ax1.imshow(T_smooth.T, origin='lower', cmap=thermal_cmap(),
                     vmin=TEMP_COLORMAP_MIN, vmax=TEMP_COLORMAP_MAX, 
                     aspect='auto', extent=[0, lx, 0, lz],
                     interpolation='bicubic')
    plt.colorbar(im1, ax=ax1, shrink=0.8, label="°C")
    ax1.set_title("Temperature Field", fontweight='bold')
    ax1.set_xlabel("Length (m)")
    ax1.set_ylabel("Height (m)")
    
    # Velocity
    ax2 = fig.add_subplot(gs[0, 1])
    vel_mag = gaussian_filter(np.sqrt(u_field**2 + w_field**2), sigma=1.5)
    im2 = ax2.imshow(vel_mag.T, origin='lower', cmap=velocity_cmap(),
                     vmin=0, vmax=VEL_COLORMAP_MAX, aspect='auto', extent=[0, lx, 0, lz],
                     interpolation='bicubic')
    plt.colorbar(im2, ax=ax2, shrink=0.8, label="m/s")
    ax2.set_title("Velocity Field", fontweight='bold')
    ax2.set_xlabel("Length (m)")
    ax2.set_ylabel("Height (m)")
    
    # Comfort Zones
    ax3 = fig.add_subplot(gs[1, 0])
    # Compute PMV-like comfort index (simplified)
    T_c = temperature_field
    comfort = 1 - np.abs(T_c - 22) / 4  # 1 = perfect, 0 = boundary
    comfort = np.clip(comfort, 0, 1)
    comfort_smooth = gaussian_filter(comfort, sigma=2)
    
    im3 = ax3.imshow(comfort_smooth.T, origin='lower', cmap='RdYlGn',
                     vmin=0, vmax=1, aspect='auto', extent=[0, lx, 0, lz],
                     interpolation='bicubic')
    plt.colorbar(im3, ax=ax3, shrink=0.8, label="Comfort Index")
    ax3.set_title("Thermal Comfort Map", fontweight='bold')
    ax3.set_xlabel("Length (m)")
    ax3.set_ylabel("Height (m)")
    
    # Streamlines overlay on velocity
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(vel_mag.T, origin='lower', cmap=velocity_cmap(),
               vmin=0, vmax=VEL_COLORMAP_MAX, aspect='auto', extent=[0, lx, 0, lz],
               interpolation='bicubic', alpha=0.6)
    
    nx, nz = u_field.shape
    x = np.linspace(0, lx, nx)
    z = np.linspace(0, lz, nz)
    X, Z = np.meshgrid(x, z)
    u_smooth = gaussian_filter(u_field, sigma=1)
    w_smooth = gaussian_filter(w_field, sigma=1)
    
    try:
        ax4.streamplot(X, Z, u_smooth.T, w_smooth.T,
                       color='black', linewidth=1, density=1.5, arrowsize=1)
    except (ValueError, IndexError):
        pass  # Streamplot may fail with insufficient data
    
    ax4.set_title("Airflow Streamlines", fontweight='bold')
    ax4.set_xlabel("Length (m)")
    ax4.set_ylabel("Height (m)")
    
    # Metrics Panel
    ax5 = fig.add_subplot(gs[:, 2])
    ax5.axis('off')
    
    # Metrics box
    metrics_text = f"""
VALIDATION RESULTS
══════════════════

Temperature
  {metrics['temperature']:.1f}°C
  {"✓ PASS" if metrics['temp_pass'] else "✗ FAIL"}

CO₂ Level
  {metrics['co2']:.0f} ppm
  {"✓ PASS" if metrics['co2_pass'] else "✗ FAIL"}

Draft Velocity
  {metrics['velocity']:.3f} m/s
  {"✓ PASS" if metrics['velocity_pass'] else "✗ FAIL"}

══════════════════
OVERALL: {"✓ ALL PASS" if metrics['overall_pass'] else "✗ FAIL"}
"""
    
    box_color = '#27ae60' if metrics['overall_pass'] else '#e74c3c'
    ax5.text(0.5, 0.5, metrics_text,
             transform=ax5.transAxes, ha='center', va='center',
             fontsize=14, fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor=box_color, alpha=0.1,
                       edgecolor=box_color, linewidth=2))
    
    fig.suptitle(f"HyperFOAM CFD Analysis — {spec.room_name}",
                 fontsize=18, fontweight='bold', y=0.98)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig
