"""
Ontic Real-Time Renderer
==============================

Production script that connects Ontic QTT engine to matplotlib visualization.
Renders a 1080p window into a trillion-point simulation without decompressing
the other 999 billion points.

Usage:
    python realtime_renderer.py

Requirements:
    - ontic_core.py (in same folder or installed)
    - numpy, matplotlib
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# --- THE CRITICAL IMPORT ---
try:
    import ontic_core as ht

    print("SUCCESS: Ontic Core loaded.")
except ImportError:
    print("CRITICAL ERROR: 'ontic_core.py' not found.")
    print("Place your engine file in this folder to run the Real Tech.")
    exit()

print("=" * 60)
print("--- ONTIC_ENGINE: REAL-TIME RENDERER ---")
print("Connecting Visualization to QTT Engine...")
print("=" * 60)

# --- CONFIGURATION ---
GRID_BITS = 20  # 2^20 x 2^20 = 1 trillion points total (use 20 for demo, 30 for full)
RENDER_RES = 256  # The window we actually see (256x256 for smooth FPS)

# --- INITIALIZE YOUR ENGINE ---
print("\nInitializing fluid state...")
fluid_state = ht.FluidState(grid_bits=GRID_BITS, rank=8, viscosity=0.01)
fluid_state.set_initial_condition("taylor_green")


# --- THE BRIDGE (Interface between Screen and Engine) ---
def get_real_data(view_x, view_y, zoom, resolution):
    """
    Translate "Screen Coordinates" into "Tensor Indices"
    and ask the engine to solve them.
    """
    # 1. Calculate the 'World Space' bounds of the camera
    half_width = (1.0 / zoom) * 0.5

    start_x_float = view_x - half_width
    end_x_float = view_x + half_width
    start_y_float = view_y - half_width
    end_y_float = view_y + half_width

    # Clamp to valid range
    start_x_float = max(0.0, min(start_x_float, 1.0))
    end_x_float = max(0.0, min(end_x_float, 1.0))
    start_y_float = max(0.0, min(start_y_float, 1.0))
    end_y_float = max(0.0, min(end_y_float, 1.0))

    # 2. Convert to INTEGER INDICES for the QTT
    total_size = 2**GRID_BITS

    idx_x_start = int(start_x_float * total_size)
    idx_x_end = int(end_x_float * total_size)
    idx_y_start = int(start_y_float * total_size)
    idx_y_end = int(end_y_float * total_size)

    # 3. CALL THE ENGINE
    pixel_data = fluid_state.contract_slice(
        x_range=(idx_x_start, idx_x_end),
        y_range=(idx_y_start, idx_y_end),
        out_shape=(resolution, resolution),
    )

    return pixel_data


# --- VISUALIZATION LOOP ---
print("\nSetting up visualization...")
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(10, 10), facecolor="black")
ax.axis("off")
ax.set_title(
    "The Ontic Engine: QTT Real-Time Fluid Simulation", color="white", fontsize=14, pad=20
)

# Initial frame
initial_data = get_real_data(0.5, 0.5, 1.0, RENDER_RES)
im = ax.imshow(
    initial_data,
    cmap="magma",
    origin="lower",
    extent=[0, 1, 0, 1],
    interpolation="bilinear",
)
plt.colorbar(im, ax=ax, label="Velocity Magnitude", shrink=0.8)

# Text overlays
memory_text = ax.text(
    0.02,
    0.98,
    "",
    transform=ax.transAxes,
    color="white",
    fontsize=10,
    verticalalignment="top",
    fontfamily="monospace",
)
stats_text = ax.text(
    0.02,
    0.02,
    "",
    transform=ax.transAxes,
    color="lime",
    fontsize=10,
    verticalalignment="bottom",
    fontfamily="monospace",
)

# Simulation State
current_zoom = 1.0
center_x = 0.5
center_y = 0.5
frame_times = []


def update(frame):
    global frame_times

    t_start = time.perf_counter()

    # 1. Step the Physics (In the Engine)
    fluid_state.step_physics()

    # 2. Get the Data (Only decompress what we see)
    data = get_real_data(center_x, center_y, current_zoom, RENDER_RES)

    # 3. Render
    im.set_array(data)
    if np.max(data) > 0:
        im.set_clim(vmin=0, vmax=np.percentile(data, 99))

    # 4. Update stats
    t_end = time.perf_counter()
    frame_times.append(t_end - t_start)
    if len(frame_times) > 30:
        frame_times = frame_times[-30:]

    avg_fps = 1.0 / np.mean(frame_times) if frame_times else 0

    stats = fluid_state.get_memory_stats()
    memory_text.set_text(
        f"Grid: {stats['grid_size']:,}² = {stats['total_points']:,} points\n"
        f"QTT: {stats['qtt_kb']:.1f} KB | Dense: {stats['dense_gb']:.2f} GB\n"
        f"Compression: {stats['compression']:,.0f}x"
    )
    stats_text.set_text(
        f"t = {fluid_state.time:.3f}s | FPS: {avg_fps:.1f} | Frame: {frame}"
    )

    return im, memory_text, stats_text


print("\n" + "=" * 60)
print("System Linked. Running The Ontic Engine simulation...")
print("Close window to exit.")
print("=" * 60 + "\n")

ani = FuncAnimation(fig, update, interval=50, blit=True, cache_frame_data=False)
plt.tight_layout()
plt.show()
