"""Visualization module for scenario output packages.

Generates:
- 2D field contour plots (PNG)
- Conservation / rank / norm time-series plots (PNG)
- Time-evolution animations (GIF + MP4)
- Per-scenario thumbnail grids

All outputs are written to ``output_dir/images/`` and
``output_dir/videos/``.
"""

from __future__ import annotations

import base64
import logging
import math
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.figure import Figure

logger = logging.getLogger("scenario_viz")

# ── Style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

# Colour maps per physical quantity
CMAPS: dict[str, str] = {
    "omega": "RdBu_r",     # vorticity: diverging
    "psi": "viridis",      # stream function: sequential
    "u": "inferno",        # concentration / temperature
    "E": "plasma",         # electric field
    "B": "cividis",        # magnetic field
}


# ═══════════════════════════════════════════════════════════════════
# Field contour plots
# ═══════════════════════════════════════════════════════════════════


def plot_field_contour(
    values: list[Any],
    field_name: str,
    unit: str,
    title: str,
    grid_info: dict[str, Any],
) -> Figure:
    """Create a filled-contour plot of a 2D field.

    Parameters
    ----------
    values : list
        Flat or nested list of field values.
    field_name : str
        Field key (for colourmap selection).
    unit : str
        Physical unit label.
    title : str
        Plot title.
    grid_info : dict
        Grid metadata (dimensions, resolution, domain_bounds).
    """
    arr = np.asarray(values, dtype=np.float64)

    # Determine 2D shape
    dims = grid_info.get("dimensions", 2)
    res = grid_info.get("resolution", [])
    if arr.ndim == 1 and dims == 2 and len(res) == 2:
        ny, nx = res
        arr = arr.reshape(ny, nx)
    elif arr.ndim == 1:
        side = int(math.isqrt(len(arr)))
        if side * side == len(arr):
            arr = arr.reshape(side, side)
        else:
            # 1D field — make a line plot instead
            return _plot_field_1d(arr, field_name, unit, title)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    bounds = grid_info.get("domain_bounds", [])
    if isinstance(bounds, list) and len(bounds) >= 2:
        # Format: [[x_min, x_max], [y_min, y_max]]
        x_min, x_max = bounds[0][0], bounds[0][1]
        y_min, y_max = bounds[1][0], bounds[1][1]
    elif isinstance(bounds, dict):
        x_min = bounds.get("x_min", 0.0)
        x_max = bounds.get("x_max", float(arr.shape[1]))
        y_min = bounds.get("y_min", 0.0)
        y_max = bounds.get("y_max", float(arr.shape[0]))
    else:
        x_min, y_min = 0.0, 0.0
        x_max = float(arr.shape[1] if arr.ndim == 2 else len(arr))
        y_max = float(arr.shape[0] if arr.ndim == 2 else 1)

    cmap = CMAPS.get(field_name, "viridis")

    fig, ax = plt.subplots(figsize=(7, 5))
    if field_name == "omega":
        vmax = max(abs(arr.min()), abs(arr.max())) or 1.0
        norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
        im = ax.imshow(
            arr, extent=[x_min, x_max, y_min, y_max],
            origin="lower", cmap=cmap, norm=norm, aspect="auto",
        )
    else:
        im = ax.imshow(
            arr, extent=[x_min, x_max, y_min, y_max],
            origin="lower", cmap=cmap, aspect="auto",
        )

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(f"{field_name} [{unit}]")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    return fig


def _plot_field_1d(
    arr: np.ndarray,
    field_name: str,
    unit: str,
    title: str,
) -> Figure:
    """Fallback: line plot for 1D fields."""
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(arr, linewidth=1.2, color="#1f77b4")
    ax.set_xlabel("Grid index")
    ax.set_ylabel(f"{field_name} [{unit}]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig


# ═══════════════════════════════════════════════════════════════════
# Time-series diagnostics
# ═══════════════════════════════════════════════════════════════════


def plot_diagnostics(
    telemetry_steps: list[dict[str, Any]],
    scenario_name: str,
) -> Figure:
    """Plot conservation error, rank, and field norms vs time step.

    Uses per-step telemetry data when available.
    """
    if not telemetry_steps:
        fig, ax = plt.subplots(figsize=(7, 2))
        ax.text(0.5, 0.5, "No telemetry data", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="gray")
        ax.set_title(f"Diagnostics — {scenario_name}")
        return fig

    steps_idx = [s.get("step", i) for i, s in enumerate(telemetry_steps)]
    ranks = [s.get("chi_max", 0) for s in telemetry_steps]
    wall_times = [s.get("wall_time_s", 0) for s in telemetry_steps]

    # Extract invariant values if present
    inv_values = []
    inv_name = ""
    for s in telemetry_steps:
        iv = s.get("invariant_values", {})
        if iv:
            inv_name = list(iv.keys())[0]
            inv_values.append(iv[inv_name])

    n_panels = 2 + (1 if inv_values else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 2.5 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    # Panel 1: Rank evolution
    axes[0].plot(steps_idx, ranks, color="#e74c3c", linewidth=1.2)
    axes[0].set_ylabel("χ_max (rank)")
    axes[0].set_title(f"Diagnostics — {scenario_name}")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Wall time per step
    axes[1].plot(steps_idx, wall_times, color="#3498db", linewidth=1.0)
    axes[1].set_ylabel("Step time [s]")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Conservation (optional)
    if inv_values and len(axes) > 2:
        initial = inv_values[0] if inv_values[0] != 0 else 1.0
        rel_drift = [(v - inv_values[0]) / abs(initial) for v in inv_values]
        axes[2].semilogy(
            steps_idx[: len(rel_drift)],
            [abs(d) + 1e-16 for d in rel_drift],
            color="#2ecc71", linewidth=1.2,
        )
        axes[2].set_ylabel(f"|Δ{inv_name}| / |{inv_name}₀|")
        axes[2].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time step")
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════
# Claims / validation summary chart
# ═══════════════════════════════════════════════════════════════════


def plot_claims_matrix(
    scenarios: list[dict[str, Any]],
) -> Figure:
    """Heatmap-style chart of claims satisfaction across scenarios."""
    # Collect all unique claim tags
    all_tags: list[str] = []
    for sc in scenarios:
        for c in sc.get("claims", []):
            if c["tag"] not in all_tags:
                all_tags.append(c["tag"])

    n_sc = len(scenarios)
    n_tags = len(all_tags) if all_tags else 1

    matrix = np.full((n_sc, n_tags), np.nan)
    labels_y = []

    for i, sc in enumerate(scenarios):
        short = sc["name"][:30]
        labels_y.append(f"[{sc['scenario_index']}] {short}")
        for c in sc.get("claims", []):
            j = all_tags.index(c["tag"])
            matrix[i, j] = 1.0 if c["satisfied"] else 0.0

    fig, ax = plt.subplots(figsize=(max(4, n_tags * 1.8), max(4, n_sc * 0.45)))
    cmap = mcolors.ListedColormap(["#e74c3c", "#2ecc71"])
    bounds = [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(n_tags))
    ax.set_xticklabels(all_tags, rotation=45, ha="right")
    ax.set_yticks(range(n_sc))
    ax.set_yticklabels(labels_y, fontsize=8)
    ax.set_title("Claims Satisfaction Matrix")

    # Add text annotations
    for i in range(n_sc):
        for j in range(n_tags):
            val = matrix[i, j]
            if not np.isnan(val):
                txt = "✓" if val == 1.0 else "✗"
                color = "white" if val == 0.0 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=11, color=color, fontweight="bold")

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════
# Summary bar chart
# ═══════════════════════════════════════════════════════════════════


def plot_performance_bars(
    scenarios: list[dict[str, Any]],
) -> Figure:
    """Horizontal bar chart of wall time and throughput per scenario."""
    names = []
    wall_times = []
    throughputs = []

    for sc in scenarios:
        names.append(f"[{sc['scenario_index']}] {sc['name'][:25]}")
        perf = sc.get("result", {}).get("performance", {})
        wall_times.append(perf.get("wall_time_s", 0))
        throughputs.append(perf.get("throughput_gp_per_s", 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(4, len(names) * 0.4)))

    y = range(len(names))
    ax1.barh(y, wall_times, color="#3498db", edgecolor="white")
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=7)
    ax1.set_xlabel("Wall time [s]")
    ax1.set_title("Execution Time")
    ax1.invert_yaxis()

    ax2.barh(y, throughputs, color="#e67e22", edgecolor="white")
    ax2.set_yticks(y)
    ax2.set_yticklabels([], fontsize=7)
    ax2.set_xlabel("Throughput [gp/s]")
    ax2.set_title("Grid-Point Throughput")
    ax2.invert_yaxis()

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════
# Time-evolution animation (GIF / MP4)
# ═══════════════════════════════════════════════════════════════════


def create_evolution_frames(
    snapshots: list[dict[str, Any]],
    field_name: str,
    unit: str,
    scenario_name: str,
    grid_info: dict[str, Any],
) -> list[np.ndarray]:
    """Render frames from a list of field snapshots.

    Each snapshot is ``{"step": int, "values": list}``.
    Returns list of RGBA numpy arrays suitable for imageio.
    """
    if not snapshots:
        return []

    frames: list[np.ndarray] = []
    # Determine global colour range across all snapshots
    all_vals = []
    for snap in snapshots:
        arr = np.asarray(snap["values"], dtype=np.float64)
        all_vals.extend(arr.ravel().tolist())
    vmin_global = min(all_vals)
    vmax_global = max(all_vals)

    cmap = CMAPS.get(field_name, "viridis")

    for snap in snapshots:
        arr = np.asarray(snap["values"], dtype=np.float64)
        side = int(math.isqrt(len(arr)))
        if side * side == len(arr) and arr.ndim == 1:
            arr = arr.reshape(side, side)

        fig, ax = plt.subplots(figsize=(5, 4))
        bounds = grid_info.get("domain_bounds", [])
        if isinstance(bounds, list) and len(bounds) >= 2:
            ext = [bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]]
        elif isinstance(bounds, dict):
            ext = [
                bounds.get("x_min", 0),
                bounds.get("x_max", arr.shape[1] if arr.ndim == 2 else len(arr)),
                bounds.get("y_min", 0),
                bounds.get("y_max", arr.shape[0] if arr.ndim == 2 else 1),
            ]
        else:
            ext = [0, arr.shape[1] if arr.ndim == 2 else len(arr),
                   0, arr.shape[0] if arr.ndim == 2 else 1]

        if field_name == "omega":
            vabs = max(abs(vmin_global), abs(vmax_global)) or 1.0
            norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vabs, vmax=vabs)
            ax.imshow(arr, extent=ext, origin="lower", cmap=cmap, norm=norm, aspect="auto")
        else:
            ax.imshow(arr, extent=ext, origin="lower", cmap=cmap,
                      vmin=vmin_global, vmax=vmax_global, aspect="auto")

        ax.set_title(f"{scenario_name}\n{field_name} [{unit}]  —  step {snap['step']}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.tight_layout()

        # Render to numpy array
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frames.append(frame.copy())
        plt.close(fig)

    return frames


def save_animation(
    frames: list[np.ndarray],
    output_path: Path,
    fps: int = 5,
) -> None:
    """Save frames as GIF and optionally MP4."""
    import imageio.v3 as iio

    if not frames:
        return

    # GIF
    gif_path = output_path.with_suffix(".gif")
    iio.imwrite(gif_path, frames, duration=1000 // fps, loop=0)
    logger.info("Wrote animation: %s", gif_path)

    # MP4 (if ffmpeg available)
    try:
        import shutil
        if shutil.which("ffmpeg"):
            mp4_path = output_path.with_suffix(".mp4")
            iio.imwrite(mp4_path, frames, fps=fps, codec="libx264")
            logger.info("Wrote video: %s", mp4_path)
    except Exception as exc:
        logger.warning("MP4 generation failed: %s", exc)


# ═══════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════


def fig_to_base64(fig: Figure) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def fig_to_file(fig: Figure, path: Path) -> None:
    """Save a figure to a PNG file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), format="png", dpi=150)
    plt.close(fig)
    logger.info("Wrote image: %s", path)


# ═══════════════════════════════════════════════════════════════════
# Master visualization pipeline
# ═══════════════════════════════════════════════════════════════════


def generate_all_visuals(
    scenarios: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Generate all images and videos for a scenario batch.

    Returns a manifest of generated files for the report generator.
    """
    images_dir = output_dir / "images"
    videos_dir = output_dir / "videos"
    images_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "images": {},
        "videos": {},
        "base64": {},
    }

    # ── Per-scenario field plots ────────────────────────────────
    for sc in scenarios:
        idx = sc["scenario_index"]
        name = sc["name"]
        result = sc.get("result", {})
        fields = result.get("fields") or {}
        grid_info = result.get("grid", {})

        if sc["status"] != "success":
            continue

        sc_images: dict[str, str] = {}
        sc_b64: dict[str, str] = {}

        for fname, fdata in fields.items():
            values = fdata.get("values", [])
            unit = fdata.get("unit", "")
            if not values:
                continue

            title = f"{name}\n{fname} [{unit}]"
            fig = plot_field_contour(values, fname, unit, title, grid_info)

            img_name = f"scenario_{idx:02d}_{fname}_contour.png"
            img_path = images_dir / img_name
            fig_to_file(fig, img_path)
            sc_images[fname] = str(img_path.relative_to(output_dir))

            # Also store base64 for HTML embedding
            fig2 = plot_field_contour(values, fname, unit, title, grid_info)
            sc_b64[fname] = fig_to_base64(fig2)

        # Diagnostics plot from telemetry
        telemetry = sc.get("telemetry_steps", [])
        if telemetry:
            diag_fig = plot_diagnostics(telemetry, name)
            diag_path = images_dir / f"scenario_{idx:02d}_diagnostics.png"
            fig_to_file(diag_fig, diag_path)
            sc_images["diagnostics"] = str(diag_path.relative_to(output_dir))

            diag_fig2 = plot_diagnostics(telemetry, name)
            sc_b64["diagnostics"] = fig_to_base64(diag_fig2)

        # Time evolution animation from snapshots
        snapshots = sc.get("field_snapshots", {})
        for fname, snaps in snapshots.items():
            if len(snaps) < 3:
                continue
            unit = fields.get(fname, {}).get("unit", "")
            frame_list = create_evolution_frames(
                snaps, fname, unit, name, grid_info,
            )
            if frame_list:
                anim_base = videos_dir / f"scenario_{idx:02d}_{fname}_evolution"
                save_animation(frame_list, anim_base, fps=4)
                manifest["videos"][f"{idx}_{fname}"] = str(
                    anim_base.with_suffix(".gif").relative_to(output_dir)
                )

        manifest["images"][idx] = sc_images
        manifest["base64"][idx] = sc_b64

    # ── Cross-scenario summary charts ───────────────────────────
    succeeded = [s for s in scenarios if s["status"] == "success"]

    claims_fig = plot_claims_matrix(succeeded)
    claims_path = images_dir / "claims_matrix.png"
    fig_to_file(claims_fig, claims_path)
    manifest["images"]["claims_matrix"] = str(claims_path.relative_to(output_dir))

    claims_fig2 = plot_claims_matrix(succeeded)
    manifest["base64"]["claims_matrix"] = fig_to_base64(claims_fig2)

    perf_fig = plot_performance_bars(succeeded)
    perf_path = images_dir / "performance_bars.png"
    fig_to_file(perf_fig, perf_path)
    manifest["images"]["performance_bars"] = str(perf_path.relative_to(output_dir))

    perf_fig2 = plot_performance_bars(succeeded)
    manifest["base64"]["performance_bars"] = fig_to_base64(perf_fig2)

    logger.info(
        "Visualization complete: %d scenario plots, %d summary charts",
        len([s for s in scenarios if s["status"] == "success"]),
        2,
    )

    return manifest
