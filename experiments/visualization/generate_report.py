#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════
# HyperTensor — Publication-Quality Report Generator
# ═══════════════════════════════════════════════════════════════════════════
#
# Reads results/industrial_qtt_gpu_simulation_results.json and generates
# a multi-page PDF report with publication-quality figures for all four
# simulation campaigns, plus individual PNG exports.
#
# Usage:
#   python visualization/generate_report.py [--output-dir DIR] [--dpi DPI]
#   python visualization/generate_report.py --png-only   # skip PDF, PNGs only
#
# Output:
#   visualization/output/report/hypertensor_report.pdf
#   visualization/output/report/fig_*.png
#
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

# ═══════════════════════════════════════════════════════════════════════════
# Style Configuration
# ═══════════════════════════════════════════════════════════════════════════

# Dark theme matching the Blender volumetric aesthetic
BACKGROUND = "#0A0E14"
SURFACE = "#111822"
TEXT_PRIMARY = "#E6EDF3"
TEXT_SECONDARY = "#8B949E"
GRID_COLOR = "#1E2A3A"
ACCENT_CYAN = "#58D5E3"
ACCENT_ORANGE = "#FF8C42"
ACCENT_MAGENTA = "#E45BFF"
ACCENT_GREEN = "#3DDC84"
ACCENT_RED = "#FF4C4C"
ACCENT_GOLD = "#FFD700"
ACCENT_BLUE = "#4A90D9"

PALETTE = [ACCENT_CYAN, ACCENT_ORANGE, ACCENT_MAGENTA, ACCENT_GREEN,
           ACCENT_RED, ACCENT_GOLD, ACCENT_BLUE]

FONT_SIZES = {
    "title": 18,
    "subtitle": 13,
    "axis_label": 12,
    "tick": 10,
    "legend": 10,
    "annotation": 9,
    "watermark": 8,
}


def apply_style() -> None:
    """Apply dark publication theme to matplotlib."""
    mpl.rcParams.update({
        "figure.facecolor": BACKGROUND,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_PRIMARY,
        "axes.grid": True,
        "axes.grid.which": "both",
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.6,
        "grid.linewidth": 0.5,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": FONT_SIZES["tick"],
        "ytick.labelsize": FONT_SIZES["tick"],
        "text.color": TEXT_PRIMARY,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "legend.facecolor": SURFACE,
        "legend.edgecolor": GRID_COLOR,
        "legend.fontsize": FONT_SIZES["legend"],
        "legend.framealpha": 0.9,
        "figure.dpi": 150,
        "savefig.facecolor": BACKGROUND,
        "savefig.edgecolor": "none",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "lines.linewidth": 2.0,
        "lines.antialiased": True,
    })


def watermark(ax: plt.Axes) -> None:
    """Add HyperTensor watermark to bottom-right of axes."""
    ax.text(0.99, 0.01, "HyperTensor-VM",
            transform=ax.transAxes, fontsize=FONT_SIZES["watermark"],
            color=TEXT_SECONDARY, alpha=0.4,
            ha="right", va="bottom", fontstyle="italic")


def styled_title(ax: plt.Axes, title: str, subtitle: str = "") -> None:
    """Set styled title with optional subtitle."""
    ax.set_title(title, fontsize=FONT_SIZES["title"],
                 fontweight="bold", color=TEXT_PRIMARY, pad=12)
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
                fontsize=FONT_SIZES["subtitle"], color=TEXT_SECONDARY,
                ha="center", va="bottom")


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_results(path: str | Path) -> dict[str, Any]:
    """Load and validate the simulation results JSON."""
    path = Path(path)
    if not path.exists():
        print(f"ERROR: Results file not found: {path}", file=sys.stderr)
        print("  Run the simulation first:", file=sys.stderr)
        print("  python scripts/industrial_qtt_gpu_simulation.py", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    required = ["metadata", "campaign_i", "campaign_ii",
                "campaign_iii", "campaign_iv"]
    missing = [k for k in required if k not in data]
    if missing:
        print(f"ERROR: Missing keys in results JSON: {missing}", file=sys.stderr)
        sys.exit(1)

    return data


# ═══════════════════════════════════════════════════════════════════════════
# Figure Generators — Campaign I: NS3D DNS
# ═══════════════════════════════════════════════════════════════════════════

def fig_ke_decay(c1: dict[str, Any]) -> plt.Figure:
    """Kinetic energy decay plot for Taylor-Green vortex DNS."""
    ke = np.array(c1["ke_history"])
    n_steps = c1["n_steps"]
    t = np.linspace(0, n_steps, len(ke))

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Normalize to initial KE
    ke_norm = ke / ke[0] if ke[0] != 0 else ke

    ax.plot(t, ke_norm, color=ACCENT_CYAN, linewidth=2.5,
            marker="o", markersize=5, markerfacecolor=ACCENT_CYAN,
            markeredgecolor="white", markeredgewidth=0.8,
            label="QTT DNS (rank-adaptive)", zorder=5)

    # Analytical reference: exponential decay envelope
    decay_rate = -np.log(ke_norm[-1] + 1e-15) / t[-1]
    t_ref = np.linspace(0, n_steps, 200)
    ke_ref = np.exp(-decay_rate * t_ref)
    ax.plot(t_ref, ke_ref, "--", color=TEXT_SECONDARY, linewidth=1.0,
            alpha=0.6, label=f"Exponential fit (λ={decay_rate:.4f})")

    ax.fill_between(t, ke_norm, alpha=0.08, color=ACCENT_CYAN)

    ax.set_xlabel("Time Step", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel("KE / KE₀ (normalized)", fontsize=FONT_SIZES["axis_label"])
    styled_title(ax, "Kinetic Energy Decay — Taylor-Green Vortex",
                 f"{c1['grid_resolution']} DNS  |  "
                 f"Relative Error: {c1['ke_relative_error']:.4e}  |  "
                 f"Compression: {c1['compression_ratio']:.0f}×")
    ax.legend(loc="upper right")
    ax.set_xlim(0, n_steps)
    ax.set_ylim(bottom=0)
    watermark(ax)
    fig.tight_layout()
    return fig


def fig_enstrophy(c1: dict[str, Any]) -> plt.Figure:
    """Enstrophy evolution showing vortex stretching and dissipation."""
    ens = np.array(c1["enstrophy_history"])
    n_steps = c1["n_steps"]
    t = np.linspace(0, n_steps, len(ens))

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(t, ens, color=ACCENT_ORANGE, linewidth=2.5,
            marker="s", markersize=5, markerfacecolor=ACCENT_ORANGE,
            markeredgecolor="white", markeredgewidth=0.8,
            label="Enstrophy", zorder=5)
    ax.fill_between(t, ens, alpha=0.08, color=ACCENT_ORANGE)

    # Mark peak
    peak_idx = np.argmax(ens)
    ax.annotate(f"Peak: {ens[peak_idx]:.2f}\nt = {t[peak_idx]:.0f}",
                xy=(t[peak_idx], ens[peak_idx]),
                xytext=(t[peak_idx] + n_steps * 0.08, ens[peak_idx] * 0.85),
                fontsize=FONT_SIZES["annotation"],
                color=ACCENT_ORANGE,
                arrowprops=dict(arrowstyle="->", color=ACCENT_ORANGE,
                                lw=1.5, connectionstyle="arc3,rad=-0.2"),
                bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE,
                          ec=ACCENT_ORANGE, alpha=0.9))

    ax.set_xlabel("Time Step", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel("Enstrophy", fontsize=FONT_SIZES["axis_label"])
    styled_title(ax, "Enstrophy Evolution — Vortex Stretching & Dissipation",
                 f"{c1['grid_resolution']} DNS")
    ax.legend(loc="upper right")
    ax.set_xlim(0, n_steps)
    watermark(ax)
    fig.tight_layout()
    return fig


def fig_rank_history(c1: dict[str, Any]) -> plt.Figure:
    """QTT rank adaptation over simulation steps."""
    ranks = np.array(c1["rank_history"])
    n_steps = c1["n_steps"]
    t = np.linspace(0, n_steps, len(ranks))

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.step(t, ranks, where="mid", color=ACCENT_GREEN, linewidth=2.5,
            label="QTT Rank", zorder=5)
    ax.fill_between(t, ranks, step="mid", alpha=0.10, color=ACCENT_GREEN)

    # Horizontal lines for min/max
    ax.axhline(y=ranks.min(), color=ACCENT_GREEN, alpha=0.3,
               linestyle=":", linewidth=1)
    ax.axhline(y=ranks.max(), color=ACCENT_GREEN, alpha=0.3,
               linestyle=":", linewidth=1)

    ax.set_xlabel("Time Step", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel("QTT Rank", fontsize=FONT_SIZES["axis_label"])
    styled_title(ax, "Rank-Adaptive QTT Compression",
                 f"Rank range: [{int(ranks.min())}–{int(ranks.max())}]  |  "
                 f"Compression: {c1['compression_ratio']:.0f}×")
    ax.legend(loc="upper right")
    ax.set_xlim(0, n_steps)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    watermark(ax)
    fig.tight_layout()
    return fig


def fig_step_timing(c1: dict[str, Any]) -> plt.Figure:
    """Per-step wall-clock timing showing GPU kernel performance."""
    times = np.array(c1["step_times_ms"])
    steps = np.arange(1, len(times) + 1)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.bar(steps, times, color=ACCENT_BLUE, alpha=0.8, edgecolor="none",
           width=0.8, label="Step time")

    # Running mean
    window = min(5, len(times))
    if window > 1:
        running_mean = np.convolve(times, np.ones(window) / window, mode="valid")
        ax.plot(steps[window - 1:], running_mean, color=ACCENT_GOLD,
                linewidth=2, label=f"Running mean ({window}-step)")

    # Stats annotation
    med = np.median(times)
    ax.axhline(y=med, color=ACCENT_CYAN, linestyle="--", linewidth=1.2,
               alpha=0.7, label=f"Median: {med:.1f} ms")

    ax.set_xlabel("Step Number", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel("Time (ms)", fontsize=FONT_SIZES["axis_label"])
    styled_title(ax, "Per-Step GPU Wall-Clock Timing",
                 f"{len(times)} steps  |  "
                 f"Median: {med:.1f} ms  |  "
                 f"Total: {c1['wall_time_sec']:.1f} s")
    ax.legend(loc="upper right")
    ax.set_xlim(0.5, len(times) + 0.5)
    watermark(ax)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Figure Generators — Campaign II: Compression Scaling
# ═══════════════════════════════════════════════════════════════════════════

def _parse_grid_n(grid_str: str) -> int:
    """Extract N from grid strings like '64 cubed', '1024 cubed'."""
    parts = grid_str.strip().split()
    return int(parts[0])


def fig_compression_scaling(c2: list[dict[str, Any]]) -> plt.Figure:
    """Log-log compression ratio vs grid points by function type."""
    # Group by function type
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in c2:
        groups[entry["function_type"]].append(entry)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "sod_shock": ACCENT_CYAN,
        "turbulent_8mode": ACCENT_ORANGE,
        "boundary_layer": ACCENT_MAGENTA,
    }
    markers = {"sod_shock": "o", "turbulent_8mode": "s", "boundary_layer": "D"}
    labels = {
        "sod_shock": "Sod Shock Tube",
        "turbulent_8mode": "Turbulent 8-Mode",
        "boundary_layer": "Boundary Layer",
    }

    for ftype, entries in sorted(groups.items()):
        entries_sorted = sorted(entries, key=lambda e: e["n_grid_points"])
        n_pts = np.array([e["n_grid_points"] for e in entries_sorted])
        ratios = np.array([e["compression_ratio"] for e in entries_sorted])
        n_grid = [_parse_grid_n(e["grid_resolution"]) for e in entries_sorted]

        color = colors.get(ftype, TEXT_SECONDARY)
        marker = markers.get(ftype, "o")
        label = labels.get(ftype, ftype)

        ax.loglog(n_pts, ratios, color=color, marker=marker,
                  markersize=7, markeredgecolor="white", markeredgewidth=0.8,
                  linewidth=2, label=label, zorder=5)

        # Annotate last point
        ax.annotate(f"{n_grid[-1]}³\n{ratios[-1]:.0f}×",
                    xy=(n_pts[-1], ratios[-1]),
                    xytext=(12, 0), textcoords="offset points",
                    fontsize=FONT_SIZES["annotation"],
                    color=color, ha="left", va="center")

    ax.set_xlabel("Grid Points (N³)", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel("Compression Ratio", fontsize=FONT_SIZES["axis_label"])
    styled_title(ax, "QTT Compression Scaling — 3D Flow Fields",
                 f"3 function types × 7 grid resolutions (64³ → 4096³)  |  "
                 f"Tolerance: 10⁻⁶")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, which="minor", alpha=0.2)
    watermark(ax)
    fig.tight_layout()
    return fig


def fig_compression_bars(c2: list[dict[str, Any]]) -> plt.Figure:
    """Grouped bar chart of compression ratios at each grid resolution."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in c2:
        groups[entry["function_type"]].append(entry)

    ftypes = sorted(groups.keys())
    # Get unique grid sizes in order
    all_grids = sorted(set(_parse_grid_n(e["grid_resolution"]) for e in c2))

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(all_grids))
    width = 0.25
    offsets = np.linspace(-width, width, len(ftypes))

    colors_list = [ACCENT_CYAN, ACCENT_ORANGE, ACCENT_MAGENTA]
    labels_map = {
        "boundary_layer": "Boundary Layer",
        "sod_shock": "Sod Shock",
        "turbulent_8mode": "Turbulent 8-Mode",
    }

    for i, ftype in enumerate(ftypes):
        entries = sorted(groups[ftype], key=lambda e: _parse_grid_n(e["grid_resolution"]))
        ratios = [e["compression_ratio"] for e in entries]

        bars = ax.bar(x + offsets[i], ratios, width=width * 0.9,
                      color=colors_list[i % len(colors_list)],
                      edgecolor="none", alpha=0.9,
                      label=labels_map.get(ftype, ftype), zorder=5)

        # Value labels on top of tall bars
        for bar, ratio in zip(bars, ratios):
            if ratio > 1000:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{ratio / 1e6:.1f}M" if ratio >= 1e6 else f"{ratio / 1e3:.0f}K",
                        ha="center", va="bottom",
                        fontsize=FONT_SIZES["annotation"] - 1,
                        color=colors_list[i % len(colors_list)],
                        rotation=45)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}³" for g in all_grids])
    ax.set_xlabel("Grid Resolution", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel("Compression Ratio (log scale)", fontsize=FONT_SIZES["axis_label"])
    styled_title(ax, "QTT Compression by Grid Resolution & Flow Type",
                 f"Rank-adaptive TCI  |  Tolerance: 10⁻⁶")
    ax.legend(loc="upper left")
    watermark(ax)
    fig.tight_layout()
    return fig


def fig_compression_rank(c2: list[dict[str, Any]]) -> plt.Figure:
    """Rank vs compression ratio scatter showing rank efficiency."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in c2:
        groups[entry["function_type"]].append(entry)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    colors = {
        "sod_shock": ACCENT_CYAN,
        "turbulent_8mode": ACCENT_ORANGE,
        "boundary_layer": ACCENT_MAGENTA,
    }
    labels = {
        "sod_shock": "Sod Shock",
        "turbulent_8mode": "Turbulent 8-Mode",
        "boundary_layer": "Boundary Layer",
    }

    for ftype, entries in sorted(groups.items()):
        ranks = [e["max_rank_actual"] for e in entries]
        ratios = [e["compression_ratio"] for e in entries]
        n_pts = [e["n_grid_points"] for e in entries]
        color = colors.get(ftype, TEXT_SECONDARY)

        sc = ax.scatter(ranks, ratios, c=color, s=np.log10(n_pts) * 25,
                        alpha=0.85, edgecolors="white", linewidths=0.5,
                        label=labels.get(ftype, ftype), zorder=5)

    ax.set_yscale("log")
    ax.set_xlabel("QTT Rank", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel("Compression Ratio (log)", fontsize=FONT_SIZES["axis_label"])
    styled_title(ax, "Rank Efficiency — Lower Rank, Higher Compression",
                 "Bubble size ∝ log(grid points)")
    ax.legend(loc="upper right")
    watermark(ax)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Figure Generators — Campaign III: Kernel Benchmarks
# ═══════════════════════════════════════════════════════════════════════════

def _normalize_kernel_name(name: str) -> str:
    """Strip backend prefix for matching."""
    prefixes = ["triton_", "legacy_", "cuda_native_"]
    for p in prefixes:
        if name.startswith(p):
            return name[len(p):]
    # Handle special cases
    if "adaptive_truncate" in name:
        return "adaptive_truncate"
    if "morton_encode" in name:
        return "morton_encode"
    return name


def fig_kernel_latency(c3: list[dict[str, Any]]) -> plt.Figure:
    """Grouped bar chart comparing kernel latency across backends."""
    # Group by normalized kernel name
    kernel_groups: dict[str, dict[str, dict]] = defaultdict(dict)
    for entry in c3:
        norm_name = _normalize_kernel_name(entry["kernel_name"])
        kernel_groups[norm_name][entry["backend"]] = entry

    # Filter to kernels with multiple backends for comparison
    # (also include single-backend ones)
    kernel_names = sorted(kernel_groups.keys())

    fig, ax = plt.subplots(figsize=(12, 6))

    backends = ["triton_native", "legacy_python", "cuda_native", "triton_jit"]
    backend_colors = {
        "triton_native": ACCENT_CYAN,
        "legacy_python": ACCENT_ORANGE,
        "cuda_native": ACCENT_GREEN,
        "triton_jit": ACCENT_MAGENTA,
    }
    backend_labels = {
        "triton_native": "Triton (fused)",
        "legacy_python": "Legacy Python",
        "cuda_native": "CUDA Native",
        "triton_jit": "Triton JIT",
    }

    x = np.arange(len(kernel_names))
    n_backends = len(backends)
    width = 0.8 / n_backends

    plotted_backends = set()

    for i, backend in enumerate(backends):
        latencies = []
        for kname in kernel_names:
            entry = kernel_groups[kname].get(backend)
            latencies.append(entry["median_ms"] if entry else 0)

        if any(v > 0 for v in latencies):
            non_zero = [v if v > 0 else np.nan for v in latencies]
            bars = ax.bar(x + (i - n_backends / 2 + 0.5) * width,
                          non_zero, width=width * 0.9,
                          color=backend_colors[backend], alpha=0.9,
                          edgecolor="none",
                          label=backend_labels[backend], zorder=5)
            plotted_backends.add(backend)

            # Value labels
            for bar, val in zip(bars, non_zero):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height(), f"{val:.2f}",
                            ha="center", va="bottom",
                            fontsize=FONT_SIZES["annotation"] - 1,
                            color=backend_colors[backend])

    ax.set_yscale("log")
    ax.set_xticks(x)

    # Pretty labels
    pretty = {
        "qtt_add": "QTT Add",
        "qtt_scale": "QTT Scale",
        "qtt_inner": "QTT Inner",
        "qtt_hadamard": "Hadamard",
        "mpo_apply": "MPO Apply",
        "adaptive_truncate": "Adaptive\nTruncate",
        "add": "Add",
        "inner": "Inner",
        "hadamard": "Hadamard",
        "morton_encode": "Morton\nEncode",
    }
    ax.set_xticklabels([pretty.get(k, k) for k in kernel_names],
                       rotation=30, ha="right")

    ax.set_xlabel("Kernel", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel("Median Latency (ms, log scale)", fontsize=FONT_SIZES["axis_label"])
    styled_title(ax, "GPU Kernel Benchmark — Latency Comparison",
                 f"{len(c3)} benchmarks across {len(plotted_backends)} backends")
    ax.legend(loc="upper right")
    watermark(ax)
    fig.tight_layout()
    return fig


def fig_kernel_throughput(c3: list[dict[str, Any]]) -> plt.Figure:
    """Horizontal bar chart of throughput (ops/sec) for all kernels."""
    # Sort by throughput descending
    entries = sorted(c3, key=lambda e: e["throughput_ops_sec"], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    backend_colors = {
        "triton_native": ACCENT_CYAN,
        "legacy_python": ACCENT_ORANGE,
        "cuda_native": ACCENT_GREEN,
        "triton_jit": ACCENT_MAGENTA,
    }

    y = np.arange(len(entries))
    throughputs = [e["throughput_ops_sec"] for e in entries]
    colors = [backend_colors.get(e["backend"], TEXT_SECONDARY) for e in entries]
    labels = [f"{e['kernel_name']} ({e['backend'].split('_')[0]})" for e in entries]

    bars = ax.barh(y, throughputs, color=colors, alpha=0.9,
                   edgecolor="none", height=0.7, zorder=5)

    # Value labels
    for bar, tp in zip(bars, throughputs):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f"  {tp:.0f} ops/s",
                ha="left", va="center",
                fontsize=FONT_SIZES["annotation"],
                color=TEXT_SECONDARY)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FONT_SIZES["tick"])
    ax.set_xscale("log")
    ax.set_xlabel("Throughput (ops/sec, log scale)", fontsize=FONT_SIZES["axis_label"])
    ax.invert_yaxis()
    styled_title(ax, "GPU Kernel Throughput Ranking")
    watermark(ax)
    fig.tight_layout()
    return fig


def fig_kernel_speedup(c3: list[dict[str, Any]]) -> plt.Figure:
    """Speedup of Triton kernels vs Legacy Python baseline."""
    # Pair by normalized kernel name
    kernel_groups: dict[str, dict[str, dict]] = defaultdict(dict)
    for entry in c3:
        norm_name = _normalize_kernel_name(entry["kernel_name"])
        kernel_groups[norm_name][entry["backend"]] = entry

    # Only include kernels present in both triton_native and legacy_python
    speedups = []
    for kname, backends in sorted(kernel_groups.items()):
        if "triton_native" in backends and "legacy_python" in backends:
            triton_ms = backends["triton_native"]["median_ms"]
            legacy_ms = backends["legacy_python"]["median_ms"]
            speedup = legacy_ms / triton_ms if triton_ms > 0 else 0
            speedups.append((kname, speedup, triton_ms, legacy_ms))

    if not speedups:
        # Nothing to compare — return empty figure with message
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No matching Triton/Legacy kernel pairs found",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=FONT_SIZES["subtitle"], color=TEXT_SECONDARY)
        return fig

    # Sort by speedup
    speedups.sort(key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    names = [s[0] for s in speedups]
    vals = [s[1] for s in speedups]
    x = np.arange(len(names))

    bar_colors = [ACCENT_GREEN if v >= 1 else ACCENT_RED for v in vals]
    bars = ax.bar(x, vals, color=bar_colors, alpha=0.9, edgecolor="none",
                  width=0.6, zorder=5)

    # Baseline line
    ax.axhline(y=1.0, color=TEXT_SECONDARY, linestyle="--", linewidth=1,
               alpha=0.5, label="Parity (1×)")

    # Annotations
    for bar, (name, spd, t_ms, l_ms) in zip(bars, speedups):
        label = f"{spd:.1f}×"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5, label,
                ha="center", va="bottom",
                fontsize=FONT_SIZES["annotation"],
                color=ACCENT_GREEN if spd >= 1 else ACCENT_RED,
                fontweight="bold")
        # Show raw timings below
        ax.text(bar.get_x() + bar.get_width() / 2, -2.5,
                f"{t_ms:.2f}ms\nvs\n{l_ms:.2f}ms",
                ha="center", va="top",
                fontsize=FONT_SIZES["annotation"] - 1,
                color=TEXT_SECONDARY)

    pretty = {
        "qtt_add": "QTT Add",
        "qtt_scale": "QTT Scale",
        "qtt_inner": "QTT Inner",
        "qtt_hadamard": "Hadamard",
    }
    ax.set_xticks(x)
    ax.set_xticklabels([pretty.get(n, n) for n in names])
    ax.set_ylabel("Speedup (Triton / Legacy)", fontsize=FONT_SIZES["axis_label"])
    styled_title(ax, "Triton Fused Kernels vs Legacy Python",
                 "Speedup measured at median latency")
    ax.legend(loc="upper right")
    watermark(ax)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Figure Generators — Campaign IV: Combustion DNS
# ═══════════════════════════════════════════════════════════════════════════

def fig_flame_profile(c4: dict[str, Any]) -> plt.Figure:
    """Temperature profile through the flame front."""
    temp = np.array(c4["temperature_profile"])
    x = np.linspace(0, 1, len(temp))

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Temperature curve with gradient fill
    ax.plot(x, temp, color=ACCENT_ORANGE, linewidth=2.5,
            label="Temperature", zorder=5)

    # Color gradient fill under curve using pcolormesh
    t_norm = (temp - temp.min()) / (temp.max() - temp.min() + 1e-12)
    for i in range(len(x) - 1):
        avg_t = (t_norm[i] + t_norm[i + 1]) / 2
        # Interpolate from blue (cold) to orange to red (hot)
        r = min(1, avg_t * 2.5)
        g = max(0, 0.5 - abs(avg_t - 0.5)) * 2
        b = max(0, 1 - avg_t * 3)
        ax.fill_between(x[i:i + 2], temp[i:i + 2], alpha=0.15,
                        color=(r, g, b))

    # Mark key temperatures
    ax.axhline(y=c4["max_temperature"], color=ACCENT_RED, linestyle="--",
               linewidth=1, alpha=0.5)
    ax.annotate(f"T_max = {c4['max_temperature']:.0f} K",
                xy=(x[np.argmax(temp)], c4["max_temperature"]),
                xytext=(0.15, c4["max_temperature"] - 200),
                fontsize=FONT_SIZES["annotation"],
                color=ACCENT_RED,
                arrowprops=dict(arrowstyle="->", color=ACCENT_RED, lw=1),
                bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE,
                          ec=ACCENT_RED, alpha=0.9))

    ax.axhline(y=c4["min_temperature"], color=ACCENT_BLUE, linestyle="--",
               linewidth=1, alpha=0.5)

    ax.set_xlabel("Normalized Position (x/L)", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel("Temperature (K)", fontsize=FONT_SIZES["axis_label"])
    styled_title(ax, "Flame Temperature Profile — Combustion DNS",
                 f"{c4['mechanism']}  |  S_L = {c4['flame_speed_estimate']:.2f} m/s  |  "
                 f"{c4['n_species']} species, {c4['n_reactions']} reactions")
    ax.legend(loc="center left")
    ax.set_xlim(0, 1)
    watermark(ax)
    fig.tight_layout()
    return fig


def fig_flame_summary(c4: dict[str, Any]) -> plt.Figure:
    """Summary metrics card for combustion campaign."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    metrics = [
        ("Mechanism", c4["mechanism"]),
        ("Species", str(c4["n_species"])),
        ("Reactions", str(c4["n_reactions"])),
        ("Grid (N_x)", str(c4["nx"])),
        ("Time Steps", str(c4["n_steps"])),
        ("T_max", f"{c4['max_temperature']:.1f} K"),
        ("T_min", f"{c4['min_temperature']:.1f} K"),
        ("Flame Speed (S_L)", f"{c4['flame_speed_estimate']:.3f} m/s"),
        ("Max Heat Release", f"{c4['max_heat_release']:.1f} W/m³"),
        ("Wall Time", f"{c4['wall_time_sec']:.2f} s"),
    ]

    # Layout: 2 columns
    n_rows = (len(metrics) + 1) // 2
    for i, (label, value) in enumerate(metrics):
        col = i // n_rows
        row = i % n_rows
        x_pos = 0.08 + col * 0.48
        y_pos = 0.88 - row * (0.8 / n_rows)

        ax.text(x_pos, y_pos, label, transform=ax.transAxes,
                fontsize=FONT_SIZES["axis_label"], color=TEXT_SECONDARY,
                fontweight="bold", va="center")
        ax.text(x_pos + 0.28, y_pos, value, transform=ax.transAxes,
                fontsize=FONT_SIZES["axis_label"], color=ACCENT_CYAN,
                va="center", fontweight="normal")

    styled_title(ax, "Campaign IV — Combustion DNS Summary")
    watermark(ax)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Summary / Title Page
# ═══════════════════════════════════════════════════════════════════════════

def fig_title_page(data: dict[str, Any]) -> plt.Figure:
    """Title page for the report."""
    meta = data["metadata"]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")

    # Title
    ax.text(0.5, 0.78, "HyperTensor-VM", transform=ax.transAxes,
            fontsize=32, color=ACCENT_CYAN, fontweight="bold",
            ha="center", va="center", fontstyle="italic")
    ax.text(0.5, 0.68, "Industrial QTT/GPU Simulation Report",
            transform=ax.transAxes, fontsize=20, color=TEXT_PRIMARY,
            ha="center", va="center")

    # Divider line
    ax.plot([0.2, 0.8], [0.62, 0.62], transform=ax.transAxes,
            color=ACCENT_CYAN, linewidth=2, alpha=0.5)

    # Campaign summary
    campaign_lines = [
        ("Campaign I", "Navier-Stokes 3D DNS (1024³ Taylor-Green Vortex)"),
        ("Campaign II", "QTT Compression Scaling (64³ → 4096³, 3 flow types)"),
        ("Campaign III", "GPU Kernel Benchmarks (Triton / CUDA / Legacy)"),
        ("Campaign IV", "Reactive Combustion DNS (H₂-Air 6-species)"),
    ]

    y_start = 0.52
    for i, (label, desc) in enumerate(campaign_lines):
        y = y_start - i * 0.08
        ax.text(0.15, y, label, transform=ax.transAxes,
                fontsize=FONT_SIZES["axis_label"], color=ACCENT_GOLD,
                fontweight="bold", va="center")
        ax.text(0.35, y, desc, transform=ax.transAxes,
                fontsize=FONT_SIZES["axis_label"], color=TEXT_PRIMARY,
                va="center")

    # Hardware info
    hw = meta.get("hardware", {})
    hw_lines = []
    if isinstance(hw, dict):
        if "gpu" in hw:
            hw_lines.append(f"GPU: {hw['gpu']}")
        if "torch" in hw:
            hw_lines.append(f"PyTorch {hw['torch']}")
        if "cuda" in hw:
            hw_lines.append(f"CUDA {hw['cuda']}")
    elif isinstance(hw, str):
        hw_lines.append(hw)

    if hw_lines:
        hw_text = "  |  ".join(hw_lines)
        ax.text(0.5, 0.12, hw_text, transform=ax.transAxes,
                fontsize=FONT_SIZES["annotation"], color=TEXT_SECONDARY,
                ha="center", va="center")

    ax.text(0.5, 0.04, "Generated by HyperTensor-VM Visualization Pipeline",
            transform=ax.transAxes, fontsize=FONT_SIZES["watermark"],
            color=TEXT_SECONDARY, ha="center", va="center", alpha=0.5)

    fig.tight_layout()
    return fig


def fig_summary_dashboard(data: dict[str, Any]) -> plt.Figure:
    """Single-page dashboard summarising key results from all campaigns."""
    c1 = data["campaign_i"]
    c2 = data["campaign_ii"]
    c3 = data["campaign_iii"]
    c4 = data["campaign_iv"]

    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # ── Panel 1: KE Decay (C1) ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ke = np.array(c1["ke_history"])
    t = np.linspace(0, c1["n_steps"], len(ke))
    ke_norm = ke / ke[0] if ke[0] != 0 else ke
    ax1.plot(t, ke_norm, color=ACCENT_CYAN, linewidth=1.8,
             marker="o", markersize=3)
    ax1.set_xlabel("Step", fontsize=9)
    ax1.set_ylabel("KE/KE₀", fontsize=9)
    ax1.set_title("KE Decay (TG Vortex)", fontsize=11,
                  fontweight="bold", color=TEXT_PRIMARY)
    ax1.text(0.95, 0.05, f"ε = {c1['ke_relative_error']:.2e}",
             transform=ax1.transAxes, fontsize=8, color=ACCENT_CYAN,
             ha="right", va="bottom")

    # ── Panel 2: Compression Scaling (C2) ────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    groups: dict[str, list] = defaultdict(list)
    for entry in c2:
        groups[entry["function_type"]].append(entry)

    c_map = {"sod_shock": ACCENT_CYAN, "turbulent_8mode": ACCENT_ORANGE,
             "boundary_layer": ACCENT_MAGENTA}
    for ftype, entries in sorted(groups.items()):
        entries_s = sorted(entries, key=lambda e: e["n_grid_points"])
        n_pts = [e["n_grid_points"] for e in entries_s]
        ratios = [e["compression_ratio"] for e in entries_s]
        ax2.loglog(n_pts, ratios, color=c_map.get(ftype, TEXT_SECONDARY),
                   marker="o", markersize=3, linewidth=1.5)
    ax2.set_xlabel("Grid Points", fontsize=9)
    ax2.set_ylabel("Compression", fontsize=9)
    ax2.set_title("QTT Compression", fontsize=11,
                  fontweight="bold", color=TEXT_PRIMARY)

    # ── Panel 3: Temperature Profile (C4) ────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    temp = np.array(c4["temperature_profile"])
    x_pos = np.linspace(0, 1, len(temp))
    ax3.plot(x_pos, temp, color=ACCENT_ORANGE, linewidth=1.8)
    ax3.fill_between(x_pos, temp, alpha=0.1, color=ACCENT_ORANGE)
    ax3.set_xlabel("x/L", fontsize=9)
    ax3.set_ylabel("T (K)", fontsize=9)
    ax3.set_title("Flame Profile", fontsize=11,
                  fontweight="bold", color=TEXT_PRIMARY)

    # ── Panel 4: Kernel Latency (C3) — bottom spanning ──────────────
    ax4 = fig.add_subplot(gs[1, :])
    entries_sorted = sorted(c3, key=lambda e: e["median_ms"])
    names = [e["kernel_name"].replace("_", "\n") for e in entries_sorted]
    latencies = [e["median_ms"] for e in entries_sorted]
    backend_colors_map = {
        "triton_native": ACCENT_CYAN,
        "legacy_python": ACCENT_ORANGE,
        "cuda_native": ACCENT_GREEN,
        "triton_jit": ACCENT_MAGENTA,
    }
    bar_colors = [backend_colors_map.get(e["backend"], TEXT_SECONDARY)
                  for e in entries_sorted]

    ax4.barh(range(len(entries_sorted)), latencies,
             color=bar_colors, alpha=0.9, height=0.7)
    ax4.set_yticks(range(len(entries_sorted)))
    ax4.set_yticklabels(names, fontsize=7)
    ax4.set_xscale("log")
    ax4.set_xlabel("Median Latency (ms)", fontsize=9)
    ax4.set_title("GPU Kernel Benchmarks — All Backends", fontsize=11,
                  fontweight="bold", color=TEXT_PRIMARY)
    ax4.invert_yaxis()

    # Add value labels
    for i, (lat, entry) in enumerate(zip(latencies, entries_sorted)):
        ax4.text(lat * 1.15, i, f"{lat:.3f} ms",
                 va="center", fontsize=7, color=TEXT_SECONDARY)

    fig.suptitle("HyperTensor-VM — Simulation Results Summary",
                 fontsize=16, fontweight="bold", color=ACCENT_CYAN, y=0.98)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Report Assembly
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(results_path: str | Path,
                    output_dir: str | Path,
                    dpi: int = 200,
                    png_only: bool = False) -> None:
    """Generate complete report: multi-page PDF + individual PNGs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_results(results_path)
    c1 = data["campaign_i"]
    c2 = data["campaign_ii"]
    c3 = data["campaign_iii"]
    c4 = data["campaign_iv"]

    apply_style()

    # Define all figures in order
    figure_specs = [
        ("title_page", "Title Page", lambda: fig_title_page(data)),
        ("summary_dashboard", "Summary Dashboard", lambda: fig_summary_dashboard(data)),
        ("c1_ke_decay", "KE Decay (Campaign I)", lambda: fig_ke_decay(c1)),
        ("c1_enstrophy", "Enstrophy (Campaign I)", lambda: fig_enstrophy(c1)),
        ("c1_rank", "Rank History (Campaign I)", lambda: fig_rank_history(c1)),
        ("c1_timing", "Step Timing (Campaign I)", lambda: fig_step_timing(c1)),
        ("c2_compression_scaling", "Compression Scaling (Campaign II)",
         lambda: fig_compression_scaling(c2)),
        ("c2_compression_bars", "Compression Bars (Campaign II)",
         lambda: fig_compression_bars(c2)),
        ("c2_rank_scatter", "Rank Efficiency (Campaign II)",
         lambda: fig_compression_rank(c2)),
        ("c3_kernel_latency", "Kernel Latency (Campaign III)",
         lambda: fig_kernel_latency(c3)),
        ("c3_kernel_throughput", "Kernel Throughput (Campaign III)",
         lambda: fig_kernel_throughput(c3)),
        ("c3_speedup", "Triton vs Legacy (Campaign III)",
         lambda: fig_kernel_speedup(c3)),
        ("c4_flame_profile", "Flame Profile (Campaign IV)",
         lambda: fig_flame_profile(c4)),
        ("c4_flame_summary", "Flame Summary (Campaign IV)",
         lambda: fig_flame_summary(c4)),
    ]

    # Generate PNGs
    figures: list[tuple[str, plt.Figure]] = []
    for fig_id, label, fig_fn in figure_specs:
        print(f"  Generating: {label}...", flush=True)
        fig = fig_fn()
        png_path = output_dir / f"fig_{fig_id}.png"
        fig.savefig(png_path, dpi=dpi, facecolor=BACKGROUND)
        print(f"    → {png_path}")
        figures.append((fig_id, fig))

    # Generate PDF
    if not png_only:
        pdf_path = output_dir / "hypertensor_report.pdf"
        print(f"\n  Assembling PDF: {pdf_path}...", flush=True)
        with PdfPages(str(pdf_path)) as pdf:
            for fig_id, fig in figures:
                pdf.savefig(fig, facecolor=BACKGROUND)

            # PDF metadata
            d = pdf.infodict()
            d["Title"] = "HyperTensor-VM Industrial Simulation Report"
            d["Author"] = "HyperTensor-VM Visualization Pipeline"
            d["Subject"] = "QTT/GPU Simulation Results"
            d["Keywords"] = "QTT, GPU, DNS, CFD, Tensor, HyperTensor"

        pdf_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        print(f"    → {pdf_path} ({pdf_size_mb:.1f} MB)")

    # Close all figures
    for _, fig in figures:
        plt.close(fig)

    # Summary
    print()
    print("═══════════════════════════════════════════════════════════════")
    print("  REPORT COMPLETE")
    print(f"  Figures: {len(figures)} generated")
    print(f"  PNGs:    {output_dir}/fig_*.png")
    if not png_only:
        print(f"  PDF:     {output_dir}/hypertensor_report.pdf")
    print("═══════════════════════════════════════════════════════════════")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate HyperTensor simulation report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualization/generate_report.py
  python visualization/generate_report.py --dpi 300
  python visualization/generate_report.py --png-only
  python visualization/generate_report.py --output-dir artifacts/report
        """,
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=None,
        help="Output directory (default: visualization/output/report)",
    )
    parser.add_argument(
        "--results", type=str,
        default=None,
        help="Path to results JSON (default: auto-detected)",
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help="PNG resolution in DPI (default: 200)",
    )
    parser.add_argument(
        "--png-only", action="store_true",
        help="Generate PNGs only, skip PDF",
    )

    args = parser.parse_args()

    # Resolve paths relative to repo root
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    results_path = (
        Path(args.results) if args.results
        else repo_root / "results" / "industrial_qtt_gpu_simulation_results.json"
    )
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else script_dir / "output" / "report"
    )

    print("═══════════════════════════════════════════════════════════════")
    print("  HyperTensor — Report Generator")
    print(f"  Results:  {results_path}")
    print(f"  Output:   {output_dir}")
    print(f"  DPI:      {args.dpi}")
    print(f"  PDF:      {'no' if args.png_only else 'yes'}")
    print("═══════════════════════════════════════════════════════════════")
    print()

    generate_report(results_path, output_dir, args.dpi, args.png_only)


if __name__ == "__main__":
    main()
