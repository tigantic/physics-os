"""
Visualization — matplotlib-based field snapshots and convergence plots.

Minimal built-in visualisation shipped with the SDK.  For production-quality
figures, export to VTK/XDMF and open in ParaView/VisIt (see ``export.py``).

Functions
---------
* ``plot_field_1d``  — line plot of a 1-D scalar field.
* ``plot_field_2d``  — filled-contour / imshow of a 2-D scalar field.
* ``plot_convergence`` — error-vs-resolution log-log plot.
* ``plot_observable_history`` — time series of observables from a SolveResult.
* ``plot_spectrum``  — power spectrum from ``fft_field`` output.

All functions return ``(fig, ax)`` so the caller can customise or save.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from ontic.platform.data_model import FieldData, StructuredMesh
from ontic.platform.protocols import SolveResult

logger = logging.getLogger(__name__)

__all__ = [
    "plot_field_1d",
    "plot_field_2d",
    "plot_convergence",
    "plot_observable_history",
    "plot_spectrum",
    "ensure_matplotlib",
]


def ensure_matplotlib() -> bool:
    """
    Check that matplotlib is importable.  Returns True if available.
    All public plotting functions call this and raise ``ImportError`` with
    a helpful message if matplotlib is missing.
    """
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


def _require_mpl() -> Any:
    if not ensure_matplotlib():
        raise ImportError(
            "Visualization requires matplotlib.  Install with: pip install matplotlib"
        )
    import matplotlib.pyplot as plt
    return plt


# ═══════════════════════════════════════════════════════════════════════════════
# 1-D Field Plot
# ═══════════════════════════════════════════════════════════════════════════════


def plot_field_1d(
    field: FieldData,
    *,
    exact: Optional[Tensor] = None,
    title: Optional[str] = None,
    xlabel: str = "x",
    ylabel: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (8.0, 4.5),
) -> Any:
    """
    Plot a 1-D scalar field.

    Parameters
    ----------
    field : scalar FieldData on a 1-D StructuredMesh.
    exact : optional exact-solution tensor for overlay.
    title : plot title (default: field name).
    save_path : if set, save the figure to this path.

    Returns
    -------
    (fig, ax)
    """
    plt = _require_mpl()
    mesh = field.mesh
    if not isinstance(mesh, StructuredMesh) or mesh.ndim != 1:
        raise TypeError("plot_field_1d requires a 1-D StructuredMesh")

    x = mesh.cell_centers().squeeze(-1).detach().cpu().numpy()
    y = field.data.detach().cpu().to(torch.float64).numpy()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(x, y, "-", linewidth=1.5, label="numerical")
    if exact is not None:
        ye = exact.detach().cpu().to(torch.float64).numpy()
        ax.plot(x, ye, "--", linewidth=1.0, label="exact", alpha=0.8)
        ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or field.name)
    ax.set_title(title or field.name)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), dpi=150, bbox_inches="tight")
        logger.info("Saved plot: %s", p)

    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════════
# 2-D Field Plot
# ═══════════════════════════════════════════════════════════════════════════════


def plot_field_2d(
    field: FieldData,
    *,
    title: Optional[str] = None,
    cmap: str = "viridis",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (8.0, 6.0),
    colorbar: bool = True,
) -> Any:
    """
    Plot a 2-D scalar field as a filled contour / imshow.

    Parameters
    ----------
    field : scalar FieldData on a 2-D StructuredMesh.
    cmap : matplotlib colormap name.
    save_path : if set, save the figure to this path.

    Returns
    -------
    (fig, ax)
    """
    plt = _require_mpl()
    mesh = field.mesh
    if not isinstance(mesh, StructuredMesh) or mesh.ndim != 2:
        raise TypeError("plot_field_2d requires a 2-D StructuredMesh")

    nx, ny = mesh.shape
    data = field.data.detach().cpu().to(torch.float64).numpy().reshape(nx, ny)

    x_lo, x_hi = mesh.domain[0]
    y_lo, y_hi = mesh.domain[1]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(
        data.T,
        origin="lower",
        extent=[x_lo, x_hi, y_lo, y_hi],
        cmap=cmap,
        aspect="auto",
    )
    if colorbar:
        fig.colorbar(im, ax=ax, label=field.units if field.units != "1" else "")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or field.name)
    fig.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), dpi=150, bbox_inches="tight")
        logger.info("Saved plot: %s", p)

    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════════
# Convergence Plot
# ═══════════════════════════════════════════════════════════════════════════════


def plot_convergence(
    resolutions: Sequence[int],
    errors: Sequence[float],
    *,
    reference_orders: Optional[Sequence[int]] = None,
    title: str = "Convergence",
    xlabel: str = "N (cells)",
    ylabel: str = "Error",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (7.0, 5.0),
) -> Any:
    """
    Log-log convergence plot with optional reference slopes.

    Parameters
    ----------
    resolutions : list of grid resolutions (number of cells).
    errors : corresponding error measurements.
    reference_orders : optional list of orders (e.g. [1, 2, 4]) to draw
                       reference slope lines.
    save_path : if set, save the figure.

    Returns
    -------
    (fig, ax)
    """
    plt = _require_mpl()
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    res = np.array(resolutions, dtype=np.float64)
    err = np.array(errors, dtype=np.float64)

    ax.loglog(res, err, "o-", linewidth=1.5, markersize=5, label="measured")

    if reference_orders:
        for p in reference_orders:
            # Anchor ref line at first point
            ref = err[0] * (res[0] / res) ** p
            ax.loglog(res, ref, "--", alpha=0.5, label=f"O(h^{p})")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), dpi=150, bbox_inches="tight")
        logger.info("Saved plot: %s", p)

    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════════
# Observable History Plot
# ═══════════════════════════════════════════════════════════════════════════════


def plot_observable_history(
    solve_result: SolveResult,
    *,
    observables: Optional[Sequence[str]] = None,
    title: str = "Observable History",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (8.0, 4.5),
) -> Any:
    """
    Plot time series of observables from a ``SolveResult``.

    Parameters
    ----------
    solve_result : SolveResult with an ``observable_history`` dict.
    observables : which observables to plot (default: all).
    save_path : if set, save the figure.

    Returns
    -------
    (fig, ax)
    """
    plt = _require_mpl()
    hist = solve_result.observable_history
    if not hist:
        raise ValueError("SolveResult has no observable_history")

    names = observables or list(hist.keys())
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for name in names:
        if name not in hist:
            continue
        vals = [v.item() if isinstance(v, Tensor) and v.ndim == 0 else float(v)
                for v in hist[name]]
        steps = list(range(1, len(vals) + 1))
        ax.plot(steps, vals, "-", linewidth=1.2, label=name)

    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), dpi=150, bbox_inches="tight")
        logger.info("Saved plot: %s", p)

    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════════
# Power Spectrum Plot
# ═══════════════════════════════════════════════════════════════════════════════


def plot_spectrum(
    frequencies: Tensor,
    power: Tensor,
    *,
    title: str = "Power Spectrum",
    xlabel: str = "Frequency",
    ylabel: str = "Power",
    log_scale: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (7.0, 4.5),
) -> Any:
    """
    Plot a power spectrum (output of ``fft_field``).

    Parameters
    ----------
    frequencies : 1-D wavenumber tensor.
    power : 1-D power tensor (same length).
    log_scale : use log-log axes.
    save_path : if set, save the figure.

    Returns
    -------
    (fig, ax)
    """
    plt = _require_mpl()
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    f = frequencies.detach().cpu().numpy()
    p = power.detach().cpu().numpy()

    if f.ndim > 1:
        # Multi-D spectrum: flatten and use magnitude of frequency vector
        f_mag = np.linalg.norm(f, axis=-1)
        order = np.argsort(f_mag)
        f = f_mag[order]
        p = p.ravel()[order]

    if log_scale:
        mask = f > 0
        ax.loglog(f[mask], p[mask], "-", linewidth=1.0)
    else:
        ax.plot(f, p, "-", linewidth=1.0)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    if save_path:
        pp = Path(save_path)
        pp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(pp), dpi=150, bbox_inches="tight")
        logger.info("Saved plot: %s", pp)

    return fig, ax
