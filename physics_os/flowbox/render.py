"""FlowBox MP4 render pipeline.

Generates MP4 video from dense vorticity frame snapshots.

Dependencies:
    - matplotlib (frame rendering)
    - ffmpeg (MP4 encoding, via matplotlib.animation.FFMpegWriter)

Falls back gracefully:
    - No ffmpeg → GIF via PillowWriter
    - No matplotlib → render skipped, metadata reports unavailable
"""

from __future__ import annotations

import io
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RenderResult:
    """MP4 render output."""

    available: bool
    format: str  # "mp4" | "gif" | "unavailable"
    frames: int
    fps: int
    duration_s: float
    data: bytes  # Raw video bytes
    size_bytes: int
    error: str | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        """Public-safe render metadata (no binary data)."""
        return {
            "available": self.available,
            "format": self.format,
            "frames": self.frames,
            "fps": self.fps,
            "duration_s": round(self.duration_s, 2),
            "size_bytes": self.size_bytes,
        }


# ═══════════════════════════════════════════════════════════════════
# Watermark
# ═══════════════════════════════════════════════════════════════════

_WATERMARK_TEXT = "FlowBox Explorer — ontic.io"


# ═══════════════════════════════════════════════════════════════════
# Main render function
# ═══════════════════════════════════════════════════════════════════


def generate_render(
    frames: list[np.ndarray],
    frame_times: list[float],
    *,
    fps: int = 30,
    colormap: str = "RdBu_r",
    watermark: bool = False,
    preset_label: str = "",
    grid: int = 512,
    viscosity: float = 0.01,
) -> RenderResult:
    """Generate an MP4 (or GIF fallback) from vorticity frames.

    Parameters
    ----------
    frames : list[np.ndarray]
        Vorticity fields at snapshot times, each shape (N, N).
    frame_times : list[float]
        Simulation times corresponding to each frame.
    fps : int
        Target frames per second.
    colormap : str
        Matplotlib colormap name.
    watermark : bool
        If True, overlay Explorer-tier watermark text.
    preset_label : str
        Preset name for title annotation.
    grid : int
        Grid resolution for title annotation.
    viscosity : float
        Viscosity for title annotation.

    Returns
    -------
    RenderResult
        Video bytes and metadata.
    """
    if not frames:
        return RenderResult(
            available=False,
            format="unavailable",
            frames=0,
            fps=fps,
            duration_s=0.0,
            data=b"",
            size_bytes=0,
            error="No frames to render.",
        )

    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
    except ImportError:
        return RenderResult(
            available=False,
            format="unavailable",
            frames=len(frames),
            fps=fps,
            duration_s=len(frames) / max(fps, 1),
            data=b"",
            size_bytes=0,
            error="matplotlib not available.",
        )

    n_frames = len(frames)
    duration = n_frames / max(fps, 1)

    # Compute symmetric colorbar limits across all frames
    vmax = max(np.abs(f).max() for f in frames)
    vmin = -vmax

    # ── Build animation ─────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02)

    im = ax.imshow(
        frames[0].T,  # Transpose for (x, y) → (row, col) display
        origin="lower",
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
        extent=[0, 1, 0, 1],
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    title = ax.set_title(
        f"{preset_label}  |  {grid}×{grid}  |  ν={viscosity:.0e}  |  t=0.000",
        fontsize=10,
        fontfamily="monospace",
    )

    if watermark:
        ax.text(
            0.5, 0.5, _WATERMARK_TEXT,
            transform=ax.transAxes,
            fontsize=16,
            color="white",
            alpha=0.35,
            ha="center",
            va="center",
            rotation=30,
            fontweight="bold",
        )

    def _update(frame_idx: int) -> list:
        im.set_data(frames[frame_idx].T)
        t = frame_times[frame_idx] if frame_idx < len(frame_times) else 0.0
        title.set_text(
            f"{preset_label}  |  {grid}×{grid}  |  "
            f"ν={viscosity:.0e}  |  t={t:.4f}"
        )
        return [im, title]

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=n_frames,
        interval=1000 // max(fps, 1),
        blit=True,
    )

    # ── Try MP4 (ffmpeg), fall back to GIF (Pillow) ─────────────
    buf = io.BytesIO()
    fmt = "mp4"

    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(buf, writer=writer, dpi=150)
    except Exception as e_mp4:
        logger.info("ffmpeg unavailable (%s), falling back to GIF", e_mp4)
        fmt = "gif"
        buf = io.BytesIO()
        try:
            writer_gif = animation.PillowWriter(fps=fps)
            anim.save(buf, writer=writer_gif, dpi=100)
        except Exception as e_gif:
            plt.close(fig)
            return RenderResult(
                available=False,
                format="unavailable",
                frames=n_frames,
                fps=fps,
                duration_s=duration,
                data=b"",
                size_bytes=0,
                error=f"Render failed: MP4={e_mp4}, GIF={e_gif}",
            )

    plt.close(fig)
    video_bytes = buf.getvalue()

    return RenderResult(
        available=True,
        format=fmt,
        frames=n_frames,
        fps=fps,
        duration_s=duration,
        data=video_bytes,
        size_bytes=len(video_bytes),
    )
