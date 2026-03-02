"""Hybrid field representation: TT backbone + localized corrections.

The core idea:
    q(x) = q_TT(x) + q_local(x)

Where:
- q_TT is a globally compressed QTT tensor (the backbone)
- q_local is a sparse set of dense correction tiles in feature-active
  regions (shocks, interfaces, boundaries, singularities)

This module provides:
1. HybridField — the summed representation
2. LocalTile — dense correction on a sub-domain
3. FeatureSensor — detects regions needing local corrections
4. TileActivationPolicy — decides which tiles get activated
5. HybridRoundPolicy — truncation policy for hybrid fields

All operations remain deterministic given seed/config per §20.2.
Local corrections do NOT violate the "never dense" rule because they
are bounded-size sub-blocks (never full-domain dense), explicitly
scoped to narrow bands/tiles.

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

# Maximum tile size (points per dimension). Anything larger is a QTT
# violation — tiles must stay small relative to the full domain.
MAX_TILE_SIZE: int = 256

# Default sensor threshold for shock detection
DEFAULT_SHOCK_THRESHOLD: float = 0.1

# Maximum fraction of domain covered by local tiles before we warn
MAX_LOCAL_COVERAGE: float = 0.25


# ─────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────


class TileKind(Enum):
    """Classification of local correction tiles."""

    SHOCK_BAND = auto()      # Narrow band around detected shock
    INTERFACE_BAND = auto()  # Narrow band around phase interface
    WALL_LAYER = auto()      # Near-wall correction layer
    SINGULARITY = auto()     # Point/line singularity neighborhood
    CUSTOM = auto()          # User-defined ROI


class SensorKind(Enum):
    """Feature sensor types."""

    GRADIENT_MAGNITUDE = auto()  # |∇q| exceeds threshold
    JUMP_INDICATOR = auto()      # Multi-scale jump indicator
    CURVATURE = auto()           # |∇²q| / |∇q| curvature detection
    PHASE_GRADIENT = auto()      # |∇φ| for phase-field interfaces


# ─────────────────────────────────────────────────────────────────────
# LocalTile — dense correction on a sub-domain
# ─────────────────────────────────────────────────────────────────────


@dataclass
class LocalTile:
    """Dense correction tile on a bounded sub-domain.

    A tile represents a small, bounded region where the QTT backbone
    is insufficient (e.g., near a shock or interface). The correction
    is stored as a dense array of shape matching the tile extent.

    Parameters
    ----------
    origin : tuple[int, ...]
        Multi-index of the tile's lower-left corner in the global grid.
    extent : tuple[int, ...]
        Number of points per dimension in the tile.
    data : NDArray
        Dense correction values. Shape must match *extent*.
    kind : TileKind
        Classification of why this tile exists.
    level : int
        Refinement level (0 = base, 1 = 2x refined, etc.).
    weight : float
        Blending weight ∈ [0, 1] for smooth tile-to-backbone transition.
    """

    origin: tuple[int, ...]
    extent: tuple[int, ...]
    data: NDArray
    kind: TileKind = TileKind.CUSTOM
    level: int = 0
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.data.shape != self.extent:
            raise ValueError(
                f"Tile data shape {self.data.shape} does not match "
                f"extent {self.extent}"
            )
        total_points = math.prod(self.extent)
        if total_points > MAX_TILE_SIZE ** len(self.extent):
            raise ValueError(
                f"Tile has {total_points} points, exceeding max "
                f"{MAX_TILE_SIZE ** len(self.extent)}"
            )
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"Weight must be in [0, 1], got {self.weight}")

    @property
    def n_points(self) -> int:
        """Total number of points in the tile."""
        return math.prod(self.extent)

    @property
    def n_dims(self) -> int:
        """Spatial dimensionality."""
        return len(self.extent)

    def global_indices(self) -> tuple[NDArray, ...]:
        """Return arrays of global indices covered by this tile.

        Returns
        -------
        tuple[NDArray, ...]
            One array per dimension of global grid indices.
        """
        ranges = [
            np.arange(o, o + e) for o, e in zip(self.origin, self.extent)
        ]
        return tuple(ranges)

    def blending_mask(self, taper_width: int = 4) -> NDArray:
        """Compute a smooth blending mask that tapers at tile edges.

        Uses a cosine taper to avoid abrupt transitions between the
        TT backbone and the local correction.

        Parameters
        ----------
        taper_width : int
            Number of points over which to taper from 0 to 1.

        Returns
        -------
        NDArray
            Mask of same shape as tile data, values in [0, 1].
        """
        masks_1d = []
        for size in self.extent:
            mask = np.ones(size, dtype=np.float64)
            tw = min(taper_width, size // 2)
            if tw > 0:
                ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(tw) / tw))
                mask[:tw] = ramp
                mask[-tw:] = ramp[::-1]
            masks_1d.append(mask)

        if len(masks_1d) == 1:
            return masks_1d[0] * self.weight
        elif len(masks_1d) == 2:
            return np.outer(masks_1d[0], masks_1d[1]) * self.weight
        else:
            result = masks_1d[0]
            for m in masks_1d[1:]:
                result = np.multiply.outer(result, m)
            return result * self.weight


# ─────────────────────────────────────────────────────────────────────
# HybridField — q = q_TT + q_local
# ─────────────────────────────────────────────────────────────────────


@dataclass
class HybridField:
    """Hybrid field representation: QTT backbone + local corrections.

    The evaluated field at any point is:
        q(x) = q_TT(x) + Σ_i  α_i(x) · tile_i(x)

    where α_i is the blending mask for tile i.

    Parameters
    ----------
    backbone : QTTTensor or None
        Global QTT-compressed field. None means the backbone is zero
        (pure local representation — unusual but allowed for testing).
    tiles : list[LocalTile]
        Sparse local corrections.
    name : str
        Human-readable field name.
    """

    backbone: object  # QTTTensor — import-free for flexibility
    tiles: list[LocalTile] = field(default_factory=list)
    name: str = ""

    @property
    def n_tiles(self) -> int:
        """Number of active local correction tiles."""
        return len(self.tiles)

    @property
    def local_point_count(self) -> int:
        """Total points stored in local tiles."""
        return sum(t.n_points for t in self.tiles)

    @property
    def backbone_point_count(self) -> int:
        """Equivalent grid size of the QTT backbone."""
        if self.backbone is None:
            return 0
        cores = getattr(self.backbone, "cores", [])
        if not cores:
            return 0
        n_sites = len(cores)
        return 2 ** n_sites

    @property
    def local_coverage_fraction(self) -> float:
        """Fraction of the global grid covered by local tiles.

        Values above MAX_LOCAL_COVERAGE indicate the hybrid approach
        is losing its compression advantage.
        """
        backbone_n = self.backbone_point_count
        if backbone_n == 0:
            return 0.0
        return self.local_point_count / backbone_n

    @property
    def is_coverage_healthy(self) -> bool:
        """True if local corrections cover ≤ MAX_LOCAL_COVERAGE of domain."""
        return self.local_coverage_fraction <= MAX_LOCAL_COVERAGE

    def add_tile(self, tile: LocalTile) -> None:
        """Add a local correction tile."""
        self.tiles.append(tile)

    def remove_tile(self, index: int) -> LocalTile:
        """Remove and return tile at given index."""
        return self.tiles.pop(index)

    def clear_tiles(self) -> None:
        """Remove all local correction tiles."""
        self.tiles.clear()

    def evaluate_on_grid(self, grid_shape: tuple[int, ...]) -> NDArray:
        """Evaluate the hybrid field on a full grid (for diagnostics ONLY).

        This materializes the full field — DO NOT use in the execution
        loop. Exists solely for post-execution diagnostics/V&V.

        Parameters
        ----------
        grid_shape : tuple[int, ...]
            Shape of the output grid.

        Returns
        -------
        NDArray
            Dense array of shape *grid_shape*.
        """
        # Start with backbone
        if self.backbone is not None:
            to_dense = getattr(self.backbone, "to_dense", None)
            if to_dense is not None:
                result = to_dense().reshape(grid_shape).copy()
            else:
                result = np.zeros(grid_shape, dtype=np.float64)
        else:
            result = np.zeros(grid_shape, dtype=np.float64)

        # Add local corrections with blending
        for tile in self.tiles:
            mask = tile.blending_mask()
            indices = tile.global_indices()
            if len(grid_shape) == 1:
                ix = indices[0]
                valid = ix < grid_shape[0]
                result[ix[valid]] += mask[valid] * tile.data[valid]
            elif len(grid_shape) == 2:
                ix, iy = indices
                ix_valid = ix[ix < grid_shape[0]]
                iy_valid = iy[iy < grid_shape[1]]
                mesh_ix, mesh_iy = np.meshgrid(ix_valid, iy_valid, indexing="ij")
                sub_mask = mask[:len(ix_valid), :len(iy_valid)]
                sub_data = tile.data[:len(ix_valid), :len(iy_valid)]
                result[mesh_ix, mesh_iy] += sub_mask * sub_data
            else:
                # General N-D: use tuple of slices
                slices = tuple(
                    slice(o, min(o + e, s))
                    for o, e, s in zip(tile.origin, tile.extent, grid_shape)
                )
                clip_extent = tuple(
                    min(o + e, s) - o
                    for o, e, s in zip(tile.origin, tile.extent, grid_shape)
                )
                clip_slices = tuple(slice(0, ce) for ce in clip_extent)
                result[slices] += mask[clip_slices] * tile.data[clip_slices]

        return result

    def diagnostics(self) -> dict[str, object]:
        """Return sanitizer-safe diagnostic summary.

        Returns
        -------
        dict
            Keys: n_tiles, local_point_count, local_coverage_fraction,
            coverage_healthy, tile_kinds.
        """
        return {
            "n_tiles": self.n_tiles,
            "local_point_count": self.local_point_count,
            "local_coverage_fraction": round(self.local_coverage_fraction, 6),
            "coverage_healthy": self.is_coverage_healthy,
            "tile_kinds": [t.kind.name for t in self.tiles],
        }


# ─────────────────────────────────────────────────────────────────────
# FeatureSensor — detects regions needing local corrections
# ─────────────────────────────────────────────────────────────────────


@dataclass
class FeatureSensorConfig:
    """Configuration for feature detection.

    Parameters
    ----------
    kind : SensorKind
        Type of feature to detect.
    threshold : float
        Activation threshold. Cells where the sensor exceeds this
        value are flagged for local correction.
    min_band_width : int
        Minimum half-width of the correction band around each
        detected feature point.
    max_band_width : int
        Maximum half-width of the correction band.
    """

    kind: SensorKind = SensorKind.GRADIENT_MAGNITUDE
    threshold: float = DEFAULT_SHOCK_THRESHOLD
    min_band_width: int = 4
    max_band_width: int = 16


def detect_features_1d(
    field_values: NDArray,
    h: float,
    config: FeatureSensorConfig | None = None,
) -> NDArray:
    """Detect feature locations in a 1D field.

    Uses gradient magnitude as the primary indicator. Returns a
    boolean mask over the field indicating active cells.

    Parameters
    ----------
    field_values : NDArray
        1D field values (dense, for sensor evaluation only).
    h : float
        Grid spacing.
    config : FeatureSensorConfig, optional
        Sensor configuration. Defaults to gradient magnitude sensor.

    Returns
    -------
    NDArray
        Boolean mask, True where feature is detected.
    """
    if config is None:
        config = FeatureSensorConfig()

    N = len(field_values)
    mask = np.zeros(N, dtype=bool)

    if config.kind == SensorKind.GRADIENT_MAGNITUDE:
        # Central differences for interior, one-sided at boundaries
        grad = np.zeros(N, dtype=np.float64)
        grad[1:-1] = (field_values[2:] - field_values[:-2]) / (2.0 * h)
        grad[0] = (field_values[1] - field_values[0]) / h
        grad[-1] = (field_values[-1] - field_values[-2]) / h
        indicator = np.abs(grad)

    elif config.kind == SensorKind.JUMP_INDICATOR:
        # Multi-scale jump indicator: compare local averages at
        # different stencil widths to detect discontinuities
        indicator = np.zeros(N, dtype=np.float64)
        for w in [1, 2, 4]:
            if N > 2 * w:
                left_avg = np.zeros(N); right_avg = np.zeros(N)
                for i in range(w, N - w):
                    left_avg[i] = np.mean(field_values[i - w:i])
                    right_avg[i] = np.mean(field_values[i:i + w])
                indicator = np.maximum(indicator, np.abs(right_avg - left_avg))

    elif config.kind == SensorKind.CURVATURE:
        # |∇²q| as curvature proxy
        lap = np.zeros(N, dtype=np.float64)
        lap[1:-1] = (
            field_values[2:] - 2.0 * field_values[1:-1] + field_values[:-2]
        ) / (h * h)
        indicator = np.abs(lap)

    elif config.kind == SensorKind.PHASE_GRADIENT:
        grad = np.zeros(N, dtype=np.float64)
        grad[1:-1] = (field_values[2:] - field_values[:-2]) / (2.0 * h)
        indicator = np.abs(grad)

    else:
        raise ValueError(f"Unknown sensor kind: {config.kind}")

    # Threshold
    active = indicator > config.threshold

    # Expand to band width
    band_hw = config.min_band_width
    for i in range(N):
        if active[i]:
            lo = max(0, i - band_hw)
            hi = min(N, i + band_hw + 1)
            mask[lo:hi] = True

    return mask


def detect_features_2d(
    field_values: NDArray,
    hx: float,
    hy: float,
    config: FeatureSensorConfig | None = None,
) -> NDArray:
    """Detect feature locations in a 2D field.

    Parameters
    ----------
    field_values : NDArray
        2D field values of shape (Nx, Ny).
    hx, hy : float
        Grid spacings in x and y.
    config : FeatureSensorConfig, optional
        Sensor configuration.

    Returns
    -------
    NDArray
        Boolean mask of shape (Nx, Ny), True where feature is detected.
    """
    if config is None:
        config = FeatureSensorConfig()

    Nx, Ny = field_values.shape
    mask = np.zeros((Nx, Ny), dtype=bool)

    if config.kind in (
        SensorKind.GRADIENT_MAGNITUDE,
        SensorKind.PHASE_GRADIENT,
    ):
        grad_x = np.zeros_like(field_values)
        grad_y = np.zeros_like(field_values)
        grad_x[1:-1, :] = (
            field_values[2:, :] - field_values[:-2, :]
        ) / (2.0 * hx)
        grad_y[:, 1:-1] = (
            field_values[:, 2:] - field_values[:, :-2]
        ) / (2.0 * hy)
        indicator = np.sqrt(grad_x**2 + grad_y**2)

    elif config.kind == SensorKind.CURVATURE:
        lap_x = np.zeros_like(field_values)
        lap_y = np.zeros_like(field_values)
        lap_x[1:-1, :] = (
            field_values[2:, :] - 2.0 * field_values[1:-1, :] +
            field_values[:-2, :]
        ) / (hx * hx)
        lap_y[:, 1:-1] = (
            field_values[:, 2:] - 2.0 * field_values[:, 1:-1] +
            field_values[:, :-2]
        ) / (hy * hy)
        indicator = np.abs(lap_x + lap_y)

    else:
        raise ValueError(f"Unsupported 2D sensor kind: {config.kind}")

    active = indicator > config.threshold

    # Expand band
    from scipy.ndimage import binary_dilation
    struct_size = 2 * config.min_band_width + 1
    struct = np.ones((struct_size, struct_size), dtype=bool)
    mask = binary_dilation(active, structure=struct)

    return mask


def tiles_from_mask_1d(
    mask: NDArray,
    correction: NDArray,
    kind: TileKind = TileKind.SHOCK_BAND,
) -> list[LocalTile]:
    """Convert a 1D boolean mask and correction values into LocalTiles.

    Contiguous True regions become individual tiles.

    Parameters
    ----------
    mask : NDArray
        Boolean mask of length N.
    correction : NDArray
        Dense correction values of length N.
    kind : TileKind
        Tile classification.

    Returns
    -------
    list[LocalTile]
        One tile per contiguous True region.
    """
    tiles: list[LocalTile] = []
    N = len(mask)
    i = 0
    while i < N:
        if mask[i]:
            start = i
            while i < N and mask[i]:
                i += 1
            end = i
            tile_data = correction[start:end].copy()
            tiles.append(
                LocalTile(
                    origin=(start,),
                    extent=(end - start,),
                    data=tile_data,
                    kind=kind,
                )
            )
        else:
            i += 1
    return tiles


# ─────────────────────────────────────────────────────────────────────
# TileActivationPolicy — decides which tiles are active
# ─────────────────────────────────────────────────────────────────────


@dataclass
class TileActivationPolicy:
    """Policy for activating and deactivating local correction tiles.

    Parameters
    ----------
    max_tiles : int
        Maximum number of simultaneous tiles.
    max_total_points : int
        Maximum total correction points across all tiles.
    deactivation_threshold : float
        If a tile's max correction magnitude drops below this,
        deactivate it (merge back into backbone on next round).
    recheck_interval : int
        Re-evaluate tile activation every N timesteps.
    """

    max_tiles: int = 32
    max_total_points: int = 65536  # 256² — bounded local budget
    deactivation_threshold: float = 1e-8
    recheck_interval: int = 10

    def should_activate(
        self,
        current_tiles: list[LocalTile],
        candidate_tiles: list[LocalTile],
    ) -> list[LocalTile]:
        """Filter candidate tiles according to budget constraints.

        Parameters
        ----------
        current_tiles : list[LocalTile]
            Already active tiles.
        candidate_tiles : list[LocalTile]
            Newly detected tiles requesting activation.

        Returns
        -------
        list[LocalTile]
            Subset of candidates that fit within budget.
        """
        current_count = len(current_tiles)
        current_points = sum(t.n_points for t in current_tiles)

        accepted: list[LocalTile] = []
        for tile in candidate_tiles:
            if current_count + len(accepted) >= self.max_tiles:
                break
            if current_points + tile.n_points > self.max_total_points:
                continue
            current_points += tile.n_points
            accepted.append(tile)
        return accepted

    def should_deactivate(self, tile: LocalTile) -> bool:
        """Check if a tile should be deactivated (correction negligible)."""
        return bool(np.max(np.abs(tile.data)) < self.deactivation_threshold)

    def prune(self, hybrid: HybridField) -> list[LocalTile]:
        """Remove tiles whose corrections have become negligible.

        Returns
        -------
        list[LocalTile]
            The tiles that were removed.
        """
        removed: list[LocalTile] = []
        keep: list[LocalTile] = []
        for tile in hybrid.tiles:
            if self.should_deactivate(tile):
                removed.append(tile)
            else:
                keep.append(tile)
        hybrid.tiles[:] = keep
        return removed


# ─────────────────────────────────────────────────────────────────────
# HybridRoundPolicy — truncation for hybrid fields
# ─────────────────────────────────────────────────────────────────────


@dataclass
class HybridRoundPolicy:
    """Truncation policy for hybrid fields.

    The backbone gets truncated more aggressively than a pure QTT
    field because local tiles capture the high-frequency features
    that would otherwise blow up the rank.

    Parameters
    ----------
    backbone_max_rank : int
        Maximum rank for the backbone (adaptive via governor).
    backbone_rel_tol : float
        Truncation tolerance for the backbone.
    tile_update_frequency : int
        How often (in timesteps) to re-evaluate tiles.
    aggressive_factor : float
        Factor by which to reduce backbone rank when tiles are
        active (> 0 tiles). E.g. 0.75 means 75% of normal rank.
    """

    backbone_max_rank: int = 64
    backbone_rel_tol: float = 1e-10
    tile_update_frequency: int = 5
    aggressive_factor: float = 0.75

    def effective_backbone_rank(self, n_tiles: int) -> int:
        """Compute effective backbone rank given active tile count.

        When tiles are active, backbone rank can be reduced because
        the tiles handle the rank-hostile features.
        """
        if n_tiles == 0:
            return self.backbone_max_rank
        reduced = int(self.backbone_max_rank * self.aggressive_factor)
        return max(4, reduced)  # Never drop below rank 4


# ─────────────────────────────────────────────────────────────────────
# Sanitizer for hybrid field diagnostics
# ─────────────────────────────────────────────────────────────────────

_HYBRID_DIAG_WHITELIST: frozenset[str] = frozenset({
    "n_tiles",
    "local_point_count",
    "local_coverage_fraction",
    "coverage_healthy",
    "tile_kinds",
    "backbone_peak_rank",
    "backbone_mean_rank",
    "tiles_activated",
    "tiles_deactivated",
    "hybrid_compression_ratio",
})


def sanitize_hybrid_diagnostics(raw: dict[str, object]) -> dict[str, object]:
    """Filter hybrid field diagnostics to whitelist-only outputs.

    Parameters
    ----------
    raw : dict
        Raw diagnostic dictionary.

    Returns
    -------
    dict
        Filtered dictionary containing only whitelisted keys.
    """
    return {k: v for k, v in raw.items() if k in _HYBRID_DIAG_WHITELIST}
