"""
Slice Engine
============

Resolution-independent slicing from QTT fields.
Extract 2D planes from 3D fields, or line profiles from 2D fields.

Features:
    - Arbitrary slice planes (xy, xz, yz, or custom normal)
    - Streaming extraction for large slices
    - Volume rendering with ray marching
    - Adaptive sampling based on field complexity
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch

# =============================================================================
# DATA STRUCTURES
# =============================================================================


class SlicePlane(Enum):
    """Standard slice planes."""

    XY = "xy"
    XZ = "xz"
    YZ = "yz"
    CUSTOM = "custom"


@dataclass
class SliceResult:
    """Result of a slice operation."""

    data: np.ndarray  # 2D slice data
    plane: SlicePlane  # Slice plane
    depth: float  # Depth in field coordinates
    bounds: tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)
    resolution: tuple[int, int]  # (width, height)

    # Metadata
    field_range: tuple[float, float] = (0.0, 1.0)  # (min, max) values
    render_time_ms: float = 0.0
    rank_used: int = 0

    def normalize(self) -> np.ndarray:
        """Normalize to [0, 1] range."""
        vmin, vmax = self.field_range
        if vmax - vmin < 1e-10:
            return np.zeros_like(self.data)
        return (self.data - vmin) / (vmax - vmin)

    def to_image(self, colormap: ColorMap = None) -> np.ndarray:
        """Convert to RGBA image."""
        from .colormaps import VIRIDIS, apply_colormap

        cm = colormap or VIRIDIS
        normalized = self.normalize()
        return apply_colormap(normalized, cm)


@dataclass
class VolumeResult:
    """Result of volume rendering."""

    image: np.ndarray  # Rendered image (H, W, 4) RGBA
    depth_buffer: np.ndarray  # Depth values (H, W)
    resolution: tuple[int, int]

    # Camera
    view_matrix: np.ndarray  # 4x4 view matrix
    proj_matrix: np.ndarray  # 4x4 projection matrix

    # Stats
    render_time_ms: float = 0.0
    samples_per_ray: int = 0


# =============================================================================
# SLICE ENGINE
# =============================================================================


class SliceEngine:
    """
    Resolution-independent slicing from QTT fields.

    Usage:
        slicer = SliceEngine(field)

        # 2D slice at depth 0.5
        result = slicer.slice(plane='xy', depth=0.5, resolution=1024)

        # Line profile
        profile = slicer.line_profile(
            start=(0, 0.5),
            end=(1, 0.5),
            samples=1000,
        )
    """

    def __init__(
        self,
        field: Field,
        device: str = "cuda",
    ):
        self.field = field
        self.device = device if torch.cuda.is_available() else "cpu"

        # Cache for common operations
        self._slice_cache: dict[str, SliceResult] = {}

    def slice(
        self,
        plane: str | SlicePlane = "xy",
        depth: float = 0.5,
        resolution: int | tuple[int, int] = 512,
        bounds: tuple[float, float, float, float] | None = None,
        max_rank: int | None = None,
    ) -> SliceResult:
        """
        Extract a 2D slice from the field.

        Args:
            plane: Slice plane ('xy', 'xz', 'yz')
            depth: Depth along perpendicular axis [0, 1]
            resolution: Output resolution (int or (width, height))
            bounds: Spatial bounds (x_min, y_min, x_max, y_max)
            max_rank: Rank cap for extraction

        Returns:
            SliceResult with 2D data
        """
        t_start = time.perf_counter()

        # Parse plane
        if isinstance(plane, str):
            plane = SlicePlane(plane.lower())

        # Parse resolution
        if isinstance(resolution, int):
            resolution = (resolution, resolution)

        # Default bounds
        if bounds is None:
            bounds = (0.0, 0.0, 1.0, 1.0)

        x_min, y_min, x_max, y_max = bounds
        width, height = resolution

        # Create sample grid
        xs = torch.linspace(x_min, x_max, width, device=self.device)
        ys = torch.linspace(y_min, y_max, height, device=self.device)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")

        # Build 3D sample points based on plane
        if plane == SlicePlane.XY:
            points = torch.stack(
                [
                    grid_x.flatten(),
                    grid_y.flatten(),
                    torch.full((width * height,), depth, device=self.device),
                ],
                dim=1,
            )
        elif plane == SlicePlane.XZ:
            points = torch.stack(
                [
                    grid_x.flatten(),
                    torch.full((width * height,), depth, device=self.device),
                    grid_y.flatten(),
                ],
                dim=1,
            )
        elif plane == SlicePlane.YZ:
            points = torch.stack(
                [
                    torch.full((width * height,), depth, device=self.device),
                    grid_x.flatten(),
                    grid_y.flatten(),
                ],
                dim=1,
            )
        else:
            raise ValueError(f"Unsupported plane: {plane}")

        # Handle 2D fields
        if self.field.dims == 2:
            points = points[:, :2]  # Use only x, y

        # Sample field
        values = self.field.sample(points)
        # D-016: Visualization output - numpy required for image display
        data = (
            values.reshape(width, height).cpu().numpy().T
        )  # Transpose for image convention

        t_end = time.perf_counter()

        return SliceResult(
            data=data,
            plane=plane,
            depth=depth,
            bounds=bounds,
            resolution=resolution,
            field_range=(float(data.min()), float(data.max())),
            render_time_ms=(t_end - t_start) * 1000,
            rank_used=max(c.shape[0] for c in self.field.cores),
        )

    def line_profile(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        samples: int = 1000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract a 1D line profile from the field.

        Args:
            start: Start point (x, y) normalized [0, 1]
            end: End point (x, y) normalized [0, 1]
            samples: Number of samples along line

        Returns:
            (positions, values) tuple
        """
        t = torch.linspace(0, 1, samples, device=self.device)
        x0, y0 = start
        x1, y1 = end

        xs = x0 + t * (x1 - x0)
        ys = y0 + t * (y1 - y0)

        points = torch.stack([xs, ys], dim=1)
        values = self.field.sample(points)

        # D-016: Visualization output for plotting
        positions = t.cpu().numpy()
        values = values.cpu().numpy()

        return positions, values

    def multi_slice(
        self,
        depths: list[float],
        plane: str | SlicePlane = "xy",
        resolution: int = 256,
    ) -> list[SliceResult]:
        """
        Extract multiple slices at different depths.

        Useful for building volume from 2D slices.
        """
        return [self.slice(plane=plane, depth=d, resolution=resolution) for d in depths]


# =============================================================================
# VOLUME RENDERER
# =============================================================================


class VolumeRenderer:
    """
    Volume rendering via ray marching through QTT field.

    Uses direct volume rendering (DVR) with opacity accumulation.
    All sampling done in QTT format - never decompresses full volume.
    """

    def __init__(
        self,
        field: Field,
        transfer_function: TransferFunction | None = None,
        device: str = "cuda",
    ):
        self.field = field
        self.transfer_fn = transfer_function
        self.device = device if torch.cuda.is_available() else "cpu"

    def render(
        self,
        resolution: tuple[int, int] = (512, 512),
        samples_per_ray: int = 128,
        camera_pos: tuple[float, float, float] = (2.0, 2.0, 2.0),
        look_at: tuple[float, float, float] = (0.5, 0.5, 0.5),
        fov: float = 45.0,
    ) -> VolumeResult:
        """
        Render volume using ray marching.

        Args:
            resolution: Output resolution (width, height)
            samples_per_ray: Number of samples per ray
            camera_pos: Camera position in world space
            look_at: Look-at point
            fov: Field of view in degrees

        Returns:
            VolumeResult with rendered image
        """
        t_start = time.perf_counter()

        width, height = resolution

        # Build camera
        view_matrix = self._look_at(camera_pos, look_at)
        proj_matrix = self._perspective(fov, width / height)

        # Generate rays
        rays_o, rays_d = self._generate_rays(width, height, camera_pos, look_at, fov)

        # Ray march
        image, depth = self._ray_march(rays_o, rays_d, samples_per_ray)

        t_end = time.perf_counter()

        return VolumeResult(
            image=image,
            depth_buffer=depth,
            resolution=resolution,
            view_matrix=view_matrix,
            proj_matrix=proj_matrix,
            render_time_ms=(t_end - t_start) * 1000,
            samples_per_ray=samples_per_ray,
        )

    def _generate_rays(
        self,
        width: int,
        height: int,
        camera_pos: tuple[float, float, float],
        look_at: tuple[float, float, float],
        fov: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate camera rays."""
        device = self.device

        # Camera basis
        cam_pos = torch.tensor(camera_pos, device=device, dtype=torch.float32)
        target = torch.tensor(look_at, device=device, dtype=torch.float32)

        forward = target - cam_pos
        forward = forward / forward.norm()

        up = torch.tensor([0.0, 1.0, 0.0], device=device)
        right = torch.linalg.cross(forward, up)
        right = right / right.norm()
        up = torch.linalg.cross(right, forward)

        # Screen coordinates
        aspect = width / height
        fov_rad = fov * np.pi / 180
        half_height = np.tan(fov_rad / 2)
        half_width = half_height * aspect

        u = torch.linspace(-half_width, half_width, width, device=device)
        v = torch.linspace(-half_height, half_height, height, device=device)
        uu, vv = torch.meshgrid(u, v, indexing="xy")

        # Ray directions
        rays_d = (
            forward[None, None, :]
            + uu[:, :, None] * right[None, None, :]
            + vv[:, :, None] * up[None, None, :]
        )
        rays_d = rays_d / rays_d.norm(dim=2, keepdim=True)

        # Ray origins (all same)
        rays_o = cam_pos.expand(width, height, 3)

        return rays_o, rays_d

    def _ray_march(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        n_samples: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Ray march through volume."""
        width, height = rays_o.shape[:2]
        device = rays_o.device

        # Intersect with unit cube [0, 1]^3
        t_near = 0.0
        t_far = 3.0  # Max ray length

        # Sample along rays
        t_vals = torch.linspace(t_near, t_far, n_samples, device=device)

        # Accumulate color and opacity
        color = torch.zeros(width, height, 3, device=device)
        alpha = torch.zeros(width, height, device=device)
        depth = torch.zeros(width, height, device=device)

        for i, t in enumerate(t_vals):
            # Sample points
            points = rays_o + t * rays_d
            points_flat = points.reshape(-1, 3)

            # Clamp to [0, 1] for field sampling
            points_flat = torch.clamp(points_flat, 0, 1)

            # Handle 2D fields
            if self.field.dims == 2:
                points_2d = points_flat[:, :2]
                values = self.field.sample(points_2d)
            else:
                values = self.field.sample(points_flat)

            values = values.reshape(width, height)

            # Simple density-to-color mapping
            density = torch.clamp(values.abs(), 0, 1)
            sample_color = density[:, :, None].expand(-1, -1, 3)
            sample_alpha = density * 0.1  # Opacity per sample

            # Front-to-back compositing
            transmittance = 1.0 - alpha
            color = (
                color
                + transmittance[:, :, None] * sample_color * sample_alpha[:, :, None]
            )

            # Update depth where first hit
            first_hit = (alpha < 0.01) & (sample_alpha > 0.01)
            depth[first_hit] = t

            alpha = alpha + transmittance * sample_alpha

        # Convert to RGBA image
        image = torch.cat([color, alpha[:, :, None]], dim=2)
        # D-016: Visualization output for display
        image = image.cpu().numpy()
        depth = depth.cpu().numpy()

        return image, depth

    def _look_at(
        self,
        eye: tuple[float, float, float],
        target: tuple[float, float, float],
    ) -> np.ndarray:
        """Build view matrix."""
        eye = np.array(eye)
        target = np.array(target)
        up = np.array([0, 1, 0])

        f = target - eye
        f = f / np.linalg.norm(f)

        r = np.cross(f, up)
        r = r / np.linalg.norm(r)

        u = np.cross(r, f)

        view = np.eye(4)
        view[0, :3] = r
        view[1, :3] = u
        view[2, :3] = -f
        view[:3, 3] = -view[:3, :3] @ eye

        return view

    def _perspective(self, fov: float, aspect: float) -> np.ndarray:
        """Build perspective projection matrix."""
        f = 1.0 / np.tan(np.radians(fov) / 2)
        near, far = 0.1, 100.0

        proj = np.zeros((4, 4))
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = 2 * far * near / (near - far)
        proj[3, 2] = -1

        return proj
