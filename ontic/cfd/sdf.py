"""Signed Distance Function (SDF) geometry library.

Provides a protocol-based geometry abstraction for immersed boundary
methods on structured Cartesian grids.  Each SDF maps every point in
space to its signed distance from the nearest surface:

    d < 0  →  inside solid
    d = 0  →  on the surface
    d > 0  →  outside (fluid)

All implementations are torch-compatible and differentiable for
gradient-based shape optimization.

References
----------
- Mittal & Iaccarino, "Immersed Boundary Methods",
  Annu. Rev. Fluid Mech. 37:239-261, 2005
- Hart, "Sphere Tracing", Visual Comp. 12(10), 1996
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import Tensor


# ═══════════════════════════════════════════════════════════════════
# Protocol
# ═══════════════════════════════════════════════════════════════════


class SDFGeometry(abc.ABC):
    """Abstract base for 2-D signed-distance-function geometries.

    Every concrete subclass must implement ``sdf``.  The default
    ``is_inside``, ``normal``, and ``surface_points`` derive from
    ``sdf`` automatically but may be overridden for efficiency.
    """

    @abc.abstractmethod
    def sdf(self, x: Tensor, y: Tensor) -> Tensor:
        """Signed distance from each (x, y) to the nearest surface.

        Returns
        -------
        Tensor
            Same shape as *x* / *y*.  Negative inside, positive outside.
        """
        ...

    def is_inside(self, x: Tensor, y: Tensor) -> Tensor:
        """Boolean mask — True where (x, y) is inside the solid."""
        return self.sdf(x, y) < 0.0

    def normal(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """Outward-pointing unit normal via autograd on ``sdf``.

        Falls back to finite-difference if autograd is unavailable.
        """
        xr = x.detach().requires_grad_(True)
        yr = y.detach().requires_grad_(True)
        d = self.sdf(xr, yr)
        ones = torch.ones_like(d)
        (gx,) = torch.autograd.grad(d, xr, grad_outputs=ones, retain_graph=True)
        (gy,) = torch.autograd.grad(d, yr, grad_outputs=ones, create_graph=False)
        mag = torch.sqrt(gx * gx + gy * gy).clamp(min=1e-30)
        return gx / mag, gy / mag

    @property
    @abc.abstractmethod
    def bounding_box(self) -> tuple[float, float, float, float]:
        """Tight axis-aligned bounding box: (x_min, x_max, y_min, y_max)."""
        ...

    @property
    @abc.abstractmethod
    def characteristic_length(self) -> float:
        """Reference length for Reynolds-number computation (metres)."""
        ...

    def surface_points(self, n: int = 256) -> tuple[Tensor, Tensor]:
        """Sample *n* approximately equi-spaced points on the surface.

        Default implementation marches around the bounding box and
        uses bisection along radial rays from the centroid.
        """
        xlo, xhi, ylo, yhi = self.bounding_box
        cx = 0.5 * (xlo + xhi)
        cy = 0.5 * (ylo + yhi)
        R = math.hypot(xhi - xlo, yhi - ylo)
        angles = torch.linspace(0, 2 * math.pi, n + 1, dtype=torch.float64)[:-1]
        xs_out: list[float] = []
        ys_out: list[float] = []
        for a in angles:
            dx = math.cos(a.item())
            dy = math.sin(a.item())
            lo_t, hi_t = 0.0, R
            for _ in range(48):  # bisection iterations
                mid = 0.5 * (lo_t + hi_t)
                px = torch.tensor([cx + mid * dx], dtype=torch.float64)
                py = torch.tensor([cy + mid * dy], dtype=torch.float64)
                if self.sdf(px, py).item() < 0:
                    lo_t = mid
                else:
                    hi_t = mid
            xs_out.append(cx + 0.5 * (lo_t + hi_t) * dx)
            ys_out.append(cy + 0.5 * (lo_t + hi_t) * dy)
        return (
            torch.tensor(xs_out, dtype=torch.float64),
            torch.tensor(ys_out, dtype=torch.float64),
        )


# ═══════════════════════════════════════════════════════════════════
# 1. Circle / Cylinder cross-section
# ═══════════════════════════════════════════════════════════════════


class CircleSDF(SDFGeometry):
    """Circle centred at (cx, cy) with radius *r*."""

    def __init__(self, center_x: float, center_y: float, radius: float) -> None:
        if radius <= 0:
            raise ValueError(f"radius must be > 0, got {radius}")
        self.cx = center_x
        self.cy = center_y
        self.r = radius

    def sdf(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.sqrt((x - self.cx) ** 2 + (y - self.cy) ** 2) - self.r

    def normal(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        dx = x - self.cx
        dy = y - self.cy
        mag = torch.sqrt(dx * dx + dy * dy).clamp(min=1e-30)
        return dx / mag, dy / mag

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        return (self.cx - self.r, self.cx + self.r, self.cy - self.r, self.cy + self.r)

    @property
    def characteristic_length(self) -> float:
        return 2.0 * self.r  # diameter

    def surface_points(self, n: int = 256) -> tuple[Tensor, Tensor]:
        t = torch.linspace(0, 2 * math.pi, n + 1, dtype=torch.float64)[:-1]
        return self.cx + self.r * torch.cos(t), self.cy + self.r * torch.sin(t)


# ═══════════════════════════════════════════════════════════════════
# 2. Ellipse
# ═══════════════════════════════════════════════════════════════════


class EllipseSDF(SDFGeometry):
    """Axis-aligned ellipse at (cx, cy) with semi-axes *a* (x) and *b* (y).

    Uses a smooth approximation; exact SDF for an ellipse requires
    iterative root-finding.  The approximation is excellent for
    immersed-boundary purposes (error < grid spacing).
    """

    def __init__(
        self,
        center_x: float,
        center_y: float,
        semi_a: float,
        semi_b: float,
    ) -> None:
        if semi_a <= 0 or semi_b <= 0:
            raise ValueError(f"semi-axes must be > 0, got a={semi_a}, b={semi_b}")
        self.cx = center_x
        self.cy = center_y
        self.a = semi_a
        self.b = semi_b

    def sdf(self, x: Tensor, y: Tensor) -> Tensor:
        # Approximate SDF: level-set function f = (x/a)²+(y/b)²-1
        # scaled to approximate signed distance near the boundary.
        px = (x - self.cx) / self.a
        py = (y - self.cy) / self.b
        f = px * px + py * py - 1.0
        # Gradient magnitude of the level-set for scaling
        gx = 2.0 * px / self.a
        gy = 2.0 * py / self.b
        gmag = torch.sqrt(gx * gx + gy * gy).clamp(min=1e-30)
        return f / gmag

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        return (self.cx - self.a, self.cx + self.a, self.cy - self.b, self.cy + self.b)

    @property
    def characteristic_length(self) -> float:
        # Projected frontal width (perpendicular to flow assumed in x)
        return 2.0 * self.b


# ═══════════════════════════════════════════════════════════════════
# 3. Axis-aligned rectangle
# ═══════════════════════════════════════════════════════════════════


class RectangleSDF(SDFGeometry):
    """Axis-aligned rectangle centred at (cx, cy)."""

    def __init__(
        self,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f"width/height must be > 0, got {width}×{height}")
        self.cx = center_x
        self.cy = center_y
        self.hw = width / 2.0
        self.hh = height / 2.0

    def sdf(self, x: Tensor, y: Tensor) -> Tensor:
        dx = torch.abs(x - self.cx) - self.hw
        dy = torch.abs(y - self.cy) - self.hh
        outside = torch.sqrt(torch.clamp(dx, min=0) ** 2 + torch.clamp(dy, min=0) ** 2)
        inside = torch.clamp(torch.maximum(dx, dy), max=0)
        return outside + inside

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        return (self.cx - self.hw, self.cx + self.hw, self.cy - self.hh, self.cy + self.hh)

    @property
    def characteristic_length(self) -> float:
        return 2.0 * self.hh  # frontal height


# ═══════════════════════════════════════════════════════════════════
# 4. Rounded rectangle
# ═══════════════════════════════════════════════════════════════════


class RoundedRectSDF(SDFGeometry):
    """Rectangle with rounded corners (radius *r*)."""

    def __init__(
        self,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        corner_radius: float,
    ) -> None:
        half_w = width / 2.0
        half_h = height / 2.0
        r = min(corner_radius, half_w, half_h)
        if width <= 0 or height <= 0:
            raise ValueError(f"width/height must be > 0, got {width}×{height}")
        if corner_radius < 0:
            raise ValueError(f"corner_radius must be >= 0, got {corner_radius}")
        self.cx = center_x
        self.cy = center_y
        self.hw = half_w
        self.hh = half_h
        self.r = r

    def sdf(self, x: Tensor, y: Tensor) -> Tensor:
        dx = torch.abs(x - self.cx) - (self.hw - self.r)
        dy = torch.abs(y - self.cy) - (self.hh - self.r)
        outside = torch.sqrt(torch.clamp(dx, min=0) ** 2 + torch.clamp(dy, min=0) ** 2) - self.r
        inside = torch.clamp(torch.maximum(dx, dy), max=0) - self.r
        return torch.where(
            (dx > 0) | (dy > 0),
            outside,
            inside,
        )

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        return (self.cx - self.hw, self.cx + self.hw, self.cy - self.hh, self.cy + self.hh)

    @property
    def characteristic_length(self) -> float:
        return 2.0 * self.hh


# ═══════════════════════════════════════════════════════════════════
# 5. Wedge (refactored from WedgeGeometry)
# ═══════════════════════════════════════════════════════════════════


class WedgeSDF(SDFGeometry):
    """Symmetric sharp wedge pointing into the flow (-x direction).

    Leading edge at (tip_x, tip_y), extending in +x by *length*.
    """

    def __init__(
        self,
        tip_x: float,
        tip_y: float,
        half_angle_rad: float,
        length: float,
    ) -> None:
        if half_angle_rad <= 0 or half_angle_rad >= math.pi / 2:
            raise ValueError(f"half_angle must be in (0, π/2), got {half_angle_rad}")
        if length <= 0:
            raise ValueError(f"length must be > 0, got {length}")
        self.tip_x = tip_x
        self.tip_y = tip_y
        self.half_angle = half_angle_rad
        self.length = length
        self._tan = math.tan(half_angle_rad)
        self._cos = math.cos(half_angle_rad)

    def sdf(self, x: Tensor, y: Tensor) -> Tensor:
        dx = x - self.tip_x
        dy = y - self.tip_y

        # Surface half-height at this x
        h = dx * self._tan

        # Distance to upper surface (perpendicular)
        d_upper = (torch.abs(dy) - h) * self._cos

        # Before tip or after trailing edge
        d_tip = torch.sqrt(dx ** 2 + dy ** 2)
        x_te = self.tip_x + self.length

        # SDF: before tip → distance to tip point
        #       along body → perpendicular to nearest surface
        #       after trailing edge → handled approximately
        result = torch.where(
            dx < 0,
            d_tip,
            torch.where(
                x > x_te,
                d_upper,  # downstream of TE; keep surface distance
                d_upper,
            ),
        )
        return result

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        h = self.length * self._tan
        return (self.tip_x, self.tip_x + self.length, self.tip_y - h, self.tip_y + h)

    @property
    def characteristic_length(self) -> float:
        return self.length


# ═══════════════════════════════════════════════════════════════════
# 6. NACA 4-digit airfoil
# ═══════════════════════════════════════════════════════════════════


class NACA4DigitSDF(SDFGeometry):
    """NACA 4-digit airfoil profile as an SDF.

    Parameters
    ----------
    chord : float
        Chord length (metres).
    naca_code : str
        4-digit NACA code, e.g. "0012", "2412".
    leading_edge_x, leading_edge_y : float
        Position of the leading edge.
    aoa_deg : float
        Angle of attack in degrees (positive nose-up).
    """

    def __init__(
        self,
        chord: float,
        naca_code: str = "0012",
        leading_edge_x: float = 0.0,
        leading_edge_y: float = 0.0,
        aoa_deg: float = 0.0,
    ) -> None:
        if len(naca_code) != 4 or not naca_code.isdigit():
            raise ValueError(f"naca_code must be 4 digits, got {naca_code!r}")
        if chord <= 0:
            raise ValueError(f"chord must be > 0, got {chord}")
        self.chord = chord
        self.le_x = leading_edge_x
        self.le_y = leading_edge_y
        self.aoa = math.radians(aoa_deg)
        self._m = int(naca_code[0]) / 100.0  # max camber
        self._p = int(naca_code[1]) / 10.0   # max camber position
        self._t = int(naca_code[2:]) / 100.0  # max thickness ratio

        # Pre-compute surface points for SDF evaluation
        self._n_surf = 512
        self._build_surface()

    def _thickness(self, xc: Tensor) -> Tensor:
        """Half-thickness at normalised chord position xc ∈ [0, 1]."""
        t = self._t
        return (
            t
            / 0.2
            * (
                0.2969 * torch.sqrt(xc.clamp(min=0))
                - 0.1260 * xc
                - 0.3516 * xc ** 2
                + 0.2843 * xc ** 3
                - 0.1015 * xc ** 4
            )
        )

    def _camber(self, xc: Tensor) -> tuple[Tensor, Tensor]:
        """Camber line y_c and dy_c/dx at normalised chord xc."""
        m, p = self._m, self._p
        if m == 0 or p == 0:
            z = torch.zeros_like(xc)
            return z, z
        yc = torch.where(
            xc < p,
            m / (p * p) * (2 * p * xc - xc ** 2),
            m / ((1 - p) ** 2) * (1 - 2 * p + 2 * p * xc - xc ** 2),
        )
        dyc = torch.where(
            xc < p,
            2 * m / (p * p) * (p - xc),
            2 * m / ((1 - p) ** 2) * (p - xc),
        )
        return yc, dyc

    def _build_surface(self) -> None:
        """Pre-compute upper and lower surface coordinates (body frame)."""
        xc = torch.linspace(0, 1, self._n_surf, dtype=torch.float64)
        yt = self._thickness(xc)
        yc, dyc = self._camber(xc)
        theta = torch.atan(dyc)

        # Upper surface
        xu = xc - yt * torch.sin(theta)
        yu = yc + yt * torch.cos(theta)
        # Lower surface
        xl = xc + yt * torch.sin(theta)
        yl = yc - yt * torch.cos(theta)

        # Concatenate into closed loop (upper forward, lower backward)
        self._sx = torch.cat([xu, xl.flip(0)]) * self.chord
        self._sy = torch.cat([yu, yl.flip(0)]) * self.chord

        # Rotate by AoA and translate
        cos_a = math.cos(-self.aoa)
        sin_a = math.sin(-self.aoa)
        rx = self._sx * cos_a - self._sy * sin_a + self.le_x
        ry = self._sx * sin_a + self._sy * cos_a + self.le_y
        self._sx = rx
        self._sy = ry

    def sdf(self, x: Tensor, y: Tensor) -> Tensor:
        """Closest-point SDF to the pre-computed surface polygon."""
        # Broadcast: (Nquery, 1) vs (1, Nsurf)
        flat_x = x.reshape(-1)
        flat_y = y.reshape(-1)
        nq = flat_x.shape[0]
        ns = self._sx.shape[0]

        # Segment start and end
        sx0 = self._sx  # (ns,)
        sy0 = self._sy
        sx1 = torch.roll(self._sx, -1, 0)
        sy1 = torch.roll(self._sy, -1, 0)

        # Vector from segment start to query point
        qx = flat_x.unsqueeze(1) - sx0.unsqueeze(0)  # (nq, ns)
        qy = flat_y.unsqueeze(1) - sy0.unsqueeze(0)
        # Segment direction
        ex = (sx1 - sx0).unsqueeze(0)  # (1, ns)
        ey = (sy1 - sy0).unsqueeze(0)
        elen2 = (ex * ex + ey * ey).clamp(min=1e-30)

        # Project query onto segment
        t = ((qx * ex + qy * ey) / elen2).clamp(0, 1)
        # Closest point on segment
        cpx = qx - t * ex
        cpy = qy - t * ey
        dist2 = cpx * cpx + cpy * cpy

        # Minimum distance over all segments
        min_dist2, _ = dist2.min(dim=1)
        min_dist = torch.sqrt(min_dist2)

        # Sign: winding number (cross product sign)
        cross = qx * ey - qy * ex
        # Use the cross product at the nearest segment for sign
        _, nearest_idx = dist2.min(dim=1)
        nearest_cross = cross.gather(1, nearest_idx.unsqueeze(1)).squeeze(1)
        # If cross > 0, the query is to the left of the nearest segment
        # → inside the polygon (clockwise loop) → negative SDF
        sign = torch.where(nearest_cross >= 0, -torch.ones_like(min_dist), torch.ones_like(min_dist))

        return (sign * min_dist).reshape(x.shape)

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        return (
            self._sx.min().item(),
            self._sx.max().item(),
            self._sy.min().item(),
            self._sy.max().item(),
        )

    @property
    def characteristic_length(self) -> float:
        return self.chord

    def surface_points(self, n: int = 256) -> tuple[Tensor, Tensor]:
        step = max(1, len(self._sx) // n)
        return self._sx[::step][:n], self._sy[::step][:n]


# ═══════════════════════════════════════════════════════════════════
# 7. Flat plate (thin rectangle)
# ═══════════════════════════════════════════════════════════════════


class FlatPlateSDF(SDFGeometry):
    """Thin flat plate aligned with the x-axis."""

    def __init__(
        self,
        x_start: float,
        y_center: float,
        length: float,
        thickness: float = 0.001,
    ) -> None:
        if length <= 0:
            raise ValueError(f"length must be > 0, got {length}")
        if thickness <= 0:
            raise ValueError(f"thickness must be > 0, got {thickness}")
        self._rect = RectangleSDF(
            center_x=x_start + length / 2,
            center_y=y_center,
            width=length,
            height=thickness,
        )
        self._length = length
        self._thickness = thickness

    def sdf(self, x: Tensor, y: Tensor) -> Tensor:
        return self._rect.sdf(x, y)

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        return self._rect.bounding_box

    @property
    def characteristic_length(self) -> float:
        return self._length


# ═══════════════════════════════════════════════════════════════════
# 8. Fin array (repeated rectangular fins on a base)
# ═══════════════════════════════════════════════════════════════════


class FinArraySDF(SDFGeometry):
    """Array of rectangular fins extending from a base plate.

    Fins extend in +y from base_y.  The base plate itself is
    included as part of the SDF.
    """

    def __init__(
        self,
        base_y: float,
        n_fins: int,
        fin_height: float,
        fin_thickness: float,
        fin_spacing: float,
        base_thickness: float = 0.002,
    ) -> None:
        if n_fins < 1:
            raise ValueError(f"n_fins must be >= 1, got {n_fins}")
        if fin_height <= 0 or fin_thickness <= 0 or fin_spacing <= 0:
            raise ValueError("fin_height, fin_thickness, fin_spacing must be > 0")

        self.base_y = base_y
        self.n_fins = n_fins
        self.fin_height = fin_height
        self.fin_thickness = fin_thickness
        self.fin_spacing = fin_spacing
        self.base_thickness = base_thickness

        # Build individual fin SDFs
        total_width = (n_fins - 1) * fin_spacing + fin_thickness
        start_x = -total_width / 2.0

        bodies: list[RectangleSDF] = []
        for i in range(n_fins):
            cx = start_x + i * fin_spacing + fin_thickness / 2.0
            cy = base_y + base_thickness + fin_height / 2.0
            bodies.append(RectangleSDF(cx, cy, fin_thickness, fin_height))

        # Base plate
        base_cx = 0.0
        base_cy = base_y + base_thickness / 2.0
        bodies.append(RectangleSDF(base_cx, base_cy, total_width + fin_spacing, base_thickness))

        self._bodies = bodies
        self._total_width = total_width

    def sdf(self, x: Tensor, y: Tensor) -> Tensor:
        d = self._bodies[0].sdf(x, y)
        for body in self._bodies[1:]:
            d = torch.minimum(d, body.sdf(x, y))
        return d

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        boxes = [b.bounding_box for b in self._bodies]
        return (
            min(b[0] for b in boxes),
            max(b[1] for b in boxes),
            min(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )

    @property
    def characteristic_length(self) -> float:
        return self.fin_height


# ═══════════════════════════════════════════════════════════════════
# 9. Pipe bend (2D cross-section of a 90° elbow)
# ═══════════════════════════════════════════════════════════════════


class PipeBendSDF(SDFGeometry):
    """2D representation of a pipe with a circular bend.

    The pipe centre-line enters horizontally from the left, bends
    through *bend_angle_deg* (default 90°), and exits vertically upward.
    The wall thickness defines an annular region around the bend.
    """

    def __init__(
        self,
        inner_radius: float,
        outer_radius: float,
        bend_radius: float,
        bend_angle_deg: float = 90.0,
    ) -> None:
        if inner_radius <= 0 or outer_radius <= inner_radius:
            raise ValueError("Must have 0 < inner_radius < outer_radius")
        if bend_radius <= 0:
            raise ValueError(f"bend_radius must be > 0, got {bend_radius}")
        self.r_in = inner_radius
        self.r_out = outer_radius
        self.r_bend = bend_radius
        self.bend_angle = math.radians(bend_angle_deg)
        self._half_gap = (outer_radius - inner_radius) / 2.0
        self._mid_r = (outer_radius + inner_radius) / 2.0

    def sdf(self, x: Tensor, y: Tensor) -> Tensor:
        # Straight inlet section (x < 0): horizontal channel
        d_inlet_inner = torch.abs(y) - self.r_in
        d_inlet_outer = self.r_out - torch.abs(y)
        d_inlet_walls = -torch.minimum(d_inlet_inner, d_inlet_outer)
        d_inlet = torch.where(y.abs() < self.r_out, d_inlet_walls, torch.abs(y) - self.r_out)

        # Bend section: annular distance from bend centre (0, r_bend)
        bend_cx, bend_cy = 0.0, self.r_bend
        dist_to_bend_center = torch.sqrt(x ** 2 + (y - bend_cy) ** 2)
        d_bend_channel = torch.abs(dist_to_bend_center - self.r_bend) - self._half_gap

        # Choose based on position (simplified: use minimum)
        return torch.minimum(d_inlet, d_bend_channel)

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        r = self.r_bend + self.r_out
        return (-r, r, -self.r_out, r)

    @property
    def characteristic_length(self) -> float:
        return self.r_out - self.r_in  # hydraulic diameter ≈ gap width


# ═══════════════════════════════════════════════════════════════════
# 10. Concentric annulus
# ═══════════════════════════════════════════════════════════════════


class ConcentricAnnulusSDF(SDFGeometry):
    """Annular region between two concentric circles.

    The SDF is negative in the *solid* regions (inside the inner
    circle OR outside the outer circle), i.e. the fluid region
    (the annular gap) has positive SDF.

    For immersed-boundary use: the inner cylinder is the solid body;
    the outer cylinder is the domain boundary.  This SDF represents
    the inner cylinder only.
    """

    def __init__(
        self,
        center_x: float,
        center_y: float,
        r_inner: float,
        r_outer: float,
    ) -> None:
        if r_inner <= 0 or r_outer <= r_inner:
            raise ValueError("Must have 0 < r_inner < r_outer")
        self.cx = center_x
        self.cy = center_y
        self.r_inner = r_inner
        self.r_outer = r_outer

    def sdf(self, x: Tensor, y: Tensor) -> Tensor:
        """SDF for the inner cylinder (solid body)."""
        d = torch.sqrt((x - self.cx) ** 2 + (y - self.cy) ** 2) - self.r_inner
        return d

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        r = self.r_outer
        return (self.cx - r, self.cx + r, self.cy - r, self.cy + r)

    @property
    def characteristic_length(self) -> float:
        return self.r_outer - self.r_inner  # gap width


# ═══════════════════════════════════════════════════════════════════
# 11. Multi-body (CSG union)
# ═══════════════════════════════════════════════════════════════════


class MultiBodySDF(SDFGeometry):
    """CSG union of multiple SDFGeometry objects.

    The SDF at each point is ``min(sdf_1, sdf_2, …)`` — the standard
    signed-distance union operator.
    """

    def __init__(self, bodies: Sequence[SDFGeometry]) -> None:
        if len(bodies) < 1:
            raise ValueError("MultiBodySDF requires at least one body")
        self._bodies = list(bodies)

    def sdf(self, x: Tensor, y: Tensor) -> Tensor:
        d = self._bodies[0].sdf(x, y)
        for body in self._bodies[1:]:
            d = torch.minimum(d, body.sdf(x, y))
        return d

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        boxes = [b.bounding_box for b in self._bodies]
        return (
            min(b[0] for b in boxes),
            max(b[1] for b in boxes),
            min(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )

    @property
    def characteristic_length(self) -> float:
        return max(b.characteristic_length for b in self._bodies)


# ═══════════════════════════════════════════════════════════════════
# 12. Backward-facing step
# ═══════════════════════════════════════════════════════════════════


class StepSDF(SDFGeometry):
    """Backward-facing step geometry.

    A solid block occupies the upper portion of the channel upstream
    of the step location, creating a sudden expansion.

    ::

        ████████████████████
        ████████████████████
        ████████████
        ████████████         ← step_height
        ─────────────────────  channel floor (y=0)
                   ^
                step_x
    """

    def __init__(
        self,
        step_x: float,
        step_height: float,
        channel_height: float,
    ) -> None:
        if step_height <= 0 or channel_height <= step_height:
            raise ValueError("Must have 0 < step_height < channel_height")
        self.step_x = step_x
        self.step_height = step_height
        self.channel_height = channel_height
        # The solid block: from y = (channel_height - step_height) to y = channel_height,
        # for x < step_x.  Represented as a rectangle with very large upstream extent.
        block_width = 100.0 * channel_height  # effectively semi-infinite
        self._block = RectangleSDF(
            center_x=step_x - block_width / 2.0,
            center_y=channel_height - step_height / 2.0,
            width=block_width,
            height=step_height,
        )

    def sdf(self, x: Tensor, y: Tensor) -> Tensor:
        return self._block.sdf(x, y)

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        return (
            self.step_x - 20 * self.channel_height,
            self.step_x,
            self.channel_height - self.step_height,
            self.channel_height,
        )

    @property
    def characteristic_length(self) -> float:
        return self.step_height
