"""
Mesh Generation & Adaptive Mesh Refinement (AMR) — Octree/quadtree,
Delaunay triangulation, h-adaptivity, Morton Z-curve.

Domain XVII.4 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Quadtree AMR
# ---------------------------------------------------------------------------

class QuadtreeNode:
    """Node in a quadtree for 2D AMR."""

    __slots__ = ('x0', 'y0', 'dx', 'dy', 'level', 'children', 'data',
                 'is_leaf', 'max_level')

    def __init__(self, x0: float, y0: float, dx: float, dy: float,
                 level: int = 0, max_level: int = 8) -> None:
        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy
        self.level = level
        self.max_level = max_level
        self.children: Optional[List[QuadtreeNode]] = None
        self.data: float = 0.0
        self.is_leaf = True

    @property
    def centre(self) -> Tuple[float, float]:
        return (self.x0 + self.dx / 2, self.y0 + self.dy / 2)

    def refine(self) -> None:
        """Split into 4 children."""
        if self.level >= self.max_level:
            return
        half_dx = self.dx / 2
        half_dy = self.dy / 2
        self.children = [
            QuadtreeNode(self.x0, self.y0, half_dx, half_dy,
                          self.level + 1, self.max_level),
            QuadtreeNode(self.x0 + half_dx, self.y0, half_dx, half_dy,
                          self.level + 1, self.max_level),
            QuadtreeNode(self.x0, self.y0 + half_dy, half_dx, half_dy,
                          self.level + 1, self.max_level),
            QuadtreeNode(self.x0 + half_dx, self.y0 + half_dy, half_dx, half_dy,
                          self.level + 1, self.max_level),
        ]
        self.is_leaf = False

    def coarsen(self) -> None:
        """Remove children (coarsen)."""
        self.children = None
        self.is_leaf = True


class QuadtreeAMR:
    """2D adaptive mesh refinement using a quadtree."""

    def __init__(self, Lx: float = 1.0, Ly: float = 1.0,
                 max_level: int = 8) -> None:
        self.root = QuadtreeNode(0, 0, Lx, Ly, 0, max_level)
        self.max_level = max_level

    def refine_by_criterion(self, criterion: Callable[[QuadtreeNode], bool]) -> int:
        """Recursively refine leaves that satisfy criterion.

        Returns number of cells refined.
        """
        count = 0

        def _refine(node: QuadtreeNode) -> None:
            nonlocal count
            if node.is_leaf:
                if criterion(node):
                    node.refine()
                    count += 1
            else:
                if node.children:
                    for child in node.children:
                        _refine(child)

        _refine(self.root)
        return count

    def get_leaves(self) -> List[QuadtreeNode]:
        """Return all leaf nodes."""
        leaves: List[QuadtreeNode] = []

        def _collect(node: QuadtreeNode) -> None:
            if node.is_leaf:
                leaves.append(node)
            elif node.children:
                for child in node.children:
                    _collect(child)

        _collect(self.root)
        return leaves

    def total_cells(self) -> int:
        return len(self.get_leaves())

    def cell_centres(self) -> NDArray:
        """Return (N, 2) array of leaf cell centres."""
        leaves = self.get_leaves()
        centres = np.array([leaf.centre for leaf in leaves])
        return centres

    def effective_resolution(self) -> float:
        """Finest cell size."""
        leaves = self.get_leaves()
        return min(leaf.dx for leaf in leaves)


# ---------------------------------------------------------------------------
#  Octree (3D AMR)
# ---------------------------------------------------------------------------

class OctreeNode:
    """Node in an octree for 3D AMR."""

    __slots__ = ('x0', 'y0', 'z0', 'dx', 'dy', 'dz', 'level',
                 'children', 'data', 'is_leaf', 'max_level')

    def __init__(self, x0: float, y0: float, z0: float,
                 dx: float, dy: float, dz: float,
                 level: int = 0, max_level: int = 6) -> None:
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.level = level
        self.max_level = max_level
        self.children: Optional[List[OctreeNode]] = None
        self.data: float = 0.0
        self.is_leaf = True

    def refine(self) -> None:
        if self.level >= self.max_level:
            return
        hx, hy, hz = self.dx / 2, self.dy / 2, self.dz / 2
        self.children = []
        for iz in range(2):
            for iy in range(2):
                for ix in range(2):
                    self.children.append(OctreeNode(
                        self.x0 + ix * hx, self.y0 + iy * hy,
                        self.z0 + iz * hz,
                        hx, hy, hz, self.level + 1, self.max_level))
        self.is_leaf = False


# ---------------------------------------------------------------------------
#  Morton Z-Curve (Space-Filling Curve)
# ---------------------------------------------------------------------------

class MortonCurve:
    r"""
    Morton Z-order space-filling curve for cache-efficient tree traversal.

    Interleave bits of (x, y) or (x, y, z) to produce a single index
    that preserves spatial locality.

    2D: $z = \text{interleave}(x, y)$.
    3D: $z = \text{interleave}(x, y, z)$.
    """

    @staticmethod
    def encode_2d(x: int, y: int) -> int:
        """Interleave bits of x and y (32-bit max)."""
        def _spread(v: int) -> int:
            v &= 0x0000FFFF
            v = (v | (v << 8)) & 0x00FF00FF
            v = (v | (v << 4)) & 0x0F0F0F0F
            v = (v | (v << 2)) & 0x33333333
            v = (v | (v << 1)) & 0x55555555
            return v
        return _spread(x) | (_spread(y) << 1)

    @staticmethod
    def decode_2d(z: int) -> Tuple[int, int]:
        """Extract x, y from Morton code."""
        def _compact(v: int) -> int:
            v &= 0x55555555
            v = (v | (v >> 1)) & 0x33333333
            v = (v | (v >> 2)) & 0x0F0F0F0F
            v = (v | (v >> 4)) & 0x00FF00FF
            v = (v | (v >> 8)) & 0x0000FFFF
            return v
        return _compact(z), _compact(z >> 1)

    @staticmethod
    def encode_3d(x: int, y: int, z: int) -> int:
        """3D Morton encoding."""
        def _spread3(v: int) -> int:
            v &= 0x000003FF
            v = (v | (v << 16)) & 0x030000FF
            v = (v | (v << 8)) & 0x0300F00F
            v = (v | (v << 4)) & 0x030C30C3
            v = (v | (v << 2)) & 0x09249249
            return v
        return _spread3(x) | (_spread3(y) << 1) | (_spread3(z) << 2)

    @staticmethod
    def sort_points_2d(points: NDArray, n_bits: int = 16) -> NDArray:
        """Sort 2D points along Morton curve.

        points: (N, 2) array.
        Returns sorted indices.
        """
        N = len(points)
        pmin = points.min(axis=0)
        pmax = points.max(axis=0)
        scale = (2**n_bits - 1) / (pmax - pmin + 1e-10)

        codes = np.zeros(N, dtype=np.int64)
        for i in range(N):
            xi = int((points[i, 0] - pmin[0]) * scale[0])
            yi = int((points[i, 1] - pmin[1]) * scale[1])
            codes[i] = MortonCurve.encode_2d(xi, yi)

        return np.argsort(codes)


# ---------------------------------------------------------------------------
#  Delaunay Triangulation (2D, Bowyer-Watson)
# ---------------------------------------------------------------------------

class DelaunayTriangulation2D:
    r"""
    2D Delaunay triangulation via the Bowyer-Watson algorithm.

    Circumcircle property: no point lies inside the circumcircle of any triangle.

    Used for unstructured mesh generation, natural neighbour interpolation,
    and Voronoi diagram construction (dual graph).
    """

    def __init__(self) -> None:
        self.points: List[Tuple[float, float]] = []
        self.triangles: List[Tuple[int, int, int]] = []

    @staticmethod
    def circumcircle(p1: Tuple[float, float], p2: Tuple[float, float],
                       p3: Tuple[float, float]) -> Tuple[float, float, float]:
        """Circumcircle centre (cx, cy) and radius² for triangle (p1, p2, p3)."""
        ax, ay = p1
        bx, by = p2
        cx, cy = p3
        D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(D) < 1e-12:
            return (0.0, 0.0, float('inf'))
        ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay)
              + (cx**2 + cy**2) * (ay - by)) / D
        uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx)
              + (cx**2 + cy**2) * (bx - ax)) / D
        r2 = (ax - ux)**2 + (ay - uy)**2
        return (ux, uy, r2)

    def triangulate(self, points: NDArray) -> List[Tuple[int, int, int]]:
        """Bowyer-Watson Delaunay triangulation.

        points: (N, 2) array.
        Returns list of triangle index tuples.
        """
        self.points = [(float(p[0]), float(p[1])) for p in points]
        N = len(self.points)

        # Super-triangle
        M = max(abs(p[0]) + abs(p[1]) for p in self.points) + 1
        st = [(-3 * M, -3 * M), (3 * M, -3 * M), (0, 3 * M)]
        for s in st:
            self.points.append(s)

        self.triangles = [(N, N + 1, N + 2)]

        for i in range(N):
            px, py = self.points[i]
            bad_triangles = []
            for tri in self.triangles:
                p1, p2, p3 = self.points[tri[0]], self.points[tri[1]], self.points[tri[2]]
                cx, cy, r2 = self.circumcircle(p1, p2, p3)
                if (px - cx)**2 + (py - cy)**2 < r2:
                    bad_triangles.append(tri)

            # Find boundary polygon
            edges: List[Tuple[int, int]] = []
            for tri in bad_triangles:
                for e in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
                    shared = False
                    for other in bad_triangles:
                        if other == tri:
                            continue
                        if e[0] in other and e[1] in other:
                            shared = True
                            break
                    if not shared:
                        edges.append(e)

            for tri in bad_triangles:
                self.triangles.remove(tri)

            for e in edges:
                self.triangles.append((i, e[0], e[1]))

        # Remove triangles with super-triangle vertices
        self.triangles = [t for t in self.triangles
                           if t[0] < N and t[1] < N and t[2] < N]
        self.points = self.points[:N]
        return self.triangles

    def mesh_quality(self) -> float:
        """Average triangle quality (ratio of circumradius to inradius)."""
        ratios = []
        for tri in self.triangles:
            p = [self.points[tri[j]] for j in range(3)]
            a = math.sqrt((p[1][0] - p[0][0])**2 + (p[1][1] - p[0][1])**2)
            b = math.sqrt((p[2][0] - p[1][0])**2 + (p[2][1] - p[1][1])**2)
            c = math.sqrt((p[0][0] - p[2][0])**2 + (p[0][1] - p[2][1])**2)
            s = (a + b + c) / 2
            area = math.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))
            if area > 1e-15:
                R = a * b * c / (4 * area)
                r = area / s
                ratios.append(r / R)  # ideal equilateral: r/R = 0.5
        return float(np.mean(ratios)) if ratios else 0.0


# ---------------------------------------------------------------------------
#  h-Adaptivity Error Estimator
# ---------------------------------------------------------------------------

class HAdaptivityEstimator:
    r"""
    h-adaptivity: local mesh refinement based on error indicators.

    Gradient-based estimator:
    $$\eta_K = h_K^{p+1}|\nabla^{p+1} u|_K$$

    Zienkiewicz-Zhu (ZZ) stress recovery estimator:
    $$\eta_K = \|(\sigma^* - \sigma^h)\|_K$$

    Dörfler marking: refine cells contributing to fraction θ of total error.
    """

    def __init__(self, threshold: float = 0.3) -> None:
        self.threshold = threshold  # Dörfler bulk parameter

    def gradient_indicator(self, u: NDArray, dx: NDArray) -> NDArray:
        """Gradient-jump error indicator: η_i = h|∇u_i − ∇u_{i-1}|."""
        grad_u = np.diff(u) / dx
        jump = np.abs(np.diff(grad_u))
        h = 0.5 * (dx[:-1] + dx[1:])
        return h * jump

    def doerfler_marking(self, eta: NDArray) -> NDArray:
        """Dörfler bulk marking strategy.

        Mark smallest set of cells whose error exceeds θ·total.
        Returns boolean mask of cells to refine.
        """
        total_error = np.sum(eta**2)
        sorted_idx = np.argsort(-eta**2)

        cumsum = 0.0
        mark = np.zeros(len(eta), dtype=bool)
        for idx in sorted_idx:
            mark[idx] = True
            cumsum += eta[idx]**2
            if cumsum >= self.threshold * total_error:
                break
        return mark

    def maximum_strategy(self, eta: NDArray, fraction: float = 0.5) -> NDArray:
        """Refine all cells with η > fraction * max(η)."""
        threshold = fraction * np.max(eta)
        return eta > threshold
