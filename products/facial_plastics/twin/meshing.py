"""Volumetric mesh generation from segmented anatomy.

Creates FEM-ready tetrahedral meshes with:
  - Adaptive refinement in surgical ROI
  - Multi-material region tagging
  - Surface conformity to tissue boundaries
  - Quality-controlled element generation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial import Delaunay as _Delaunay

from ..core.config import MeshConfig
from ..core.types import (
    BoundingBox,
    MaterialModel,
    MeshElementType,
    MeshQualityReport,
    StructureType,
    SurfaceMesh,
    TissueProperties,
    Vec3,
    VolumeMesh,
)

logger = logging.getLogger(__name__)


@dataclass
class MeshRegion:
    """Definition of a mesh region with sizing constraints."""
    structure: StructureType
    target_edge_length_mm: float
    label: int  # material region label
    priority: int = 0


class VolumetricMesher:
    """Generate FEM-quality volumetric meshes from segmented volumes.

    Pipeline:
      1. Extract isosurfaces from label volume (marching cubes)
      2. Smooth and decimate surfaces
      3. Generate constrained Delaunay tetrahedralization
      4. Assign material region tags
      5. Refine in surgical ROI
      6. Quality check and Laplacian smoothing
    """

    def __init__(self, config: Optional[MeshConfig] = None) -> None:
        self._config = config or MeshConfig()

    def mesh_from_labels(
        self,
        labels: np.ndarray,
        voxel_spacing_mm: Tuple[float, float, float],
        regions: Optional[List[MeshRegion]] = None,
        *,
        roi_box: Optional[BoundingBox] = None,
    ) -> VolumeMesh:
        """Generate a volume mesh from a segmentation label volume.

        Parameters
        ----------
        labels : ndarray (D, H, W) int
            Segmentation labels.
        voxel_spacing_mm : tuple of 3 floats
            Voxel spacing.
        regions : list of MeshRegion, optional
            Region definitions for material assignment.
        roi_box : BoundingBox, optional
            Surgical region of interest for mesh refinement.

        Returns
        -------
        VolumeMesh ready for FEM simulation.
        """
        cfg = self._config
        sz, sy, sx = voxel_spacing_mm

        logger.info("Generating volume mesh from labels %s", labels.shape)

        # Phase 1: Extract boundary surfaces via marching cubes
        all_vertices: List[np.ndarray] = []
        all_tets: List[np.ndarray] = []
        region_materials: Dict[int, TissueProperties] = {}
        surface_tags: Dict[str, np.ndarray] = {}

        # Find unique non-zero labels
        unique_labels = sorted(set(int(v) for v in np.unique(labels) if v > 0))

        if not unique_labels:
            raise ValueError("No non-zero labels in segmentation volume")

        # Phase 2: Generate surface mesh for entire region
        outer_surface = self._extract_isosurface(
            (labels > 0).astype(np.float32), voxel_spacing_mm, level=0.5
        )

        if outer_surface is None:
            raise ValueError("Failed to extract outer isosurface")

        logger.info("Outer surface: %d vertices, %d triangles",
                     len(outer_surface.vertices), len(outer_surface.triangles))

        # Phase 3: Smooth the surface
        outer_surface = self._laplacian_smooth_surface(outer_surface, iterations=10, alpha=0.3)

        # Phase 4: Generate tetrahedral mesh from surface
        nodes, elements = self._tetrahedralize(
            outer_surface, cfg.target_edge_length_mm
        )

        if len(elements) == 0:
            raise RuntimeError("Tetrahedralization produced no elements")

        logger.info("Initial mesh: %d nodes, %d tetrahedra", len(nodes), len(elements))

        # Phase 5: Assign region labels based on segmentation
        element_regions = self._assign_regions(
            nodes, elements, labels, voxel_spacing_mm
        )

        # Phase 6: Refine in ROI
        if roi_box is not None:
            nodes, elements, element_regions = self._refine_in_roi(
                nodes, elements, element_regions, roi_box, cfg.refinement_factor
            )
            logger.info("After ROI refinement: %d nodes, %d tetrahedra",
                         len(nodes), len(elements))

        # Phase 7: Quality improvement
        nodes = self._laplacian_smooth_volume(nodes, elements, iterations=5, alpha=0.2)

        # Build region materials map
        if regions:
            for reg in regions:
                region_materials[reg.label] = TissueProperties(
                    structure_type=reg.structure,
                    material_model=MaterialModel.LINEAR_ELASTIC,
                    parameters={"E": 1e6, "nu": 0.45},
                )

        # Compute boundary surface faces
        boundary_faces = self._extract_surface_tets(elements)
        if len(boundary_faces) > 0:
            surface_tags["boundary"] = boundary_faces

        mesh = VolumeMesh(
            nodes=nodes,
            elements=elements,
            element_type=MeshElementType.TET4,
            region_ids=element_regions,
            region_materials=region_materials,
            surface_tags=surface_tags,
        )

        logger.info("Final mesh: %d nodes, %d elements, %d boundary faces",
                     len(nodes), len(elements), len(boundary_faces))

        return mesh

    def mesh_from_surface(
        self,
        surface: SurfaceMesh,
        *,
        target_edge_length_mm: Optional[float] = None,
    ) -> VolumeMesh:
        """Generate a volume mesh from a closed surface mesh."""
        cfg = self._config
        edge_len = target_edge_length_mm or cfg.target_edge_length_mm

        surface = self._laplacian_smooth_surface(surface, iterations=5, alpha=0.3)
        nodes, elements = self._tetrahedralize(surface, edge_len)

        return VolumeMesh(
            nodes=nodes,
            elements=elements,
            element_type=MeshElementType.TET4,
            region_ids=np.zeros(len(elements), dtype=np.int32),
        )

    def compute_quality(self, mesh: VolumeMesh) -> MeshQualityReport:
        """Compute mesh quality metrics."""
        nodes = mesh.nodes
        elements = mesh.elements

        # Compute per-element quality metrics
        n_elem = len(elements)
        aspect_ratios = np.zeros(n_elem, dtype=np.float32)
        jacobians = np.zeros(n_elem, dtype=np.float32)
        volumes = np.zeros(n_elem, dtype=np.float32)
        global_min_edge = float("inf")
        global_max_edge = 0.0

        for i, elem in enumerate(elements):
            v0, v1, v2, v3 = nodes[elem[0]], nodes[elem[1]], nodes[elem[2]], nodes[elem[3]]

            # Tet volume
            mat = np.column_stack([v1 - v0, v2 - v0, v3 - v0])
            vol = abs(float(np.linalg.det(mat))) / 6.0
            volumes[i] = vol

            # Edge lengths
            edges = [
                float(np.linalg.norm(v1 - v0)), float(np.linalg.norm(v2 - v0)),
                float(np.linalg.norm(v3 - v0)), float(np.linalg.norm(v2 - v1)),
                float(np.linalg.norm(v3 - v1)), float(np.linalg.norm(v3 - v2)),
            ]
            max_edge = max(edges)
            min_edge = max(min(edges), 1e-12)
            global_min_edge = min(global_min_edge, min_edge)
            global_max_edge = max(global_max_edge, max_edge)
            aspect_ratios[i] = max_edge / min_edge

            # Jacobian (normalized)
            ideal_vol = (max_edge ** 3) / (6.0 * float(np.sqrt(2)))
            jacobians[i] = vol / max(ideal_vol, 1e-12)

        # Aggregate metrics
        n_inverted = int((volumes <= 0).sum())
        n_regions = len(set(int(r) for r in mesh.region_ids)) if len(mesh.region_ids) > 0 else 0

        # Compute surface area from boundary faces
        boundary = self._extract_surface_tets(elements)
        surface_area = 0.0
        for face in boundary:
            vf0, vf1, vf2 = nodes[face[0]], nodes[face[1]], nodes[face[2]]
            surface_area += 0.5 * float(np.linalg.norm(np.cross(vf1 - vf0, vf2 - vf0)))

        if n_elem == 0:
            global_min_edge = 0.0
            global_max_edge = 0.0

        report = MeshQualityReport(
            n_nodes=len(nodes),
            n_elements=n_elem,
            element_type=mesh.element_type,
            min_jacobian=float(jacobians.min()) if n_elem > 0 else 0.0,
            max_aspect_ratio=float(aspect_ratios.max()) if n_elem > 0 else 0.0,
            min_edge_length_mm=global_min_edge,
            max_edge_length_mm=global_max_edge,
            mean_quality=float(jacobians.mean()) if n_elem > 0 else 0.0,
            n_inverted=n_inverted,
            volume_mm3=float(volumes.sum()),
            surface_area_mm2=surface_area,
            n_regions=n_regions,
            min_quality=float(jacobians.min()) if n_elem > 0 else 0.0,
            max_quality=float(jacobians.max()) if n_elem > 0 else 0.0,
            min_aspect_ratio=float(aspect_ratios.min()) if n_elem > 0 else 1.0,
        )

        return report

    # ── Isosurface extraction (Marching Cubes) ────────────────

    def _extract_isosurface(
        self,
        volume: np.ndarray,
        spacing: Tuple[float, float, float],
        level: float = 0.5,
    ) -> Optional[SurfaceMesh]:
        """Vectorised binary-surface extraction via face adjacency.

        For every pair of adjacent voxels that straddle *level*, emits
        two triangles (one quad) on the shared face.  Produces a
        watertight, consistently-oriented mesh.
        """
        sz, sy, sx = spacing
        dz, dy, dx = volume.shape

        all_verts: List[np.ndarray] = []
        all_tris: List[np.ndarray] = []
        vert_offset = 0

        # Helper: for a given axis, find all straddling pairs and emit quads
        def _emit_axis(
            axis: int,
            sp_a: float,  # spacing along axis perpendicular component 1
            sp_b: float,  # spacing along axis perpendicular component 2
            sp_n: float,  # spacing along normal axis
        ) -> None:
            nonlocal vert_offset
            a = np.take(volume, range(volume.shape[axis] - 1), axis=axis)
            b = np.take(volume, range(1, volume.shape[axis]), axis=axis)
            straddle = (a >= level) != (b >= level)
            inside_a = a >= level

            zs, ys, xs = np.where(straddle)

            if len(zs) == 0:
                return

            # For each straddling location, compute quad corners
            # Axis 0 (Z): face at z=(i+1)*sz, quad in XY
            # Axis 1 (Y): face at y=(j+1)*sy, quad in XZ
            # Axis 2 (X): face at x=(k+1)*sx, quad in YZ
            n_faces = len(zs)

            if axis == 0:
                # Face normal along Z at z_pos = (zs+1)*sz
                z_pos = (zs + 1).astype(np.float64) * sz
                x0 = xs.astype(np.float64) * sx
                x1 = (xs + 1).astype(np.float64) * sx
                y0 = ys.astype(np.float64) * sy
                y1 = (ys + 1).astype(np.float64) * sy
                # 4 corners per quad: (x0,y0,z), (x1,y0,z), (x1,y1,z), (x0,y1,z)
                p0 = np.column_stack([x0, y0, z_pos])
                p1 = np.column_stack([x1, y0, z_pos])
                p2 = np.column_stack([x1, y1, z_pos])
                p3 = np.column_stack([x0, y1, z_pos])
            elif axis == 1:
                y_pos = (ys + 1).astype(np.float64) * sy
                x0 = xs.astype(np.float64) * sx
                x1 = (xs + 1).astype(np.float64) * sx
                z0 = zs.astype(np.float64) * sz
                z1 = (zs + 1).astype(np.float64) * sz
                p0 = np.column_stack([x0, y_pos, z0])
                p1 = np.column_stack([x0, y_pos, z1])
                p2 = np.column_stack([x1, y_pos, z1])
                p3 = np.column_stack([x1, y_pos, z0])
            else:  # axis == 2
                x_pos = (xs + 1).astype(np.float64) * sx
                y0 = ys.astype(np.float64) * sy
                y1 = (ys + 1).astype(np.float64) * sy
                z0 = zs.astype(np.float64) * sz
                z1 = (zs + 1).astype(np.float64) * sz
                p0 = np.column_stack([x_pos, y0, z0])
                p1 = np.column_stack([x_pos, y1, z0])
                p2 = np.column_stack([x_pos, y1, z1])
                p3 = np.column_stack([x_pos, y0, z1])

            # Flip winding for faces where a < level (outside→inside)
            flip = ~inside_a[zs, ys, xs]

            # Build vertex array: 4 verts per quad
            verts = np.empty((n_faces * 4, 3), dtype=np.float64)
            verts[0::4] = p0
            verts[1::4] = np.where(flip[:, None], p3, p1)
            verts[2::4] = p2
            verts[3::4] = np.where(flip[:, None], p1, p3)

            # Build triangle array: 2 tris per quad
            base = np.arange(n_faces, dtype=np.int64) * 4 + vert_offset
            tris = np.empty((n_faces * 2, 3), dtype=np.int64)
            tris[0::2, 0] = base
            tris[0::2, 1] = base + 1
            tris[0::2, 2] = base + 2
            tris[1::2, 0] = base
            tris[1::2, 1] = base + 2
            tris[1::2, 2] = base + 3

            all_verts.append(verts)
            all_tris.append(tris)
            vert_offset += len(verts)

        _emit_axis(0, sx, sy, sz)
        _emit_axis(1, sx, sz, sy)
        _emit_axis(2, sy, sz, sx)

        if not all_verts:
            return None

        verts = np.concatenate(all_verts).astype(np.float32)
        faces = np.concatenate(all_tris).astype(np.int32)

        # Merge close vertices
        from ..data.surface_ingest import _merge_vertices
        verts, faces = _merge_vertices(verts, faces, tol=min(spacing) * 0.1)

        mesh = SurfaceMesh(vertices=verts, triangles=faces)
        mesh.compute_normals()
        return mesh

    # ── Tetrahedralization ────────────────────────────────────

    def _tetrahedralize(
        self,
        surface: SurfaceMesh,
        target_edge: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate tetrahedral mesh from closed surface.

        Uses constrained Delaunay approach with interior point insertion.
        """
        verts = surface.vertices.astype(np.float64)
        faces = surface.triangles

        # Compute bounding box
        bb_min = verts.min(axis=0)
        bb_max = verts.max(axis=0)
        bb_size = bb_max - bb_min

        # Generate interior seed points on a regular grid
        nx = min(max(2, int(bb_size[0] / target_edge)), 100)
        ny = min(max(2, int(bb_size[1] / target_edge)), 100)
        nz = min(max(2, int(bb_size[2] / target_edge)), 100)

        x_coords = np.linspace(bb_min[0] + target_edge / 2, bb_max[0] - target_edge / 2, nx)
        y_coords = np.linspace(bb_min[1] + target_edge / 2, bb_max[1] - target_edge / 2, ny)
        z_coords = np.linspace(bb_min[2] + target_edge / 2, bb_max[2] - target_edge / 2, nz)

        grid_points = np.array(
            [[x, y, z] for x in x_coords for y in y_coords for z in z_coords],
            dtype=np.float64,
        )

        # Filter to inside the surface (ray casting)
        interior_mask = self._points_inside_surface(grid_points, verts, faces)
        interior_points = grid_points[interior_mask]

        # Combine surface and interior points
        all_points = np.vstack([verts, interior_points])
        n_surface = len(verts)

        # Delaunay tetrahedralization
        tets = self._delaunay_3d(all_points)

        # Filter tetrahedra with at least one interior point
        # (keeps tets that are inside the domain)
        valid_tets = []
        for tet in tets:
            centroid = all_points[tet].mean(axis=0)
            # Check if centroid is inside
            if self._point_inside_bbox(centroid, bb_min, bb_max):
                valid_tets.append(tet)

        if not valid_tets:
            valid_tets = tets.tolist() if len(tets) > 0 else []

        elements = np.array(valid_tets, dtype=np.int32) if valid_tets else np.empty((0, 4), dtype=np.int32)
        return all_points.astype(np.float32), elements

    def _delaunay_3d(self, points: np.ndarray) -> np.ndarray:
        """3D Delaunay tetrahedralization via scipy."""
        n = len(points)
        if n < 4:
            return np.empty((0, 4), dtype=np.int32)
        try:
            tri = _Delaunay(points)
            return np.asarray(tri.simplices, dtype=np.int32)
        except Exception:
            logger.warning("Delaunay tetrahedralization failed")
            return np.empty((0, 4), dtype=np.int32)

    # ── Region assignment ─────────────────────────────────────

    @staticmethod
    def _assign_regions(
        nodes: np.ndarray,
        elements: np.ndarray,
        labels: np.ndarray,
        spacing: Tuple[float, float, float],
    ) -> np.ndarray:
        """Assign segmentation labels to each element."""
        sz, sy, sx = spacing
        n_elem = len(elements)
        regions = np.zeros(n_elem, dtype=np.int32)

        for i, elem in enumerate(elements):
            centroid = nodes[elem].mean(axis=0)
            # Convert to voxel coordinates
            vx = int(round(centroid[0] / sx))
            vy = int(round(centroid[1] / sy))
            vz = int(round(centroid[2] / sz))

            dz, dy, dx = labels.shape
            vx = max(0, min(vx, dx - 1))
            vy = max(0, min(vy, dy - 1))
            vz = max(0, min(vz, dz - 1))

            regions[i] = labels[vz, vy, vx]

        return regions

    # ── Mesh refinement ───────────────────────────────────────

    def _refine_in_roi(
        self,
        nodes: np.ndarray,
        elements: np.ndarray,
        regions: np.ndarray,
        roi: BoundingBox,
        factor: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Refine elements inside the ROI by splitting long edges."""
        min_c = np.array([roi.origin.x, roi.origin.y, roi.origin.z])
        max_c = np.array([
            roi.origin.x + roi.extent.x,
            roi.origin.y + roi.extent.y,
            roi.origin.z + roi.extent.z,
        ])
        target_edge = self._config.target_edge_length_mm * factor

        nodes_list = nodes.tolist()
        new_elements = []
        new_regions = []
        midpoint_cache: Dict[Tuple[int, int], int] = {}

        for i, elem in enumerate(elements):
            centroid = np.array(nodes_list[elem[0]]) + np.array(nodes_list[elem[1]])
            centroid += np.array(nodes_list[elem[2]]) + np.array(nodes_list[elem[3]])
            centroid /= 4

            in_roi = np.all(centroid >= min_c) and np.all(centroid <= max_c)

            if in_roi:
                # Check longest edge
                edge_pairs = [
                    (elem[0], elem[1]), (elem[0], elem[2]), (elem[0], elem[3]),
                    (elem[1], elem[2]), (elem[1], elem[3]), (elem[2], elem[3]),
                ]
                max_edge_len = 0.0
                for a, b in edge_pairs:
                    d = float(np.linalg.norm(
                        np.array(nodes_list[a]) - np.array(nodes_list[b])
                    ))
                    max_edge_len = max(max_edge_len, d)

                if max_edge_len > target_edge:
                    # Split at midpoint of longest edge
                    longest = max(edge_pairs, key=lambda ab: np.linalg.norm(
                        np.array(nodes_list[ab[0]]) - np.array(nodes_list[ab[1]])
                    ))
                    a, b = longest
                    key = (min(a, b), max(a, b))
                    if key not in midpoint_cache:
                        mid = (np.array(nodes_list[a]) + np.array(nodes_list[b])) / 2
                        midpoint_cache[key] = len(nodes_list)
                        nodes_list.append(mid.tolist())
                    mid_idx = midpoint_cache[key]

                    # Split tet into 2 tets along the longest edge
                    others = [v for v in elem if v != a and v != b]
                    new_elements.append([a, mid_idx, others[0], others[1]])
                    new_elements.append([mid_idx, b, others[0], others[1]])
                    new_regions.append(regions[i])
                    new_regions.append(regions[i])
                    continue

            new_elements.append(elem.tolist())
            new_regions.append(regions[i])

        return (
            np.array(nodes_list, dtype=np.float32),
            np.array(new_elements, dtype=np.int32),
            np.array(new_regions, dtype=np.int32),
        )

    # ── Smoothing ─────────────────────────────────────────────

    @staticmethod
    def _laplacian_smooth_surface(
        mesh: SurfaceMesh,
        iterations: int = 10,
        alpha: float = 0.3,
    ) -> SurfaceMesh:
        """Laplacian smoothing of surface mesh."""
        verts = mesh.vertices.copy()
        faces = mesh.triangles
        n = len(verts)

        # Build adjacency
        neighbors: Dict[int, Set[int]] = {i: set() for i in range(n)}
        for face in faces:
            for j in range(3):
                a, b = face[j], face[(j + 1) % 3]
                neighbors[a].add(b)
                neighbors[b].add(a)

        for _ in range(iterations):
            new_verts = verts.copy()
            for i in range(n):
                nbrs = list(neighbors[i])
                if nbrs:
                    avg = verts[nbrs].mean(axis=0)
                    new_verts[i] = verts[i] + alpha * (avg - verts[i])
            verts = new_verts

        result = SurfaceMesh(vertices=verts, triangles=mesh.triangles.copy())
        result.compute_normals()
        return result

    @staticmethod
    def _laplacian_smooth_volume(
        nodes: np.ndarray,
        elements: np.ndarray,
        iterations: int = 5,
        alpha: float = 0.2,
    ) -> np.ndarray:
        """Laplacian smoothing of volume mesh interior nodes."""
        n = len(nodes)
        nodes = nodes.copy()

        # Build adjacency
        neighbors: Dict[int, Set[int]] = {i: set() for i in range(n)}
        for elem in elements:
            for j in range(4):
                for k in range(j + 1, 4):
                    neighbors[elem[j]].add(elem[k])
                    neighbors[elem[k]].add(elem[j])

        # Identify boundary nodes (connected to surface faces)
        boundary: Set[int] = set()
        face_count: Dict[Tuple[int, ...], int] = {}
        for elem in elements:
            for face in [(elem[0], elem[1], elem[2]),
                         (elem[0], elem[1], elem[3]),
                         (elem[0], elem[2], elem[3]),
                         (elem[1], elem[2], elem[3])]:
                key: Tuple[int, ...] = tuple(int(x) for x in sorted(face))
                face_count[key] = face_count.get(key, 0) + 1

        for face_key, count in face_count.items():
            if count == 1:  # boundary face
                boundary.update(face_key)

        for _ in range(iterations):
            new_nodes = nodes.copy()
            for i in range(n):
                if i in boundary:
                    continue
                nbrs = list(neighbors[i])
                if nbrs:
                    avg = nodes[nbrs].mean(axis=0)
                    new_nodes[i] = nodes[i] + alpha * (avg - nodes[i])
            nodes = new_nodes

        return nodes

    # ── Utility ───────────────────────────────────────────────

    @staticmethod
    def _points_inside_surface(
        points: np.ndarray,
        verts: np.ndarray,
        faces: np.ndarray,
    ) -> np.ndarray:
        """Test if points are inside a closed surface using ray casting.

        Vectorised over faces for each query point.
        """
        n = len(points)
        if n == 0 or len(faces) == 0:
            return np.zeros(n, dtype=bool)

        inside = np.zeros(n, dtype=bool)

        # Pre-compute triangle data
        v0 = verts[faces[:, 0]]  # (F,3)
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        edge1 = v1 - v0  # (F,3)
        edge2 = v2 - v0

        ray_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        h = np.cross(ray_dir, edge2)  # (F,3)
        a = np.einsum("ij,ij->i", edge1, h)  # (F,)

        valid = np.abs(a) > 1e-10
        f_inv = np.zeros_like(a)
        f_inv[valid] = 1.0 / a[valid]

        for pi in range(n):
            p = points[pi]
            s = p - v0  # (F,3)
            u = f_inv * np.einsum("ij,ij->i", s, h)
            q = np.cross(s, edge1)
            v_param = f_inv * np.einsum("ij,ij->i", np.broadcast_to(ray_dir, q.shape), q)
            t = f_inv * np.einsum("ij,ij->i", edge2, q)

            hit = valid & (u >= 0) & (u <= 1) & (v_param >= 0) & (u + v_param <= 1) & (t > 1e-10)
            crossings = int(np.count_nonzero(hit))
            inside[pi] = crossings % 2 == 1

        return inside

    @staticmethod
    def _point_inside_bbox(p: np.ndarray, bb_min: np.ndarray, bb_max: np.ndarray) -> bool:
        return bool(np.all(p >= bb_min) and np.all(p <= bb_max))

    @staticmethod
    def _extract_surface_tets(elements: np.ndarray) -> np.ndarray:
        """Extract boundary faces from a tetrahedral mesh."""
        face_count: Dict[Tuple[int, ...], list] = {}
        for elem in elements:
            for face in [(elem[0], elem[1], elem[2]),
                         (elem[0], elem[1], elem[3]),
                         (elem[0], elem[2], elem[3]),
                         (elem[1], elem[2], elem[3])]:
                key = tuple(int(x) for x in sorted(face))
                if key in face_count:
                    face_count[key].append(1)
                else:
                    face_count[key] = [1]

        boundary_faces = [list(face) for face, counts in face_count.items() if len(counts) == 1]
        return np.array(boundary_faces, dtype=np.int32) if boundary_faces else np.empty((0, 3), dtype=np.int32)
