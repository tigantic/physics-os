"""Surface mesh ingestion — OBJ / STL / PLY loading and normalization."""

from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

from ..core.types import SurfaceMesh

logger = logging.getLogger(__name__)


class SurfaceIngester:
    """Load surface meshes from common file formats.

    Supported: OBJ, STL (binary + ASCII), PLY (ASCII + binary LE).
    All outputs are standardized SurfaceMesh with float32 vertices
    and int32 face indices.
    """

    def ingest(self, path: str | Path) -> SurfaceMesh:
        """Load a surface mesh from file.

        Parameters
        ----------
        path : str or Path
            Path to OBJ, STL, or PLY file.

        Returns
        -------
        SurfaceMesh with vertices (N,3) and faces (F,3).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Mesh file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".obj":
            verts, faces = self._load_obj(path)
        elif suffix == ".stl":
            verts, faces = self._load_stl(path)
        elif suffix == ".ply":
            verts, faces = self._load_ply(path)
        else:
            raise ValueError(f"Unsupported mesh format: {suffix}")

        mesh = SurfaceMesh(vertices=verts, triangles=faces)
        mesh.compute_normals()

        logger.info(
            "Loaded %s: %d vertices, %d faces, area=%.1f mm²",
            path.name, len(verts), len(faces), mesh.surface_area_mm2(),
        )
        return mesh

    # ── OBJ ───────────────────────────────────────────────────

    @staticmethod
    def _load_obj(path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Parse Wavefront OBJ file."""
        verts = []
        faces = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("v "):
                    parts = line.split()
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith("f "):
                    parts = line.split()[1:]
                    # Handle face formats: "v", "v/vt", "v/vt/vn", "v//vn"
                    indices = []
                    for part in parts:
                        idx = int(part.split("/")[0])
                        indices.append(idx - 1)  # OBJ is 1-indexed
                    # Triangulate polygons via fan triangulation
                    for i in range(1, len(indices) - 1):
                        faces.append([indices[0], indices[i], indices[i + 1]])

        if not verts:
            raise ValueError(f"No vertices found in {path}")
        if not faces:
            raise ValueError(f"No faces found in {path}")

        return (
            np.array(verts, dtype=np.float32),
            np.array(faces, dtype=np.int32),
        )

    # ── STL ───────────────────────────────────────────────────

    @staticmethod
    def _load_stl(path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Parse STL file (auto-detect binary vs ASCII)."""
        with open(path, "rb") as f:
            header = f.read(80)
            # Check if ASCII
            try:
                header_str = header.decode("ascii")
                if header_str.strip().lower().startswith("solid"):
                    # Might be ASCII, verify by checking next line
                    next_line = f.readline()
                    try:
                        next_str = next_line.decode("ascii").strip()
                        if next_str.startswith("facet") or next_str == "":
                            return SurfaceIngester._load_stl_ascii(path)
                    except UnicodeDecodeError:
                        pass
            except UnicodeDecodeError:
                pass

        return SurfaceIngester._load_stl_binary(path)

    @staticmethod
    def _load_stl_binary(path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Parse binary STL."""
        with open(path, "rb") as f:
            f.read(80)  # header
            n_triangles = struct.unpack("<I", f.read(4))[0]

            if n_triangles == 0:
                raise ValueError(f"Empty STL file: {path}")

            # Each triangle: 12 floats (normal + 3 vertices) + 2 bytes attribute
            verts = np.empty((n_triangles * 3, 3), dtype=np.float32)
            faces = np.empty((n_triangles, 3), dtype=np.int32)

            for i in range(n_triangles):
                data = f.read(50)  # 12*4 + 2
                if len(data) < 50:
                    break
                vals = struct.unpack("<12fH", data)
                # Skip normal (vals[0:3]), read vertices
                verts[i * 3] = vals[3:6]
                verts[i * 3 + 1] = vals[6:9]
                verts[i * 3 + 2] = vals[9:12]
                faces[i] = [i * 3, i * 3 + 1, i * 3 + 2]

        # Merge duplicate vertices
        verts, faces = _merge_vertices(verts, faces)
        return verts, faces

    @staticmethod
    def _load_stl_ascii(path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Parse ASCII STL."""
        verts: list[list[float]] = []
        faces: list[list[int]] = []
        with open(path, "r") as f:
            tri_verts = []
            for line in f:
                line = line.strip()
                if line.startswith("vertex"):
                    parts = line.split()
                    tri_verts.append(
                        [float(parts[1]), float(parts[2]), float(parts[3])]
                    )
                    if len(tri_verts) == 3:
                        base = len(verts)
                        verts.extend(tri_verts)
                        faces.append([base, base + 1, base + 2])
                        tri_verts = []

        if not verts:
            raise ValueError(f"No vertices in ASCII STL: {path}")

        v = np.array(verts, dtype=np.float32)
        f_arr = np.array(faces, dtype=np.int32)
        v, f_arr = _merge_vertices(v, f_arr)
        return v, f_arr

    # ── PLY ───────────────────────────────────────────────────

    @staticmethod
    def _load_ply(path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Parse PLY file (ASCII or binary little-endian)."""
        with open(path, "rb") as f:
            # Parse header
            header_lines = []
            while True:
                line = f.readline().decode("ascii", errors="replace").strip()
                header_lines.append(line)
                if line == "end_header":
                    break

            n_verts = 0
            n_faces = 0
            fmt = "ascii"
            for line in header_lines:
                if line.startswith("element vertex"):
                    n_verts = int(line.split()[-1])
                elif line.startswith("element face"):
                    n_faces = int(line.split()[-1])
                elif line.startswith("format"):
                    parts = line.split()
                    fmt = parts[1]

            if n_verts == 0:
                raise ValueError(f"PLY has 0 vertices: {path}")

            if fmt == "ascii":
                return SurfaceIngester._load_ply_ascii(f, n_verts, n_faces)
            elif fmt == "binary_little_endian":
                return SurfaceIngester._load_ply_binary_le(f, n_verts, n_faces)
            else:
                raise ValueError(f"Unsupported PLY format: {fmt}")

    @staticmethod
    def _load_ply_ascii(f: Any, n_verts: int, n_faces: int) -> Tuple[np.ndarray, np.ndarray]:
        """Parse PLY ASCII data section."""
        verts = np.empty((n_verts, 3), dtype=np.float32)
        for i in range(n_verts):
            parts = f.readline().decode("ascii").split()
            verts[i] = [float(parts[0]), float(parts[1]), float(parts[2])]

        faces = []
        for _ in range(n_faces):
            parts = f.readline().decode("ascii").split()
            n = int(parts[0])
            indices = [int(parts[j + 1]) for j in range(n)]
            # Fan triangulate
            for j in range(1, n - 1):
                faces.append([indices[0], indices[j], indices[j + 1]])

        return verts, np.array(faces, dtype=np.int32) if faces else np.empty((0, 3), dtype=np.int32)

    @staticmethod
    def _load_ply_binary_le(f: Any, n_verts: int, n_faces: int) -> Tuple[np.ndarray, np.ndarray]:
        """Parse PLY binary little-endian data section."""
        # Assume vertices are 3 floats (most common case)
        vert_data = f.read(n_verts * 12)
        verts = np.frombuffer(vert_data, dtype=np.float32).reshape(n_verts, 3).copy()

        faces = []
        for _ in range(n_faces):
            n_bytes = f.read(1)
            if not n_bytes:
                break
            n = struct.unpack("B", n_bytes)[0]
            idx_data = f.read(n * 4)
            indices = list(struct.unpack(f"<{n}i", idx_data))
            for j in range(1, n - 1):
                faces.append([indices[0], indices[j], indices[j + 1]])

        return verts, np.array(faces, dtype=np.int32) if faces else np.empty((0, 3), dtype=np.int32)


# ── Utilities ─────────────────────────────────────────────────────

def _merge_vertices(
    verts: np.ndarray,
    faces: np.ndarray,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge duplicate vertices within tolerance.

    Uses grid-based hashing for O(N) performance on typical meshes.
    """
    n = len(verts)
    if n == 0:
        return verts, faces

    # Quantize to grid
    scale = 1.0 / max(tol, 1e-12)
    quantized = (verts * scale).astype(np.int64)

    # Hash each vertex to a canonical index
    seen: dict = {}
    remap = np.empty(n, dtype=np.int32)
    unique_verts = []
    uid = 0

    for i in range(n):
        key = (quantized[i, 0], quantized[i, 1], quantized[i, 2])
        if key in seen:
            remap[i] = seen[key]
        else:
            seen[key] = uid
            remap[i] = uid
            unique_verts.append(verts[i])
            uid += 1

    new_verts = np.array(unique_verts, dtype=np.float32)
    new_faces = remap[faces]

    # Remove degenerate faces
    valid = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 0] != new_faces[:, 2])
    )
    new_faces = new_faces[valid]

    return new_verts, new_faces
