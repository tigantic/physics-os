"""MakeHuman CC0 face mesh generator for the Sovereign facial-plastics pipeline.

Loads the MakeHuman Community base mesh (CC0 licensed, public domain),
extracts the head/face region, triangulates quads, assigns anatomical
tissue regions, and outputs a ``SurfaceMesh`` compatible with the
existing rendering and simulation pipeline.

The generated mesh features real human face topology with:
- Proper orbital concavities
- Nasal bridge and dorsum
- Philtrum and vermillion border
- Alar lobule and columella
- Jaw line and mental protuberance

Coordinate system (Frankfurt Horizontal, millimetres):
    X : right (+) / left (-)
    Y : superior (+) / inferior (-)
    Z : anterior (+) / posterior (-)
    Origin ≈ subnasale

Tissue region IDs (matching existing pipeline):
    0  bone              (nasal bridge / glabella)
    1  cartilage_upper   (upper lateral cartilage)
    2  cartilage_lower   (lower lateral cartilage)
    3  cartilage_septal  (columellar / septal)
    4  skin_dorsal       (dorsal skin envelope)
    5  skin_tip          (tip / supratip skin)
    6  soft_tissue       (periorbital / buccal)
    7  skin_alar         (alar lobule skin)
    8  cartilage_columella (columella)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────
_BASE_OBJ = Path(__file__).parent / "data" / "makehuman" / "base.obj"

# MakeHuman coordinate landmarks (measured from base.obj)
_MH_SUBNASALE_Y = 6.56       # approximate subnasale Y in MH coords
_MH_SUBNASALE_Z = 1.55       # approximate subnasale Z in MH coords
_MH_HEAD_Y_CUTOFF = 5.80     # neck cutoff — includes full jaw/chin
_MH_SCALE = 65.0             # scale factor: MH units → mm (half-width ≈ 65 mm)

# Anatomical region IDs — must match pipeline constants
REGION_BONE = 0
REGION_CARTILAGE_UPPER = 1
REGION_CARTILAGE_LOWER = 2
REGION_CARTILAGE_SEPTAL = 3
REGION_SKIN_DORSAL = 4
REGION_SKIN_TIP = 5
REGION_SOFT_TISSUE = 6
REGION_SKIN_ALAR = 7
REGION_CARTILAGE_COLUMELLA = 8

REGION_NAMES = [
    "bone", "cartilage_upper", "cartilage_lower",
    "cartilage_septal", "skin_dorsal", "skin_tip",
    "soft_tissue", "skin_alar", "cartilage_columella",
]

# Cache for loaded mesh (module-level singleton)
_mesh_cache: Optional[Dict[str, Any]] = None


def _parse_obj(path: Path) -> Tuple[np.ndarray, List[List[int]], Dict[str, List[int]]]:
    """Parse a Wavefront OBJ file.

    Returns
    -------
    vertices : ndarray (N, 3)
    faces : list of lists (each face is a list of 0-indexed vertex IDs)
    groups : dict mapping group name → list of face indices
    """
    vertices: List[List[float]] = []
    faces: List[List[int]] = []
    groups: Dict[str, List[int]] = {}
    current_group = "default"

    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("g "):
                current_group = line[2:].strip()
            elif line.startswith("f "):
                tokens = line.split()[1:]
                face_verts = [int(t.split("/")[0]) - 1 for t in tokens]
                if current_group not in groups:
                    groups[current_group] = []
                groups[current_group].append(len(faces))
                faces.append(face_verts)

    return np.array(vertices, dtype=np.float64), faces, groups


def _triangulate_quads(
    faces: List[List[int]],
) -> np.ndarray:
    """Convert a list of polygon faces (tris or quads) to triangles.

    Returns ndarray (T, 3) of int32.
    """
    tris: List[List[int]] = []
    for face in faces:
        if len(face) == 3:
            tris.append(face)
        elif len(face) == 4:
            # Ear-clip: split quad ABCD → ABC, ACD
            tris.append([face[0], face[1], face[2]])
            tris.append([face[0], face[2], face[3]])
        else:
            # General polygon fan triangulation
            for i in range(1, len(face) - 1):
                tris.append([face[0], face[i], face[i + 1]])
    return np.array(tris, dtype=np.int32)


def _extract_head(
    vertices: np.ndarray,
    faces: List[List[int]],
    groups: Dict[str, List[int]],
    y_cutoff: float = _MH_HEAD_Y_CUTOFF,
) -> Tuple[np.ndarray, List[List[int]]]:
    """Extract the full head from the full-body mesh.

    Keeps all faces from the "body" group (plus eye/teeth/hair helpers)
    whose vertices are entirely above *y_cutoff* in MakeHuman Y
    coordinate (height axis).  No posterior or skull-cap crop — the
    complete head surface is preserved so there are no jagged
    boundary edges.

    Returns (vertices_Mx3, faces_list_of_lists_with_new_indices).
    """
    # Use only the "body" group — helper meshes (eyeballs, teeth,
    # tongue, eyelashes) are internal structures that create visual
    # artifacts (bulging spheres, protruding geometry) and interfere
    # with landmark detection.
    all_face_indices: List[int] = list(groups.get("body", []))

    source_faces = [faces[i] for i in all_face_indices]

    # Filter to head height: keep faces where ALL vertices are above cutoff
    head_vert_set = set(
        vi for vi in range(len(vertices)) if vertices[vi, 1] > y_cutoff
    )
    head_faces = [f for f in source_faces if all(vi in head_vert_set for vi in f)]

    # Collect actually-used vertices and re-index
    used_verts = sorted(set(vi for f in head_faces for vi in f))
    old_to_new = {old: new for new, old in enumerate(used_verts)}

    new_verts = vertices[used_verts]
    new_faces = [[old_to_new[vi] for vi in f] for f in head_faces]

    return new_verts, new_faces


def _transform_to_pipeline_coords(
    vertices: np.ndarray,
) -> np.ndarray:
    """Transform MakeHuman coordinates to pipeline coordinates.

    MakeHuman:  X=right, Y=up, Z=forward   (decimetres)
    Pipeline:   X=right, Y=up, Z=forward   (millimetres, origin at subnasale)

    Steps:
    1. Re-centre so subnasale is at origin
    2. Scale to millimetres
    """
    out = vertices.copy()

    # Re-centre on subnasale
    out[:, 1] -= _MH_SUBNASALE_Y
    out[:, 2] -= _MH_SUBNASALE_Z

    # Scale: measure current half-width and normalise to _MH_SCALE mm
    x_extent = max(abs(out[:, 0].min()), abs(out[:, 0].max()))
    if x_extent > 1e-6:
        scale = _MH_SCALE / x_extent
    else:
        scale = _MH_SCALE
    out *= scale

    return out


def _compute_normals(
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    """Compute per-vertex normals via face-area-weighted averaging."""
    normals = np.zeros_like(vertices)

    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    face_normals = np.cross(v1 - v0, v2 - v0)  # (T, 3), length = 2 × area

    # Accumulate face normals to vertices
    for axis in range(3):
        np.add.at(normals[:, axis], triangles[:, 0], face_normals[:, axis])
        np.add.at(normals[:, axis], triangles[:, 1], face_normals[:, axis])
        np.add.at(normals[:, axis], triangles[:, 2], face_normals[:, axis])

    # Normalise
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths < 1e-12] = 1.0
    normals /= lengths

    return normals


def _assign_regions(
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    """Assign anatomical tissue region IDs to each triangle.

    Uses spatial position of triangle centroids in pipeline coordinates
    (origin at subnasale, mm).  Calibrated against the MakeHuman base
    mesh where:
        Nose tip   ≈ (  0,  26,  10)
        Subnasale  ≈ (  0,   0,   1)
        Chin       ≈ (  0, -17,  -4)
        L-orbit    ≈ (-19,  45, -30)
        R-orbit    ≈ ( 19,  45, -30)

    Returns ndarray (T,) of int32 region IDs.
    """
    centroids = vertices[triangles].mean(axis=1)  # (T, 3)
    cx = centroids[:, 0]  # lateral
    cy = centroids[:, 1]  # vertical (+ = superior)
    cz = centroids[:, 2]  # AP (+ = anterior)

    abs_cx = np.abs(cx)
    n_tris = len(triangles)

    # Default: soft_tissue (periorbital, cheek, forehead, etc.)
    region_ids = np.full(n_tris, REGION_SOFT_TISSUE, dtype=np.int32)

    # Apply from BROADEST to MOST SPECIFIC so specific regions win.

    # ── 1. Skin dorsal — broad nasal skin envelope, Y=8-38 ──
    dorsal_mask = (
        (abs_cx < 15.0) &
        (cy > 8.0) & (cy < 38.0) &
        (cz > -5.0)
    )
    region_ids[dorsal_mask] = REGION_SKIN_DORSAL

    # ── 2. Nasal bone — bony vault, Y=28-42, midline ──
    bone_mask = (
        (abs_cx < 10.0) &
        (cy > 28.0) & (cy < 42.0) &
        (cz > -2.0)
    )
    region_ids[bone_mask] = REGION_BONE

    # ── 3. Upper lateral cartilage — cartilaginous vault Y=14-30 ──
    ulc_mask = (
        (abs_cx < 14.0) &
        (cy > 14.0) & (cy < 30.0) &
        (cz > 0.0)
    )
    region_ids[ulc_mask] = REGION_CARTILAGE_UPPER

    # ── 4. Lower lateral cartilage — caudal to ULC, Y=-2 to 16 ──
    llc_mask = (
        (abs_cx < 16.0) &
        (cy > -2.0) & (cy < 16.0) &
        (cz > 0.0)
    )
    region_ids[llc_mask] = REGION_CARTILAGE_LOWER

    # ── 5. Skin alar — lateral nostril wings, Y=-8 to 10 ──
    alar_mask = (
        (abs_cx > 3.0) & (abs_cx < 22.0) &
        (cy > -8.0) & (cy < 10.0) &
        (cz > -5.0)
    )
    region_ids[alar_mask] = REGION_SKIN_ALAR

    # ── 6. Cartilage septal — deep midline, Y=-5 to 18 ──
    septal_mask = (
        (abs_cx < 5.0) &
        (cy > -5.0) & (cy < 18.0) &
        (cz > -2.0)
    )
    region_ids[septal_mask] = REGION_CARTILAGE_SEPTAL

    # ── 7. Skin tip — nasal tip apex, most anterior, Y=15-30 ──
    tip_mask = (
        (abs_cx < 10.0) &
        (cy > 15.0) & (cy < 30.0) &
        (cz > 4.0)
    )
    region_ids[tip_mask] = REGION_SKIN_TIP

    # ── 8. Cartilage columella — midline inferior, Y=-10 to 2 ──
    columella_mask = (
        (abs_cx < 5.0) &
        (cy > -10.0) & (cy < 2.0) &
        (cz > -3.0)
    )
    region_ids[columella_mask] = REGION_CARTILAGE_COLUMELLA

    return region_ids


def load_makehuman_face(
    obj_path: Optional[Path] = None,
    *,
    seed: int = 0,
    scale_mm: float = _MH_SCALE,
) -> Dict[str, Any]:
    """Load and process the MakeHuman base mesh into a pipeline-ready face.

    Parameters
    ----------
    obj_path : Path, optional
        Path to base.obj.  Defaults to the bundled CC0 mesh.
    seed : int
        Seed for any stochastic morph variation (reserved for future use).
    scale_mm : float
        Target half-width in millimetres.

    Returns
    -------
    dict with keys:
        positions   : list of [x, y, z] — nested, pipeline coords (mm)
        indices     : list of [i, j, k] — triangle vertex indices
        normals     : list of [nx, ny, nz]
        n_vertices  : int
        n_triangles : int
        region_ids  : list of int (per-triangle)
        vertex_labels : list of int (per-vertex, majority-vote from triangles)
        metadata    : dict with provenance info
    """
    global _mesh_cache, _MH_SCALE

    if scale_mm != _MH_SCALE:
        _MH_SCALE = scale_mm
        _mesh_cache = None  # invalidate cache on parameter change

    if _mesh_cache is not None:
        return _mesh_cache

    if obj_path is None:
        obj_path = _BASE_OBJ

    if not obj_path.exists():
        raise FileNotFoundError(
            f"MakeHuman base mesh not found at {obj_path}.  "
            f"Expected CC0-licensed base.obj from makehumancommunity."
        )

    logger.info("Loading MakeHuman base mesh from %s", obj_path)

    # ── 1. Parse OBJ ──
    vertices, faces, groups = _parse_obj(obj_path)
    logger.info("Parsed OBJ: %d vertices, %d faces, %d groups",
                len(vertices), len(faces), len(groups))

    # ── 2. Extract head/face ──
    head_verts, head_faces = _extract_head(vertices, faces, groups)
    logger.info("Extracted head: %d vertices, %d faces",
                len(head_verts), len(head_faces))

    # ── 3. Transform to pipeline coordinates ──
    head_verts = _transform_to_pipeline_coords(head_verts)

    # ── 4. Triangulate ──
    triangles = _triangulate_quads(head_faces)
    logger.info("Triangulated: %d triangles", len(triangles))

    # ── 5. Compute normals ──
    normals = _compute_normals(head_verts, triangles)

    # ── 6. Assign anatomical regions ──
    region_ids = _assign_regions(head_verts, triangles)

    # ── 7. Per-vertex labels (majority vote from adjacent triangles) ──
    vertex_labels = np.full(len(head_verts), REGION_SOFT_TISSUE, dtype=np.int32)
    vote_counts: Dict[int, Dict[int, int]] = {}
    for ti, tri in enumerate(triangles):
        rid = int(region_ids[ti])
        for vi in tri:
            vi_int = int(vi)
            if vi_int not in vote_counts:
                vote_counts[vi_int] = {}
            vote_counts[vi_int][rid] = vote_counts[vi_int].get(rid, 0) + 1
    for vi, counts in vote_counts.items():
        vertex_labels[vi] = max(counts, key=counts.get)  # type: ignore[arg-type]

    region_counts = {REGION_NAMES[i]: int(np.sum(region_ids == i))
                     for i in range(len(REGION_NAMES))
                     if np.any(region_ids == i)}

    result: Dict[str, Any] = {
        "positions": head_verts.tolist(),
        "indices": triangles.tolist(),
        "normals": normals.tolist(),
        "n_vertices": len(head_verts),
        "n_triangles": len(triangles),
        "region_ids": region_ids.tolist(),
        "vertex_labels": vertex_labels.tolist(),
        "metadata": {
            "source": "MakeHuman Community base mesh",
            "license": "CC0 1.0 Universal (Public Domain)",
            "topology": "real_human_face",
            "coordinate_system": "Frankfurt Horizontal, mm",
            "origin": "subnasale",
            "region_counts": region_counts,
        },
    }

    _mesh_cache = result
    logger.info(
        "MakeHuman face mesh ready: %d verts, %d tris, regions: %s",
        result["n_vertices"], result["n_triangles"], region_counts,
    )
    return result


def load_makehuman_surface_mesh(
    obj_path: Optional[Path] = None,
) -> Any:
    """Load the MakeHuman face as a ``SurfaceMesh`` dataclass.

    This returns the mesh in the exact format consumed by
    ``CaseBundle.save_surface_mesh`` and ``api._bundle_surface_mesh``.
    """
    from .core.types import SurfaceMesh

    data = load_makehuman_face(obj_path)

    return SurfaceMesh(
        vertices=np.array(data["positions"], dtype=np.float64),
        triangles=np.array(data["indices"], dtype=np.int64),
        normals=np.array(data["normals"], dtype=np.float64),
        vertex_labels=np.array(data["vertex_labels"], dtype=np.int8),
        metadata=data["metadata"],
    )


def invalidate_cache() -> None:
    """Clear the cached mesh so the next call re-loads from disk."""
    global _mesh_cache
    _mesh_cache = None


def _snap_to_surface(
    vertices: np.ndarray,
    target: List[float],
) -> List[float]:
    """Find the nearest mesh vertex to a target 3D position.

    Guarantees the returned position lies exactly on the mesh surface.
    """
    t = np.array(target)
    dists = np.linalg.norm(vertices - t, axis=1)
    idx = int(dists.argmin())
    v = vertices[idx]
    return [float(v[0]), float(v[1]), float(v[2])]


def generate_makehuman_landmarks() -> List[Dict[str, Any]]:
    """Generate anatomical landmarks positioned on the MakeHuman face mesh.

    Uses profile-curve analysis on the anterior face surface to detect
    anatomical features.  Every landmark is snapped to the nearest
    mesh vertex so it lies exactly on the rendered surface.

    The full head mesh includes the posterior skull, so all searches
    are restricted to the anterior face (Z > Z_ANTERIOR_CUTOFF).
    """
    data = load_makehuman_face()
    vertices = np.array(data["positions"])

    # ── Anterior-face filter ──────────────────────────────────
    # Only search vertices on the facial surface, not the posterior
    # skull / occiput.  The face surface spans roughly Z > -15 mm.
    Z_ANTERIOR = -15.0
    face_mask = vertices[:, 2] > Z_ANTERIOR
    face_verts = vertices[face_mask]

    def _find_anterior(
        y_lo: float, y_hi: float,
        x_lo: float = -8.0, x_hi: float = 8.0,
    ) -> List[float]:
        """Find the most anterior (max Z) face vertex in a Y/X band."""
        mask = (
            (face_verts[:, 1] >= y_lo) & (face_verts[:, 1] <= y_hi) &
            (face_verts[:, 0] >= x_lo) & (face_verts[:, 0] <= x_hi)
        )
        if not mask.any():
            # Relax X constraint
            mask = (face_verts[:, 1] >= y_lo) & (face_verts[:, 1] <= y_hi)
        if not mask.any():
            return [0.0, (y_lo + y_hi) / 2.0, 0.0]
        subset = face_verts[mask]
        best = subset[subset[:, 2].argmax()]
        return [float(best[0]), float(best[1]), float(best[2])]

    def _find_lateral(
        side: float, y_lo: float, y_hi: float,
        x_min: float = 8.0,
    ) -> List[float]:
        """Find a lateral face-surface point (max Z with |X|>x_min)."""
        if side > 0:
            mask = (
                (face_verts[:, 0] > x_min) &
                (face_verts[:, 1] >= y_lo) & (face_verts[:, 1] <= y_hi)
            )
        else:
            mask = (
                (face_verts[:, 0] < -x_min) &
                (face_verts[:, 1] >= y_lo) & (face_verts[:, 1] <= y_hi)
            )
        if not mask.any():
            return [side * 20.0, (y_lo + y_hi) / 2.0, -5.0]
        subset = face_verts[mask]
        best = subset[subset[:, 2].argmax()]
        return [float(best[0]), float(best[1]), float(best[2])]

    def _find_most_lateral(
        side: float, y_lo: float, y_hi: float,
    ) -> List[float]:
        """Find the most lateral face vertex (max |X|) in a Y band."""
        mask = (face_verts[:, 1] >= y_lo) & (face_verts[:, 1] <= y_hi)
        if not mask.any():
            return [side * 20.0, (y_lo + y_hi) / 2.0, -5.0]
        subset = face_verts[mask]
        if side > 0:
            best = subset[subset[:, 0].argmax()]
        else:
            best = subset[subset[:, 0].argmin()]
        return [float(best[0]), float(best[1]), float(best[2])]

    # ── Profile-curve feature detection ───────────────────────
    # Midline profile on anterior face only
    midline_mask = (np.abs(face_verts[:, 0]) < 5.0)
    mid = face_verts[midline_mask]
    mid = mid[mid[:, 1].argsort()]  # sort inferior → superior

    # Pronasale: max Z on midline = nose tip
    pronasale_idx = mid[:, 2].argmax()
    pronasale_pos = [float(mid[pronasale_idx, 0]),
                     float(mid[pronasale_idx, 1]),
                     float(mid[pronasale_idx, 2])]
    tip_y = pronasale_pos[1]
    tip_z = pronasale_pos[2]

    # Subnasale: Z minimum between lip area and nose tip on midline
    sub_mask = (mid[:, 1] > tip_y - 30) & (mid[:, 1] < tip_y - 5)
    if sub_mask.any():
        sub_band = mid[sub_mask]
        # Find local Z maximum (upper lip / subnasale protrusion)
        sub_best = sub_band[sub_band[:, 2].argmax()]
        subnasale_pos = [float(sub_best[0]), float(sub_best[1]), float(sub_best[2])]
    else:
        subnasale_pos = [0.0, tip_y - 15.0, tip_z - 5.0]

    # Pogonion: max Z on chin (Y band below lips)
    pog_mask = (mid[:, 1] > tip_y - 55) & (mid[:, 1] < tip_y - 30)
    if pog_mask.any():
        pog_band = mid[pog_mask]
        pog_best = pog_band[pog_band[:, 2].argmax()]
        pogonion_pos = [float(pog_best[0]), float(pog_best[1]), float(pog_best[2])]
    else:
        pogonion_pos = [0.0, tip_y - 45.0, tip_z - 15.0]

    # Menton: lowest Y vertex on anterior midline chin
    men_mask = (mid[:, 1] > tip_y - 60) & (mid[:, 1] < pogonion_pos[1] + 3)
    if men_mask.any():
        men_band = mid[men_mask]
        men_best = men_band[men_band[:, 1].argmin()]
        menton_pos = [float(men_best[0]), float(men_best[1]), float(men_best[2])]
    else:
        menton_pos = [0.0, pogonion_pos[1] - 5.0, pogonion_pos[2] - 2.0]

    # Stomion: Z local max between subnasale and pogonion (lips)
    sto_mask = (mid[:, 1] > pogonion_pos[1]) & (mid[:, 1] < subnasale_pos[1])
    if sto_mask.any():
        sto_band = mid[sto_mask]
        # Upper third = upper lip, lower third = lower lip
        y_range = subnasale_pos[1] - pogonion_pos[1]
        upper_mask = sto_band[:, 1] > (pogonion_pos[1] + y_range * 0.5)
        lower_mask = sto_band[:, 1] < (pogonion_pos[1] + y_range * 0.5)

        if upper_mask.any():
            lab_sup = sto_band[upper_mask]
            labrale_sup_pos = [float(lab_sup[lab_sup[:, 2].argmax(), 0]),
                               float(lab_sup[lab_sup[:, 2].argmax(), 1]),
                               float(lab_sup[lab_sup[:, 2].argmax(), 2])]
        else:
            labrale_sup_pos = subnasale_pos

        if lower_mask.any():
            lab_inf = sto_band[lower_mask]
            labrale_inf_pos = [float(lab_inf[lab_inf[:, 2].argmax(), 0]),
                               float(lab_inf[lab_inf[:, 2].argmax(), 1]),
                               float(lab_inf[lab_inf[:, 2].argmax(), 2])]
        else:
            labrale_inf_pos = pogonion_pos

        stomion_y = (labrale_sup_pos[1] + labrale_inf_pos[1]) / 2.0
        stomion_pos = _find_anterior(stomion_y - 2, stomion_y + 2, -3.0, 3.0)
    else:
        mid_y = (subnasale_pos[1] + pogonion_pos[1]) / 2.0
        stomion_pos = [0.0, mid_y, subnasale_pos[2] - 2.0]
        labrale_sup_pos = [0.0, mid_y + 3.0, subnasale_pos[2] - 1.0]
        labrale_inf_pos = [0.0, mid_y - 3.0, subnasale_pos[2] - 3.0]

    # Nasion: Z minimum between nasal dorsum and glabella
    nas_mask = (mid[:, 1] > tip_y + 5) & (mid[:, 1] < tip_y + 30)
    if nas_mask.any():
        nas_band = mid[nas_mask]
        nas_best = nas_band[nas_band[:, 2].argmin()]  # deepest point = nasion
        nasion_pos = [float(nas_best[0]), float(nas_best[1]), float(nas_best[2])]
    else:
        nasion_pos = _find_anterior(tip_y + 10, tip_y + 25)

    # Rhinion: midpoint between pronasale and nasion on dorsum
    rhinion_y = (tip_y + nasion_pos[1]) / 2.0
    rhinion_pos = _find_anterior(rhinion_y - 3, rhinion_y + 3)

    # Sellion: deepest point of nasofrontal angle (near nasion)
    sel_mask = (mid[:, 1] > nasion_pos[1] - 5) & (mid[:, 1] < nasion_pos[1] + 8)
    if sel_mask.any():
        sel_band = mid[sel_mask]
        sel_best = sel_band[sel_band[:, 2].argmin()]
        sellion_pos = [float(sel_best[0]), float(sel_best[1]), float(sel_best[2])]
    else:
        sellion_pos = nasion_pos

    # Glabella: Z max above nasion (brow prominence)
    gla_mask = (mid[:, 1] > nasion_pos[1] + 3) & (mid[:, 1] < nasion_pos[1] + 30)
    if gla_mask.any():
        gla_band = mid[gla_mask]
        gla_best = gla_band[gla_band[:, 2].argmax()]
        glabella_pos = [float(gla_best[0]), float(gla_best[1]), float(gla_best[2])]
    else:
        glabella_pos = _find_anterior(nasion_pos[1] + 5, nasion_pos[1] + 25)

    # Supratip breakpoint: on dorsum between pronasale and rhinion.
    # Using max-Z search would just re-find pronasale, so interpolate
    # along the dorsum and snap to surface.
    supratip_pos = [
        pronasale_pos[0] * 0.65 + rhinion_pos[0] * 0.35,
        pronasale_pos[1] * 0.65 + rhinion_pos[1] * 0.35,
        pronasale_pos[2] * 0.65 + rhinion_pos[2] * 0.35,
    ]

    # Columella breakpoint: on columella between pronasale and subnasale.
    # Interpolate and snap — max-Z would re-find pronasale.
    columella_pos = [
        pronasale_pos[0] * 0.4 + subnasale_pos[0] * 0.6,
        pronasale_pos[1] * 0.4 + subnasale_pos[1] * 0.6,
        pronasale_pos[2] * 0.4 + subnasale_pos[2] * 0.6,
    ]

    # ── Snap all to nearest surface vertex ────────────────────
    landmarks = [
        {"type": "pronasale",
         "position": _snap_to_surface(vertices, pronasale_pos),
         "confidence": 0.95},
        {"type": "subnasale",
         "position": _snap_to_surface(vertices, subnasale_pos),
         "confidence": 0.95},
        {"type": "rhinion",
         "position": _snap_to_surface(vertices, rhinion_pos),
         "confidence": 0.92},
        {"type": "nasion",
         "position": _snap_to_surface(vertices, nasion_pos),
         "confidence": 0.90},
        {"type": "glabella",
         "position": _snap_to_surface(vertices, glabella_pos),
         "confidence": 0.88},
        {"type": "sellion",
         "position": _snap_to_surface(vertices, sellion_pos),
         "confidence": 0.88},
        {"type": "pogonion",
         "position": _snap_to_surface(vertices, pogonion_pos),
         "confidence": 0.90},
        {"type": "menton",
         "position": _snap_to_surface(vertices, menton_pos),
         "confidence": 0.85},
        {"type": "stomion",
         "position": _snap_to_surface(vertices, stomion_pos),
         "confidence": 0.90},
        {"type": "labrale_superius",
         "position": _snap_to_surface(vertices, labrale_sup_pos),
         "confidence": 0.88},
        {"type": "labrale_inferius",
         "position": _snap_to_surface(vertices, labrale_inf_pos),
         "confidence": 0.88},
        {"type": "supratip_breakpoint",
         "position": _snap_to_surface(vertices, supratip_pos),
         "confidence": 0.85},
        {"type": "columella_breakpoint",
         "position": _snap_to_surface(vertices, columella_pos),
         "confidence": 0.82},
        {"type": "alar_rim_left",
         "position": _snap_to_surface(vertices, _find_lateral(-1, subnasale_pos[1] - 3, subnasale_pos[1] + 8, 6.0)),
         "confidence": 0.88},
        {"type": "alar_rim_right",
         "position": _snap_to_surface(vertices, _find_lateral(1, subnasale_pos[1] - 3, subnasale_pos[1] + 8, 6.0)),
         "confidence": 0.88},
        {"type": "alar_crease_left",
         "position": _snap_to_surface(vertices, _find_lateral(-1, subnasale_pos[1] - 6, subnasale_pos[1] + 3, 10.0)),
         "confidence": 0.82},
        {"type": "alar_crease_right",
         "position": _snap_to_surface(vertices, _find_lateral(1, subnasale_pos[1] - 6, subnasale_pos[1] + 3, 10.0)),
         "confidence": 0.82},
        {"type": "tip_defining_point_left",
         "position": _snap_to_surface(vertices, [-4.0, tip_y, tip_z - 1.0]),
         "confidence": 0.80},
        {"type": "tip_defining_point_right",
         "position": _snap_to_surface(vertices, [4.0, tip_y, tip_z - 1.0]),
         "confidence": 0.80},
        {"type": "endocanthion_left",
         "position": _snap_to_surface(vertices, _find_anterior(nasion_pos[1] - 8, nasion_pos[1] + 2, -18, -6)),
         "confidence": 0.85},
        {"type": "endocanthion_right",
         "position": _snap_to_surface(vertices, _find_anterior(nasion_pos[1] - 8, nasion_pos[1] + 2, 6, 18)),
         "confidence": 0.85},
        {"type": "exocanthion_left",
         "position": _snap_to_surface(vertices, _find_lateral(-1, nasion_pos[1] - 10, nasion_pos[1] + 5, 20.0)),
         "confidence": 0.82},
        {"type": "exocanthion_right",
         "position": _snap_to_surface(vertices, _find_lateral(1, nasion_pos[1] - 10, nasion_pos[1] + 5, 20.0)),
         "confidence": 0.82},
        {"type": "cheilion_left",
         "position": _snap_to_surface(vertices, _find_lateral(-1, stomion_pos[1] - 5, stomion_pos[1] + 5, 12.0)),
         "confidence": 0.82},
        {"type": "cheilion_right",
         "position": _snap_to_surface(vertices, _find_lateral(1, stomion_pos[1] - 5, stomion_pos[1] + 5, 12.0)),
         "confidence": 0.82},
    ]
    return landmarks
