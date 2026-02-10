"""Shared test fixtures and helpers for the facial plastics test suite."""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict

from products.facial_plastics.core.types import (
    BoundingBox,
    ClinicalMeasurement,
    Landmark,
    LandmarkType,
    MaterialModel,
    MeshElementType,
    Modality,
    ProcedureType,
    QualityLevel,
    StructureType,
    SurfaceMesh,
    TissueProperties,
    Vec3,
    VolumeMesh,
)


def make_box_surface_mesh(
    size: float = 20.0,
    n_per_edge: int = 5,
) -> SurfaceMesh:
    """Generate a simple box surface mesh for testing.

    Returns a triangulated mesh of a cube [-size/2, size/2]^3.
    """
    half = size / 2.0
    verts: list[list[float]] = []
    tris: list[list[int]] = []

    # Generate a grid on each face of the cube
    faces_def = [
        # (axis, sign, u_axis, v_axis)
        (2, +1, 0, 1),  # +Z
        (2, -1, 0, 1),  # -Z
        (1, +1, 0, 2),  # +Y
        (1, -1, 0, 2),  # -Y
        (0, +1, 1, 2),  # +X
        (0, -1, 1, 2),  # -X
    ]

    for axis, sign, u_ax, v_ax in faces_def:
        base_idx = len(verts)
        for i in range(n_per_edge):
            for j in range(n_per_edge):
                u = -half + (i / (n_per_edge - 1)) * size
                v = -half + (j / (n_per_edge - 1)) * size
                pt = [0.0, 0.0, 0.0]
                pt[axis] = sign * half
                pt[u_ax] = u
                pt[v_ax] = v
                verts.append(pt)

        for i in range(n_per_edge - 1):
            for j in range(n_per_edge - 1):
                idx = base_idx + i * n_per_edge + j
                tris.append([idx, idx + 1, idx + n_per_edge])
                tris.append([idx + 1, idx + n_per_edge + 1, idx + n_per_edge])

    return SurfaceMesh(
        vertices=np.array(verts, dtype=np.float64),
        triangles=np.array(tris, dtype=np.int64),
    )


def make_nose_surface_mesh(n_verts: int = 200) -> SurfaceMesh:
    """Generate a simplified nose-like surface mesh for testing.

    Creates a paraboloid shape roughly resembling a nasal surface.
    """
    # Parametric paraboloid with some noise
    rng = np.random.RandomState(42)
    u = rng.uniform(-1, 1, n_verts)
    v = rng.uniform(-1, 1, n_verts)

    x = u * 15.0  # mm, lateral
    y = v * 20.0  # mm, vertical (inferior-superior)
    z = 30.0 - 0.5 * (u ** 2 + v ** 2) * 10.0  # mm, AP projection

    vertices = np.column_stack([x, y, z])

    # Simple Delaunay-like triangulation via grid
    # Sort by u, v and create a grid triangulation
    n_side = int(np.sqrt(n_verts))
    actual_n = n_side * n_side
    grid_verts = []
    for i in range(n_side):
        for j in range(n_side):
            uu = -1.0 + 2.0 * i / (n_side - 1)
            vv = -1.0 + 2.0 * j / (n_side - 1)
            xx = uu * 15.0
            yy = vv * 20.0
            zz = 30.0 - 0.5 * (uu ** 2 + vv ** 2) * 10.0
            grid_verts.append([xx, yy, zz])

    tris = []
    for i in range(n_side - 1):
        for j in range(n_side - 1):
            idx = i * n_side + j
            tris.append([idx, idx + 1, idx + n_side])
            tris.append([idx + 1, idx + n_side + 1, idx + n_side])

    return SurfaceMesh(
        vertices=np.array(grid_verts, dtype=np.float64),
        triangles=np.array(tris, dtype=np.int64),
    )


def make_volume_mesh(
    n_nodes: int = 125,
    n_elements: int = 384,
) -> VolumeMesh:
    """Generate a simple tetrahedral volume mesh of a cube."""
    n_side = 5
    nodes = []
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                nodes.append([
                    i * 5.0,  # mm
                    j * 5.0,
                    k * 5.0,
                ])

    # Generate tetrahedra by subdividing each hex into 6 tets
    elements = []
    for i in range(n_side - 1):
        for j in range(n_side - 1):
            for k in range(n_side - 1):
                # 8 corners of the hex cell
                c = [
                    i * n_side * n_side + j * n_side + k,
                    i * n_side * n_side + j * n_side + (k + 1),
                    i * n_side * n_side + (j + 1) * n_side + k,
                    i * n_side * n_side + (j + 1) * n_side + (k + 1),
                    (i + 1) * n_side * n_side + j * n_side + k,
                    (i + 1) * n_side * n_side + j * n_side + (k + 1),
                    (i + 1) * n_side * n_side + (j + 1) * n_side + k,
                    (i + 1) * n_side * n_side + (j + 1) * n_side + (k + 1),
                ]
                # 6-tet subdivision of hex
                elements.append([c[0], c[1], c[3], c[5]])
                elements.append([c[0], c[3], c[2], c[6]])
                elements.append([c[0], c[5], c[4], c[6]])
                elements.append([c[3], c[5], c[6], c[7]])
                elements.append([c[0], c[3], c[5], c[6]])
                elements.append([c[0], c[6], c[5], c[3]])  # degenerate pair

    nodes_arr = np.array(nodes, dtype=np.float64)
    elem_arr = np.array(elements[:n_elements] if len(elements) > n_elements
                        else elements, dtype=np.int64)

    region_ids = np.zeros(elem_arr.shape[0], dtype=np.int32)

    return VolumeMesh(
        nodes=nodes_arr,
        elements=elem_arr,
        element_type=MeshElementType.TET4,
        region_ids=region_ids,
        region_materials={
            0: TissueProperties(
                structure_type=StructureType.FAT_SUBCUTANEOUS,
                material_model=MaterialModel.NEO_HOOKEAN,
                parameters={"mu": 10.0, "kappa": 100.0},
            ),
        },
    )


def make_rhinoplasty_landmarks() -> Dict[LandmarkType, Vec3]:
    """Standard set of rhinoplasty landmarks for testing."""
    return {
        LandmarkType.NASION: Vec3(0.0, 35.0, 15.0),
        LandmarkType.SELLION: Vec3(0.0, 33.0, 14.0),
        LandmarkType.RHINION: Vec3(0.0, 20.0, 25.0),
        LandmarkType.PRONASALE: Vec3(0.0, 5.0, 35.0),
        LandmarkType.SUBNASALE: Vec3(0.0, 0.0, 25.0),
        LandmarkType.COLUMELLA_BREAKPOINT: Vec3(0.0, 2.0, 28.0),
        LandmarkType.SUPRATIP_BREAKPOINT: Vec3(0.0, 8.0, 33.0),
        LandmarkType.TIP_DEFINING_POINT_LEFT: Vec3(-3.0, 5.0, 34.0),
        LandmarkType.TIP_DEFINING_POINT_RIGHT: Vec3(3.0, 5.0, 34.0),
        LandmarkType.ALAR_RIM_LEFT: Vec3(-17.0, 2.0, 20.0),
        LandmarkType.ALAR_RIM_RIGHT: Vec3(17.0, 2.0, 20.0),
        LandmarkType.ALAR_CREASE_LEFT: Vec3(-16.0, 0.0, 18.0),
        LandmarkType.ALAR_CREASE_RIGHT: Vec3(16.0, 0.0, 18.0),
        LandmarkType.POGONION: Vec3(0.0, -30.0, 10.0),
        LandmarkType.MENTON: Vec3(0.0, -40.0, 5.0),
        LandmarkType.STOMION: Vec3(0.0, -15.0, 15.0),
        LandmarkType.LABRALE_SUPERIUS: Vec3(0.0, -10.0, 18.0),
        LandmarkType.ENDOCANTHION_LEFT: Vec3(-15.0, 35.0, 10.0),
        LandmarkType.ENDOCANTHION_RIGHT: Vec3(15.0, 35.0, 10.0),
        LandmarkType.EXOCANTHION_LEFT: Vec3(-30.0, 35.0, 5.0),
        LandmarkType.EXOCANTHION_RIGHT: Vec3(30.0, 35.0, 5.0),
        LandmarkType.GLABELLA: Vec3(0.0, 40.0, 12.0),
        LandmarkType.TRICHION: Vec3(0.0, 60.0, 5.0),
    }
