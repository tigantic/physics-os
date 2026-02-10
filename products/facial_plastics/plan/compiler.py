"""Plan compiler — translate SurgicalPlan → solver boundary conditions.

The compiler walks the plan DAG, converting each SurgicalOp
into concrete boundary conditions (BCs) that the FEM/CFD
solvers can consume:

  - Nodal displacements (Dirichlet BCs)
  - Nodal/surface forces (Neumann BCs)
  - Constraint equations (tied contacts, symmetry)
  - Material modifications (graft insertion, scoring)
  - Mesh modifications (element removal, insertion, splitting)
  - CFD boundary changes (wall geometry, inlets, outlets)

The compiler is deterministic: same plan + same mesh = same BCs.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from ..core.provenance import hash_dict
from ..core.types import (
    MaterialModel,
    MeshElementType,
    SolverType,
    StructureType,
    TissueProperties,
    Vec3,
    VolumeMesh,
)
from .dsl import (
    BranchNode,
    CompositeOp,
    OpCategory,
    PlanValidationError,
    SequenceNode,
    SurgicalOp,
    SurgicalPlan,
)

logger = logging.getLogger(__name__)


# ── BC type definitions ───────────────────────────────────────────

class BCType(str, Enum):
    """Types of boundary conditions produced by compilation."""
    NODAL_DISPLACEMENT = "nodal_displacement"
    NODAL_FORCE = "nodal_force"
    SURFACE_PRESSURE = "surface_pressure"
    CONTACT_TIE = "contact_tie"
    SYMMETRY = "symmetry"
    FIXED = "fixed"
    ELEMENT_REMOVAL = "element_removal"
    ELEMENT_INSERTION = "element_insertion"
    MATERIAL_OVERRIDE = "material_override"
    STIFFNESS_MODIFICATION = "stiffness_modification"
    WALL_GEOMETRY = "wall_geometry"
    CFD_INLET = "cfd_inlet"
    CFD_OUTLET = "cfd_outlet"
    CFD_WALL = "cfd_wall"


@dataclass
class BoundaryCondition:
    """A single boundary condition for the solver."""
    bc_type: BCType
    node_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    element_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    values: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    direction: Optional[np.ndarray] = None  # (3,) unit vector
    magnitude: float = 0.0
    region_id: int = -1
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_op: str = ""  # name of the originating SurgicalOp

    def content_hash(self) -> str:
        data = {
            "bc_type": self.bc_type.value,
            "n_nodes": len(self.node_ids),
            "n_elements": len(self.element_ids),
            "magnitude": self.magnitude,
            "source_op": self.source_op,
        }
        return hash_dict(data)


@dataclass
class MaterialModification:
    """Modification to material properties in a region."""
    region_id: int
    element_ids: np.ndarray
    original_model: MaterialModel
    modified_model: MaterialModel
    modified_params: Dict[str, float]
    source_op: str = ""
    description: str = ""


@dataclass
class MeshModification:
    """Modification to the mesh topology."""
    mod_type: str  # "remove", "insert", "split", "refine"
    element_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    new_nodes: Optional[np.ndarray] = None    # (N,3)
    new_elements: Optional[np.ndarray] = None  # (E,K)
    new_region_id: int = -1
    source_op: str = ""
    description: str = ""


@dataclass
class CompilationResult:
    """Complete compilation output — everything the solver needs."""
    boundary_conditions: List[BoundaryCondition] = field(default_factory=list)
    material_modifications: List[MaterialModification] = field(default_factory=list)
    mesh_modifications: List[MeshModification] = field(default_factory=list)
    solver_type: SolverType = SolverType.FEM_QUASISTATIC
    n_load_steps: int = 1
    plan_hash: str = ""
    mesh_hash: str = ""
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    @property
    def n_bcs(self) -> int:
        return len(self.boundary_conditions)

    @property
    def n_material_mods(self) -> int:
        return len(self.material_modifications)

    @property
    def n_mesh_mods(self) -> int:
        return len(self.mesh_modifications)

    def content_hash(self) -> str:
        bc_hashes = [bc.content_hash() for bc in self.boundary_conditions]
        return hash_dict({
            "plan_hash": self.plan_hash,
            "mesh_hash": self.mesh_hash,
            "n_bcs": self.n_bcs,
            "bc_hashes": bc_hashes,
            "solver_type": self.solver_type.value,
            "n_load_steps": self.n_load_steps,
        })

    def summary(self) -> str:
        lines = [
            f"Compilation result: {self.n_bcs} BCs, "
            f"{self.n_material_mods} material mods, "
            f"{self.n_mesh_mods} mesh mods",
            f"  Solver: {self.solver_type.value}, load steps: {self.n_load_steps}",
            f"  Plan hash: {self.plan_hash[:16]}",
        ]
        if self.warnings:
            lines.append(f"  Warnings: {len(self.warnings)}")
        if self.errors:
            lines.append(f"  ERRORS: {len(self.errors)}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_hash": self.plan_hash,
            "mesh_hash": self.mesh_hash,
            "solver_type": self.solver_type.value,
            "n_load_steps": self.n_load_steps,
            "n_bcs": self.n_bcs,
            "n_material_mods": self.n_material_mods,
            "n_mesh_mods": self.n_mesh_mods,
            "content_hash": self.content_hash(),
            "warnings": self.warnings,
            "errors": self.errors,
        }


# ── Geometry helpers ──────────────────────────────────────────────

def _find_nodes_in_region(
    mesh: VolumeMesh,
    region_id: int,
) -> np.ndarray:
    """Return node IDs belonging to elements in a given region."""
    mask = mesh.region_ids == region_id
    elem_ids = np.where(mask)[0]
    node_set: Set[int] = set()
    for eid in elem_ids:
        node_set.update(mesh.elements[eid].tolist())
    return np.array(sorted(node_set), dtype=np.int64)


def _find_elements_by_structure(
    mesh: VolumeMesh,
    structure: StructureType,
) -> np.ndarray:
    """Return element IDs assigned to a given structure type."""
    target_regions: List[int] = []
    for rid, props in mesh.region_materials.items():
        if props.structure_type == structure:
            target_regions.append(rid)
    if not target_regions:
        return np.array([], dtype=np.int64)
    mask = np.isin(mesh.region_ids, target_regions)
    return np.where(mask)[0].astype(np.int64)


def _find_nodes_by_structure(
    mesh: VolumeMesh,
    structure: StructureType,
) -> np.ndarray:
    """Return node IDs for elements assigned to a structure."""
    elem_ids = _find_elements_by_structure(mesh, structure)
    if len(elem_ids) == 0:
        return np.array([], dtype=np.int64)
    node_set: Set[int] = set()
    for eid in elem_ids:
        node_set.update(mesh.elements[eid].tolist())
    return np.array(sorted(node_set), dtype=np.int64)


def _find_surface_nodes(
    mesh: VolumeMesh,
    tag: str,
) -> np.ndarray:
    """Return node IDs from a named surface tag."""
    if tag not in mesh.surface_tags:
        return np.array([], dtype=np.int64)
    face_indices = mesh.surface_tags[tag]
    node_set: Set[int] = set()
    for fi in face_indices:
        node_set.add(int(fi))
    return np.array(sorted(node_set), dtype=np.int64)


def _compute_dorsal_profile_nodes(
    mesh: VolumeMesh,
    start_frac: float,
    end_frac: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find dorsal ridge nodes and compute displacement field.

    Returns (node_ids, displacement_vectors) for dorsal reduction.
    The dorsal ridge is approximated as the superior-most nodes
    of the nasal bone and upper lateral cartilage, between the
    start and end fractions along the nasion→tip axis.
    """
    bone_nodes = _find_nodes_by_structure(mesh, StructureType.BONE_NASAL)
    cart_nodes = _find_nodes_by_structure(mesh, StructureType.CARTILAGE_UPPER_LATERAL)
    all_nodes = np.union1d(bone_nodes, cart_nodes)

    if len(all_nodes) == 0:
        return np.array([], dtype=np.int64), np.zeros((0, 3), dtype=np.float64)

    positions = mesh.nodes[all_nodes]

    # Dorsal ridge: approximate as nodes along the superior edge
    # Use the Y-axis (anterior-posterior) as the primary direction
    # and Z-axis (superior-inferior) for identifying the ridge
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    y_range = y_max - y_min
    if y_range < 1e-6:
        return np.array([], dtype=np.int64), np.zeros((0, 3), dtype=np.float64)

    # Normalize along the dorsal axis (Y)
    y_frac = (positions[:, 1] - y_min) / y_range

    # Select nodes in the specified range
    mask = (y_frac >= start_frac) & (y_frac <= end_frac)
    selected = all_nodes[mask]
    selected_pos = positions[mask]

    if len(selected) == 0:
        return np.array([], dtype=np.int64), np.zeros((0, 3), dtype=np.float64)

    # The dorsal ridge is the top surface — select upper percentile in Z
    z_vals = selected_pos[:, 2]
    z_thresh = np.percentile(z_vals, 75)
    ridge_mask = z_vals >= z_thresh
    ridge_nodes = selected[ridge_mask]

    return ridge_nodes, mesh.nodes[ridge_nodes]


def _compute_osteotomy_plane(
    mesh: VolumeMesh,
    angle_deg: float,
    side: str,
    low_to_low: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute osteotomy cut plane for bone element splitting.

    Returns (element_ids_to_split, cut_plane_normal).
    """
    bone_elems = _find_elements_by_structure(mesh, StructureType.BONE_NASAL)
    maxilla_elems = _find_elements_by_structure(mesh, StructureType.BONE_MAXILLA)
    all_elems = np.union1d(bone_elems, maxilla_elems)

    if len(all_elems) == 0:
        return np.array([], dtype=np.int64), np.zeros(3, dtype=np.float64)

    # Compute element centroids
    centroids = np.zeros((len(all_elems), 3), dtype=np.float64)
    for i, eid in enumerate(all_elems):
        elem_nodes = mesh.elements[eid]
        centroids[i] = mesh.nodes[elem_nodes].mean(axis=0)

    # Define cut plane based on angle
    angle_rad = np.radians(angle_deg)
    if low_to_low:
        # Low-to-low: starts at piriform aperture base, stays low
        normal = np.array([np.sin(angle_rad), 0.0, np.cos(angle_rad)])
    else:
        # Low-to-high: starts low, angles upward
        normal = np.array([np.sin(angle_rad), np.sin(angle_rad * 0.5), np.cos(angle_rad)])

    # Mirror for right side
    if side == "right":
        normal[0] = -normal[0]

    # Normalize
    norm = np.linalg.norm(normal)
    if norm > 1e-12:
        normal = normal / norm

    # Find elements intersecting the cut plane
    # Elements whose centroids are within a band around the plane
    center = centroids.mean(axis=0)
    distances = np.abs(np.dot(centroids - center, normal))
    band_width = 1.5  # mm — osteotomy cut width
    cut_mask = distances < band_width

    # Lateralize: only cut on the specified side
    if side in ("left", "bilateral"):
        left_mask = centroids[:, 0] >= center[0] - 1.0
        cut_mask = cut_mask & left_mask
    if side in ("right", "bilateral"):
        right_mask = centroids[:, 0] <= center[0] + 1.0
        if side == "right":
            cut_mask = cut_mask & right_mask

    cut_elems = all_elems[cut_mask]
    return cut_elems, normal


# ── Operator-specific compilers ──────────────────────────────────

def _compile_dorsal_reduction(
    op: SurgicalOp,
    mesh: VolumeMesh,
    result: CompilationResult,
) -> None:
    """Compile dorsal reduction → nodal displacements."""
    amount = op.params.get("amount_mm", 2.0)
    start_frac = op.params.get("start_fraction", 0.0)
    end_frac = op.params.get("end_fraction", 1.0)
    taper = op.params.get("taper", True)

    ridge_nodes, ridge_positions = _compute_dorsal_profile_nodes(
        mesh, start_frac, end_frac,
    )

    if len(ridge_nodes) == 0:
        result.warnings.append(
            f"dorsal_reduction: no dorsal ridge nodes found "
            f"(start={start_frac}, end={end_frac})"
        )
        return

    # Compute displacement: move nodes inferiorly (negative Z)
    displacements = np.zeros((len(ridge_nodes), 3), dtype=np.float64)
    displacements[:, 2] = -amount

    if taper:
        # Taper at endpoints using a smooth cosine window
        y_vals = ridge_positions[:, 1]
        y_min, y_max = y_vals.min(), y_vals.max()
        y_range = y_max - y_min
        if y_range > 1e-6:
            y_frac = (y_vals - y_min) / y_range
            taper_width = 0.15
            # Taper at start
            start_mask = y_frac < taper_width
            if np.any(start_mask):
                t = y_frac[start_mask] / taper_width
                displacements[start_mask, 2] *= 0.5 * (1.0 - np.cos(np.pi * t))
            # Taper at end
            end_mask = y_frac > (1.0 - taper_width)
            if np.any(end_mask):
                t = (1.0 - y_frac[end_mask]) / taper_width
                displacements[end_mask, 2] *= 0.5 * (1.0 - np.cos(np.pi * t))

    # Elements to remove above the cut
    bone_elems = _find_elements_by_structure(mesh, StructureType.BONE_NASAL)
    if len(bone_elems) > 0:
        centroids = np.zeros((len(bone_elems), 3), dtype=np.float64)
        for i, eid in enumerate(bone_elems):
            centroids[i] = mesh.nodes[mesh.elements[eid]].mean(axis=0)

        # Remove elements that are above the new dorsal line
        removal_z = ridge_positions[:, 2].mean() - amount * 0.5
        above = centroids[:, 2] > removal_z
        if np.any(above):
            result.mesh_modifications.append(MeshModification(
                mod_type="remove",
                element_ids=bone_elems[above],
                source_op=op.name,
                description=f"Remove {int(np.sum(above))} bone elements above dorsal reduction",
            ))

    result.boundary_conditions.append(BoundaryCondition(
        bc_type=BCType.NODAL_DISPLACEMENT,
        node_ids=ridge_nodes,
        values=displacements,
        magnitude=amount,
        source_op=op.name,
        metadata={"taper": taper},
    ))


def _compile_osteotomy(
    op: SurgicalOp,
    mesh: VolumeMesh,
    result: CompilationResult,
    lateral: bool = True,
) -> None:
    """Compile lateral/medial osteotomy → element splitting + displacement."""
    side = op.params.get("side", "bilateral")
    angle_deg = op.params.get("angle_deg", 30.0)
    low_to_low = op.params.get("low_to_low", True)

    sides = ["left", "right"] if side == "bilateral" else [side]

    for s in sides:
        cut_elems, cut_normal = _compute_osteotomy_plane(
            mesh, angle_deg, s, low_to_low,
        )

        if len(cut_elems) == 0:
            result.warnings.append(
                f"{op.name}: no elements found for osteotomy on {s} side"
            )
            continue

        # Mesh modification: split elements along the cut plane
        result.mesh_modifications.append(MeshModification(
            mod_type="split",
            element_ids=cut_elems,
            source_op=op.name,
            description=f"Osteotomy cut ({s}) at {angle_deg}°",
        ))

        # After cut, the bone segment will be free to move
        # Add a displacement BC to model the infracture that follows
        cut_nodes: Set[int] = set()
        for eid in cut_elems:
            cut_nodes.update(mesh.elements[eid].tolist())
        node_arr = np.array(sorted(cut_nodes), dtype=np.int64)

        # Generate displacement field (will be applied by subsequent infracture op)
        result.boundary_conditions.append(BoundaryCondition(
            bc_type=BCType.NODAL_DISPLACEMENT,
            node_ids=node_arr,
            values=np.zeros((len(node_arr), 3), dtype=np.float64),
            direction=cut_normal,
            magnitude=0.0,  # Will be set by infracture
            source_op=op.name,
            metadata={"side": s, "angle_deg": angle_deg, "is_osteotomy_plane": True},
        ))


def _compile_infracture(
    op: SurgicalOp,
    mesh: VolumeMesh,
    result: CompilationResult,
) -> None:
    """Compile bone infracture → medial displacement of nasal bone segment."""
    side = op.params.get("side", "bilateral")
    displacement_mm = op.params.get("displacement_mm", 2.0)

    bone_nodes = _find_nodes_by_structure(mesh, StructureType.BONE_NASAL)
    if len(bone_nodes) == 0:
        result.warnings.append("bone_infracture: no nasal bone nodes found")
        return

    positions = mesh.nodes[bone_nodes]
    center_x = positions[:, 0].mean()

    sides = ["left", "right"] if side == "bilateral" else [side]

    for s in sides:
        if s == "left":
            mask = positions[:, 0] > center_x
            direction = np.array([-1.0, 0.0, 0.0])
        else:
            mask = positions[:, 0] < center_x
            direction = np.array([1.0, 0.0, 0.0])

        side_nodes = bone_nodes[mask]
        if len(side_nodes) == 0:
            continue

        displacements = np.outer(
            np.ones(len(side_nodes), dtype=np.float64) * displacement_mm,
            direction,
        )

        result.boundary_conditions.append(BoundaryCondition(
            bc_type=BCType.NODAL_DISPLACEMENT,
            node_ids=side_nodes,
            values=displacements,
            direction=direction,
            magnitude=displacement_mm,
            source_op=op.name,
            metadata={"side": s},
        ))


def _compile_septoplasty(
    op: SurgicalOp,
    mesh: VolumeMesh,
    result: CompilationResult,
) -> None:
    """Compile septoplasty → element removal/repositioning."""
    extent = op.params.get("resection_extent", "partial")

    septal_elems = _find_elements_by_structure(mesh, StructureType.CARTILAGE_SEPTUM)
    if len(septal_elems) == 0:
        result.warnings.append("septoplasty: no septal cartilage elements found")
        return

    # Compute centroids of septal elements
    centroids = np.zeros((len(septal_elems), 3), dtype=np.float64)
    for i, eid in enumerate(septal_elems):
        centroids[i] = mesh.nodes[mesh.elements[eid]].mean(axis=0)

    center_x = centroids[:, 0].mean()

    if extent == "partial":
        # Remove deviated portion — posterior-inferior quadrant
        y_mid = np.median(centroids[:, 1])
        z_mid = np.median(centroids[:, 2])
        deviated = (centroids[:, 1] < y_mid) & (centroids[:, 2] < z_mid)
        remove_fraction = 0.3
    elif extent == "submucous":
        # Remove larger section, preserving L-strut
        y_range = centroids[:, 1].max() - centroids[:, 1].min()
        z_range = centroids[:, 2].max() - centroids[:, 2].min()
        l_strut_y = centroids[:, 1].max() - 10.0  # 10mm dorsal L-strut
        l_strut_z = centroids[:, 2].max() - 10.0  # 10mm caudal L-strut
        deviated = (centroids[:, 1] < l_strut_y) & (centroids[:, 2] < l_strut_z)
        remove_fraction = 0.5
    elif extent == "extracorporeal":
        # Remove entire septum (it will be reimplanted)
        deviated = np.ones(len(septal_elems), dtype=bool)
        remove_fraction = 0.9
    else:
        result.errors.append(f"septoplasty: unknown extent '{extent}'")
        return

    # Limit removal to deviated elements
    if np.any(deviated):
        remove_ids = septal_elems[deviated]
        # Random subset based on fraction
        n_remove = max(1, int(len(remove_ids) * remove_fraction))
        if n_remove < len(remove_ids):
            # Deterministic selection: take the most deviated ones
            dev_x = np.abs(centroids[deviated, 0] - center_x)
            order = np.argsort(dev_x)[::-1]
            remove_ids = remove_ids[order[:n_remove]]

        result.mesh_modifications.append(MeshModification(
            mod_type="remove",
            element_ids=remove_ids,
            source_op=op.name,
            description=f"Septoplasty ({extent}): remove {len(remove_ids)} elements",
        ))


def _compile_turbinate_reduction(
    op: SurgicalOp,
    mesh: VolumeMesh,
    result: CompilationResult,
) -> None:
    """Compile turbinate reduction → element removal + stiffness mod."""
    reduction_pct = op.params.get("reduction_pct", 30.0)
    method = op.params.get("method", "submucosal")

    turb_elems = _find_elements_by_structure(mesh, StructureType.TURBINATE_INFERIOR)
    if len(turb_elems) == 0:
        result.warnings.append("turbinate_reduction: no turbinate elements found")
        return

    n_remove = max(1, int(len(turb_elems) * reduction_pct / 100.0))

    if method == "submucosal":
        # Submucosal resection — remove interior elements, preserve mucosa
        centroids = np.zeros((len(turb_elems), 3), dtype=np.float64)
        for i, eid in enumerate(turb_elems):
            centroids[i] = mesh.nodes[mesh.elements[eid]].mean(axis=0)
        # Sort by distance from surface (interior first)
        center = centroids.mean(axis=0)
        dists = np.linalg.norm(centroids - center, axis=1)
        order = np.argsort(dists)  # interior first
        remove_ids = turb_elems[order[:n_remove]]
    elif method == "cautery":
        # Cautery — stiffen and shrink mucosal surface
        remove_ids = turb_elems[:n_remove]
        # Also modify remaining elements to be stiffer (cauterized tissue)
        remaining = turb_elems[n_remove:]
        if len(remaining) > 0:
            for rid, props in mesh.region_materials.items():
                if props.structure_type == StructureType.TURBINATE_INFERIOR:
                    result.material_modifications.append(MaterialModification(
                        region_id=rid,
                        element_ids=remaining,
                        original_model=props.material_model,
                        modified_model=props.material_model,
                        modified_params={"E": props.parameters.get("E", 1e4) * 2.0},
                        source_op=op.name,
                        description="Cauterized tissue — increased stiffness",
                    ))
                    break
    elif method in ("outfracture", "partial_resection"):
        # Simple volume removal
        remove_ids = turb_elems[:n_remove]
    else:
        result.errors.append(f"turbinate_reduction: unknown method '{method}'")
        return

    result.mesh_modifications.append(MeshModification(
        mod_type="remove",
        element_ids=remove_ids,
        source_op=op.name,
        description=f"Turbinate reduction ({method}): remove {len(remove_ids)} elements",
    ))


def _compile_graft(
    op: SurgicalOp,
    mesh: VolumeMesh,
    result: CompilationResult,
) -> None:
    """Compile graft insertion → new elements + material + contact."""
    length_mm = op.params.get("length_mm", 18.0)
    width_mm = op.params.get("width_mm", 3.0)
    thickness_mm = op.params.get("thickness_mm", 1.5)
    height_mm = op.params.get("height_mm", 8.0)
    source_material = op.params.get("source", "septal")

    # Determine graft material properties based on source
    graft_props: Dict[str, float] = {}
    if source_material == "septal":
        graft_props = {"E": 5.0e6, "nu": 0.3}  # Septal cartilage ~5 MPa
    elif source_material == "ear":
        graft_props = {"E": 3.0e6, "nu": 0.3}  # Auricular cartilage ~3 MPa
    elif source_material == "costal":
        graft_props = {"E": 12.0e6, "nu": 0.3}  # Costal cartilage ~12 MPa

    # Find insertion site based on graft type
    affected = op.affected_structures
    if not affected:
        result.warnings.append(f"{op.name}: no affected structures defined")
        return

    primary_structure = affected[0]
    target_nodes = _find_nodes_by_structure(mesh, primary_structure)

    if len(target_nodes) == 0:
        result.warnings.append(f"{op.name}: no target nodes found for {primary_structure.value}")
        return

    target_positions = mesh.nodes[target_nodes]
    graft_center = target_positions.mean(axis=0)

    # Generate simplified graft geometry: a rectangular slab
    # In a full implementation this would be a detailed mesh
    # For now, define the graft as new elements at the insertion site
    n_graft_elems_x = max(2, int(length_mm / 2.0))
    n_graft_elems_y = max(1, int(width_mm / 1.5))
    n_graft_elems_z = max(1, int(thickness_mm / 1.0))

    # Create a structured grid of nodes for the graft
    nx, ny, nz = n_graft_elems_x + 1, n_graft_elems_y + 1, n_graft_elems_z + 1
    dx = length_mm / n_graft_elems_x
    dy = width_mm / n_graft_elems_y
    dz = thickness_mm / n_graft_elems_z

    graft_nodes = np.zeros((nx * ny * nz, 3), dtype=np.float64)
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                graft_nodes[idx] = graft_center + np.array([
                    (ix - nx // 2) * dx,
                    (iy - ny // 2) * dy,
                    (iz - nz // 2) * dz,
                ])
                idx += 1

    # Create hex elements (or tets)
    graft_elements = []
    for iz in range(n_graft_elems_z):
        for iy in range(n_graft_elems_y):
            for ix in range(n_graft_elems_x):
                n0 = iz * ny * nx + iy * nx + ix
                n1 = n0 + 1
                n2 = n0 + nx + 1
                n3 = n0 + nx
                n4 = n0 + ny * nx
                n5 = n4 + 1
                n6 = n4 + nx + 1
                n7 = n4 + nx
                # Split hex into 5 tets for tet-only mesh compatibility
                graft_elements.extend([
                    [n0, n1, n3, n4],
                    [n1, n2, n3, n6],
                    [n1, n4, n5, n6],
                    [n3, n4, n6, n7],
                    [n1, n3, n4, n6],
                ])

    graft_elem_array = np.array(graft_elements, dtype=np.int64)
    new_region_id = max(mesh.region_ids) + 100 if len(mesh.region_ids) > 0 else 100

    result.mesh_modifications.append(MeshModification(
        mod_type="insert",
        new_nodes=graft_nodes,
        new_elements=graft_elem_array,
        new_region_id=new_region_id,
        source_op=op.name,
        description=f"Graft insertion: {op.name} ({source_material})",
    ))

    # Tie graft to surrounding tissue
    result.boundary_conditions.append(BoundaryCondition(
        bc_type=BCType.CONTACT_TIE,
        node_ids=target_nodes[:min(50, len(target_nodes))],
        source_op=op.name,
        metadata={
            "graft_region_id": new_region_id,
            "source_material": source_material,
            "graft_props": graft_props,
        },
    ))

    # Material assignment for the graft
    result.material_modifications.append(MaterialModification(
        region_id=new_region_id,
        element_ids=np.arange(len(graft_elements), dtype=np.int64),
        original_model=MaterialModel.LINEAR_ELASTIC,
        modified_model=MaterialModel.LINEAR_ELASTIC,
        modified_params=graft_props,
        source_op=op.name,
        description=f"Graft material: {source_material} cartilage",
    ))


def _compile_suture(
    op: SurgicalOp,
    mesh: VolumeMesh,
    result: CompilationResult,
) -> None:
    """Compile suture → spring/force BC between node pairs."""
    technique = op.params.get("technique", "transdomal")
    tension = op.params.get("tension", 0.5)

    llc_nodes = _find_nodes_by_structure(mesh, StructureType.CARTILAGE_LOWER_LATERAL)
    if len(llc_nodes) == 0:
        result.warnings.append(f"tip_suture: no lower lateral cartilage nodes found")
        return

    positions = mesh.nodes[llc_nodes]
    center = positions.mean(axis=0)

    # Tension force magnitude scaled by the tension parameter
    # Typical suture forces: 0.1-2.0 N
    force_n = 0.1 + tension * 1.9  # 0.1N to 2.0N

    if technique == "transdomal":
        # Transdomal suture: compress the dome width
        # Find left and right dome nodes (lateral-most on each side)
        left_mask = positions[:, 0] > center[0]
        right_mask = positions[:, 0] < center[0]

        if np.any(left_mask) and np.any(right_mask):
            left_nodes = llc_nodes[left_mask]
            right_nodes = llc_nodes[right_mask]

            # Apply compressive force toward midline
            left_forces = np.zeros((len(left_nodes), 3), dtype=np.float64)
            left_forces[:, 0] = -force_n
            right_forces = np.zeros((len(right_nodes), 3), dtype=np.float64)
            right_forces[:, 0] = force_n

            result.boundary_conditions.append(BoundaryCondition(
                bc_type=BCType.NODAL_FORCE,
                node_ids=left_nodes,
                values=left_forces,
                magnitude=force_n,
                source_op=op.name,
                metadata={"suture": technique, "side": "left"},
            ))
            result.boundary_conditions.append(BoundaryCondition(
                bc_type=BCType.NODAL_FORCE,
                node_ids=right_nodes,
                values=right_forces,
                magnitude=force_n,
                source_op=op.name,
                metadata={"suture": technique, "side": "right"},
            ))

    elif technique == "interdomal":
        # Interdomal suture: bring domes together medially
        # Similar to transdomal but only at the dome peaks
        z_thresh = np.percentile(positions[:, 2], 80)
        dome_mask = positions[:, 2] > z_thresh
        dome_nodes = llc_nodes[dome_mask]
        dome_pos = positions[dome_mask]
        if len(dome_nodes) > 1:
            dome_center = dome_pos.mean(axis=0)
            forces = (dome_center - dome_pos) * force_n
            norms = np.linalg.norm(forces, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            forces = forces / norms * force_n

            result.boundary_conditions.append(BoundaryCondition(
                bc_type=BCType.NODAL_FORCE,
                node_ids=dome_nodes,
                values=forces,
                magnitude=force_n,
                source_op=op.name,
                metadata={"suture": technique},
            ))

    elif technique in ("lateral_crural_spanning", "medial_crural_fixation",
                       "tongue_in_groove"):
        # Generic suture: add spring-like force toward target position
        target_pos = center.copy()
        if technique == "tongue_in_groove":
            # Pull caudal septum posterior
            target_pos[1] -= 3.0  # mm
        forces = np.outer(np.ones(len(llc_nodes)), target_pos - center) * force_n * 0.1
        result.boundary_conditions.append(BoundaryCondition(
            bc_type=BCType.NODAL_FORCE,
            node_ids=llc_nodes,
            values=forces,
            magnitude=force_n,
            source_op=op.name,
            metadata={"suture": technique},
        ))


def _compile_cephalic_trim(
    op: SurgicalOp,
    mesh: VolumeMesh,
    result: CompilationResult,
) -> None:
    """Compile cephalic trim → element removal of LLC cephalic portion."""
    residual_strip_mm = op.params.get("residual_strip_mm", 6.0)
    side = op.params.get("side", "bilateral")

    llc_elems = _find_elements_by_structure(mesh, StructureType.CARTILAGE_LOWER_LATERAL)
    if len(llc_elems) == 0:
        result.warnings.append("cephalic_trim: no LLC elements found")
        return

    centroids = np.zeros((len(llc_elems), 3), dtype=np.float64)
    for i, eid in enumerate(llc_elems):
        centroids[i] = mesh.nodes[mesh.elements[eid]].mean(axis=0)

    center = centroids.mean(axis=0)

    # The cephalic portion is the superior part of the LLC
    z_range = centroids[:, 2].max() - centroids[:, 2].min()
    if z_range < 1e-6:
        return

    # Keep the caudal strip (residual_strip_mm)
    strip_fraction = min(0.9, residual_strip_mm / z_range)
    z_threshold = centroids[:, 2].min() + z_range * strip_fraction

    cephalic_mask = centroids[:, 2] > z_threshold

    if side != "bilateral":
        if side == "left":
            cephalic_mask = cephalic_mask & (centroids[:, 0] > center[0])
        elif side == "right":
            cephalic_mask = cephalic_mask & (centroids[:, 0] < center[0])

    if np.any(cephalic_mask):
        result.mesh_modifications.append(MeshModification(
            mod_type="remove",
            element_ids=llc_elems[cephalic_mask],
            source_op=op.name,
            description=f"Cephalic trim: remove {int(np.sum(cephalic_mask))} elements, "
                        f"preserve {residual_strip_mm}mm strip",
        ))


def _compile_alar_base_reduction(
    op: SurgicalOp,
    mesh: VolumeMesh,
    result: CompilationResult,
) -> None:
    """Compile alar base reduction → element removal + skin closure."""
    amount_mm = op.params.get("amount_mm", 2.0)
    technique = op.params.get("technique", "weir")
    side = op.params.get("side", "bilateral")

    skin_elems = _find_elements_by_structure(mesh, StructureType.SKIN_THICK)
    alar_elems = _find_elements_by_structure(mesh, StructureType.CARTILAGE_ALAR)
    target_elems = np.union1d(skin_elems, alar_elems) if len(alar_elems) > 0 else skin_elems

    if len(target_elems) == 0:
        result.warnings.append("alar_base_reduction: no target elements found")
        return

    centroids = np.zeros((len(target_elems), 3), dtype=np.float64)
    for i, eid in enumerate(target_elems):
        centroids[i] = mesh.nodes[mesh.elements[eid]].mean(axis=0)

    center = centroids.mean(axis=0)

    # Find alar base elements (most lateral, most inferior)
    lateral_scores = np.abs(centroids[:, 0] - center[0])
    z_scores = center[2] - centroids[:, 2]  # inferior is positive
    base_score = lateral_scores + z_scores
    n_remove = max(1, int(len(target_elems) * amount_mm / 10.0))
    order = np.argsort(base_score)[::-1]

    if side != "bilateral":
        if side == "left":
            side_mask = centroids[:, 0] > center[0]
        else:
            side_mask = centroids[:, 0] < center[0]
        valid = order[side_mask[order]]
        remove_ids = target_elems[valid[:n_remove]]
    else:
        remove_ids = target_elems[order[:n_remove]]

    result.mesh_modifications.append(MeshModification(
        mod_type="remove",
        element_ids=remove_ids,
        source_op=op.name,
        description=f"Alar base reduction ({technique}): {len(remove_ids)} elements",
    ))


def _compile_cartilage_scoring(
    op: SurgicalOp,
    mesh: VolumeMesh,
    result: CompilationResult,
) -> None:
    """Compile cartilage scoring → stiffness reduction along score lines."""
    depth_fraction = op.params.get("depth_fraction", 0.5)
    n_scores = op.params.get("n_scores", 3)
    structure_name = op.params.get("structure", "lower_lateral_cartilage")

    # Map structure name to StructureType
    name_map = {
        "lower_lateral_cartilage": StructureType.CARTILAGE_LOWER_LATERAL,
        "upper_lateral_cartilage": StructureType.CARTILAGE_UPPER_LATERAL,
        "septal_cartilage": StructureType.CARTILAGE_SEPTUM,
    }
    structure = name_map.get(structure_name)
    if structure is None:
        result.errors.append(f"cartilage_scoring: unknown structure '{structure_name}'")
        return

    elems = _find_elements_by_structure(mesh, structure)
    if len(elems) == 0:
        result.warnings.append(f"cartilage_scoring: no {structure_name} elements found")
        return

    # Compute centroids
    centroids = np.zeros((len(elems), 3), dtype=np.float64)
    for i, eid in enumerate(elems):
        centroids[i] = mesh.nodes[mesh.elements[eid]].mean(axis=0)

    # Score lines are evenly spaced along the primary axis
    y_min, y_max = centroids[:, 1].min(), centroids[:, 1].max()
    y_range = y_max - y_min
    if y_range < 1e-6:
        return

    score_width = 0.5  # mm — width of each score line
    stiffness_reduction = 1.0 - depth_fraction  # deeper score → softer

    for i_score in range(n_scores):
        y_pos = y_min + (i_score + 1) * y_range / (n_scores + 1)
        score_mask = np.abs(centroids[:, 1] - y_pos) < score_width
        score_elems = elems[score_mask]

        if len(score_elems) == 0:
            continue

        for rid, props in mesh.region_materials.items():
            if props.structure_type == structure:
                result.material_modifications.append(MaterialModification(
                    region_id=rid,
                    element_ids=score_elems,
                    original_model=props.material_model,
                    modified_model=props.material_model,
                    modified_params={
                        k: v * stiffness_reduction
                        for k, v in props.parameters.items()
                        if isinstance(v, (int, float))
                    },
                    source_op=op.name,
                    description=f"Score line {i_score + 1}/{n_scores} at y={y_pos:.1f}mm",
                ))
                break


# ── Operator dispatch ─────────────────────────────────────────────

_OP_COMPILERS = {
    "dorsal_reduction": _compile_dorsal_reduction,
    "lateral_osteotomy": lambda op, mesh, res: _compile_osteotomy(op, mesh, res, lateral=True),
    "medial_osteotomy": lambda op, mesh, res: _compile_osteotomy(op, mesh, res, lateral=False),
    "bone_infracture": _compile_infracture,
    "septoplasty": _compile_septoplasty,
    "turbinate_reduction": _compile_turbinate_reduction,
    "spreader_graft": _compile_graft,
    "columellar_strut": _compile_graft,
    "shield_graft": _compile_graft,
    "cephalic_trim": _compile_cephalic_trim,
    "tip_suture": _compile_suture,
    "alar_base_reduction": _compile_alar_base_reduction,
    "cartilage_scoring": _compile_cartilage_scoring,
}


# ── Main compiler ─────────────────────────────────────────────────

class PlanCompiler:
    """Compile a SurgicalPlan into solver-ready boundary conditions.

    The compiler walks the plan DAG in execution order,
    translating each SurgicalOp into BCs, material modifications,
    and mesh modifications that can be consumed by the FEM/CFD solvers.
    """

    def __init__(self, mesh: VolumeMesh) -> None:
        self._mesh = mesh
        self._mesh_hash = self._compute_mesh_hash()

    def compile(self, plan: SurgicalPlan) -> CompilationResult:
        """Compile a full surgical plan.

        Returns a CompilationResult containing all BCs and modifications.
        """
        # Validate plan first
        validation_errors = plan.validate()
        if validation_errors:
            return CompilationResult(
                plan_hash=plan.content_hash(),
                mesh_hash=self._mesh_hash,
                errors=validation_errors,
            )

        result = CompilationResult(
            plan_hash=plan.content_hash(),
            mesh_hash=self._mesh_hash,
        )

        # Walk the plan in execution order
        ops = plan.all_ops()
        logger.info("Compiling %d operations from plan '%s'", len(ops), plan.name)

        for op in ops:
            self._compile_op(op, result)

        # Determine solver type based on operations
        result.solver_type = self._infer_solver_type(ops)
        result.n_load_steps = self._infer_load_steps(ops, result)

        # Add boundary fixation BCs
        self._add_boundary_fixation(result)

        logger.info("Compilation complete: %s", result.summary())
        return result

    def _compile_op(self, op: SurgicalOp, result: CompilationResult) -> None:
        """Compile a single SurgicalOp."""
        compiler_fn = _OP_COMPILERS.get(op.name)
        if compiler_fn is None:
            result.warnings.append(
                f"No compiler registered for operator '{op.name}', skipping"
            )
            return

        try:
            compiler_fn(op, self._mesh, result)
        except Exception as exc:
            result.errors.append(f"Error compiling '{op.name}': {exc}")
            logger.error("Error compiling '%s': %s", op.name, exc, exc_info=True)

    def _infer_solver_type(self, ops: List[SurgicalOp]) -> SolverType:
        """Infer the appropriate solver type from operations."""
        has_osteotomy = any(o.category == OpCategory.OSTEOTOMY for o in ops)
        has_suture = any(o.category == OpCategory.SUTURE for o in ops)
        has_large_deformation = any(
            o.params.get("amount_mm", 0) > 3.0 or
            o.params.get("displacement_mm", 0) > 3.0
            for o in ops
        )

        if has_osteotomy or has_large_deformation:
            return SolverType.FEM_QUASISTATIC
        if has_suture:
            return SolverType.FEM_QUASISTATIC
        return SolverType.FEM_STATIC

    def _infer_load_steps(
        self,
        ops: List[SurgicalOp],
        result: CompilationResult,
    ) -> int:
        """Infer number of load steps for quasi-static analysis."""
        max_disp = 0.0
        for bc in result.boundary_conditions:
            if bc.bc_type == BCType.NODAL_DISPLACEMENT:
                max_disp = max(max_disp, bc.magnitude)

        if max_disp < 1.0:
            return 5
        elif max_disp < 3.0:
            return 10
        elif max_disp < 5.0:
            return 20
        else:
            return 50

    def _add_boundary_fixation(self, result: CompilationResult) -> None:
        """Add fixed BCs at the mesh boundary to prevent rigid body motion."""
        # Fix nodes at the posterior boundary (furthest from nose)
        if self._mesh.n_nodes == 0:
            return

        y_min = self._mesh.nodes[:, 1].min()
        y_range = self._mesh.nodes[:, 1].max() - y_min
        if y_range < 1e-6:
            return

        # Fix posterior 10% of nodes
        posterior_thresh = y_min + y_range * 0.10
        posterior_mask = self._mesh.nodes[:, 1] < posterior_thresh
        posterior_nodes = np.where(posterior_mask)[0].astype(np.int64)

        if len(posterior_nodes) > 0:
            result.boundary_conditions.append(BoundaryCondition(
                bc_type=BCType.FIXED,
                node_ids=posterior_nodes,
                source_op="boundary_fixation",
                metadata={"region": "posterior"},
            ))

        # Check for surface-tagged boundaries
        for tag in ("fixed", "posterior", "boundary"):
            if tag in self._mesh.surface_tags:
                tagged_nodes = self._mesh.surface_tags[tag]
                result.boundary_conditions.append(BoundaryCondition(
                    bc_type=BCType.FIXED,
                    node_ids=tagged_nodes.astype(np.int64),
                    source_op="boundary_fixation",
                    metadata={"tag": tag},
                ))

    def _compute_mesh_hash(self) -> str:
        """Compute a deterministic hash of the mesh geometry."""
        data = {
            "n_nodes": self._mesh.n_nodes,
            "n_elements": self._mesh.n_elements,
            "element_type": self._mesh.element_type.value,
        }
        if self._mesh.n_nodes > 0:
            # Hash a subsample for efficiency
            step = max(1, self._mesh.n_nodes // 1000)
            sample = self._mesh.nodes[::step].tobytes()
            data["node_sample_hash"] = hashlib.sha256(sample).hexdigest()[:16]
        return hash_dict(data)
