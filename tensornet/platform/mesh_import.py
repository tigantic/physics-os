"""
Mesh Import — GMSH ``.msh`` (v2 and v4 ASCII) and raw node/element formats.

Converts external mesh files into platform ``StructuredMesh`` or
``UnstructuredMesh`` objects so that users can bring their own geometry.

Supported formats
-----------------
* **GMSH v2 ASCII** (``.msh``) — triangles, quads, tetrahedra, hexahedra.
* **GMSH v4 ASCII** (``.msh``) — same element types, v4 section layout.
* **Raw arrays** — direct construction from (nodes, elements) numpy/torch arrays.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from tensornet.platform.data_model import (
    Mesh,
    StructuredMesh,
    UnstructuredMesh,
)

logger = logging.getLogger(__name__)

__all__ = [
    "import_gmsh",
    "import_raw",
    "detect_mesh_format",
    "MeshImportError",
]


class MeshImportError(Exception):
    """Raised when a mesh file cannot be parsed."""


# ═══════════════════════════════════════════════════════════════════════════════
# Format Detection
# ═══════════════════════════════════════════════════════════════════════════════


def detect_mesh_format(path: Union[str, Path]) -> str:
    """
    Detect mesh format from file header.

    Returns ``'gmsh2'``, ``'gmsh4'``, or ``'unknown'``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Mesh file not found: {p}")

    with p.open("r") as f:
        first_line = f.readline().strip()

    if first_line == "$MeshFormat":
        with p.open("r") as f:
            f.readline()  # $MeshFormat
            version_line = f.readline().strip()
        version = version_line.split()[0]
        major = int(version.split(".")[0])
        if major >= 4:
            return "gmsh4"
        return "gmsh2"

    return "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# GMSH v2 ASCII Import
# ═══════════════════════════════════════════════════════════════════════════════

# GMSH element type → (name, n_nodes)
_GMSH_ELEM_TYPES: Dict[int, Tuple[str, int]] = {
    1: ("line", 2),
    2: ("triangle", 3),
    3: ("quad", 4),
    4: ("tetrahedron", 4),
    5: ("hexahedron", 8),
    15: ("point", 1),
}


def _parse_gmsh_v2(path: Path) -> UnstructuredMesh:
    """Parse a GMSH v2 ASCII ``.msh`` file."""
    nodes: List[List[float]] = []
    elements: List[List[int]] = []
    physical_tags: List[int] = []
    elem_max_dim = 0

    with path.open("r") as f:
        line = ""
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()

            if line == "$Nodes":
                n_nodes = int(f.readline().strip())
                for _ in range(n_nodes):
                    parts = f.readline().strip().split()
                    # parts: node_id x y z
                    nodes.append([float(parts[1]), float(parts[2]), float(parts[3])])
                f.readline()  # $EndNodes

            elif line == "$Elements":
                n_elems = int(f.readline().strip())
                for _ in range(n_elems):
                    parts = f.readline().strip().split()
                    etype = int(parts[1])
                    if etype not in _GMSH_ELEM_TYPES:
                        continue  # skip unsupported element types
                    ename, enodes = _GMSH_ELEM_TYPES[etype]
                    n_tags = int(parts[2])
                    phys_tag = int(parts[3]) if n_tags > 0 else 0
                    node_start = 3 + n_tags
                    elem_nodes = [
                        int(parts[node_start + i]) - 1  # 0-based
                        for i in range(enodes)
                    ]

                    # Track element dimensionality
                    elem_dim = {
                        "point": 0, "line": 1, "triangle": 2,
                        "quad": 2, "tetrahedron": 3, "hexahedron": 3,
                    }[ename]
                    if elem_dim > elem_max_dim:
                        elem_max_dim = elem_dim
                        elements.clear()
                        physical_tags.clear()

                    if elem_dim == elem_max_dim:
                        elements.append(elem_nodes)
                        physical_tags.append(phys_tag)

                f.readline()  # $EndElements

    if not nodes:
        raise MeshImportError(f"No nodes found in {path}")
    if not elements:
        raise MeshImportError(f"No volume/surface elements found in {path}")

    nodes_arr = np.array(nodes, dtype=np.float64)
    elems_arr = np.array(elements, dtype=np.int64)

    # Determine spatial dimension from node coordinates
    ndim = 3
    if np.allclose(nodes_arr[:, 2], 0.0):
        ndim = 2
        if np.allclose(nodes_arr[:, 1], 0.0):
            ndim = 1

    nodes_t = torch.tensor(nodes_arr[:, :ndim], dtype=torch.float64)
    elems_t = torch.tensor(elems_arr, dtype=torch.long)

    mesh = UnstructuredMesh(nodes=nodes_t, elements=elems_t)
    logger.info(
        "Imported GMSH v2: %s (%d nodes, %d elements, %dD)",
        path, len(nodes), len(elements), ndim,
    )
    return mesh


# ═══════════════════════════════════════════════════════════════════════════════
# GMSH v4 ASCII Import
# ═══════════════════════════════════════════════════════════════════════════════


def _parse_gmsh_v4(path: Path) -> UnstructuredMesh:
    """Parse a GMSH v4 ASCII ``.msh`` file."""
    nodes_dict: Dict[int, List[float]] = {}
    elements: List[List[int]] = []
    elem_max_dim = 0

    with path.open("r") as f:
        line = ""
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()

            if line == "$Nodes":
                header = f.readline().strip().split()
                n_blocks = int(header[0])
                for _ in range(n_blocks):
                    block_header = f.readline().strip().split()
                    # entity_dim, entity_tag, parametric, n_nodes_block
                    n_nodes_block = int(block_header[3])
                    node_ids: List[int] = []
                    for _ in range(n_nodes_block):
                        node_ids.append(int(f.readline().strip()))
                    for nid in node_ids:
                        coords = [float(c) for c in f.readline().strip().split()]
                        nodes_dict[nid] = coords
                f.readline()  # $EndNodes

            elif line == "$Elements":
                header = f.readline().strip().split()
                n_blocks = int(header[0])
                for _ in range(n_blocks):
                    block_header = f.readline().strip().split()
                    entity_dim = int(block_header[0])
                    etype = int(block_header[2])
                    n_elems_block = int(block_header[3])

                    if etype not in _GMSH_ELEM_TYPES:
                        for _ in range(n_elems_block):
                            f.readline()
                        continue

                    ename, enodes = _GMSH_ELEM_TYPES[etype]
                    elem_dim = {
                        "point": 0, "line": 1, "triangle": 2,
                        "quad": 2, "tetrahedron": 3, "hexahedron": 3,
                    }[ename]

                    for _ in range(n_elems_block):
                        parts = [int(x) for x in f.readline().strip().split()]
                        if elem_dim > elem_max_dim:
                            elem_max_dim = elem_dim
                            elements.clear()
                        if elem_dim == elem_max_dim:
                            # parts[0] is element tag, rest are node tags
                            elem_nodes = [p - 1 for p in parts[1: 1 + enodes]]
                            elements.append(elem_nodes)

                f.readline()  # $EndElements

    if not nodes_dict:
        raise MeshImportError(f"No nodes found in {path}")
    if not elements:
        raise MeshImportError(f"No volume/surface elements found in {path}")

    # Build contiguous node array (node IDs may not be sequential)
    max_id = max(nodes_dict.keys())
    id_remap: Dict[int, int] = {}
    ordered_nodes: List[List[float]] = []
    for i, (nid, coords) in enumerate(sorted(nodes_dict.items())):
        id_remap[nid] = i
        ordered_nodes.append(coords)

    # Remap element node indices
    remapped_elements = [
        [id_remap[n + 1] for n in elem]  # elem nodes are 0-based, dict keys are 1-based
        for elem in elements
    ]

    nodes_arr = np.array(ordered_nodes, dtype=np.float64)
    elems_arr = np.array(remapped_elements, dtype=np.int64)

    ndim = 3
    if np.allclose(nodes_arr[:, 2], 0.0):
        ndim = 2
        if np.allclose(nodes_arr[:, 1], 0.0):
            ndim = 1

    nodes_t = torch.tensor(nodes_arr[:, :ndim], dtype=torch.float64)
    elems_t = torch.tensor(elems_arr, dtype=torch.long)

    mesh = UnstructuredMesh(nodes=nodes_t, elements=elems_t)
    logger.info(
        "Imported GMSH v4: %s (%d nodes, %d elements, %dD)",
        path, len(ordered_nodes), len(elements), ndim,
    )
    return mesh


# ═══════════════════════════════════════════════════════════════════════════════
# Public Import Functions
# ═══════════════════════════════════════════════════════════════════════════════


def import_gmsh(path: Union[str, Path]) -> UnstructuredMesh:
    """
    Import a GMSH ``.msh`` file (v2 or v4 ASCII).

    Parameters
    ----------
    path : path to the ``.msh`` file.

    Returns
    -------
    UnstructuredMesh

    Raises
    ------
    MeshImportError
        If the file cannot be parsed.
    FileNotFoundError
        If the file does not exist.
    """
    p = Path(path)
    fmt = detect_mesh_format(p)
    if fmt == "gmsh2":
        return _parse_gmsh_v2(p)
    elif fmt == "gmsh4":
        return _parse_gmsh_v4(p)
    else:
        raise MeshImportError(
            f"Unrecognised mesh format in {p}.  Expected GMSH v2 or v4 ASCII."
        )


def import_raw(
    nodes: Union[np.ndarray, Tensor],
    elements: Union[np.ndarray, Tensor],
) -> UnstructuredMesh:
    """
    Create an ``UnstructuredMesh`` from raw node/element arrays.

    Parameters
    ----------
    nodes : (n_nodes, ndim) array
    elements : (n_elements, nodes_per_element) array (0-based)

    Returns
    -------
    UnstructuredMesh
    """
    if isinstance(nodes, np.ndarray):
        nodes = torch.tensor(nodes, dtype=torch.float64)
    else:
        nodes = nodes.to(torch.float64)
    if isinstance(elements, np.ndarray):
        elements = torch.tensor(elements, dtype=torch.long)
    else:
        elements = elements.to(torch.long)

    return UnstructuredMesh(nodes=nodes, elements=elements)
