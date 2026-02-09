"""
Export Layer — VTK/VTU, HDF5-XDMF, CSV, JSON result serialisation.

Converts platform ``FieldData`` / ``SimulationState`` objects into standard
engineering file formats for interoperability with ParaView, VisIt, and
downstream analysis pipelines.

Supported formats
-----------------
* **VTK / VTU** — XML-based VTK UnstructuredGrid (``.vtu``).  Works with
  both structured and unstructured meshes (structured grids are exported as
  explicit unstructured grids for maximum compatibility).
* **XDMF + HDF5** — Lightweight XML topology pointing at binary HDF5 arrays.
  Requires ``h5py`` at runtime.
* **CSV** — Scalar observable histories and convergence data.
* **JSON** — Metadata, provenance, observable snapshots.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import struct
import xml.etree.ElementTree as ET
from base64 import b64encode
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from tensornet.platform.data_model import (
    FieldData,
    Mesh,
    SimulationState,
    StructuredMesh,
    UnstructuredMesh,
)

logger = logging.getLogger(__name__)

__all__ = [
    "export_vtu",
    "export_xdmf_hdf5",
    "export_csv",
    "export_json",
    "ExportBundle",
]


# ═══════════════════════════════════════════════════════════════════════════════
# VTK / VTU Export
# ═══════════════════════════════════════════════════════════════════════════════


def _mesh_to_points_cells(mesh: Mesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a ``Mesh`` to raw numpy arrays for VTU export.

    Returns
    -------
    points : (n_points, 3)  float64
    connectivity : (n_cells * vpn,)  int64
    offsets : (n_cells,)  int64
    """
    if isinstance(mesh, StructuredMesh):
        return _structured_to_vtu_arrays(mesh)
    elif isinstance(mesh, UnstructuredMesh):
        return _unstructured_to_vtu_arrays(mesh)
    else:
        # Generic fallback: one vertex per cell, at cell centers
        centers = mesh.cell_centers().detach().cpu().numpy()
        n = centers.shape[0]
        pts = np.zeros((n, 3), dtype=np.float64)
        pts[:, : centers.shape[1]] = centers
        conn = np.arange(n, dtype=np.int64)
        offsets = np.arange(1, n + 1, dtype=np.int64)
        return pts, conn, offsets


def _structured_to_vtu_arrays(
    mesh: StructuredMesh,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a ``StructuredMesh`` to vertex-based VTU arrays.

    For a 1-D mesh with N cells we produce N+1 vertices and N line cells.
    For 2-D (Nx, Ny) we produce (Nx+1)*(Ny+1) vertices and Nx*Ny quad cells.
    For 3-D analogous hexahedra.
    """
    ndim = mesh.ndim
    shape = mesh.shape

    # Build per-axis vertex coordinates
    vertex_coords: List[np.ndarray] = []
    for dim_idx in range(ndim):
        lo, hi = mesh.domain[dim_idx]
        n = shape[dim_idx]
        vertex_coords.append(np.linspace(lo, hi, n + 1, dtype=np.float64))

    if ndim == 1:
        pts = np.zeros((shape[0] + 1, 3), dtype=np.float64)
        pts[:, 0] = vertex_coords[0]
        conn = np.empty(shape[0] * 2, dtype=np.int64)
        for i in range(shape[0]):
            conn[2 * i] = i
            conn[2 * i + 1] = i + 1
        offsets = np.arange(2, 2 * shape[0] + 1, 2, dtype=np.int64)
        return pts, conn, offsets

    elif ndim == 2:
        nx, ny = shape
        gx, gy = np.meshgrid(vertex_coords[0], vertex_coords[1], indexing="ij")
        nv = (nx + 1) * (ny + 1)
        pts = np.zeros((nv, 3), dtype=np.float64)
        pts[:, 0] = gx.ravel()
        pts[:, 1] = gy.ravel()

        conn_list: List[int] = []
        for i in range(nx):
            for j in range(ny):
                v0 = i * (ny + 1) + j
                v1 = v0 + 1
                v2 = (i + 1) * (ny + 1) + j + 1
                v3 = (i + 1) * (ny + 1) + j
                conn_list.extend([v0, v1, v2, v3])
        conn = np.array(conn_list, dtype=np.int64)
        offsets = np.arange(4, 4 * nx * ny + 1, 4, dtype=np.int64)
        return pts, conn, offsets

    else:  # 3-D
        nx, ny, nz = shape
        gx, gy, gz = np.meshgrid(
            vertex_coords[0], vertex_coords[1], vertex_coords[2], indexing="ij"
        )
        nv = (nx + 1) * (ny + 1) * (nz + 1)
        pts = np.zeros((nv, 3), dtype=np.float64)
        pts[:, 0] = gx.ravel()
        pts[:, 1] = gy.ravel()
        pts[:, 2] = gz.ravel()

        def vidx(i: int, j: int, k: int) -> int:
            return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k

        conn_list = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    conn_list.extend([
                        vidx(i, j, k),
                        vidx(i + 1, j, k),
                        vidx(i + 1, j + 1, k),
                        vidx(i, j + 1, k),
                        vidx(i, j, k + 1),
                        vidx(i + 1, j, k + 1),
                        vidx(i + 1, j + 1, k + 1),
                        vidx(i, j + 1, k + 1),
                    ])
        conn = np.array(conn_list, dtype=np.int64)
        offsets = np.arange(8, 8 * nx * ny * nz + 1, 8, dtype=np.int64)
        return pts, conn, offsets


def _unstructured_to_vtu_arrays(
    mesh: UnstructuredMesh,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes = mesh.nodes.detach().cpu().numpy()
    pts = np.zeros((nodes.shape[0], 3), dtype=np.float64)
    pts[:, : nodes.shape[1]] = nodes

    elems = mesh.elements.detach().cpu().numpy().astype(np.int64)
    vpn = elems.shape[1]
    conn = elems.ravel()
    offsets = np.arange(vpn, vpn * mesh.n_cells + 1, vpn, dtype=np.int64)
    return pts, conn, offsets


def _vtk_cell_type(mesh: Mesh) -> int:
    """Return VTK cell-type integer for the mesh."""
    if isinstance(mesh, StructuredMesh):
        ndim = mesh.ndim
        if ndim == 1:
            return 3  # VTK_LINE
        elif ndim == 2:
            return 9  # VTK_QUAD
        else:
            return 12  # VTK_HEXAHEDRON
    elif isinstance(mesh, UnstructuredMesh):
        vpn = mesh.elements.shape[1]
        _map = {2: 3, 3: 5, 4: 10, 8: 12}  # line, tri, tet, hex
        return _map.get(vpn, 7)  # VTK_POLYGON fallback
    return 1  # VTK_VERTEX


def _encode_array_b64(arr: np.ndarray) -> str:
    """Encode a numpy array as base64 for inline VTU data."""
    raw = arr.tobytes()
    # VTK appended/inline base64 expects 4-byte header with byte count
    header = struct.pack("<I", len(raw))
    return b64encode(header + raw).decode("ascii")


def export_vtu(
    state: SimulationState,
    path: Union[str, Path],
    *,
    fields: Optional[Sequence[str]] = None,
    binary: bool = True,
) -> Path:
    """
    Export a ``SimulationState`` as a VTK UnstructuredGrid (``.vtu``).

    Parameters
    ----------
    state : SimulationState
    path : file path (directory + filename, or just filename)
    fields : list of field names to include (default: all)
    binary : if True, inline base64 encoding; else ASCII

    Returns
    -------
    Path to the written file.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    mesh = state.mesh
    pts, conn, offsets = _mesh_to_points_cells(mesh)
    n_pts = pts.shape[0]
    n_cells = mesh.n_cells
    cell_type = _vtk_cell_type(mesh)

    # Build XML tree
    root = ET.Element("VTKFile", type="UnstructuredGrid", version="1.0",
                       byte_order="LittleEndian")
    ug = ET.SubElement(root, "UnstructuredGrid")
    piece = ET.SubElement(ug, "Piece",
                          NumberOfPoints=str(n_pts),
                          NumberOfCells=str(n_cells))

    # -- Points --
    points_elem = ET.SubElement(piece, "Points")
    _write_data_array(points_elem, "Points", pts.ravel(), 3, binary)

    # -- Cells --
    cells_elem = ET.SubElement(piece, "Cells")
    _write_data_array(cells_elem, "connectivity", conn, 1, binary, dtype_name="Int64")
    _write_data_array(cells_elem, "offsets", offsets, 1, binary, dtype_name="Int64")
    types_arr = np.full(n_cells, cell_type, dtype=np.uint8)
    _write_data_array(cells_elem, "types", types_arr, 1, binary, dtype_name="UInt8")

    # -- CellData --
    field_names = fields or list(state.fields.keys())
    if field_names:
        cd = ET.SubElement(piece, "CellData")
        for fname in field_names:
            if fname not in state.fields:
                continue
            fdata = state.fields[fname]
            arr = fdata.data.detach().cpu().to(torch.float64).numpy()
            ncomp = fdata.components
            if arr.ndim == 1:
                _write_data_array(cd, fname, arr, 1, binary)
            else:
                _write_data_array(cd, fname, arr.ravel(), ncomp, binary)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(out), xml_declaration=True, encoding="utf-8")
    logger.info("Exported VTU: %s (%d points, %d cells)", out, n_pts, n_cells)
    return out


def _write_data_array(
    parent: ET.Element,
    name: str,
    data: np.ndarray,
    n_components: int,
    binary: bool,
    dtype_name: Optional[str] = None,
) -> None:
    """Append a DataArray element to *parent*."""
    if dtype_name is None:
        dtype_name = "Float64"
    fmt = "binary" if binary else "ascii"
    da = ET.SubElement(parent, "DataArray",
                       type=dtype_name,
                       Name=name,
                       NumberOfComponents=str(n_components),
                       format=fmt)
    if binary:
        if dtype_name == "Float64":
            arr = data.astype(np.float64)
        elif dtype_name == "Int64":
            arr = data.astype(np.int64)
        elif dtype_name == "UInt8":
            arr = data.astype(np.uint8)
        else:
            arr = data
        da.text = _encode_array_b64(arr)
    else:
        da.text = " ".join(str(v) for v in data.ravel())


# ═══════════════════════════════════════════════════════════════════════════════
# XDMF + HDF5 Export
# ═══════════════════════════════════════════════════════════════════════════════


def export_xdmf_hdf5(
    state: SimulationState,
    path: Union[str, Path],
    *,
    fields: Optional[Sequence[str]] = None,
) -> Tuple[Path, Path]:
    """
    Export a ``SimulationState`` as XDMF + HDF5.

    Returns (xdmf_path, hdf5_path).

    Requires ``h5py``.  If not installed, raises ``ImportError`` with a
    clear message.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "XDMF/HDF5 export requires h5py.  Install with: pip install h5py"
        ) from exc

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    h5_path = out.with_suffix(".h5")
    xdmf_path = out.with_suffix(".xdmf")

    mesh = state.mesh
    pts, conn, offsets = _mesh_to_points_cells(mesh)
    n_pts = pts.shape[0]
    n_cells = mesh.n_cells

    # Determine topology type string for XDMF
    topo_map = {3: "Polyline", 5: "Triangle", 9: "Quadrilateral",
                10: "Tetrahedron", 12: "Hexahedron"}
    ct = _vtk_cell_type(mesh)
    topo_type = topo_map.get(ct, "Mixed")

    # Compute vertices per cell from offsets
    if len(offsets) > 0:
        vpn = int(offsets[0])
    else:
        vpn = 1

    # Write HDF5
    with h5py.File(str(h5_path), "w") as hf:
        hf.create_dataset("mesh/points", data=pts)
        hf.create_dataset("mesh/connectivity", data=conn.reshape(n_cells, vpn))
        field_names = fields or list(state.fields.keys())
        for fname in field_names:
            if fname not in state.fields:
                continue
            arr = state.fields[fname].data.detach().cpu().to(torch.float64).numpy()
            hf.create_dataset(f"fields/{fname}", data=arr)
        hf.attrs["t"] = state.t
        hf.attrs["step_index"] = state.step_index

    # Write XDMF
    h5_name = h5_path.name
    xdmf_root = ET.Element("Xdmf", Version="3.0")
    domain = ET.SubElement(xdmf_root, "Domain")
    grid = ET.SubElement(domain, "Grid", Name="mesh", GridType="Uniform")

    # Time
    ET.SubElement(grid, "Time", Value=str(state.t))

    # Topology
    topo = ET.SubElement(grid, "Topology", TopologyType=topo_type,
                         NumberOfElements=str(n_cells))
    topo_data = ET.SubElement(topo, "DataItem",
                              Dimensions=f"{n_cells} {vpn}",
                              NumberType="Int", Format="HDF")
    topo_data.text = f"{h5_name}:/mesh/connectivity"

    # Geometry
    geom = ET.SubElement(grid, "Geometry", GeometryType="XYZ")
    geom_data = ET.SubElement(geom, "DataItem",
                              Dimensions=f"{n_pts} 3",
                              NumberType="Float", Precision="8",
                              Format="HDF")
    geom_data.text = f"{h5_name}:/mesh/points"

    # Attributes (fields)
    field_names = fields or list(state.fields.keys())
    for fname in field_names:
        if fname not in state.fields:
            continue
        fdata = state.fields[fname]
        attr_type = "Scalar" if fdata.components == 1 else "Vector"
        attr = ET.SubElement(grid, "Attribute", Name=fname,
                             AttributeType=attr_type, Center="Cell")
        dims = str(n_cells) if fdata.components == 1 else f"{n_cells} {fdata.components}"
        aitem = ET.SubElement(attr, "DataItem", Dimensions=dims,
                              NumberType="Float", Precision="8",
                              Format="HDF")
        aitem.text = f"{h5_name}:/fields/{fname}"

    tree = ET.ElementTree(xdmf_root)
    ET.indent(tree, space="  ")
    tree.write(str(xdmf_path), xml_declaration=True, encoding="utf-8")

    logger.info("Exported XDMF+HDF5: %s + %s", xdmf_path, h5_path)
    return xdmf_path, h5_path


# ═══════════════════════════════════════════════════════════════════════════════
# CSV Export
# ═══════════════════════════════════════════════════════════════════════════════


def export_csv(
    data: Dict[str, List[float]],
    path: Union[str, Path],
    *,
    header_comment: Optional[str] = None,
) -> Path:
    """
    Export tabular data (observable histories, convergence data) to CSV.

    Parameters
    ----------
    data : dict mapping column-name → list of values
        All lists must have the same length.
    path : output file path
    header_comment : optional comment line prepended as ``# ...``

    Returns
    -------
    Path to the written file.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    columns = list(data.keys())
    n_rows = len(next(iter(data.values())))
    for col, vals in data.items():
        if len(vals) != n_rows:
            raise ValueError(
                f"Column '{col}' has {len(vals)} rows, expected {n_rows}"
            )

    with out.open("w", newline="") as f:
        if header_comment:
            f.write(f"# {header_comment}\n")
        writer = csv.writer(f)
        writer.writerow(columns)
        for i in range(n_rows):
            writer.writerow([data[col][i] for col in columns])

    logger.info("Exported CSV: %s (%d rows, %d cols)", out, n_rows, len(columns))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# JSON Export
# ═══════════════════════════════════════════════════════════════════════════════


def export_json(
    payload: Dict[str, Any],
    path: Union[str, Path],
) -> Path:
    """
    Export a dictionary as pretty-printed JSON.

    Tensor values are converted to nested lists.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    def _default(obj: Any) -> Any:
        if isinstance(obj, Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with out.open("w") as f:
        json.dump(payload, f, indent=2, default=_default)

    logger.info("Exported JSON: %s", out)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Export Bundle — convenience wrapper
# ═══════════════════════════════════════════════════════════════════════════════


class ExportBundle:
    """
    Convenience class that exports a ``SimulationState`` to multiple formats
    in one call.

    Usage::

        bundle = ExportBundle(state, output_dir="results/run_001")
        bundle.vtu("solution.vtu")
        bundle.csv(observable_history, "convergence.csv")
        bundle.json(provenance_dict, "provenance.json")
        paths = bundle.all("solution")  # VTU + JSON metadata
    """

    def __init__(
        self,
        state: SimulationState,
        output_dir: Union[str, Path] = ".",
    ) -> None:
        self.state = state
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._written: List[Path] = []

    def vtu(
        self,
        filename: str = "solution.vtu",
        *,
        fields: Optional[Sequence[str]] = None,
    ) -> Path:
        p = export_vtu(self.state, self.output_dir / filename, fields=fields)
        self._written.append(p)
        return p

    def xdmf(
        self,
        filename: str = "solution",
        *,
        fields: Optional[Sequence[str]] = None,
    ) -> Tuple[Path, Path]:
        xp, hp = export_xdmf_hdf5(
            self.state, self.output_dir / filename, fields=fields
        )
        self._written.extend([xp, hp])
        return xp, hp

    def csv(
        self,
        data: Dict[str, List[float]],
        filename: str = "observables.csv",
    ) -> Path:
        p = export_csv(data, self.output_dir / filename)
        self._written.append(p)
        return p

    def json(
        self,
        payload: Dict[str, Any],
        filename: str = "metadata.json",
    ) -> Path:
        p = export_json(payload, self.output_dir / filename)
        self._written.append(p)
        return p

    def all(
        self,
        stem: str = "solution",
        *,
        fields: Optional[Sequence[str]] = None,
    ) -> List[Path]:
        """Export VTU + JSON metadata.  Returns list of paths written."""
        paths: List[Path] = []
        paths.append(self.vtu(f"{stem}.vtu", fields=fields))
        meta = {
            "t": self.state.t,
            "step_index": self.state.step_index,
            "fields": list(self.state.fields.keys()),
            "n_cells": self.state.mesh.n_cells,
            "ndim": self.state.mesh.ndim,
        }
        meta.update(self.state.metadata)
        paths.append(self.json(meta, f"{stem}_meta.json"))
        return paths

    @property
    def written_files(self) -> List[Path]:
        return list(self._written)
