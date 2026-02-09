"""
Checkpoint / Serialization Layer
=================================

Provides ``save_checkpoint`` / ``load_checkpoint`` for ``SimulationState``
objects plus extensible ``CheckpointStore`` for different backends.

Format
------
Each checkpoint is a directory containing:

    checkpoint/
      meta.json        – provenance, step index, timestamp, schema version
      fields/
        <name>.pt      – per-field tensor data (torch.save)
      mesh.pt          – serialized mesh description
      observables.json – observable history (if present)

The format is intentionally simple (no HDF5 dependency at V0.1).
A Zarr/HDF5 backend can be added at V0.4+ via the ``CheckpointStore``
protocol.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor

from tensornet.platform.data_model import (
    FieldData,
    Mesh,
    SimulationState,
    StructuredMesh,
    UnstructuredMesh,
)
from tensornet.platform.reproduce import ArtifactHash, hash_tensor

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


# ═══════════════════════════════════════════════════════════════════════════════
# CheckpointStore ABC
# ═══════════════════════════════════════════════════════════════════════════════


class CheckpointStore:
    """
    Pluggable store for checkpoint backends.

    The default implementation writes to local filesystem.  Override for
    cloud / HDF5 / Zarr backends.
    """

    def __init__(self, root: Union[str, Path]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def list_checkpoints(self) -> List[str]:
        return sorted(
            d.name for d in self.root.iterdir()
            if d.is_dir() and (d / "meta.json").exists()
        )

    def checkpoint_path(self, name: str) -> Path:
        return self.root / name


# ═══════════════════════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════════════════════


def save_checkpoint(
    state: SimulationState,
    path: Union[str, Path],
    *,
    name: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Persist a ``SimulationState`` to disk.

    Parameters
    ----------
    state : SimulationState
    path : str or Path
        Directory to write into.  Created if absent.
    name : str, optional
        Checkpoint subdirectory name.  Defaults to ``step_{step_index}``.
    extra_metadata : dict, optional
        Additional metadata to embed in ``meta.json``.

    Returns
    -------
    Path
        The checkpoint directory.
    """
    base = Path(path)
    ckpt_name = name or f"step_{state.step_index:06d}"
    ckpt_dir = base / ckpt_name
    fields_dir = ckpt_dir / "fields"
    fields_dir.mkdir(parents=True, exist_ok=True)

    # ── fields ──
    field_meta: Dict[str, Dict[str, Any]] = {}
    for fname, fdata in state.fields.items():
        torch.save(fdata.data, fields_dir / f"{fname}.pt")
        field_meta[fname] = {
            "components": fdata.components,
            "units": fdata.units,
            "shape": list(fdata.data.shape),
            "dtype": str(fdata.data.dtype),
            "hash": hash_tensor(fdata.data).digest,
        }

    # ── mesh ──
    mesh_dict = _serialize_mesh(state.mesh)
    torch.save(mesh_dict, ckpt_dir / "mesh.pt")

    # ── meta ──
    meta: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "t": state.t,
        "step_index": state.step_index,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "field_meta": field_meta,
        "mesh_type": type(state.mesh).__name__,
        "metadata": state.metadata,
    }
    if extra_metadata:
        meta["extra"] = extra_metadata

    with (ckpt_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2, default=str)

    logger.info("Saved checkpoint: %s (step=%d, t=%.6g)", ckpt_dir, state.step_index, state.t)
    return ckpt_dir


# ═══════════════════════════════════════════════════════════════════════════════
# Load
# ═══════════════════════════════════════════════════════════════════════════════


def load_checkpoint(path: Union[str, Path]) -> SimulationState:
    """
    Restore a ``SimulationState`` from a checkpoint directory.

    Parameters
    ----------
    path : str or Path
        Directory that contains ``meta.json``, ``mesh.pt``, ``fields/*.pt``.

    Returns
    -------
    SimulationState
    """
    ckpt_dir = Path(path)
    if not (ckpt_dir / "meta.json").exists():
        raise FileNotFoundError(f"No meta.json in {ckpt_dir}")

    with (ckpt_dir / "meta.json").open() as f:
        meta = json.load(f)

    schema_v = meta.get("schema_version", 0)
    if schema_v > SCHEMA_VERSION:
        raise ValueError(
            f"Checkpoint schema version {schema_v} > supported {SCHEMA_VERSION}"
        )

    # ── mesh ──
    mesh_dict = torch.load(ckpt_dir / "mesh.pt", weights_only=False)
    mesh = _deserialize_mesh(mesh_dict)

    # ── fields ──
    fields: Dict[str, FieldData] = {}
    field_meta = meta.get("field_meta", {})
    fields_dir = ckpt_dir / "fields"
    for fname, fmeta in field_meta.items():
        data = torch.load(fields_dir / f"{fname}.pt", weights_only=False)
        fields[fname] = FieldData(
            name=fname,
            data=data,
            mesh=mesh,
            components=fmeta.get("components", 1),
            units=fmeta.get("units", "1"),
        )

    state = SimulationState(
        t=meta["t"],
        fields=fields,
        mesh=mesh,
        metadata=meta.get("metadata", {}),
        step_index=meta.get("step_index", 0),
    )

    # Verify field hashes
    for fname, fmeta in field_meta.items():
        expected = fmeta.get("hash")
        if expected:
            actual = hash_tensor(fields[fname].data).digest
            if actual != expected:
                logger.warning(
                    "Hash mismatch for field '%s': expected %s, got %s",
                    fname,
                    expected[:16],
                    actual[:16],
                )

    logger.info("Loaded checkpoint: %s (step=%d, t=%.6g)", ckpt_dir, state.step_index, state.t)
    return state


# ═══════════════════════════════════════════════════════════════════════════════
# Mesh Serialization Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _serialize_mesh(mesh: Mesh) -> Dict[str, Any]:
    if isinstance(mesh, StructuredMesh):
        return {
            "type": "StructuredMesh",
            "shape": list(mesh.shape),
            "domain": [list(d) for d in mesh.domain],
        }
    elif isinstance(mesh, UnstructuredMesh):
        return {
            "type": "UnstructuredMesh",
            "nodes": mesh.nodes,
            "elements": mesh.elements,
        }
    else:
        return {
            "type": "Mesh",
            "ndim": mesh.ndim,
            "n_cells": mesh.n_cells,
        }


def _deserialize_mesh(d: Dict[str, Any]) -> Mesh:
    mtype = d.get("type", "Mesh")
    if mtype == "StructuredMesh":
        return StructuredMesh(
            shape=tuple(d["shape"]),
            domain=tuple(tuple(dd) for dd in d["domain"]),
        )
    elif mtype == "UnstructuredMesh":
        return UnstructuredMesh(nodes=d["nodes"], elements=d["elements"])
    else:
        return Mesh(ndim=d.get("ndim", 1), n_cells=d.get("n_cells", 0))
