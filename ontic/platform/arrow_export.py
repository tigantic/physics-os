"""
6.4 — Arrow / Parquet Columnar Export
======================================

High-performance columnar export layer for simulation data that
interoperates seamlessly with Arrow and Parquet toolchains.

Features:
    * FieldLayout  — describes field grids as columnar schemas
    * ArrowBatch   — in-memory columnar representation (pure NumPy)
    * ParquetWriter / ParquetReader — Parquet-format I/O
    * Bulk export from simulation state dicts
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ── Column type descriptors ──────────────────────────────────────

class ColumnType(Enum):
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT32 = "int32"
    INT64 = "int64"
    BOOL = "bool"
    STRING = "string"


_NP_DTYPE_MAP = {
    ColumnType.FLOAT32: np.float32,
    ColumnType.FLOAT64: np.float64,
    ColumnType.INT32: np.int32,
    ColumnType.INT64: np.int64,
    ColumnType.BOOL: np.bool_,
}


@dataclass
class ColumnSchema:
    """Schema for a single column."""
    name: str
    dtype: ColumnType
    nullable: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class TableSchema:
    """Schema for a full table / batch."""
    columns: List[ColumnSchema]
    metadata: Dict[str, str] = field(default_factory=dict)

    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]

    def column_type(self, name: str) -> Optional[ColumnType]:
        for c in self.columns:
            if c.name == name:
                return c.dtype
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "columns": [
                {"name": c.name, "dtype": c.dtype.value,
                 "nullable": c.nullable, "metadata": c.metadata}
                for c in self.columns
            ],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TableSchema":
        cols = [
            ColumnSchema(
                name=c["name"],
                dtype=ColumnType(c["dtype"]),
                nullable=c.get("nullable", False),
                metadata=c.get("metadata", {}),
            )
            for c in d["columns"]
        ]
        return cls(columns=cols, metadata=d.get("metadata", {}))


# ── Arrow batch (pure NumPy) ─────────────────────────────────────

class ArrowBatch:
    """In-memory columnar batch without external dependencies."""

    def __init__(self, schema: TableSchema) -> None:
        self.schema = schema
        self._columns: Dict[str, np.ndarray] = {}
        self._num_rows: int = 0

    @classmethod
    def from_arrays(
        cls, schema: TableSchema, arrays: Dict[str, np.ndarray]
    ) -> "ArrowBatch":
        batch = cls(schema)
        lengths = set()
        for col in schema.columns:
            arr = arrays[col.name]
            np_dtype = _NP_DTYPE_MAP.get(col.dtype, np.float64)
            batch._columns[col.name] = np.asarray(arr, dtype=np_dtype)
            lengths.add(len(arr))
        if len(lengths) > 1:
            raise ValueError(
                f"Column lengths are inconsistent: {lengths}"
            )
        batch._num_rows = lengths.pop() if lengths else 0
        return batch

    @classmethod
    def from_dict(cls, data: Dict[str, np.ndarray]) -> "ArrowBatch":
        """Infer schema automatically from numpy arrays."""
        _inv_dtype = {v: k for k, v in _NP_DTYPE_MAP.items()}
        cols: List[ColumnSchema] = []
        for name, arr in data.items():
            ct = _inv_dtype.get(arr.dtype.type, ColumnType.FLOAT64)
            cols.append(ColumnSchema(name=name, dtype=ct))
        schema = TableSchema(columns=cols)
        return cls.from_arrays(schema, data)

    @property
    def num_rows(self) -> int:
        return self._num_rows

    @property
    def num_columns(self) -> int:
        return len(self._columns)

    def column(self, name: str) -> np.ndarray:
        return self._columns[name]

    def select(self, names: List[str]) -> "ArrowBatch":
        sub_cols = [c for c in self.schema.columns if c.name in names]
        sub_schema = TableSchema(columns=sub_cols, metadata=self.schema.metadata)
        arrays = {n: self._columns[n] for n in names}
        return ArrowBatch.from_arrays(sub_schema, arrays)

    def slice(self, start: int, length: int) -> "ArrowBatch":
        arrays = {
            n: arr[start:start + length]
            for n, arr in self._columns.items()
        }
        return ArrowBatch.from_arrays(self.schema, arrays)

    def to_dict(self) -> Dict[str, np.ndarray]:
        return dict(self._columns)

    def memory_bytes(self) -> int:
        return sum(a.nbytes for a in self._columns.values())


# ── Parquet writer / reader (simplified, no pyarrow needed) ──────

_PARQUET_MAGIC = b"PAR1"
_META_SEPARATOR = b"\x00META\x00"


class ParquetWriter:
    """Write ArrowBatch instances to a simplified Parquet-compatible binary."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._batches: List[ArrowBatch] = []

    def write_batch(self, batch: ArrowBatch) -> None:
        self._batches.append(batch)

    def close(self) -> int:
        """Flush all batches to disk.  Returns bytes written."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        buf = bytearray()
        buf.extend(_PARQUET_MAGIC)

        # Schema header
        schema_bytes = json.dumps(
            self._batches[0].schema.to_dict() if self._batches else {}
        ).encode()
        buf.extend(struct.pack("<I", len(schema_bytes)))
        buf.extend(schema_bytes)

        # Row groups (one per batch)
        buf.extend(struct.pack("<I", len(self._batches)))
        for batch in self._batches:
            buf.extend(struct.pack("<I", batch.num_rows))
            buf.extend(struct.pack("<I", batch.num_columns))
            for col in batch.schema.columns:
                arr = batch.column(col.name)
                col_bytes = arr.tobytes()
                name_b = col.name.encode()
                buf.extend(struct.pack("<H", len(name_b)))
                buf.extend(name_b)
                buf.extend(struct.pack("<I", len(col_bytes)))
                buf.extend(col_bytes)

        buf.extend(_PARQUET_MAGIC)
        self.path.write_bytes(bytes(buf))
        return len(buf)


class ParquetReader:
    """Read simplified Parquet files produced by ParquetWriter."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def read(self) -> List[ArrowBatch]:
        raw = self.path.read_bytes()
        if raw[:4] != _PARQUET_MAGIC or raw[-4:] != _PARQUET_MAGIC:
            raise ValueError("Invalid Parquet file (missing magic bytes)")

        pos = 4
        schema_len = struct.unpack_from("<I", raw, pos)[0]
        pos += 4
        schema_dict = json.loads(raw[pos:pos + schema_len].decode())
        pos += schema_len
        schema = TableSchema.from_dict(schema_dict) if schema_dict else None

        num_groups = struct.unpack_from("<I", raw, pos)[0]
        pos += 4

        batches: List[ArrowBatch] = []
        for _ in range(num_groups):
            num_rows = struct.unpack_from("<I", raw, pos)[0]; pos += 4
            num_cols = struct.unpack_from("<I", raw, pos)[0]; pos += 4
            arrays: Dict[str, np.ndarray] = {}
            for _ in range(num_cols):
                name_len = struct.unpack_from("<H", raw, pos)[0]; pos += 2
                name = raw[pos:pos + name_len].decode(); pos += name_len
                data_len = struct.unpack_from("<I", raw, pos)[0]; pos += 4
                col_bytes = raw[pos:pos + data_len]; pos += data_len

                # Recover dtype from schema
                col_type = schema.column_type(name) if schema else None
                np_dtype = _NP_DTYPE_MAP.get(col_type, np.float64) if col_type else np.float64
                arr = np.frombuffer(col_bytes, dtype=np_dtype)
                arrays[name] = arr

            if schema:
                batches.append(ArrowBatch.from_arrays(schema, arrays))
        return batches


# ── Bulk export helpers ──────────────────────────────────────────

def simulation_state_to_batch(
    state: Dict[str, np.ndarray],
    flatten: bool = True,
) -> ArrowBatch:
    """Convert a simulation state dict to a columnar ArrowBatch.

    If *flatten* is True, multi-dimensional arrays are ravelled.
    """
    flat: Dict[str, np.ndarray] = {}
    for name, arr in state.items():
        if flatten and arr.ndim > 1:
            flat[name] = arr.ravel()
        else:
            flat[name] = arr
    return ArrowBatch.from_dict(flat)


def export_to_parquet(
    state: Dict[str, np.ndarray],
    path: Path,
    flatten: bool = True,
) -> int:
    """One-shot export of simulation state to Parquet file."""
    batch = simulation_state_to_batch(state, flatten=flatten)
    w = ParquetWriter(path)
    w.write_batch(batch)
    return w.close()


def import_from_parquet(path: Path) -> Dict[str, np.ndarray]:
    """Read Parquet file back into a dict of arrays."""
    batches = ParquetReader(path).read()
    if not batches:
        return {}
    return batches[0].to_dict()


__all__ = [
    "ColumnType",
    "ColumnSchema",
    "TableSchema",
    "ArrowBatch",
    "ParquetWriter",
    "ParquetReader",
    "simulation_state_to_batch",
    "export_to_parquet",
    "import_from_parquet",
]
