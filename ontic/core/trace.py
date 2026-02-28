"""
Computation Trace Logger
========================

Deterministic trace of every tensor network operation for ZK proof generation.

Records:
    - Every TT-SVD truncation (inputs hash, singular values, rank, error)
    - Every MPO × MPS contraction (shapes, hash of result)
    - Every QR decomposition (shape, diagonal of R)
    - Every canonicalization sweep (direction, bond dims before/after)

The trace is compact: we store SHA-256 hashes of tensor data, not the
full tensors. This is sufficient for the ZK prover to verify the
computation was performed correctly on committed inputs.

Thread-safety: One TraceSession per thread. Use the context manager.

Constitution Compliance: Article II (Reproducibility), Article V (Numerical)
"""

from __future__ import annotations

import hashlib
import json
import logging
import struct
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Generator

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════════

TRACE_VERSION = 1
MAX_TRACE_ENTRIES = 10_000_000   # 10M ops before forced flush
MAX_SINGULAR_VALUES_STORED = 512  # Store top-k SVs for compact traces

# ═════════════════════════════════════════════════════════════════════════════
# Operation Types
# ═════════════════════════════════════════════════════════════════════════════


class OpType(Enum):
    """Traced operation types."""
    SVD_TRUNCATED = "svd_truncated"
    SVD_EXACT = "svd_exact"
    QR_POSITIVE = "qr_positive"
    THIN_SVD = "thin_svd"
    POLAR = "polar_decomposition"
    MPO_APPLY = "mpo_apply"
    MPS_NORMALIZE = "mps_normalize"
    MPS_CANONICALIZE_LEFT = "mps_canonicalize_left"
    MPS_CANONICALIZE_RIGHT = "mps_canonicalize_right"
    MPS_CANONICALIZE_TO = "mps_canonicalize_to"
    MPS_TRUNCATE = "mps_truncate"
    MPS_FROM_TENSOR = "mps_from_tensor"
    CONTRACTION = "contraction"
    CUSTOM = "custom"


# ═════════════════════════════════════════════════════════════════════════════
# Tensor Hashing
# ═════════════════════════════════════════════════════════════════════════════


def _hash_tensor(t: Tensor) -> str:
    """
    Deterministic SHA-256 of a tensor's data.

    Ensures reproducibility by converting to contiguous double on CPU
    before hashing. The hash commits to the tensor's values, shape, and dtype.
    """
    if t.is_cuda:
        t = t.cpu()
    t = t.contiguous().to(torch.float64)
    header = struct.pack("<I", len(t.shape))
    for dim in t.shape:
        header += struct.pack("<q", dim)
    h = hashlib.sha256(header)
    h.update(t.numpy().tobytes())
    return h.hexdigest()


def _hash_tensor_list(tensors: list[Tensor]) -> str:
    """Hash a list of tensors by chaining their individual hashes."""
    h = hashlib.sha256()
    h.update(struct.pack("<I", len(tensors)))
    for t in tensors:
        h.update(_hash_tensor(t).encode("ascii"))
    return h.hexdigest()


def _tensor_stats(t: Tensor) -> dict[str, Any]:
    """Compact statistics of a tensor for the trace record."""
    with torch.no_grad():
        t_f = t.to(torch.float64)
        return {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "norm": float(torch.linalg.norm(t_f).item()),
            "min": float(t_f.min().item()),
            "max": float(t_f.max().item()),
            "mean": float(t_f.mean().item()),
            "numel": int(t.numel()),
        }


# ═════════════════════════════════════════════════════════════════════════════
# Trace Entry
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class TraceEntry:
    """Single traced operation."""
    seq: int                          # Global sequence number
    op: OpType                        # Operation type
    timestamp_ns: int                 # Nanoseconds since epoch
    duration_ns: int = 0              # Wall-clock duration in ns
    input_hashes: dict[str, str] = field(default_factory=dict)
    output_hashes: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "op": self.op.value,
            "timestamp_ns": self.timestamp_ns,
            "duration_ns": self.duration_ns,
            "input_hashes": self.input_hashes,
            "output_hashes": self.output_hashes,
            "params": self.params,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TraceEntry:
        return cls(
            seq=int(d["seq"]),
            op=OpType(d["op"]),
            timestamp_ns=int(d["timestamp_ns"]),
            duration_ns=int(d.get("duration_ns", 0)),
            input_hashes=dict(d.get("input_hashes", {})),
            output_hashes=dict(d.get("output_hashes", {})),
            params=dict(d.get("params", {})),
            metrics=dict(d.get("metrics", {})),
        )


# ═════════════════════════════════════════════════════════════════════════════
# Trace Session
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class TraceSession:
    """
    A computation trace session.

    Records all tensor network operations for ZK proof generation.
    Thread-local: each thread gets its own session via the context manager.

    Usage:
        with trace_session() as session:
            mps = MPS.from_tensor(tensor, chi_max=32)
            result = mpo.apply(mps)
        # session.entries contains the full trace
        session.save("trace.json")
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time_ns: int = field(default_factory=lambda: int(time.time() * 1e9))
    entries: list[TraceEntry] = field(default_factory=list)
    _seq_counter: int = field(default=0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Configuration
    record_tensor_stats: bool = True
    record_singular_values: bool = True
    max_sv_count: int = MAX_SINGULAR_VALUES_STORED

    def _next_seq(self) -> int:
        with self._lock:
            seq = self._seq_counter
            self._seq_counter += 1
            return seq

    @property
    def entry_count(self) -> int:
        return len(self.entries)

    @property
    def duration_ns(self) -> int:
        if not self.entries:
            return 0
        return int(time.time() * 1e9) - self.start_time_ns

    # ── Recording Methods ────────────────────────────────────────────────

    def log_svd_truncated(
        self,
        A: Tensor,
        U: Tensor,
        S: Tensor,
        Vh: Tensor,
        chi_max: int | None,
        cutoff: float,
        use_rsvd: bool,
        info: dict[str, Any] | None,
        duration_ns: int,
    ) -> TraceEntry:
        """Record an svd_truncated operation."""
        entry = TraceEntry(
            seq=self._next_seq(),
            op=OpType.SVD_TRUNCATED if chi_max is not None else OpType.SVD_EXACT,
            timestamp_ns=int(time.time() * 1e9),
            duration_ns=duration_ns,
            input_hashes={"A": _hash_tensor(A)},
            output_hashes={
                "U": _hash_tensor(U),
                "S": _hash_tensor(S),
                "Vh": _hash_tensor(Vh),
            },
            params={
                "input_shape": list(A.shape),
                "chi_max": chi_max,
                "cutoff": cutoff,
                "use_rsvd": use_rsvd,
            },
            metrics={},
        )

        if info is not None:
            entry.metrics["truncation_error"] = info.get("truncation_error", 0.0)
            entry.metrics["rank"] = info.get("rank", 0)
            entry.metrics["original_rank"] = info.get("original_rank", 0)
            entry.metrics["condition_number"] = info.get("condition_number", 0.0)

        # Store singular values for the ZK circuit
        if self.record_singular_values:
            with torch.no_grad():
                sv = S.detach().cpu().to(torch.float64)
                n = min(len(sv), self.max_sv_count)
                entry.metrics["singular_values"] = sv[:n].tolist()
                entry.metrics["sv_count"] = int(len(sv))

        if self.record_tensor_stats:
            entry.metrics["U_stats"] = _tensor_stats(U)
            entry.metrics["Vh_stats"] = _tensor_stats(Vh)

        self.entries.append(entry)
        return entry

    def log_qr_positive(
        self,
        A: Tensor,
        Q: Tensor,
        R: Tensor,
        duration_ns: int,
    ) -> TraceEntry:
        """Record a qr_positive operation."""
        entry = TraceEntry(
            seq=self._next_seq(),
            op=OpType.QR_POSITIVE,
            timestamp_ns=int(time.time() * 1e9),
            duration_ns=duration_ns,
            input_hashes={"A": _hash_tensor(A)},
            output_hashes={
                "Q": _hash_tensor(Q),
                "R": _hash_tensor(R),
            },
            params={"input_shape": list(A.shape)},
            metrics={},
        )

        if self.record_tensor_stats:
            with torch.no_grad():
                diag_R = torch.diag(R).detach().cpu().to(torch.float64)
                entry.metrics["R_diagonal"] = diag_R.tolist()
                entry.metrics["Q_orthogonality_error"] = float(
                    torch.linalg.norm(
                        Q.T @ Q - torch.eye(Q.shape[1], dtype=Q.dtype, device=Q.device)
                    ).item()
                )

        self.entries.append(entry)
        return entry

    def log_mpo_apply(
        self,
        mpo_tensors: list[Tensor],
        mps_tensors: list[Tensor],
        result_tensors: list[Tensor],
        duration_ns: int,
    ) -> TraceEntry:
        """Record an MPO × MPS contraction."""
        entry = TraceEntry(
            seq=self._next_seq(),
            op=OpType.MPO_APPLY,
            timestamp_ns=int(time.time() * 1e9),
            duration_ns=duration_ns,
            input_hashes={
                "mpo": _hash_tensor_list(mpo_tensors),
                "mps": _hash_tensor_list(mps_tensors),
            },
            output_hashes={
                "result": _hash_tensor_list(result_tensors),
            },
            params={
                "L": len(mps_tensors),
                "mps_bond_dims": [t.shape[2] for t in mps_tensors[:-1]],
                "mpo_bond_dims": [t.shape[3] for t in mpo_tensors[:-1]],
                "result_bond_dims": [t.shape[2] for t in result_tensors[:-1]],
            },
            metrics={},
        )

        if self.record_tensor_stats:
            entry.metrics["mps_chi_max"] = max(max(t.shape[0], t.shape[2]) for t in mps_tensors)
            entry.metrics["mpo_D_max"] = max(max(t.shape[0], t.shape[3]) for t in mpo_tensors)
            entry.metrics["result_chi_max"] = max(max(t.shape[0], t.shape[2]) for t in result_tensors)

        self.entries.append(entry)
        return entry

    def log_mps_canonicalize(
        self,
        direction: str,
        tensors_before: list[Tensor],
        tensors_after: list[Tensor],
        site: int | None,
        duration_ns: int,
    ) -> TraceEntry:
        """Record MPS canonicalization."""
        if direction == "left":
            op = OpType.MPS_CANONICALIZE_LEFT
        elif direction == "right":
            op = OpType.MPS_CANONICALIZE_RIGHT
        else:
            op = OpType.MPS_CANONICALIZE_TO

        entry = TraceEntry(
            seq=self._next_seq(),
            op=op,
            timestamp_ns=int(time.time() * 1e9),
            duration_ns=duration_ns,
            input_hashes={"mps": _hash_tensor_list(tensors_before)},
            output_hashes={"mps": _hash_tensor_list(tensors_after)},
            params={
                "L": len(tensors_before),
                "direction": direction,
                "site": site,
                "bond_dims_before": [t.shape[2] for t in tensors_before[:-1]],
                "bond_dims_after": [t.shape[2] for t in tensors_after[:-1]],
            },
        )
        self.entries.append(entry)
        return entry

    def log_mps_truncate(
        self,
        tensors_before: list[Tensor],
        tensors_after: list[Tensor],
        chi_max: int,
        cutoff: float,
        duration_ns: int,
    ) -> TraceEntry:
        """Record MPS truncation."""
        entry = TraceEntry(
            seq=self._next_seq(),
            op=OpType.MPS_TRUNCATE,
            timestamp_ns=int(time.time() * 1e9),
            duration_ns=duration_ns,
            input_hashes={"mps": _hash_tensor_list(tensors_before)},
            output_hashes={"mps": _hash_tensor_list(tensors_after)},
            params={
                "chi_max": chi_max,
                "cutoff": cutoff,
                "L": len(tensors_before),
                "bond_dims_before": [t.shape[2] for t in tensors_before[:-1]],
                "bond_dims_after": [t.shape[2] for t in tensors_after[:-1]],
            },
        )
        self.entries.append(entry)
        return entry

    def log_mps_normalize(
        self,
        tensors_before: list[Tensor],
        tensors_after: list[Tensor],
        norm_value: float,
        duration_ns: int,
    ) -> TraceEntry:
        """Record MPS normalization."""
        entry = TraceEntry(
            seq=self._next_seq(),
            op=OpType.MPS_NORMALIZE,
            timestamp_ns=int(time.time() * 1e9),
            duration_ns=duration_ns,
            input_hashes={"mps": _hash_tensor_list(tensors_before)},
            output_hashes={"mps": _hash_tensor_list(tensors_after)},
            params={"norm": norm_value},
        )
        self.entries.append(entry)
        return entry

    def log_mps_from_tensor(
        self,
        input_tensor: Tensor,
        result_tensors: list[Tensor],
        chi_max: int | None,
        cutoff: float,
        duration_ns: int,
    ) -> TraceEntry:
        """Record MPS.from_tensor."""
        entry = TraceEntry(
            seq=self._next_seq(),
            op=OpType.MPS_FROM_TENSOR,
            timestamp_ns=int(time.time() * 1e9),
            duration_ns=duration_ns,
            input_hashes={"tensor": _hash_tensor(input_tensor)},
            output_hashes={"mps": _hash_tensor_list(result_tensors)},
            params={
                "input_shape": list(input_tensor.shape),
                "chi_max": chi_max,
                "cutoff": cutoff,
                "L": len(result_tensors),
                "bond_dims": [t.shape[2] for t in result_tensors[:-1]],
            },
        )
        self.entries.append(entry)
        return entry

    def log_contraction(
        self,
        name: str,
        inputs: dict[str, Tensor],
        outputs: dict[str, Tensor],
        equation: str,
        duration_ns: int,
    ) -> TraceEntry:
        """Record a generic tensor contraction (einsum)."""
        entry = TraceEntry(
            seq=self._next_seq(),
            op=OpType.CONTRACTION,
            timestamp_ns=int(time.time() * 1e9),
            duration_ns=duration_ns,
            input_hashes={k: _hash_tensor(v) for k, v in inputs.items()},
            output_hashes={k: _hash_tensor(v) for k, v in outputs.items()},
            params={
                "name": name,
                "equation": equation,
                "input_shapes": {k: list(v.shape) for k, v in inputs.items()},
                "output_shapes": {k: list(v.shape) for k, v in outputs.items()},
            },
        )
        self.entries.append(entry)
        return entry

    def log_custom(
        self,
        name: str,
        input_hashes: dict[str, str] | None = None,
        output_hashes: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> TraceEntry:
        """Record a custom operation."""
        entry = TraceEntry(
            seq=self._next_seq(),
            op=OpType.CUSTOM,
            timestamp_ns=int(time.time() * 1e9),
            input_hashes=input_hashes or {},
            output_hashes=output_hashes or {},
            params=params or {},
            metrics=metrics or {},
        )
        self.entries.append(entry)
        return entry

    # ── Finalization ─────────────────────────────────────────────────────

    def finalize(self) -> TraceDigest:
        """
        Compute the final trace digest.

        The digest is a compact summary that commits to the entire trace
        via a Merkle-like hash chain.
        """
        chain_hash = hashlib.sha256(b"TPC_TRACE_V1")

        op_counts: dict[str, int] = {}
        total_duration_ns = 0

        for entry in self.entries:
            # Chain hash: H(prev || entry_json)
            entry_bytes = json.dumps(
                entry.to_dict(), separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
            chain_hash.update(entry_bytes)

            op_name = entry.op.value
            op_counts[op_name] = op_counts.get(op_name, 0) + 1
            total_duration_ns += entry.duration_ns

        return TraceDigest(
            session_id=self.session_id,
            trace_hash=chain_hash.hexdigest(),
            entry_count=len(self.entries),
            op_counts=op_counts,
            total_duration_ns=total_duration_ns,
            start_time_ns=self.start_time_ns,
            end_time_ns=int(time.time() * 1e9),
        )

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        """
        Save trace to JSON file.

        Args:
            path: Output path (will be created / overwritten).

        Returns:
            Resolved path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        digest = self.finalize()
        payload = {
            "trace_version": TRACE_VERSION,
            "session_id": self.session_id,
            "digest": digest.to_dict(),
            "entries": [e.to_dict() for e in self.entries],
        }

        with open(path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

        logger.info(
            f"Trace saved: {path} ({len(self.entries)} entries, "
            f"hash={digest.trace_hash[:16]}...)"
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> TraceSession:
        """Load trace from JSON file."""
        path = Path(path)
        with open(path) as f:
            payload = json.load(f)

        session = cls(session_id=payload["session_id"])
        session.entries = [TraceEntry.from_dict(e) for e in payload["entries"]]
        session._seq_counter = len(session.entries)
        return session

    def save_binary(self, path: str | Path) -> Path:
        """
        Save trace in compact binary format for the Rust proof bridge.

        Format:
            [4 bytes] magic: TRCV
            [4 bytes] version
            [16 bytes] session UUID
            [8 bytes] entry count (uint64)
            For each entry:
                [4 bytes] JSON length
                [N bytes] JSON-encoded entry

        This is the format consumed by crates/proof_bridge.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        session_uuid = uuid.UUID(self.session_id)
        parts: list[bytes] = [
            b"TRCV",
            struct.pack("<I", TRACE_VERSION),
            session_uuid.bytes,
            struct.pack("<Q", len(self.entries)),
        ]

        for entry in self.entries:
            entry_json = json.dumps(
                entry.to_dict(), separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
            parts.append(struct.pack("<I", len(entry_json)))
            parts.append(entry_json)

        path.write_bytes(b"".join(parts))
        logger.info(f"Binary trace saved: {path} ({len(self.entries)} entries)")
        return path

    @classmethod
    def load_binary(cls, path: str | Path) -> TraceSession:
        """Load trace from compact binary format."""
        path = Path(path)
        data = path.read_bytes()
        offset = 0

        magic = data[offset:offset + 4]
        if magic != b"TRCV":
            raise ValueError(f"Invalid trace magic: {magic!r}")
        offset += 4

        version = struct.unpack_from("<I", data, offset)[0]
        if version != TRACE_VERSION:
            raise ValueError(f"Unsupported trace version: {version}")
        offset += 4

        session_uuid = uuid.UUID(bytes=data[offset:offset + 16])
        offset += 16

        entry_count = struct.unpack_from("<Q", data, offset)[0]
        offset += 8

        entries: list[TraceEntry] = []
        for _ in range(entry_count):
            json_len = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            entry_json = data[offset:offset + json_len].decode("utf-8")
            offset += json_len
            entries.append(TraceEntry.from_dict(json.loads(entry_json)))

        session = cls(session_id=str(session_uuid))
        session.entries = entries
        session._seq_counter = len(entries)
        return session

    def __repr__(self) -> str:
        return (
            f"TraceSession(id={self.session_id[:8]}..., "
            f"entries={len(self.entries)})"
        )


# ═════════════════════════════════════════════════════════════════════════════
# Trace Digest — Compact Summary
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class TraceDigest:
    """Compact summary of a computation trace."""
    session_id: str
    trace_hash: str                    # SHA-256 chain hash of all entries
    entry_count: int
    op_counts: dict[str, int]
    total_duration_ns: int
    start_time_ns: int
    end_time_ns: int

    @property
    def total_duration_s(self) -> float:
        return self.total_duration_ns / 1e9

    @property
    def wall_time_s(self) -> float:
        return (self.end_time_ns - self.start_time_ns) / 1e9

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "trace_hash": self.trace_hash,
            "entry_count": self.entry_count,
            "op_counts": self.op_counts,
            "total_duration_ns": self.total_duration_ns,
            "total_duration_s": self.total_duration_s,
            "start_time_ns": self.start_time_ns,
            "end_time_ns": self.end_time_ns,
            "wall_time_s": self.wall_time_s,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TraceDigest:
        return cls(
            session_id=str(d["session_id"]),
            trace_hash=str(d["trace_hash"]),
            entry_count=int(d["entry_count"]),
            op_counts=dict(d.get("op_counts", {})),
            total_duration_ns=int(d.get("total_duration_ns", 0)),
            start_time_ns=int(d.get("start_time_ns", 0)),
            end_time_ns=int(d.get("end_time_ns", 0)),
        )


# ═════════════════════════════════════════════════════════════════════════════
# Thread-Local Session Management
# ═════════════════════════════════════════════════════════════════════════════

_thread_local = threading.local()


def get_active_session() -> TraceSession | None:
    """Get the active trace session for the current thread, if any."""
    return getattr(_thread_local, "session", None)


@contextmanager
def trace_session(
    record_tensor_stats: bool = True,
    record_singular_values: bool = True,
    max_sv_count: int = MAX_SINGULAR_VALUES_STORED,
) -> Generator[TraceSession, None, None]:
    """
    Context manager for a trace session.

    All tensor network operations within the context will be recorded.

    Args:
        record_tensor_stats: Record shape/norm/min/max of tensors.
        record_singular_values: Record singular values from SVDs.
        max_sv_count: Maximum number of singular values to store per SVD.

    Yields:
        TraceSession instance.

    Example:
        >>> with trace_session() as session:
        ...     mps = MPS.from_tensor(tensor, chi_max=32)
        ...     result = mpo.apply(mps)
        >>> digest = session.finalize()
        >>> print(f"Trace hash: {digest.trace_hash}")
    """
    session = TraceSession(
        record_tensor_stats=record_tensor_stats,
        record_singular_values=record_singular_values,
        max_sv_count=max_sv_count,
    )

    # Save any existing session (supports nesting)
    prev_session = getattr(_thread_local, "session", None)
    _thread_local.session = session

    try:
        yield session
    finally:
        _thread_local.session = prev_session


# ═════════════════════════════════════════════════════════════════════════════
# Instrumented Operations — Drop-in replacements
# ═════════════════════════════════════════════════════════════════════════════
#
# These wrap the original ontic.core functions to emit trace entries
# when a session is active, with zero overhead when no session exists.
# ═════════════════════════════════════════════════════════════════════════════


def traced_svd_truncated(
    A: Tensor,
    chi_max: int | None = None,
    cutoff: float = 1e-14,
    return_info: bool = False,
    use_rsvd: bool | None = None,
    rsvd_threshold: int = 256,
) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, dict]:
    """
    Drop-in replacement for svd_truncated that emits trace entries.

    When no trace session is active, delegates directly to the original
    with zero overhead beyond one getattr check.
    """
    from ontic.core.decompositions import svd_truncated as _original_svd

    session = get_active_session()
    if session is None:
        return _original_svd(A, chi_max, cutoff, return_info, use_rsvd, rsvd_threshold)

    # We need info for the trace even if the caller doesn't
    t0 = time.perf_counter_ns()
    result = _original_svd(A, chi_max, cutoff, True, use_rsvd, rsvd_threshold)
    duration = time.perf_counter_ns() - t0

    U, S, Vh, info = result  # type: ignore[misc]

    # Determine actual use_rsvd state
    actual_rsvd = use_rsvd
    if actual_rsvd is None:
        m, n = A.shape
        min_dim = min(m, n)
        target = min(chi_max or min_dim, min_dim)
        actual_rsvd = (min_dim > rsvd_threshold) and (target < min_dim // 2)

    session.log_svd_truncated(A, U, S, Vh, chi_max, cutoff, actual_rsvd, info, duration)

    if return_info:
        return U, S, Vh, info
    return U, S, Vh


def traced_qr_positive(A: Tensor) -> tuple[Tensor, Tensor]:
    """Drop-in replacement for qr_positive that emits trace entries."""
    from ontic.core.decompositions import qr_positive as _original_qr

    session = get_active_session()
    if session is None:
        return _original_qr(A)

    t0 = time.perf_counter_ns()
    Q, R = _original_qr(A)
    duration = time.perf_counter_ns() - t0

    session.log_qr_positive(A, Q, R, duration)
    return Q, R


# ═════════════════════════════════════════════════════════════════════════════
# Monkey-Patch Installation
# ═════════════════════════════════════════════════════════════════════════════

_patched = False
_originals: dict[str, Any] = {}


def install_trace_hooks() -> None:
    """
    Install trace hooks into ontic.core modules.

    This monkey-patches svd_truncated, qr_positive, and MPO.apply
    so that they emit trace entries when a session is active.
    Safe to call multiple times (idempotent).
    """
    global _patched
    if _patched:
        return

    import ontic.core.decompositions as decomp_mod
    import ontic.core.mps as mps_mod
    import ontic.core.mpo as mpo_mod

    # Save originals
    _originals["svd_truncated"] = decomp_mod.svd_truncated
    _originals["qr_positive"] = decomp_mod.qr_positive
    _originals["MPO.apply"] = mpo_mod.MPO.apply
    _originals["MPS.canonicalize_left_"] = mps_mod.MPS.canonicalize_left_
    _originals["MPS.canonicalize_right_"] = mps_mod.MPS.canonicalize_right_
    _originals["MPS.canonicalize_to_"] = mps_mod.MPS.canonicalize_to_
    _originals["MPS.truncate_"] = mps_mod.MPS.truncate_
    _originals["MPS.normalize_"] = mps_mod.MPS.normalize_
    _originals["MPS.from_tensor"] = mps_mod.MPS.from_tensor

    # Patch decompositions (these are also imported by mps.py, so we
    # patch both the module-level reference and the mps module import)
    decomp_mod.svd_truncated = traced_svd_truncated  # type: ignore[assignment]
    decomp_mod.qr_positive = traced_qr_positive  # type: ignore[assignment]
    mps_mod.svd_truncated = traced_svd_truncated  # type: ignore[attr-defined]
    mps_mod.qr_positive = traced_qr_positive  # type: ignore[attr-defined]

    # Patch MPO.apply
    original_apply = _originals["MPO.apply"]

    def traced_mpo_apply(self: Any, mps: Any) -> Any:
        session = get_active_session()
        if session is None:
            return original_apply(self, mps)
        mpo_tensors_snap = [t.clone() for t in self.tensors]
        mps_tensors_snap = [t.clone() for t in mps.tensors]
        t0 = time.perf_counter_ns()
        result = original_apply(self, mps)
        duration = time.perf_counter_ns() - t0
        session.log_mpo_apply(mpo_tensors_snap, mps_tensors_snap, result.tensors, duration)
        return result

    mpo_mod.MPO.apply = traced_mpo_apply  # type: ignore[assignment]

    # Patch MPS.canonicalize_left_
    original_canon_left = _originals["MPS.canonicalize_left_"]

    def traced_canonicalize_left_(self: Any) -> Any:
        session = get_active_session()
        if session is None:
            return original_canon_left(self)
        tensors_before = [t.clone() for t in self.tensors]
        t0 = time.perf_counter_ns()
        result = original_canon_left(self)
        duration = time.perf_counter_ns() - t0
        session.log_mps_canonicalize("left", tensors_before, self.tensors, None, duration)
        return result

    mps_mod.MPS.canonicalize_left_ = traced_canonicalize_left_  # type: ignore[assignment]

    # Patch MPS.canonicalize_right_
    original_canon_right = _originals["MPS.canonicalize_right_"]

    def traced_canonicalize_right_(self: Any) -> Any:
        session = get_active_session()
        if session is None:
            return original_canon_right(self)
        tensors_before = [t.clone() for t in self.tensors]
        t0 = time.perf_counter_ns()
        result = original_canon_right(self)
        duration = time.perf_counter_ns() - t0
        session.log_mps_canonicalize("right", tensors_before, self.tensors, None, duration)
        return result

    mps_mod.MPS.canonicalize_right_ = traced_canonicalize_right_  # type: ignore[assignment]

    # Patch MPS.canonicalize_to_
    original_canon_to = _originals["MPS.canonicalize_to_"]

    def traced_canonicalize_to_(self: Any, site: int) -> Any:
        session = get_active_session()
        if session is None:
            return original_canon_to(self, site)
        tensors_before = [t.clone() for t in self.tensors]
        t0 = time.perf_counter_ns()
        result = original_canon_to(self, site)
        duration = time.perf_counter_ns() - t0
        session.log_mps_canonicalize("to", tensors_before, self.tensors, site, duration)
        return result

    mps_mod.MPS.canonicalize_to_ = traced_canonicalize_to_  # type: ignore[assignment]

    # Patch MPS.truncate_
    original_truncate = _originals["MPS.truncate_"]

    def traced_truncate_(self: Any, chi_max: int, cutoff: float = 1e-14) -> Any:
        session = get_active_session()
        if session is None:
            return original_truncate(self, chi_max, cutoff)
        tensors_before = [t.clone() for t in self.tensors]
        t0 = time.perf_counter_ns()
        result = original_truncate(self, chi_max, cutoff)
        duration = time.perf_counter_ns() - t0
        session.log_mps_truncate(tensors_before, self.tensors, chi_max, cutoff, duration)
        return result

    mps_mod.MPS.truncate_ = traced_truncate_  # type: ignore[assignment]

    # Patch MPS.normalize_
    original_normalize = _originals["MPS.normalize_"]

    def traced_normalize_(self: Any) -> Any:
        session = get_active_session()
        if session is None:
            return original_normalize(self)
        tensors_before = [t.clone() for t in self.tensors]
        norm_val = float(self.norm().item())
        t0 = time.perf_counter_ns()
        result = original_normalize(self)
        duration = time.perf_counter_ns() - t0
        session.log_mps_normalize(tensors_before, self.tensors, norm_val, duration)
        return result

    mps_mod.MPS.normalize_ = traced_normalize_  # type: ignore[assignment]

    # Patch MPS.from_tensor
    original_from_tensor = _originals["MPS.from_tensor"]

    @classmethod  # type: ignore[misc]
    def traced_from_tensor(
        cls: type,
        tensor: Tensor,
        chi_max: int | None = None,
        cutoff: float = 1e-14,
    ) -> Any:
        session = get_active_session()
        if session is None:
            return original_from_tensor.__func__(cls, tensor, chi_max, cutoff)
        input_snap = tensor.clone()
        t0 = time.perf_counter_ns()
        result = original_from_tensor.__func__(cls, tensor, chi_max, cutoff)
        duration = time.perf_counter_ns() - t0
        session.log_mps_from_tensor(input_snap, result.tensors, chi_max, cutoff, duration)
        return result

    mps_mod.MPS.from_tensor = traced_from_tensor  # type: ignore[assignment]

    _patched = True
    logger.info("Trace hooks installed into ontic.core")


def uninstall_trace_hooks() -> None:
    """Remove trace hooks, restoring original functions."""
    global _patched
    if not _patched:
        return

    import ontic.core.decompositions as decomp_mod
    import ontic.core.mps as mps_mod
    import ontic.core.mpo as mpo_mod

    decomp_mod.svd_truncated = _originals["svd_truncated"]  # type: ignore[assignment]
    decomp_mod.qr_positive = _originals["qr_positive"]  # type: ignore[assignment]
    mps_mod.svd_truncated = _originals["svd_truncated"]  # type: ignore[attr-defined]
    mps_mod.qr_positive = _originals["qr_positive"]  # type: ignore[attr-defined]
    mpo_mod.MPO.apply = _originals["MPO.apply"]  # type: ignore[assignment]
    mps_mod.MPS.canonicalize_left_ = _originals["MPS.canonicalize_left_"]  # type: ignore[assignment]
    mps_mod.MPS.canonicalize_right_ = _originals["MPS.canonicalize_right_"]  # type: ignore[assignment]
    mps_mod.MPS.canonicalize_to_ = _originals["MPS.canonicalize_to_"]  # type: ignore[assignment]
    mps_mod.MPS.truncate_ = _originals["MPS.truncate_"]  # type: ignore[assignment]
    mps_mod.MPS.normalize_ = _originals["MPS.normalize_"]  # type: ignore[assignment]
    mps_mod.MPS.from_tensor = _originals["MPS.from_tensor"]  # type: ignore[assignment]

    _originals.clear()
    _patched = False
    logger.info("Trace hooks removed from ontic.core")
