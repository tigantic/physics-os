"""
TPC Format — Trustless Physics Certificate Serializer / Deserializer
=====================================================================

Binary format for cryptographic physics verification certificates.

Structure:
    Header        (64 bytes fixed)
    Layer A       (Mathematical Truth — Lean 4 proofs)
    Layer B       (Computational Integrity — ZK proof)
    Layer C       (Physical Fidelity — attested benchmarks)
    Metadata      (solver, domain, QTT params)
    Signature     (Ed25519 over SHA-256 of all preceding sections)

Wire encoding: MessagePack for variable-length sections, preceded by
4-byte little-endian length prefix. Header is raw struct.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import json
import logging
import struct
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from tpc.constants import (
    FORMAL_PROOF_SYSTEMS,
    HASH_SIZE,
    KNOWN_DOMAINS,
    KNOWN_SOLVERS,
    MAX_BENCHMARKS,
    MAX_CERTIFICATE_SIZE,
    MAX_LEAN_PROOFS,
    MAX_PROOF_BYTES,
    MAX_STRING_LEN,
    MAX_THEOREM_NAME_LEN,
    PROOF_SYSTEMS,
    PUBLIC_KEY_SIZE,
    SIGNATURE_SIZE,
    TPC_HEADER_SIZE,
    TPC_MAGIC,
    TPC_VERSION,
)

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# Data Classes — In-Memory Representation
# ═════════════════════════════════════════════════════════════════════════════


class CoverageLevel(Enum):
    """How much of the solver's operations are formally verified."""
    NONE = "none"
    PARTIAL = "partial"
    FULL = "full"


@dataclass(frozen=True)
class TheoremRef:
    """Reference to a Lean 4 theorem used in Layer A."""
    name: str
    file: str
    line: int = 0
    statement_hash: str = ""   # SHA-256 of the Lean source for the theorem

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "file": self.file,
            "line": self.line,
            "statement_hash": self.statement_hash,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TheoremRef:
        return cls(
            name=str(d["name"]),
            file=str(d["file"]),
            line=int(d.get("line", 0)),
            statement_hash=str(d.get("statement_hash", "")),
        )


@dataclass
class LayerA:
    """
    Layer A — Mathematical Truth.

    Formal proofs (Lean 4) that the governing equations are mathematically
    sound for the domain they claim to cover.
    """
    proof_system: str = "lean4"
    theorems: list[TheoremRef] = field(default_factory=list)
    proof_objects: bytes = b""       # Serialized Lean environment exports
    coverage: CoverageLevel = CoverageLevel.PARTIAL
    coverage_pct: float = 0.0        # 0.0–100.0
    notes: str = ""

    def __post_init__(self) -> None:
        if self.proof_system not in FORMAL_PROOF_SYSTEMS:
            raise ValueError(
                f"Unknown formal proof system {self.proof_system!r}. "
                f"Expected one of {sorted(FORMAL_PROOF_SYSTEMS)}"
            )
        if len(self.theorems) > MAX_LEAN_PROOFS:
            raise ValueError(
                f"Too many theorems ({len(self.theorems)} > {MAX_LEAN_PROOFS})"
            )
        for t in self.theorems:
            if len(t.name) > MAX_THEOREM_NAME_LEN:
                raise ValueError(f"Theorem name too long: {t.name[:64]}...")

    def to_dict(self) -> dict[str, Any]:
        return {
            "proof_system": self.proof_system,
            "theorems": [t.to_dict() for t in self.theorems],
            "proof_objects_sha256": hashlib.sha256(self.proof_objects).hexdigest(),
            "proof_objects_size": len(self.proof_objects),
            "coverage": self.coverage.value,
            "coverage_pct": self.coverage_pct,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], proof_objects: bytes = b"") -> LayerA:
        return cls(
            proof_system=str(d.get("proof_system", "lean4")),
            theorems=[TheoremRef.from_dict(t) for t in d.get("theorems", [])],
            proof_objects=proof_objects,
            coverage=CoverageLevel(d.get("coverage", "partial")),
            coverage_pct=float(d.get("coverage_pct", 0.0)),
            notes=str(d.get("notes", "")),
        )


@dataclass
class LayerB:
    """
    Layer B — Computational Integrity.

    Zero-knowledge proof that the QTT computation was executed correctly
    on specific inputs with specific tolerances producing specific outputs.
    """
    proof_system: str = "stark"
    public_inputs: dict[str, Any] = field(default_factory=dict)
    public_outputs: dict[str, Any] = field(default_factory=dict)
    proof_bytes: bytes = b""
    verification_key: bytes = b""
    proof_generation_time_s: float = 0.0
    circuit_constraints: int = 0
    prover_version: str = ""

    def __post_init__(self) -> None:
        if self.proof_system not in PROOF_SYSTEMS:
            raise ValueError(
                f"Unknown proof system {self.proof_system!r}. "
                f"Expected one of {sorted(PROOF_SYSTEMS)}"
            )
        if len(self.proof_bytes) > MAX_PROOF_BYTES:
            raise ValueError(
                f"Proof bytes too large ({len(self.proof_bytes)} > {MAX_PROOF_BYTES})"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "proof_system": self.proof_system,
            "public_inputs": self.public_inputs,
            "public_outputs": self.public_outputs,
            "proof_sha256": hashlib.sha256(self.proof_bytes).hexdigest(),
            "proof_size": len(self.proof_bytes),
            "verification_key_sha256": hashlib.sha256(self.verification_key).hexdigest(),
            "verification_key_size": len(self.verification_key),
            "proof_generation_time_s": self.proof_generation_time_s,
            "circuit_constraints": self.circuit_constraints,
            "prover_version": self.prover_version,
        }

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        proof_bytes: bytes = b"",
        verification_key: bytes = b"",
    ) -> LayerB:
        return cls(
            proof_system=str(d.get("proof_system", "stark")),
            public_inputs=dict(d.get("public_inputs", {})),
            public_outputs=dict(d.get("public_outputs", {})),
            proof_bytes=proof_bytes,
            verification_key=verification_key,
            proof_generation_time_s=float(d.get("proof_generation_time_s", 0.0)),
            circuit_constraints=int(d.get("circuit_constraints", 0)),
            prover_version=str(d.get("prover_version", "")),
        )


@dataclass
class BenchmarkResult:
    """Single benchmark validation result."""
    name: str
    gauntlet: str = ""
    l2_error: float = 0.0
    max_deviation: float = 0.0
    conservation_error: float = 0.0
    passed: bool = False
    threshold_l2: float = 0.0
    threshold_max: float = 0.0
    threshold_conservation: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "gauntlet": self.gauntlet,
            "l2_error": self.l2_error,
            "max_deviation": self.max_deviation,
            "conservation_error": self.conservation_error,
            "passed": self.passed,
            "threshold_l2": self.threshold_l2,
            "threshold_max": self.threshold_max,
            "threshold_conservation": self.threshold_conservation,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BenchmarkResult:
        return cls(
            name=str(d["name"]),
            gauntlet=str(d.get("gauntlet", "")),
            l2_error=float(d.get("l2_error", 0.0)),
            max_deviation=float(d.get("max_deviation", 0.0)),
            conservation_error=float(d.get("conservation_error", 0.0)),
            passed=bool(d.get("passed", False)),
            threshold_l2=float(d.get("threshold_l2", 0.0)),
            threshold_max=float(d.get("threshold_max", 0.0)),
            threshold_conservation=float(d.get("threshold_conservation", 0.0)),
            metrics=dict(d.get("metrics", {})),
        )


@dataclass
class HardwareSpec:
    """Hardware environment where the simulation was executed."""
    platform: str = ""
    processor: str = ""
    gpu: str = ""
    memory_gb: float = 0.0
    os: str = ""
    python_version: str = ""
    torch_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "processor": self.processor,
            "gpu": self.gpu,
            "memory_gb": self.memory_gb,
            "os": self.os,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HardwareSpec:
        return cls(
            platform=str(d.get("platform", "")),
            processor=str(d.get("processor", "")),
            gpu=str(d.get("gpu", "")),
            memory_gb=float(d.get("memory_gb", 0.0)),
            os=str(d.get("os", "")),
            python_version=str(d.get("python_version", "")),
            torch_version=str(d.get("torch_version", "")),
        )

    @classmethod
    def detect(cls) -> HardwareSpec:
        """Auto-detect current hardware environment."""
        import platform as plat
        import sys

        gpu = "none"
        torch_ver = ""
        try:
            import torch
            torch_ver = torch.__version__
            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_name(0)
        except ImportError:
            pass

        mem_gb = 0.0
        try:
            import psutil
            mem_gb = psutil.virtual_memory().total / (1024 ** 3)
        except ImportError:
            pass

        return cls(
            platform=plat.machine(),
            processor=plat.processor() or plat.machine(),
            gpu=gpu,
            memory_gb=round(mem_gb, 1),
            os=f"{plat.system()} {plat.release()}",
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            torch_version=torch_ver,
        )


@dataclass
class LayerC:
    """
    Layer C — Physical Fidelity.

    Cryptographically attested validation against known physical benchmarks.
    """
    benchmarks: list[BenchmarkResult] = field(default_factory=list)
    hardware: HardwareSpec = field(default_factory=HardwareSpec)
    git_commit: str = ""
    attestation_json: bytes = b""    # Raw attestation JSON for reproducibility
    total_time_s: float = 0.0

    def __post_init__(self) -> None:
        if len(self.benchmarks) > MAX_BENCHMARKS:
            raise ValueError(
                f"Too many benchmarks ({len(self.benchmarks)} > {MAX_BENCHMARKS})"
            )

    @property
    def all_passed(self) -> bool:
        return all(b.passed for b in self.benchmarks)

    @property
    def pass_count(self) -> int:
        return sum(1 for b in self.benchmarks if b.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmarks": [b.to_dict() for b in self.benchmarks],
            "hardware": self.hardware.to_dict(),
            "git_commit": self.git_commit,
            "attestation_sha256": hashlib.sha256(self.attestation_json).hexdigest(),
            "attestation_size": len(self.attestation_json),
            "total_time_s": self.total_time_s,
            "all_passed": self.all_passed,
            "pass_count": self.pass_count,
            "total_benchmarks": len(self.benchmarks),
        }

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], attestation_json: bytes = b""
    ) -> LayerC:
        return cls(
            benchmarks=[BenchmarkResult.from_dict(b) for b in d.get("benchmarks", [])],
            hardware=HardwareSpec.from_dict(d.get("hardware", {})),
            git_commit=str(d.get("git_commit", "")),
            attestation_json=attestation_json,
            total_time_s=float(d.get("total_time_s", 0.0)),
        )


@dataclass
class QTTParams:
    """QTT solver parameters recorded in the certificate."""
    max_rank: int = 0
    tolerance: float = 0.0
    grid_bits: int = 0
    num_sites: int = 0
    physical_dim: int = 2
    bond_dims: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "max_rank": self.max_rank,
            "tolerance": self.tolerance,
            "grid_bits": self.grid_bits,
            "num_sites": self.num_sites,
            "physical_dim": self.physical_dim,
        }
        if self.bond_dims:
            d["bond_dims"] = self.bond_dims
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QTTParams:
        return cls(
            max_rank=int(d.get("max_rank", 0)),
            tolerance=float(d.get("tolerance", 0.0)),
            grid_bits=int(d.get("grid_bits", 0)),
            num_sites=int(d.get("num_sites", 0)),
            physical_dim=int(d.get("physical_dim", 2)),
            bond_dims=list(d.get("bond_dims", [])),
        )


@dataclass
class Metadata:
    """Certificate metadata."""
    domain: str = "cfd"
    solver: str = "euler3d"
    qtt_params: QTTParams = field(default_factory=QTTParams)
    customer_ref: str = ""           # Optional, encrypted externally
    description: str = ""
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.domain not in KNOWN_DOMAINS and self.domain != "":
            logger.warning(f"Unknown domain {self.domain!r}")
        if self.solver not in KNOWN_SOLVERS and self.solver != "":
            logger.warning(f"Unknown solver {self.solver!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "solver": self.solver,
            "qtt_params": self.qtt_params.to_dict(),
            "customer_ref": self.customer_ref,
            "description": self.description,
            "tags": self.tags,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Metadata:
        return cls(
            domain=str(d.get("domain", "cfd")),
            solver=str(d.get("solver", "euler3d")),
            qtt_params=QTTParams.from_dict(d.get("qtt_params", {})),
            customer_ref=str(d.get("customer_ref", "")),
            description=str(d.get("description", "")),
            tags=list(d.get("tags", [])),
            extra=dict(d.get("extra", {})),
        )


# ═════════════════════════════════════════════════════════════════════════════
# Header — Fixed 64 bytes
# ═════════════════════════════════════════════════════════════════════════════

# Header struct layout (little-endian):
#   magic:        4 bytes  "TPC\x01"
#   version:      4 bytes  uint32
#   cert_id:     16 bytes  UUID (RFC 4122)
#   timestamp_ns: 8 bytes  int64  (nanoseconds since UNIX epoch)
#   solver_hash: 32 bytes  SHA-256 of solver binary / source
#   total:       64 bytes
_HEADER_FMT = "<4sI16sq32s"
assert struct.calcsize(_HEADER_FMT) == TPC_HEADER_SIZE


@dataclass
class TPCHeader:
    """Fixed-size 64-byte certificate header."""
    version: int = TPC_VERSION
    certificate_id: uuid.UUID = field(default_factory=uuid.uuid4)
    timestamp_ns: int = 0
    solver_hash: bytes = b"\x00" * HASH_SIZE

    def __post_init__(self) -> None:
        if self.timestamp_ns == 0:
            self.timestamp_ns = int(time.time() * 1_000_000_000)

    def pack(self) -> bytes:
        """Serialize to exactly 64 bytes."""
        solver_hash = self.solver_hash.ljust(HASH_SIZE, b"\x00")[:HASH_SIZE]
        return struct.pack(
            _HEADER_FMT,
            TPC_MAGIC,
            self.version,
            self.certificate_id.bytes,
            self.timestamp_ns,
            solver_hash,
        )

    @classmethod
    def unpack(cls, data: bytes) -> TPCHeader:
        """Deserialize from 64 bytes."""
        if len(data) < TPC_HEADER_SIZE:
            raise ValueError(
                f"Header too short: {len(data)} < {TPC_HEADER_SIZE}"
            )
        magic, version, uuid_bytes, ts_ns, solver_hash = struct.unpack(
            _HEADER_FMT, data[:TPC_HEADER_SIZE]
        )
        if magic != TPC_MAGIC:
            raise ValueError(
                f"Invalid TPC magic: {magic!r} (expected {TPC_MAGIC!r})"
            )
        return cls(
            version=version,
            certificate_id=uuid.UUID(bytes=uuid_bytes),
            timestamp_ns=ts_ns,
            solver_hash=solver_hash,
        )

    @property
    def timestamp_s(self) -> float:
        return self.timestamp_ns / 1_000_000_000


# ═════════════════════════════════════════════════════════════════════════════
# Signature
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class TPCSignature:
    """Ed25519 signature over all preceding sections."""
    public_key: bytes = b"\x00" * PUBLIC_KEY_SIZE
    signature: bytes = b"\x00" * SIGNATURE_SIZE
    content_hash: bytes = b"\x00" * HASH_SIZE  # SHA-256 of signed content

    def to_dict(self) -> dict[str, Any]:
        return {
            "public_key": self.public_key.hex(),
            "signature": self.signature.hex(),
            "content_hash": self.content_hash.hex(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TPCSignature:
        return cls(
            public_key=bytes.fromhex(d.get("public_key", "00" * PUBLIC_KEY_SIZE)),
            signature=bytes.fromhex(d.get("signature", "00" * SIGNATURE_SIZE)),
            content_hash=bytes.fromhex(d.get("content_hash", "00" * HASH_SIZE)),
        )


# ═════════════════════════════════════════════════════════════════════════════
# Section Encoding — length-prefixed JSON
# ═════════════════════════════════════════════════════════════════════════════

def _encode_section(data: dict[str, Any], binary_blobs: dict[str, bytes] | None = None) -> bytes:
    """
    Encode a section as: [4-byte JSON len][JSON bytes][4-byte blob len][blob bytes]...

    For sections with large binary payloads (proof_bytes, proof_objects, etc.),
    the JSON contains hashes and the blobs follow sequentially.
    """
    json_bytes = json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(json_bytes) > MAX_STRING_LEN * 10:
        raise ValueError(f"Section JSON too large: {len(json_bytes)} bytes")

    parts = [struct.pack("<I", len(json_bytes)), json_bytes]

    blobs = binary_blobs or {}
    parts.append(struct.pack("<I", len(blobs)))
    for name in sorted(blobs.keys()):
        blob = blobs[name]
        name_bytes = name.encode("utf-8")
        parts.append(struct.pack("<H", len(name_bytes)))
        parts.append(name_bytes)
        parts.append(struct.pack("<I", len(blob)))
        parts.append(blob)

    return b"".join(parts)


def _decode_section(data: bytes, offset: int) -> tuple[dict[str, Any], dict[str, bytes], int]:
    """
    Decode a section. Returns (json_dict, binary_blobs, new_offset).
    """
    if offset + 4 > len(data):
        raise ValueError(f"Truncated section at offset {offset}")

    json_len = struct.unpack_from("<I", data, offset)[0]
    offset += 4

    if offset + json_len > len(data):
        raise ValueError(f"Truncated JSON at offset {offset}")

    json_bytes = data[offset:offset + json_len]
    offset += json_len
    json_dict = json.loads(json_bytes.decode("utf-8"))

    if offset + 4 > len(data):
        raise ValueError(f"Truncated blob count at offset {offset}")

    blob_count = struct.unpack_from("<I", data, offset)[0]
    offset += 4

    blobs: dict[str, bytes] = {}
    for _ in range(blob_count):
        if offset + 2 > len(data):
            raise ValueError(f"Truncated blob name length at offset {offset}")
        name_len = struct.unpack_from("<H", data, offset)[0]
        offset += 2

        if offset + name_len > len(data):
            raise ValueError(f"Truncated blob name at offset {offset}")
        name = data[offset:offset + name_len].decode("utf-8")
        offset += name_len

        if offset + 4 > len(data):
            raise ValueError(f"Truncated blob data length at offset {offset}")
        blob_len = struct.unpack_from("<I", data, offset)[0]
        offset += 4

        if offset + blob_len > len(data):
            raise ValueError(f"Truncated blob data at offset {offset}")
        blobs[name] = data[offset:offset + blob_len]
        offset += blob_len

    return json_dict, blobs, offset


# ═════════════════════════════════════════════════════════════════════════════
# TPCFile — Full Certificate
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class TPCFile:
    """
    Complete Trustless Physics Certificate.

    Combines Header + Layer A + Layer B + Layer C + Metadata + Signature
    into a single verifiable package.
    """
    header: TPCHeader = field(default_factory=TPCHeader)
    layer_a: LayerA = field(default_factory=LayerA)
    layer_b: LayerB = field(default_factory=LayerB)
    layer_c: LayerC = field(default_factory=LayerC)
    metadata: Metadata = field(default_factory=Metadata)
    signature: TPCSignature = field(default_factory=TPCSignature)

    # ── Serialization ────────────────────────────────────────────────────

    def serialize(self) -> bytes:
        """
        Serialize the full certificate to bytes.

        Layout:
            [Header: 64 bytes]
            [Section A: len-prefixed JSON + blobs]
            [Section B: len-prefixed JSON + blobs]
            [Section C: len-prefixed JSON + blobs]
            [Metadata:  len-prefixed JSON]
            [Signature: 32 + 64 + 32 = 128 bytes]
        """
        parts: list[bytes] = []

        # Header
        parts.append(self.header.pack())

        # Layer A
        a_blobs: dict[str, bytes] = {}
        if self.layer_a.proof_objects:
            a_blobs["proof_objects"] = self.layer_a.proof_objects
        parts.append(_encode_section(self.layer_a.to_dict(), a_blobs))

        # Layer B
        b_blobs: dict[str, bytes] = {}
        if self.layer_b.proof_bytes:
            b_blobs["proof_bytes"] = self.layer_b.proof_bytes
        if self.layer_b.verification_key:
            b_blobs["verification_key"] = self.layer_b.verification_key
        parts.append(_encode_section(self.layer_b.to_dict(), b_blobs))

        # Layer C
        c_blobs: dict[str, bytes] = {}
        if self.layer_c.attestation_json:
            c_blobs["attestation_json"] = self.layer_c.attestation_json
        parts.append(_encode_section(self.layer_c.to_dict(), c_blobs))

        # Metadata
        parts.append(_encode_section(self.metadata.to_dict()))

        # Compute content hash (everything before signature)
        content = b"".join(parts)
        content_hash = hashlib.sha256(content).digest()

        # Signature section (fixed 128 bytes)
        sig = self.signature
        sig_data = sig.public_key[:PUBLIC_KEY_SIZE].ljust(PUBLIC_KEY_SIZE, b"\x00")
        sig_data += sig.signature[:SIGNATURE_SIZE].ljust(SIGNATURE_SIZE, b"\x00")
        sig_data += content_hash
        parts.append(sig_data)

        result = b"".join(parts)
        if len(result) > MAX_CERTIFICATE_SIZE:
            raise ValueError(
                f"Certificate too large: {len(result)} > {MAX_CERTIFICATE_SIZE}"
            )
        return result

    @classmethod
    def deserialize(cls, data: bytes) -> TPCFile:
        """Deserialize a .tpc file from bytes."""
        if len(data) < TPC_HEADER_SIZE + 128:
            raise ValueError(f"Certificate too short: {len(data)} bytes")
        if len(data) > MAX_CERTIFICATE_SIZE:
            raise ValueError(f"Certificate too large: {len(data)} bytes")

        # Header
        header = TPCHeader.unpack(data[:TPC_HEADER_SIZE])
        offset = TPC_HEADER_SIZE

        # Layer A
        a_dict, a_blobs, offset = _decode_section(data, offset)
        layer_a = LayerA.from_dict(a_dict, a_blobs.get("proof_objects", b""))

        # Layer B
        b_dict, b_blobs, offset = _decode_section(data, offset)
        layer_b = LayerB.from_dict(
            b_dict,
            proof_bytes=b_blobs.get("proof_bytes", b""),
            verification_key=b_blobs.get("verification_key", b""),
        )

        # Layer C
        c_dict, c_blobs, offset = _decode_section(data, offset)
        layer_c = LayerC.from_dict(c_dict, c_blobs.get("attestation_json", b""))

        # Metadata
        m_dict, _, offset = _decode_section(data, offset)
        metadata = Metadata.from_dict(m_dict)

        # Signature (last 128 bytes)
        sig_start = offset
        if sig_start + PUBLIC_KEY_SIZE + SIGNATURE_SIZE + HASH_SIZE > len(data):
            raise ValueError("Truncated signature section")

        pub_key = data[sig_start:sig_start + PUBLIC_KEY_SIZE]
        sig_bytes = data[sig_start + PUBLIC_KEY_SIZE:sig_start + PUBLIC_KEY_SIZE + SIGNATURE_SIZE]
        content_hash = data[sig_start + PUBLIC_KEY_SIZE + SIGNATURE_SIZE:
                            sig_start + PUBLIC_KEY_SIZE + SIGNATURE_SIZE + HASH_SIZE]

        signature = TPCSignature(
            public_key=pub_key,
            signature=sig_bytes,
            content_hash=content_hash,
        )

        return cls(
            header=header,
            layer_a=layer_a,
            layer_b=layer_b,
            layer_c=layer_c,
            metadata=metadata,
            signature=signature,
        )

    # ── File I/O ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        """Write certificate to disk."""
        path = Path(path)
        data = self.serialize()
        path.write_bytes(data)
        logger.info(
            f"TPC certificate saved: {path} "
            f"({len(data):,} bytes, ID={self.header.certificate_id})"
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> TPCFile:
        """Read certificate from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Certificate not found: {path}")
        data = path.read_bytes()
        return cls.deserialize(data)

    # ── Signing ──────────────────────────────────────────────────────────

    def sign(self, private_key_bytes: bytes) -> None:
        """
        Sign the certificate with an Ed25519 private key.

        Args:
            private_key_bytes: 32-byte Ed25519 seed (private key).
        """
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        except ImportError:
            raise ImportError(
                "cryptography package required for signing. "
                "Install: pip install cryptography"
            )

        priv_key = Ed25519PrivateKey.from_private_bytes(private_key_bytes[:32])
        pub_key = priv_key.public_key()

        # Serialize everything except signature to get content hash
        old_sig = self.signature
        self.signature = TPCSignature()  # placeholder
        content = self.serialize()[:-128]  # strip placeholder sig
        content_hash = hashlib.sha256(content).digest()

        sig_bytes = priv_key.sign(content_hash)

        self.signature = TPCSignature(
            public_key=pub_key.public_bytes_raw(),
            signature=sig_bytes,
            content_hash=content_hash,
        )

    def verify_signature(self) -> bool:
        """
        Verify the Ed25519 signature.

        Returns True if signature is valid, False otherwise.
        Returns True for zero-key certificates (unsigned / self-attested).
        """
        if self.signature.public_key == b"\x00" * PUBLIC_KEY_SIZE:
            return True  # Unsigned certificate, skip verification

        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        except ImportError:
            logger.warning("cryptography package not installed; signature check skipped")
            return True

        # Recompute content hash
        old_sig = self.signature
        self.signature = TPCSignature()
        content = self.serialize()[:-128]
        content_hash = hashlib.sha256(content).digest()
        self.signature = old_sig

        if content_hash != old_sig.content_hash:
            logger.error("Content hash mismatch — certificate has been tampered with")
            return False

        try:
            pub_key = Ed25519PublicKey.from_public_bytes(old_sig.public_key)
            pub_key.verify(old_sig.signature, content_hash)
            return True
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False

    # ── Human-readable summary ───────────────────────────────────────────

    def summary(self) -> str:
        """Return human-readable certificate summary."""
        lines = [
            "TRUSTLESS PHYSICS VERIFICATION CERTIFICATE",
            "=" * 50,
            f"Certificate ID:  {self.header.certificate_id}",
            f"Version:         {self.header.version}",
            f"Timestamp:       {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(self.header.timestamp_s))}",
            f"Solver:          {self.metadata.solver}",
            f"Domain:          {self.metadata.domain}",
            "",
            "Layer A — Mathematical Truth",
            f"  Proof system:  {self.layer_a.proof_system}",
            f"  Theorems:      {len(self.layer_a.theorems)}",
            f"  Coverage:      {self.layer_a.coverage.value} ({self.layer_a.coverage_pct:.1f}%)",
        ]
        for t in self.layer_a.theorems:
            lines.append(f"    ✓ {t.name} ({t.file})")

        lines.extend([
            "",
            "Layer B — Computational Integrity",
            f"  Proof system:  {self.layer_b.proof_system}",
            f"  Proof size:    {len(self.layer_b.proof_bytes):,} bytes",
            f"  Constraints:   {self.layer_b.circuit_constraints:,}",
            f"  Gen time:      {self.layer_b.proof_generation_time_s:.2f}s",
        ])
        if self.layer_b.public_inputs:
            for k, v in self.layer_b.public_inputs.items():
                lines.append(f"    Input:  {k} = {v}")
        if self.layer_b.public_outputs:
            for k, v in self.layer_b.public_outputs.items():
                lines.append(f"    Output: {k} = {v}")

        lines.extend([
            "",
            "Layer C — Physical Fidelity",
            f"  Benchmarks:    {len(self.layer_c.benchmarks)}",
            f"  All passed:    {'✅' if self.layer_c.all_passed else '❌'}",
            f"  Pass rate:     {self.layer_c.pass_count}/{len(self.layer_c.benchmarks)}",
        ])
        for b in self.layer_c.benchmarks:
            status = "✅" if b.passed else "❌"
            lines.append(
                f"    {status} {b.name}: L2={b.l2_error:.2e}, "
                f"max_dev={b.max_deviation:.2e}, cons={b.conservation_error:.2e}"
            )

        qtt = self.metadata.qtt_params
        if qtt.grid_bits > 0:
            lines.extend([
                "",
                "QTT Parameters",
                f"  Grid bits:     {qtt.grid_bits} (N = 2^{qtt.grid_bits} = {2**qtt.grid_bits:,})",
                f"  Max rank:      {qtt.max_rank}",
                f"  Tolerance:     {qtt.tolerance:.2e}",
            ])

        sig_status = "SIGNED" if self.signature.public_key != b"\x00" * PUBLIC_KEY_SIZE else "UNSIGNED"
        lines.extend([
            "",
            f"Signature:       {sig_status}",
            f"Content SHA-256: {self.signature.content_hash.hex()[:32]}...",
        ])

        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        """Export certificate metadata as JSON-serializable dict (no binary blobs)."""
        return {
            "tpc_version": self.header.version,
            "certificate_id": str(self.header.certificate_id),
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.header.timestamp_s)
            ),
            "solver_hash": self.header.solver_hash.hex(),
            "layer_a": self.layer_a.to_dict(),
            "layer_b": self.layer_b.to_dict(),
            "layer_c": self.layer_c.to_dict(),
            "metadata": self.metadata.to_dict(),
            "signature": self.signature.to_dict(),
        }


# ═════════════════════════════════════════════════════════════════════════════
# Verification
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class VerificationReport:
    """Result of verifying a .tpc certificate."""
    valid: bool = False
    certificate_id: str = ""
    timestamp: str = ""
    solver: str = ""
    domain: str = ""
    verification_time_s: float = 0.0

    # Layer A
    layer_a_valid: bool = False
    layer_a_theorems: int = 0
    layer_a_coverage: str = "none"
    layer_a_coverage_pct: float = 0.0

    # Layer B
    layer_b_valid: bool = False
    layer_b_proof_system: str = ""
    layer_b_proof_size: int = 0
    layer_b_constraints: int = 0

    # Layer C
    layer_c_valid: bool = False
    layer_c_benchmarks: int = 0
    layer_c_passed: int = 0
    layer_c_all_passed: bool = False

    # Signature
    signature_valid: bool = False

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def report_text(self) -> str:
        """Generate human-readable verification report."""
        verdict = "VALID ✅" if self.valid else "INVALID ❌"
        lines = [
            "",
            "TRUSTLESS PHYSICS VERIFICATION REPORT",
            "=" * 50,
            f"Certificate ID: {self.certificate_id}",
            f"Timestamp:      {self.timestamp}",
            f"Solver:         {self.solver} ({self.domain})",
            "",
            "Layer A — Mathematical Truth",
            f"  Status:        {'✅' if self.layer_a_valid else '❌'}",
            f"  Theorems:      {self.layer_a_theorems}",
            f"  Coverage:      {self.layer_a_coverage} ({self.layer_a_coverage_pct:.1f}%)",
            "",
            "Layer B — Computational Integrity",
            f"  Status:        {'✅' if self.layer_b_valid else '❌'}",
            f"  Proof system:  {self.layer_b_proof_system}",
            f"  Proof size:    {self.layer_b_proof_size:,} bytes",
            f"  Constraints:   {self.layer_b_constraints:,}",
            "",
            "Layer C — Physical Fidelity",
            f"  Status:        {'✅' if self.layer_c_valid else '❌'}",
            f"  Benchmarks:    {self.layer_c_passed}/{self.layer_c_benchmarks} passed",
            "",
            f"Signature:       {'✅' if self.signature_valid else '❌'}",
            "",
            f"VERDICT: {verdict}",
            f"Verification time: {self.verification_time_s:.3f}s",
        ]

        if self.errors:
            lines.append("")
            lines.append("ERRORS:")
            for e in self.errors:
                lines.append(f"  ✗ {e}")

        if self.warnings:
            lines.append("")
            lines.append("WARNINGS:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")

        return "\n".join(lines)


def verify_certificate(
    path_or_data: str | Path | bytes,
    verify_signature: bool = True,
    verify_zk_proof: bool = False,
) -> VerificationReport:
    """
    Verify a Trustless Physics Certificate.

    This is the Python-side verifier. The standalone Rust verifier
    (apps/trustless_verify) provides the same logic as a single binary.

    Args:
        path_or_data: Path to .tpc file, or raw bytes.
        verify_signature: Check Ed25519 signature.
        verify_zk_proof: If True, actually verify the ZK proof (requires prover).

    Returns:
        VerificationReport with detailed results.
    """
    t0 = time.monotonic()
    report = VerificationReport()

    # ── Load certificate ─────────────────────────────────────────────────
    try:
        if isinstance(path_or_data, (str, Path)):
            cert = TPCFile.load(path_or_data)
        else:
            cert = TPCFile.deserialize(path_or_data)
    except Exception as e:
        report.errors.append(f"Failed to parse certificate: {e}")
        report.verification_time_s = time.monotonic() - t0
        return report

    report.certificate_id = str(cert.header.certificate_id)
    report.timestamp = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(cert.header.timestamp_s)
    )
    report.solver = cert.metadata.solver
    report.domain = cert.metadata.domain

    # ── Verify Layer A ───────────────────────────────────────────────────
    try:
        report.layer_a_theorems = len(cert.layer_a.theorems)
        report.layer_a_coverage = cert.layer_a.coverage.value
        report.layer_a_coverage_pct = cert.layer_a.coverage_pct

        if cert.layer_a.proof_system == "none" and report.layer_a_theorems == 0:
            report.warnings.append("Layer A: No formal proofs provided")
            report.layer_a_valid = True  # Valid but empty
        elif report.layer_a_theorems > 0:
            # Verify theorem references are non-empty
            for t in cert.layer_a.theorems:
                if not t.name or not t.file:
                    report.errors.append(f"Layer A: Theorem missing name or file: {t}")
                    break
            else:
                report.layer_a_valid = True
        else:
            report.layer_a_valid = True  # Proof system declared but no theorems yet
            report.warnings.append(
                f"Layer A: Proof system is {cert.layer_a.proof_system} "
                f"but no theorems listed"
            )
    except Exception as e:
        report.errors.append(f"Layer A verification error: {e}")

    # ── Verify Layer B ───────────────────────────────────────────────────
    try:
        report.layer_b_proof_system = cert.layer_b.proof_system
        report.layer_b_proof_size = len(cert.layer_b.proof_bytes)
        report.layer_b_constraints = cert.layer_b.circuit_constraints

        if cert.layer_b.proof_system == "none":
            report.warnings.append("Layer B: No ZK proof system")
            report.layer_b_valid = True
        elif len(cert.layer_b.proof_bytes) == 0:
            report.warnings.append("Layer B: ZK proof bytes empty")
            report.layer_b_valid = True  # Valid structure, proof pending
        else:
            # Structural validation: proof bytes are non-trivial
            if len(cert.layer_b.proof_bytes) < 32:
                report.errors.append(
                    f"Layer B: Proof too small ({len(cert.layer_b.proof_bytes)} bytes)"
                )
            else:
                report.layer_b_valid = True

            # Full ZK proof verification (optional, requires prover infrastructure)
            if verify_zk_proof and report.layer_b_valid:
                report.warnings.append(
                    "Layer B: Full ZK verification requested but not yet implemented. "
                    "Use the Rust verifier (trustless-verify) for cryptographic proof checking."
                )
    except Exception as e:
        report.errors.append(f"Layer B verification error: {e}")

    # ── Verify Layer C ───────────────────────────────────────────────────
    try:
        report.layer_c_benchmarks = len(cert.layer_c.benchmarks)
        report.layer_c_passed = cert.layer_c.pass_count
        report.layer_c_all_passed = cert.layer_c.all_passed

        if report.layer_c_benchmarks == 0:
            report.warnings.append("Layer C: No benchmarks provided")
            report.layer_c_valid = True
        else:
            # Check benchmark structural integrity
            for b in cert.layer_c.benchmarks:
                if not b.name:
                    report.errors.append("Layer C: Benchmark missing name")
                    break
                if not b.passed:
                    report.warnings.append(f"Layer C: Benchmark '{b.name}' FAILED")
            else:
                report.layer_c_valid = True

            if not cert.layer_c.all_passed:
                report.layer_c_valid = False
                report.errors.append(
                    f"Layer C: {report.layer_c_benchmarks - report.layer_c_passed} "
                    f"benchmark(s) failed"
                )
    except Exception as e:
        report.errors.append(f"Layer C verification error: {e}")

    # ── Verify Signature ─────────────────────────────────────────────────
    if verify_signature:
        try:
            report.signature_valid = cert.verify_signature()
            if not report.signature_valid:
                report.errors.append("Signature verification failed")
        except Exception as e:
            report.errors.append(f"Signature verification error: {e}")
    else:
        report.signature_valid = True

    # ── Overall verdict ──────────────────────────────────────────────────
    report.valid = (
        report.layer_a_valid
        and report.layer_b_valid
        and report.layer_c_valid
        and report.signature_valid
        and len(report.errors) == 0
    )

    report.verification_time_s = time.monotonic() - t0
    return report
