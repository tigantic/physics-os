"""
TPC Certificate parser and local verification.

Handles the binary TPC certificate format:
  - Header (64 bytes): magic + version + UUID + timestamp + solver_hash
  - Layer sections (variable): JSON metadata + optional binary blobs
  - Signature section (128 bytes): Ed25519 pubkey + signature + SHA-256 hash
"""

from __future__ import annotations

import hashlib
import json
import struct
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from nacl.signing import VerifyKey
from nacl.exceptions import BadSignatureError

from fluidelite_verify.errors import InvalidCertificate, InvalidHash, InvalidSignature

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

TPC_MAGIC = b"TPC\x01"
HEADER_SIZE = 64
SIGNATURE_SECTION_SIZE = 128
PUBKEY_SIZE = 32
SIGNATURE_SIZE = 64
HASH_SIZE = 32

DOMAIN_MAP = {
    0: "thermal",
    1: "euler3d",
    2: "ns_imex",
    3: "fluidelite",
}


# ═══════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CertificateHeader:
    """Parsed TPC certificate header (64 bytes)."""
    magic: bytes
    version: int
    certificate_id: str  # UUID string
    timestamp_ns: int
    solver_hash: bytes  # 32 bytes

    @property
    def timestamp(self) -> datetime:
        """Convert nanosecond timestamp to datetime."""
        return datetime.fromtimestamp(
            self.timestamp_ns / 1_000_000_000, tz=timezone.utc
        )


@dataclass
class Blob:
    """Binary blob attached to a layer."""
    name: str
    data: bytes


@dataclass
class Layer:
    """A certificate layer (JSON metadata + optional blobs)."""
    metadata: dict[str, Any]
    blobs: list[Blob] = field(default_factory=list)


@dataclass
class SignatureSection:
    """Parsed signature section (128 bytes)."""
    pubkey: bytes  # 32 bytes Ed25519
    signature: bytes  # 64 bytes
    content_hash: bytes  # 32 bytes SHA-256

    @property
    def pubkey_hex(self) -> str:
        return self.pubkey.hex()

    @property
    def signature_hex(self) -> str:
        return self.signature.hex()

    @property
    def content_hash_hex(self) -> str:
        return self.content_hash.hex()


@dataclass
class VerificationResult:
    """Result of local certificate verification."""
    valid: bool
    hash_valid: bool
    signature_valid: bool
    certificate_id: str
    signer_pubkey: str
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════
# Certificate
# ═══════════════════════════════════════════════════════════════════════════

class Certificate:
    """TPC Certificate: parse, inspect, and verify locally."""

    def __init__(self, raw: bytes) -> None:
        self._raw = raw
        self._header: Optional[CertificateHeader] = None
        self._layers: Optional[list[Layer]] = None
        self._signature_section: Optional[SignatureSection] = None
        self._parse()

    @classmethod
    def from_file(cls, path: str | Path) -> Certificate:
        """Load a TPC certificate from a file."""
        data = Path(path).read_bytes()
        return cls(data)

    @classmethod
    def from_hex(cls, hex_str: str) -> Certificate:
        """Decode a TPC certificate from hex string."""
        return cls(bytes.fromhex(hex_str))

    @classmethod
    def from_bytes(cls, data: bytes) -> Certificate:
        """Create a certificate from raw bytes."""
        return cls(data)

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def raw(self) -> bytes:
        """Raw certificate bytes."""
        return self._raw

    @property
    def header(self) -> CertificateHeader:
        """Parsed certificate header."""
        assert self._header is not None
        return self._header

    @property
    def layers(self) -> list[Layer]:
        """Parsed certificate layers."""
        assert self._layers is not None
        return self._layers

    @property
    def signature_section(self) -> SignatureSection:
        """Parsed signature section."""
        assert self._signature_section is not None
        return self._signature_section

    @property
    def certificate_id(self) -> str:
        """Certificate UUID."""
        return self.header.certificate_id

    @property
    def content_hash(self) -> str:
        """SHA-256 hash of the certificate content (hex)."""
        content = self._raw[:-SIGNATURE_SECTION_SIZE]
        return hashlib.sha256(content).hexdigest()

    @property
    def domain(self) -> Optional[str]:
        """Domain extracted from layer metadata, if available."""
        for layer in self.layers:
            if "domain" in layer.metadata:
                return layer.metadata["domain"]
        return None

    @property
    def size_bytes(self) -> int:
        """Total certificate size in bytes."""
        return len(self._raw)

    # ── Parsing ──────────────────────────────────────────────────────────

    def _parse(self) -> None:
        """Parse the binary TPC certificate format."""
        data = self._raw

        if len(data) < HEADER_SIZE + SIGNATURE_SECTION_SIZE:
            raise InvalidCertificate(
                f"Certificate too short: {len(data)} bytes "
                f"(minimum {HEADER_SIZE + SIGNATURE_SECTION_SIZE})"
            )

        # Validate magic
        if data[:4] != TPC_MAGIC:
            raise InvalidCertificate(
                f"Invalid TPC magic: {data[:4]!r} (expected {TPC_MAGIC!r})"
            )

        # Parse header (64 bytes)
        magic = data[0:4]
        version = struct.unpack_from("<I", data, 4)[0]
        cert_id_bytes = data[8:24]
        timestamp_ns = struct.unpack_from("<q", data, 24)[0]
        solver_hash = data[32:64]

        cert_id = str(uuid.UUID(bytes=cert_id_bytes))

        self._header = CertificateHeader(
            magic=magic,
            version=version,
            certificate_id=cert_id,
            timestamp_ns=timestamp_ns,
            solver_hash=solver_hash,
        )

        # Parse signature section (last 128 bytes)
        sig_start = len(data) - SIGNATURE_SECTION_SIZE
        sig_data = data[sig_start:]

        self._signature_section = SignatureSection(
            pubkey=sig_data[:PUBKEY_SIZE],
            signature=sig_data[PUBKEY_SIZE : PUBKEY_SIZE + SIGNATURE_SIZE],
            content_hash=sig_data[PUBKEY_SIZE + SIGNATURE_SIZE :],
        )

        # Parse layers (between header and signature section)
        self._layers = []
        offset = HEADER_SIZE
        layer_end = sig_start

        while offset < layer_end:
            if offset + 4 > layer_end:
                break

            # Read JSON length
            json_len = struct.unpack_from("<I", data, offset)[0]
            offset += 4

            if offset + json_len > layer_end:
                break

            # Read JSON metadata
            json_bytes = data[offset : offset + json_len]
            offset += json_len

            try:
                metadata = json.loads(json_bytes)
            except json.JSONDecodeError:
                metadata = {"raw": json_bytes.hex()}

            # Read blob count
            if offset + 4 > layer_end:
                self._layers.append(Layer(metadata=metadata))
                break

            blob_count = struct.unpack_from("<I", data, offset)[0]
            offset += 4

            blobs: list[Blob] = []
            for _ in range(blob_count):
                if offset + 2 > layer_end:
                    break

                # Blob name length (u16)
                name_len = struct.unpack_from("<H", data, offset)[0]
                offset += 2

                if offset + name_len > layer_end:
                    break

                name = data[offset : offset + name_len].decode("utf-8", errors="replace")
                offset += name_len

                if offset + 4 > layer_end:
                    break

                # Blob data length (u32)
                blob_len = struct.unpack_from("<I", data, offset)[0]
                offset += 4

                if offset + blob_len > layer_end:
                    break

                blob_data = data[offset : offset + blob_len]
                offset += blob_len

                blobs.append(Blob(name=name, data=blob_data))

            self._layers.append(Layer(metadata=metadata, blobs=blobs))

    # ── Verification ─────────────────────────────────────────────────────

    def verify(self) -> VerificationResult:
        """
        Verify the certificate's integrity:
        1. SHA-256 content hash matches stored hash
        2. Ed25519 signature over the hash is valid

        Returns a VerificationResult with detailed status.
        """
        sig = self.signature_section
        cert_id = self.header.certificate_id

        # Step 1: Verify content hash
        content = self._raw[:-SIGNATURE_SECTION_SIZE]
        computed_hash = hashlib.sha256(content).digest()
        hash_valid = computed_hash == sig.content_hash

        if not hash_valid:
            return VerificationResult(
                valid=False,
                hash_valid=False,
                signature_valid=False,
                certificate_id=cert_id,
                signer_pubkey=sig.pubkey_hex,
                error="Content hash mismatch",
            )

        # Step 2: Verify Ed25519 signature
        try:
            vk = VerifyKey(sig.pubkey)
            vk.verify(computed_hash, sig.signature)
            signature_valid = True
        except (BadSignatureError, Exception) as e:
            return VerificationResult(
                valid=False,
                hash_valid=True,
                signature_valid=False,
                certificate_id=cert_id,
                signer_pubkey=sig.pubkey_hex,
                error=f"Signature verification failed: {e}",
            )

        return VerificationResult(
            valid=True,
            hash_valid=True,
            signature_valid=True,
            certificate_id=cert_id,
            signer_pubkey=sig.pubkey_hex,
        )

    def verify_strict(self) -> None:
        """Verify the certificate, raising exceptions on failure."""
        result = self.verify()
        if not result.hash_valid:
            raise InvalidHash(
                f"Content hash mismatch for certificate {result.certificate_id}"
            )
        if not result.signature_valid:
            raise InvalidSignature(
                f"Signature invalid for certificate {result.certificate_id}: "
                f"{result.error}"
            )

    # ── Display ──────────────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Return a human-readable summary of the certificate."""
        return {
            "certificate_id": self.header.certificate_id,
            "version": self.header.version,
            "timestamp": self.header.timestamp.isoformat(),
            "domain": self.domain,
            "solver_hash": self.header.solver_hash.hex(),
            "signer_pubkey": self.signature_section.pubkey_hex,
            "content_hash": self.content_hash,
            "size_bytes": self.size_bytes,
            "layers": len(self.layers),
            "total_blobs": sum(len(l.blobs) for l in self.layers),
        }

    def __repr__(self) -> str:
        return (
            f"Certificate(id={self.certificate_id!r}, "
            f"domain={self.domain!r}, size={self.size_bytes})"
        )
