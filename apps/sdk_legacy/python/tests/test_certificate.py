"""Tests for the fluidelite-verify Python SDK."""

from __future__ import annotations

import hashlib
import struct
import uuid
from unittest.mock import MagicMock, patch

import pytest
from nacl.signing import SigningKey

from fluidelite_verify.certificate import (
    Certificate,
    CertificateHeader,
    Layer,
    SignatureSection,
    VerificationResult,
    HEADER_SIZE,
    SIGNATURE_SECTION_SIZE,
    TPC_MAGIC,
)
from fluidelite_verify.errors import (
    InvalidCertificate,
    InvalidHash,
    InvalidSignature,
)
from fluidelite_verify.verifier import TPCVerifier, OnChainResult
from fluidelite_verify.client import TPCClient, IssueResult


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

def _build_tpc_certificate(
    domain: str = "thermal",
    proof: bytes = b"\xde\xad\xbe\xef",
    signing_key: SigningKey | None = None,
) -> bytes:
    """Build a valid TPC certificate for testing."""
    if signing_key is None:
        signing_key = SigningKey(b"\x42" * 32)

    cert_id = uuid.uuid4()
    timestamp_ns = 1700000000_000000000

    # Header (64 bytes)
    data = bytearray()
    data.extend(TPC_MAGIC)                                  # magic (4)
    data.extend(struct.pack("<I", 1))                       # version (4)
    data.extend(cert_id.bytes)                              # UUID (16)
    data.extend(struct.pack("<q", timestamp_ns))            # timestamp (8)
    data.extend(b"\x00" * 32)                               # solver_hash (32)
    assert len(data) == HEADER_SIZE

    # Layer A: Mathematical Truth
    import json
    layer_a = json.dumps({"type": "mathematical_truth", "domain": domain}).encode()
    data.extend(struct.pack("<I", len(layer_a)))
    data.extend(layer_a)
    data.extend(struct.pack("<I", 0))  # 0 blobs

    # Layer B: Computational Integrity
    proof_hash = hashlib.sha256(proof).hexdigest()
    layer_b = json.dumps({
        "type": "computational_integrity",
        "proof_hash": proof_hash,
    }).encode()
    data.extend(struct.pack("<I", len(layer_b)))
    data.extend(layer_b)
    # 1 blob: proof
    data.extend(struct.pack("<I", 1))
    blob_name = b"proof"
    data.extend(struct.pack("<H", len(blob_name)))
    data.extend(blob_name)
    data.extend(struct.pack("<I", len(proof)))
    data.extend(proof)

    # Layer C: Physical Fidelity
    layer_c = json.dumps({"type": "physical_fidelity", "domain": domain}).encode()
    data.extend(struct.pack("<I", len(layer_c)))
    data.extend(layer_c)
    data.extend(struct.pack("<I", 0))  # 0 blobs

    # Metadata layer
    meta = json.dumps({"ca_version": "1.0.0"}).encode()
    data.extend(struct.pack("<I", len(meta)))
    data.extend(meta)
    data.extend(struct.pack("<I", 0))  # 0 blobs

    # Sign
    content = bytes(data)
    content_hash = hashlib.sha256(content).digest()
    signature = signing_key.sign(content_hash).signature

    # Signature section (128 bytes)
    data.extend(signing_key.verify_key.encode())  # pubkey (32)
    data.extend(signature)                          # signature (64)
    data.extend(content_hash)                       # hash (32)

    return bytes(data)


@pytest.fixture
def signing_key() -> SigningKey:
    return SigningKey(b"\x42" * 32)


@pytest.fixture
def valid_cert(signing_key: SigningKey) -> bytes:
    return _build_tpc_certificate(signing_key=signing_key)


# ═══════════════════════════════════════════════════════════════════════════
# Certificate Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCertificate:
    def test_parse_valid_certificate(self, valid_cert: bytes) -> None:
        cert = Certificate(valid_cert)
        assert cert.header.magic == TPC_MAGIC
        assert cert.header.version == 1
        assert cert.size_bytes == len(valid_cert)
        assert len(cert.layers) == 4
        assert cert.domain == "thermal"

    def test_parse_too_short(self) -> None:
        with pytest.raises(InvalidCertificate, match="too short"):
            Certificate(b"\x00" * 10)

    def test_parse_wrong_magic(self) -> None:
        data = b"\x00" * (HEADER_SIZE + SIGNATURE_SECTION_SIZE)
        with pytest.raises(InvalidCertificate, match="Invalid TPC magic"):
            Certificate(data)

    def test_verify_valid(self, valid_cert: bytes) -> None:
        cert = Certificate(valid_cert)
        result = cert.verify()
        assert result.valid is True
        assert result.hash_valid is True
        assert result.signature_valid is True
        assert result.error is None

    def test_verify_tampered_content(self, valid_cert: bytes) -> None:
        tampered = bytearray(valid_cert)
        tampered[20] ^= 0xFF
        cert = Certificate(bytes(tampered))
        result = cert.verify()
        assert result.valid is False
        assert result.hash_valid is False

    def test_verify_tampered_signature(self, valid_cert: bytes) -> None:
        tampered = bytearray(valid_cert)
        # Tamper signature byte within signature section
        sig_offset = len(tampered) - SIGNATURE_SECTION_SIZE + 32 + 10
        tampered[sig_offset] ^= 0xFF
        cert = Certificate(bytes(tampered))
        result = cert.verify()
        assert result.valid is False
        assert result.signature_valid is False

    def test_verify_strict_valid(self, valid_cert: bytes) -> None:
        cert = Certificate(valid_cert)
        cert.verify_strict()  # Should not raise

    def test_verify_strict_raises_on_tampered(self, valid_cert: bytes) -> None:
        tampered = bytearray(valid_cert)
        tampered[20] ^= 0xFF
        cert = Certificate(bytes(tampered))
        with pytest.raises(InvalidHash):
            cert.verify_strict()

    def test_from_hex(self, valid_cert: bytes) -> None:
        cert = Certificate.from_hex(valid_cert.hex())
        assert cert.verify().valid

    def test_summary(self, valid_cert: bytes) -> None:
        cert = Certificate(valid_cert)
        s = cert.summary()
        assert "certificate_id" in s
        assert "version" in s
        assert s["layers"] == 4
        assert s["domain"] == "thermal"

    def test_layers_with_blobs(self, valid_cert: bytes) -> None:
        cert = Certificate(valid_cert)
        # Layer B should have the proof blob
        found_blob = False
        for layer in cert.layers:
            for blob in layer.blobs:
                if blob.name == "proof":
                    assert blob.data == b"\xde\xad\xbe\xef"
                    found_blob = True
        assert found_blob, "Expected to find proof blob"

    def test_multiple_domains(self, signing_key: SigningKey) -> None:
        for domain in ["thermal", "euler3d", "ns_imex", "fluidelite"]:
            data = _build_tpc_certificate(domain=domain, signing_key=signing_key)
            cert = Certificate(data)
            assert cert.domain == domain
            assert cert.verify().valid

    def test_content_hash(self, valid_cert: bytes) -> None:
        cert = Certificate(valid_cert)
        expected = hashlib.sha256(valid_cert[:-SIGNATURE_SECTION_SIZE]).hexdigest()
        assert cert.content_hash == expected

    def test_repr(self, valid_cert: bytes) -> None:
        cert = Certificate(valid_cert)
        r = repr(cert)
        assert "Certificate(" in r
        assert "thermal" in r


# ═══════════════════════════════════════════════════════════════════════════
# Verifier Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestTPCVerifier:
    def test_local_only_verification(self, valid_cert: bytes) -> None:
        verifier = TPCVerifier()
        cert = Certificate(valid_cert)
        result = verifier.verify_local(cert)
        assert result.valid is True
        assert result.on_chain is None

    def test_verify_without_on_chain_config(self, valid_cert: bytes) -> None:
        verifier = TPCVerifier()
        cert = Certificate(valid_cert)
        result = verifier.verify(cert)
        assert result.valid is True
        assert result.on_chain is None

    def test_on_chain_without_config_raises(self, valid_cert: bytes) -> None:
        from fluidelite_verify.errors import OnChainVerificationFailed
        verifier = TPCVerifier()
        cert = Certificate(valid_cert)
        with pytest.raises(OnChainVerificationFailed, match="required"):
            verifier.verify_on_chain(cert)


# ═══════════════════════════════════════════════════════════════════════════
# Client Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestTPCClient:
    def test_init_with_api_key(self) -> None:
        client = TPCClient(ca_url="http://localhost:8444", api_key="test-key")
        assert "Bearer test-key" in client._session.headers.get("Authorization", "")

    def test_init_without_api_key(self) -> None:
        client = TPCClient(ca_url="http://localhost:8444")
        assert "Authorization" not in client._session.headers

    @patch("fluidelite_verify.client.requests.Session")
    def test_issue_success(self, mock_session_cls: MagicMock) -> None:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "certificate_id": "test-uuid",
            "content_hash": "abcd1234",
            "signer_pubkey": "00" * 32,
            "domain": "thermal",
            "size_bytes": 512,
            "issued_at": "2024-01-01T00:00:00Z",
            "on_chain_status": "pending",
        }
        mock_session.post.return_value = mock_response

        client = TPCClient.__new__(TPCClient)
        client._base_url = "http://localhost:8444"
        client._timeout = 30.0
        client._session = mock_session

        result = client.issue(
            domain="thermal",
            proof=b"\xde\xad",
            public_inputs=["0x01"],
        )

        assert result.certificate_id == "test-uuid"
        assert result.domain == "thermal"

    @patch("fluidelite_verify.client.requests.Session")
    def test_issue_failure(self, mock_session_cls: MagicMock) -> None:
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad request"
        mock_session.post.return_value = mock_response

        client = TPCClient.__new__(TPCClient)
        client._base_url = "http://localhost:8444"
        client._timeout = 30.0
        client._session = mock_session

        from fluidelite_verify.errors import TPCError
        with pytest.raises(TPCError, match="400"):
            client.issue(domain="thermal", proof=b"\x00")


# ═══════════════════════════════════════════════════════════════════════════
# Header Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCertificateHeader:
    def test_timestamp_conversion(self) -> None:
        header = CertificateHeader(
            magic=TPC_MAGIC,
            version=1,
            certificate_id="test-id",
            timestamp_ns=1700000000_000000000,
            solver_hash=b"\x00" * 32,
        )
        ts = header.timestamp
        assert ts.year == 2023
        assert ts.month == 11


class TestSignatureSection:
    def test_hex_properties(self) -> None:
        section = SignatureSection(
            pubkey=b"\x01" * 32,
            signature=b"\x02" * 64,
            content_hash=b"\x03" * 32,
        )
        assert section.pubkey_hex == "01" * 32
        assert section.signature_hex == "02" * 64
        assert section.content_hash_hex == "03" * 32


class TestOnChainResult:
    def test_status_name(self) -> None:
        assert OnChainResult(registered=True, status=0).status_name == "valid"
        assert OnChainResult(registered=True, status=1).status_name == "revoked"
        assert OnChainResult(registered=True, status=2).status_name == "superseded"
        assert OnChainResult(registered=True, status=99).status_name == "unknown"
        assert OnChainResult(registered=False).status_name == "unknown"
