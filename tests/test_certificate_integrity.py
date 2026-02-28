"""G6 — Certificate Integrity adversarial test suite.

Tests T1–T12 from CERTIFICATE_TEST_MATRIX.md.
Covers: sign/verify round-trip, tamper detection, IP boundary,
key management, and determinism.
"""

from __future__ import annotations

import copy
import json
import os
import uuid

import pytest

from physics_os.core.certificates import (
    _init_keys,
    _SIGNING_KEY,
    _USE_ED25519,
    _VERIFY_KEY,
    get_public_key_hex,
    issue_certificate,
    verify_certificate,
    verify_signature,
)
from physics_os.core.evidence import generate_claims, generate_validation_report
from physics_os.core.hasher import canonical_json, content_hash


# ── Helpers ──────────────────────────────────────────────────────────


def _make_certificate(**overrides: object) -> dict:
    """Issue a certificate with sensible defaults."""
    defaults = {
        "job_id": str(uuid.uuid4()),
        "claims": [
            {
                "tag": "CONSERVATION",
                "claim": "energy preserved to 1.23e-08 relative error",
                "witness": {
                    "initial": 1.0,
                    "final": 0.9999999877,
                    "relative_error": 1.23e-08,
                    "threshold": 1e-4,
                },
                "satisfied": True,
            },
            {
                "tag": "STABILITY",
                "claim": "Simulation completed without numerical divergence",
                "witness": {
                    "wall_time_s": 0.34,
                    "time_steps": 100,
                    "completed": True,
                },
                "satisfied": True,
            },
            {
                "tag": "BOUND",
                "claim": "All field values bounded (max |value| = 1.23e+00)",
                "witness": {
                    "max_absolute_value": 1.23,
                    "threshold": 1e15,
                },
                "satisfied": True,
            },
        ],
        "input_manifest_hash": "sha256:abc123",
        "result_hash": "sha256:def456",
        "config_hash": "sha256:ghi789",
        "runtime_version": "3.1.0",
        "seed": None,
        "device_class": "cpu",
    }
    defaults.update(overrides)
    return issue_certificate(**defaults)


# ═══════════════════════════════════════════════════════════════════
# T1: Happy path — round-trip sign + verify
# ═══════════════════════════════════════════════════════════════════


class TestT1RoundTrip:
    def test_issue_and_verify(self) -> None:
        cert = _make_certificate()
        assert "signature" in cert
        assert verify_certificate(cert) is True

    def test_certificate_has_required_fields(self) -> None:
        cert = _make_certificate()
        required = {
            "certificate_version",
            "job_id",
            "issued_at",
            "issuer",
            "claims",
            "input_manifest_hash",
            "result_hash",
            "replay_metadata",
            "signature",
        }
        assert required.issubset(cert.keys())

    def test_certificate_version(self) -> None:
        cert = _make_certificate()
        assert cert["certificate_version"] == "1.0.0"

    def test_issuer(self) -> None:
        cert = _make_certificate()
        assert cert["issuer"] == "physics_os-runtime"


# ═══════════════════════════════════════════════════════════════════
# T2: Tampered claim — signature invalidated
# ═══════════════════════════════════════════════════════════════════


class TestT2ClaimTampering:
    def test_flip_satisfied_flag(self) -> None:
        cert = _make_certificate()
        assert verify_certificate(cert) is True

        tampered = copy.deepcopy(cert)
        tampered["claims"][0]["satisfied"] = False
        assert verify_certificate(tampered) is False

    def test_modify_claim_text(self) -> None:
        cert = _make_certificate()
        tampered = copy.deepcopy(cert)
        tampered["claims"][0]["claim"] = "FORGED: energy not preserved"
        assert verify_certificate(tampered) is False

    def test_modify_witness_value(self) -> None:
        cert = _make_certificate()
        tampered = copy.deepcopy(cert)
        tampered["claims"][0]["witness"]["relative_error"] = 0.5
        assert verify_certificate(tampered) is False

    def test_add_extra_claim(self) -> None:
        cert = _make_certificate()
        tampered = copy.deepcopy(cert)
        tampered["claims"].append({
            "tag": "FORGED",
            "claim": "injected claim",
            "witness": {},
            "satisfied": True,
        })
        assert verify_certificate(tampered) is False

    def test_remove_claim(self) -> None:
        cert = _make_certificate()
        tampered = copy.deepcopy(cert)
        tampered["claims"].pop()
        assert verify_certificate(tampered) is False


# ═══════════════════════════════════════════════════════════════════
# T3: Tampered result hash — signature invalidated
# ═══════════════════════════════════════════════════════════════════


class TestT3ResultHashTampering:
    def test_change_result_hash(self) -> None:
        cert = _make_certificate()
        tampered = copy.deepcopy(cert)
        tampered["result_hash"] = "sha256:FORGED000000"
        assert verify_certificate(tampered) is False

    def test_empty_result_hash(self) -> None:
        cert = _make_certificate()
        tampered = copy.deepcopy(cert)
        tampered["result_hash"] = ""
        assert verify_certificate(tampered) is False


# ═══════════════════════════════════════════════════════════════════
# T4: Tampered job ID — signature invalidated
# ═══════════════════════════════════════════════════════════════════


class TestT4JobIdTampering:
    def test_change_job_id(self) -> None:
        cert = _make_certificate()
        tampered = copy.deepcopy(cert)
        tampered["job_id"] = str(uuid.uuid4())
        assert verify_certificate(tampered) is False


# ═══════════════════════════════════════════════════════════════════
# T5: Replay metadata tampering — signature invalidated
# ═══════════════════════════════════════════════════════════════════


class TestT5ReplayMetadataTampering:
    def test_change_runtime_version(self) -> None:
        cert = _make_certificate()
        tampered = copy.deepcopy(cert)
        tampered["replay_metadata"]["runtime_version"] = "9.9.9"
        assert verify_certificate(tampered) is False

    def test_change_device_class(self) -> None:
        cert = _make_certificate()
        tampered = copy.deepcopy(cert)
        tampered["replay_metadata"]["device_class"] = "tpu"
        assert verify_certificate(tampered) is False

    def test_change_config_hash(self) -> None:
        cert = _make_certificate()
        tampered = copy.deepcopy(cert)
        tampered["replay_metadata"]["config_hash"] = "sha256:INJECTED"
        assert verify_certificate(tampered) is False

    def test_change_seed(self) -> None:
        cert = _make_certificate()
        tampered = copy.deepcopy(cert)
        tampered["replay_metadata"]["seed"] = 42
        assert verify_certificate(tampered) is False


# ═══════════════════════════════════════════════════════════════════
# T6: Missing signature — verification fails
# ═══════════════════════════════════════════════════════════════════


class TestT6MissingSignature:
    def test_no_signature_key(self) -> None:
        cert = _make_certificate()
        del cert["signature"]
        assert verify_certificate(cert) is False

    def test_empty_signature(self) -> None:
        cert = _make_certificate()
        cert["signature"] = ""
        assert verify_certificate(cert) is False

    def test_garbage_signature(self) -> None:
        cert = _make_certificate()
        cert["signature"] = "not-a-real-signature"
        assert verify_certificate(cert) is False


# ═══════════════════════════════════════════════════════════════════
# T7: Wrong key — cross-key verification fails
# ═══════════════════════════════════════════════════════════════════


class TestT7WrongKey:
    def test_different_key_rejects(self) -> None:
        """Sign with current key, re-init keys, try to verify with new key."""
        import physics_os.core.certificates as cert_mod

        cert = _make_certificate()
        assert verify_certificate(cert) is True

        # Force key re-generation
        old_signing = cert_mod._SIGNING_KEY
        old_verify = cert_mod._VERIFY_KEY

        cert_mod._SIGNING_KEY = None
        cert_mod._VERIFY_KEY = None
        cert_mod._init_keys()

        # With new keys, old cert should fail (unless by astronomical chance
        # the same ephemeral key was generated)
        if cert_mod._USE_ED25519:
            result = verify_certificate(cert)
            assert result is False, "Certificate verified with wrong key"

        # Restore original keys for remaining tests
        cert_mod._SIGNING_KEY = old_signing
        cert_mod._VERIFY_KEY = old_verify


# ═══════════════════════════════════════════════════════════════════
# T8: HMAC fallback — round-trip
# ═══════════════════════════════════════════════════════════════════


class TestT8HMACFallback:
    def test_hmac_round_trip(self) -> None:
        """Force HMAC mode and verify round-trip."""
        import physics_os.core.certificates as cert_mod

        # Save state
        saved_key = cert_mod._SIGNING_KEY
        saved_verify = cert_mod._VERIFY_KEY
        saved_ed = cert_mod._USE_ED25519

        try:
            # Force HMAC mode
            secret = b"test-secret-key-for-hmac-testing"
            cert_mod._SIGNING_KEY = secret
            cert_mod._VERIFY_KEY = secret
            cert_mod._USE_ED25519 = False

            cert = _make_certificate()
            assert cert["signature"].startswith("hmac-sha256:")
            assert verify_certificate(cert) is True

            # Tamper → should fail
            tampered = copy.deepcopy(cert)
            tampered["claims"][0]["satisfied"] = False
            assert verify_certificate(tampered) is False
        finally:
            # Restore
            cert_mod._SIGNING_KEY = saved_key
            cert_mod._VERIFY_KEY = saved_verify
            cert_mod._USE_ED25519 = saved_ed


# ═══════════════════════════════════════════════════════════════════
# T9: Canonical JSON determinism
# ═══════════════════════════════════════════════════════════════════


class TestT9CanonicalDeterminism:
    def test_key_order_independent(self) -> None:
        d1 = {"z": 1, "a": 2, "m": 3}
        d2 = {"a": 2, "m": 3, "z": 1}
        assert canonical_json(d1) == canonical_json(d2)

    def test_nested_key_order(self) -> None:
        d1 = {"outer": {"z": 1, "a": 2}}
        d2 = {"outer": {"a": 2, "z": 1}}
        assert canonical_json(d1) == canonical_json(d2)

    def test_repeated_calls_identical(self) -> None:
        data = {"key": "value", "number": 3.14, "list": [1, 2, 3]}
        results = [canonical_json(data) for _ in range(100)]
        assert len(set(results)) == 1

    def test_content_hash_stable(self) -> None:
        data = {"key": "value", "another": [1, 2, 3]}
        hashes = [content_hash(data) for _ in range(100)]
        assert len(set(hashes)) == 1


# ═══════════════════════════════════════════════════════════════════
# T10: Certificate contains no forbidden fields
# ═══════════════════════════════════════════════════════════════════


class TestT10NoForbiddenFields:
    FORBIDDEN_KEYWORDS = {
        "bond_dim",
        "chi_max",
        "chi_mean",
        "chi_final",
        "compression_ratio",
        "singular_value",
        "svd_spectrum",
        "tt_core",
        "core_shape",
        "rank_evolution",
        "rank_history",
        "saturation_rate",
        "scaling_class",
        "scaling_classification",
        "opcode",
        "instruction_count",
        "register_count",
        "ir_ops",
        "ir_instructions",
        "signing_key",
        "hmac_secret",
        "private_key",
    }

    def _flatten_keys(self, obj: object, prefix: str = "") -> set[str]:
        keys: set[str] = set()
        if isinstance(obj, dict):
            for k, v in obj.items():
                full_key = f"{prefix}.{k}" if prefix else k
                keys.add(k.lower())
                keys |= self._flatten_keys(v, full_key)
        elif isinstance(obj, list):
            for item in obj:
                keys |= self._flatten_keys(item, prefix)
        return keys

    def test_no_forbidden_keys_in_certificate(self) -> None:
        cert = _make_certificate()
        all_keys = self._flatten_keys(cert)
        found_forbidden = all_keys & self.FORBIDDEN_KEYWORDS
        assert not found_forbidden, f"Forbidden keys found in certificate: {found_forbidden}"

    def test_no_forbidden_in_serialized_cert(self) -> None:
        cert = _make_certificate()
        cert_json = json.dumps(cert).lower()
        for keyword in self.FORBIDDEN_KEYWORDS:
            assert keyword not in cert_json, f"Forbidden keyword '{keyword}' in serialized cert"


# ═══════════════════════════════════════════════════════════════════
# T11: Claim tags are from allowlist
# ═══════════════════════════════════════════════════════════════════


class TestT11ClaimTagAllowlist:
    ALLOWED_TAGS = {"CONSERVATION", "STABILITY", "BOUND"}

    def test_all_claim_tags_in_allowlist(self) -> None:
        cert = _make_certificate()
        for claim in cert["claims"]:
            assert claim["tag"] in self.ALLOWED_TAGS, (
                f"Unregistered claim tag: {claim['tag']}"
            )

    def test_generate_claims_tags_subset(self) -> None:
        """Verify generate_claims() only produces allowed tags."""
        # Minimal sanitized result
        result = {
            "conservation": {
                "quantity": "L2_norm",
                "initial_value": 1.0,
                "final_value": 0.9999,
                "relative_error": 1e-5,
                "status": "conserved",
            },
            "performance": {
                "wall_time_s": 0.5,
                "time_steps": 100,
            },
            "fields": {
                "u": {"values": [0.1, 0.2, 0.3]},
            },
        }
        claims = generate_claims(result, "burgers")
        for claim in claims:
            assert claim["tag"] in self.ALLOWED_TAGS, (
                f"generate_claims produced unregistered tag: {claim['tag']}"
            )


# ═══════════════════════════════════════════════════════════════════
# T12: Issued-at timestamp is UTC
# ═══════════════════════════════════════════════════════════════════


class TestT12TimestampUTC:
    def test_issued_at_is_utc(self) -> None:
        cert = _make_certificate()
        issued_at = cert["issued_at"]
        # Python's datetime.isoformat() with timezone.utc produces +00:00
        assert issued_at.endswith("+00:00") or issued_at.endswith("Z"), (
            f"issued_at is not UTC: {issued_at}"
        )


# ═══════════════════════════════════════════════════════════════════
# G6 gate-specific aggregate tests (map to LAUNCH_GATE_MATRIX.json)
# ═══════════════════════════════════════════════════════════════════


class TestG6_1_ClaimTamperingDetected:
    """G6.1: Any modification to claims breaks the certificate."""

    def test_satisfied_flip(self) -> None:
        cert = _make_certificate()
        tampered = copy.deepcopy(cert)
        tampered["claims"][0]["satisfied"] = not tampered["claims"][0]["satisfied"]
        assert verify_certificate(tampered) is False


class TestG6_2_EnvelopeTamperingDetected:
    """G6.2: Any modification to the envelope breaks the certificate."""

    def test_issued_at_change(self) -> None:
        cert = _make_certificate()
        tampered = copy.deepcopy(cert)
        tampered["issued_at"] = "2000-01-01T00:00:00+00:00"
        assert verify_certificate(tampered) is False

    def test_issuer_change(self) -> None:
        cert = _make_certificate()
        tampered = copy.deepcopy(cert)
        tampered["issuer"] = "evil-runtime"
        assert verify_certificate(tampered) is False

    def test_version_change(self) -> None:
        cert = _make_certificate()
        tampered = copy.deepcopy(cert)
        tampered["certificate_version"] = "9.9.9"
        assert verify_certificate(tampered) is False


class TestG6_3_SignatureCorruptionDetected:
    """G6.3: Corrupted signature bytes are rejected."""

    def test_truncated_signature(self) -> None:
        cert = _make_certificate()
        cert["signature"] = cert["signature"][:20]
        assert verify_certificate(cert) is False

    def test_flipped_byte(self) -> None:
        cert = _make_certificate()
        sig = cert["signature"]
        # Flip a character near the end of the hex signature
        prefix, hex_part = sig.split(":", 1)
        if hex_part:
            flipped = hex_part[:-1] + ("0" if hex_part[-1] != "0" else "1")
            cert["signature"] = f"{prefix}:{flipped}"
        assert verify_certificate(cert) is False


class TestG6_4_WrongKeyRejected:
    """G6.4: Certificate signed with key A fails verify with key B."""

    def test_wrong_key(self) -> None:
        # Covered by TestT7WrongKey — re-assert here for gate clarity
        import physics_os.core.certificates as cert_mod

        cert = _make_certificate()
        old_signing = cert_mod._SIGNING_KEY
        old_verify = cert_mod._VERIFY_KEY

        cert_mod._SIGNING_KEY = None
        cert_mod._VERIFY_KEY = None
        cert_mod._init_keys()

        if cert_mod._USE_ED25519:
            assert verify_certificate(cert) is False

        cert_mod._SIGNING_KEY = old_signing
        cert_mod._VERIFY_KEY = old_verify


class TestG6_5_ReplayDifferentPayload:
    """G6.5: Signature from cert A does not verify cert B."""

    def test_replay_signature_on_different_body(self) -> None:
        cert_a = _make_certificate(result_hash="sha256:original_result_aaa")
        cert_b = _make_certificate(result_hash="sha256:different_result_bbb")

        # Take A's signature and attach to B's body
        franken = copy.deepcopy(cert_b)
        franken["signature"] = cert_a["signature"]
        assert verify_certificate(franken) is False


class TestG6_6_InvalidValidationRejected:
    """G6.6: Certificate from result with failed validation
    has claims with satisfied=False."""

    def test_failed_validation_claims(self) -> None:
        # A result that will fail validation (null values, bad conservation)
        bad_result = {
            "conservation": {
                "quantity": "energy",
                "initial_value": 1.0,
                "final_value": 100.0,
                "relative_error": 99.0,
                "status": "drift",
            },
            "performance": {"wall_time_s": 0.5, "time_steps": 100},
            "fields": {"u": {"values": [None, None, None]}},
        }
        claims = generate_claims(bad_result, "burgers")
        conservation_claim = next(c for c in claims if c["tag"] == "CONSERVATION")
        assert conservation_claim["satisfied"] is False

        validation = generate_validation_report(bad_result, "burgers")
        assert validation["valid"] is False
