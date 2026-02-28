"""Trust certificate generation and verification.

Certificates are signed attestations that a computation was performed
correctly and that specific claims (conservation laws, bounds,
stability) were satisfied.

The signing key is server-side only and NEVER distributed.  Clients
verify certificates using the public key, which is available via
the ``/v1/contracts`` endpoint and shipped with the SDK.

Signature scheme: Ed25519 (via ``cryptography`` library if available,
with a HMAC-SHA256 fallback for environments without it).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

from .hasher import canonical_json, content_hash

logger = logging.getLogger(__name__)

# ── Key management ──────────────────────────────────────────────────

_SIGNING_KEY: bytes | None = None
_VERIFY_KEY: bytes | None = None
_USE_ED25519: bool = False


def _init_keys() -> None:
    """Initialize signing keys.

    Attempts Ed25519 first.  Falls back to HMAC-SHA256 with a
    server-local secret.
    """
    global _SIGNING_KEY, _VERIFY_KEY, _USE_ED25519

    # Check for pre-configured key material
    key_path = os.environ.get("ONTIC_SIGNING_KEY_PATH")
    if key_path and os.path.exists(key_path):
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )
            from cryptography.hazmat.primitives.serialization import (
                load_pem_private_key,
            )

            with open(key_path, "rb") as f:
                private_key = load_pem_private_key(f.read(), password=None)
            if isinstance(private_key, Ed25519PrivateKey):
                _SIGNING_KEY = private_key.private_bytes_raw()
                _VERIFY_KEY = private_key.public_key().public_bytes_raw()
                _USE_ED25519 = True
                logger.info("Certificate signing: Ed25519 (key from %s)", key_path)
                return
        except Exception as exc:
            logger.warning("Failed to load Ed25519 key: %s.  Falling back to HMAC.", exc)

    # Try generating ephemeral Ed25519 key
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        private_key = Ed25519PrivateKey.generate()
        _SIGNING_KEY = private_key.private_bytes_raw()
        _VERIFY_KEY = private_key.public_key().public_bytes_raw()
        _USE_ED25519 = True
        logger.info("Certificate signing: Ed25519 (ephemeral key)")
        return
    except ImportError:
        pass

    # Fallback: HMAC-SHA256 with server-local secret
    secret = os.environ.get("ONTIC_HMAC_SECRET", "").encode("utf-8")
    if not secret:
        secret = os.urandom(32)
        logger.warning(
            "Certificate signing: HMAC-SHA256 (random ephemeral secret).  "
            "Set ONTIC_SIGNING_KEY_PATH or ONTIC_HMAC_SECRET "
            "for persistent signatures."
        )
    _SIGNING_KEY = secret
    _VERIFY_KEY = secret  # HMAC uses same key for sign+verify
    _USE_ED25519 = False
    logger.info("Certificate signing: HMAC-SHA256")


def _ensure_keys() -> None:
    if _SIGNING_KEY is None:
        _init_keys()


def _sign(data: bytes) -> str:
    """Sign data and return a prefixed signature string."""
    _ensure_keys()
    assert _SIGNING_KEY is not None

    if _USE_ED25519:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        private_key = Ed25519PrivateKey.from_private_bytes(_SIGNING_KEY)
        sig = private_key.sign(data)
        return f"ed25519:{sig.hex()}"
    else:
        sig = hmac.new(_SIGNING_KEY, data, hashlib.sha256).hexdigest()
        return f"hmac-sha256:{sig}"


def verify_signature(data: bytes, signature: str) -> bool:
    """Verify a signature against the current key."""
    _ensure_keys()
    assert _VERIFY_KEY is not None

    if signature.startswith("ed25519:"):
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PublicKey,
            )

            sig_bytes = bytes.fromhex(signature[len("ed25519:"):])
            public_key = Ed25519PublicKey.from_public_bytes(_VERIFY_KEY)
            public_key.verify(sig_bytes, data)
            return True
        except Exception:
            return False

    elif signature.startswith("hmac-sha256:"):
        expected = hmac.new(_VERIFY_KEY, data, hashlib.sha256).hexdigest()
        return hmac.compare_digest(signature[len("hmac-sha256:"):], expected)

    return False


# ── Certificate generation ──────────────────────────────────────────


def issue_certificate(
    job_id: str,
    claims: list[dict[str, Any]],
    input_manifest_hash: str,
    result_hash: str,
    config_hash: str,
    runtime_version: str,
    seed: int | None = None,
    device_class: str = "cpu",
) -> dict[str, Any]:
    """Issue a signed trust certificate for a completed job.

    The certificate is self-contained: a verifier needs only the
    public key and the certificate JSON to check all claims.
    """
    cert_body = {
        "certificate_version": "1.0.0",
        "job_id": job_id,
        "issued_at": datetime.now(timezone.utc).isoformat(),
        "issuer": "physics_os-runtime",
        "claims": claims,
        "input_manifest_hash": input_manifest_hash,
        "result_hash": result_hash,
        "replay_metadata": {
            "runtime_version": runtime_version,
            "config_hash": config_hash,
            "seed": seed,
            "device_class": device_class,
        },
    }

    # Sign the canonical representation of the certificate body
    payload = canonical_json(cert_body)
    cert_body["signature"] = _sign(payload)

    return cert_body


def verify_certificate(certificate: dict[str, Any]) -> bool:
    """Verify a certificate's signature.

    Returns True if the signature is valid for the certificate body.
    """
    cert_copy = dict(certificate)
    signature = cert_copy.pop("signature", "")
    payload = canonical_json(cert_copy)
    return verify_signature(payload, signature)


def get_public_key_hex() -> str | None:
    """Return the public verification key as a hex string.

    Returns None if using HMAC (where the key cannot be shared).
    """
    _ensure_keys()
    if _USE_ED25519 and _VERIFY_KEY is not None:
        return _VERIFY_KEY.hex()
    return None
