#!/usr/bin/env python3
"""
PQC Manifest Signer
===================

Signs validation manifests using post-quantum cryptography (Dilithium2).

Falls back to ECDSA if PQC library not available.

Usage:
    python sign_manifest.py --manifest manifest.json --algorithm dilithium2 --output signed.json

Constitution Compliance: Article IV.1 (Verification), Phase 3 Automation
Tags: [V&V] [PROVENANCE] [CRYPTOGRAPHY] [PQC]
"""

import argparse
import base64
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

# Try to import PQC library
PQC_AVAILABLE = False
try:
    import oqs

    PQC_AVAILABLE = True
except ImportError:
    pass

# Try to import cryptography for ECDSA fallback
ECDSA_AVAILABLE = False
try:
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    ECDSA_AVAILABLE = True
except ImportError:
    pass


class DilithiumSigner:
    """Post-quantum Dilithium2 signer."""

    def __init__(self, algorithm: str = "Dilithium2"):
        if not PQC_AVAILABLE:
            raise RuntimeError(
                "liboqs-python not available. Install with: pip install liboqs-python"
            )

        self.algorithm = algorithm
        self.signer = oqs.Signature(algorithm)
        self.public_key = None
        self.secret_key = None

    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """Generate a new keypair."""
        self.public_key = self.signer.generate_keypair()
        self.secret_key = self.signer.export_secret_key()
        return self.public_key, self.secret_key

    def load_keypair(self, public_key: bytes, secret_key: bytes):
        """Load existing keypair."""
        self.public_key = public_key
        # liboqs doesn't support importing secret keys directly in some versions
        # This is a limitation we work around by regenerating
        pass

    def sign(self, message: bytes) -> bytes:
        """Sign a message."""
        if self.secret_key is None:
            self.generate_keypair()
        return self.signer.sign(message)

    def verify(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify a signature."""
        verifier = oqs.Signature(self.algorithm)
        return verifier.verify(message, signature, public_key)


class ECDSASigner:
    """ECDSA fallback signer (P-256)."""

    def __init__(self):
        if not ECDSA_AVAILABLE:
            raise RuntimeError(
                "cryptography library not available. Install with: pip install cryptography"
            )

        self.private_key = None
        self.public_key = None

    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """Generate a new keypair."""
        self.private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
        self.public_key = self.private_key.public_key()

        public_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        private_bytes = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        return public_bytes, private_bytes

    def sign(self, message: bytes) -> bytes:
        """Sign a message."""
        if self.private_key is None:
            self.generate_keypair()
        return self.private_key.sign(message, ec.ECDSA(hashes.SHA256()))

    def verify(self, message: bytes, signature: bytes, public_key_pem: bytes) -> bool:
        """Verify a signature."""
        public_key = serialization.load_pem_public_key(
            public_key_pem, default_backend()
        )
        try:
            public_key.verify(signature, message, ec.ECDSA(hashes.SHA256()))
            return True
        except Exception:
            return False


class FallbackSigner:
    """Hash-based fallback when no crypto available."""

    def __init__(self):
        self.secret = os.urandom(32)

    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """Generate a 'keypair' (just hashes)."""
        public = hashlib.sha256(self.secret).digest()
        return public, self.secret

    def sign(self, message: bytes) -> bytes:
        """Create HMAC-like signature."""
        import hmac

        return hmac.new(self.secret, message, hashlib.sha256).digest()

    def verify(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify (limited - needs secret)."""
        return False  # Can't verify without secret


def get_signer(algorithm: str):
    """Get appropriate signer based on algorithm and availability."""
    if algorithm.lower() in ("dilithium2", "dilithium3", "dilithium5"):
        if PQC_AVAILABLE:
            return DilithiumSigner(algorithm.capitalize())
        else:
            print(f"Warning: {algorithm} requested but liboqs not available")

    if algorithm.lower() == "ecdsa" or (not PQC_AVAILABLE and ECDSA_AVAILABLE):
        print("Using ECDSA (P-256) fallback")
        return ECDSASigner()

    print("Warning: No crypto libraries available, using hash-based fallback")
    return FallbackSigner()


def sign_manifest(manifest_path: Path, algorithm: str, output_path: Path) -> dict:
    """Sign a validation manifest."""

    # Load manifest
    manifest = json.loads(manifest_path.read_text())

    # Get signer
    signer = get_signer(algorithm)

    # Generate keypair
    public_key, secret_key = signer.generate_keypair()

    # Create message to sign (manifest hash)
    manifest_hash = manifest.get("manifest_hash", "")
    if not manifest_hash:
        # Compute hash if not present
        manifest_for_hash = {k: v for k, v in manifest.items() if k != "signature"}
        manifest_json = json.dumps(manifest_for_hash, sort_keys=True)
        manifest_hash = hashlib.sha256(manifest_json.encode()).hexdigest()

    message = manifest_hash.encode()

    # Sign
    signature = signer.sign(message)

    # Determine algorithm name
    if isinstance(signer, DilithiumSigner):
        algo_name = signer.algorithm
    elif isinstance(signer, ECDSASigner):
        algo_name = "ECDSA-P256-SHA256"
    else:
        algo_name = "HMAC-SHA256-FALLBACK"

    # Add signature to manifest
    manifest["signature"] = {
        "algorithm": algo_name,
        "public_key": base64.b64encode(public_key).decode(),
        "signature": base64.b64encode(signature).decode(),
        "signed_at": datetime.utcnow().isoformat() + "Z",
        "signer": "Ontic VOntic V&VV CI/CD",
    }

    return manifest


def verify_manifest(manifest: dict) -> bool:
    """Verify a signed manifest."""
    sig_info = manifest.get("signature")
    if not sig_info:
        print("No signature found")
        return False

    algorithm = sig_info.get("algorithm", "")
    public_key = base64.b64decode(sig_info["public_key"])
    signature = base64.b64decode(sig_info["signature"])

    # Get message
    manifest_hash = manifest.get("manifest_hash", "")
    message = manifest_hash.encode()

    # Get verifier
    if "Dilithium" in algorithm:
        if not PQC_AVAILABLE:
            print("Cannot verify Dilithium signature: liboqs not available")
            return False
        signer = DilithiumSigner(algorithm)
        return signer.verify(message, signature, public_key)
    elif "ECDSA" in algorithm:
        if not ECDSA_AVAILABLE:
            print("Cannot verify ECDSA signature: cryptography not available")
            return False
        signer = ECDSASigner()
        return signer.verify(message, signature, public_key)
    else:
        print(f"Unknown algorithm: {algorithm}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Sign Validation Manifest")
    parser.add_argument(
        "--manifest", type=Path, required=True, help="Input manifest file"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="dilithium2",
        choices=["dilithium2", "dilithium3", "dilithium5", "ecdsa"],
        help="Signing algorithm",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output signed manifest"
    )
    parser.add_argument("--verify", action="store_true", help="Verify instead of sign")
    args = parser.parse_args()

    if args.verify:
        print("Verifying manifest signature...")
        manifest = json.loads(args.manifest.read_text())
        valid = verify_manifest(manifest)
        if valid:
            print("✅ Signature valid")
            sys.exit(0)
        else:
            print("❌ Signature invalid")
            sys.exit(1)

    print("Signing Validation Manifest")
    print("=" * 40)
    print(f"Algorithm: {args.algorithm}")
    print(f"Input: {args.manifest}")

    signed_manifest = sign_manifest(args.manifest, args.algorithm, args.output)

    # Write signed manifest
    args.output.write_text(json.dumps(signed_manifest, indent=2))

    print(f"\n✅ Manifest signed successfully")
    print(f"Output: {args.output}")
    print(f"Algorithm: {signed_manifest['signature']['algorithm']}")
    print(f"Signed at: {signed_manifest['signature']['signed_at']}")


if __name__ == "__main__":
    main()
