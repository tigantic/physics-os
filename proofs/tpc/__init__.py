"""
Trustless Physics Certificate (TPC)
====================================

Cryptographic verification certificates for physics simulations.
Three-layer verification: Mathematical Truth × Computational Integrity × Physical Fidelity.

Architecture:
    tpc.format       — .tpc binary serializer / deserializer
    tpc.generator    — Certificate builder (bundles Lean + ZK + attestation)
    tpc.constants    — Magic bytes, version, limits

Usage:
    from tpc import CertificateGenerator, TPCFile, verify_certificate

    # Generate
    cert = CertificateGenerator(solver="euler3d", ...)
    cert.save("simulation.tpc")

    # Verify
    result = verify_certificate("simulation.tpc")
    assert result.valid

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from tpc.format import (
    LayerA,
    LayerB,
    LayerC,
    Metadata,
    TPCFile,
    TPCHeader,
    TPCSignature,
    VerificationReport,
    verify_certificate,
)
from tpc.generator import CertificateGenerator

__version__ = "1.0.0"

__all__ = [
    # Format
    "TPCFile",
    "TPCHeader",
    "LayerA",
    "LayerB",
    "LayerC",
    "Metadata",
    "TPCSignature",
    "VerificationReport",
    "verify_certificate",
    # Generator
    "CertificateGenerator",
]
