"""
fluidelite-verify — TPC Certificate Verification SDK

Verify Trustless Physics Certificates locally and against on-chain state.

Usage:
    from fluidelite_verify import TPCVerifier, Certificate

    # Local (offline) verification
    cert = Certificate.from_file("proof.tpc")
    result = cert.verify()
    assert result.valid

    # On-chain verification
    verifier = TPCVerifier(rpc_url="https://sepolia.infura.io/v3/YOUR_KEY")
    on_chain = verifier.verify_on_chain(cert)
    assert on_chain.registered

    # CA-backed issuance + verification
    client = TPCClient(ca_url="https://ca.fluidelite.io", api_key="YOUR_KEY")
    cert = client.issue(domain="thermal", proof=proof_bytes, public_inputs=["0x01"])
    assert cert.verify().valid
"""

from fluidelite_verify.certificate import Certificate, CertificateHeader, Layer, SignatureSection
from fluidelite_verify.verifier import TPCVerifier, VerificationResult, OnChainResult
from fluidelite_verify.client import TPCClient
from fluidelite_verify.errors import (
    TPCError,
    InvalidCertificate,
    InvalidSignature,
    InvalidHash,
    CertificateNotFound,
    OnChainVerificationFailed,
)

__version__ = "1.0.0"
__all__ = [
    "Certificate",
    "CertificateHeader",
    "Layer",
    "SignatureSection",
    "TPCVerifier",
    "VerificationResult",
    "OnChainResult",
    "TPCClient",
    "TPCError",
    "InvalidCertificate",
    "InvalidSignature",
    "InvalidHash",
    "CertificateNotFound",
    "OnChainVerificationFailed",
]
