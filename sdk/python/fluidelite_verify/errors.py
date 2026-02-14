"""Custom exceptions for TPC certificate verification."""


class TPCError(Exception):
    """Base exception for all TPC verification errors."""
    pass


class InvalidCertificate(TPCError):
    """Certificate format is invalid (wrong magic, truncated, bad structure)."""
    pass


class InvalidSignature(TPCError):
    """Ed25519 signature verification failed."""
    pass


class InvalidHash(TPCError):
    """SHA-256 content hash mismatch."""
    pass


class CertificateNotFound(TPCError):
    """Certificate not found (local or on CA)."""
    pass


class OnChainVerificationFailed(TPCError):
    """On-chain certificate lookup or verification failed."""
    pass
