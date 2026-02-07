"""
TPC Constants
=============

Magic bytes, version numbers, size limits, and cryptographic parameters
for the Trustless Physics Certificate format.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Wire format
# ─────────────────────────────────────────────────────────────────────────────
TPC_MAGIC = b"TPC\x01"
TPC_VERSION: int = 1
TPC_HEADER_SIZE: int = 64  # bytes

# ─────────────────────────────────────────────────────────────────────────────
# Size limits (DoS protection)
# ─────────────────────────────────────────────────────────────────────────────
MAX_CERTIFICATE_SIZE: int = 256 * 1024 * 1024  # 256 MB
MAX_PROOF_BYTES: int = 64 * 1024 * 1024         # 64 MB
MAX_LEAN_PROOFS: int = 1024
MAX_BENCHMARKS: int = 4096
MAX_THEOREM_NAME_LEN: int = 512
MAX_STRING_LEN: int = 65536

# ─────────────────────────────────────────────────────────────────────────────
# Cryptographic parameters
# ─────────────────────────────────────────────────────────────────────────────
HASH_ALGORITHM = "sha256"
SIGNATURE_ALGORITHM = "ed25519"
SIGNATURE_SIZE: int = 64        # ed25519 signature bytes
PUBLIC_KEY_SIZE: int = 32       # ed25519 public key bytes
HASH_SIZE: int = 32             # SHA-256 digest bytes

# ─────────────────────────────────────────────────────────────────────────────
# Proof system identifiers
# ─────────────────────────────────────────────────────────────────────────────
PROOF_SYSTEMS = frozenset({"plonk", "stark", "groth16", "halo2", "none"})

# ─────────────────────────────────────────────────────────────────────────────
# Solver identifiers
# ─────────────────────────────────────────────────────────────────────────────
KNOWN_SOLVERS = frozenset({
    "euler1d", "euler2d", "euler3d",
    "ns_imex", "ns3d",
    "vlasov6d", "vlasov_poisson",
    "heat1d", "heat3d",
    "wave1d", "wave3d",
    "structural_fem",
    "mhd3d",
    "custom",
})

# ─────────────────────────────────────────────────────────────────────────────
# Domain identifiers
# ─────────────────────────────────────────────────────────────────────────────
KNOWN_DOMAINS = frozenset({
    "cfd", "structural", "thermal", "multi-physics",
    "plasma", "electromagnetics", "acoustics",
    "biological", "chemical", "custom",
})

# ─────────────────────────────────────────────────────────────────────────────
# Layer A: formal proof system identifiers
# ─────────────────────────────────────────────────────────────────────────────
FORMAL_PROOF_SYSTEMS = frozenset({"lean4", "coq", "isabelle", "none"})

# ─────────────────────────────────────────────────────────────────────────────
# File extension
# ─────────────────────────────────────────────────────────────────────────────
TPC_EXTENSION = ".tpc"
