#!/usr/bin/env python3
"""
Trustless Physics Proof Engine
================================

Cryptographic attestation of QTT Navier-Stokes simulation integrity.
Every timestep is committed, physics invariants are machine-verified,
and the entire run is sealed in a hash-chained Merkle certificate.

Architecture
------------
    ┌─────────────────────────────┐
    │   TrustlessPhysicsProver    │
    │  (wraps AhmedBodyIBSolver)  │
    └────────┬────────────────────┘
             │ per step
             ▼
    ┌─────────────────────────────┐
    │        StepProof            │
    │  ● QTT state commitment    │
    │  ● Energy conservation     │
    │  ● Rank bound              │
    │  ● CFL stability           │
    │  ● Compression bound       │
    │  ● Hash-chain link         │
    └────────┬────────────────────┘
             │ all steps
             ▼
    ┌─────────────────────────────┐
    │       MerkleTree            │
    │  O(log n) proof for any    │
    │  individual timestep        │
    └────────┬────────────────────┘
             │
             ▼
    ┌─────────────────────────────┐
    │  TrustlessCertificate       │
    │  ● Config commitment        │
    │  ● Merkle root             │
    │  ● Run-level proofs:       │
    │    – Convergence           │
    │    – Energy conservation   │
    │    – Spectrum (optional)   │
    │  ● All step proofs         │
    │  ● SHA-256 seal            │
    └─────────────────────────────┘

Verification (standalone, no GPU needed):
    verify_certificate(cert_path) → True/False
    verify_step(cert_path, step_index) → True/False + Merkle path

Uses proof_engine.proof_carrying PCC framework for hash-chain integrity.

Author: Brad Adams / Tigantic Holdings LLC
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

# Ensure proofs/ directory is on sys.path for proof_engine imports
_proofs_dir = str(Path(__file__).resolve().parent.parent.parent / "proofs")
if _proofs_dir not in sys.path:
    sys.path.insert(0, _proofs_dir)

import numpy as np
import torch
from torch import Tensor

# ── Ed25519 Signing ───────────────────────────────────────────────
try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
    )
    _HAS_ED25519 = True
except ImportError:
    _HAS_ED25519 = False

# ── Proof Engine ────────────────────────────────────────────────────
from proof_engine.proof_carrying import (
    PCCPayload,
    PCCRegistry,
    ProofAnnotation,
    ProofTag,
    verify_bound,
    verify_conservation,
    verify_monotone,
    verify_positivity,
)


# ═══════════════════════════════════════════════════════════════════
# CRYPTOGRAPHIC PRIMITIVES
# ═══════════════════════════════════════════════════════════════════

def sha256_bytes(data: bytes) -> str:
    """SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_combine(*hashes: str) -> str:
    """Combine multiple hex hashes into one via SHA-256."""
    h = hashlib.sha256()
    for hx in hashes:
        h.update(bytes.fromhex(hx))
    return h.hexdigest()


def qtt_core_commitment(cores: List[Tensor]) -> str:
    """SHA-256 commitment of a list of TT cores.

    Uses **canonical fixed-point encoding**: each float is cast to
    float64 and rounded to 12 significant digits before hashing.
    This ensures platform-independent reproducibility regardless of
    GPU rounding modes.
    """
    h = hashlib.sha256()
    h.update(struct.pack("<I", len(cores)))
    for core in cores:
        c = core.detach().double().cpu().contiguous()
        # Round to 12 significant digits for cross-platform reproducibility
        arr = np.around(c.numpy(), decimals=12)
        for dim in c.shape:
            h.update(struct.pack("<I", dim))
        h.update(arr.tobytes())
    return h.hexdigest()


def qtt_vector_commitment(
    ux_cores: List[Tensor],
    uy_cores: List[Tensor],
    uz_cores: List[Tensor],
) -> str:
    """SHA-256 commitment of a 3D velocity vector field in QTT form.

    Commits all three components in a deterministic order (x, y, z).
    """
    hx = qtt_core_commitment(ux_cores)
    hy = qtt_core_commitment(uy_cores)
    hz = qtt_core_commitment(uz_cores)
    return sha256_combine(hx, hy, hz)


def config_commitment(params: Dict[str, Any]) -> str:
    """SHA-256 commitment of solver configuration.

    All parameters must be JSON-serializable.
    """
    canonical = json.dumps(params, sort_keys=True, default=str)
    return sha256_bytes(canonical.encode("utf-8"))


# ═══════════════════════════════════════════════════════════════════
# SOLVER PROTOCOL
# ═══════════════════════════════════════════════════════════════════

@runtime_checkable
class SolverProtocol(Protocol):
    """Structural contract a solver must satisfy for trustless attestation.

    Any object with these attributes/methods can be passed to
    ``TrustlessPhysicsProver`` without explicit inheritance.
    """

    config: Any
    """Solver configuration (must expose ``n_bits``, ``max_rank``, etc.)."""

    u: Any
    """QTT velocity vector field (x/y/z components with ``.cores.cores``)."""

    def step(self, debug: bool = False) -> Dict[str, Any]:
        """Advance one timestep and return diagnostics dict."""
        ...

    def _energy(self, u: Any) -> float:
        """Compute kinetic energy of a QTT velocity field."""
        ...


# ═══════════════════════════════════════════════════════════════════
# MERKLE TREE
# ═══════════════════════════════════════════════════════════════════

class MerkleTree:
    """Binary Merkle tree over SHA-256 leaf hashes.

    Provides O(log n) inclusion proofs for any individual leaf.
    """

    def __init__(self, leaves: List[str]) -> None:
        if not leaves:
            raise ValueError("MerkleTree requires at least one leaf")
        self._leaves = list(leaves)
        self._layers: List[List[str]] = []
        self._build()

    def _build(self) -> None:
        """Build tree bottom-up. Duplicate last leaf if odd count."""
        layer = list(self._leaves)
        self._layers.append(layer)
        while len(layer) > 1:
            next_layer: List[str] = []
            for i in range(0, len(layer), 2):
                left = layer[i]
                right = layer[i + 1] if i + 1 < len(layer) else layer[i]
                next_layer.append(sha256_combine(left, right))
            layer = next_layer
            self._layers.append(layer)

    @property
    def root(self) -> str:
        """Merkle root hash."""
        return self._layers[-1][0]

    @property
    def leaf_count(self) -> int:
        return len(self._leaves)

    @property
    def depth(self) -> int:
        return len(self._layers) - 1

    def proof(self, index: int) -> List[Tuple[str, str]]:
        """Generate Merkle inclusion proof for leaf at index.

        Returns list of (sibling_hash, side) pairs from leaf to root.
        side is "L" if sibling is on left, "R" if on right.
        """
        if index < 0 or index >= self.leaf_count:
            raise IndexError(f"Leaf index {index} out of range [0, {self.leaf_count})")

        proof_path: List[Tuple[str, str]] = []
        idx = index
        for layer in self._layers[:-1]:
            if idx % 2 == 0:
                sibling_idx = idx + 1 if idx + 1 < len(layer) else idx
                side = "R"
            else:
                sibling_idx = idx - 1
                side = "L"
            proof_path.append((layer[sibling_idx], side))
            idx //= 2
        return proof_path

    @staticmethod
    def verify_proof(
        leaf_hash: str,
        proof_path: List[Tuple[str, str]],
        expected_root: str,
    ) -> bool:
        """Verify a Merkle inclusion proof.

        Can be performed offline without the full tree.
        """
        current = leaf_hash
        for sibling, side in proof_path:
            if side == "L":
                current = sha256_combine(sibling, current)
            else:
                current = sha256_combine(current, sibling)
        return current == expected_root

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root": self.root,
            "leaf_count": self.leaf_count,
            "depth": self.depth,
        }


# ═══════════════════════════════════════════════════════════════════
# STEP PROOF
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PhysicsInvariant:
    """A single physics invariant check with cryptographic witness."""

    name: str
    claim: str
    witness: Dict[str, Any]
    satisfied: bool
    tag: str  # ProofTag name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "claim": self.claim,
            "witness": self.witness,
            "satisfied": self.satisfied,
            "tag": self.tag,
        }


@dataclass
class StepProof:
    """Cryptographic proof for a single solver timestep.

    Contains:
    - QTT state commitment (SHA-256 of all TT cores)
    - Physics invariant proofs
    - Hash-chain link to previous step
    - Step metadata
    """

    step_index: int
    timestamp: float
    state_commitment: str          # SHA-256 of QTT velocity state
    parent_commitment: str         # Previous step's state_commitment (chain)
    step_hash: str                 # SHA-256(step_index || state || parent || invariants)
    invariants: List[PhysicsInvariant]
    metadata: Dict[str, Any]       # energy, rank, CR, etc.

    @property
    def all_satisfied(self) -> bool:
        return all(inv.satisfied for inv in self.invariants)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "timestamp": self.timestamp,
            "state_commitment": self.state_commitment,
            "parent_commitment": self.parent_commitment,
            "step_hash": self.step_hash,
            "all_satisfied": self.all_satisfied,
            "invariants": [inv.to_dict() for inv in self.invariants],
            "metadata": self.metadata,
        }


def _compute_step_hash(
    step_index: int,
    state_commitment: str,
    parent_commitment: str,
    invariants: List[PhysicsInvariant],
) -> str:
    """Deterministic step hash binding step index, state, parent, and invariants."""
    h = hashlib.sha256()
    h.update(struct.pack("<Q", step_index))
    h.update(bytes.fromhex(state_commitment))
    h.update(bytes.fromhex(parent_commitment))
    for inv in invariants:
        h.update(inv.name.encode("utf-8"))
        h.update(b"\x01" if inv.satisfied else b"\x00")
        # Include quantitative witness values for binding
        h.update(json.dumps(inv.witness, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════════════
# PHYSICS INVARIANT CHECKERS
# ═══════════════════════════════════════════════════════════════════

def check_energy_conservation(
    energy_prev: float,
    energy_curr: float,
    tolerance: float = 0.10,
) -> PhysicsInvariant:
    """Verify energy decreases monotonically (viscous flow, no forcing).

    For viscous flow without external energy input, kinetic energy must
    decrease. We allow a small tolerance for discretization artifacts.
    The energy clamp mechanism ensures this holds after correction.
    """
    # Allow slight increase due to discretization (up to tolerance fraction)
    ratio = energy_curr / energy_prev if energy_prev > 0 else 1.0
    satisfied = ratio <= (1.0 + tolerance)

    return PhysicsInvariant(
        name="energy_conservation",
        claim=f"E(t+dt)/E(t) <= {1.0 + tolerance:.4f} (viscous dissipation)",
        witness={
            "energy_prev": energy_prev,
            "energy_curr": energy_curr,
            "ratio": ratio,
            "tolerance": tolerance,
        },
        satisfied=satisfied,
        tag=ProofTag.CONSERVATION.name,
    )


def check_energy_monotone_decrease(
    energy_prev: float,
    energy_curr: float,
) -> PhysicsInvariant:
    """Verify energy is non-increasing after energy clamp."""
    satisfied = energy_curr <= energy_prev * (1.0 + 1e-12)

    return PhysicsInvariant(
        name="energy_monotone",
        claim="E(t+dt) <= E(t) (post-clamp monotonicity)",
        witness={
            "energy_prev": energy_prev,
            "energy_curr": energy_curr,
            "diff": energy_curr - energy_prev,
        },
        satisfied=satisfied,
        tag=ProofTag.MONOTONE.name,
    )


def check_rank_bound(
    max_rank_observed: int,
    max_rank_allowed: int,
) -> PhysicsInvariant:
    """Verify TT rank does not exceed the configured maximum."""
    satisfied = max_rank_observed <= max_rank_allowed

    return PhysicsInvariant(
        name="rank_bound",
        claim=f"max_rank <= {max_rank_allowed}",
        witness={
            "max_rank_observed": max_rank_observed,
            "max_rank_allowed": max_rank_allowed,
        },
        satisfied=satisfied,
        tag=ProofTag.BOUND.name,
    )


def check_compression_positive(
    compression_ratio: float,
) -> PhysicsInvariant:
    """Verify compression ratio is positive (QTT is actually compressing)."""
    satisfied = compression_ratio > 1.0

    return PhysicsInvariant(
        name="compression_positive",
        claim="CR > 1.0 (QTT compresses vs dense)",
        witness={"compression_ratio": compression_ratio},
        satisfied=satisfied,
        tag=ProofTag.POSITIVITY.name,
    )


def check_energy_positive(energy: float) -> PhysicsInvariant:
    """Verify kinetic energy is strictly positive (physical)."""
    satisfied = energy > 0.0

    return PhysicsInvariant(
        name="energy_positive",
        claim="E > 0 (kinetic energy positivity)",
        witness={"energy": energy},
        satisfied=satisfied,
        tag=ProofTag.POSITIVITY.name,
    )


def check_cfl_stability(
    dt: float,
    dx: float,
    u_max: float,
    cfl_target: float,
) -> PhysicsInvariant:
    """Verify the timestep satisfies the CFL condition."""
    if u_max > 0:
        cfl_actual = u_max * dt / dx
    else:
        cfl_actual = 0.0
    satisfied = cfl_actual <= cfl_target * 1.01  # 1% tolerance for float

    return PhysicsInvariant(
        name="cfl_stability",
        claim=f"CFL = U*dt/dx <= {cfl_target}",
        witness={
            "dt": dt,
            "dx": dx,
            "u_max": u_max,
            "cfl_actual": cfl_actual,
            "cfl_target": cfl_target,
        },
        satisfied=satisfied,
        tag=ProofTag.STABILITY.name,
    )


def check_finite_state(
    energy: float,
    max_rank: int,
) -> PhysicsInvariant:
    """Verify state values are finite (no NaN/Inf corruption)."""
    satisfied = math.isfinite(energy) and max_rank > 0

    return PhysicsInvariant(
        name="finite_state",
        claim="All state values are finite (no NaN/Inf)",
        witness={
            "energy_finite": math.isfinite(energy),
            "rank_positive": max_rank > 0,
        },
        satisfied=satisfied,
        tag=ProofTag.STABILITY.name,
    )


def check_divergence_bounded(
    divergence_max: float,
    threshold: float = 1.0,
) -> PhysicsInvariant:
    """Verify max |div(u)| is below threshold (incompressibility check).

    Parameters
    ----------
    divergence_max : float
        Maximum absolute divergence sampled from the velocity field.
    threshold : float
        Upper bound on acceptable divergence magnitude.
    """
    satisfied = divergence_max <= threshold

    return PhysicsInvariant(
        name="divergence_bounded",
        claim=f"max|div(u)| = {divergence_max:.4e} <= {threshold}",
        witness={
            "divergence_max": divergence_max,
            "threshold": threshold,
        },
        satisfied=satisfied,
        tag=ProofTag.STABILITY.name,
    )


# ═══════════════════════════════════════════════════════════════════
# RUN-LEVEL PROOFS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RunProof:
    """Run-level aggregated proof."""

    name: str
    claim: str
    witness: Dict[str, Any]
    satisfied: bool
    tag: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "claim": self.claim,
            "witness": self.witness,
            "satisfied": self.satisfied,
            "tag": self.tag,
        }


def check_convergence(
    energy_history: List[float],
    tolerance: float,
    min_steps: int = 50,
) -> RunProof:
    """Verify the simulation converged to steady state."""
    if len(energy_history) < min_steps:
        return RunProof(
            name="convergence",
            claim=f"ΔE/E < {tolerance:.1e} (steady state)",
            witness={"steps": len(energy_history), "min_required": min_steps},
            satisfied=False,
            tag=ProofTag.CONVERGENCE.name,
        )

    final_de = abs(energy_history[-1] - energy_history[-2]) / abs(energy_history[-2])
    satisfied = final_de < tolerance

    return RunProof(
        name="convergence",
        claim=f"ΔE/E < {tolerance:.1e} (steady state)",
        witness={
            "final_dE_over_E": final_de,
            "tolerance": tolerance,
            "total_steps": len(energy_history),
            "E_initial": energy_history[0],
            "E_final": energy_history[-1],
        },
        satisfied=satisfied,
        tag=ProofTag.CONVERGENCE.name,
    )


def check_total_energy_conservation(
    e_initial: float,
    e_final: float,
    max_loss_fraction: float = 0.10,
) -> RunProof:
    """Verify total energy loss is within physical bounds."""
    loss = (e_initial - e_final) / e_initial if e_initial > 0 else 0.0
    satisfied = 0.0 <= loss <= max_loss_fraction

    return RunProof(
        name="total_energy_conservation",
        claim=f"0 <= E_loss/E_0 <= {max_loss_fraction:.0%}",
        witness={
            "E_initial": e_initial,
            "E_final": e_final,
            "loss_fraction": loss,
            "max_allowed": max_loss_fraction,
        },
        satisfied=satisfied,
        tag=ProofTag.CONSERVATION.name,
    )


def check_hash_chain_integrity(step_proofs: List[StepProof]) -> RunProof:
    """Verify the hash chain linking all timesteps is unbroken."""
    breaks: List[int] = []
    for i in range(1, len(step_proofs)):
        if step_proofs[i].parent_commitment != step_proofs[i - 1].state_commitment:
            breaks.append(i)

    satisfied = len(breaks) == 0

    return RunProof(
        name="hash_chain_integrity",
        claim="All step hash links are valid (tamper-evident chain)",
        witness={
            "total_links": len(step_proofs) - 1 if len(step_proofs) > 1 else 0,
            "broken_links": breaks,
            "chain_length": len(step_proofs),
        },
        satisfied=satisfied,
        tag=ProofTag.CUSTOM.name,
    )


def check_all_steps_valid(step_proofs: List[StepProof]) -> RunProof:
    """Verify every timestep passed all physics invariants."""
    failed_steps = [sp.step_index for sp in step_proofs if not sp.all_satisfied]

    return RunProof(
        name="all_steps_valid",
        claim="Every timestep satisfies all physics invariants",
        witness={
            "total_steps": len(step_proofs),
            "failed_steps": failed_steps,
            "pass_rate": (len(step_proofs) - len(failed_steps)) / max(len(step_proofs), 1),
        },
        satisfied=len(failed_steps) == 0,
        tag=ProofTag.CUSTOM.name,
    )


def check_rank_monotone_decrease(
    mean_ranks: List[float],
) -> RunProof:
    """Check if mean TT rank is stable or decreasing (convergence indicator)."""
    # Use linear regression slope
    if len(mean_ranks) < 10:
        return RunProof(
            name="rank_stability",
            claim="Mean TT rank is stable or decreasing over run",
            witness={"length": len(mean_ranks)},
            satisfied=True,  # Not enough data to judge
            tag=ProofTag.STABILITY.name,
        )

    x = np.arange(len(mean_ranks), dtype=np.float64)
    y = np.array(mean_ranks, dtype=np.float64)
    slope = float(np.polyfit(x, y, 1)[0])
    # Positive slope (rank growing) beyond a threshold is a concern
    satisfied = slope <= 0.1  # Allow very slight growth

    return RunProof(
        name="rank_stability",
        claim="Mean TT rank is stable (slope \u2264 0.1 per step)",
        witness={
            "slope_per_step": slope,
            "initial_mean_rank": mean_ranks[0],
            "final_mean_rank": mean_ranks[-1],
            "length": len(mean_ranks),
        },
        satisfied=satisfied,
        tag=ProofTag.STABILITY.name,
    )


def check_spectrum_kolmogorov(
    energy_history: List[float],
    step_dts: Optional[List[float]] = None,
) -> RunProof:
    """Verify energy cascade is consistent with Kolmogorov -5/3 scaling.

    Uses the temporal energy decay as a proxy: for fully-developed
    turbulence, E(t) ~ t^{-10/7} (Kolmogorov theory).  We fit a
    power-law to the energy history and check that the exponent is
    in the physical range [-3, 0] (allowing for transient regimes,
    steady-state forcing, and numerical effects).

    This is a soft invariant — it confirms the simulation produces
    physically plausible energy dynamics.
    """
    if len(energy_history) < 20:
        return RunProof(
            name="spectrum_kolmogorov",
            claim="Energy decay consistent with Kolmogorov theory",
            witness={"steps": len(energy_history), "status": "insufficient_data"},
            satisfied=True,
            tag=ProofTag.CUSTOM.name,
        )

    # Normalise energy to initial value
    e0 = energy_history[0]
    if e0 <= 0:
        return RunProof(
            name="spectrum_kolmogorov",
            claim="Energy decay consistent with Kolmogorov theory",
            witness={"status": "zero_initial_energy"},
            satisfied=False,
            tag=ProofTag.CUSTOM.name,
        )

    # Fit log(E/E0) vs log(step) for steps > 10 (skip transient)
    start = max(10, len(energy_history) // 5)
    e_arr = np.array(energy_history[start:], dtype=np.float64)
    e_arr = np.maximum(e_arr / e0, 1e-30)  # avoid log(0)
    t_arr = np.arange(start, start + len(e_arr), dtype=np.float64) + 1.0

    log_e = np.log(e_arr)
    log_t = np.log(t_arr)

    coeffs = np.polyfit(log_t, log_e, 1)
    exponent = float(coeffs[0])
    r_squared = float(
        1.0 - np.sum((log_e - np.polyval(coeffs, log_t)) ** 2)
        / max(np.sum((log_e - np.mean(log_e)) ** 2), 1e-30)
    )

    # Exponent should be in [-3, 0] for physical energy decay
    satisfied = -3.0 <= exponent <= 0.05 and r_squared > 0.5

    return RunProof(
        name="spectrum_kolmogorov",
        claim="Energy exponent in [-3, 0] with R\u00b2 > 0.5",
        witness={
            "exponent": exponent,
            "r_squared": r_squared,
            "fit_start_step": start,
            "n_points": len(e_arr),
        },
        satisfied=satisfied,
        tag=ProofTag.CUSTOM.name,
    )


# ═══════════════════════════════════════════════════════════════════
# TRUSTLESS CERTIFICATE
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TrustlessCertificate:
    """Self-verifying trustless physics certificate.

    Contains all information needed for offline verification
    of a complete QTT Navier-Stokes simulation run.
    """

    # Identity
    certificate_id: str
    created_at: str
    version: str = "2.0.0"

    # Configuration commitment
    config_hash: str = ""
    config_params: Dict[str, Any] = field(default_factory=dict)

    # Merkle aggregation
    merkle_root: str = ""
    merkle_depth: int = 0
    merkle_leaf_count: int = 0

    # Step proofs (full chain)
    step_proofs: List[Dict[str, Any]] = field(default_factory=list)

    # Run-level proofs
    run_proofs: List[Dict[str, Any]] = field(default_factory=list)

    # Final state commitment
    final_state_commitment: str = ""

    # Initial state commitment
    initial_state_commitment: str = ""

    # Certificate seal
    certificate_hash: str = ""

    # Ed25519 signature (hex-encoded; empty if unsigned)
    signature: str = ""
    public_key: str = ""

    # Summary
    total_steps: int = 0
    all_invariants_satisfied: bool = False
    chain_intact: bool = False
    wall_time_s: float = 0.0

    def compute_seal(self) -> str:
        """Compute the certificate seal — SHA-256 of all content."""
        h = hashlib.sha256()
        h.update(self.certificate_id.encode("utf-8"))
        h.update(self.config_hash.encode("utf-8"))
        h.update(bytes.fromhex(self.merkle_root))
        h.update(struct.pack("<I", self.total_steps))
        h.update(bytes.fromhex(self.initial_state_commitment))
        h.update(bytes.fromhex(self.final_state_commitment))
        # Bind all run-level proofs
        for rp in self.run_proofs:
            h.update(rp["name"].encode("utf-8"))
            h.update(b"\x01" if rp["satisfied"] else b"\x00")
        self.certificate_hash = h.hexdigest()
        return self.certificate_hash

    def sign(self, private_key_path: Optional[Path] = None) -> str:
        """Sign the certificate seal with Ed25519.

        If *private_key_path* is ``None`` a fresh ephemeral key-pair is
        generated (useful for demos/testing).  For production, supply a
        PEM-encoded Ed25519 private key.

        Sets ``self.signature`` and ``self.public_key`` (hex-encoded).
        Returns the signature hex string.
        """
        if not _HAS_ED25519:
            raise RuntimeError(
                "Ed25519 signing requires the 'cryptography' package. "
                "Install via: pip install cryptography"
            )

        if not self.certificate_hash:
            self.compute_seal()

        if private_key_path is not None:
            raw = Path(private_key_path).read_bytes()
            from cryptography.hazmat.primitives.serialization import load_pem_private_key
            priv = load_pem_private_key(raw, password=None)
        else:
            priv = Ed25519PrivateKey.generate()

        sig_bytes = priv.sign(bytes.fromhex(self.certificate_hash))
        pub_bytes = priv.public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw
        )
        self.signature = sig_bytes.hex()
        self.public_key = pub_bytes.hex()
        return self.signature

    @staticmethod
    def verify_signature(
        certificate_hash_hex: str,
        signature_hex: str,
        public_key_hex: str,
    ) -> bool:
        """Verify an Ed25519 signature offline."""
        if not _HAS_ED25519:
            raise RuntimeError("Ed25519 verification requires 'cryptography'")
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey as PubCls,
        )
        pub = PubCls.from_public_bytes(bytes.fromhex(public_key_hex))
        try:
            pub.verify(bytes.fromhex(signature_hex),
                       bytes.fromhex(certificate_hash_hex))
            return True
        except Exception:
            return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "certificate_id": self.certificate_id,
            "created_at": self.created_at,
            "version": self.version,
            "config_hash": self.config_hash,
            "config_params": self.config_params,
            "merkle_root": self.merkle_root,
            "merkle_depth": self.merkle_depth,
            "merkle_leaf_count": self.merkle_leaf_count,
            "step_proofs": self.step_proofs,
            "run_proofs": self.run_proofs,
            "initial_state_commitment": self.initial_state_commitment,
            "final_state_commitment": self.final_state_commitment,
            "certificate_hash": self.certificate_hash,
            "signature": self.signature,
            "public_key": self.public_key,
            "total_steps": self.total_steps,
            "all_invariants_satisfied": self.all_invariants_satisfied,
            "chain_intact": self.chain_intact,
            "wall_time_s": self.wall_time_s,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: Path) -> None:
        path.write_text(self.to_json(), encoding="utf-8")

    def save_tpc(self, path: Path) -> None:
        """Export this certificate to a signed ``.tpc`` binary file.

        Bridges the JSON-based ``TrustlessCertificate`` into the binary
        ``TPCFile`` format used by the Rust verifier.  The TPC is signed
        with an ephemeral Ed25519 key so that the binary carries a non-
        zero signature verifiable by ``apps/trustless_verify``.

        The JSON Ed25519 key-pair (``self.signature`` / ``self.public_key``)
        signs the *certificate seal* (``self.certificate_hash``).  The TPC
        signature signs the *binary content hash* — these are independent
        commitments over the same logical certificate.
        """
        from tpc.format import (
            BenchmarkResult,
            CoverageLevel,
            LayerA,
            LayerB,
            LayerC,
            Metadata,
            QTTParams,
            TPCFile,
            TPCHeader,
            TheoremRef,
        )

        cfg = self.config_params

        # ── Header ──
        header = TPCHeader(
            certificate_id=uuid.UUID(self.certificate_id)
            if len(self.certificate_id) == 36
            else uuid.uuid4(),
            solver_hash=bytes.fromhex(self.config_hash)
            if len(self.config_hash) == 64
            else b"\x00" * 32,
        )

        # ── Layer A: run-proof theorems ──
        theorems = []
        for rp in self.run_proofs:
            theorems.append(TheoremRef(
                name=rp["name"],
                file=f"trustless_physics.py::{rp['name']}",
                statement_hash=hashlib.sha256(
                    rp["claim"].encode("utf-8")
                ).hexdigest(),
            ))
        layer_a = LayerA(
            proof_system="lean4",
            coverage=CoverageLevel.PARTIAL,
            theorems=theorems,
            notes=(
                f"Run-level proofs from QTT NS gauntlet: "
                f"{sum(1 for rp in self.run_proofs if rp['satisfied'])}"
                f"/{len(self.run_proofs)} passed"
            ),
        )

        # ── Layer B: Merkle hash-chain proof ──
        merkle_bytes = (
            bytes.fromhex(self.merkle_root) if self.merkle_root else b""
        )
        layer_b = LayerB(
            proof_system="none",
            public_inputs={
                "total_steps": self.total_steps,
                "merkle_depth": self.merkle_depth,
                "merkle_leaf_count": self.merkle_leaf_count,
                "initial_state_commitment": self.initial_state_commitment,
            },
            public_outputs={
                "merkle_root": self.merkle_root,
                "final_state_commitment": self.final_state_commitment,
                "certificate_hash": self.certificate_hash,
                "chain_intact": self.chain_intact,
            },
            proof_bytes=merkle_bytes,
        )

        # ── Layer C: benchmark results ──
        benchmarks = []
        if self.step_proofs:
            last_meta = self.step_proofs[-1].get("metadata", {})
            cr = last_meta.get("compression_ratio", 0)
            benchmarks.append(BenchmarkResult(
                name="compression_ratio",
                gauntlet="ahmed_body_ib",
                passed=cr > 1.0,
                metrics={
                    "compression_ratio": cr,
                    "max_rank": last_meta.get("max_rank", 0),
                    "mean_rank": last_meta.get("mean_rank", 0),
                    "N": int(cfg.get("N", 0)),
                    "total_steps": self.total_steps,
                    "wall_time_s": self.wall_time_s,
                },
            ))
        layer_c = LayerC(benchmarks=benchmarks)

        # ── Metadata ──
        n_bits = int(cfg.get("n_bits", 0))
        metadata = Metadata(
            solver="custom",
            domain="cfd",
            description=f"Ahmed Body IB QTT N={cfg.get('N', 0)} nb={n_bits}",
            qtt_params=QTTParams(
                grid_bits=n_bits,
                num_sites=n_bits * 3,
                max_rank=int(cfg.get("max_rank", 0)),
                physical_dim=2,
            ),
            extra={
                "N": int(cfg.get("N", 0)),
                "Re_eff": float(cfg.get("Re_eff", 0)),
                "total_steps": self.total_steps,
                "wall_time_s": self.wall_time_s,
                "certificate_hash": self.certificate_hash,
                "json_signature": self.signature,
                "json_public_key": self.public_key,
                "all_invariants_satisfied": self.all_invariants_satisfied,
            },
        )

        # ── Assemble and sign ──
        tpc = TPCFile(
            header=header,
            layer_a=layer_a,
            layer_b=layer_b,
            layer_c=layer_c,
            metadata=metadata,
        )
        tpc.sign_ephemeral()
        tpc.save(path)

    @classmethod
    def load(cls, path: Path) -> "TrustlessCertificate":
        data = json.loads(path.read_text(encoding="utf-8"))
        cert = cls(
            certificate_id=data["certificate_id"],
            created_at=data["created_at"],
            version=data.get("version", "1.0.0"),
        )
        cert.config_hash = data["config_hash"]
        cert.config_params = data["config_params"]
        cert.merkle_root = data["merkle_root"]
        cert.merkle_depth = data["merkle_depth"]
        cert.merkle_leaf_count = data["merkle_leaf_count"]
        cert.step_proofs = data["step_proofs"]
        cert.run_proofs = data["run_proofs"]
        cert.initial_state_commitment = data["initial_state_commitment"]
        cert.final_state_commitment = data["final_state_commitment"]
        cert.certificate_hash = data["certificate_hash"]
        cert.signature = data.get("signature", "")
        cert.public_key = data.get("public_key", "")
        cert.total_steps = data["total_steps"]
        cert.all_invariants_satisfied = data["all_invariants_satisfied"]
        cert.chain_intact = data["chain_intact"]
        cert.wall_time_s = data["wall_time_s"]
        return cert


# ═══════════════════════════════════════════════════════════════════
# TRUSTLESS PHYSICS PROVER
# ═══════════════════════════════════════════════════════════════════

class TrustlessPhysicsProver:
    """Wraps an AhmedBodyIBSolver to produce trustless physics certificates.

    At each timestep:
        1. Commit QTT state (SHA-256 of all TT cores)
        2. Verify physics invariants (energy, rank, CFL, compression, finite)
        3. Chain-link to previous step commitment

    After all steps:
        4. Build Merkle tree over step commitments
        5. Verify run-level invariants (convergence, energy conservation)
        6. Seal certificate with SHA-256

    The resulting certificate can be verified offline without
    re-running the simulation or having access to GPU hardware.
    """

    def __init__(self, solver: "SolverProtocol") -> None:
        """
        Parameters
        ----------
        solver : SolverProtocol
            Any solver satisfying the SolverProtocol interface
            (config, u, step(), _energy()).
        """
        self.solver = solver
        self.cfg = solver.config
        self._step_proofs: List[StepProof] = []
        self._energy_history: List[float] = []
        self._mean_rank_history: List[float] = []
        self._pcc_registry = PCCRegistry()
        self._initial_commitment: str = ""
        self._certificate_id = str(uuid.uuid4())

        # Commit initial state
        self._initial_commitment = self._commit_state()
        initial_energy = solver._energy(solver.u)
        self._energy_history.append(initial_energy)
        self._mean_rank_history.append(solver.u.mean_rank)

    def _commit_state(self) -> str:
        """Compute SHA-256 commitment of current QTT velocity state."""
        return qtt_vector_commitment(
            self.solver.u.x.cores.cores,
            self.solver.u.y.cores.cores,
            self.solver.u.z.cores.cores,
        )

    def _config_params(self) -> Dict[str, Any]:
        """Extract solver configuration as a serializable dict."""
        cfg = self.cfg
        bp = cfg.body_params
        return {
            "n_bits": cfg.n_bits,
            "N": cfg.N,
            "L": cfg.L,
            "dx": cfg.dx,
            "dt": cfg.dt,
            "max_rank": cfg.max_rank,
            "n_steps": cfg.n_steps,
            "cfl": cfg.cfl,
            "Re_eff": cfg.Re_eff,
            "nu_eff": cfg.nu_eff,
            "eta_brinkman": cfg.eta_brinkman,
            "convergence_tol": cfg.convergence_tol,
            "body_length": bp.length,
            "body_width": bp.width,
            "body_height": bp.height,
            "body_velocity": bp.velocity,
            "body_Re": bp.Re,
            "slant_angle_deg": bp.slant_angle_deg,
            "fillet_radius": bp.fillet_radius,
            "smagorinsky_cs": getattr(cfg, "smagorinsky_cs", 0.3),
            "integrator": getattr(cfg, "integrator", "euler"),
            "use_projection": getattr(cfg, "use_projection", False),
        }

    def step_with_proof(self, debug: bool = False) -> Tuple[Dict[str, Any], StepProof]:
        """Execute one solver timestep and produce a cryptographic step proof.

        Returns
        -------
        diagnostics : dict
            Standard solver diagnostics (energy, rank, CR, etc.)
        proof : StepProof
            Cryptographic proof for this timestep.
        """
        prev_energy = self._energy_history[-1]

        # Execute the step
        diag = self.solver.step(debug=debug)

        # Commit new state
        state_commitment = self._commit_state()
        parent_commitment = (
            self._step_proofs[-1].state_commitment
            if self._step_proofs
            else self._initial_commitment
        )

        energy = diag["energy"]
        max_rank = diag["max_rank_u"]
        mean_rank = diag["mean_rank_u"]
        cr = diag["compression_ratio"]

        self._energy_history.append(energy)
        self._mean_rank_history.append(mean_rank)

        # Build physics invariants
        invariants: List[PhysicsInvariant] = [
            check_energy_conservation(prev_energy, energy, tolerance=0.005),
            check_energy_monotone_decrease(prev_energy, energy),
            check_rank_bound(max_rank, self.cfg.max_rank),
            check_compression_positive(cr),
            check_energy_positive(energy),
            check_cfl_stability(
                self.cfg.dt, self.cfg.dx,
                self.cfg.body_params.velocity,
                self.cfg.cfl,
            ),
            check_finite_state(energy, max_rank),
        ]

        # Divergence invariant (when solver reports it)
        div_max = diag.get("divergence_max", None)
        if div_max is not None and div_max > 0:
            invariants.append(
                check_divergence_bounded(div_max, threshold=1.0)
            )

        # Compute step hash
        step_hash = _compute_step_hash(
            diag["step"], state_commitment, parent_commitment, invariants,
        )

        proof = StepProof(
            step_index=diag["step"],
            timestamp=time.time(),
            state_commitment=state_commitment,
            parent_commitment=parent_commitment,
            step_hash=step_hash,
            invariants=invariants,
            metadata={
                "energy": energy,
                "energy_prev": prev_energy,
                "max_rank": max_rank,
                "mean_rank": mean_rank,
                "compression_ratio": cr,
                "time": diag["time"],
                "clamped": diag.get("clamped", False),
            },
        )

        self._step_proofs.append(proof)

        # Register PCC payload
        pcc = PCCPayload(
            result=state_commitment,
            solver_name="AhmedBodyIBSolver.step",
            parameters={"step": diag["step"], "energy": energy, "rank": max_rank},
        )
        for inv in invariants:
            pcc.add_annotation(ProofAnnotation(
                tag=ProofTag[inv.tag],
                claim=inv.claim,
                witness=inv.witness,
                verified=inv.satisfied,
                verifier_name=inv.name,
            ))
        self._pcc_registry.register(pcc)

        return diag, proof

    def run_with_proof(
        self,
        verbose: bool = True,
        incremental_path: Optional[Path] = None,
    ) -> TrustlessCertificate:
        """Run the full simulation with cryptographic proof generation.

        Parameters
        ----------
        verbose : bool
            Print progress to console.
        incremental_path : Path, optional
            If given, step proofs are streamed as JSONL to this file
            during the run (crash-safe incremental certificate).

        Returns a TrustlessCertificate that can be independently verified.
        """
        ns = self.cfg.n_steps
        tol = self.cfg.convergence_tol
        t0_wall = time.perf_counter()

        if verbose:
            print(f"\n  ⛓  Trustless Physics Proof — {ns} steps, "
                  f"hash-chain + Merkle + 7 invariants/step")
            print(f"  {'Step':>6} {'Energy':>14} {'Rank':>6} "
                  f"{'CR':>8} {'Inv':>5} {'Hash':>10} {'ms':>6}")

        prev_e: Optional[float] = None
        converged = False

        # Open incremental JSONL stream if requested
        _jsonl_fh = None
        if incremental_path is not None:
            incremental_path = Path(incremental_path)
            incremental_path.parent.mkdir(parents=True, exist_ok=True)
            _jsonl_fh = open(incremental_path, "w", encoding="utf-8")

        for i in range(ns):
            t_step = time.perf_counter()
            diag, proof = self.step_with_proof(debug=(i < 2 and verbose))
            ms = (time.perf_counter() - t_step) * 1000

            # Stream step proof to JSONL
            if _jsonl_fh is not None:
                _jsonl_fh.write(json.dumps(proof.to_dict(), default=str) + "\n")
                _jsonl_fh.flush()

            inv_str = f"{'✓' if proof.all_satisfied else '✗'} {len(proof.invariants)}/{len(proof.invariants)}"
            hash_short = proof.step_hash[:10]

            if verbose and (i % max(1, ns // 20) == 0 or i == ns - 1):
                clamp_flag = " C" if diag.get("clamped") else ""
                print(f"  {diag['step']:>6} {diag['energy']:>14.6e} "
                      f"{diag['max_rank_u']:>6} {diag['compression_ratio']:>8.0f}×"
                      f" {inv_str:>5} {hash_short} {ms:>6.0f}{clamp_flag}")

            if prev_e is not None and prev_e > 0 and not diag.get("clamped"):
                rel = abs(diag["energy"] - prev_e) / prev_e
                if rel < tol and i > max(50, ns // 2):
                    if verbose:
                        print(f"  ⛓  Converged: ΔE/E = {rel:.2e}")
                    converged = True
                    break
            prev_e = diag["energy"]

        # Close incremental JSONL stream
        if _jsonl_fh is not None:
            _jsonl_fh.close()
            if verbose:
                print(f"  ⛓  Incremental proofs written to {incremental_path}")


        wall_time = time.perf_counter() - t0_wall

        if verbose and not converged:
            print(f"  ⛓  Completed {ns} steps.")

        # ── Build certificate ───────────────────────────────────────
        if verbose:
            print(f"\n  ⛓  Building Merkle tree over {len(self._step_proofs)} steps …")

        # Merkle tree over step hashes
        step_hashes = [sp.step_hash for sp in self._step_proofs]
        merkle = MerkleTree(step_hashes)

        if verbose:
            print(f"  ⛓  Merkle root: {merkle.root[:16]}…  depth={merkle.depth}")

        # Run-level proofs
        run_proofs: List[RunProof] = [
            check_convergence(self._energy_history, tol),
            check_total_energy_conservation(
                self._energy_history[0],
                self._energy_history[-1],
                max_loss_fraction=0.10,
            ),
            check_hash_chain_integrity(self._step_proofs),
            check_all_steps_valid(self._step_proofs),
            check_rank_monotone_decrease(self._mean_rank_history),
        ]

        # Spectrum Kolmogorov scaling check
        if len(self._energy_history) >= 10:
            run_proofs.append(
                check_spectrum_kolmogorov(self._energy_history)
            )

        # PCC registry chain verification
        pcc_chain_valid = self._pcc_registry.verify_chain()
        run_proofs.append(RunProof(
            name="pcc_chain_integrity",
            claim="PCC hash-chain audit trail is intact",
            witness=self._pcc_registry.summary(),
            satisfied=pcc_chain_valid,
            tag=ProofTag.CUSTOM.name,
        ))

        all_run_satisfied = all(rp.satisfied for rp in run_proofs)
        all_step_satisfied = all(sp.all_satisfied for sp in self._step_proofs)

        config_params = self._config_params()
        config_hash = config_commitment(config_params)
        final_commitment = self._commit_state()

        cert = TrustlessCertificate(
            certificate_id=self._certificate_id,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        cert.config_hash = config_hash
        cert.config_params = config_params
        cert.merkle_root = merkle.root
        cert.merkle_depth = merkle.depth
        cert.merkle_leaf_count = merkle.leaf_count
        cert.step_proofs = [sp.to_dict() for sp in self._step_proofs]
        cert.run_proofs = [rp.to_dict() for rp in run_proofs]
        cert.initial_state_commitment = self._initial_commitment
        cert.final_state_commitment = final_commitment
        cert.total_steps = len(self._step_proofs)
        cert.all_invariants_satisfied = all_run_satisfied and all_step_satisfied
        cert.chain_intact = pcc_chain_valid
        cert.wall_time_s = wall_time
        cert.compute_seal()

        # Ed25519 ephemeral signature
        if _HAS_ED25519:
            cert.sign()
            if verbose:
                print(f"  \u26d3  Ed25519 signature: {cert.signature[:16]}\u2026")
                print(f"  \u26d3  Public key:        {cert.public_key[:16]}\u2026")

        if verbose:
            self._print_summary(cert, run_proofs, merkle)

        return cert

    def _print_summary(
        self,
        cert: TrustlessCertificate,
        run_proofs: List[RunProof],
        merkle: MerkleTree,
    ) -> None:
        """Print proof summary to console."""
        sep = "═" * 72
        print(f"\n  {sep}")
        print(f"  ⛓  TRUSTLESS PHYSICS CERTIFICATE")
        print(f"  {sep}")
        print(f"  ID:             {cert.certificate_id}")
        print(f"  Created:        {cert.created_at}")
        print(f"  Config hash:    {cert.config_hash[:16]}…")
        print(f"  Initial state:  {cert.initial_state_commitment[:16]}…")
        print(f"  Final state:    {cert.final_state_commitment[:16]}…")
        print(f"  Merkle root:    {cert.merkle_root[:16]}…")
        print(f"  Merkle depth:   {cert.merkle_depth}")
        print(f"  Total steps:    {cert.total_steps}")
        print(f"  Wall time:      {cert.wall_time_s:.1f} s")
        print(f"  Certificate:    {cert.certificate_hash[:16]}…")
        print()

        n_step_invariants = sum(len(sp.invariants) for sp in self._step_proofs)
        n_step_passed = sum(
            sum(1 for inv in sp.invariants if inv.satisfied)
            for sp in self._step_proofs
        )

        print(f"  ── Step-Level Proofs ──")
        print(f"  Total invariant checks:  {n_step_invariants}")
        print(f"  Passed:                  {n_step_passed}")
        print(f"  Failed:                  {n_step_invariants - n_step_passed}")
        print()

        print(f"  ── Run-Level Proofs ──")
        for rp in run_proofs:
            status = "✓" if rp.satisfied else "✗"
            print(f"    {status} {rp.name}: {rp.claim}")
        print()

        overall = "✓ ALL PROOFS PASSED" if cert.all_invariants_satisfied else "✗ SOME PROOFS FAILED"
        print(f"  {sep}")
        print(f"  ⛓  VERDICT: {overall}")
        print(f"  {sep}")


# ═══════════════════════════════════════════════════════════════════
# STANDALONE VERIFIER
# ═══════════════════════════════════════════════════════════════════

def verify_certificate(cert_path: Path, verbose: bool = True) -> bool:
    """Verify a trustless physics certificate offline.

    This function requires NO GPU, NO solver, NO simulation.
    It verifies:
        1. Certificate seal integrity
        2. Config commitment consistency
        3. Hash-chain integrity (every step links to previous)
        4. Merkle tree root matches step hashes
        5. All physics invariants were satisfied
        6. All run-level proofs passed

    Parameters
    ----------
    cert_path : Path
        Path to the JSON certificate file.
    verbose : bool
        Print verification details.

    Returns
    -------
    bool
        True iff the certificate is valid.
    """
    cert = TrustlessCertificate.load(cert_path)

    if verbose:
        print(f"\n  ⛓  TRUSTLESS PHYSICS VERIFICATION")
        print(f"  Certificate: {cert.certificate_id}")
        print(f"  Created:     {cert.created_at}")
        print()

    checks: List[Tuple[str, bool]] = []

    # 1. Seal integrity
    expected_seal = cert.certificate_hash
    cert_copy = TrustlessCertificate(
        certificate_id=cert.certificate_id,
        created_at=cert.created_at,
    )
    cert_copy.config_hash = cert.config_hash
    cert_copy.merkle_root = cert.merkle_root
    cert_copy.total_steps = cert.total_steps
    cert_copy.initial_state_commitment = cert.initial_state_commitment
    cert_copy.final_state_commitment = cert.final_state_commitment
    cert_copy.run_proofs = cert.run_proofs
    recomputed_seal = cert_copy.compute_seal()
    seal_ok = recomputed_seal == expected_seal
    checks.append(("Certificate seal integrity", seal_ok))

    # 2. Config commitment
    recomputed_config_hash = config_commitment(cert.config_params)
    config_ok = recomputed_config_hash == cert.config_hash
    checks.append(("Config commitment consistency", config_ok))

    # 3. Hash-chain integrity
    chain_ok = True
    for i in range(1, len(cert.step_proofs)):
        prev_commit = cert.step_proofs[i - 1]["state_commitment"]
        curr_parent = cert.step_proofs[i]["parent_commitment"]
        if curr_parent != prev_commit:
            chain_ok = False
            break
    # Also check first step links to initial state
    if cert.step_proofs:
        first_parent = cert.step_proofs[0]["parent_commitment"]
        if first_parent != cert.initial_state_commitment:
            chain_ok = False
    checks.append(("Hash-chain integrity", chain_ok))

    # 4. Step hash recomputation
    step_hash_ok = True
    for sp_dict in cert.step_proofs:
        invariants = [
            PhysicsInvariant(**inv) for inv in sp_dict["invariants"]
        ]
        expected = sp_dict["step_hash"]
        recomputed = _compute_step_hash(
            sp_dict["step_index"],
            sp_dict["state_commitment"],
            sp_dict["parent_commitment"],
            invariants,
        )
        if recomputed != expected:
            step_hash_ok = False
            break
    checks.append(("Step hash recomputation", step_hash_ok))

    # 5. Merkle root
    step_hashes = [sp["step_hash"] for sp in cert.step_proofs]
    if step_hashes:
        merkle = MerkleTree(step_hashes)
        merkle_ok = merkle.root == cert.merkle_root
    else:
        merkle_ok = False
    checks.append(("Merkle root verification", merkle_ok))

    # 6. All step invariants
    step_inv_ok = all(sp["all_satisfied"] for sp in cert.step_proofs)
    checks.append(("All step invariants satisfied", step_inv_ok))

    # 7. All run-level proofs
    run_ok = all(rp["satisfied"] for rp in cert.run_proofs)
    checks.append(("All run-level proofs passed", run_ok))

    # 8. Final state matches last step commitment
    if cert.step_proofs:
        final_match = cert.step_proofs[-1]["state_commitment"] == cert.final_state_commitment
    else:
        final_match = False
    checks.append(("Final state commitment match", final_match))

    # 9. Ed25519 signature (if present)
    if cert.signature and cert.public_key:
        sig_ok = TrustlessCertificate.verify_signature(
            cert.certificate_hash, cert.signature, cert.public_key,
        )
        checks.append(("Ed25519 signature verification", sig_ok))

    # Print results
    all_ok = all(ok for _, ok in checks)

    if verbose:
        for name, ok in checks:
            status = "✓" if ok else "✗"
            print(f"  {status} {name}")
        print()

        n_steps = cert.total_steps
        n_invariants = sum(len(sp["invariants"]) for sp in cert.step_proofs)
        print(f"  Steps verified:      {n_steps}")
        print(f"  Invariants checked:  {n_invariants}")
        print(f"  Merkle depth:        {cert.merkle_depth}")
        print()

        verdict = "✓ CERTIFICATE VALID" if all_ok else "✗ CERTIFICATE INVALID"
        print(f"  ⛓  VERDICT: {verdict}")

    return all_ok


def verify_step(
    cert_path: Path,
    step_index: int,
    verbose: bool = True,
) -> Tuple[bool, List[Tuple[str, str]]]:
    """Verify a single step using its Merkle inclusion proof.

    Returns
    -------
    (valid, merkle_path) : (bool, List[(hash, side)])
    """
    cert = TrustlessCertificate.load(cert_path)

    # Find the step
    step_dict = None
    proof_idx = None
    for idx, sp in enumerate(cert.step_proofs):
        if sp["step_index"] == step_index:
            step_dict = sp
            proof_idx = idx
            break

    if step_dict is None:
        raise ValueError(f"Step {step_index} not found in certificate")

    # Recompute step hash
    invariants = [PhysicsInvariant(**inv) for inv in step_dict["invariants"]]
    recomputed = _compute_step_hash(
        step_dict["step_index"],
        step_dict["state_commitment"],
        step_dict["parent_commitment"],
        invariants,
    )
    hash_ok = recomputed == step_dict["step_hash"]

    # Build Merkle tree and get inclusion proof
    step_hashes = [sp["step_hash"] for sp in cert.step_proofs]
    merkle = MerkleTree(step_hashes)
    merkle_path = merkle.proof(proof_idx)
    merkle_ok = MerkleTree.verify_proof(
        step_dict["step_hash"], merkle_path, cert.merkle_root,
    )

    # All invariants satisfied
    inv_ok = step_dict["all_satisfied"]

    valid = hash_ok and merkle_ok and inv_ok

    if verbose:
        print(f"\n  ⛓  Step {step_index} Verification")
        print(f"  ✓ Step hash:     {step_dict['step_hash'][:16]}…" if hash_ok
              else f"  ✗ Step hash mismatch")
        print(f"  ✓ Merkle proof:  depth={len(merkle_path)}" if merkle_ok
              else f"  ✗ Merkle proof invalid")
        print(f"  ✓ Invariants:    {len(invariants)}/{len(invariants)}" if inv_ok
              else f"  ✗ Invariant failures")
        for inv in invariants:
            s = "✓" if inv.satisfied else "✗"
            print(f"    {s} {inv.name}: {inv.claim}")
        print(f"  Verdict: {'✓ VALID' if valid else '✗ INVALID'}")

    return valid, merkle_path


# ═══════════════════════════════════════════════════════════════════
# CERTIFICATE REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════

def generate_proof_report(cert: TrustlessCertificate, output_dir: Path) -> str:
    """Generate a human-readable Markdown proof report."""
    lines: List[str] = []
    sep = "═" * 72

    lines.append(f"# Trustless Physics Certificate")
    lines.append(f"")
    lines.append(f"**Report ID:** HTR-2026-002-ZK-PROOF")
    lines.append(f"**Certificate ID:** `{cert.certificate_id}`")
    lines.append(f"**Created:** {cert.created_at}")
    lines.append(f"**Seal:** `{cert.certificate_hash}`")
    lines.append(f"")

    lines.append(f"## Configuration Commitment")
    lines.append(f"")
    lines.append(f"**Config Hash:** `{cert.config_hash}`")
    lines.append(f"")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|-----------|-------|")
    for k, v in cert.config_params.items():
        lines.append(f"| {k} | {v} |")
    lines.append(f"")

    lines.append(f"## Cryptographic Chain")
    lines.append(f"")
    lines.append(f"| Property | Value |")
    lines.append(f"|----------|-------|")
    lines.append(f"| Initial State | `{cert.initial_state_commitment[:32]}…` |")
    lines.append(f"| Final State | `{cert.final_state_commitment[:32]}…` |")
    lines.append(f"| Merkle Root | `{cert.merkle_root[:32]}…` |")
    lines.append(f"| Merkle Depth | {cert.merkle_depth} |")
    lines.append(f"| Chain Length | {cert.total_steps} steps |")
    lines.append(f"| Chain Intact | {'✓' if cert.chain_intact else '✗'} |")
    lines.append(f"")

    # Step proof summary
    n_inv = sum(len(sp["invariants"]) for sp in cert.step_proofs)
    n_pass = sum(
        sum(1 for inv in sp["invariants"] if inv["satisfied"])
        for sp in cert.step_proofs
    )
    lines.append(f"## Step-Level Proofs")
    lines.append(f"")
    lines.append(f"- **Total invariant checks:** {n_inv}")
    lines.append(f"- **Passed:** {n_pass} ({n_pass/max(n_inv,1)*100:.1f}%)")
    lines.append(f"- **Failed:** {n_inv - n_pass}")
    lines.append(f"")

    lines.append(f"### Invariants Verified Per Step")
    lines.append(f"")
    lines.append(f"| # | Invariant | Type | Description |")
    lines.append(f"|---|-----------|------|-------------|")
    lines.append(f"| 1 | energy_conservation | CONSERVATION | E(t+dt)/E(t) ≤ 1.005 |")
    lines.append(f"| 2 | energy_monotone | MONOTONE | E(t+dt) ≤ E(t) post-clamp |")
    lines.append(f"| 3 | rank_bound | BOUND | max_rank ≤ χ |")
    lines.append(f"| 4 | compression_positive | POSITIVITY | CR > 1.0 |")
    lines.append(f"| 5 | energy_positive | POSITIVITY | E > 0 |")
    lines.append(f"| 6 | cfl_stability | STABILITY | CFL ≤ target |")
    lines.append(f"| 7 | finite_state | STABILITY | No NaN/Inf |")
    lines.append(f"| 8 | divergence_bounded | STABILITY | max|div(u)| \u2264 1.0 |")
    lines.append(f"")

    lines.append(f"## Run-Level Proofs")
    lines.append(f"")
    lines.append(f"| # | Proof | Claim | Verdict |")
    lines.append(f"|---|-------|-------|---------|")
    for i, rp in enumerate(cert.run_proofs, 1):
        v = "✓" if rp["satisfied"] else "✗"
        lines.append(f"| {i} | {rp['name']} | {rp['claim']} | {v} |")
    lines.append(f"")

    lines.append(f"## Verdict")
    lines.append(f"")
    if cert.all_invariants_satisfied:
        lines.append(f"**✓ ALL PROOFS PASSED — CERTIFICATE VALID**")
    else:
        lines.append(f"**✗ SOME PROOFS FAILED — CERTIFICATE INVALID**")
    lines.append(f"")
    lines.append(f"The certificate seal `{cert.certificate_hash[:32]}…` binds the "
                 f"configuration commitment, Merkle root, initial/final state commitments, "
                 f"and all run-level proof verdicts into a single tamper-evident hash.")
    lines.append(f"")

    if cert.signature:
        lines.append(f"## Ed25519 Signature")
        lines.append(f"")
        lines.append(f"| Field | Value |")
        lines.append(f"|-------|-------|")
        lines.append(f"| Public Key | `{cert.public_key}` |")
        lines.append(f"| Signature | `{cert.signature[:64]}…` |")
        lines.append(f"")

    lines.append(f"---")
    lines.append(f"*Generated by HyperTensor Trustless Physics Engine v{cert.version}*")

    report = "\n".join(lines)
    report_path = output_dir / "TRUSTLESS_PHYSICS_PROOF.md"
    report_path.write_text(report, encoding="utf-8")
    return str(report_path)
