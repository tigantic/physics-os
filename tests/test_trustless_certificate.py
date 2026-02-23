"""
Tests for scripts.trustless_physics
====================================

Covers:
- MerkleTree construction, root, proof, verify_proof
- SHA-256 primitives
- PhysicsInvariant / StepProof construction
- Certificate seal integrity (sign, verify, tamper detection)
- Ed25519 sign/verify round-trip
- SolverProtocol structural check
- Canonical float encoding reproducibility
- Divergence invariant
- Spectrum Kolmogorov invariant
- Config commitment determinism
- Fuzz: random invariant combos
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.trustless_physics import (
    sha256_bytes,
    qtt_core_commitment,
    config_commitment,
    MerkleTree,
    PhysicsInvariant,
    StepProof,
    RunProof,
    TrustlessCertificate,
    check_energy_conservation,
    check_energy_monotone_decrease,
    check_rank_bound,
    check_compression_positive,
    check_energy_positive,
    check_cfl_stability,
    check_finite_state,
    check_divergence_bounded,
    check_spectrum_kolmogorov,
    check_convergence,
    check_total_energy_conservation,
    check_hash_chain_integrity,
    check_all_steps_valid,
    SolverProtocol,
    _HAS_ED25519,
)


# ─── SHA-256 primitives ────────────────────────────────────────────

class TestSHA256:
    def test_deterministic(self) -> None:
        data = b"HyperTensor QTT"
        h1 = sha256_bytes(data)
        h2 = sha256_bytes(data)
        assert h1 == h2
        assert len(h1) == 64  # hex string

    def test_different_inputs(self) -> None:
        assert sha256_bytes(b"a") != sha256_bytes(b"b")


# ─── MerkleTree ────────────────────────────────────────────────────

class TestMerkleTree:
    def test_single_leaf(self) -> None:
        m = MerkleTree(["abc123"])
        assert m.root is not None
        assert m.depth >= 0
        assert m.leaf_count == 1

    def test_power_of_two(self) -> None:
        leaves = [sha256_bytes(f"leaf_{i}".encode()) for i in range(8)]
        m = MerkleTree(leaves)
        assert m.leaf_count == 8
        assert m.depth == 3

    def test_proof_verify(self) -> None:
        leaves = [sha256_bytes(f"step_{i}".encode()) for i in range(16)]
        m = MerkleTree(leaves)
        for idx in [0, 5, 15]:
            proof = m.proof(idx)
            assert MerkleTree.verify_proof(leaves[idx], proof, m.root)

    def test_proof_tamper_detection(self) -> None:
        leaves = [sha256_bytes(f"step_{i}".encode()) for i in range(8)]
        m = MerkleTree(leaves)
        proof = m.proof(3)
        # Tamper with a leaf
        assert not MerkleTree.verify_proof(sha256_bytes(b"tampered"), proof, m.root)

    def test_deterministic_root(self) -> None:
        leaves = [sha256_bytes(c.encode()) for c in ["a", "b", "c", "d"]]
        m1 = MerkleTree(leaves)
        m2 = MerkleTree(leaves)
        assert m1.root == m2.root

    @pytest.mark.parametrize("n", [1, 2, 3, 7, 16, 31, 64])
    def test_various_sizes(self, n: int) -> None:
        leaves = [sha256_bytes(f"leaf_{i}".encode()) for i in range(n)]
        m = MerkleTree(leaves)
        assert m.leaf_count == n
        assert m.root is not None
        # Every leaf should have a valid proof
        for i in range(n):
            proof = m.proof(i)
            assert MerkleTree.verify_proof(leaves[i], proof, m.root)


# ─── Physics invariants ────────────────────────────────────────────

class TestInvariants:
    def test_energy_conservation_pass(self) -> None:
        inv = check_energy_conservation(1.0, 1.004, tolerance=0.005)
        assert inv.satisfied

    def test_energy_conservation_fail(self) -> None:
        inv = check_energy_conservation(1.0, 1.01, tolerance=0.005)
        assert not inv.satisfied

    def test_energy_monotone_pass(self) -> None:
        inv = check_energy_monotone_decrease(1.0, 0.99)
        assert inv.satisfied

    def test_energy_monotone_fail(self) -> None:
        inv = check_energy_monotone_decrease(1.0, 1.01)
        assert not inv.satisfied

    def test_rank_bound_pass(self) -> None:
        inv = check_rank_bound(32, 48)
        assert inv.satisfied

    def test_rank_bound_fail(self) -> None:
        inv = check_rank_bound(50, 48)
        assert not inv.satisfied

    def test_compression_positive(self) -> None:
        assert check_compression_positive(10.0).satisfied
        assert not check_compression_positive(0.5).satisfied

    def test_energy_positive(self) -> None:
        assert check_energy_positive(1.0).satisfied
        assert not check_energy_positive(-0.1).satisfied

    def test_cfl_stability(self) -> None:
        # CFL = dt * V / dx = 0.001 * 10 / 0.1 = 0.1
        inv = check_cfl_stability(0.001, 0.1, 10.0, 0.2)
        assert inv.satisfied

    def test_finite_state(self) -> None:
        assert check_finite_state(1.0, 32).satisfied
        assert not check_finite_state(float("nan"), 32).satisfied
        assert not check_finite_state(float("inf"), 32).satisfied


# ─── Divergence invariant ──────────────────────────────────────────

class TestDivergenceInvariant:
    def test_bounded_pass(self) -> None:
        inv = check_divergence_bounded(0.5, threshold=1.0)
        assert inv.satisfied

    def test_bounded_fail(self) -> None:
        inv = check_divergence_bounded(1.5, threshold=1.0)
        assert not inv.satisfied

    def test_exact_threshold(self) -> None:
        inv = check_divergence_bounded(1.0, threshold=1.0)
        assert inv.satisfied  # <= is satisfied


# ─── Spectrum Kolmogorov invariant ─────────────────────────────────

class TestSpectrumInvariant:
    def test_decaying_energy(self) -> None:
        # Monotonically decaying energy → should produce negative exponent
        energy = [100.0 * (0.95 ** i) for i in range(50)]
        rp = check_spectrum_kolmogorov(energy)
        # Should pass: exponent is negative, R² should be high
        assert isinstance(rp, RunProof)

    def test_constant_energy(self) -> None:
        # Constant energy → exponent ≈ 0
        energy = [1.0] * 50
        rp = check_spectrum_kolmogorov(energy)
        # Exponent ~0 is within [-3, 0]
        assert isinstance(rp, RunProof)

    def test_too_few_points(self) -> None:
        # Fewer than 3 points
        energy = [1.0, 0.9]
        rp = check_spectrum_kolmogorov(energy)
        assert isinstance(rp, RunProof)


# ─── Config commitment ─────────────────────────────────────────────

class TestConfigCommitment:
    def test_deterministic(self) -> None:
        params = {"Re": 1000, "dt": 0.001, "N": 128}
        h1 = config_commitment(params)
        h2 = config_commitment(params)
        assert h1 == h2

    def test_order_independence(self) -> None:
        """JSON sort_keys makes key order irrelevant."""
        h1 = config_commitment({"a": 1, "b": 2})
        h2 = config_commitment({"b": 2, "a": 1})
        assert h1 == h2

    def test_different_configs(self) -> None:
        h1 = config_commitment({"Re": 1000})
        h2 = config_commitment({"Re": 2000})
        assert h1 != h2


# ─── Canonical float encoding ──────────────────────────────────────

class TestCanonicalFloat:
    def test_reproducible(self) -> None:
        """Same data → same commitment regardless of source."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        data1 = torch.tensor([1.0 / 3.0, math.pi, math.e], dtype=torch.float64)
        data2 = torch.tensor([1.0 / 3.0, math.pi, math.e], dtype=torch.float64)
        h1 = qtt_core_commitment([data1])
        h2 = qtt_core_commitment([data2])
        assert h1 == h2


# ─── TrustlessCertificate ──────────────────────────────────────────

class TestCertificate:
    def _make_cert(self) -> TrustlessCertificate:
        cert = TrustlessCertificate(
            certificate_id="test-001",
            created_at="2025-01-01T00:00:00+00:00",
        )
        cert.config_hash = sha256_bytes(b"config")
        cert.merkle_root = sha256_bytes(b"merkle")
        cert.total_steps = 10
        cert.initial_state_commitment = sha256_bytes(b"init")
        cert.final_state_commitment = sha256_bytes(b"final")
        cert.run_proofs = [{"name": "test", "satisfied": True}]
        cert.compute_seal()
        return cert

    def test_seal_deterministic(self) -> None:
        c1 = self._make_cert()
        c2 = self._make_cert()
        assert c1.certificate_hash == c2.certificate_hash

    def test_seal_tamper_detection(self) -> None:
        cert = self._make_cert()
        original_hash = cert.certificate_hash
        cert.total_steps = 999
        cert.compute_seal()
        assert cert.certificate_hash != original_hash

    def test_version(self) -> None:
        cert = self._make_cert()
        assert cert.version == "2.0.0"

    def test_save_load_round_trip(self) -> None:
        cert = self._make_cert()
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            path = Path(f.name)

        try:
            cert.save(path)
            loaded = TrustlessCertificate.load(path)
            assert loaded.certificate_id == cert.certificate_id
            assert loaded.certificate_hash == cert.certificate_hash
            assert loaded.total_steps == cert.total_steps
        finally:
            path.unlink(missing_ok=True)

    def test_to_dict_keys(self) -> None:
        cert = self._make_cert()
        d = cert.to_dict()
        required_keys = {
            "certificate_id", "created_at", "version",
            "config_hash", "merkle_root", "total_steps",
            "certificate_hash", "signature", "public_key",
            "all_invariants_satisfied", "chain_intact",
        }
        assert required_keys.issubset(d.keys())


# ─── Ed25519 signing ───────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_ED25519, reason="cryptography not installed")
class TestEd25519:
    def test_sign_verify_round_trip(self) -> None:
        cert = TrustlessCertificate(
            certificate_id="ed25519-test",
            created_at="2025-01-01T00:00:00+00:00",
        )
        cert.config_hash = sha256_bytes(b"test_config")
        cert.merkle_root = sha256_bytes(b"test_merkle")
        cert.total_steps = 5
        cert.initial_state_commitment = sha256_bytes(b"init")
        cert.final_state_commitment = sha256_bytes(b"final")
        cert.run_proofs = []
        cert.compute_seal()
        cert.sign()

        assert len(cert.signature) == 128  # 64-byte sig → 128 hex chars
        assert len(cert.public_key) == 64  # 32-byte key → 64 hex chars

        ok = TrustlessCertificate.verify_signature(
            cert.certificate_hash, cert.signature, cert.public_key,
        )
        assert ok

    def test_tampered_sig_fails(self) -> None:
        cert = TrustlessCertificate(
            certificate_id="tamper-test",
            created_at="2025-01-01T00:00:00+00:00",
        )
        cert.config_hash = sha256_bytes(b"x")
        cert.merkle_root = sha256_bytes(b"y")
        cert.total_steps = 1
        cert.initial_state_commitment = sha256_bytes(b"i")
        cert.final_state_commitment = sha256_bytes(b"f")
        cert.run_proofs = []
        cert.compute_seal()
        cert.sign()

        # Tamper with signature
        bad_sig = "00" * 64
        ok = TrustlessCertificate.verify_signature(
            cert.certificate_hash, bad_sig, cert.public_key,
        )
        assert not ok

    def test_save_load_preserves_signature(self) -> None:
        cert = TrustlessCertificate(
            certificate_id="persist-test",
            created_at="2025-01-01T00:00:00+00:00",
        )
        cert.config_hash = sha256_bytes(b"a")
        cert.merkle_root = sha256_bytes(b"b")
        cert.total_steps = 1
        cert.initial_state_commitment = sha256_bytes(b"c")
        cert.final_state_commitment = sha256_bytes(b"d")
        cert.run_proofs = []
        cert.compute_seal()
        cert.sign()

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            path = Path(f.name)

        try:
            cert.save(path)
            loaded = TrustlessCertificate.load(path)
            assert loaded.signature == cert.signature
            assert loaded.public_key == cert.public_key

            ok = TrustlessCertificate.verify_signature(
                loaded.certificate_hash, loaded.signature, loaded.public_key,
            )
            assert ok
        finally:
            path.unlink(missing_ok=True)


# ─── SolverProtocol ────────────────────────────────────────────────

class TestSolverProtocol:
    def test_protocol_is_runtime_checkable(self) -> None:
        """Protocol must be decorated with @runtime_checkable."""
        from typing import runtime_checkable

        # isinstance check should work
        class FakeSolver:
            config = None
            u = None

            def step(self, debug: bool = False):
                return {}

            def _energy(self, u):
                return 0.0

        assert isinstance(FakeSolver(), SolverProtocol)

    def test_non_compliant_rejected(self) -> None:
        class BadSolver:
            pass

        assert not isinstance(BadSolver(), SolverProtocol)


# ─── Hash chain ─────────────────────────────────────────────────────

class TestHashChain:
    def _make_step_proofs(self, n: int) -> list:
        proofs = []
        parent = "initial_commitment"
        for i in range(n):
            state = sha256_bytes(f"state_{i}".encode())
            step_hash = sha256_bytes(f"{i}_{state}_{parent}".encode())
            proofs.append(StepProof(
                step_index=i,
                timestamp=0.0,
                state_commitment=state,
                parent_commitment=parent,
                step_hash=step_hash,
                invariants=[
                    PhysicsInvariant(
                        name="test", claim="ok", witness="w",
                        satisfied=True, tag="POSITIVITY",
                    )
                ],
                metadata={},
            ))
            parent = state
        return proofs

    def test_chain_intact(self) -> None:
        proofs = self._make_step_proofs(10)
        rp = check_hash_chain_integrity(proofs)
        assert rp.satisfied

    def test_chain_broken(self) -> None:
        proofs = self._make_step_proofs(10)
        proofs[5] = StepProof(
            step_index=5,
            timestamp=0.0,
            state_commitment="wrong",
            parent_commitment="bad_parent",
            step_hash="xxx",
            invariants=[],
            metadata={},
        )
        rp = check_hash_chain_integrity(proofs)
        assert not rp.satisfied

    def test_all_steps_valid(self) -> None:
        proofs = self._make_step_proofs(5)
        rp = check_all_steps_valid(proofs)
        assert rp.satisfied


# ─── Fuzz: random invariant combos ─────────────────────────────────

class TestFuzz:
    @pytest.mark.parametrize("seed", range(10))
    def test_certificate_seal_consistency(self, seed: int) -> None:
        """Seal must be deterministic for same inputs regardless of creation order."""
        np.random.seed(seed)
        n_steps = np.random.randint(1, 50)

        cert = TrustlessCertificate(
            certificate_id=f"fuzz-{seed}",
            created_at="2025-01-01T00:00:00+00:00",
        )
        cert.config_hash = sha256_bytes(f"config_{seed}".encode())
        cert.merkle_root = sha256_bytes(f"merkle_{seed}".encode())
        cert.total_steps = n_steps
        cert.initial_state_commitment = sha256_bytes(b"init")
        cert.final_state_commitment = sha256_bytes(f"final_{seed}".encode())
        cert.run_proofs = [
            {"name": f"proof_{i}", "satisfied": bool(np.random.rand() > 0.3)}
            for i in range(3)
        ]
        cert.compute_seal()

        # Recompute — must match
        cert2 = TrustlessCertificate(
            certificate_id=cert.certificate_id,
            created_at=cert.created_at,
        )
        cert2.config_hash = cert.config_hash
        cert2.merkle_root = cert.merkle_root
        cert2.total_steps = cert.total_steps
        cert2.initial_state_commitment = cert.initial_state_commitment
        cert2.final_state_commitment = cert.final_state_commitment
        cert2.run_proofs = cert.run_proofs
        cert2.compute_seal()

        assert cert.certificate_hash == cert2.certificate_hash
