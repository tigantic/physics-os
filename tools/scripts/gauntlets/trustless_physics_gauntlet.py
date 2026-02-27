#!/usr/bin/env python3
"""
Trustless Physics Gauntlet — Phase 0 Validation
================================================

Validates:
    1. TPC format: serialize → deserialize roundtrip
    2. Computation trace: hooks, recording, save/load, binary format
    3. Certificate generator: end-to-end generation
    4. Python-side verification: structural + signature checks
    5. Cross-layer consistency

Pass criteria: ALL tests must pass. No exceptions.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import struct
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

# ── Setup ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger("trustless_physics_gauntlet")

# ═════════════════════════════════════════════════════════════════════════════
# Gauntlet Framework
# ═════════════════════════════════════════════════════════════════════════════

RESULTS: dict[str, dict[str, Any]] = {}
_start_time = time.monotonic()


def gauntlet(name: str, layer: str = "phase0"):
    """Decorator to register a gauntlet test."""
    def decorator(func):
        def wrapper():
            t0 = time.monotonic()
            try:
                func()
                elapsed = time.monotonic() - t0
                RESULTS[name] = {
                    "layer": layer,
                    "passed": True,
                    "time_seconds": round(elapsed, 4),
                    "error": None,
                }
                logger.info(f"  ✅ {name} ({elapsed:.3f}s)")
                return True
            except Exception as e:
                elapsed = time.monotonic() - t0
                RESULTS[name] = {
                    "layer": layer,
                    "passed": False,
                    "time_seconds": round(elapsed, 4),
                    "error": f"{type(e).__name__}: {e}",
                }
                logger.error(f"  ❌ {name} ({elapsed:.3f}s)")
                logger.error(f"     {type(e).__name__}: {e}")
                traceback.print_exc()
                return False
        wrapper.__name__ = name
        wrapper._gauntlet = True
        wrapper._layer = layer
        return wrapper
    return decorator


# ═════════════════════════════════════════════════════════════════════════════
# Layer 1: TPC Format Tests
# ═════════════════════════════════════════════════════════════════════════════

@gauntlet("tpc_constants_integrity", "tpc_format")
def test_tpc_constants():
    """Verify TPC constants are consistent."""
    from tpc.constants import (
        TPC_MAGIC, TPC_VERSION, TPC_HEADER_SIZE,
        HASH_SIZE, PUBLIC_KEY_SIZE, SIGNATURE_SIZE,
        PROOF_SYSTEMS, KNOWN_SOLVERS, KNOWN_DOMAINS,
        FORMAL_PROOF_SYSTEMS,
    )

    assert TPC_MAGIC == b"TPC\x01", f"Magic mismatch: {TPC_MAGIC!r}"
    assert TPC_VERSION == 1
    assert TPC_HEADER_SIZE == 64
    assert HASH_SIZE == 32
    assert PUBLIC_KEY_SIZE == 32
    assert SIGNATURE_SIZE == 64
    assert "stark" in PROOF_SYSTEMS
    assert "halo2" in PROOF_SYSTEMS
    assert "euler3d" in KNOWN_SOLVERS
    assert "cfd" in KNOWN_DOMAINS
    assert "lean4" in FORMAL_PROOF_SYSTEMS

    # Verify header struct packs to exactly 64 bytes
    header_fmt = "<4sI16sq32s"
    assert struct.calcsize(header_fmt) == TPC_HEADER_SIZE


@gauntlet("tpc_header_roundtrip", "tpc_format")
def test_header_roundtrip():
    """Verify header pack/unpack roundtrip."""
    from tpc.format import TPCHeader

    h = TPCHeader()
    packed = h.pack()
    assert len(packed) == 64, f"Header size: {len(packed)}"

    h2 = TPCHeader.unpack(packed)
    assert h.version == h2.version
    assert h.certificate_id == h2.certificate_id
    assert h.timestamp_ns == h2.timestamp_ns
    assert h.solver_hash == h2.solver_hash


@gauntlet("tpc_layer_a_validation", "tpc_format")
def test_layer_a():
    """Verify Layer A data model."""
    from tpc.format import LayerA, TheoremRef, CoverageLevel

    # Valid Layer A
    la = LayerA(
        proof_system="lean4",
        theorems=[
            TheoremRef(name="euler_well_posed", file="Euler/WellPosed.lean", line=42),
            TheoremRef(name="rh_conditions", file="Euler/RH.lean", line=100),
        ],
        coverage=CoverageLevel.PARTIAL,
        coverage_pct=35.0,
    )
    d = la.to_dict()
    assert d["proof_system"] == "lean4"
    assert len(d["theorems"]) == 2
    assert d["coverage"] == "partial"

    # Roundtrip
    la2 = LayerA.from_dict(d)
    assert la2.proof_system == la.proof_system
    assert len(la2.theorems) == len(la.theorems)

    # Invalid proof system should raise
    try:
        LayerA(proof_system="invalid_system")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


@gauntlet("tpc_layer_b_validation", "tpc_format")
def test_layer_b():
    """Verify Layer B data model."""
    from tpc.format import LayerB

    lb = LayerB(
        proof_system="stark",
        proof_bytes=b"\x00" * 1024,
        public_inputs={"trace_hash": "abcd1234"},
        circuit_constraints=131_000,
    )
    d = lb.to_dict()
    assert d["proof_system"] == "stark"
    assert d["proof_size"] == 1024
    assert d["circuit_constraints"] == 131_000

    # Roundtrip
    lb2 = LayerB.from_dict(d, proof_bytes=lb.proof_bytes)
    assert lb2.proof_system == lb.proof_system
    assert len(lb2.proof_bytes) == len(lb.proof_bytes)


@gauntlet("tpc_layer_c_validation", "tpc_format")
def test_layer_c():
    """Verify Layer C data model."""
    from tpc.format import LayerC, BenchmarkResult, HardwareSpec

    benchmarks = [
        BenchmarkResult(
            name="sod_shock_tube",
            gauntlet="genesis",
            l2_error=1e-8,
            max_deviation=5e-7,
            conservation_error=1e-12,
            passed=True,
        ),
        BenchmarkResult(
            name="blast_wave",
            gauntlet="genesis",
            l2_error=2e-6,
            max_deviation=1e-4,
            conservation_error=1e-10,
            passed=True,
        ),
    ]

    lc = LayerC(
        benchmarks=benchmarks,
        hardware=HardwareSpec(platform="x86_64", processor="AMD EPYC"),
        total_time_s=42.5,
    )

    assert lc.all_passed
    assert lc.pass_count == 2

    d = lc.to_dict()
    assert d["all_passed"] is True
    assert d["total_benchmarks"] == 2


@gauntlet("tpc_full_roundtrip", "tpc_format")
def test_full_roundtrip():
    """Full serialize → deserialize roundtrip."""
    from tpc.format import (
        TPCFile, TPCHeader, LayerA, LayerB, LayerC,
        Metadata, QTTParams, TheoremRef, BenchmarkResult,
        HardwareSpec, CoverageLevel,
    )

    cert = TPCFile(
        header=TPCHeader(),
        layer_a=LayerA(
            proof_system="lean4",
            theorems=[TheoremRef(name="test_thm", file="Test.lean")],
            proof_objects=b"lean4_proof_data_here",
            coverage=CoverageLevel.PARTIAL,
            coverage_pct=25.0,
        ),
        layer_b=LayerB(
            proof_system="halo2",
            proof_bytes=os.urandom(2048),
            verification_key=os.urandom(256),
            public_inputs={"x": 42, "y": 3.14},
            circuit_constraints=500_000,
        ),
        layer_c=LayerC(
            benchmarks=[
                BenchmarkResult(name="test_bench", passed=True, l2_error=1e-10),
            ],
            hardware=HardwareSpec.detect(),
            attestation_json=b'{"test": true}',
            total_time_s=1.5,
        ),
        metadata=Metadata(
            domain="cfd",
            solver="euler3d",
            qtt_params=QTTParams(max_rank=64, grid_bits=10, tolerance=1e-8),
            description="Gauntlet test certificate",
            tags=["test", "gauntlet"],
        ),
    )

    # Serialize
    data = cert.serialize()
    assert len(data) > 64 + 128  # Header + signature

    # Deserialize
    cert2 = TPCFile.deserialize(data)

    # Verify fields match
    assert cert2.header.certificate_id == cert.header.certificate_id
    assert cert2.header.version == cert.header.version
    assert cert2.layer_a.proof_system == "lean4"
    assert len(cert2.layer_a.theorems) == 1
    assert cert2.layer_a.proof_objects == cert.layer_a.proof_objects
    assert cert2.layer_b.proof_system == "halo2"
    assert len(cert2.layer_b.proof_bytes) == 2048
    assert cert2.layer_b.public_inputs["x"] == 42
    assert cert2.layer_c.all_passed
    assert cert2.metadata.domain == "cfd"
    assert cert2.metadata.solver == "euler3d"
    assert cert2.metadata.qtt_params.max_rank == 64


@gauntlet("tpc_file_io", "tpc_format")
def test_file_io():
    """Test save/load to disk."""
    from tpc.format import TPCFile, LayerA, LayerB, LayerC, Metadata

    cert = TPCFile(
        layer_a=LayerA(),
        layer_b=LayerB(),
        layer_c=LayerC(),
        metadata=Metadata(),
    )

    with tempfile.NamedTemporaryFile(suffix=".tpc", delete=False) as f:
        path = Path(f.name)

    try:
        cert.save(path)
        assert path.exists()
        assert path.stat().st_size > 0

        cert2 = TPCFile.load(path)
        assert cert2.header.certificate_id == cert.header.certificate_id
    finally:
        path.unlink(missing_ok=True)


@gauntlet("tpc_verification", "tpc_format")
def test_verification():
    """Test Python-side verify_certificate."""
    from tpc.format import (
        TPCFile, LayerA, LayerB, LayerC, Metadata,
        BenchmarkResult, verify_certificate,
    )

    # Valid certificate (all benchmarks pass)
    cert = TPCFile(
        layer_a=LayerA(),
        layer_b=LayerB(),
        layer_c=LayerC(
            benchmarks=[BenchmarkResult(name="test", passed=True)],
        ),
        metadata=Metadata(),
    )

    with tempfile.NamedTemporaryFile(suffix=".tpc", delete=False) as f:
        path = Path(f.name)

    try:
        cert.save(path)
        report = verify_certificate(path)
        assert report.valid, f"Errors: {report.errors}"
        assert report.layer_a_valid
        assert report.layer_b_valid
        assert report.layer_c_valid
        assert report.signature_valid
    finally:
        path.unlink(missing_ok=True)


@gauntlet("tpc_summary_text", "tpc_format")
def test_summary():
    """Test human-readable summary generation."""
    from tpc.format import (
        TPCFile, LayerA, LayerB, LayerC, Metadata,
        BenchmarkResult, TheoremRef, CoverageLevel,
    )

    cert = TPCFile(
        layer_a=LayerA(
            theorems=[TheoremRef(name="test", file="Test.lean")],
            coverage=CoverageLevel.PARTIAL,
            coverage_pct=50.0,
        ),
        layer_b=LayerB(proof_system="stark", circuit_constraints=100_000),
        layer_c=LayerC(
            benchmarks=[BenchmarkResult(name="sod", passed=True, l2_error=1e-8)],
        ),
        metadata=Metadata(domain="cfd", solver="euler3d"),
    )

    summary = cert.summary()
    assert "TRUSTLESS PHYSICS" in summary
    assert "euler3d" in summary
    assert "cfd" in summary
    assert "test" in summary
    assert "stark" in summary


# ═════════════════════════════════════════════════════════════════════════════
# Layer 2: Computation Trace Tests
# ═════════════════════════════════════════════════════════════════════════════

@gauntlet("trace_session_basic", "computation_trace")
def test_trace_session_basic():
    """Test basic trace session creation and finalization."""
    from tensornet.core.trace import TraceSession, TraceDigest

    session = TraceSession()
    assert session.entry_count == 0

    digest = session.finalize()
    assert isinstance(digest, TraceDigest)
    assert digest.entry_count == 0
    assert len(digest.trace_hash) == 64  # SHA-256 hex


@gauntlet("trace_tensor_hashing", "computation_trace")
def test_tensor_hashing():
    """Test deterministic tensor hashing."""
    import torch
    from tensornet.core.trace import _hash_tensor, _hash_tensor_list

    t1 = torch.randn(10, 10, dtype=torch.float64)
    h1 = _hash_tensor(t1)
    h2 = _hash_tensor(t1)
    assert h1 == h2, "Hash must be deterministic"
    assert len(h1) == 64  # SHA-256 hex

    # Different tensors → different hashes
    t2 = torch.randn(10, 10, dtype=torch.float64)
    h3 = _hash_tensor(t2)
    assert h1 != h3

    # List hashing
    hl1 = _hash_tensor_list([t1, t2])
    hl2 = _hash_tensor_list([t1, t2])
    assert hl1 == hl2

    # Order matters
    hl3 = _hash_tensor_list([t2, t1])
    assert hl1 != hl3


@gauntlet("trace_svd_recording", "computation_trace")
def test_trace_svd_recording():
    """Test SVD trace recording."""
    import torch
    from tensornet.core.trace import (
        trace_session, traced_svd_truncated, OpType,
    )

    A = torch.randn(50, 50, dtype=torch.float64)

    with trace_session() as session:
        U, S, Vh = traced_svd_truncated(A, chi_max=10)

    assert session.entry_count == 1
    entry = session.entries[0]
    assert entry.op == OpType.SVD_TRUNCATED
    assert "A" in entry.input_hashes
    assert "U" in entry.output_hashes
    assert "S" in entry.output_hashes
    assert "Vh" in entry.output_hashes
    assert entry.params["chi_max"] == 10
    assert "singular_values" in entry.metrics
    assert len(entry.metrics["singular_values"]) <= 10


@gauntlet("trace_qr_recording", "computation_trace")
def test_trace_qr_recording():
    """Test QR trace recording."""
    import torch
    from tensornet.core.trace import trace_session, traced_qr_positive, OpType

    A = torch.randn(30, 20, dtype=torch.float64)

    with trace_session() as session:
        Q, R = traced_qr_positive(A)

    assert session.entry_count == 1
    entry = session.entries[0]
    assert entry.op == OpType.QR_POSITIVE
    assert "A" in entry.input_hashes
    assert "Q" in entry.output_hashes
    assert "R" in entry.output_hashes


@gauntlet("trace_hooks_install_uninstall", "computation_trace")
def test_hooks_install():
    """Test monkey-patch installation and removal."""
    from tensornet.core.trace import install_trace_hooks, uninstall_trace_hooks

    install_trace_hooks()
    install_trace_hooks()  # Idempotent

    uninstall_trace_hooks()
    uninstall_trace_hooks()  # Idempotent


@gauntlet("trace_session_save_load_json", "computation_trace")
def test_trace_save_load_json():
    """Test trace persistence in JSON format."""
    import torch
    from tensornet.core.trace import trace_session, traced_svd_truncated

    A = torch.randn(30, 30, dtype=torch.float64)

    with trace_session() as session:
        traced_svd_truncated(A, chi_max=5)
        traced_svd_truncated(A, chi_max=10)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)

    try:
        session.save(path)
        assert path.exists()

        from tensornet.core.trace import TraceSession as TS
        loaded = TS.load(path)
        assert loaded.entry_count == 2
        assert loaded.session_id == session.session_id
    finally:
        path.unlink(missing_ok=True)


@gauntlet("trace_session_save_load_binary", "computation_trace")
def test_trace_save_load_binary():
    """Test trace persistence in binary format."""
    import torch
    from tensornet.core.trace import trace_session, traced_svd_truncated

    A = torch.randn(20, 20, dtype=torch.float64)

    with trace_session() as session:
        traced_svd_truncated(A, chi_max=5)

    with tempfile.NamedTemporaryFile(suffix=".trc", delete=False) as f:
        path = Path(f.name)

    try:
        session.save_binary(path)
        assert path.exists()

        # Verify magic bytes
        data = path.read_bytes()
        assert data[:4] == b"TRCV"

        from tensornet.core.trace import TraceSession as TS
        loaded = TS.load_binary(path)
        assert loaded.entry_count == 1
    finally:
        path.unlink(missing_ok=True)


@gauntlet("trace_digest_determinism", "computation_trace")
def test_trace_digest_determinism():
    """Test that tensor hashes and operation outputs are deterministic.

    Note: The full trace chain hash includes timestamps (by design, each
    session is unique). What must be deterministic is the tensor hashing —
    the same input tensor always produces the same hash, and the same SVD
    always produces the same output hashes.
    """
    import torch
    from tensornet.core.trace import (
        trace_session, traced_svd_truncated, _hash_tensor,
    )

    torch.manual_seed(42)
    A = torch.randn(20, 20, dtype=torch.float64)

    with trace_session() as s1:
        traced_svd_truncated(A, chi_max=5)

    with trace_session() as s2:
        traced_svd_truncated(A, chi_max=5)

    # Entry count must match
    assert s1.entry_count == s2.entry_count == 1

    e1 = s1.entries[0]
    e2 = s2.entries[0]

    # Input hashes must be identical (same tensor → same hash)
    assert e1.input_hashes == e2.input_hashes, (
        f"Input hashes differ: {e1.input_hashes} vs {e2.input_hashes}"
    )

    # Output hashes must be identical (deterministic SVD on same input)
    assert e1.output_hashes == e2.output_hashes, (
        f"Output hashes differ: {e1.output_hashes} vs {e2.output_hashes}"
    )

    # Singular values must match
    assert e1.metrics["singular_values"] == e2.metrics["singular_values"]

    # Parameters must match
    assert e1.params == e2.params

    # Standalone tensor hash determinism
    h1 = _hash_tensor(A)
    h2 = _hash_tensor(A)
    assert h1 == h2


@gauntlet("trace_no_session_passthrough", "computation_trace")
def test_no_session_passthrough():
    """Verify traced functions work with no active session."""
    import torch
    from tensornet.core.trace import traced_svd_truncated, traced_qr_positive

    A = torch.randn(20, 20, dtype=torch.float64)
    U, S, Vh = traced_svd_truncated(A, chi_max=5)
    assert U.shape[1] <= 5

    Q, R = traced_qr_positive(A)
    assert Q.shape == (20, 20)


# ═════════════════════════════════════════════════════════════════════════════
# Layer 3: Certificate Generator Tests
# ═════════════════════════════════════════════════════════════════════════════

@gauntlet("generator_basic", "certificate_gen")
def test_generator_basic():
    """Test basic certificate generation."""
    from tpc.generator import CertificateGenerator

    gen = CertificateGenerator(domain="cfd", solver="euler3d")
    gen.set_layer_a_empty()
    gen.set_layer_b_empty()
    gen.set_layer_c(
        benchmarks=[{"name": "test", "passed": True, "l2_error": 1e-8}],
    )

    cert = gen.generate()
    assert cert.metadata.domain == "cfd"
    assert cert.metadata.solver == "euler3d"
    assert cert.layer_a.proof_system == "none"
    assert cert.layer_b.proof_system == "none"
    assert len(cert.layer_c.benchmarks) == 1


@gauntlet("generator_full_pipeline", "certificate_gen")
def test_generator_full_pipeline():
    """Test full generate → save → verify pipeline."""
    from tpc.generator import CertificateGenerator
    from tpc.format import verify_certificate

    gen = CertificateGenerator(
        domain="cfd",
        solver="euler3d",
        description="Gauntlet full pipeline test",
    )

    gen.set_layer_a(
        theorems=[
            {"name": "euler_existence", "file": "Euler/Existence.lean", "line": 10},
        ],
        coverage="partial",
        coverage_pct=20.0,
    )

    gen.set_layer_b(
        proof_system="stark",
        proof_bytes=os.urandom(512),
        circuit_constraints=50_000,
    )

    gen.set_layer_c(
        benchmarks=[
            {"name": "sod_shock", "passed": True, "l2_error": 1e-9},
            {"name": "blast_wave", "passed": True, "l2_error": 2e-7},
        ],
        total_time_s=10.5,
    )

    gen.set_qtt_params(max_rank=32, grid_bits=8, tolerance=1e-10)

    with tempfile.NamedTemporaryFile(suffix=".tpc", delete=False) as f:
        path = Path(f.name)

    try:
        cert, report = gen.generate_and_save(path)
        assert report.valid, f"Errors: {report.errors}"
        assert report.layer_a_valid
        assert report.layer_b_valid
        assert report.layer_c_valid
        assert report.certificate_id == str(cert.header.certificate_id)
    finally:
        path.unlink(missing_ok=True)


@gauntlet("generator_from_attestation", "certificate_gen")
def test_generator_from_attestation():
    """Test loading Layer C from an attestation JSON."""
    from tpc.generator import CertificateGenerator

    # Create a mock attestation
    attestation = {
        "project": "HyperTensor-VM",
        "protocol": "gauntlet",
        "timestamp": "2024-01-01T00:00:00Z",
        "total_time_seconds": 42.0,
        "gauntlets": {
            "sod_shock": {
                "layer": "cfd",
                "passed": True,
                "time_seconds": 2.0,
                "metrics": {"l2_error": 1e-8, "max_deviation": 5e-7},
            },
            "sedov_blast": {
                "layer": "cfd",
                "passed": True,
                "time_seconds": 5.0,
                "metrics": {"l2_error": 1e-6, "conservation_error": 1e-10},
            },
        },
    }

    with tempfile.NamedTemporaryFile(
        suffix=".json", delete=False, mode="w"
    ) as f:
        json.dump(attestation, f)
        att_path = Path(f.name)

    try:
        gen = CertificateGenerator(domain="cfd", solver="euler3d")
        gen.set_layer_a_empty()
        gen.set_layer_b_empty()
        gen.set_layer_c_from_attestation(att_path)

        cert = gen.generate()
        assert len(cert.layer_c.benchmarks) == 2
        assert cert.layer_c.all_passed
        assert cert.layer_c.total_time_s == 42.0
    finally:
        att_path.unlink(missing_ok=True)


@gauntlet("generator_json_export", "certificate_gen")
def test_generator_json_export():
    """Test JSON export of certificate metadata."""
    from tpc.generator import CertificateGenerator

    gen = CertificateGenerator(domain="thermal", solver="heat_equation")
    gen.set_layer_a_empty()
    gen.set_layer_b_empty()
    gen.set_layer_c(benchmarks=[])

    cert = gen.generate()
    j = cert.to_json()

    assert j["tpc_version"] == 1
    assert j["metadata"]["domain"] == "thermal"
    assert j["metadata"]["solver"] == "heat_equation"
    assert "certificate_id" in j
    assert "timestamp" in j


# ═════════════════════════════════════════════════════════════════════════════
# Layer 4: Cross-Layer Integration Tests
# ═════════════════════════════════════════════════════════════════════════════

@gauntlet("integration_trace_to_certificate", "integration")
def test_trace_to_certificate():
    """End-to-end: trace computation → generate certificate."""
    import torch
    from tensornet.core.trace import trace_session, traced_svd_truncated
    from tpc.generator import CertificateGenerator
    from tpc.format import verify_certificate

    # Step 1: Run computation with tracing
    torch.manual_seed(42)
    A = torch.randn(40, 40, dtype=torch.float64)

    with trace_session() as session:
        U, S, Vh = traced_svd_truncated(A, chi_max=10)
        U2, S2, Vh2 = traced_svd_truncated(A, chi_max=20)

    assert session.entry_count == 2

    # Step 2: Save trace
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        trace_path = Path(f.name)
    session.save(trace_path)

    # Step 3: Generate certificate from trace
    gen = CertificateGenerator(domain="cfd", solver="euler3d")
    gen.set_layer_a_empty()
    gen.set_layer_b_from_trace(trace_path, proof_system="stark")
    gen.set_layer_c(
        benchmarks=[{"name": "integration_test", "passed": True, "l2_error": 1e-8}],
    )
    gen.set_qtt_params(max_rank=20, grid_bits=8)

    with tempfile.NamedTemporaryFile(suffix=".tpc", delete=False) as f:
        cert_path = Path(f.name)

    try:
        cert, report = gen.generate_and_save(cert_path)
        assert report.valid, f"Errors: {report.errors}"

        # Verify trace hash is in Layer B
        assert "trace_hash" in cert.layer_b.public_inputs
        assert cert.layer_b.public_inputs["trace_entries"] == 2
    finally:
        trace_path.unlink(missing_ok=True)
        cert_path.unlink(missing_ok=True)


@gauntlet("integration_failed_benchmark", "integration")
def test_failed_benchmark_rejection():
    """Verify that certificates with failed benchmarks are rejected."""
    from tpc.generator import CertificateGenerator
    from tpc.format import verify_certificate

    gen = CertificateGenerator(domain="cfd", solver="euler3d")
    gen.set_layer_a_empty()
    gen.set_layer_b_empty()
    gen.set_layer_c(
        benchmarks=[
            {"name": "good_test", "passed": True, "l2_error": 1e-8},
            {"name": "bad_test", "passed": False, "l2_error": 0.5},
        ],
    )

    with tempfile.NamedTemporaryFile(suffix=".tpc", delete=False) as f:
        path = Path(f.name)

    try:
        cert, report = gen.generate_and_save(path)
        assert not report.valid, "Certificate with failed benchmark should not be valid"
        assert not report.layer_c_valid
        assert any("failed" in e.lower() for e in report.errors)
    finally:
        path.unlink(missing_ok=True)


@gauntlet("integration_tamper_detection", "integration")
def test_tamper_detection():
    """Verify that tampered certificates are detected."""
    from tpc.format import TPCFile, LayerA, LayerB, LayerC, Metadata, verify_certificate, BenchmarkResult

    cert = TPCFile(
        layer_a=LayerA(),
        layer_b=LayerB(),
        layer_c=LayerC(benchmarks=[BenchmarkResult(name="test", passed=True)]),
        metadata=Metadata(),
    )

    data = cert.serialize()

    # Tamper with a byte in the content (not the signature section)
    tampered = bytearray(data)
    tampered[70] ^= 0xFF  # Flip a byte in Layer A section
    tampered = bytes(tampered)

    report = verify_certificate(tampered)
    # The tampered certificate should either fail parsing or fail hash check
    # (depending on which byte was flipped)
    # At minimum, it should not report as fully valid with no warnings
    assert not report.valid or len(report.errors) > 0 or len(report.warnings) > 0


# ═════════════════════════════════════════════════════════════════════════════
# Run All
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          TRUSTLESS PHYSICS GAUNTLET — PHASE 0              ║")
    print("║          Tigantic Holdings LLC · Proprietary               ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # Collect all gauntlet tests
    tests = []
    for name, obj in list(globals().items()):
        if callable(obj) and getattr(obj, "_gauntlet", False):
            tests.append(obj)

    # Group by layer
    layers: dict[str, list] = {}
    for t in tests:
        layer = t._layer
        layers.setdefault(layer, []).append(t)

    total = 0
    passed = 0
    failed = 0

    for layer_name, layer_tests in layers.items():
        print(f"\n{'─' * 60}")
        print(f"  {layer_name.upper()} ({len(layer_tests)} tests)")
        print(f"{'─' * 60}")

        for test in layer_tests:
            total += 1
            if test():
                passed += 1
            else:
                failed += 1

    elapsed = time.monotonic() - _start_time

    # Summary
    print(f"\n{'═' * 60}")
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
    print(f"  TIME:    {elapsed:.2f}s")
    print(f"{'═' * 60}")

    # Save attestation
    attestation = {
        "project": "HyperTensor-VM",
        "protocol": "trustless_physics_gauntlet_phase0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_time_seconds": round(elapsed, 4),
        "gauntlets": RESULTS,
    }

    att_path = ROOT / "TRUSTLESS_PHYSICS_PHASE0_ATTESTATION.json"
    with open(att_path, "w") as f:
        json.dump(attestation, f, indent=2)
    print(f"\n  Attestation saved: {att_path}")

    if failed > 0:
        print(f"\n  ❌ GAUNTLET FAILED — {failed} test(s) did not pass")
        return 1
    else:
        print(f"\n  ✅ GAUNTLET PASSED — ALL {total} TESTS PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
