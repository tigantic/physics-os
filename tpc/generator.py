"""
Certificate Generator
=====================

High-level API for producing Trustless Physics Certificates (.tpc files).

Combines:
    - Layer A: Lean 4 proof references
    - Layer B: ZK proof from fluidelite-zk (via proof bridge)
    - Layer C: Physical fidelity benchmarks (gauntlet results)
    - Metadata: solver, domain, QTT parameters

Usage:
    gen = CertificateGenerator(domain="cfd", solver="euler3d")
    gen.set_layer_a(theorems=[...], coverage="partial")
    gen.set_layer_b_from_trace(trace_session, proof_system="stark")
    gen.set_layer_c_from_attestation("attestation.json")
    cert = gen.generate()
    cert.save("simulation.tpc")

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tpc.constants import (
    KNOWN_DOMAINS,
    KNOWN_SOLVERS,
    PROOF_SYSTEMS,
)
from tpc.format import (
    BenchmarkResult,
    CoverageLevel,
    HardwareSpec,
    LayerA,
    LayerB,
    LayerC,
    Metadata,
    QTTParams,
    TPCFile,
    TPCHeader,
    TPCSignature,
    TheoremRef,
    VerificationReport,
    verify_certificate,
)

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# Certificate Generator
# ═════════════════════════════════════════════════════════════════════════════


class CertificateGenerator:
    """
    High-level builder for Trustless Physics Certificates.

    Assembles all three layers + metadata into a signed .tpc file.
    """

    def __init__(
        self,
        domain: str = "cfd",
        solver: str = "euler3d",
        description: str = "",
    ) -> None:
        """
        Initialize generator.

        Args:
            domain: Physics domain (cfd, structural, thermal, ...).
            solver: Solver identifier (euler3d, navier_stokes, ...).
            description: Human-readable description.
        """
        self.domain = domain
        self.solver = solver
        self.description = description
        self.tags: list[str] = []
        self.extra: dict[str, Any] = {}

        # Layers (populated by set_layer_X methods)
        self._layer_a: LayerA | None = None
        self._layer_b: LayerB | None = None
        self._layer_c: LayerC | None = None
        self._qtt_params: QTTParams = QTTParams()
        self._solver_hash: bytes = b"\x00" * 32

    # ── Layer A — Mathematical Truth ─────────────────────────────────────

    def set_layer_a(
        self,
        theorems: list[dict[str, Any]] | list[TheoremRef] | None = None,
        proof_objects: bytes = b"",
        coverage: str = "partial",
        coverage_pct: float = 0.0,
        notes: str = "",
        proof_system: str = "lean4",
    ) -> CertificateGenerator:
        """
        Set Layer A (Mathematical Truth).

        Args:
            theorems: List of theorem references (dicts or TheoremRef objects).
            proof_objects: Serialized Lean 4 environment exports.
            coverage: Coverage level ("none", "partial", "full").
            coverage_pct: Coverage percentage (0-100).
            notes: Additional notes.
            proof_system: Formal proof system ("lean4", "coq", "isabelle").

        Returns:
            self (for chaining).
        """
        theorem_refs: list[TheoremRef] = []
        if theorems:
            for t in theorems:
                if isinstance(t, TheoremRef):
                    theorem_refs.append(t)
                elif isinstance(t, dict):
                    theorem_refs.append(TheoremRef.from_dict(t))
                else:
                    raise TypeError(f"Expected TheoremRef or dict, got {type(t)}")

        self._layer_a = LayerA(
            proof_system=proof_system,
            theorems=theorem_refs,
            proof_objects=proof_objects,
            coverage=CoverageLevel(coverage),
            coverage_pct=coverage_pct,
            notes=notes,
        )
        return self

    def set_layer_a_empty(self) -> CertificateGenerator:
        """Set Layer A to empty (no formal proofs yet)."""
        self._layer_a = LayerA(
            proof_system="none",
            coverage=CoverageLevel.NONE,
            notes="Formal proofs pending — Phase 1 deliverable.",
        )
        return self

    # ── Layer B — Computational Integrity ────────────────────────────────

    def set_layer_b(
        self,
        proof_system: str = "stark",
        proof_bytes: bytes = b"",
        verification_key: bytes = b"",
        public_inputs: dict[str, Any] | None = None,
        public_outputs: dict[str, Any] | None = None,
        proof_generation_time_s: float = 0.0,
        circuit_constraints: int = 0,
        prover_version: str = "",
    ) -> CertificateGenerator:
        """
        Set Layer B (Computational Integrity) directly.

        Args:
            proof_system: ZK proof system ("stark", "halo2", "groth16", "plonk").
            proof_bytes: Serialized ZK proof.
            verification_key: Verification key bytes.
            public_inputs: Dict of public inputs.
            public_outputs: Dict of public outputs.
            proof_generation_time_s: Time to generate proof.
            circuit_constraints: Number of constraints.
            prover_version: Prover version string.

        Returns:
            self (for chaining).
        """
        self._layer_b = LayerB(
            proof_system=proof_system,
            proof_bytes=proof_bytes,
            verification_key=verification_key,
            public_inputs=public_inputs or {},
            public_outputs=public_outputs or {},
            proof_generation_time_s=proof_generation_time_s,
            circuit_constraints=circuit_constraints,
            prover_version=prover_version,
        )
        return self

    def set_layer_b_from_trace(
        self,
        trace_path: str | Path,
        proof_system: str = "stark",
        proof_bridge_binary: str | None = None,
    ) -> CertificateGenerator:
        """
        Generate Layer B from a computation trace using the proof bridge.

        This invokes the Rust `trace-to-proof` binary to convert the trace
        to circuit inputs, then (in future phases) invokes the prover.

        Args:
            trace_path: Path to trace file (JSON or binary).
            proof_system: Target proof system.
            proof_bridge_binary: Path to trace-to-proof binary.

        Returns:
            self (for chaining).
        """
        trace_path = Path(trace_path)
        if not trace_path.exists():
            raise FileNotFoundError(f"Trace file not found: {trace_path}")

        # Read trace to get chain hash as public input
        trace_data = trace_path.read_text() if trace_path.suffix == ".json" else None
        trace_hash = ""
        entry_count = 0

        if trace_data:
            payload = json.loads(trace_data)
            digest = payload.get("digest", {})
            trace_hash = digest.get("trace_hash", "")
            entry_count = digest.get("entry_count", 0)

        # Set Layer B with trace metadata
        # In Phase 0, we record the trace commitment but don't generate
        # the actual ZK proof (that requires the full prover from Phase 1)
        self._layer_b = LayerB(
            proof_system=proof_system,
            public_inputs={
                "trace_hash": trace_hash,
                "trace_entries": entry_count,
                "trace_path": str(trace_path),
            },
            public_outputs={},
            proof_bytes=b"",  # Proof generation deferred to Phase 1
            prover_version="proof-bridge-0.1.0",
        )

        logger.info(
            f"Layer B configured from trace: hash={trace_hash[:16]}..., "
            f"entries={entry_count}, system={proof_system}"
        )
        return self

    def set_layer_b_empty(self) -> CertificateGenerator:
        """Set Layer B to empty (no ZK proof yet)."""
        self._layer_b = LayerB(
            proof_system="none",
            prover_version="none",
        )
        return self

    # ── Layer C — Physical Fidelity ──────────────────────────────────────

    def set_layer_c(
        self,
        benchmarks: list[dict[str, Any]] | list[BenchmarkResult] | None = None,
        hardware: HardwareSpec | None = None,
        git_commit: str = "",
        attestation_json: bytes = b"",
        total_time_s: float = 0.0,
    ) -> CertificateGenerator:
        """
        Set Layer C (Physical Fidelity) directly.

        Args:
            benchmarks: List of benchmark results.
            hardware: Hardware specification.
            git_commit: Git commit hash.
            attestation_json: Raw attestation JSON.
            total_time_s: Total time for all benchmarks.

        Returns:
            self (for chaining).
        """
        benchmark_objs: list[BenchmarkResult] = []
        if benchmarks:
            for b in benchmarks:
                if isinstance(b, BenchmarkResult):
                    benchmark_objs.append(b)
                elif isinstance(b, dict):
                    benchmark_objs.append(BenchmarkResult.from_dict(b))
                else:
                    raise TypeError(f"Expected BenchmarkResult or dict, got {type(b)}")

        self._layer_c = LayerC(
            benchmarks=benchmark_objs,
            hardware=hardware or HardwareSpec.detect(),
            git_commit=git_commit or self._detect_git_commit(),
            attestation_json=attestation_json,
            total_time_s=total_time_s,
        )
        return self

    def set_layer_c_from_attestation(
        self,
        attestation_path: str | Path,
    ) -> CertificateGenerator:
        """
        Populate Layer C from an existing attestation JSON file.

        Parses the gauntlet-format attestation and extracts benchmark results.

        Args:
            attestation_path: Path to attestation JSON.

        Returns:
            self (for chaining).
        """
        attestation_path = Path(attestation_path)
        if not attestation_path.exists():
            raise FileNotFoundError(f"Attestation not found: {attestation_path}")

        raw = attestation_path.read_bytes()
        data = json.loads(raw)

        benchmarks: list[BenchmarkResult] = []
        gauntlets = data.get("gauntlets", {})

        for name, gauntlet in gauntlets.items():
            metrics = gauntlet.get("metrics", {})
            benchmarks.append(BenchmarkResult(
                name=name,
                gauntlet=gauntlet.get("layer", ""),
                l2_error=float(metrics.get("l2_error", 0.0)),
                max_deviation=float(metrics.get("max_deviation", 0.0)),
                conservation_error=float(metrics.get("conservation_error", 0.0)),
                passed=bool(gauntlet.get("passed", False)),
                threshold_l2=float(metrics.get("threshold_l2", 0.0)),
                threshold_max=float(metrics.get("threshold_max", 0.0)),
                threshold_conservation=float(metrics.get("threshold_conservation", 0.0)),
                metrics=metrics,
            ))

        total_time = float(data.get("total_time_seconds", 0.0))

        self._layer_c = LayerC(
            benchmarks=benchmarks,
            hardware=HardwareSpec.detect(),
            git_commit=self._detect_git_commit(),
            attestation_json=raw,
            total_time_s=total_time,
        )

        passed = sum(1 for b in benchmarks if b.passed)
        logger.info(
            f"Layer C loaded from attestation: {passed}/{len(benchmarks)} benchmarks passed, "
            f"time={total_time:.1f}s"
        )
        return self

    # ── QTT Parameters ───────────────────────────────────────────────────

    def set_qtt_params(
        self,
        max_rank: int = 0,
        tolerance: float = 0.0,
        grid_bits: int = 0,
        num_sites: int = 0,
        physical_dim: int = 2,
        bond_dims: list[int] | None = None,
    ) -> CertificateGenerator:
        """Set QTT solver parameters."""
        self._qtt_params = QTTParams(
            max_rank=max_rank,
            tolerance=tolerance,
            grid_bits=grid_bits,
            num_sites=num_sites,
            physical_dim=physical_dim,
            bond_dims=bond_dims or [],
        )
        return self

    # ── Solver Hash ──────────────────────────────────────────────────────

    def set_solver_hash(self, solver_path: str | Path | None = None) -> CertificateGenerator:
        """
        Compute SHA-256 of the solver source/binary for the certificate header.

        Args:
            solver_path: Path to solver file. If None, hashes the current module.
        """
        if solver_path is not None:
            path = Path(solver_path)
            if path.is_file():
                self._solver_hash = hashlib.sha256(path.read_bytes()).digest()
            elif path.is_dir():
                # Hash all .py files in the directory
                h = hashlib.sha256()
                for f in sorted(path.rglob("*.py")):
                    h.update(f.read_bytes())
                self._solver_hash = h.digest()
        return self

    # ── Generation ───────────────────────────────────────────────────────

    def generate(self) -> TPCFile:
        """
        Generate the complete Trustless Physics Certificate.

        Returns:
            TPCFile ready to be saved or signed.

        Raises:
            ValueError: If required layers are not set.
        """
        # Default layers if not set
        if self._layer_a is None:
            self.set_layer_a_empty()
        if self._layer_b is None:
            self.set_layer_b_empty()
        if self._layer_c is None:
            self.set_layer_c(benchmarks=[], total_time_s=0.0)

        header = TPCHeader(solver_hash=self._solver_hash)

        metadata = Metadata(
            domain=self.domain,
            solver=self.solver,
            qtt_params=self._qtt_params,
            description=self.description,
            tags=self.tags,
            extra=self.extra,
        )

        cert = TPCFile(
            header=header,
            layer_a=self._layer_a,  # type: ignore[arg-type]
            layer_b=self._layer_b,  # type: ignore[arg-type]
            layer_c=self._layer_c,  # type: ignore[arg-type]
            metadata=metadata,
        )

        logger.info(
            f"Certificate generated: ID={header.certificate_id}, "
            f"domain={self.domain}, solver={self.solver}, "
            f"layer_a={self._layer_a.coverage.value}, "  # type: ignore[union-attr]
            f"layer_b={self._layer_b.proof_system}, "  # type: ignore[union-attr]
            f"layer_c={len(self._layer_c.benchmarks)} benchmarks"  # type: ignore[union-attr]
        )

        return cert

    def generate_and_save(
        self,
        path: str | Path,
        private_key: bytes | None = None,
    ) -> tuple[TPCFile, VerificationReport]:
        """
        Generate, optionally sign, save, and verify the certificate.

        Args:
            path: Output .tpc file path.
            private_key: 32-byte Ed25519 private key for signing. If None, unsigned.

        Returns:
            Tuple of (TPCFile, VerificationReport).
        """
        cert = self.generate()

        if private_key is not None:
            cert.sign(private_key)

        cert.save(path)

        report = verify_certificate(path)

        if report.valid:
            logger.info(f"Certificate verified: {path}")
        else:
            logger.error(
                f"Certificate verification FAILED: {path}\n"
                f"Errors: {report.errors}"
            )

        return cert, report

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _detect_git_commit() -> str:
        """Auto-detect current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return ""

    def __repr__(self) -> str:
        layers = []
        if self._layer_a is not None:
            layers.append(f"A({self._layer_a.coverage.value})")
        if self._layer_b is not None:
            layers.append(f"B({self._layer_b.proof_system})")
        if self._layer_c is not None:
            layers.append(f"C({len(self._layer_c.benchmarks)} benchmarks)")
        return (
            f"CertificateGenerator(domain={self.domain!r}, solver={self.solver!r}, "
            f"layers=[{', '.join(layers)}])"
        )
