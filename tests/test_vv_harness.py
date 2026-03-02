"""Tests for Phase D: Benchmark Harness + Evidence Pipeline.

Tests cover:
- Registry loading and querying
- Gate evaluation logic
- Claim generation (all 7 registered tags)
- Public scorecard generation (ScorecardPublicV1 schema compliance)
- Private pack generation
- VVHarness orchestration
- Evidence pipeline extension (4 new claim tags)
- Schema validation against scorecard_public_v1.schema.json

References
----------
- Universal_Discretization_Execution.md §5.1, §5.2
- CLAIM_REGISTRY.md
- Platform Spec §13.3, §20.4
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from ontic.platform.vv.harness import (
    BenchmarkResult,
    GateResult,
    GateVerdict,
    QoIValue,
    VVHarness,
    evaluate_gates,
    generate_claims_from_qoi,
    generate_private_pack,
    generate_public_scorecard,
    get_benchmark_spec,
    list_benchmark_ids,
    load_registry,
    make_qtt_qoi_extractor,
    make_qtt_run_fn,
)
from physics_os.core.evidence import generate_claims, generate_validation_report


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent
    / "ontic"
    / "platform"
    / "vv"
    / "registry.yaml"
)

SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent
    / "contracts"
    / "v1"
    / "schemas"
    / "scorecard_public_v1.schema.json"
)


@pytest.fixture
def registry() -> dict[str, Any]:
    """Load the benchmark registry."""
    return load_registry(REGISTRY_PATH)


@pytest.fixture
def mock_exec_result() -> MagicMock:
    """Create a mock VM execution result with telemetry."""
    result = MagicMock()
    result.success = True

    telem = MagicMock()
    telem.total_wall_time_s = 1.5
    telem.n_steps = 100
    telem.invariant_name = "L2_norm"
    telem.invariant_initial = 1.0
    telem.invariant_final = 0.9999999
    telem.invariant_error = 1e-7

    pub = MagicMock()
    pub.config_hash = "abc123def456"
    pub.determinism_tier = MagicMock()
    pub.determinism_tier.value = "reproducible"
    telem.public = pub

    priv = MagicMock()
    priv.to_dict.return_value = {
        "chi_max": 32,
        "chi_mean": 16.5,
        "compression_ratio_final": 128.0,
        "scaling_class": "B",
    }
    telem.private = priv

    result.telemetry = telem
    result.fields = {}
    return result


@pytest.fixture
def sample_benchmark_result() -> BenchmarkResult:
    """Create a sample benchmark result for scorecard tests."""
    return BenchmarkResult(
        benchmark_id="V010_MMS_GRADIENT_1D",
        domain_key="advection_diffusion",
        status="succeeded",
        started_utc="2025-01-01T00:00:00+00:00",
        finished_utc="2025-01-01T00:00:01+00:00",
        wall_seconds=1.0,
        determinism_tier="reproducible",
        qoi_values=[
            QoIValue(name="L2_error_dudx", value=1.5e-4, units="1"),
            QoIValue(
                name="observed_order_L2_error_dudx",
                value=2.13,
                units="1",
            ),
        ],
        gate_results=[
            GateResult(
                gate_type="observed_order_min",
                qoi_name="L2_error_dudx",
                threshold=2.0,
                observed=2.13,
                verdict=GateVerdict.PASS,
                detail="order 2.130 >= 2.0",
            ),
        ],
        claims=[
            {
                "tag": "STABILITY",
                "claim": "Completed",
                "witness": {"wall_time_s": 1.0, "completed": True},
                "satisfied": True,
            },
            {
                "tag": "CONVERGENCE",
                "claim": "Grid convergence verified",
                "witness": {"qoi": "L2_error_dudx", "observed_order": 2.13},
                "satisfied": True,
            },
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Registry Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegistryLoading:
    """Tests for benchmark registry loading and querying."""

    def test_load_registry_exists(self, registry: dict[str, Any]) -> None:
        """Registry YAML loads successfully."""
        assert registry is not None
        assert "version" in registry
        assert registry["version"] == "1.0"

    def test_registry_has_benchmarks(self, registry: dict[str, Any]) -> None:
        """Registry contains benchmark definitions."""
        benchmarks = registry.get("benchmarks", [])
        assert len(benchmarks) >= 12  # 4 verification + 8 CFD

    def test_registry_has_global_defaults(self, registry: dict[str, Any]) -> None:
        """Registry contains global default settings."""
        defaults = registry.get("global_defaults", {})
        assert defaults.get("determinism_tier_required") == "reproducible"
        assert "convergence" in defaults
        assert "gates" in defaults

    def test_list_benchmark_ids(self, registry: dict[str, Any]) -> None:
        """Can list all benchmark IDs."""
        ids = list_benchmark_ids(registry)
        assert "V010_MMS_GRADIENT_1D" in ids
        assert "C250_LID_DRIVEN_CAVITY_2D" in ids

    def test_get_benchmark_spec(self, registry: dict[str, Any]) -> None:
        """Can retrieve a benchmark by ID."""
        spec = get_benchmark_spec("V010_MMS_GRADIENT_1D", registry)
        assert spec["category"] == "verification"
        assert spec["dimensions"] == 1
        assert "refinement_plan" in spec

    def test_get_benchmark_spec_not_found(self, registry: dict[str, Any]) -> None:
        """KeyError raised for nonexistent benchmark ID."""
        with pytest.raises(KeyError, match="NONEXISTENT"):
            get_benchmark_spec("NONEXISTENT", registry)

    def test_benchmark_structure_complete(self, registry: dict[str, Any]) -> None:
        """Each benchmark has required fields: id, category, domain_key, qoi."""
        for b in registry["benchmarks"]:
            assert "id" in b, f"Missing 'id' in benchmark"
            assert "category" in b, f"Missing 'category' in {b.get('id')}"
            assert "domain_key" in b, f"Missing 'domain_key' in {b.get('id')}"
            assert "qoi" in b, f"Missing 'qoi' in {b.get('id')}"

    def test_registry_load_not_found(self) -> None:
        """FileNotFoundError for missing registry file."""
        with pytest.raises(FileNotFoundError):
            load_registry(Path("/nonexistent/registry.yaml"))


# ═══════════════════════════════════════════════════════════════════════════════
# Gate Evaluation Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestGateEvaluation:
    """Tests for gate pass/fail logic."""

    def test_absolute_max_pass(self) -> None:
        """Gate passes when observed <= threshold."""
        gates = [{"type": "absolute_max", "qoi": "L2_error", "max": 1e-3}]
        qoi_map = {"L2_error": 5e-4}
        results = evaluate_gates(gates, qoi_map)
        assert len(results) == 1
        assert results[0].verdict == GateVerdict.PASS

    def test_absolute_max_fail(self) -> None:
        """Gate fails when observed > threshold."""
        gates = [{"type": "absolute_max", "qoi": "L2_error", "max": 1e-3}]
        qoi_map = {"L2_error": 5e-2}
        results = evaluate_gates(gates, qoi_map)
        assert results[0].verdict == GateVerdict.FAIL

    def test_absolute_max_skip_missing(self) -> None:
        """Gate skipped when QoI not available."""
        gates = [{"type": "absolute_max", "qoi": "missing_qoi", "max": 1e-3}]
        qoi_map = {}
        results = evaluate_gates(gates, qoi_map)
        assert results[0].verdict == GateVerdict.SKIP

    def test_observed_order_pass(self) -> None:
        """Convergence order gate passes when order >= min_order."""
        gates = [{"type": "observed_order_min", "qoi": "L2_error", "min_order": 2.0}]
        qoi_map = {"observed_order_L2_error": 2.13}
        results = evaluate_gates(gates, qoi_map)
        assert results[0].verdict == GateVerdict.PASS

    def test_observed_order_fail(self) -> None:
        """Convergence order gate fails when order < min_order."""
        gates = [{"type": "observed_order_min", "qoi": "L2_error", "min_order": 2.0}]
        qoi_map = {"observed_order_L2_error": 1.5}
        results = evaluate_gates(gates, qoi_map)
        assert results[0].verdict == GateVerdict.FAIL

    def test_boundedness_pass(self) -> None:
        """Boundedness gate passes when field min > 0."""
        gates = [{"type": "boundedness", "require_positive": ["rho", "p"]}]
        qoi_map = {"min_rho": 0.1, "min_p": 100.0}
        results = evaluate_gates(gates, qoi_map)
        assert all(r.verdict == GateVerdict.PASS for r in results)

    def test_boundedness_fail(self) -> None:
        """Boundedness gate fails when field min <= 0."""
        gates = [{"type": "boundedness", "require_positive": ["rho"]}]
        qoi_map = {"min_rho": -0.01}
        results = evaluate_gates(gates, qoi_map)
        assert results[0].verdict == GateVerdict.FAIL

    def test_unknown_gate_type_skipped(self) -> None:
        """Unknown gate type is skipped gracefully."""
        gates = [{"type": "unknown_gate", "qoi": "something"}]
        results = evaluate_gates(gates, {})
        assert results[0].verdict == GateVerdict.SKIP

    def test_multiple_gates(self) -> None:
        """Multiple gates evaluated correctly."""
        gates = [
            {"type": "absolute_max", "qoi": "L2_error", "max": 1e-3},
            {"type": "observed_order_min", "qoi": "L2_error", "min_order": 2.0},
        ]
        qoi_map = {"L2_error": 5e-4, "observed_order_L2_error": 2.5}
        results = evaluate_gates(gates, qoi_map)
        assert len(results) == 2
        assert all(r.verdict == GateVerdict.PASS for r in results)


# ═══════════════════════════════════════════════════════════════════════════════
# Claim Generation Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestClaimGeneration:
    """Tests for claim-witness pair generation (all 7 tags)."""

    def test_stability_claim(self, mock_exec_result: MagicMock) -> None:
        """STABILITY claim generated from execution result."""
        claims = generate_claims_from_qoi(mock_exec_result, [], {})
        stability = [c for c in claims if c["tag"] == "STABILITY"]
        assert len(stability) == 1
        assert stability[0]["satisfied"] is True
        assert stability[0]["witness"]["completed"] is True

    def test_bound_claim(self, mock_exec_result: MagicMock) -> None:
        """BOUND claim generated from QoI values."""
        qois = [QoIValue(name="L2_error", value=1e-4)]
        claims = generate_claims_from_qoi(mock_exec_result, qois, {})
        bound = [c for c in claims if c["tag"] == "BOUND"]
        assert len(bound) == 1
        assert bound[0]["satisfied"] is True

    def test_conservation_claim(self, mock_exec_result: MagicMock) -> None:
        """CONSERVATION claim generated from telemetry invariants."""
        claims = generate_claims_from_qoi(mock_exec_result, [], {})
        conservation = [c for c in claims if c["tag"] == "CONSERVATION"]
        assert len(conservation) == 1
        assert conservation[0]["satisfied"] is True
        assert conservation[0]["witness"]["error_value"] == pytest.approx(1e-7)

    def test_convergence_claim(self, mock_exec_result: MagicMock) -> None:
        """CONVERGENCE claim generated from observed_order QoIs."""
        qois = [QoIValue(name="observed_order_L2_error", value=2.13)]
        spec = {"gates": [{"type": "observed_order_min", "qoi": "L2_error", "min_order": 2.0}]}
        claims = generate_claims_from_qoi(mock_exec_result, qois, spec)
        conv = [c for c in claims if c["tag"] == "CONVERGENCE"]
        assert len(conv) == 1
        assert conv[0]["satisfied"] is True
        assert conv[0]["witness"]["observed_order"] == 2.13

    def test_cfl_satisfied_claim(self, mock_exec_result: MagicMock) -> None:
        """CFL_SATISFIED claim generated from max_cfl QoI."""
        qois = [QoIValue(name="max_cfl", value=0.45)]
        claims = generate_claims_from_qoi(mock_exec_result, qois, {})
        cfl = [c for c in claims if c["tag"] == "CFL_SATISFIED"]
        assert len(cfl) == 1
        assert cfl[0]["satisfied"] is True
        assert cfl[0]["witness"]["max_cfl"] == 0.45

    def test_energy_bound_claim(self, mock_exec_result: MagicMock) -> None:
        """ENERGY_BOUND claim generated from energy QoI."""
        qois = [QoIValue(name="kinetic_energy", value=42.0)]
        claims = generate_claims_from_qoi(mock_exec_result, qois, {})
        energy = [c for c in claims if c["tag"] == "ENERGY_BOUND"]
        assert len(energy) == 1
        assert energy[0]["satisfied"] is True

    def test_reproducibility_claim(self, mock_exec_result: MagicMock) -> None:
        """REPRODUCIBILITY claim generated from determinism metadata."""
        claims = generate_claims_from_qoi(mock_exec_result, [], {})
        repro = [c for c in claims if c["tag"] == "REPRODUCIBILITY"]
        assert len(repro) == 1
        assert repro[0]["satisfied"] is True
        assert "abc123" in repro[0]["witness"]["config_hash"]


# ═══════════════════════════════════════════════════════════════════════════════
# Proof Pack Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestProofPacks:
    """Tests for two-tier proof pack generation."""

    def test_public_scorecard_structure(
        self, sample_benchmark_result: BenchmarkResult
    ) -> None:
        """Public scorecard has required fields per ScorecardPublicV1 schema."""
        scorecard = generate_public_scorecard(sample_benchmark_result)
        required = [
            "schema_version", "job_id", "domain_key", "status",
            "timestamps", "determinism", "evidence", "qoi", "performance",
        ]
        for key in required:
            assert key in scorecard, f"Missing required field: {key}"

    def test_public_scorecard_no_forbidden_fields(
        self, sample_benchmark_result: BenchmarkResult
    ) -> None:
        """Public scorecard must not contain any forbidden internal fields."""
        scorecard = generate_public_scorecard(sample_benchmark_result)
        scorecard_str = json.dumps(scorecard)
        forbidden_substrings = [
            "tt_core", "bond_dim", "svd_spectr", "compiler_ir",
            "opcode_trace", "rank_distrib", "chi_max", "chi_mean",
            "compression_ratio", "scaling_class", "truncation",
        ]
        for forbidden in forbidden_substrings:
            assert forbidden not in scorecard_str.lower(), (
                f"Forbidden field '{forbidden}' found in public scorecard"
            )

    def test_public_scorecard_schema_version(
        self, sample_benchmark_result: BenchmarkResult
    ) -> None:
        """Schema version matches pattern ^1\\.[0-9]+$."""
        scorecard = generate_public_scorecard(sample_benchmark_result)
        import re
        assert re.match(r"^1\.[0-9]+$", scorecard["schema_version"])

    def test_public_scorecard_determinism_tier(
        self, sample_benchmark_result: BenchmarkResult
    ) -> None:
        """Determinism tier is a valid enum value."""
        scorecard = generate_public_scorecard(sample_benchmark_result)
        valid_tiers = {"bitwise", "reproducible", "physically_equivalent"}
        assert scorecard["determinism"]["tier"] in valid_tiers

    def test_private_pack_contains_telemetry(
        self, sample_benchmark_result: BenchmarkResult
    ) -> None:
        """Private pack contains internal telemetry data."""
        sample_benchmark_result.private_telemetry = {
            "chi_max": 64,
            "compression_ratio_final": 256.0,
        }
        pack = generate_private_pack(sample_benchmark_result)
        assert pack["private_telemetry"]["chi_max"] == 64
        assert pack["benchmark_id"] == "V010_MMS_GRADIENT_1D"

    def test_public_scorecard_has_qoi_values(
        self, sample_benchmark_result: BenchmarkResult
    ) -> None:
        """Public scorecard QoI array matches input."""
        scorecard = generate_public_scorecard(sample_benchmark_result)
        assert len(scorecard["qoi"]) == 2
        names = {q["name"] for q in scorecard["qoi"]}
        assert "L2_error_dudx" in names

    def test_public_scorecard_convergence_section(
        self, sample_benchmark_result: BenchmarkResult
    ) -> None:
        """Convergence section included when summary is provided."""
        sample_benchmark_result.convergence_summary = "Order 2.13 observed"
        scorecard = generate_public_scorecard(sample_benchmark_result)
        assert "convergence" in scorecard
        assert scorecard["convergence"]["performed"] is True

    def test_public_scorecard_json_serializable(
        self, sample_benchmark_result: BenchmarkResult
    ) -> None:
        """Public scorecard is fully JSON-serializable."""
        scorecard = generate_public_scorecard(sample_benchmark_result)
        serialized = json.dumps(scorecard)
        assert isinstance(serialized, str)
        roundtrip = json.loads(serialized)
        assert roundtrip["domain_key"] == "advection_diffusion"


# ═══════════════════════════════════════════════════════════════════════════════
# Evidence Pipeline Extension Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvidencePipelineExtension:
    """Tests for the 4 new claim tags in physics_os.core.evidence."""

    def test_convergence_claim_in_evidence(self) -> None:
        """CONVERGENCE claim generated from convergence data in sanitized result."""
        result = {
            "performance": {"wall_time_s": 1.0, "time_steps": 100},
            "convergence": {
                "qoi": "L2_error_dudx",
                "observed_order": 2.13,
                "required_order": 2.0,
                "levels": 3,
            },
        }
        claims = generate_claims(result, "advection_diffusion")
        conv = [c for c in claims if c["tag"] == "CONVERGENCE"]
        assert len(conv) == 1
        assert conv[0]["satisfied"] is True
        assert conv[0]["witness"]["observed_order"] == 2.13

    def test_convergence_claim_fails_when_order_low(self) -> None:
        """CONVERGENCE claim unsatisfied when observed order < required."""
        result = {
            "performance": {"wall_time_s": 1.0},
            "convergence": {
                "qoi": "L2_error",
                "observed_order": 0.8,
                "required_order": 2.0,
            },
        }
        claims = generate_claims(result, "burgers")
        conv = [c for c in claims if c["tag"] == "CONVERGENCE"]
        assert len(conv) == 1
        assert conv[0]["satisfied"] is False

    def test_reproducibility_claim_in_evidence(self) -> None:
        """REPRODUCIBILITY claim generated from reproducibility data."""
        result = {
            "performance": {"wall_time_s": 1.0},
            "reproducibility": {
                "config_hash": "abc123def456",
                "determinism_tier": "reproducible",
                "seed": 42,
            },
        }
        claims = generate_claims(result, "burgers")
        repro = [c for c in claims if c["tag"] == "REPRODUCIBILITY"]
        assert len(repro) == 1
        assert repro[0]["satisfied"] is True

    def test_energy_bound_claim_in_evidence(self) -> None:
        """ENERGY_BOUND claim generated from energy_bound data."""
        result = {
            "performance": {"wall_time_s": 1.0},
            "energy_bound": {
                "quantity": "kinetic_energy",
                "value": 42.0,
                "threshold": 1e15,
            },
        }
        claims = generate_claims(result, "navier_stokes_2d")
        energy = [c for c in claims if c["tag"] == "ENERGY_BOUND"]
        assert len(energy) == 1
        assert energy[0]["satisfied"] is True

    def test_energy_bound_claim_fails(self) -> None:
        """ENERGY_BOUND claim unsatisfied when energy exceeds threshold."""
        result = {
            "performance": {"wall_time_s": 1.0},
            "energy_bound": {
                "quantity": "total_energy",
                "value": 2e15,
                "threshold": 1e15,
            },
        }
        claims = generate_claims(result, "navier_stokes_2d")
        energy = [c for c in claims if c["tag"] == "ENERGY_BOUND"]
        assert len(energy) == 1
        assert energy[0]["satisfied"] is False

    def test_cfl_satisfied_claim_in_evidence(self) -> None:
        """CFL_SATISFIED claim generated from cfl data."""
        result = {
            "performance": {"wall_time_s": 1.0},
            "cfl": {"max_cfl": 0.5, "cfl_limit": 1.0},
        }
        claims = generate_claims(result, "burgers")
        cfl = [c for c in claims if c["tag"] == "CFL_SATISFIED"]
        assert len(cfl) == 1
        assert cfl[0]["satisfied"] is True

    def test_cfl_satisfied_claim_fails(self) -> None:
        """CFL_SATISFIED claim unsatisfied when CFL exceeds limit."""
        result = {
            "performance": {"wall_time_s": 1.0},
            "cfl": {"max_cfl": 1.5, "cfl_limit": 1.0},
        }
        claims = generate_claims(result, "burgers")
        cfl = [c for c in claims if c["tag"] == "CFL_SATISFIED"]
        assert len(cfl) == 1
        assert cfl[0]["satisfied"] is False

    def test_existing_claims_still_work(self) -> None:
        """Existing CONSERVATION, STABILITY, BOUND claims still generated."""
        result = {
            "conservation": {
                "quantity": "L2_norm",
                "initial_value": 1.0,
                "final_value": 0.999,
                "status": "conserved",
                "error_value": 1e-3,
                "tier_threshold": 1e-2,
            },
            "performance": {"wall_time_s": 1.0, "time_steps": 100},
            "fields": {
                "u": {"values": [0.1, 0.5, -0.3]},
            },
        }
        claims = generate_claims(result, "burgers")
        tags = {c["tag"] for c in claims}
        assert "CONSERVATION" in tags
        assert "STABILITY" in tags
        assert "BOUND" in tags


# ═══════════════════════════════════════════════════════════════════════════════
# VVHarness Orchestration Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestVVHarness:
    """Tests for the VVHarness orchestration class."""

    def test_harness_runs_benchmark(self, registry: dict[str, Any]) -> None:
        """Harness can execute a benchmark with mock run_fn."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.telemetry = MagicMock()
        mock_result.telemetry.total_wall_time_s = 0.5
        mock_result.telemetry.n_steps = 50
        mock_result.telemetry.invariant_name = ""
        mock_result.telemetry.invariant_initial = 0.0

        def run_fn(spec: dict[str, Any]) -> Any:
            return mock_result

        def qoi_extractor(exec_result: Any, spec: dict[str, Any]) -> list[QoIValue]:
            return [QoIValue(name="L2_error_dudx", value=1e-4)]

        harness = VVHarness(
            run_fn=run_fn,
            qoi_extractor=qoi_extractor,
            registry_path=REGISTRY_PATH,
        )
        result = harness.run_benchmark("V010_MMS_GRADIENT_1D", registry)
        assert result.status == "succeeded"
        assert result.benchmark_id == "V010_MMS_GRADIENT_1D"
        assert len(result.qoi_values) >= 1

    def test_harness_handles_failure(self, registry: dict[str, Any]) -> None:
        """Harness captures benchmark execution failures gracefully."""

        def run_fn(spec: dict[str, Any]) -> Any:
            raise RuntimeError("Simulation diverged")

        def qoi_extractor(exec_result: Any, spec: dict[str, Any]) -> list[QoIValue]:
            return []

        harness = VVHarness(
            run_fn=run_fn,
            qoi_extractor=qoi_extractor,
            registry_path=REGISTRY_PATH,
        )
        result = harness.run_benchmark("V010_MMS_GRADIENT_1D", registry)
        assert result.status == "failed"
        assert "diverged" in result.error_message

    def test_harness_run_all_with_filter(self, registry: dict[str, Any]) -> None:
        """Harness can run all benchmarks filtered by category."""
        call_count = 0

        def run_fn(spec: dict[str, Any]) -> Any:
            nonlocal call_count
            call_count += 1
            mock = MagicMock()
            mock.success = True
            mock.telemetry = MagicMock()
            mock.telemetry.total_wall_time_s = 0.1
            mock.telemetry.n_steps = 10
            mock.telemetry.invariant_name = ""
            mock.telemetry.invariant_initial = 0.0
            return mock

        def qoi_extractor(exec_result: Any, spec: dict[str, Any]) -> list[QoIValue]:
            return [QoIValue(name="wall_time", value=0.1)]

        harness = VVHarness(
            run_fn=run_fn,
            qoi_extractor=qoi_extractor,
            registry_path=REGISTRY_PATH,
        )
        results = harness.run_all(registry, category="verification")
        # Should only run verification benchmarks
        verification_count = sum(
            1 for b in registry["benchmarks"] if b["category"] == "verification"
        )
        assert len(results) == verification_count
        assert call_count == verification_count

    def test_harness_generate_report(self, registry: dict[str, Any]) -> None:
        """Harness generates a summary report from results."""
        results = [
            BenchmarkResult(
                benchmark_id="V010",
                domain_key="test",
                status="succeeded",
                gate_results=[
                    GateResult("test", "qoi", 1.0, 0.5, GateVerdict.PASS),
                ],
            ),
            BenchmarkResult(
                benchmark_id="V020",
                domain_key="test",
                status="failed",
                error_message="boom",
            ),
        ]
        harness = VVHarness(
            run_fn=lambda s: None,
            qoi_extractor=lambda r, s: [],
            registry_path=REGISTRY_PATH,
        )
        report = harness.generate_report(results)
        assert report["total_benchmarks"] == 2
        assert report["passed"] == 1
        assert report["failed"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Schema File Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSchemaFile:
    """Tests for the scorecard JSON schema file."""

    def test_schema_file_exists(self) -> None:
        """Scorecard schema file exists at expected path."""
        assert SCHEMA_PATH.exists(), f"Schema not found: {SCHEMA_PATH}"

    def test_schema_is_valid_json(self) -> None:
        """Schema file is valid JSON."""
        with open(SCHEMA_PATH) as f:
            schema = json.load(f)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"

    def test_schema_has_required_properties(self) -> None:
        """Schema defines the required top-level properties."""
        with open(SCHEMA_PATH) as f:
            schema = json.load(f)
        required = schema.get("required", [])
        expected = [
            "schema_version", "job_id", "domain_key", "status",
            "timestamps", "determinism", "evidence", "qoi", "performance",
        ]
        for prop in expected:
            assert prop in required, f"Missing required property: {prop}"

    def test_schema_evidence_claims_include_new_tags(self) -> None:
        """Schema evidence.claims enum includes all 7 registered tags."""
        with open(SCHEMA_PATH) as f:
            schema = json.load(f)
        claims_enum = (
            schema["properties"]["evidence"]["properties"]["claims"]["items"]["enum"]
        )
        expected_tags = [
            "CONSERVATION", "STABILITY", "BOUND",
            "CONVERGENCE", "REPRODUCIBILITY", "ENERGY_BOUND", "CFL_SATISFIED",
        ]
        for tag in expected_tags:
            assert tag in claims_enum, f"Missing claim tag in schema: {tag}"

    def test_schema_no_forbidden_properties(self) -> None:
        """Schema does not define any forbidden internal fields."""
        with open(SCHEMA_PATH) as f:
            schema_text = f.read()
        forbidden = [
            "tt_core", "bond_dim", "svd_spectr", "compiler_ir",
            "opcode_trace", "rank_distrib",
        ]
        for f in forbidden:
            assert f not in schema_text.lower(), (
                f"Forbidden field '{f}' found in scorecard schema"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# QTT VM Adapter Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestQTTAdapters:
    """Tests for QTT VM convenience adapter functions."""

    def test_make_qtt_run_fn(self) -> None:
        """QTT run_fn adapter calls compiler and runtime correctly."""
        mock_runtime = MagicMock()
        mock_program = MagicMock()
        mock_compiler = MagicMock()
        mock_compiler.compile.return_value = mock_program

        run_fn = make_qtt_run_fn(
            runtime=mock_runtime,
            compiler_factory=lambda spec: mock_compiler,
        )
        result = run_fn({"id": "test"})
        mock_compiler.compile.assert_called_once()
        mock_runtime.execute.assert_called_once_with(mock_program)

    def test_make_qtt_qoi_extractor_basic(self) -> None:
        """QTT QoI extractor captures wall time from telemetry."""
        mock_result = MagicMock()
        mock_result.telemetry = MagicMock()
        mock_result.telemetry.total_wall_time_s = 2.5
        mock_result.telemetry.invariant_name = ""
        mock_result.telemetry.invariant_error = 0.0

        extractor = make_qtt_qoi_extractor()
        qois = extractor(mock_result, {})
        names = {q.name for q in qois}
        assert "wall_time_s" in names

    def test_make_qtt_qoi_extractor_custom(self) -> None:
        """QTT QoI extractor with custom extraction function."""

        def custom_fn(exec_result: Any, spec: dict[str, Any]) -> list[QoIValue]:
            return [QoIValue(name="custom_metric", value=42.0)]

        extractor = make_qtt_qoi_extractor(custom_extractors={"custom": custom_fn})
        mock_result = MagicMock()
        mock_result.telemetry = MagicMock()
        mock_result.telemetry.total_wall_time_s = 1.0
        mock_result.telemetry.invariant_name = ""

        qois = extractor(mock_result, {})
        names = {q.name for q in qois}
        assert "custom_metric" in names
        assert "wall_time_s" in names


# ═══════════════════════════════════════════════════════════════════════════════
# BenchmarkResult Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestBenchmarkResult:
    """Tests for BenchmarkResult properties."""

    def test_all_gates_passed_true(self) -> None:
        """all_gates_passed is True when all gates pass."""
        result = BenchmarkResult(
            benchmark_id="test",
            domain_key="test",
            gate_results=[
                GateResult("a", "q", 1.0, 0.5, GateVerdict.PASS),
                GateResult("b", "q", 1.0, 0.5, GateVerdict.PASS),
                GateResult("c", "q", 1.0, 0.0, GateVerdict.SKIP),
            ],
        )
        assert result.all_gates_passed is True

    def test_all_gates_passed_false(self) -> None:
        """all_gates_passed is False when any gate fails."""
        result = BenchmarkResult(
            benchmark_id="test",
            domain_key="test",
            gate_results=[
                GateResult("a", "q", 1.0, 0.5, GateVerdict.PASS),
                GateResult("b", "q", 1.0, 2.0, GateVerdict.FAIL),
            ],
        )
        assert result.all_gates_passed is False

    def test_empty_gates_pass(self) -> None:
        """all_gates_passed is True with no gates."""
        result = BenchmarkResult(benchmark_id="test", domain_key="test")
        assert result.all_gates_passed is True
