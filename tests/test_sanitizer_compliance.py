"""Sanitizer IP boundary compliance tests.

Verifies that ``sanitize_result()`` provably contains none of the
25+ forbidden field categories (§20.4), that the execution fence
prevents ``to_dense()`` inside VM dispatch, and that the two-world
telemetry split keeps private metrics out of public outputs.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from physics_os.core.sanitizer import (
    FORBIDDEN_FIELDS,
    SanitizerIPViolation,
    sanitize_result,
    _assert_no_forbidden_fields,
    _collect_all_keys,
)
from ontic.engine.vm.execution_fence import (
    DenseInDispatchError,
    vm_dispatch_context,
    assert_not_in_dispatch,
)
from ontic.engine.vm.telemetry import (
    DeterminismTier,
    PublicMetrics,
    PrivateMetrics,
    ProgramTelemetry,
    StepTelemetry,
    TelemetryCollector,
    compute_config_hash,
    detect_device_class,
)
from ontic.engine.vm.qtt_tensor import QTTTensor

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────


def _make_qtt(n_bits: int = 8) -> QTTTensor:
    """Create a simple QTT tensor for testing."""
    cores = [np.ones((1, 2, 1), dtype=np.float64) * 0.1 for _ in range(n_bits)]
    return QTTTensor(cores=cores, bits_per_dim=(n_bits,), domain=((0.0, 1.0),))


def _make_execution_result(n_bits: int = 8, n_steps: int = 5):
    """Create a mock ExecutionResult."""
    qtt = _make_qtt(n_bits)

    telemetry = ProgramTelemetry(
        domain="burgers",
        domain_label="Burgers equation",
        n_bits=n_bits,
        n_dims=1,
        n_steps=n_steps,
        n_fields=1,
        dt=0.001,
        total_wall_time_s=1.23,
        chi_max=4,
        chi_mean=2.5,
        chi_final=3,
        compression_ratio_final=85.0,
        invariant_error=1e-12,
        invariant_name="momentum",
        invariant_initial=1.0,
        invariant_final=1.0 - 1e-12,
        scaling_class="A",
        saturation_rate=0.0,
        total_truncations=50,
        max_rank_policy=64,
        n_instructions=10,
        ir_opcodes_used=["LOAD_FIELD", "GRAD", "TRUNCATE", "STORE_FIELD"],
    )

    result = MagicMock()
    result.telemetry = telemetry
    result.fields = {"u": qtt}
    return result


# ─────────────────────────────────────────────────────────────────────
# Test: FORBIDDEN_FIELDS constant is comprehensive
# ─────────────────────────────────────────────────────────────────────


class TestForbiddenFields:
    """Verify the FORBIDDEN_FIELDS constant covers all §20.4 categories."""

    def test_forbidden_fields_is_frozenset(self) -> None:
        assert isinstance(FORBIDDEN_FIELDS, frozenset)

    def test_minimum_coverage(self) -> None:
        """Must cover at minimum the critical categories."""
        critical = {
            "tt_cores", "cores", "bond_dim", "chi_max", "chi_mean",
            "svd_spectra", "singular_values", "compression_ratio",
            "ir_opcodes", "scaling_class", "saturation_rate",
            "rank_distribution", "rank_history", "private",
        }
        missing = critical - FORBIDDEN_FIELDS
        assert not missing, f"Missing critical forbidden fields: {missing}"

    def test_at_least_25_categories(self) -> None:
        """§20.4 lists 25 forbidden categories."""
        assert len(FORBIDDEN_FIELDS) >= 25


# ─────────────────────────────────────────────────────────────────────
# Test: sanitize_result strips all forbidden fields
# ─────────────────────────────────────────────────────────────────────


class TestSanitizeResult:
    """Verify sanitize_result outputs contain NO forbidden fields."""

    def test_sanitized_output_has_no_forbidden_keys(self) -> None:
        result = _make_execution_result()
        sanitized = sanitize_result(result, domain_key="burgers")

        all_keys = _collect_all_keys(sanitized)
        leaked = {k for k in all_keys if k.lower() in FORBIDDEN_FIELDS}
        assert not leaked, (
            f"Forbidden fields detected in sanitized output: {sorted(leaked)}"
        )

    def test_sanitized_output_has_expected_sections(self) -> None:
        result = _make_execution_result()
        sanitized = sanitize_result(result, domain_key="burgers")

        assert "grid" in sanitized
        assert "performance" in sanitized
        assert "conservation" in sanitized

    def test_private_telemetry_absent(self) -> None:
        """Private metrics must NEVER appear."""
        result = _make_execution_result()
        sanitized = sanitize_result(result, domain_key="burgers")

        all_keys = _collect_all_keys(sanitized)
        private_keys = {"chi_max", "chi_mean", "chi_final",
                        "compression_ratio", "scaling_class",
                        "saturation_rate", "ir_opcodes_used",
                        "private", "private_metrics", "rank_history"}
        leaked = all_keys & private_keys
        assert not leaked, f"Private metrics leaked: {sorted(leaked)}"


# ─────────────────────────────────────────────────────────────────────
# Test: _assert_no_forbidden_fields raises on injection
# ─────────────────────────────────────────────────────────────────────


class TestAssertNoForbidden:
    """Verify the enforcement function catches injected forbidden keys."""

    def test_clean_dict_passes(self) -> None:
        _assert_no_forbidden_fields({
            "grid": {"dimensions": 1},
            "performance": {"wall_time_s": 1.0},
        })

    def test_forbidden_key_at_top_level_raises(self) -> None:
        with pytest.raises(SanitizerIPViolation, match="FORBIDDEN"):
            _assert_no_forbidden_fields({
                "grid": {},
                "chi_max": 42,
            })

    def test_forbidden_key_nested_raises(self) -> None:
        with pytest.raises(SanitizerIPViolation, match="FORBIDDEN"):
            _assert_no_forbidden_fields({
                "inner": {"cores": [1, 2, 3]},
            })

    def test_forbidden_key_in_list_of_dicts_raises(self) -> None:
        with pytest.raises(SanitizerIPViolation, match="FORBIDDEN"):
            _assert_no_forbidden_fields({
                "items": [{"name": "ok"}, {"svd_spectra": [0.1]}],
            })


# ─────────────────────────────────────────────────────────────────────
# Test: Execution Fence (to_dense guard)
# ─────────────────────────────────────────────────────────────────────


class TestExecutionFence:
    """Verify to_dense() is blocked inside VM dispatch context."""

    def test_to_dense_outside_dispatch_works(self) -> None:
        qtt = _make_qtt(4)
        dense = qtt.to_dense()
        assert dense.shape == (16,)

    def test_to_dense_inside_dispatch_raises(self) -> None:
        qtt = _make_qtt(4)
        with vm_dispatch_context():
            with pytest.raises(DenseInDispatchError):
                qtt.to_dense()

    def test_assert_not_in_dispatch_outside_is_fine(self) -> None:
        assert_not_in_dispatch()  # should not raise

    def test_assert_not_in_dispatch_inside_raises(self) -> None:
        with vm_dispatch_context():
            with pytest.raises(DenseInDispatchError):
                assert_not_in_dispatch()

    def test_dispatch_context_nesting(self) -> None:
        """Nested dispatch contexts must still block to_dense."""
        qtt = _make_qtt(4)
        with vm_dispatch_context():
            with vm_dispatch_context():
                with pytest.raises(DenseInDispatchError):
                    qtt.to_dense()
            # Still in outer context
            with pytest.raises(DenseInDispatchError):
                qtt.to_dense()
        # Outside all contexts — should work
        dense = qtt.to_dense()
        assert dense.shape == (16,)


# ─────────────────────────────────────────────────────────────────────
# Test: Two-World Telemetry Split
# ─────────────────────────────────────────────────────────────────────


class TestTelemetrySplit:
    """Verify PublicMetrics / PrivateMetrics separation."""

    def test_public_metrics_has_no_rank_fields(self) -> None:
        pm = PublicMetrics()
        d = pm.to_dict()
        forbidden_in_public = {"chi_max", "chi_mean", "chi_final",
                               "compression_ratio", "scaling_class",
                               "ir_opcodes_used"}
        assert not (set(d.keys()) & forbidden_in_public)

    def test_private_metrics_contains_rank_fields(self) -> None:
        pm = PrivateMetrics(chi_max=10, chi_mean=5.0, chi_final=8)
        d = pm.to_dict()
        assert d["chi_max"] == 10
        assert d["chi_mean"] == 5.0

    def test_program_telemetry_populates_both(self) -> None:
        """After finalize, both public and private are populated."""
        collector = TelemetryCollector(
            domain="burgers",
            domain_label="Burgers",
            n_bits=8,
            n_dims=1,
            n_steps=2,
            n_fields=1,
            dt=0.001,
            n_instructions=5,
            ir_opcodes=["LOAD_FIELD", "GRAD"],
            max_rank_policy=64,
            invariant_name="momentum",
        )
        collector.begin_program()

        qtt = _make_qtt(8)
        for step in range(2):
            collector.begin_step(step)
            collector.record_field("u", qtt)
            collector.record_invariant("momentum", 1.0)
            collector.end_step(n_truncations=3, peak_rank=2)

        result = collector.finalize()

        # Public has conservation and timing
        assert result.public.invariant_name == "momentum"
        assert result.public.wall_time_s > 0

        # Private has rank info
        assert result.private.chi_max >= 0
        assert result.private.max_rank_policy == 64
        assert result.private.ir_opcodes_used == ["LOAD_FIELD", "GRAD"]


# ─────────────────────────────────────────────────────────────────────
# Test: Determinism Tier
# ─────────────────────────────────────────────────────────────────────


class TestDeterminismTier:
    """Verify determinism tier enum and config hash."""

    def test_tier_values(self) -> None:
        assert DeterminismTier.BITWISE.value == "bitwise"
        assert DeterminismTier.REPRODUCIBLE.value == "reproducible"
        assert DeterminismTier.PHYSICALLY_EQUIVALENT.value == "physically_equivalent"

    def test_config_hash_deterministic(self) -> None:
        cfg = {"domain": "burgers", "n_bits": 10, "dt": 0.001}
        h1 = compute_config_hash(cfg)
        h2 = compute_config_hash(cfg)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex length

    def test_config_hash_differs_on_change(self) -> None:
        h1 = compute_config_hash({"domain": "burgers", "n_bits": 10})
        h2 = compute_config_hash({"domain": "burgers", "n_bits": 11})
        assert h1 != h2

    def test_detect_device_class_returns_string(self) -> None:
        dc = detect_device_class()
        assert dc in ("cpu", "gpu_consumer", "gpu_datacenter")
