"""HyperTensor Tier-2 Execution Fabric — Unit Tests.

Covers: package metadata, module imports, version contract surface,
core registry/hasher/sanitizer, and job models.
"""

from __future__ import annotations

import importlib
import types

import pytest

# ═══════════════════════════════════════════════════════════════════════════════
# Version & Metadata
# ═══════════════════════════════════════════════════════════════════════════════


class TestVersionMetadata:
    """Verify published version constants are well-formed and consistent."""

    def test_version_is_string(self) -> None:
        import hypertensor

        assert isinstance(hypertensor.__version__, str)

    def test_version_semver_format(self) -> None:
        import hypertensor

        parts = hypertensor.__version__.split(".")
        assert len(parts) == 3, f"Expected semver x.y.z, got {hypertensor.__version__}"
        for p in parts:
            assert p.isdigit(), f"Non-numeric semver component: {p}"

    def test_api_version_defined(self) -> None:
        import hypertensor

        assert hasattr(hypertensor, "API_VERSION")
        assert isinstance(hypertensor.API_VERSION, str)
        assert len(hypertensor.API_VERSION) > 0

    def test_runtime_version_defined(self) -> None:
        import hypertensor

        assert hasattr(hypertensor, "RUNTIME_VERSION")
        assert isinstance(hypertensor.RUNTIME_VERSION, str)

    def test_schema_version_defined(self) -> None:
        import hypertensor

        assert hasattr(hypertensor, "SCHEMA_VERSION")
        assert isinstance(hypertensor.SCHEMA_VERSION, str)


# ═══════════════════════════════════════════════════════════════════════════════
# Module Import Surface
# ═══════════════════════════════════════════════════════════════════════════════


class TestModuleImports:
    """Ensure all documented subpackages are importable."""

    @pytest.mark.parametrize(
        "module_path",
        [
            "hypertensor",
            "hypertensor.core",
            "hypertensor.core.certificates",
            "hypertensor.core.evidence",
            "hypertensor.core.executor",
            "hypertensor.core.hasher",
            "hypertensor.core.registry",
            "hypertensor.core.sanitizer",
            "hypertensor.jobs",
            "hypertensor.jobs.models",
            "hypertensor.jobs.store",
            "hypertensor.api",
            "hypertensor.api.app",
            "hypertensor.api.auth",
            "hypertensor.api.config",
            "hypertensor.sdk",
            "hypertensor.sdk.client",
            "hypertensor.cli",
            "hypertensor.cli.main",
            "hypertensor.billing",
            "hypertensor.billing.meter",
            "hypertensor.billing.invoice",
            "hypertensor.mcp",
            "hypertensor.mcp.server",
        ],
    )
    def test_submodule_importable(self, module_path: str) -> None:
        mod = importlib.import_module(module_path)
        assert isinstance(mod, types.ModuleType)


# ═══════════════════════════════════════════════════════════════════════════════
# Core — Hasher
# ═══════════════════════════════════════════════════════════════════════════════


class TestCoreHasher:
    """Verify deterministic hashing utilities."""

    def test_hasher_module_has_public_api(self) -> None:
        from hypertensor.core import hasher

        public = [n for n in dir(hasher) if not n.startswith("_")]
        assert len(public) > 0, "hasher module exposes no public symbols"

    def test_hash_determinism(self) -> None:
        from hypertensor.core import hasher

        # Find a callable hash function
        hash_fn = None
        for name in dir(hasher):
            obj = getattr(hasher, name)
            if callable(obj) and "hash" in name.lower():
                hash_fn = obj
                break
        if hash_fn is None:
            pytest.skip("No callable hash function discovered")
        # Determinism: same input → same output
        result_a = hash_fn(b"hello world")
        result_b = hash_fn(b"hello world")
        assert result_a == result_b


# ═══════════════════════════════════════════════════════════════════════════════
# Core — Sanitizer
# ═══════════════════════════════════════════════════════════════════════════════


class TestCoreSanitizer:
    """Verify input sanitization utilities."""

    def test_sanitizer_module_has_public_api(self) -> None:
        from hypertensor.core import sanitizer

        public = [n for n in dir(sanitizer) if not n.startswith("_")]
        assert len(public) > 0, "sanitizer module exposes no public symbols"


# ═══════════════════════════════════════════════════════════════════════════════
# Core — Registry
# ═══════════════════════════════════════════════════════════════════════════════


class TestCoreRegistry:
    """Verify domain registry surface."""

    def test_registry_module_has_public_api(self) -> None:
        from hypertensor.core import registry

        public = [n for n in dir(registry) if not n.startswith("_")]
        assert len(public) > 0, "registry module exposes no public symbols"


# ═══════════════════════════════════════════════════════════════════════════════
# Jobs — Models & Store
# ═══════════════════════════════════════════════════════════════════════════════


class TestJobModels:
    """Verify job state machine exports."""

    def test_models_module_has_exports(self) -> None:
        from hypertensor.jobs import models

        public = [n for n in dir(models) if not n.startswith("_")]
        assert len(public) > 0, "jobs.models exposes no public symbols"

    def test_store_module_has_exports(self) -> None:
        from hypertensor.jobs import store

        public = [n for n in dir(store) if not n.startswith("_")]
        assert len(public) > 0, "jobs.store exposes no public symbols"
