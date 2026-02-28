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
        import physics_os

        assert isinstance(physics_os.__version__, str)

    def test_version_semver_format(self) -> None:
        import physics_os

        parts = physics_os.__version__.split(".")
        assert len(parts) == 3, f"Expected semver x.y.z, got {physics_os.__version__}"
        for p in parts:
            assert p.isdigit(), f"Non-numeric semver component: {p}"

    def test_api_version_defined(self) -> None:
        import physics_os

        assert hasattr(physics_os, "API_VERSION")
        assert isinstance(physics_os.API_VERSION, str)
        assert len(physics_os.API_VERSION) > 0

    def test_runtime_version_defined(self) -> None:
        import physics_os

        assert hasattr(physics_os, "RUNTIME_VERSION")
        assert isinstance(physics_os.RUNTIME_VERSION, str)

    def test_schema_version_defined(self) -> None:
        import physics_os

        assert hasattr(physics_os, "SCHEMA_VERSION")
        assert isinstance(physics_os.SCHEMA_VERSION, str)


# ═══════════════════════════════════════════════════════════════════════════════
# Module Import Surface
# ═══════════════════════════════════════════════════════════════════════════════


class TestModuleImports:
    """Ensure all documented subpackages are importable."""

    @pytest.mark.parametrize(
        "module_path",
        [
            "physics_os",
            "physics_os.core",
            "physics_os.core.certificates",
            "physics_os.core.evidence",
            "physics_os.core.executor",
            "physics_os.core.hasher",
            "physics_os.core.registry",
            "physics_os.core.sanitizer",
            "physics_os.jobs",
            "physics_os.jobs.models",
            "physics_os.jobs.store",
            "physics_os.api",
            "physics_os.api.app",
            "physics_os.api.auth",
            "physics_os.api.config",
            "physics_os.sdk",
            "physics_os.sdk.client",
            "physics_os.cli",
            "physics_os.cli.main",
            "physics_os.billing",
            "physics_os.billing.meter",
            "physics_os.billing.invoice",
            "physics_os.mcp",
            "physics_os.mcp.server",
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
        from physics_os.core import hasher

        public = [n for n in dir(hasher) if not n.startswith("_")]
        assert len(public) > 0, "hasher module exposes no public symbols"

    def test_hash_determinism(self) -> None:
        from physics_os.core import hasher

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
        from physics_os.core import sanitizer

        public = [n for n in dir(sanitizer) if not n.startswith("_")]
        assert len(public) > 0, "sanitizer module exposes no public symbols"


# ═══════════════════════════════════════════════════════════════════════════════
# Core — Registry
# ═══════════════════════════════════════════════════════════════════════════════


class TestCoreRegistry:
    """Verify domain registry surface."""

    def test_registry_module_has_public_api(self) -> None:
        from physics_os.core import registry

        public = [n for n in dir(registry) if not n.startswith("_")]
        assert len(public) > 0, "registry module exposes no public symbols"


# ═══════════════════════════════════════════════════════════════════════════════
# Jobs — Models & Store
# ═══════════════════════════════════════════════════════════════════════════════


class TestJobModels:
    """Verify job state machine exports."""

    def test_models_module_has_exports(self) -> None:
        from physics_os.jobs import models

        public = [n for n in dir(models) if not n.startswith("_")]
        assert len(public) > 0, "jobs.models exposes no public symbols"

    def test_store_module_has_exports(self) -> None:
        from physics_os.jobs import store

        public = [n for n in dir(store) if not n.startswith("_")]
        assert len(public) > 0, "jobs.store exposes no public symbols"
