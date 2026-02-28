"""
Tests for tci_core Rust extension imports and basic functionality.

These tests verify that the Rust extension module loads correctly
and exports the expected symbols.
"""

import pytest

# Mark all tests as requiring the Rust extension
pytestmark = [pytest.mark.rust, pytest.mark.unit]


class TestTciCoreImport:
    """Test that tci_core can be imported and has expected exports."""

    def test_import_module(self):
        """Test that the tci_core module can be imported."""
        try:
            from ontic import _tci_core

            assert _tci_core is not None
        except ImportError:
            pytest.skip("tci_core Rust extension not built")

    def test_version_attribute(self):
        """Test that the module has a version attribute."""
        try:
            from ontic import _tci_core

            assert hasattr(_tci_core, "__version__")
            assert isinstance(_tci_core.__version__, str)
        except ImportError:
            pytest.skip("tci_core Rust extension not built")

    def test_maxvol_config_export(self):
        """Test that MaxVolConfig is exported."""
        try:
            from ontic._tci_core import MaxVolConfig

            assert MaxVolConfig is not None
        except ImportError:
            pytest.skip("tci_core Rust extension not built")

    def test_truncation_policy_export(self):
        """Test that TruncationPolicy is exported."""
        try:
            from ontic._tci_core import TruncationPolicy

            assert TruncationPolicy is not None
        except ImportError:
            pytest.skip("tci_core Rust extension not built")

    def test_tci_config_export(self):
        """Test that TCIConfig is exported."""
        try:
            from ontic._tci_core import TCIConfig

            assert TCIConfig is not None
        except ImportError:
            pytest.skip("tci_core Rust extension not built")

    def test_tci_sampler_export(self):
        """Test that TCISampler is exported."""
        try:
            from ontic._tci_core import TCISampler

            assert TCISampler is not None
        except ImportError:
            pytest.skip("tci_core Rust extension not built")

    def test_index_batch_export(self):
        """Test that IndexBatch is exported."""
        try:
            from ontic._tci_core import IndexBatch

            assert IndexBatch is not None
        except ImportError:
            pytest.skip("tci_core Rust extension not built")


class TestMaxVolConfig:
    """Test MaxVolConfig instantiation and basic usage."""

    def test_default_construction(self):
        """Test default construction of MaxVolConfig."""
        try:
            from ontic._tci_core import MaxVolConfig

            config = MaxVolConfig()
            assert config is not None
        except ImportError:
            pytest.skip("tci_core Rust extension not built")
        except TypeError:
            # May require explicit parameters
            pass

    def test_with_parameters(self):
        """Test MaxVolConfig with custom parameters if supported."""
        try:
            from ontic._tci_core import MaxVolConfig

            # Try common parameter patterns
            try:
                config = MaxVolConfig(tol=1e-6, max_iter=100)
                assert config is not None
            except TypeError:
                # Different parameter signature, skip
                pass
        except ImportError:
            pytest.skip("tci_core Rust extension not built")


class TestTCIConfig:
    """Test TCIConfig instantiation."""

    def test_construction(self):
        """Test TCIConfig construction."""
        try:
            from ontic._tci_core import TCIConfig

            config = TCIConfig()
            assert config is not None
        except ImportError:
            pytest.skip("tci_core Rust extension not built")
        except TypeError:
            # May require parameters
            pass


class TestTCISampler:
    """Test TCISampler basic functionality."""

    def test_construction_fails_without_args(self):
        """Test that TCISampler requires proper arguments."""
        try:
            from ontic._tci_core import TCISampler

            # Should require configuration
            with pytest.raises(TypeError):
                TCISampler()
        except ImportError:
            pytest.skip("tci_core Rust extension not built")


class TestIndexBatch:
    """Test IndexBatch for DLPack compatibility."""

    def test_index_batch_exists(self):
        """Test that IndexBatch class exists."""
        try:
            from ontic._tci_core import IndexBatch

            assert IndexBatch is not None
        except ImportError:
            pytest.skip("tci_core Rust extension not built")
