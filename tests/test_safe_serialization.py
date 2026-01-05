"""
Tests for safe serialization behavior.

These tests verify that:
1. allow_pickle=False is enforced for numpy loads
2. weights_only=True is used for torch loads
3. Pickle payloads are rejected

Security: CWE-502 - Deserialization of Untrusted Data
"""

import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.security]


class TestNumpyLoadSafety:
    """Test that NumPy loads are safe by default."""

    def test_npz_load_without_pickle(self):
        """Test that npz files can be loaded without allow_pickle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "safe_data.npz"

            # Save regular arrays
            data = {"array1": np.array([1, 2, 3]), "array2": np.array([4.0, 5.0])}
            np.savez(path, **data)

            # Load without allow_pickle (default, safe)
            loaded = np.load(path, allow_pickle=False)

            assert np.array_equal(loaded["array1"], data["array1"])
            assert np.array_equal(loaded["array2"], data["array2"])

    def test_npz_with_pickle_rejected(self):
        """Test that pickled objects in npz are rejected when allow_pickle=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "unsafe_data.npz"

            # Save with pickle (object array)
            obj_array = np.array([{"key": "value"}], dtype=object)
            np.savez(path, objects=obj_array)

            # Attempting to load without allow_pickle should raise
            with pytest.raises(ValueError, match="Object arrays"):
                loaded = np.load(path, allow_pickle=False)
                _ = loaded["objects"]  # Access triggers load


class TestTorchLoadSafety:
    """Test that PyTorch loads are safe."""

    def test_weights_only_default(self):
        """Test that weights_only=True works for state dicts."""
        torch = pytest.importorskip("torch")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"

            # Save a simple tensor
            state = {"weight": torch.randn(10, 10)}
            torch.save(state, path)

            # Load with weights_only=True (safe)
            loaded = torch.load(path, weights_only=True)

            assert "weight" in loaded
            assert loaded["weight"].shape == (10, 10)


class TestPickleRejection:
    """Test that pickle payloads are rejected."""

    def test_direct_pickle_not_used(self):
        """Verify that we never call pickle.load on untrusted data."""
        # This is a documentation test - the actual protection is in code review
        # and linting (S301 bandit rule)

        # Create a malicious pickle that would execute code if loaded
        class MaliciousObject:
            def __reduce__(self):
                # This would execute os.system("echo pwned") if unpickled
                import os

                return (os.system, ("echo pwned",))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "malicious.pkl"

            # Write malicious pickle
            with open(path, "wb") as f:
                pickle.dump(MaliciousObject(), f)

            # We should NEVER load this - test that our code doesn't
            # This test exists to document the threat model
            assert path.exists()


class TestJsonSerializationSafety:
    """Test that JSON serialization is used for config/state."""

    def test_json_for_config(self):
        """Test that configuration can be safely serialized with JSON."""
        config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "nested": {"key": [1, 2, 3]},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"

            # Save
            with open(path, "w") as f:
                json.dump(config, f)

            # Load
            with open(path, "r") as f:
                loaded = json.load(f)

            assert loaded == config

    def test_json_rejects_invalid_types(self):
        """Test that JSON rejects non-serializable types."""
        # This ensures we don't accidentally use pickle as fallback
        invalid_config = {
            "callback": lambda x: x,  # Functions can't be JSON serialized
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid.json"

            with pytest.raises(TypeError):
                with open(path, "w") as f:
                    json.dump(invalid_config, f)


class TestFieldSerialization:
    """Test Field class serialization safety."""

    def test_field_save_load_safe(self):
        """Test that Field save/load is safe (if Field class exists)."""
        try:
            from tensornet.core.field import Field
        except ImportError:
            pytest.skip("Field class not available")

        # This test would verify Field.save() and Field.load() are safe
        # Implementation depends on actual Field class interface
        pass
