"""
Shared pytest fixtures and configuration for HyperTensor test suite.

Provides:
    - Deterministic random seeding for reproducible tests
    - Common test fixtures for tensornet components
    - Test markers for categorization

Usage:
    Fixtures are automatically available to all tests in the tests/ directory.
    No need to import - pytest discovers conftest.py automatically.
"""

import os
import random
from typing import Generator

import numpy as np
import pytest

# =============================================================================
# DETERMINISTIC SEEDING
# =============================================================================
DEFAULT_SEED = 42


@pytest.fixture(autouse=True)
def deterministic_seed(request) -> Generator[int, None, None]:
    """
    Automatically seed all random number generators for deterministic tests.

    This fixture runs before every test to ensure reproducibility.
    The seed can be overridden via the HYPERTENSOR_TEST_SEED environment variable.

    Yields:
        The seed value used for this test.
    """
    # Allow override via environment variable
    seed = int(os.environ.get("HYPERTENSOR_TEST_SEED", DEFAULT_SEED))

    # Seed Python's random
    random.seed(seed)

    # Seed NumPy
    np.random.seed(seed)

    # Seed PyTorch if available
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Ensure deterministic algorithms where possible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    yield seed

    # No cleanup needed - next test will reseed


@pytest.fixture
def random_state() -> np.random.RandomState:
    """
    Provide an isolated RandomState for tests that need explicit control.

    Returns:
        A seeded numpy RandomState instance.
    """
    return np.random.RandomState(DEFAULT_SEED)


# =============================================================================
# PYTEST MARKERS (V&V Taxonomy per HYPERTENSOR_VV_FRAMEWORK.md)
# =============================================================================
def pytest_configure(config):
    """Register custom markers for test categorization.

    Aligned with ASME V&V 10-2019 test classification taxonomy.
    """
    # --- Core Test Categories ---
    config.addinivalue_line(
        "markers", "unit: Unit tests - function-level correctness (<100ms)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests - component interaction (<10s)"
    )
    config.addinivalue_line("markers", "slow: Long-running tests (>5 seconds)")

    # --- V&V Categories (ASME V&V 10-2019) ---
    config.addinivalue_line(
        "markers", "benchmark: Known-solution validation - Tier 1/2 benchmarks (<60s)"
    )
    config.addinivalue_line(
        "markers",
        "mms: Method of Manufactured Solutions - discretization verification (<60s)",
    )
    config.addinivalue_line(
        "markers",
        "conservation: Conservation law verification - mass/momentum/energy (<60s)",
    )
    config.addinivalue_line(
        "markers", "convergence: Order of accuracy verification (<5min)"
    )
    config.addinivalue_line(
        "markers", "regression: Regression prevention - catch breakage (<60s)"
    )
    config.addinivalue_line("markers", "stress: Large-scale stress tests (>5min)")

    # --- Domain-Specific ---
    config.addinivalue_line(
        "markers", "physics: Physics validation test (CFD, quantum, etc.)"
    )
    config.addinivalue_line("markers", "performance: Speed/memory benchmarks (<5min)")

    # --- Security ---
    config.addinivalue_line(
        "markers", "security: Security validation test (SBOM, license, audit)"
    )

    # --- Hardware Requirements ---
    config.addinivalue_line("markers", "gpu: Requires GPU (CUDA)")
    config.addinivalue_line("markers", "rust: Requires Rust TCI extension")


# =============================================================================
# COMMON FIXTURES
# =============================================================================
@pytest.fixture
def temp_dir(tmp_path):
    """
    Provide a temporary directory for test file I/O.

    The directory is automatically cleaned up after the test.

    Returns:
        pathlib.Path to a temporary directory.
    """
    return tmp_path


@pytest.fixture
def sample_tensor_2d(random_state) -> np.ndarray:
    """
    Provide a sample 2D tensor for testing.

    Returns:
        A (64, 64) float32 array with values in [0, 1].
    """
    return random_state.rand(64, 64).astype(np.float32)


@pytest.fixture
def sample_tensor_3d(random_state) -> np.ndarray:
    """
    Provide a sample 3D tensor for testing.

    Returns:
        A (32, 32, 32) float32 array with values in [0, 1].
    """
    return random_state.rand(32, 32, 32).astype(np.float32)


@pytest.fixture
def sample_vector_field_2d(random_state) -> tuple:
    """
    Provide a sample 2D vector field (u, v components).

    Returns:
        Tuple of (u, v) arrays, each (64, 64) float32.
    """
    u = random_state.randn(64, 64).astype(np.float32)
    v = random_state.randn(64, 64).astype(np.float32)
    return u, v


# =============================================================================
# SKIP CONDITIONS
# =============================================================================
@pytest.fixture
def requires_torch():
    """Skip test if PyTorch is not available."""
    pytest.importorskip("torch")


@pytest.fixture
def requires_cuda():
    """Skip test if CUDA is not available."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.fixture
def requires_rust_tci():
    """Skip test if Rust TCI extension is not available."""
    try:
        import tci_core
    except ImportError:
        pytest.skip("Rust TCI extension (tci_core) not available")
