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
# PYTEST MARKERS
# =============================================================================
def pytest_configure(config):
    """Register custom markers for test categorization."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test (may be slow)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (>5 seconds)"
    )
    config.addinivalue_line(
        "markers", "physics: mark test as a physics validation test"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "rust: mark test as requiring Rust TCI extension"
    )


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
