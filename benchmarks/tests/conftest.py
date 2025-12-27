"""
Shared fixtures and configuration for benchmark tests.
"""

import pytest


@pytest.fixture
def small_grid_size():
    """Small grid size for quick benchmark tests."""
    return 64


@pytest.fixture
def medium_grid_size():
    """Medium grid size for moderate benchmark tests."""
    return 256


@pytest.fixture
def benchmark_tolerance():
    """Standard tolerance for benchmark comparisons."""
    return 1e-6


@pytest.fixture
def max_benchmark_rank():
    """Maximum rank allowed in TCI/QTT benchmarks."""
    return 64
