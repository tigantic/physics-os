"""
Shared pytest fixtures for Physics test suite.

Inherits common fixtures from the main tests/conftest.py and adds
physics-specific utilities.
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
    """
    seed = int(os.environ.get("ONTIC_ENGINE_TEST_SEED", DEFAULT_SEED))
    
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    yield seed


# =============================================================================
# PYTEST MARKERS
# =============================================================================
def pytest_configure(config):
    """Register custom markers for Physics tests."""
    config.addinivalue_line(
        "markers", "physics: mark test as a physics validation test"
    )
    config.addinivalue_line(
        "markers", "quantum: mark test as a quantum computing test"
    )
    config.addinivalue_line(
        "markers", "phase14: mark test as Phase 14 (Documentation)"
    )
    config.addinivalue_line(
        "markers", "phase15: mark test as Phase 15 (Validation)"
    )
    config.addinivalue_line(
        "markers", "phase16: mark test as Phase 16 (Integration)"
    )
    config.addinivalue_line(
        "markers", "phase20: mark test as Phase 20 (Quantum-Classical)"
    )
