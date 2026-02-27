"""
Legacy Test Configuration
=========================

These tests import from tensornet.hvac.projection_solver which has been
superseded by the hyperfoam package. They are kept for reference but
skipped by default.

To run legacy tests explicitly:
    pytest tests/legacy/ --run-legacy
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "legacy: mark test as legacy (requires --run-legacy to run)"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-legacy", default=False):
        skip_legacy = pytest.mark.skip(reason="Legacy test - use --run-legacy to run")
        for item in items:
            if "legacy" in str(item.fspath):
                item.add_marker(skip_legacy)


def pytest_addoption(parser):
    parser.addoption(
        "--run-legacy",
        action="store_true",
        default=False,
        help="Run legacy tests that depend on deprecated modules"
    )
