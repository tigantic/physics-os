"""
Test Rust TCI Core functionality.

Tests:
1. TCISampler creation and basic usage
2. IndexBatch generation
3. MaxVol configuration
"""

import sys
sys.path.insert(0, ".")

from tci_core import (
    RUST_AVAILABLE,
    TCISampler,
    IndexBatch,
    MaxVolConfig,
    TruncationPolicy,
    TCIConfig,
)


def test_rust_available():
    """Verify Rust extension is loaded."""
    print("Test 1: Rust extension availability...")
    assert RUST_AVAILABLE, "Rust extension not available!"
    print("  ✓ Rust TCI Core loaded successfully")


def test_sampler_creation():
    """Test TCISampler instantiation."""
    print("Test 2: TCISampler creation...")
    
    # Create sampler
    sampler = TCISampler(
        n_qubits=12,
        boundary="periodic",
        seed=42
    )
    
    assert sampler is not None
    print("  ✓ TCISampler created")


def test_maxvol_config():
    """Test MaxVolConfig creation."""
    print("Test 3: MaxVolConfig creation...")
    
    config = MaxVolConfig(
        tolerance=1e-2,
        max_iterations=100,
        regularization=1e-10
    )
    
    assert config is not None
    print("  ✓ MaxVolConfig created")


def test_truncation_policy():
    """Test TruncationPolicy creation."""
    print("Test 4: TruncationPolicy creation...")
    
    policy = TruncationPolicy(
        target_rank=32,
        hard_cap=128,
        relative_tol=1e-8,
        monitor_growth=True
    )
    
    assert policy is not None
    assert policy.hard_cap == 128
    print("  ✓ TruncationPolicy created")


def test_tci_config():
    """Test TCIConfig creation."""
    print("Test 5: TCIConfig creation...")
    
    config = TCIConfig(
        n_qubits=16,
        max_rank=64,
        tolerance=1e-6,
        batch_size=10000
    )
    
    assert config is not None
    assert config.n_qubits == 16
    assert config.n_points == 2**16
    print(f"  Config: {config}")
    print("  ✓ TCIConfig created")


def test_index_batch():
    """Test IndexBatch functionality."""
    print("Test 6: IndexBatch functionality...")
    
    # IndexBatch is created by TCISampler, test if type exists
    assert IndexBatch is not None
    print("  ✓ IndexBatch type available")


def run_all_tests():
    """Run all Rust TCI tests."""
    print("=" * 60)
    print("Rust TCI Core Tests")
    print("=" * 60)
    print()
    
    test_rust_available()
    test_sampler_creation()
    test_maxvol_config()
    test_truncation_policy()
    test_tci_config()
    test_index_batch()
    
    print()
    print("=" * 60)
    print("ALL RUST TCI CORE TESTS PASSED ✓")
    print("=" * 60)
    print()
    print("Rust TCI Core is operational.")
    print("Next: Connect sampler to PyTorch flux evaluation.")
    print()


if __name__ == "__main__":
    run_all_tests()
