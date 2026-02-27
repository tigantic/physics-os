"""
Phase 4 Quick Validation Test

Tests that all Phase 4 components are properly structured.
Run this to validate the implementation before full integration test.
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all Phase 4 modules import correctly."""
    print("=" * 60)
    print("Phase 4 Component Validation")
    print("=" * 60)

    tests_passed = 0
    tests_failed = 0

    # Test 1: QTT Slice Extractor
    try:
        from tensornet.infra.sovereign.qtt_slice_extractor import (QTT3DState,
                                                             QTTSliceExtractor)

        print("✓ QTT Slice Extractor imports successfully")
        print(f"  - QTTSliceExtractor class: OK")
        print(f"  - QTT3DState dataclass: OK")
        tests_passed += 1
    except Exception as e:
        print(f"✗ QTT Slice Extractor import failed: {e}")
        tests_failed += 1

    # Test 2: Real-Time Streamer (modified)
    try:
        from tensornet.infra.sovereign.realtime_tensor_stream import \
            RealtimeTensorStream

        streamer = RealtimeTensorStream.__dict__
        if "stream_from_qtt" in dir(RealtimeTensorStream):
            print("✓ Real-Time Streamer updated successfully")
            print(f"  - stream_from_qtt() method: OK")
            tests_passed += 1
        else:
            print("✗ stream_from_qtt() method not found")
            tests_failed += 1
    except Exception as e:
        print(f"✗ Real-Time Streamer import failed: {e}")
        tests_failed += 1

    # Test 3: Integration Test Script
    try:
        test_file = Path("test_phase4_integration.py")
        if test_file.exists():
            print(f"✓ Integration test script exists")
            print(f"  - Path: {test_file}")
            print(f"  - Size: {test_file.stat().st_size} bytes")
            tests_passed += 1
        else:
            print(f"✗ Integration test script not found")
            tests_failed += 1
    except Exception as e:
        print(f"✗ Integration test check failed: {e}")
        tests_failed += 1

    # Test 4: Phase 4 Plan Documentation
    try:
        plan_file = Path("PHASE4_PLAN.md")
        if plan_file.exists():
            print(f"✓ Phase 4 plan documentation exists")
            print(f"  - Path: {plan_file}")
            lines = len(plan_file.read_text().splitlines())
            print(f"  - Lines: {lines}")
            tests_passed += 1
        else:
            print(f"✗ Phase 4 plan not found")
            tests_failed += 1
    except Exception as e:
        print(f"✗ Phase 4 plan check failed: {e}")
        tests_failed += 1

    # Test 5: Check for required dependencies
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        print(f"✓ PyTorch installed")
        print(f"  - Version: {torch.__version__}")
        print(f"  - CUDA available: {cuda_available}")
        if cuda_available:
            print(f"  - GPU: {torch.cuda.get_device_name(0)}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ PyTorch check failed: {e}")
        tests_failed += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"Results: {tests_passed}/{tests_passed + tests_failed} tests passed")
    print("=" * 60)

    if tests_failed == 0:
        print("\n✓ All Phase 4 components validated successfully!")
        print("\nNext steps:")
        print("  1. Run: python test_phase4_integration.py 10 --grid-size 64")
        print(
            "  2. In another terminal: cd glass-cockpit && cargo run --release --bin phase3"
        )
        return True
    else:
        print(f"\n✗ {tests_failed} validation(s) failed")
        print("Please fix the issues before running integration tests")
        return False


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
