"""
Layer 9 Audit: Engine Integration Validation
=============================================

Validates the Python bridge code for Unreal/Unity integration.
NOTE: Cannot test actual engine integration without Unreal/Unity installed.
Tests validate that the bridge classes work correctly in isolation.
"""

import sys
import os
import numpy as np

# Project root needed for integrations path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_unreal_bridge_classes():
    """Test Unreal bridge class structure."""
    # Add integrations path for bridge imports
    unreal_path = os.path.join(PROJECT_ROOT, "integrations", "unreal")
    if unreal_path not in sys.path:
        sys.path.insert(0, unreal_path)
    
    from python_bridge import (
        MessageType, 
        FieldConfig, 
        BridgeStats,
    )
    
    # Test MessageType enum
    assert MessageType.INIT.value == 0x01
    assert MessageType.SAMPLE.value == 0x10
    assert MessageType.RESPONSE_OK.value == 0xF0
    
    # Test FieldConfig
    config = FieldConfig(size_x=128, size_y=128, size_z=64)
    assert config.size_x == 128
    assert config.size_z == 64
    assert config.field_type == "vector"
    
    # Test BridgeStats
    stats = BridgeStats()
    assert stats.messages_received == 0
    assert stats.uptime >= 0
    
    return True, "MessageType, FieldConfig, BridgeStats validated"


def test_unity_package_structure():
    """Test Unity package structure exists."""
    unity_dir = os.path.join(PROJECT_ROOT, "integrations", "unity")
    
    required = ["package.json", "README.md", "Runtime", "Editor"]
    missing = [f for f in required if not os.path.exists(os.path.join(unity_dir, f))]
    
    if missing:
        return False, f"Missing: {missing}"
    
    # Check package.json
    import json
    with open(os.path.join(unity_dir, "package.json")) as f:
        pkg = json.load(f)
    
    assert "name" in pkg, "package.json missing 'name'"
    assert "version" in pkg, "package.json missing 'version'"
    
    return True, f"Unity package: {pkg.get('name', 'unknown')}"


def test_unreal_plugin_structure():
    """Test Unreal plugin structure exists."""
    unreal_dir = os.path.join(PROJECT_ROOT, "integrations", "unreal")
    
    required = ["HyperTensor.uplugin", "Source", "python_bridge.py", "README.md"]
    missing = [f for f in required if not os.path.exists(os.path.join(unreal_dir, f))]
    
    if missing:
        return False, f"Missing: {missing}"
    
    # Check uplugin file
    import json
    with open(os.path.join(unreal_dir, "HyperTensor.uplugin")) as f:
        plugin = json.load(f)
    
    assert "FriendlyName" in plugin or "VersionName" in plugin, "Invalid uplugin format"
    
    return True, f"Unreal plugin: HyperTensor.uplugin"


def test_integration_module():
    """Test tensornet.integration module."""
    from tensornet.integration import (
        Configuration,
        HealthStatus,
        get_logger,
    )
    
    # Test Configuration
    config = Configuration()
    assert config is not None
    
    # Test HealthStatus enum
    assert HealthStatus.HEALTHY is not None
    assert HealthStatus.UNHEALTHY is not None
    
    # Test logger
    logger = get_logger("test")
    assert logger is not None
    
    return True, "tensornet.integration module validated"


def run_audit():
    print()
    print("=" * 66)
    print("           LAYER 9 AUDIT: Engine Integration")
    print("=" * 66)
    print()
    print("  NOTE: Full engine integration requires Unreal/Unity installed.")
    print("  These tests validate Python-side bridge and structure only.")
    print()
    
    results = []
    
    tests = [
        ("Unreal Bridge Classes", test_unreal_bridge_classes),
        ("Unity Package Structure", test_unity_package_structure),
        ("Unreal Plugin Structure", test_unreal_plugin_structure),
        ("Integration Module", test_integration_module),
    ]
    
    for name, test_fn in tests:
        print(f"Test: {name}...")
        try:
            passed, detail = test_fn()
            results.append((name, passed, detail))
            print(f"  {'PASS' if passed else 'FAIL'} | {detail}")
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"  FAIL | {e}")
    
    print()
    all_passed = all(r[1] for r in results)
    n_passed = sum(1 for r in results if r[1])
    
    print("=" * 66)
    if all_passed:
        print("  ALL TESTS PASSED (Python-side validated)")
        print("  Full validation requires Unreal/Unity engines")
    else:
        print(f"  {n_passed}/{len(results)} TESTS PASSED")
    print("=" * 66)
    
    return all_passed


if __name__ == "__main__":
    success = run_audit()
    sys.exit(0 if success else 1)
