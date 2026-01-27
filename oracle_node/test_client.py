#!/usr/bin/env python3
"""
Test the Oracle Node locally without Docker.

Usage:
    python oracle_node/test_client.py
"""

import json
import numpy as np
import requests

BASE_URL = "http://localhost:8080"


def generate_climate_data():
    """Generate synthetic climate distribution shift."""
    np.random.seed(42)
    # Historical: centered at 15°C
    historical = np.random.normal(15, 3, 10000)
    # Current: shifted to 17.5°C (climate change)
    current = np.random.normal(17.5, 3.5, 10000)
    return historical.tolist(), current.tolist()


def generate_finance_data():
    """Generate synthetic market distribution shift."""
    np.random.seed(123)
    # Normal market: log-normal returns
    normal = np.random.lognormal(0, 0.2, 10000)
    # Crash: fat tails, shifted left
    crash = np.random.lognormal(-0.3, 0.5, 10000)
    return normal.tolist(), crash.tolist()


def generate_medical_data():
    """Generate synthetic blood flow distribution."""
    np.random.seed(456)
    # Healthy: normal flow
    healthy = np.random.normal(100, 10, 10000)
    # Blocked: bimodal (clot + compensation)
    blocked = np.concatenate([
        np.random.normal(60, 5, 5000),   # Reduced flow
        np.random.normal(120, 8, 5000),  # Compensatory increase
    ])
    return healthy.tolist(), blocked.tolist()


def test_analyze(name: str, data_a: list, data_b: list, domain: str):
    """Run analysis and print results."""
    print(f"\n{'='*60}")
    print(f"  TEST: {name}")
    print(f"{'='*60}")
    
    response = requests.post(
        f"{BASE_URL}/analyze",
        json={
            "distribution_a": data_a,
            "distribution_b": data_b,
            "domain_hint": domain,
        },
    )
    
    if response.status_code != 200:
        print(f"ERROR: {response.status_code} - {response.text}")
        return
    
    result = response.json()
    
    print(f"\n  RESULTS:")
    print(f"  ────────────────────────────────────────")
    print(f"  Stage 1 (OT):   W₂ = {result['stage_1_ot']['wasserstein_2']:.4f}")
    print(f"  Stage 2 (SGW):  Scale = {result['stage_2_sgw']['dominant_scale']}")
    print(f"  Stage 3 (RKHS): Anomaly = {result['stage_3_rkhs']['anomaly_level']}")
    print(f"  Stage 4 (PH):   Shape = {result['stage_4_ph']['shape_type']}")
    print(f"  Stage 5 (GA):   Trend = {result['stage_5_ga']['trend']}")
    print(f"  ────────────────────────────────────────")
    print(f"  Total Time: {result['total_time_seconds']*1000:.1f} ms")
    print(f"  SHA256: {result['sha256'][:32]}...")
    
    # Save attestation
    filename = f"ORACLE_ATTESTATION_{domain.upper()}.json"
    with open(filename, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {filename}")


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    O R A C L E   N O D E   T E S T   C L I E N T            ║
║                                                                              ║
║                Testing Domain-Agnostic Structure Analysis                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Health check
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print(f"✓ Oracle Node is running: {response.json()}")
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to Oracle Node. Start it first:")
        print("  python oracle_node/server.py")
        return
    
    # Test 1: Climate
    hist, curr = generate_climate_data()
    test_analyze("Climate Distribution Shift", hist, curr, "climate")
    
    # Test 2: Finance
    normal, crash = generate_finance_data()
    test_analyze("Market Crash Detection", normal, crash, "finance")
    
    # Test 3: Medical
    healthy, blocked = generate_medical_data()
    test_analyze("Blood Flow Anomaly", healthy, blocked, "medical")
    
    print("\n" + "="*60)
    print("  ALL TESTS COMPLETE")
    print("="*60)
    print("\n  The SAME pipeline analyzed:")
    print("  • Climate temperature shifts")
    print("  • Financial market crashes")
    print("  • Medical blood flow anomalies")
    print("\n  Zero domain-specific code. Pure structure analysis.")


if __name__ == "__main__":
    main()
