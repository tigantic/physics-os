#!/usr/bin/env python3
"""
Proof Test: DeFi Discovery Pipeline

Validates Phase 1 DeFi discovery capabilities.
"""

import sys
import json
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_ingester_token_flows():
    """Test token flow ingestion and distribution building."""
    from tensornet.discovery.ingest.defi import DeFiIngester
    
    ingester = DeFiIngester(grid_bits=10)
    
    flows = [
        {"from": "0xA", "to": "0xB", "amount": 100},
        {"from": "0xB", "to": "0xC", "amount": 250},
        {"from": "0xC", "to": "0xD", "amount": 50000},  # Whale
    ]
    
    dist = ingester.build_token_flow_distribution(flows)
    
    assert dist.shape[0] > 0, "Distribution should have elements"
    assert abs(float(dist.sum()) - 1.0) < 1e-5, "Distribution should sum to 1"
    assert float(dist.max()) > 0, "Should have non-zero max"
    
    return True


def test_ingester_call_graph():
    """Test call graph tensor building."""
    from tensornet.discovery.ingest.defi import DeFiIngester, ContractData
    
    ingester = DeFiIngester()
    
    contract = ContractData(
        address="0x1234",
        functions=[
            {"name": "swap", "stateMutability": "nonpayable"},
            {"name": "quote", "stateMutability": "view"},
            {"name": "transfer", "stateMutability": "nonpayable"},
        ],
        call_graph=[("swap", "quote")],
    )
    
    adj = ingester.build_call_graph_tensor(contract)
    
    assert adj.shape == (3, 3), f"Expected (3,3), got {adj.shape}"
    # swap(0) -> quote(1) should have edge (explicit or heuristic)
    assert float(adj[0, 1]) >= 0.5, f"Edge swap->quote should exist, got {float(adj[0, 1])}"
    
    return True


def test_ingester_price_series():
    """Test price series tensor building."""
    from tensornet.discovery.ingest.defi import DeFiIngester
    
    ingester = DeFiIngester()
    
    prices = [100 + i for i in range(50)]
    tensor = ingester.build_price_series_tensor(prices, target_length=32)
    
    assert tensor.shape[0] == 32, f"Expected 32, got {tensor.shape[0]}"
    
    return True


def test_pool_analysis():
    """Test DEX pool analysis."""
    from tensornet.discovery.pipelines.defi_pipeline import DeFiDiscoveryPipeline
    
    pipeline = DeFiDiscoveryPipeline()
    
    swaps = [{"amount0": i * 100, "tick": 1000 + i} for i in range(30)]
    swaps.append({"amount0": 1000000, "tick": 5000})  # Anomaly
    
    result = pipeline.analyze_pool(
        pool_address="0xTestPool",
        swap_events=swaps,
        liquidity_events=[],
    )
    
    assert len(result.findings) > 0, "Should detect findings"
    assert result.attestation_hash, "Should have attestation"
    
    return True


def test_lending_analysis():
    """Test lending protocol analysis."""
    from tensornet.discovery.pipelines.defi_pipeline import DeFiDiscoveryPipeline
    
    pipeline = DeFiDiscoveryPipeline()
    
    result = pipeline.analyze_lending_protocol(
        protocol_name="TestLend",
        borrow_events=[{"amount": 1000 * i} for i in range(1, 10)],
        repay_events=[{"amount": 500 * i} for i in range(1, 5)],
        liquidation_events=[{"amount": 50000}],
    )
    
    assert len(result.findings) > 0, "Should detect findings"
    
    # Check that protocol context was added
    has_protocol = any(
        f.evidence.get("protocol") == "TestLend" 
        for f in result.findings
    )
    assert has_protocol, "Findings should have protocol context"
    
    return True


def test_report_generation():
    """Test Immunefi report generation."""
    from tensornet.discovery.pipelines.defi_pipeline import DeFiDiscoveryPipeline
    
    pipeline = DeFiDiscoveryPipeline()
    
    # Use more swap events to avoid numerical issues
    result = pipeline.analyze_pool(
        pool_address="0xReportTest",
        swap_events=[{"amount0": 100 * i} for i in range(1, 20)],
        liquidity_events=[{"liquidity": 500}],
    )
    
    report = pipeline.generate_immunefi_report(result, "TestProtocol")
    
    assert "TestProtocol" in report, "Report should contain protocol name"
    assert "Attestation" in report, "Report should have attestation section"
    assert result.attestation_hash[:16] in report, "Should include hash"
    
    return True


def run_proof_tests():
    """Run all Phase 1 proof tests."""
    tests = [
        ("ingester_token_flows", test_ingester_token_flows),
        ("ingester_call_graph", test_ingester_call_graph),
        ("ingester_price_series", test_ingester_price_series),
        ("pool_analysis", test_pool_analysis),
        ("lending_analysis", test_lending_analysis),
        ("report_generation", test_report_generation),
    ]
    
    results = []
    all_passed = True
    
    print("=" * 60)
    print("PROOF: DeFi Discovery Pipeline (Phase 1)")
    print("=" * 60)
    
    for name, test_fn in tests:
        start = time.perf_counter()
        try:
            passed = test_fn()
            elapsed = (time.perf_counter() - start) * 1000
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_passed = False
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            status = "FAIL"
            all_passed = False
            print(f"  ERROR: {e}")
        
        results.append({
            "name": name,
            "status": status,
            "time_ms": elapsed,
        })
        
        print(f"  [{status}] {name} ({elapsed:.1f}ms)")
    
    print("-" * 60)
    
    passed_count = sum(1 for r in results if r["status"] == "PASS")
    print(f"  {passed_count}/{len(tests)} tests passed")
    
    # Generate attestation
    attestation = {
        "proof_type": "defi_discovery_pipeline",
        "phase": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "all_passed": all_passed,
    }
    
    content = json.dumps(attestation, sort_keys=True)
    attestation["hash"] = hashlib.sha256(content.encode()).hexdigest()
    
    # Write attestation
    output_path = Path(__file__).parent / "proof_defi_pipeline.json"
    with open(output_path, "w") as f:
        json.dump(attestation, f, indent=2)
    
    print(f"\n  Attestation: {attestation['hash'][:32]}...")
    print(f"  Written to: {output_path.name}")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_proof_tests())
