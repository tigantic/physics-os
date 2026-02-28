#!/usr/bin/env python3
"""
DeFi Discovery Pipeline

Full pipeline: DeFi Data → Ingest → Discovery → Hypotheses → Report
"""

import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..engine_v2 import DiscoveryEngineV2, Finding, DiscoveryResult
from ..ingest.defi import DeFiIngester, ContractData, DeFiSnapshot
from ..hypothesis.generator import HypothesisGenerator, Hypothesis


@dataclass
class DeFiDiscoveryConfig:
    """Configuration for DeFi discovery."""
    grid_bits: int = 12
    generate_hypotheses: bool = True
    severity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "ot_drift": 2.0,
        "sgw_anomaly": 500.0,
        "rkhs_mmd": 0.3,
        "ph_persistence": 0.2,
        "ga_geometric": 5.0,
    })
    enable_stages: List[str] = field(default_factory=lambda: [
        "ot", "sgw", "rkhs", "ph", "ga"
    ])


class DeFiDiscoveryPipeline:
    """
    End-to-end DeFi discovery pipeline.
    
    1. Ingest: Raw DeFi data → Tensors
    2. Discover: Tensors → Findings
    3. Hypothesize: Findings → Hypotheses
    4. Report: Hypotheses → Actionable output
    """
    
    def __init__(self, config: DeFiDiscoveryConfig = None):
        self.config = config or DeFiDiscoveryConfig()
        self.ingester = DeFiIngester(grid_bits=self.config.grid_bits)
        self.engine = DiscoveryEngineV2(grid_bits=self.config.grid_bits)
        self.hypothesis_gen = HypothesisGenerator(domain="defi")
    
    def analyze_pool(
        self,
        pool_address: str,
        swap_events: List[Dict],
        liquidity_events: List[Dict],
        historical_swaps: List[Dict] = None,
        generate_hypotheses: bool = True,
    ) -> DiscoveryResult:
        """
        Analyze DEX pool for anomalies.
        
        Args:
            pool_address: Pool contract address
            swap_events: Recent swap events
            liquidity_events: Recent liquidity events
            historical_swaps: Historical data for comparison
            
        Returns:
            DiscoveryResult with findings
        """
        # Ingest current data
        pool_data = self.ingester.ingest_uniswap_v3_pool(
            pool_address=pool_address,
            swap_events=swap_events,
            liquidity_events=liquidity_events,
        )
        
        # Build discovery input
        current_dist = pool_data["swap_distribution"]
        
        # Get historical baseline
        if historical_swaps:
            historical_pool = self.ingester.ingest_uniswap_v3_pool(
                pool_address=pool_address,
                swap_events=historical_swaps,
                liquidity_events=[],
            )
            historical_dist = historical_pool["swap_distribution"]
        else:
            # Use uniform as baseline
            historical_dist = torch.ones_like(current_dist) / len(current_dist)
        
        # Build combined input tensor
        # Stack current and historical for comparison
        discovery_input = torch.stack([current_dist, historical_dist])
        
        # Run discovery
        result = self.engine.discover(discovery_input)
        
        # Add pool context to findings via evidence dict
        for finding in result.findings:
            finding.evidence["pool_address"] = pool_address
            finding.evidence["analysis_type"] = "pool_swap_distribution"
        
        return result
    
    def analyze_lending_protocol(
        self,
        protocol_name: str,
        borrow_events: List[Dict],
        repay_events: List[Dict],
        liquidation_events: List[Dict],
        historical_borrows: List[Dict] = None,
    ) -> DiscoveryResult:
        """
        Analyze lending protocol for anomalies.
        
        Focus areas:
        - Unusual borrow patterns
        - Liquidation cascades
        - Health factor distributions
        """
        # Ingest data
        protocol_data = self.ingester.ingest_lending_protocol(
            protocol_name=protocol_name,
            borrow_events=borrow_events,
            repay_events=repay_events,
            liquidation_events=liquidation_events,
        )
        
        current_dist = protocol_data["borrow_distribution"]
        
        if historical_borrows:
            historical_data = self.ingester.ingest_lending_protocol(
                protocol_name=protocol_name,
                borrow_events=historical_borrows,
                repay_events=[],
                liquidation_events=[],
            )
            historical_dist = historical_data["borrow_distribution"]
        else:
            historical_dist = torch.ones_like(current_dist) / len(current_dist)
        
        discovery_input = torch.stack([current_dist, historical_dist])
        result = self.engine.discover(discovery_input)
        
        # Check liquidation concentration
        if len(liquidation_events) > 0:
            liq_dist = protocol_data["liquidation_distribution"]
            # High concentration in liquidations = potential cascade
            max_concentration = float(liq_dist.max())
            if max_concentration > 0.1:  # More than 10% in one bin
                result.findings.append(Finding(
                    primitive="INGEST",
                    severity="CRITICAL",
                    summary=f"Liquidation concentration detected: {max_concentration*100:.1f}% in single bucket",
                    evidence={
                        "protocol": protocol_name,
                        "num_liquidations": len(liquidation_events),
                        "concentration": max_concentration,
                    }
                ))
        
        for finding in result.findings:
            finding.evidence["protocol"] = protocol_name
            finding.evidence["analysis_type"] = "lending_protocol"
        
        return result
    
    def analyze_token_flows(
        self,
        token_address: str,
        transfer_events: List[Dict],
        time_window_hours: int = 24,
    ) -> DiscoveryResult:
        """
        Analyze ERC20 token transfer patterns.
        
        Detects:
        - Wash trading
        - Whale movements
        - Unusual concentration
        """
        flow_dist = self.ingester.build_token_flow_distribution(transfer_events)
        
        # Build uniform baseline
        uniform = torch.ones_like(flow_dist) / len(flow_dist)
        
        discovery_input = torch.stack([flow_dist, uniform])
        result = self.engine.discover(discovery_input)
        
        # Quick anomaly check
        anomaly_features = self.ingester.compute_anomaly_features(flow_dist, uniform)
        
        if anomaly_features["max_ratio"] > 50:
            result.findings.append(Finding(
                primitive="INGEST",
                severity="HIGH",
                summary=f"Whale concentration: {anomaly_features['max_ratio']:.1f}x normal",
                evidence={
                    "token": token_address,
                    "kl_divergence": anomaly_features["kl_divergence"],
                    "max_ratio": anomaly_features["max_ratio"],
                }
            ))
        
        for finding in result.findings:
            finding.evidence["token_address"] = token_address
            finding.evidence["time_window_hours"] = time_window_hours
        
        return result
    
    def generate_hypotheses(self, result: DiscoveryResult) -> List[Hypothesis]:
        """Generate hypotheses from discovery result."""
        return self.hypothesis_gen.generate(result)
    
    def generate_immunefi_report(
        self,
        result: DiscoveryResult,
        protocol_name: str,
        include_hypotheses: bool = True,
    ) -> str:
        """
        Generate Immunefi-ready vulnerability report.
        """
        critical_findings = [f for f in result.findings if f.severity == "CRITICAL"]
        high_findings = [f for f in result.findings if f.severity == "HIGH"]
        
        # Generate hypotheses
        hypotheses = self.generate_hypotheses(result) if include_hypotheses else []
        
        stage_names = [s.get("name", "unknown") for s in result.stages]
        
        report = f"""# {protocol_name} Vulnerability Report

**Generated**: {datetime.now(timezone.utc).isoformat()}
**Attestation**: {result.attestation_hash[:16]}...
**Pipeline Stages**: {', '.join(stage_names)}

## Executive Summary

- **CRITICAL Findings**: {len(critical_findings)}
- **HIGH Findings**: {len(high_findings)}
- **Total Findings**: {len(result.findings)}
- **Hypotheses Generated**: {len(hypotheses)}

"""
        
        # Add hypotheses section
        if hypotheses:
            report += """## Synthesized Hypotheses

*Cross-primitive analysis synthesized the following actionable hypotheses:*

"""
            for i, h in enumerate(hypotheses[:3], 1):  # Top 3 hypotheses
                report += f"""### Hypothesis {i}: {h.title}

**Confidence**: {h.confidence:.0%} | **Severity**: {h.severity}

{h.description}

**Recommended Action**: {h.recommended_action}

---

"""
        
        report += """## Critical Findings

"""
        for i, f in enumerate(critical_findings, 1):
            report += f"""### {i}. {f.primitive}

**Severity**: {f.severity}

{f.summary}

**Evidence**:
```json
{f.evidence}
```

---

"""
        
        report += """## High Severity Findings

"""
        for i, f in enumerate(high_findings, 1):
            report += f"""### {i}. {f.primitive}

**Severity**: {f.severity}

{f.summary}

---

"""
        
        report += f"""## Attestation

This analysis was performed using the Ontic Engine Autonomous Discovery Engine.

**Full Hash**: `{result.attestation_hash}`
**Execution Time**: {result.total_time*1000:.1f}ms
"""
        
        return report


def main():
    """Demo the DeFi discovery pipeline."""
    print("=" * 60)
    print("DeFi Discovery Pipeline - Demo")
    print("=" * 60)
    
    pipeline = DeFiDiscoveryPipeline()
    
    # Simulate pool data with anomaly
    normal_swaps = [{"amount0": i * 100, "tick": 1000 + i} for i in range(50)]
    anomalous_swaps = normal_swaps + [{"amount0": 1000000, "tick": 5000}]  # Whale swap
    
    print("\n[1] Analyzing DEX Pool...")
    result = pipeline.analyze_pool(
        pool_address="0xPool123",
        swap_events=anomalous_swaps,
        liquidity_events=[{"liquidity": 1000}],
        historical_swaps=normal_swaps,
    )
    
    print(f"    Findings: {len(result.findings)}")
    for f in result.findings[:3]:
        print(f"    - [{f.severity}] {f.primitive}: {f.summary[:50]}")
    
    # Simulate lending protocol with liquidations
    print("\n[2] Analyzing Lending Protocol...")
    result2 = pipeline.analyze_lending_protocol(
        protocol_name="TestLend",
        borrow_events=[{"amount": 1000 * i} for i in range(1, 20)],
        repay_events=[{"amount": 800 * i} for i in range(1, 15)],
        liquidation_events=[
            {"amount": 50000, "healthFactor": 0.95},
            {"amount": 48000, "healthFactor": 0.92},
            {"amount": 52000, "healthFactor": 0.88},
        ],
    )
    
    print(f"    Findings: {len(result2.findings)}")
    for f in result2.findings[:3]:
        print(f"    - [{f.severity}] {f.primitive}: {f.summary[:50]}")
    
    # Generate report
    print("\n[3] Generating Immunefi Report...")
    report = pipeline.generate_immunefi_report(result2, "TestLend Protocol")
    print(f"    Report length: {len(report)} chars")
    
    print("\n" + "=" * 60)
    print("✅ DeFi Discovery Pipeline operational")
    print("=" * 60)


if __name__ == "__main__":
    main()
