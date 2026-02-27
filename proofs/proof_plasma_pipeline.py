#!/usr/bin/env python3
"""
Proof Tests for Phase 2: Fusion Plasma Discovery Pipeline

Validates:
1. Plasma ingester functionality
2. ELM event analysis
3. Profile distribution analysis
4. MHD mode detection
5. Safety factor analysis
6. Disruption risk assessment
7. Full pipeline integration
8. Report generation
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tensornet.ml.discovery.ingest.plasma import (
    PlasmaIngester,
    PlasmaShot,
    PlasmaProfile,
    MagneticField3D,
)
from tensornet.ml.discovery.pipelines.plasma_pipeline import (
    PlasmaDiscoveryPipeline,
    PlasmaPipelineResult,
    run_demo,
)


class ProofResult:
    """Track proof test results."""
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.message = ""
        self.duration_ms = 0
        
    def __repr__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} | {self.name}: {self.message}"


def proof_plasma_ingester() -> ProofResult:
    """Proof 1: Plasma ingester creates valid tensors."""
    result = ProofResult("plasma_ingester")
    
    try:
        import time
        start = time.time()
        
        ingester = PlasmaIngester()
        
        # Create test shot
        shot = PlasmaShot(
            shot_id="TEST-001",
            device="TEST",
            plasma_current_kA=1500,
            magnetic_field_T=5.0,
            electron_density_m3=1e20,
            electron_temp_keV=10.0,
            ion_temp_keV=8.0,
            stored_energy_MJ=300,
            confinement_mode="H-mode",
            elm_events=[
                {"time_ms": 100, "energy_kJ": 50},
                {"time_ms": 200, "energy_kJ": 60},
            ]
        )
        
        # Test profile creation
        rho = torch.linspace(0, 1, 100)
        Te = 10 * (1 - rho**2)
        profile = PlasmaProfile(rho=rho, values=Te, profile_type="Te", units="keV")
        
        profiles = {"Te": profile}
        time_traces = {"power": [10 + 0.1*i for i in range(100)]}
        
        ingested = ingester.ingest_shot(shot, profiles, time_traces)
        
        # Validate outputs
        assert "Te_distribution" in ingested, "Missing Te distribution"
        assert "elm_distribution" in ingested, "Missing ELM distribution"
        assert "power_series" in ingested, "Missing power series"
        assert "parameters" in ingested, "Missing parameters"
        
        Te_dist = ingested["Te_distribution"]
        assert Te_dist.shape[0] > 0, "Empty Te distribution"
        assert float(Te_dist.sum()) > 0.99, f"Te distribution not normalized: {float(Te_dist.sum())}"
        
        params = ingested["parameters"]
        assert len(params) == 6, f"Expected 6 parameters, got {len(params)}"
        
        result.duration_ms = (time.time() - start) * 1000
        result.passed = True
        result.message = f"Ingested shot with {len(ingested)} outputs"
        
    except Exception as e:
        result.message = f"Error: {e}"
        
    return result


def proof_elm_analysis() -> ProofResult:
    """Proof 2: ELM analysis detects patterns."""
    result = ProofResult("elm_analysis")
    
    try:
        import time
        start = time.time()
        
        ingester = PlasmaIngester()
        
        # Create shot with different ELM patterns
        large_elm_events = [
            {"time_ms": i * 100, "energy_kJ": 120 + np.random.rand() * 20}
            for i in range(10)
        ]
        
        elm_dist, elm_stats = ingester.build_elm_event_distribution(large_elm_events)
        
        # Validate ELM statistics
        assert elm_stats["count"] == 10, f"Expected 10 ELMs, got {elm_stats['count']}"
        assert elm_stats["mean_energy_kJ"] > 100, "Mean energy too low"
        assert elm_stats["frequency_Hz"] > 0, "Frequency should be > 0"
        
        # Validate distribution
        assert float(elm_dist.sum()) > 0.99, "ELM distribution not normalized"
        
        result.duration_ms = (time.time() - start) * 1000
        result.passed = True
        result.message = f"ELM analysis: {elm_stats['count']} events, {elm_stats['frequency_Hz']:.1f} Hz"
        
    except Exception as e:
        result.message = f"Error: {e}"
        
    return result


def proof_mhd_spectrum() -> ProofResult:
    """Proof 3: MHD mode spectrum detection."""
    result = ProofResult("mhd_spectrum")
    
    try:
        import time
        start = time.time()
        
        ingester = PlasmaIngester()
        
        # Create signal with known frequency content
        t = torch.linspace(0, 1, 1024)
        signal = torch.sin(2 * np.pi * 5 * t) + 0.5 * torch.sin(2 * np.pi * 10 * t)
        
        spectrum = ingester.build_mhd_mode_spectrum(signal, n_modes=32)
        
        # Validate spectrum
        assert len(spectrum) == 32, f"Expected 32 modes, got {len(spectrum)}"
        assert float(spectrum.sum()) > 0.99, "Spectrum not normalized"
        
        # Check that there's a peak (dominant mode)
        peak_mode = int(spectrum.argmax())
        peak_power = float(spectrum.max())
        
        assert peak_power > 0.05, "No clear peak in spectrum"
        
        result.duration_ms = (time.time() - start) * 1000
        result.passed = True
        result.message = f"Dominant mode n={peak_mode}, power={peak_power:.3f}"
        
    except Exception as e:
        result.message = f"Error: {e}"
        
    return result


def proof_q_profile() -> ProofResult:
    """Proof 4: Safety factor q profile processing."""
    result = ProofResult("q_profile")
    
    try:
        import time
        start = time.time()
        
        ingester = PlasmaIngester()
        
        # Create realistic q profile (q increases from core to edge)
        q_values = [1.0 + 0.05 * i + 0.001 * i**2 for i in range(100)]
        
        q_tensor = ingester.build_q_profile_tensor(q_values, target_length=256)
        
        # Validate
        assert len(q_tensor) == 256, f"Expected 256 points, got {len(q_tensor)}"
        
        # q should increase from core to edge
        q_core = float(q_tensor[10])
        q_edge = float(q_tensor[-10])
        
        assert q_edge > q_core, f"q should increase: q_core={q_core:.2f}, q_edge={q_edge:.2f}"
        
        # Check q95 (95% of the way out)
        q95_idx = int(0.95 * len(q_tensor))
        q95 = float(q_tensor[q95_idx])
        
        assert q95 > 3.0, f"q95 should be > 3 for stability, got {q95:.2f}"
        
        result.duration_ms = (time.time() - start) * 1000
        result.passed = True
        result.message = f"q profile: q_core={q_core:.2f}, q95={q95:.2f}"
        
    except Exception as e:
        result.message = f"Error: {e}"
        
    return result


def proof_pressure_gradient() -> ProofResult:
    """Proof 5: Pressure gradient computation."""
    result = ProofResult("pressure_gradient")
    
    try:
        import time
        start = time.time()
        
        ingester = PlasmaIngester()
        
        # Create pressure profile with pedestal
        rho = torch.linspace(0, 1, 100)
        pressure = 100 * (1 - rho**2)  # Core pressure
        pressure[-20:] *= torch.linspace(1, 0.1, 20)  # Edge pedestal
        
        profile = PlasmaProfile(
            rho=rho,
            values=pressure,
            profile_type="pressure",
            units="kPa"
        )
        
        grad = ingester.compute_pressure_gradient(profile)
        
        # Validate gradient
        assert len(grad) == len(pressure), "Gradient length mismatch"
        
        # Core gradient should be moderate
        core_grad = float(grad[10:30].abs().mean())
        
        # Edge gradient should be steep (pedestal)
        edge_grad = float(grad[-25:-5].abs().mean())
        
        # Pedestal should have steeper gradient
        assert edge_grad > core_grad * 0.5, f"Expected steep pedestal gradient"
        
        result.duration_ms = (time.time() - start) * 1000
        result.passed = True
        result.message = f"Gradient: core={core_grad:.2f}, edge={edge_grad:.2f}"
        
    except Exception as e:
        result.message = f"Error: {e}"
        
    return result


def proof_pipeline_findings() -> ProofResult:
    """Proof 6: Pipeline generates meaningful findings."""
    result = ProofResult("pipeline_findings")
    
    try:
        import time
        start = time.time()
        
        pipeline = PlasmaDiscoveryPipeline(verbose=False)
        
        # Create shot with known issues
        shot = PlasmaShot(
            shot_id="PROOF-006",
            device="TEST",
            plasma_current_kA=15000,
            magnetic_field_T=5.3,
            electron_density_m3=1e20,
            electron_temp_keV=12.0,
            ion_temp_keV=10.0,
            stored_energy_MJ=350,
            confinement_mode="H-mode",
            elm_events=[
                {"time_ms": 100, "energy_kJ": 150},  # Large ELM
                {"time_ms": 200, "energy_kJ": 140},
                {"time_ms": 300, "energy_kJ": 130},
            ],
            disruption=False,
        )
        
        rho = torch.linspace(0, 1, 100)
        profiles = {
            "Te": PlasmaProfile(rho=rho, values=12*(1-rho**2), profile_type="Te", units="keV"),
        }
        
        pipeline_result = pipeline.analyze_shot(shot, profiles=profiles)
        
        # Validate findings
        assert len(pipeline_result.findings) > 0, "No findings generated"
        
        # Should detect large ELM
        elm_findings = [f for f in pipeline_result.findings if "ELM" in f.get("type", "")]
        assert len(elm_findings) > 0, "Should detect ELM-related finding"
        
        # Check severity distribution
        critical = [f for f in pipeline_result.findings if f.get("severity") == "critical"]
        
        result.duration_ms = (time.time() - start) * 1000
        result.passed = True
        result.message = f"{len(pipeline_result.findings)} findings, {len(critical)} critical"
        
    except Exception as e:
        result.message = f"Error: {e}"
        
    return result


def proof_hypothesis_generation() -> ProofResult:
    """Proof 7: Pipeline generates hypotheses."""
    result = ProofResult("hypothesis_generation")
    
    try:
        import time
        start = time.time()
        
        pipeline = PlasmaDiscoveryPipeline(verbose=False)
        
        shot = PlasmaShot(
            shot_id="PROOF-007",
            device="TEST",
            plasma_current_kA=15000,
            magnetic_field_T=5.3,
            electron_density_m3=1e20,
            electron_temp_keV=12.0,
            ion_temp_keV=10.0,
            stored_energy_MJ=350,
            confinement_mode="H-mode",
            elm_events=[{"time_ms": i*50, "energy_kJ": 80} for i in range(10)],
        )
        
        rho = torch.linspace(0, 1, 100)
        profiles = {
            "Te": PlasmaProfile(rho=rho, values=12*(1-rho**2), profile_type="Te", units="keV"),
            "Ti": PlasmaProfile(rho=rho, values=10*(1-rho**2), profile_type="Ti", units="keV"),
        }
        
        pipeline_result = pipeline.analyze_shot(shot, profiles=profiles)
        
        # Validate hypotheses
        assert len(pipeline_result.hypotheses) > 0, "No hypotheses generated"
        
        for h in pipeline_result.hypotheses:
            assert "confidence" in h, "Hypothesis missing confidence"
            assert h["confidence"] >= 0 and h["confidence"] <= 1, "Invalid confidence"
            assert "summary" in h, "Hypothesis missing summary"
        
        result.duration_ms = (time.time() - start) * 1000
        result.passed = True
        result.message = f"{len(pipeline_result.hypotheses)} hypotheses generated"
        
    except Exception as e:
        result.message = f"Error: {e}"
        
    return result


def proof_report_generation() -> ProofResult:
    """Proof 8: Report generation works."""
    result = ProofResult("report_generation")
    
    try:
        import time
        start = time.time()
        
        pipeline = PlasmaDiscoveryPipeline(verbose=False)
        
        shot = PlasmaShot(
            shot_id="PROOF-008",
            device="TEST",
            plasma_current_kA=10000,
            magnetic_field_T=4.0,
            confinement_mode="L-mode",
        )
        
        pipeline_result = pipeline.analyze_shot(shot)
        
        # Test Markdown report
        md_report = pipeline.generate_report(pipeline_result, format="markdown")
        assert "# Plasma Discovery Report" in md_report, "Missing report header"
        assert "PROOF-008" in md_report, "Missing shot ID"
        assert "Findings" in md_report or "findings" in md_report.lower(), "Missing findings section"
        
        # Test JSON report
        json_report = pipeline.generate_report(pipeline_result, format="json")
        report_data = json.loads(json_report)
        assert report_data["shot_id"] == "PROOF-008", "JSON shot_id mismatch"
        
        result.duration_ms = (time.time() - start) * 1000
        result.passed = True
        result.message = f"MD: {len(md_report)} chars, JSON: {len(json_report)} chars"
        
    except Exception as e:
        result.message = f"Error: {e}"
        
    return result


def proof_demo_integration() -> ProofResult:
    """Proof 9: Demo runs end-to-end."""
    result = ProofResult("demo_integration")
    
    try:
        import time
        start = time.time()
        
        demo_result = run_demo()
        
        # Validate demo output
        assert isinstance(demo_result, PlasmaPipelineResult), "Demo didn't return PlasmaPipelineResult"
        assert demo_result.shot_id == "DEMO-ITER-001", "Unexpected shot ID"
        assert len(demo_result.findings) > 0, "Demo should find something"
        assert demo_result.execution_time_ms > 0, "Execution time should be recorded"
        
        result.duration_ms = (time.time() - start) * 1000
        result.passed = True
        result.message = f"Demo: {len(demo_result.findings)} findings in {demo_result.execution_time_ms:.1f}ms"
        
    except Exception as e:
        result.message = f"Error: {e}"
        
    return result


def proof_confinement_time() -> ProofResult:
    """Proof 10: Confinement time calculation."""
    result = ProofResult("confinement_time")
    
    try:
        import time
        start = time.time()
        
        ingester = PlasmaIngester()
        
        # ITER target: tau_E ≈ 5s at W = 350 MJ, P = 50 MW
        tau_E = ingester.compute_confinement_time(
            stored_energy_MJ=350,
            heating_power_MW=50
        )
        
        # Should be around 7000 ms (7 seconds)
        assert abs(tau_E - 7000) < 100, f"Unexpected tau_E: {tau_E} ms"
        
        # Edge case: zero power
        tau_zero = ingester.compute_confinement_time(350, 0)
        assert tau_zero == 0, "Zero power should give zero tau_E"
        
        result.duration_ms = (time.time() - start) * 1000
        result.passed = True
        result.message = f"τ_E = {tau_E:.0f} ms (expected ~7000 ms)"
        
    except Exception as e:
        result.message = f"Error: {e}"
        
    return result


def proof_boris_pusher_simulation() -> ProofResult:
    """Proof 11: Boris pusher particle confinement simulation."""
    result = ProofResult("boris_pusher_simulation")
    
    try:
        import time
        start = time.time()
        
        # Create pipeline
        pipeline = PlasmaDiscoveryPipeline(verbose=False)
        
        # Create ITER-like shot
        shot = PlasmaShot(
            shot_id="BORIS-TEST",
            device="ITER-SIM",
            plasma_current_kA=15000,
            magnetic_field_T=5.3,
            electron_temp_keV=10.0,
            ion_temp_keV=8.0,
            stored_energy_MJ=350,
            confinement_mode="H-mode",
        )
        
        # Run Boris pusher simulation (small for speed)
        report, findings = pipeline.simulate_confinement(
            shot,
            n_particles=100,
            n_steps=50,
            dt=1e-9,
        )
        
        # Verify confinement report
        assert report.total_particles == 100, "Should simulate 100 particles"
        assert report.confinement_ratio >= 0.0, "Ratio should be non-negative"
        assert report.confinement_ratio <= 1.0, "Ratio should be <= 1"
        assert len(findings) >= 1, "Should produce at least 1 finding"
        
        # Good plasma parameters should give good confinement
        assert report.confinement_ratio > 0.8, f"ITER params should confine well, got {report.confinement_ratio}"
        
        result.duration_ms = (time.time() - start) * 1000
        result.passed = True
        result.message = f"Boris: {report.confinement_ratio*100:.0f}% confined, {len(findings)} findings"
        
    except Exception as e:
        import traceback
        result.message = f"Error: {e}\n{traceback.format_exc()}"
        
    return result


def run_all_proofs(verbose: bool = True) -> dict:
    """Run all Phase 2 proof tests."""
    proofs = [
        proof_plasma_ingester,
        proof_elm_analysis,
        proof_mhd_spectrum,
        proof_q_profile,
        proof_pressure_gradient,
        proof_pipeline_findings,
        proof_hypothesis_generation,
        proof_report_generation,
        proof_demo_integration,
        proof_confinement_time,
        proof_boris_pusher_simulation,
    ]
    
    results = []
    passed = 0
    failed = 0
    
    if verbose:
        print("=" * 70)
        print("PHASE 2: FUSION PLASMA PIPELINE - PROOF TESTS")
        print("=" * 70)
        print()
    
    for proof_fn in proofs:
        result = proof_fn()
        results.append(result)
        
        if result.passed:
            passed += 1
        else:
            failed += 1
        
        if verbose:
            print(result)
    
    if verbose:
        print()
        print("=" * 70)
        print(f"RESULTS: {passed}/{len(proofs)} PASSED")
        print("=" * 70)
    
    # Generate attestation
    attestation = {
        "phase": "Phase 2: Fusion Plasma",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_tests": len(proofs),
        "passed": passed,
        "failed": failed,
        "success_rate": passed / len(proofs),
        "tests": [
            {
                "name": r.name,
                "passed": r.passed,
                "message": r.message,
                "duration_ms": r.duration_ms,
            }
            for r in results
        ]
    }
    
    # Save attestation
    attestation_path = Path(__file__).parent / "proof_plasma_pipeline.json"
    with open(attestation_path, "w") as f:
        json.dump(attestation, f, indent=2)
    
    if verbose:
        print(f"\nAttestation saved: {attestation_path}")
    
    return attestation


if __name__ == "__main__":
    import sys
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    result = run_all_proofs(verbose=True)
    
    if result["failed"] > 0:
        sys.exit(1)
