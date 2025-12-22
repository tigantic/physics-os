"""
Phase 23 Proofs: Infrastructure Hardening (TMR)
===============================================

Formal proofs for Phase 23 implementations:
1. TMR bit flip correction
2. Conservation watchdog detection
3. Checkpoint rollback recovery

Constitution Compliance: Article V (Mission Assurance)
"""

import sys
import json
import math
import torch
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Results storage
RESULTS = {
    "phase": "23",
    "title": "Infrastructure Hardening (TMR)",
    "proofs": []
}


def record_proof(name: str, passed: bool, details: dict):
    """Record proof result."""
    RESULTS["proofs"].append({
        "name": name,
        "passed": passed,
        "details": details
    })
    status = "PASS PASSED" if passed else "FAIL FAILED"
    print(f"  {status}: {name}")
    if not passed:
        print(f"    Details: {details}")


# =============================================================================
# Proof 23.1: TMR Bit Flip Correction
# =============================================================================

def proof_23_1_tmr_bit_flip():
    """
    Verify TMR detects and corrects single bit flips.
    
    Inject a bit flip into one replica and verify:
    1. The flip is detected
    2. The correct value is recovered via voting
    """
    print("\nProof 23.1: TMR Bit Flip Correction")
    
    from tensornet.deployment.rad_hard import MajorityVoter, TMRExecutor, TMRConfig
    
    # Create three identical tensors (correct state)
    correct_value = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    
    t1 = correct_value.clone()
    t2 = correct_value.clone()
    t3 = correct_value.clone()
    
    # Inject bit flip into replica 2 at position 2
    t2[2] = 999.0  # Simulated SEU
    
    # Vote
    voter = MajorityVoter(tolerance_abs=1e-10, tolerance_rel=1e-8)
    result, events = voter.vote_with_detection(t1, t2, t3)
    
    # Test 1: SEU should be detected
    seu_detected = len(events) > 0
    
    # Test 2: Correct value should be recovered
    error = torch.abs(result - correct_value).max().item()
    value_recovered = error < 1e-10
    
    # Test 3: Location should be identified
    if events:
        location_correct = events[0].location == (2,)
    else:
        location_correct = False
    
    passed = seu_detected and value_recovered and location_correct
    
    record_proof("TMR Bit Flip Correction", passed, {
        "seu_detected": seu_detected,
        "value_recovered": value_recovered,
        "max_error": error,
        "location_correct": location_correct,
        "n_events": len(events)
    })
    
    return passed


# =============================================================================
# Proof 23.2: TMR Multiple Bit Flips
# =============================================================================

def proof_23_2_tmr_multiple_flips():
    """
    Verify TMR handles multiple simultaneous bit flips.
    
    With 2-of-3 voting, single flips in different positions
    should all be corrected.
    """
    print("\nProof 23.2: TMR Multiple Bit Flips")
    
    from tensornet.deployment.rad_hard import MajorityVoter
    
    # Create test tensor
    correct_value = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    
    t1 = correct_value.clone()
    t2 = correct_value.clone()
    t3 = correct_value.clone()
    
    # Inject different single flips in each replica at DIFFERENT positions
    t1[0] = 100.0  # Flip at position 0
    t2[2] = 200.0  # Flip at position 2
    t3[4] = 300.0  # Flip at position 4
    
    # Vote (median should pick correct value at each position)
    voter = MajorityVoter()
    result, events = voter.vote_with_detection(t1, t2, t3)
    
    # For median voting:
    # pos 0: [100, 1, 1] -> median = 1 PASS
    # pos 1: [2, 2, 2] -> median = 2 PASS
    # pos 2: [3, 200, 3] -> median = 3 PASS
    # pos 3: [4, 4, 4] -> median = 4 PASS
    # pos 4: [5, 5, 300] -> median = 5 PASS
    
    error = torch.abs(result - correct_value).max().item()
    all_corrected = error < 1e-10
    
    # All three flips should be detected
    flips_detected = len(events) >= 3
    
    passed = all_corrected and flips_detected
    
    record_proof("TMR Multiple Bit Flips", passed, {
        "all_corrected": all_corrected,
        "max_error": error,
        "flips_detected": len(events)
    })
    
    return passed


# =============================================================================
# Proof 23.3: Conservation Watchdog Detection
# =============================================================================

def proof_23_3_conservation_watchdog():
    """
    Verify conservation watchdog detects non-physical energy spikes.
    
    Inject artificial energy increase and verify detection.
    """
    print("\nProof 23.3: Conservation Watchdog Detection")
    
    from tensornet.deployment.rad_hard import ConservationWatchdog
    
    # Use larger thresholds for normal CFD time-stepping
    watchdog = ConservationWatchdog(
        energy_threshold=0.05,  # 5% threshold (typical for explicit CFD)
        mass_threshold=0.01     # 1% threshold
    )
    
    # Create physical state
    nx = 50
    dx = 0.02
    
    rho_prev = torch.ones(nx, dtype=torch.float64)
    E_prev = torch.ones(nx, dtype=torch.float64) * 2.5  # Energy density
    
    # Test 1: Normal evolution (small change - within threshold)
    rho_good = rho_prev * 1.001  # 0.1% change (below 1% threshold)
    E_good = E_prev * 0.99      # 1% change (below 5% threshold)
    
    energy_score_good = watchdog.check_energy(E_good, E_prev, dx)
    mass_score_good = watchdog.check_mass(rho_good, rho_prev, dx)
    
    normal_ok = energy_score_good < 1.0 and mass_score_good < 1.0
    
    # Test 2: Anomalous energy spike (50% increase)
    E_bad = E_prev * 1.5
    
    energy_score_bad = watchdog.check_energy(E_bad, E_prev, dx)
    spike_detected = energy_score_bad > 1.0
    
    # Test 3: Negative density (non-physical)
    rho_negative = rho_prev.clone()
    rho_negative[10] = -1.0
    p_test = torch.ones(nx, dtype=torch.float64)
    
    positivity_check = not watchdog.check_positivity(rho_negative, p_test)
    
    # Test 4: NaN detection
    E_nan = E_prev.clone()
    E_nan[5] = float('nan')
    
    state_nan = {'E': E_nan, 'rho': rho_prev}
    prev_state = {'E': E_prev, 'rho': rho_prev}
    
    is_valid_nan, scores = watchdog.check_state(state_nan, prev_state, dx)
    nan_detected = not is_valid_nan
    
    passed = all([normal_ok, spike_detected, positivity_check, nan_detected])
    
    record_proof("Conservation Watchdog Detection", passed, {
        "normal_evolution_ok": normal_ok,
        "energy_spike_detected": spike_detected,
        "energy_score_bad": energy_score_bad,
        "negative_density_detected": positivity_check,
        "nan_detected": nan_detected
    })
    
    return passed


# =============================================================================
# Proof 23.4: Checkpoint Rollback Recovery
# =============================================================================

def proof_23_4_checkpoint_rollback():
    """
    Verify checkpoint manager enables recovery from corrupted state.
    """
    print("\nProof 23.4: Checkpoint Rollback Recovery")
    
    from tensornet.deployment.rad_hard import CheckpointManager, ConservationWatchdog
    
    # Use temporary directory for checkpoints
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(checkpoint_dir=tmpdir, keep_last_n=3)
        watchdog = ConservationWatchdog()
        
        # Create good state
        good_state = {
            'rho': torch.ones(50, dtype=torch.float64),
            'E': torch.ones(50, dtype=torch.float64) * 2.5
        }
        
        # Save checkpoint
        manager.save(good_state, step=100)
        
        # Verify checkpoint exists
        checkpoint_saved = 100 in manager.available_steps
        
        # Simulate corruption
        bad_state = {
            'rho': torch.ones(50, dtype=torch.float64) * -1.0,  # Non-physical
            'E': torch.ones(50, dtype=torch.float64) * 100.0
        }
        
        # Detect and rollback
        recovered = watchdog.rollback_if_anomaly(
            bad_state, good_state, good_state, dx=0.02
        )
        
        # Check recovery
        rollback_correct = torch.allclose(recovered['rho'], good_state['rho'])
        
        # Test 2: Load from checkpoint
        loaded_state, metadata = manager.load(100)
        load_correct = torch.allclose(loaded_state['rho'], good_state['rho'])
        
        # Test 3: Multiple checkpoints with pruning
        for step in [200, 300, 400, 500]:
            manager.save(good_state, step=step)
        
        # Should only have 3 checkpoints (keep_last_n=3)
        pruning_correct = len(manager.available_steps) == 3
        
        # Latest should be 500
        _, _, latest_step = manager.load_latest()
        latest_correct = latest_step == 500
    
    passed = all([checkpoint_saved, rollback_correct, load_correct, pruning_correct, latest_correct])
    
    record_proof("Checkpoint Rollback Recovery", passed, {
        "checkpoint_saved": checkpoint_saved,
        "rollback_correct": rollback_correct,
        "load_correct": load_correct,
        "pruning_correct": pruning_correct,
        "latest_correct": latest_correct
    })
    
    return passed


# =============================================================================
# Proof 23.5: TMR Executor Integration
# =============================================================================

def proof_23_5_tmr_executor():
    """
    Verify TMR executor runs computations with protection.
    """
    print("\nProof 23.5: TMR Executor Integration")
    
    from tensornet.deployment.rad_hard import TMRExecutor, TMRConfig
    
    # Define a simple kernel
    def my_kernel(x: torch.Tensor) -> torch.Tensor:
        return x ** 2 + 1
    
    config = TMRConfig(enabled=True)
    executor = TMRExecutor(my_kernel, config)
    
    # Execute with TMR
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    result = executor.execute(x)
    
    # Check correctness
    expected = x ** 2 + 1
    error = torch.abs(result - expected).max().item()
    correct = error < 1e-10
    
    # Check stats
    stats = executor.stats
    stats_ok = 'retry_count' in stats and 'seu_count' in stats
    
    # Test tuple output
    def multi_output_kernel(x: torch.Tensor) -> tuple:
        return x * 2, x ** 2
    
    executor2 = TMRExecutor(multi_output_kernel, config)
    out1, out2 = executor2.execute(x)
    
    multi_correct = (
        torch.allclose(out1, x * 2) and
        torch.allclose(out2, x ** 2)
    )
    
    passed = correct and stats_ok and multi_correct
    
    record_proof("TMR Executor Integration", passed, {
        "single_output_correct": correct,
        "max_error": error,
        "stats_ok": stats_ok,
        "multi_output_correct": multi_correct
    })
    
    return passed


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Phase 23 Proofs: Infrastructure Hardening (TMR)")
    print("=" * 60)
    
    all_passed = True
    
    # Run all proofs
    all_passed &= proof_23_1_tmr_bit_flip()
    all_passed &= proof_23_2_tmr_multiple_flips()
    all_passed &= proof_23_3_conservation_watchdog()
    all_passed &= proof_23_4_checkpoint_rollback()
    all_passed &= proof_23_5_tmr_executor()
    
    # Summary
    print("\n" + "=" * 60)
    n_passed = sum(1 for p in RESULTS["proofs"] if p["passed"])
    n_total = len(RESULTS["proofs"])
    print(f"Phase 23 Proofs: {n_passed}/{n_total} PASSED")
    
    RESULTS["all_passed"] = all_passed
    RESULTS["summary"] = f"{n_passed}/{n_total}"
    
    # Save results
    output_path = Path(__file__).parent / "proof_23_results.json"
    with open(output_path, 'w') as f:
        json.dump(RESULTS, f, indent=2, default=str)
    
    print(f"Results saved to: {output_path}")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
