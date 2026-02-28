"""
Integration test for TCI-based QTT CFD flux computation.

This test verifies:
1. QTT evaluation works correctly (qtt_eval.py)
2. Rusanov flux formula is correct (tci_flux.py)
3. The full TCI→Flux→QTT pipeline is sound

Note: This does NOT test the Rust TCI core (requires maturin build).
It tests the Python-side components that integrate with TCI.
"""

import math

import torch

from ontic.cfd.qtt_eval import (QTTContiguous, dense_to_qtt_cores,
                                    qtt_eval_batch, qtt_eval_multi_field_batch,
                                    qtt_to_dense, verify_qtt_evaluation)
from ontic.cfd.tci_flux import (TCIFluxConfig, rusanov_flux,
                                    verify_sound_speed_formula)


def test_qtt_sine_wave():
    """Test QTT evaluation on sine wave."""
    print("Test 1: QTT sine wave evaluation...")

    n_qubits = 12  # 4096 points
    N = 2**n_qubits

    # Create sine wave
    x = torch.linspace(0, 2 * math.pi, N)
    values = torch.sin(x)

    # Convert to QTT
    cores = dense_to_qtt_cores(values, max_rank=8)

    # Verify evaluation
    max_err, mean_err = verify_qtt_evaluation(cores, n_test=200)

    assert max_err < 1e-5, f"Max error {max_err} too large"
    assert mean_err < 1e-6, f"Mean error {mean_err} too large"

    print(f"  ✓ Evaluation error: max={max_err:.2e}, mean={mean_err:.2e}")


def test_qtt_step_function():
    """Test QTT evaluation on step function (harder case)."""
    print("Test 2: QTT step function evaluation...")

    n_qubits = 10  # 1024 points
    N = 2**n_qubits

    # Create step function (discontinuous - high rank)
    x = torch.linspace(0, 1, N)
    values = (x > 0.5).float()

    # Convert to QTT (will need higher rank)
    cores = dense_to_qtt_cores(values, max_rank=32)

    # Verify evaluation
    max_err, mean_err = verify_qtt_evaluation(cores, n_test=200)

    # Step function will have some error due to rank truncation
    assert max_err < 0.5, f"Max error {max_err} too large for step"

    print(f"  ✓ Step function error: max={max_err:.2e}, mean={mean_err:.2e}")


def test_rusanov_flux_sod():
    """Test Rusanov flux on Sod shock tube initial condition."""
    print("Test 3: Rusanov flux computation...")

    # Sod shock tube IC (left and right states)
    # Left: ρ=1, u=0, p=1
    # Right: ρ=0.125, u=0, p=0.1
    gamma = 1.4

    # Left state
    rho_L = torch.tensor([1.0])
    u_L = torch.tensor([0.0])
    p_L = torch.tensor([1.0])
    E_L = p_L / (gamma - 1) + 0.5 * rho_L * u_L**2
    rho_u_L = rho_L * u_L

    # Right state
    rho_R = torch.tensor([0.125])
    u_R = torch.tensor([0.0])
    p_R = torch.tensor([0.1])
    E_R = p_R / (gamma - 1) + 0.5 * rho_R * u_R**2
    rho_u_R = rho_R * u_R

    # Compute Rusanov flux
    F_rho, F_rho_u, F_E = rusanov_flux(rho_L, rho_u_L, E_L, rho_R, rho_u_R, E_R, gamma)

    # Verify flux is finite
    assert torch.isfinite(F_rho).all(), "F_rho is not finite"
    assert torch.isfinite(F_rho_u).all(), "F_rho_u is not finite"
    assert torch.isfinite(F_E).all(), "F_E is not finite"

    # Verify dissipation (flux should move from high to low pressure)
    # Mass flux should be negative (from left to right)
    # Since u=0 on both sides, pure dissipation term

    # Sound speeds
    c_L = math.sqrt(gamma * p_L.item() / rho_L.item())
    c_R = math.sqrt(gamma * p_R.item() / rho_R.item())
    lambda_max = max(abs(u_L.item()) + c_L, abs(u_R.item()) + c_R)

    print(f"  Sound speeds: c_L={c_L:.3f}, c_R={c_R:.3f}")
    print(f"  Max wave speed: λ_max={lambda_max:.3f}")
    print(
        f"  Fluxes: F_ρ={F_rho.item():.4f}, F_ρu={F_rho_u.item():.4f}, F_E={F_E.item():.4f}"
    )

    # The dissipation should drive flux in the upwind direction
    # With u=0, Rusanov reduces to pure central + dissipation
    expected_F_rho = -0.5 * lambda_max * (rho_R.item() - rho_L.item())

    # Use reasonable floating point tolerance
    assert (
        abs(F_rho.item() - expected_F_rho) < 1e-6
    ), f"F_rho mismatch: {F_rho.item()} vs {expected_F_rho}"

    print("  ✓ Rusanov flux correct")


def test_batched_flux():
    """Test batched flux computation (critical for TCI)."""
    print("Test 4: Batched flux computation...")

    batch_size = 10000
    gamma = 1.4

    # Random states (physically plausible)
    rho_L = 0.5 + torch.rand(batch_size)
    u_L = 0.5 * (torch.rand(batch_size) - 0.5)
    p_L = 0.5 + torch.rand(batch_size)
    E_L = p_L / (gamma - 1) + 0.5 * rho_L * u_L**2
    rho_u_L = rho_L * u_L

    rho_R = 0.5 + torch.rand(batch_size)
    u_R = 0.5 * (torch.rand(batch_size) - 0.5)
    p_R = 0.5 + torch.rand(batch_size)
    E_R = p_R / (gamma - 1) + 0.5 * rho_R * u_R**2
    rho_u_R = rho_R * u_R

    # Compute flux
    F_rho, F_rho_u, F_E = rusanov_flux(rho_L, rho_u_L, E_L, rho_R, rho_u_R, E_R, gamma)

    assert F_rho.shape == (batch_size,)
    assert F_rho_u.shape == (batch_size,)
    assert F_E.shape == (batch_size,)

    # All should be finite
    assert torch.isfinite(F_rho).all(), "NaN/Inf in F_rho"
    assert torch.isfinite(F_rho_u).all(), "NaN/Inf in F_rho_u"
    assert torch.isfinite(F_E).all(), "NaN/Inf in F_E"

    print(f"  ✓ Batched flux: {batch_size} points computed")


def test_multi_field_qtt():
    """Test evaluating multiple QTT fields at same indices (for CFD)."""
    print("Test 5: Multi-field QTT evaluation...")

    n_qubits = 10
    N = 2**n_qubits
    gamma = 1.4

    # Create Sod shock tube IC in QTT format
    x = torch.linspace(0, 1, N)

    # Left: ρ=1, u=0, p=1
    # Right: ρ=0.125, u=0, p=0.1
    rho = torch.where(x < 0.5, torch.ones_like(x), 0.125 * torch.ones_like(x))
    u = torch.zeros(N)
    p = torch.where(x < 0.5, torch.ones_like(x), 0.1 * torch.ones_like(x))

    rho_u = rho * u
    E = p / (gamma - 1) + 0.5 * rho * u**2

    # Convert to QTT
    rho_cores = dense_to_qtt_cores(rho, max_rank=32)
    rho_u_cores = dense_to_qtt_cores(rho_u, max_rank=32)
    E_cores = dense_to_qtt_cores(E, max_rank=32)

    # Batch of test indices
    test_indices = torch.randint(0, N, (100,))

    # Evaluate each field
    rho_vals = qtt_eval_batch(rho_cores, test_indices)
    rho_u_vals = qtt_eval_batch(rho_u_cores, test_indices)
    E_vals = qtt_eval_batch(E_cores, test_indices)

    # Compare to dense
    rho_dense = rho[test_indices]
    rho_u_dense = rho_u[test_indices]
    E_dense = E[test_indices]

    err_rho = (rho_vals - rho_dense).abs().max().item()
    err_rho_u = (rho_u_vals - rho_u_dense).abs().max().item()
    err_E = (E_vals - E_dense).abs().max().item()

    print(f"  Field errors: ρ={err_rho:.2e}, ρu={err_rho_u:.2e}, E={err_E:.2e}")

    # Should be reasonable (step function is hard)
    assert err_rho < 0.5, f"ρ error too large: {err_rho}"
    assert err_E < 1.5, f"E error too large: {err_E}"

    print("  ✓ Multi-field QTT evaluation works")


def test_neighbor_indices_logic():
    """Test the neighbor index logic (what Rust TCI core would do)."""
    print("Test 6: Neighbor index logic...")

    N = 1024

    # Test periodic boundary
    def periodic_neighbor(i, N):
        left = (i - 1) % N
        right = (i + 1) % N
        return left, right

    # Interior
    l, r = periodic_neighbor(500, N)
    assert l == 499 and r == 501, "Interior neighbor failed"

    # Left boundary
    l, r = periodic_neighbor(0, N)
    assert l == N - 1 and r == 1, f"Left boundary failed: {l}, {r}"

    # Right boundary
    l, r = periodic_neighbor(N - 1, N)
    assert l == N - 2 and r == 0, f"Right boundary failed: {l}, {r}"

    print("  ✓ Neighbor index logic correct")


def test_sound_speed_formula():
    """Verify sound speed formula (critical bug check)."""
    print("Test 7: Sound speed formula verification...")

    c = verify_sound_speed_formula()

    # Should be ~340 m/s for air at STP
    assert abs(c - 340) < 5, f"Sound speed {c} m/s, expected ~340"

    print(f"  ✓ Sound speed = {c:.1f} m/s (correct!)")


def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("TCI-QTT CFD Integration Tests")
    print("=" * 60)
    print()

    test_qtt_sine_wave()
    test_qtt_step_function()
    test_rusanov_flux_sod()
    test_batched_flux()
    test_multi_field_qtt()
    test_neighbor_indices_logic()
    test_sound_speed_formula()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
    print()
    print("Python-side TCI integration is ready.")
    print("To complete native TCI:")
    print("  1. cd tci_core && maturin develop --release")
    print(
        "  2. python -c 'from tci_core import TCISampler; print(\"Rust TCI loaded!\")'"
    )
    print()


if __name__ == "__main__":
    run_all_tests()
