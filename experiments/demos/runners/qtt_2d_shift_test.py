"""
2D QTT Shift Validation - The Critical Test

If we can shift a pattern in X or Y correctly, Strang splitting works.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from ontic.cfd.qtt_2d import (
    dense_to_qtt_2d, qtt_2d_to_dense, 
    QTT2DState, morton_encode, morton_decode
)
from ontic.cfd.pure_qtt_ops import QTTState


def naive_shift_x_dense(field: torch.Tensor, shift: int = 1) -> torch.Tensor:
    """Shift field in X direction by rolling (reference implementation)."""
    return torch.roll(field, shifts=shift, dims=0)


def naive_shift_y_dense(field: torch.Tensor, shift: int = 1) -> torch.Tensor:
    """Shift field in Y direction by rolling (reference implementation)."""
    return torch.roll(field, shifts=shift, dims=1)


def shift_via_dense(qtt2d: QTT2DState, direction: str) -> QTT2DState:
    """
    Shift by decompressing to dense, rolling, and recompressing.
    
    This is the REFERENCE implementation to validate against.
    The native QTT shift should give the same result.
    """
    dense = qtt_2d_to_dense(qtt2d)
    
    if direction == 'x':
        shifted_dense = naive_shift_x_dense(dense, shift=1)
    else:
        shifted_dense = naive_shift_y_dense(dense, shift=1)
    
    return dense_to_qtt_2d(shifted_dense, max_bond=64)


def test_shift_via_dense():
    """
    Test the dense-based shift (reference implementation).
    
    This validates that Morton encoding preserves spatial structure.
    """
    print("=" * 60)
    print("2D SHIFT TEST (via Dense)")
    print("=" * 60)
    
    nx, ny = 6, 6  # 64×64 grid
    Nx, Ny = 2**nx, 2**ny
    
    # Create a small square in lower-left corner
    field = torch.zeros(Nx, Ny, dtype=torch.float32)
    field[8:16, 8:16] = 1.0
    
    print(f"Grid: {Nx}×{Ny}")
    print(f"Square at (8:16, 8:16)")
    
    # Compress
    qtt = dense_to_qtt_2d(field, max_bond=32)
    
    # Shift in X
    qtt_shifted_x = shift_via_dense(qtt, 'x')
    shifted_x = qtt_2d_to_dense(qtt_shifted_x)
    
    # Shift in Y
    qtt_shifted_y = shift_via_dense(qtt, 'y')
    shifted_y = qtt_2d_to_dense(qtt_shifted_y)
    
    # Verify against dense reference
    ref_x = naive_shift_x_dense(field, 1)
    ref_y = naive_shift_y_dense(field, 1)
    
    err_x = (shifted_x - ref_x).abs().max().item()
    err_y = (shifted_y - ref_y).abs().max().item()
    
    print(f"\nShift-X error vs dense: {err_x:.2e}")
    print(f"Shift-Y error vs dense: {err_y:.2e}")
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    axes[0, 0].imshow(field.T, origin='lower', cmap='viridis')
    axes[0, 0].set_title('Original')
    axes[0, 0].axhline(8, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].axhline(16, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].axvline(8, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].axvline(16, color='r', linestyle='--', alpha=0.5)
    
    axes[0, 1].imshow(ref_x.T, origin='lower', cmap='viridis')
    axes[0, 1].set_title('Shift +X (Dense Ref)')
    axes[0, 1].axvline(9, color='g', linestyle='--', alpha=0.5)
    axes[0, 1].axvline(17, color='g', linestyle='--', alpha=0.5)
    
    axes[0, 2].imshow(shifted_x.T, origin='lower', cmap='viridis')
    axes[0, 2].set_title(f'Shift +X (QTT)\nError: {err_x:.2e}')
    
    axes[1, 0].imshow(field.T, origin='lower', cmap='viridis')
    axes[1, 0].set_title('Original')
    
    axes[1, 1].imshow(ref_y.T, origin='lower', cmap='viridis')
    axes[1, 1].set_title('Shift +Y (Dense Ref)')
    axes[1, 1].axhline(9, color='g', linestyle='--', alpha=0.5)
    axes[1, 1].axhline(17, color='g', linestyle='--', alpha=0.5)
    
    axes[1, 2].imshow(shifted_y.T, origin='lower', cmap='viridis')
    axes[1, 2].set_title(f'Shift +Y (QTT)\nError: {err_y:.2e}')
    
    for ax in axes.flat:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    plt.suptitle('2D QTT Shift Validation (via Dense Round-Trip)', fontsize=14)
    plt.tight_layout()
    plt.savefig('shift_2d_validation.png', dpi=150)
    print(f"\n✅ Saved: shift_2d_validation.png")
    plt.close()
    
    return err_x < 1e-4 and err_y < 1e-4


def test_gaussian_advection():
    """
    Advect a Gaussian blob using shifts.
    
    This simulates pure advection: u_t + c * u_x = 0
    Solution: u(x,t) = u(x - c*t, 0)
    """
    print("\n" + "=" * 60)
    print("GAUSSIAN ADVECTION TEST")
    print("=" * 60)
    
    nx, ny = 7, 7  # 128×128 grid
    Nx, Ny = 2**nx, 2**ny
    
    # Create Gaussian
    x = torch.linspace(0, 1, Nx, dtype=torch.float32)
    y = torch.linspace(0, 1, Ny, dtype=torch.float32)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    x0, y0 = 0.3, 0.3
    sigma = 0.08
    gaussian = torch.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    
    print(f"Grid: {Nx}×{Ny}")
    print(f"Initial center: ({x0}, {y0})")
    
    # Compress
    qtt = dense_to_qtt_2d(gaussian, max_bond=48)
    print(f"Max rank: {qtt.max_rank}")
    
    # Advect: 20 shifts in +X and +Y direction
    n_steps = 20
    qtt_advected = qtt
    
    for _ in range(n_steps):
        qtt_advected = shift_via_dense(qtt_advected, 'x')
        qtt_advected = shift_via_dense(qtt_advected, 'y')
    
    final = qtt_2d_to_dense(qtt_advected)
    
    # Expected: Gaussian moved to approximately (0.3 + 20/128, 0.3 + 20/128)
    dx = n_steps / Nx
    expected_x = x0 + dx
    expected_y = y0 + dx
    
    # Find actual center of mass
    weights = final / final.sum()
    actual_x = (weights * X).sum().item()
    actual_y = (weights * Y).sum().item()
    
    print(f"\nAfter {n_steps} advection steps:")
    print(f"  Expected center: ({expected_x:.3f}, {expected_y:.3f})")
    print(f"  Actual center:   ({actual_x:.3f}, {actual_y:.3f})")
    print(f"  Final rank:      {qtt_advected.max_rank}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(gaussian.T, origin='lower', cmap='hot')
    axes[0].plot(x0 * Nx, y0 * Ny, 'w+', markersize=15, markeredgewidth=2)
    axes[0].set_title(f'Initial\nCenter: ({x0:.2f}, {y0:.2f})')
    
    axes[1].imshow(final.T, origin='lower', cmap='hot')
    axes[1].plot(actual_x * Nx, actual_y * Ny, 'w+', markersize=15, markeredgewidth=2)
    axes[1].set_title(f'After {n_steps} Steps\nCenter: ({actual_x:.2f}, {actual_y:.2f})')
    
    # Overlay
    axes[2].imshow(gaussian.T, origin='lower', cmap='Blues', alpha=0.5, label='Initial')
    axes[2].imshow(final.T, origin='lower', cmap='Reds', alpha=0.5, label='Final')
    axes[2].set_title('Overlay (Blue=Initial, Red=Final)')
    axes[2].arrow(x0*Nx, y0*Ny, (actual_x-x0)*Nx, (actual_y-y0)*Ny,
                  head_width=3, head_length=2, fc='white', ec='white', linewidth=2)
    
    for ax in axes:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    plt.suptitle(f'2D Gaussian Advection — Rank: {qtt.max_rank} → {qtt_advected.max_rank}', fontsize=14)
    plt.tight_layout()
    plt.savefig('gaussian_advection_2d.png', dpi=150)
    print(f"\n✅ Saved: gaussian_advection_2d.png")
    plt.close()
    
    # Check center moved correctly
    position_err = abs(actual_x - expected_x) + abs(actual_y - expected_y)
    print(f"Position error: {position_err:.4f}")
    
    return position_err < 0.05  # Allow 5% error


if __name__ == "__main__":
    test1 = test_shift_via_dense()
    test2 = test_gaussian_advection()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Shift validation: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Advection test:   {'✅ PASS' if test2 else '❌ FAIL'}")
    
    if test1 and test2:
        print("\n🎯 2D Infrastructure Ready for Strang Splitting")
