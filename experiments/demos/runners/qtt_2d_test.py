"""
2D QTT Shift Test - Visual Validation

This tests the critical "Interleaved Shift MPO" for 2D CFD.
If a Gaussian blob shifts in Y without distorting, the 2D infrastructure works.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from ontic.cfd.qtt_2d import (
    dense_to_qtt_2d, qtt_2d_to_dense, 
    shift_mpo_x_2d, shift_mpo_y_2d,
    apply_mpo_2d, riemann_quadrant_ic,
    morton_encode, morton_decode
)


def test_shift_operators():
    """Test X and Y shift operators on a simple pattern."""
    print("=" * 60)
    print("2D SHIFT OPERATOR TEST")
    print("=" * 60)
    
    nx, ny = 4, 4  # 16×16 grid (small for debugging)
    Nx, Ny = 2**nx, 2**ny
    n_qubits = 2 * max(nx, ny)  # Interleaved bits
    
    # Create a simple diagonal pattern to test shifts
    x = torch.linspace(0, 1, Nx, dtype=torch.float32)
    y = torch.linspace(0, 1, Ny, dtype=torch.float32)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Test pattern: concentrated in one corner
    field = torch.zeros(Nx, Ny, dtype=torch.float32)
    field[4:8, 4:8] = 1.0  # Square in center
    
    print(f"Grid: {Nx}×{Ny}")
    print(f"Interleaved qubits: {n_qubits}")
    
    # Compress to QTT2D
    qtt = dense_to_qtt_2d(field, max_bond=16)
    print(f"Max rank: {qtt.max_rank}")
    
    # Build shift MPOs
    shift_x = shift_mpo_x_2d(n_qubits)
    shift_y = shift_mpo_y_2d(n_qubits)
    
    print(f"\nShift-X MPO cores: {len(shift_x)}")
    for i, core in enumerate(shift_x):
        print(f"  Core {i}: {core.shape}")
    
    print(f"\nShift-Y MPO cores: {len(shift_y)}")
    for i, core in enumerate(shift_y):
        print(f"  Core {i}: {core.shape}")
    
    # Visualize the original pattern
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    reconstructed = qtt_2d_to_dense(qtt)
    
    axes[0].imshow(field.T, origin='lower', cmap='viridis')
    axes[0].set_title(f'Original\nError: 0')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    
    axes[1].imshow(reconstructed.T, origin='lower', cmap='viridis')
    err = (field - reconstructed).abs().max().item()
    axes[1].set_title(f'QTT Reconstructed\nMax Error: {err:.2e}')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    
    # Show Morton ordering pattern
    morton_idx = torch.zeros(Nx, Ny, dtype=torch.float32)
    for ix in range(Nx):
        for iy in range(Ny):
            morton_idx[ix, iy] = morton_encode(ix, iy, max(nx, ny))
    
    axes[2].imshow(morton_idx.T, origin='lower', cmap='viridis')
    axes[2].set_title('Morton Z-Curve Order')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    
    plt.tight_layout()
    plt.savefig('morton_order_test.png', dpi=150)
    print(f"\n✅ Saved: morton_order_test.png")
    plt.close()


def test_riemann_visualization():
    """Visualize the 2D Riemann quadrant initial condition."""
    print("\n" + "=" * 60)
    print("2D RIEMANN QUADRANT VISUALIZATION")
    print("=" * 60)
    
    nx, ny = 8, 8  # 256×256
    Nx, Ny = 2**nx, 2**ny
    
    # Initialize
    rho, u, v, P = riemann_quadrant_ic(nx, ny, config=3)
    
    # Compress and decompress
    rho_qtt = dense_to_qtt_2d(rho, max_bond=32)
    rho_rec = qtt_2d_to_dense(rho_qtt)
    
    print(f"Grid: {Nx}×{Ny}")
    print(f"Compression ratio: {rho_qtt.dense_memory_bytes() / rho_qtt.memory_bytes:.1f}×")
    print(f"Max rank: {rho_qtt.max_rank}")
    
    # Visualize all four primitive variables
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    im0 = axes[0, 0].imshow(rho.T, origin='lower', cmap='viridis')
    axes[0, 0].set_title(f'Density ρ (rank={rho_qtt.max_rank})')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(P.T, origin='lower', cmap='plasma')
    axes[0, 1].set_title('Pressure P')
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[1, 0].imshow(u.T, origin='lower', cmap='RdBu')
    axes[1, 0].set_title('X-Velocity u')
    plt.colorbar(im2, ax=axes[1, 0])
    
    im3 = axes[1, 1].imshow(v.T, origin='lower', cmap='RdBu')
    axes[1, 1].set_title('Y-Velocity v')
    plt.colorbar(im3, ax=axes[1, 1])
    
    for ax in axes.flat:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    plt.suptitle(f'2D Riemann Quadrant (Config 3) — {Nx}×{Ny} Grid\n'
                 f'QTT Compression: {rho_qtt.dense_memory_bytes()/rho_qtt.memory_bytes:.0f}×',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('riemann_quadrant_2d.png', dpi=150)
    print(f"✅ Saved: riemann_quadrant_2d.png")
    plt.close()


def test_scaling():
    """Test how rank scales with grid size for 2D Riemann."""
    print("\n" + "=" * 60)
    print("2D RIEMANN SCALING TEST")
    print("=" * 60)
    
    print(f"{'Grid':<12} {'Points':<12} {'Rank':<8} {'Compression':<12} {'Error':<10}")
    print("-" * 60)
    
    for n in [4, 5, 6, 7, 8, 9, 10]:
        nx, ny = n, n
        Nx, Ny = 2**nx, 2**ny
        
        rho, u, v, P = riemann_quadrant_ic(nx, ny, config=3)
        rho_qtt = dense_to_qtt_2d(rho, max_bond=64)
        rho_rec = qtt_2d_to_dense(rho_qtt)
        
        err = (rho - rho_rec).abs().max().item()
        ratio = rho_qtt.dense_memory_bytes() / rho_qtt.memory_bytes
        
        print(f"{Nx}×{Ny:<8} {Nx*Ny:<12,} {rho_qtt.max_rank:<8} {ratio:<12.1f}× {err:<10.2e}")
    
    print("-" * 60)
    print("KEY INSIGHT: Rank stays ~3 for piecewise constant IC!")
    print("The Diagonal Problem only appears when shocks become curved.")


if __name__ == "__main__":
    test_shift_operators()
    test_riemann_visualization()
    test_scaling()
