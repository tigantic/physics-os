#!/usr/bin/env python3
"""
Resolution Independence Demo
=============================

Demonstrates the core insight:

    A field represented in QTT form can be sampled at any resolution
    without changing the field itself, while memory stays bounded
    *for bounded rank r*.

The FIELD is the physical reality (density, velocity, vorticity).
QTT is how we REPRESENT that field compactly.

Key insight:
    - Dense storage: O(N^d) memory, resolution-dependent
    - QTT storage:   O(d × r²) memory, resolution-INDEPENDENT for bounded rank r

This demo:
    1. Creates a REAL dense Taylor-Green vortex field on a moderate grid
    2. Compresses it to QTT using actual TT-SVD
    3. Samples the QTT at various resolutions (via full contraction)
    4. Shows that memory is constant while sample resolution varies

Usage:
    python demos/resolution_independence.py
    python demos/resolution_independence.py --interactive
"""

import sys
import os
import time
import argparse
import math

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np


def format_bytes(n: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def format_number(n: int) -> str:
    """Format large numbers with commas."""
    return f"{n:,}"


def dense_memory_for_resolution(resolution: int, dims: int = 2, dtype_bytes: int = 4) -> int:
    """Calculate memory for dense storage at given resolution."""
    return (resolution ** dims) * dtype_bytes


def print_header():
    """Print demo header."""
    print("\n" + "=" * 74)
    print("  RESOLUTION INDEPENDENCE DEMO")
    print("  A field represented in QTT form (for bounded rank r)")
    print("=" * 74)
    print("""
  The FIELD is the physical quantity: Taylor-Green vorticity ω(x,y).
  QTT is how we REPRESENT that field in compressed tensor-train format.

  Key property (for bounded rank r):
    • Dense storage scales as O(N^d) — doubles resolution = 4× memory
    • QTT storage scales as O(d×r²) — resolution-INDEPENDENT

  The method:
    1. Create dense Taylor-Green field on moderate grid (256×256)
    2. Compress to QTT via TT-SVD (actual implementation)
    3. Sample QTT at various resolutions via contraction
    4. Memory = Σ(core sizes) stays constant regardless of sample resolution

  Note: Memory table includes hypothetical dense sizes; 
        we only sample up to 256² in this demo.
""")


def create_taylor_green_vorticity(resolution: int, time: float = 0.0, nu: float = 0.01) -> np.ndarray:
    """
    Create actual Taylor-Green vorticity field.
    
    ω(x,y,t) = 2k cos(kx) cos(ky) exp(-2νk²t)
    
    where k = 2π (single wavelength in [0,1]² domain).
    
    Args:
        resolution: Grid resolution
        time: Physical time (for decay)
        nu: Kinematic viscosity
        
    Returns:
        Vorticity field as 2D numpy array
    """
    k = 2 * np.pi  # Wavenumber for [0,1] domain
    decay = np.exp(-2 * nu * k**2 * time)
    
    x = np.linspace(0, 1, resolution, endpoint=False)
    y = np.linspace(0, 1, resolution, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Vorticity: ω = ∂v/∂x - ∂u/∂y = 2k cos(kx) cos(ky) exp(-2νk²t)
    omega = 2 * k * np.cos(k * X) * np.cos(k * Y) * decay
    
    return omega.astype(np.float32)


# =============================================================================
# QTT Implementation (using real TT-SVD)
# =============================================================================

def tt_svd(
    tensor: torch.Tensor,
    shape: tuple,
    chi_max: int = 32,
    tol: float = 1e-10
) -> tuple:
    """
    Tensor Train SVD decomposition.
    
    Decomposes a tensor T[i₁,i₂,...,i_L] into a train of tensors:
    T ≈ A₁[i₁] · A₂[i₂] · ... · A_L[i_L]
    
    where each A_k has shape (χ_{k-1}, d_k, χ_k).
    
    This is the REAL TT-SVD algorithm from tensornet/cfd/qtt.py.
    """
    L = len(shape)
    dtype = tensor.dtype
    device = tensor.device
    
    # Reshape tensor to target shape
    T = tensor.reshape(shape)
    
    cores = []
    total_error_sq = 0.0
    frobenius_norm = torch.norm(T).item()
    
    # Left-to-right sweep with SVD truncation
    current = T.reshape(shape[0], -1)
    chi_left = 1
    
    for k in range(L - 1):
        d_k = shape[k]
        current = current.reshape(chi_left * d_k, -1)
        
        # Truncated SVD
        U, S, Vh = torch.linalg.svd(current, full_matrices=False)
        
        # Determine truncation
        if tol > 0 and len(S) > 1:
            cumsum = torch.cumsum(S**2, dim=0)
            total_sq = cumsum[-1].item()
            threshold = tol**2 * frobenius_norm**2
            keep = len(S)
            for i in range(len(S) - 1, 0, -1):
                tail_sq = total_sq - cumsum[i - 1].item()
                if tail_sq > threshold:
                    keep = i + 1
                    break
                keep = i
            keep = max(1, min(keep, chi_max))
        else:
            keep = min(chi_max, len(S))
        
        # Truncate
        U = U[:, :keep]
        S_kept = S[:keep]
        Vh = Vh[:keep, :]
        
        # Track error
        if keep < len(S):
            total_error_sq += torch.sum(S[keep:]**2).item()
        
        # Form core tensor
        core = U.reshape(chi_left, d_k, keep)
        cores.append(core)
        
        # Propagate S @ Vh
        current = torch.diag(S_kept) @ Vh
        chi_left = keep
    
    # Last core
    d_last = shape[-1]
    last_core = current.reshape(chi_left, d_last, 1)
    cores.append(last_core)
    
    total_error = math.sqrt(total_error_sq)
    
    return cores, total_error


def field_to_qtt(field: np.ndarray, chi_max: int = 32, tol: float = 1e-10) -> dict:
    """
    Compress a 2D field to QTT format using real TT-SVD.
    
    Args:
        field: 2D numpy array
        chi_max: Maximum bond dimension
        tol: Truncation tolerance
        
    Returns:
        Dictionary with QTT data
    """
    tensor = torch.from_numpy(field.flatten()).float()
    original_shape = field.shape
    N = tensor.numel()
    
    # Pad to power of 2
    N_padded = 1 << (N - 1).bit_length()
    if N_padded > N:
        padded = torch.zeros(N_padded)
        padded[:N] = tensor
        tensor = padded
    
    num_qubits = int(math.log2(N_padded))
    qtt_shape = tuple([2] * num_qubits)
    
    # Real TT-SVD
    cores, truncation_error = tt_svd(tensor, qtt_shape, chi_max=chi_max, tol=tol)
    
    # Memory calculation
    memory = sum(c.numel() * c.element_size() for c in cores)
    bond_dims = [c.shape[2] for c in cores[:-1]]
    
    return {
        'cores': cores,
        'original_shape': original_shape,
        'num_qubits': num_qubits,
        'truncation_error': truncation_error,
        'memory_bytes': memory,
        'bond_dims': bond_dims,
        'max_rank': max(bond_dims) if bond_dims else 1,
    }


def qtt_to_field(qtt: dict, target_resolution: int = None) -> np.ndarray:
    """
    Reconstruct field from QTT via full contraction.
    
    This contracts all cores to get the full vector, then reshapes.
    For large fields, this is expensive but exact.
    
    Args:
        qtt: QTT dictionary from field_to_qtt
        target_resolution: If specified, resample to this resolution
        
    Returns:
        2D numpy array
    """
    cores = qtt['cores']
    original_shape = qtt['original_shape']
    
    # Full contraction
    vec = cores[0]  # (1, d, chi)
    for core in cores[1:]:
        vec = torch.tensordot(vec, core, dims=([-1], [0]))
    
    # Squeeze boundaries and flatten
    vec = vec.squeeze(0).squeeze(-1).flatten()
    
    # Unpad
    original_size = original_shape[0] * original_shape[1]
    vec = vec[:original_size]
    
    # Reshape
    field = vec.reshape(original_shape).numpy()
    
    # Resample if requested
    if target_resolution is not None and target_resolution != original_shape[0]:
        # Simple resample via interpolation
        try:
            from scipy.ndimage import zoom
            factor = target_resolution / original_shape[0]
            field = zoom(field, factor, order=1)
        except ImportError:
            # Fallback: nearest-neighbor via numpy
            old_res = original_shape[0]
            new_res = target_resolution
            indices = np.linspace(0, old_res - 1, new_res).astype(int)
            field = field[np.ix_(indices, indices)]
    
    return field


def sample_qtt_at_resolution(qtt: dict, resolution: int) -> np.ndarray:
    """
    Sample QTT at specified resolution.
    
    Method: Contract to full field at original resolution,
    then resample to target. This is the honest approach.
    
    For truly resolution-independent sampling, we would need
    to implement QTT evaluation at arbitrary points via
    partial contractions with interpolation weights.
    
    Args:
        qtt: QTT dictionary
        resolution: Target resolution
        
    Returns:
        2D numpy array at target resolution
    """
    return qtt_to_field(qtt, target_resolution=resolution)


class QTTField:
    """
    A continuous field stored in Quantized Tensor Train format.
    
    This wraps real TT-SVD compression of a dense analytic field.
    The field can be sampled at any resolution; memory stays bounded
    because we store the compressed QTT cores, not the samples.
    """
    
    def __init__(self, source_resolution: int = 256, chi_max: int = 32, 
                 tol: float = 1e-10, nu: float = 0.01):
        """
        Create a QTT-compressed Taylor-Green vorticity field.
        
        Args:
            source_resolution: Resolution for initial dense field
            chi_max: Maximum bond dimension
            tol: Truncation tolerance for TT-SVD
            nu: Kinematic viscosity for Taylor-Green decay
        """
        self.source_resolution = source_resolution
        self.chi_max = chi_max
        self.tol = tol
        self.nu = nu
        self.time = 0.0
        
        # Create and compress the initial field
        self._compress_field()
    
    def _compress_field(self):
        """Create dense Taylor-Green and compress to QTT."""
        # Create actual Taylor-Green vorticity
        dense = create_taylor_green_vorticity(
            self.source_resolution, 
            time=self.time, 
            nu=self.nu
        )
        
        # Store field range for consistent visualization
        self._vmin = dense.min()
        self._vmax = dense.max()
        
        # Compress via real TT-SVD
        self.qtt = field_to_qtt(dense, chi_max=self.chi_max, tol=self.tol)
    
    @property
    def memory_bytes(self) -> int:
        """Total memory of QTT cores."""
        return self.qtt['memory_bytes']
    
    @property
    def max_rank(self) -> int:
        """Maximum bond dimension achieved."""
        return self.qtt['max_rank']
    
    @property
    def truncation_error(self) -> float:
        """Truncation error from TT-SVD."""
        return self.qtt['truncation_error']
    
    @property
    def num_qubits(self) -> int:
        """Number of QTT cores (= log2 of padded size)."""
        return self.qtt['num_qubits']
    
    @property
    def field_range(self) -> tuple:
        """Fixed (vmin, vmax) for consistent visualization."""
        return (self._vmin, self._vmax)
    
    def sample(self, resolution: int) -> np.ndarray:
        """Sample the field at specified resolution."""
        return sample_qtt_at_resolution(self.qtt, resolution)
    
    def evolve(self, dt: float = 0.1):
        """
        Evolve the field forward in time.
        
        Re-creates the analytic Taylor-Green at the new time
        and re-compresses to QTT. Memory stays bounded.
        """
        self.time += dt
        self._compress_field()


def run_demo(interactive: bool = False):
    """Run the resolution independence demo."""
    print_header()
    
    # Create the field
    print("Creating Taylor-Green vorticity field...")
    print("  Source: 256×256 dense grid, compressed via TT-SVD")
    print("  χ_max: 32 (maximum bond dimension)")
    print()
    
    field = QTTField(source_resolution=256, chi_max=32, tol=1e-10)
    
    qtt_memory = field.memory_bytes
    print(f"  QTT Memory:        {format_bytes(qtt_memory)}")
    print(f"  Achieved rank:     {field.max_rank}")
    print(f"  Number of cores:   {field.num_qubits}")
    print(f"  Truncation error:  {field.truncation_error:.2e}")
    print()
    
    # Memory comparison table
    print("-" * 74)
    print("  MEMORY COMPARISON TABLE")
    print("  (QTT memory is FIXED; dense memory grows as O(N²))")
    print("-" * 74)
    print()
    print("  Note: This table shows *hypothetical* dense sizes.")
    print("        We only sample up to 256² in this demo.")
    print()
    print("  Resolution      │ Points            │ Dense Would Be │ QTT Memory │ Ratio")
    print("  ────────────────┼───────────────────┼────────────────┼────────────┼────────────")
    
    resolutions = [64, 256, 1024, 4096, 16384, 65536, 1048576]
    
    for res in resolutions:
        n_points = res * res
        dense_mem = dense_memory_for_resolution(res)
        memory_ratio = dense_mem / qtt_memory
        
        res_str = f"{format_number(res)}²"
        points_str = format_number(n_points)
        
        print(f"  {res_str:>15} │ {points_str:>17} │ {format_bytes(dense_mem):>14} │ {format_bytes(qtt_memory):>10} │ {memory_ratio:>10,.0f}×")
    
    print()
    print("  ─────────────────────────────────────────────────────────────────────────")
    print(f"  QTT storage: {format_bytes(qtt_memory)} — constant for bounded rank r={field.max_rank}")
    print()
    
    # Actual sampling demonstration
    print("-" * 74)
    print("  ACTUAL SAMPLING (proving the QTT can be evaluated)")
    print("-" * 74)
    print()
    print("  Extracting grids at increasing resolutions from SAME QTT representation:")
    print("  (Method: Full contraction → resample. Memory = QTT cores only)")
    print()
    
    sample_resolutions = [32, 64, 128, 256]
    
    for res in sample_resolutions:
        start = time.perf_counter()
        samples = field.sample(res)
        elapsed = time.perf_counter() - start
        
        n_points = res * res
        points_per_sec = n_points / elapsed if elapsed > 0 else 0
        
        print(f"  {res:>4}×{res:<4}: {format_number(n_points):>8} points in {elapsed*1000:>6.1f}ms "
              f"| range: [{samples.min():>7.2f}, {samples.max():>7.2f}]")
    
    print()
    
    # Evolution demonstration
    print("-" * 74)
    print("  EVOLVING THE FIELD OVER TIME")
    print("  (Taylor-Green decays as exp(-2νk²t); re-compress at each step)")
    print("-" * 74)
    print()
    print("  Time    │ Field Statistics (128×128)    │ Memory     │ Rank  │ Error")
    print("  ────────┼───────────────────────────────┼────────────┼───────┼──────────")
    
    for step in range(5):
        samples = field.sample(128)
        mem = field.memory_bytes
        
        print(f"  t={field.time:>5.2f} │ min={samples.min():>7.2f} max={samples.max():>7.2f} σ={samples.std():>6.2f} │ {format_bytes(mem):>10} │ {field.max_rank:>5} │ {field.truncation_error:.2e}")
        
        # Evolve
        field.evolve(dt=0.2)
    
    print()
    print("-" * 74)
    print("  CONCLUSION")
    print("-" * 74)
    print(f"""
  The FIELD is physical reality: Taylor-Green vorticity ω(x,y,t).
  
  We stored it in QTT format via real TT-SVD compression:
    • Source: 256×256 dense field (262,144 points)
    • Compressed: {format_bytes(qtt_memory)} with rank r={field.max_rank}
    • Truncation error: {field.truncation_error:.2e}
    
  The QTT can be sampled at any resolution ≤ source:
    • 32×32, 64×64, 128×128, 256×256 — all from same {format_bytes(qtt_memory)} storage
    
  Hypothetical comparison at 1M×1M (if we had that source):
    • Dense memory: {format_bytes(dense_memory_for_resolution(1048576))} (4 TB!)
    • QTT memory:   O(log(N) × r²) — still bounded for bounded rank

  Key qualifier: "resolution-independent" holds *for bounded rank r*.
  Highly turbulent/discontinuous fields may require rank growth.
""")
    
    if interactive:
        run_interactive_visualization(field)


def run_interactive_visualization(field: QTTField):
    """Run interactive matplotlib visualization."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, Button
    except ImportError:
        print("  [matplotlib not installed — skipping interactive visualization]")
        return
    
    print()
    print("-" * 74)
    print("  INTERACTIVE VISUALIZATION")
    print("-" * 74)
    print("  Use the slider to change sample resolution.")
    print("  Memory stays constant because we store QTT cores, not samples.")
    print()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(bottom=0.25)
    
    # Get fixed color range from initial field (prevents "Nintendo" rescaling)
    vmin, vmax = field.field_range
    
    # Initial plot
    resolution = 64
    samples = field.sample(resolution)
    
    im = axes[0].imshow(samples, cmap='RdBu_r', origin='lower',
                        extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Taylor-Green Vorticity at {resolution}×{resolution}')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im, ax=axes[0], label='ω(x,y)')
    
    # Memory comparison bar chart
    qtt_mem = field.memory_bytes
    resolutions_for_bar = [32, 64, 128, 256]
    dense_mems = [dense_memory_for_resolution(r) / 1024 for r in resolutions_for_bar]  # KB
    qtt_mems = [qtt_mem / 1024] * len(resolutions_for_bar)  # KB
    
    x_pos = np.arange(len(resolutions_for_bar))
    width = 0.35
    
    bars1 = axes[1].bar(x_pos - width/2, dense_mems, width, 
                        label='Dense (would need)', color='red', alpha=0.7)
    bars2 = axes[1].bar(x_pos + width/2, qtt_mems, width, 
                        label=f'QTT (actual, r={field.max_rank})', color='green', alpha=0.7)
    
    axes[1].set_ylabel('Memory (KB)')
    axes[1].set_xlabel('Sample Resolution')
    axes[1].set_title('Memory: Dense vs QTT Storage')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f'{r}²' for r in resolutions_for_bar])
    axes[1].legend()
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    # Resolution slider
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Resolution', 16, 256, valinit=64, valstep=16)
    
    # Time step button
    ax_button = plt.axes([0.8, 0.02, 0.1, 0.04])
    button = Button(ax_button, 'Evolve')
    
    def update_resolution(val):
        res = int(val)
        start = time.perf_counter()
        samples = field.sample(res)
        elapsed = time.perf_counter() - start
        
        im.set_data(samples)
        # Keep fixed color scale (no rescaling = honest visualization)
        axes[0].set_title(f'Taylor-Green at {res}×{res} ({elapsed*1000:.1f}ms, QTT: {format_bytes(field.memory_bytes)})')
        fig.canvas.draw_idle()
    
    def evolve_step(event):
        field.evolve(dt=0.2)
        # Update color range for new field state
        vmin, vmax = field.field_range
        im.set_clim(vmin, vmax)
        update_resolution(slider.val)
    
    slider.on_changed(update_resolution)
    button.on_clicked(evolve_step)
    
    plt.suptitle('Resolution Independence: Same QTT, Any Sample Resolution, Constant Memory\n'
                 '(for bounded rank r)',
                 fontsize=12, fontweight='bold')
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Demonstrate resolution-independent sampling from QTT-represented fields'
    )
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Launch interactive matplotlib visualization')
    args = parser.parse_args()
    
    run_demo(interactive=args.interactive)


if __name__ == '__main__':
    main()
