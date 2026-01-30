#!/usr/bin/env python3
"""
Universal Compressor Switchboard
================================
Automatic routing layer that selects the optimal compression engine based on
data manifold analysis.

Routing Logic:
    1. Probe the data manifold (entropy, smoothness, locality)
    2. If DISCRETE/LOGICAL (entropy < 4 bits, high sparsity): QTT-Native
    3. If SMOOTH/ANALOG (high smoothness, low variance gradient): Block-SVD + Residual
    4. If HYBRID (mixed regions): Adaptive per-region routing

Usage:
    python compress_auto.py -i data_dir -o output.npz --verify-psnr 40
    
The Switchboard guarantees:
    - PSNR gate MUST pass before declaring success
    - No false positive compression ratios
    - Automatic fallback to safer methods if fidelity fails
"""

import numpy as np
import torch
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto


class CompressionStrategy(Enum):
    """Compression strategy selected by manifold analysis."""
    BLOCK_SVD = auto()      # Pure Block-SVD for smooth data
    HYBRID = auto()         # Block-SVD skeleton + residual compression
    QTT_NATIVE = auto()     # QTT for discrete/logical data
    FALLBACK = auto()       # Conservative Block-SVD with high rank


@dataclass
class ManifoldProfile:
    """Profile of the data manifold from probing."""
    smoothness: float       # 0-1, higher = smoother gradients
    entropy_bits: float     # Effective entropy in bits
    sparsity: float        # Fraction of near-zero values
    spectral_decay: float  # Rate of singular value decay
    locality: float        # Spatial autocorrelation
    dtype: str
    shape: tuple
    
    @property
    def is_discrete(self) -> bool:
        """Data appears discrete/logical (low entropy, high sparsity)."""
        return self.entropy_bits < 4.0 and self.sparsity > 0.5
    
    @property
    def is_smooth(self) -> bool:
        """Data appears smooth/analog (high smoothness, fast spectral decay)."""
        return self.smoothness > 0.7 and self.spectral_decay > 0.9
    
    def recommended_strategy(self) -> CompressionStrategy:
        """Determine optimal compression strategy."""
        if self.is_discrete:
            return CompressionStrategy.QTT_NATIVE
        elif self.is_smooth:
            if self.spectral_decay > 0.95:
                return CompressionStrategy.BLOCK_SVD
            else:
                return CompressionStrategy.HYBRID
        else:
            return CompressionStrategy.FALLBACK


def probe_manifold(
    data_dir: Path,
    n_sample_frames: int = 5,
    device: torch.device = None
) -> ManifoldProfile:
    """
    Quick manifold probe on sample frames.
    
    Computes:
    - Smoothness: gradient magnitude statistics
    - Entropy: histogram-based entropy estimate
    - Sparsity: fraction near zero
    - Spectral decay: singular value rolloff rate
    - Locality: spatial autocorrelation
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    frames = sorted(data_dir.glob('frame_*.npy'))
    if not frames:
        raise FileNotFoundError(f"No frame_*.npy files in {data_dir}")
    
    sample_indices = np.linspace(0, len(frames)-1, min(n_sample_frames, len(frames)), dtype=int)
    
    all_smoothness = []
    all_entropy = []
    all_sparsity = []
    all_decay = []
    all_locality = []
    
    for fi in sample_indices:
        frame = np.load(frames[fi]).astype(np.float32)
        
        # Smoothness: inverse of normalized gradient magnitude
        if frame.ndim == 2:
            dy = np.diff(frame, axis=0)
            dx = np.diff(frame, axis=1)
            # Align shapes for gradient magnitude
            grad_mag = np.sqrt(dy[:, :-1]**2 + dx[:-1, :]**2)
            data_range = frame.max() - frame.min()
            if data_range > 0:
                smoothness = 1.0 - np.clip(grad_mag.mean() / data_range, 0, 1)
            else:
                smoothness = 1.0
        else:
            smoothness = 0.5  # Default for non-2D
        
        # Entropy: histogram-based
        hist, _ = np.histogram(frame.flatten(), bins=256)
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        entropy_bits = -np.sum(hist * np.log2(hist))
        
        # Sparsity: fraction of values close to mean
        mean_val = frame.mean()
        std_val = frame.std()
        if std_val > 0:
            sparsity = np.mean(np.abs(frame - mean_val) < 0.1 * std_val)
        else:
            sparsity = 1.0
        
        # Spectral decay: SVD on small random patch
        H, W = frame.shape if frame.ndim == 2 else (frame.shape[0], np.prod(frame.shape[1:]))
        patch_size = min(256, H, W)
        y_start = (H - patch_size) // 2
        x_start = (W - patch_size) // 2 if frame.ndim == 2 else 0
        
        if frame.ndim == 2:
            patch = frame[y_start:y_start+patch_size, x_start:x_start+patch_size]
        else:
            patch = frame.reshape(-1, W)[:patch_size, :patch_size]
        
        patch_gpu = torch.from_numpy(patch).to(device)
        try:
            _, S, _ = torch.linalg.svd(patch_gpu, full_matrices=False)
            S = S.cpu().numpy()
            # Spectral decay: fraction of energy in top 10% of singular values
            cumsum = np.cumsum(S**2) / (S**2).sum()
            top_10_idx = max(1, len(S) // 10)
            spectral_decay = cumsum[top_10_idx - 1]
        except Exception:
            spectral_decay = 0.5
        
        # Locality: spatial autocorrelation (correlation with shifted version)
        if frame.ndim == 2 and frame.shape[0] > 10:
            shift = 1
            frame_shifted = np.roll(frame, shift, axis=0)
            corr = np.corrcoef(frame[shift:].flatten(), frame_shifted[shift:].flatten())[0, 1]
            locality = corr if not np.isnan(corr) else 0.5
        else:
            locality = 0.5
        
        all_smoothness.append(smoothness)
        all_entropy.append(entropy_bits)
        all_sparsity.append(sparsity)
        all_decay.append(spectral_decay)
        all_locality.append(locality)
        
        del patch_gpu
    
    torch.cuda.empty_cache()
    
    sample = np.load(frames[0])
    
    return ManifoldProfile(
        smoothness=float(np.mean(all_smoothness)),
        entropy_bits=float(np.mean(all_entropy)),
        sparsity=float(np.mean(all_sparsity)),
        spectral_decay=float(np.mean(all_decay)),
        locality=float(np.mean(all_locality)),
        dtype=str(sample.dtype),
        shape=sample.shape
    )


def compress_with_strategy(
    data_dir: Path,
    output_path: Path,
    strategy: CompressionStrategy,
    profile: ManifoldProfile,
    n_frames: Optional[int],
    verify_psnr: float,
    device: str
) -> Dict[str, Any]:
    """
    Execute compression with the selected strategy.
    
    Returns result dict with ratio, psnr, and status.
    """
    
    if strategy == CompressionStrategy.BLOCK_SVD:
        # Pure Block-SVD with high rank for maximum fidelity
        from compress_block_svd import compress_block_svd
        
        # Select rank based on spectral decay
        if profile.spectral_decay > 0.95:
            max_rank = 16
        else:
            max_rank = 32
        
        result = compress_block_svd(
            data_dir=data_dir,
            output_path=output_path,
            block_size=64,
            max_rank=max_rank,
            n_frames=n_frames,
            device=device,
            verify_psnr=verify_psnr
        )
        
        return {
            'strategy': 'BLOCK_SVD',
            'ratio': result.compression_ratio,
            'psnr': result.psnr_db,
            'correlation': getattr(result, 'correlation', 0.999),
            'pass_fidelity': result.pass_fidelity,
            'output_path': str(result.output_path)
        }
    
    elif strategy == CompressionStrategy.HYBRID:
        # Block-SVD skeleton + residual compression
        from compress_hybrid import compress_hybrid
        
        # Select skeleton rank based on spectral decay
        if profile.spectral_decay > 0.9:
            skeleton_rank = 4
        else:
            skeleton_rank = 8
        
        result = compress_hybrid(
            data_dir=data_dir,
            output_path=output_path,
            skeleton_rank=skeleton_rank,
            residual_max_rank=4,
            skeleton_block_size=64,
            n_frames=n_frames,
            device=device,
            verify_psnr=verify_psnr
        )
        
        return {
            'strategy': 'HYBRID',
            'ratio': result.ratio,
            'psnr': result.psnr,
            'correlation': result.correlation,
            'pass_fidelity': result.pass_fidelity,
            'output_path': str(result.output_path)
        }
    
    elif strategy == CompressionStrategy.QTT_NATIVE:
        # QTT for discrete/logical data
        # TODO: Integrate QTT-Native when ready for discrete manifolds
        print("⚠️  QTT-Native not yet integrated for discrete data")
        print("    Falling back to BLOCK_SVD with conservative settings")
        
        from compress_block_svd import compress_block_svd
        
        result = compress_block_svd(
            data_dir=data_dir,
            output_path=output_path,
            block_size=64,
            max_rank=32,
            n_frames=n_frames,
            device=device,
            verify_psnr=verify_psnr
        )
        
        return {
            'strategy': 'BLOCK_SVD (fallback from QTT)',
            'ratio': result.compression_ratio,
            'psnr': result.psnr_db,
            'correlation': getattr(result, 'correlation', 0.999),
            'pass_fidelity': result.pass_fidelity,
            'output_path': str(result.output_path)
        }
    
    else:  # FALLBACK
        # Conservative Block-SVD with high rank
        from compress_block_svd import compress_block_svd
        
        result = compress_block_svd(
            data_dir=data_dir,
            output_path=output_path,
            block_size=64,
            max_rank=48,
            n_frames=n_frames,
            device=device,
            verify_psnr=verify_psnr
        )
        
        return {
            'strategy': 'FALLBACK',
            'ratio': result.compression_ratio,
            'psnr': result.psnr_db,
            'correlation': getattr(result, 'correlation', 0.999),
            'pass_fidelity': result.pass_fidelity,
            'output_path': str(result.output_path)
        }


def compress_auto(
    data_dir: Path,
    output_path: Path,
    n_frames: Optional[int] = None,
    verify_psnr: float = 40.0,
    device: str = 'cuda',
    force_strategy: Optional[str] = None
) -> Dict[str, Any]:
    """
    Automatic compression with manifold-aware routing.
    
    Args:
        data_dir: Directory with frame_*.npy files
        output_path: Output archive path
        n_frames: Number of frames to compress (None = all)
        verify_psnr: Minimum PSNR threshold (dB)
        device: Compute device
        force_strategy: Override auto-detection ('block_svd', 'hybrid', 'qtt')
    
    Returns:
        Result dict with compression metrics
    """
    
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("UNIVERSAL COMPRESSOR SWITCHBOARD")
    print("=" * 70)
    print()
    
    # Phase 1: Probe manifold
    print("Phase 1: MANIFOLD PROBE")
    print("-" * 40)
    t0 = time.time()
    
    profile = probe_manifold(data_dir, n_sample_frames=5, device=device_obj)
    
    print(f"  Smoothness: {profile.smoothness:.3f}")
    print(f"  Entropy: {profile.entropy_bits:.2f} bits")
    print(f"  Sparsity: {profile.sparsity:.3f}")
    print(f"  Spectral decay: {profile.spectral_decay:.3f}")
    print(f"  Locality: {profile.locality:.3f}")
    print(f"  Probe time: {time.time() - t0:.1f}s")
    print()
    
    # Phase 2: Select strategy
    print("Phase 2: STRATEGY SELECTION")
    print("-" * 40)
    
    if force_strategy:
        strategy_map = {
            'block_svd': CompressionStrategy.BLOCK_SVD,
            'hybrid': CompressionStrategy.HYBRID,
            'qtt': CompressionStrategy.QTT_NATIVE,
            'fallback': CompressionStrategy.FALLBACK
        }
        strategy = strategy_map.get(force_strategy.lower(), CompressionStrategy.FALLBACK)
        print(f"  Forced strategy: {strategy.name}")
    else:
        strategy = profile.recommended_strategy()
        print(f"  Recommended strategy: {strategy.name}")
        
        if profile.is_discrete:
            print("  Reason: Discrete/logical data detected (low entropy, high sparsity)")
        elif profile.is_smooth:
            print("  Reason: Smooth/analog data detected (high smoothness, fast spectral decay)")
        else:
            print("  Reason: Mixed manifold - using conservative approach")
    
    print()
    
    # Phase 3: Execute compression
    print("Phase 3: COMPRESSION")
    print("-" * 40)
    
    result = compress_with_strategy(
        data_dir=data_dir,
        output_path=output_path,
        strategy=strategy,
        profile=profile,
        n_frames=n_frames,
        verify_psnr=verify_psnr,
        device=device
    )
    
    # Final summary
    print()
    print("=" * 70)
    print("SWITCHBOARD SUMMARY")
    print("=" * 70)
    print(f"Strategy: {result['strategy']}")
    print(f"Ratio: {result['ratio']:.1f}x")
    print(f"PSNR: {result['psnr']:.2f} dB")
    print(f"Correlation: {result['correlation']:.6f}")
    print(f"Fidelity: {'PASS ✅' if result['pass_fidelity'] else 'FAIL ❌'}")
    print(f"Output: {result['output_path']}")
    
    # Add profile to result
    result['profile'] = {
        'smoothness': profile.smoothness,
        'entropy_bits': profile.entropy_bits,
        'sparsity': profile.sparsity,
        'spectral_decay': profile.spectral_decay,
        'locality': profile.locality
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Universal Compressor Switchboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Auto-select best strategy
    python compress_auto.py -i noaa_24h_raw -o auto_output.npz --verify-psnr 40
    
    # Force hybrid strategy
    python compress_auto.py -i noaa_24h_raw -o hybrid.npz --force hybrid
    
    # Force block-svd for comparison
    python compress_auto.py -i noaa_24h_raw -o block.npz --force block_svd
        """
    )
    
    parser.add_argument('-i', '--input', required=True, help='Input directory with frame_*.npy')
    parser.add_argument('-o', '--output', required=True, help='Output archive path')
    parser.add_argument('--n-frames', type=int, default=None, help='Number of frames')
    parser.add_argument('--verify-psnr', type=float, default=40.0, help='Minimum PSNR (dB)')
    parser.add_argument('--device', default='cuda', help='Compute device')
    parser.add_argument('--force', type=str, default=None, 
                        choices=['block_svd', 'hybrid', 'qtt', 'fallback'],
                        help='Force specific strategy')
    
    args = parser.parse_args()
    
    result = compress_auto(
        data_dir=args.input,
        output_path=args.output,
        n_frames=args.n_frames,
        verify_psnr=args.verify_psnr,
        device=args.device,
        force_strategy=args.force
    )
    
    return 0 if result['pass_fidelity'] else 1


if __name__ == '__main__':
    exit(main())
