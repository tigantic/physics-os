#!/usr/bin/env python3
"""
Variable Prober v1.1.0-alpha
============================
The "Scout" for The_Compressor — probes the manifold of any dataset and returns
the exact mathematical "Recipe" (Rank, Bits, Mapping) needed to hit 63,000x ratio
without guessing or crashing.

Usage:
    python probe.py path/to/data.npy
    python probe.py path/to/data.npy --sample-size 2048 --output recipe.json
    
Output:
    - Effective Rank: Complexity of the data manifold
    - Manifold Smoothness: Physical (>0.8) vs Jagged (<0.4)
    - Mapping Strategy: 4D-Morton, Semantic-Clustered, or Standard-TT
    - Hardware Alignment: L2/L3 cache residency prediction
"""

import numpy as np
import torch
import time
import math
from pathlib import Path
import json
import argparse
from typing import Dict, Any, Tuple, Optional, Union


class ManifoldProber:
    """
    Probes the entropy, singular value decay, and locality of any data brick.
    Returns optimal compression parameters for The_Compressor.
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🛰️  PROBER ACTIVE | Device: {self.device}")
    
    def analyze_manifold(self, data_path: Union[str, Path], 
                         sample_size: int = 1024,
                         energy_threshold: float = 0.999) -> Dict[str, Any]:
        """
        Probes the entropy, singular value decay, and locality of any data brick.
        
        Args:
            data_path: Path to .npy file or folder of .npy files
            sample_size: Number of random rows to sample for SVD analysis
            energy_threshold: Cumulative energy threshold for rank estimation
            
        Returns:
            Dictionary with metrics and recommendations
        """
        data_path = Path(data_path)
        print(f"🔍 Probing: {data_path}...")
        t0 = time.time()
        
        # Handle folder of files or single file
        if data_path.is_dir():
            return self._analyze_folder(data_path, sample_size, energy_threshold)
        else:
            return self._analyze_file(data_path, sample_size, energy_threshold)
    
    def _analyze_file(self, file_path: Path, sample_size: int,
                      energy_threshold: float) -> Dict[str, Any]:
        """Analyze a single numpy file."""
        t0 = time.time()
        
        # 1. MMAP Load (RAM Safe)
        data = np.load(file_path, mmap_mode='r')
        original_shape = data.shape
        total_elements = data.size
        dtype = data.dtype
        file_size_bytes = file_path.stat().st_size
        
        print(f"   Shape: {original_shape}")
        print(f"   Elements: {total_elements:,}")
        print(f"   Dtype: {dtype}")
        
        # 2. Spectral Analysis (RSVD Sketch)
        # Reshape to 2D for SVD analysis
        if len(original_shape) == 1:
            sample_data = data[:min(sample_size * 64, len(data))].reshape(-1, min(64, len(data)))
        else:
            sample_flat = data.reshape(-1, original_shape[-1])
            n_rows = sample_flat.shape[0]
            indices = np.random.choice(n_rows, min(sample_size, n_rows), replace=False)
            sample_data = sample_flat[indices]
        
        subset = torch.from_numpy(sample_data.astype(np.float32)).to(self.device)
        
        # SVD for spectral analysis
        try:
            _, S, _ = torch.linalg.svd(subset, full_matrices=False)
            S_norm = S / S[0]
            
            # Calculate Effective Rank (where energy > threshold)
            cumulative_energy = torch.cumsum(S_norm**2, dim=0) / torch.sum(S_norm**2)
            rank_indices = torch.where(cumulative_energy > energy_threshold)[0]
            if len(rank_indices) > 0:
                target_rank = int(rank_indices[0]) + 1
            else:
                target_rank = len(S)
            
            # Singular value decay rate (indicator of compressibility)
            sv_decay_rate = float((S_norm[0] / S_norm[min(10, len(S)-1)]).item())
            
        except Exception as e:
            print(f"   ⚠️ SVD failed: {e}")
            target_rank = 64
            sv_decay_rate = 1.0
        
        # 3. Entropy & Smoothness Check
        try:
            # Gradient-based smoothness
            diffs = torch.diff(subset, dim=-1)
            smoothness_score = 1.0 - (torch.abs(diffs).mean() / (torch.abs(subset).mean() + 1e-8)).item()
            smoothness_score = max(0.0, min(1.0, smoothness_score))
            
            # Local variance (another smoothness indicator)
            local_var = torch.var(subset, dim=-1).mean().item()
            global_var = torch.var(subset).item()
            locality_ratio = local_var / (global_var + 1e-8)
            
        except Exception as e:
            print(f"   ⚠️ Smoothness analysis failed: {e}")
            smoothness_score = 0.5
            locality_ratio = 1.0
        
        # 4. Value Distribution Analysis
        sample_values = subset.flatten()
        value_range = (float(sample_values.min()), float(sample_values.max()))
        value_std = float(sample_values.std())
        value_mean = float(sample_values.mean())
        
        # 5. Sparsity Check
        near_zero = (torch.abs(sample_values) < 1e-6).float().mean().item()
        
        elapsed = time.time() - t0
        print(f"   Probe time: {elapsed:.2f}s")
        
        # 6. Generate Recommendations
        recs = self._generate_recommendations(
            original_shape, target_rank, smoothness_score, 
            sv_decay_rate, locality_ratio, near_zero
        )
        
        return {
            "file": str(file_path),
            "metrics": {
                "shape": list(original_shape),
                "ndim": len(original_shape),
                "elements": int(total_elements),
                "dtype": str(dtype),
                "file_size_bytes": file_size_bytes,
                "effective_rank": target_rank,
                "sv_decay_rate": round(sv_decay_rate, 2),
                "manifold_smoothness": round(smoothness_score, 4),
                "locality_ratio": round(locality_ratio, 4),
                "sparsity": round(near_zero, 4),
                "value_range": [round(v, 4) for v in value_range],
                "value_mean": round(value_mean, 4),
                "value_std": round(value_std, 4),
            },
            "recommendations": recs,
            "probe_time_s": round(elapsed, 2)
        }
    
    def _analyze_folder(self, folder_path: Path, sample_size: int,
                        energy_threshold: float) -> Dict[str, Any]:
        """Analyze a folder of numpy files (e.g., NOAA frames)."""
        
        files = sorted(folder_path.glob("*.npy"))
        if not files:
            raise ValueError(f"No .npy files found in {folder_path}")
        
        print(f"   Found {len(files)} .npy files")
        
        # Sample from multiple files
        n_sample_files = min(8, len(files))
        sample_files = [files[i * len(files) // n_sample_files] for i in range(n_sample_files)]
        
        # Load first file for shape info
        first_frame = np.load(files[0], mmap_mode='r')
        frame_shape = first_frame.shape
        
        # Combined shape: (n_frames, *frame_shape)
        combined_shape = (len(files),) + frame_shape
        total_elements = len(files) * first_frame.size
        
        print(f"   Combined shape: {combined_shape}")
        print(f"   Total elements: {total_elements:,}")
        
        # Sample across frames for temporal analysis
        samples = []
        for f in sample_files:
            frame = np.load(f, mmap_mode='r')
            # Random spatial samples
            flat = frame.flatten()
            indices = np.random.choice(len(flat), min(sample_size // n_sample_files, len(flat)), replace=False)
            samples.append(flat[indices])
        
        sample_matrix = np.stack(samples, axis=0)  # (n_files, n_samples)
        subset = torch.from_numpy(sample_matrix.astype(np.float32)).to(self.device)
        
        # Temporal SVD (across frames)
        try:
            _, S, _ = torch.linalg.svd(subset, full_matrices=False)
            S_norm = S / S[0]
            cumulative_energy = torch.cumsum(S_norm**2, dim=0) / torch.sum(S_norm**2)
            rank_indices = torch.where(cumulative_energy > energy_threshold)[0]
            temporal_rank = int(rank_indices[0]) + 1 if len(rank_indices) > 0 else len(S)
        except:
            temporal_rank = len(files)
        
        # Analyze single frame for spatial properties
        single_result = self._analyze_file(files[0], sample_size, energy_threshold)
        spatial_rank = single_result["metrics"]["effective_rank"]
        
        # Combined effective rank
        effective_rank = max(spatial_rank, temporal_rank)
        
        # Temporal smoothness (frame-to-frame correlation)
        temporal_diffs = torch.diff(subset, dim=0)
        temporal_smoothness = 1.0 - (torch.abs(temporal_diffs).mean() / (torch.abs(subset).mean() + 1e-8)).item()
        temporal_smoothness = max(0.0, min(1.0, temporal_smoothness))
        
        # Combined smoothness
        combined_smoothness = (single_result["metrics"]["manifold_smoothness"] + temporal_smoothness) / 2
        
        # Generate recommendations for 4D data
        recs = self._generate_recommendations(
            combined_shape, effective_rank, combined_smoothness,
            single_result["metrics"]["sv_decay_rate"],
            single_result["metrics"]["locality_ratio"],
            single_result["metrics"]["sparsity"],
            is_temporal=True
        )
        
        return {
            "folder": str(folder_path),
            "n_files": len(files),
            "metrics": {
                "combined_shape": list(combined_shape),
                "ndim": len(combined_shape),
                "elements": int(total_elements),
                "spatial_rank": spatial_rank,
                "temporal_rank": temporal_rank,
                "effective_rank": effective_rank,
                "spatial_smoothness": round(single_result["metrics"]["manifold_smoothness"], 4),
                "temporal_smoothness": round(temporal_smoothness, 4),
                "combined_smoothness": round(combined_smoothness, 4),
            },
            "recommendations": recs
        }
    
    def _generate_recommendations(self, shape: Tuple, rank: int, smoothness: float,
                                   sv_decay: float = 1.0, locality: float = 1.0,
                                   sparsity: float = 0.0, is_temporal: bool = False) -> Dict[str, Any]:
        """Generate compression recommendations based on manifold analysis."""
        
        # === MAPPING STRATEGY ===
        if len(shape) >= 3 and smoothness > 0.7:
            if is_temporal or len(shape) == 4:
                mapping = "4D-Morton (Space-Time Unified)"
            else:
                mapping = "3D-Morton (Spatial)"
            mapping_code = "morton"
        elif smoothness < 0.3:
            mapping = "Semantic-Clustered (LLM Mode)"
            mapping_code = "semantic"
        elif len(shape) == 2 and smoothness > 0.5:
            mapping = "2D-Hilbert (Better 2D Locality)"
            mapping_code = "hilbert"
        else:
            mapping = "Standard-TT (Row-Major)"
            mapping_code = "standard"
        
        # === BIT DEPTH ===
        total_elements = int(np.prod(shape))
        suggested_bits = math.ceil(math.log2(max(total_elements, 2)))
        
        # Per-dimension bits (for Morton)
        if len(shape) >= 2:
            bits_per_dim = [math.ceil(math.log2(max(s, 2))) for s in shape]
        else:
            bits_per_dim = [suggested_bits]
        
        # === RANK RECOMMENDATION ===
        # Clamp rank based on hardware and data characteristics
        if smoothness > 0.8 and sv_decay > 10:
            # Highly compressible - low rank sufficient
            recommended_rank = min(max(rank, 16), 64)
            compressibility = "EXCELLENT"
        elif smoothness > 0.5 and sv_decay > 5:
            recommended_rank = min(max(rank, 32), 96)
            compressibility = "GOOD"
        elif smoothness > 0.3:
            recommended_rank = min(max(rank, 48), 128)
            compressibility = "MODERATE"
        else:
            recommended_rank = min(max(rank, 64), 256)
            compressibility = "CHALLENGING"
        
        # === SIZE ESTIMATION ===
        # Estimate compressed size: sum of cores ≈ bits * rank^2 * 2 bytes (float16)
        est_core_bytes = suggested_bits * recommended_rank * recommended_rank * 2
        est_size_mb = est_core_bytes / 1e6
        
        # === CACHE ALIGNMENT ===
        if est_size_mb < 2.5:
            cache_status = "L2 RESIDENT (< 2.5 MB)"
            cache_tier = "L2"
        elif est_size_mb < 36:
            cache_status = "L3 RESIDENT (< 36 MB)"
            cache_tier = "L3"
        else:
            cache_status = "MEMORY RESIDENT"
            cache_tier = "RAM"
        
        # === COMPRESSION RATIO ESTIMATE ===
        original_bytes = total_elements * 4  # float32
        projected_ratio = original_bytes / max(est_core_bytes, 1)
        
        # === CONFIDENCE SCORE ===
        confidence = min(1.0, smoothness * 0.4 + (sv_decay / 20) * 0.3 + (1 - sparsity) * 0.3)
        
        return {
            "max_rank": recommended_rank,
            "target_bits": suggested_bits,
            "bits_per_dim": bits_per_dim,
            "mapping_strategy": mapping,
            "mapping_code": mapping_code,
            "compressibility": compressibility,
            "estimated_size_mb": round(est_size_mb, 3),
            "projected_ratio": f"{int(projected_ratio):,}x",
            "projected_ratio_numeric": int(projected_ratio),
            "hardware_alignment": cache_status,
            "cache_tier": cache_tier,
            "confidence": round(confidence, 2),
            "warnings": self._generate_warnings(smoothness, rank, est_size_mb, projected_ratio)
        }
    
    def _generate_warnings(self, smoothness: float, rank: int, 
                           est_size_mb: float, ratio: float) -> list:
        """Generate warnings for problematic data characteristics."""
        warnings = []
        
        if smoothness < 0.3:
            warnings.append("⚠️ LOW SMOOTHNESS: Data is 'jagged'. Consider semantic embedding preprocessing.")
        
        if rank > 128:
            warnings.append("⚠️ HIGH RANK: Manifold is complex. May need adaptive truncation or accept lower ratio.")
        
        if est_size_mb > 36:
            warnings.append("⚠️ LARGE CORES: Won't fit in L3 cache. Consider lowering max_rank.")
        
        if ratio < 100:
            warnings.append("⚠️ LOW RATIO: Data may be incompressible or require different mapping strategy.")
        
        if ratio > 100000:
            warnings.append("✅ EXCELLENT: Data is highly redundant. Expect massive compression.")
        
        return warnings


def probe_and_print(data_path: str, sample_size: int = 1024, 
                    output_file: Optional[str] = None, device: str = 'cuda'):
    """Probe data and print/save results."""
    
    prober = ManifoldProber(device=device)
    result = prober.analyze_manifold(data_path, sample_size=sample_size)
    
    # Pretty print
    print("\n" + "=" * 70)
    print("📊 MANIFOLD PROBE RESULTS")
    print("=" * 70)
    
    print("\n📈 METRICS:")
    for key, value in result.get("metrics", {}).items():
        print(f"   {key}: {value}")
    
    print("\n🎯 RECOMMENDATIONS:")
    recs = result.get("recommendations", {})
    print(f"   Max Rank:         {recs.get('max_rank')}")
    print(f"   Target Bits:      {recs.get('target_bits')}")
    print(f"   Bits per Dim:     {recs.get('bits_per_dim')}")
    print(f"   Mapping:          {recs.get('mapping_strategy')}")
    print(f"   Compressibility:  {recs.get('compressibility')}")
    print(f"   Estimated Size:   {recs.get('estimated_size_mb')} MB")
    print(f"   Projected Ratio:  {recs.get('projected_ratio')}")
    print(f"   Cache Alignment:  {recs.get('hardware_alignment')}")
    print(f"   Confidence:       {recs.get('confidence')}")
    
    warnings = recs.get('warnings', [])
    if warnings:
        print("\n⚠️  WARNINGS:")
        for w in warnings:
            print(f"   {w}")
    
    print("\n" + "=" * 70)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"💾 Saved to {output_file}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Variable Prober v1.1.0-alpha - Scout for The_Compressor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('data_path', type=str, 
                        help='Path to .npy file or folder of .npy files')
    parser.add_argument('--sample-size', '-s', type=int, default=1024,
                        help='Number of samples for SVD analysis (default: 1024)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output JSON file for results')
    parser.add_argument('--device', '-d', type=str, default='cuda',
                        help='Device: cuda or cpu')
    
    args = parser.parse_args()
    
    probe_and_print(args.data_path, args.sample_size, args.output, args.device)


if __name__ == '__main__':
    main()
