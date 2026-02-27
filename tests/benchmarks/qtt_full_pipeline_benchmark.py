#!/usr/bin/env python3
"""
Full QTT Pipeline Benchmark Suite

Profiles the complete QTT workflow from raw data to rendered output:
1. Dense → QTT compression (TT-SVD)
2. QTT storage and memory footprint
3. QTT point evaluation
4. QTT 2D slice rendering (our optimized separable path)
5. QTT → Dense decompression
6. End-to-end pipeline timing

This benchmark identifies bottlenecks across the entire workflow.

Run with:
    python tests/benchmarks/qtt_full_pipeline_benchmark.py
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_cuda() -> bool:
    """Check if PyTorch with CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@dataclass
class PipelineConfig:
    """Configuration for pipeline benchmark."""
    # Grid sizes to test
    grid_sizes: list[int]
    # Rank settings
    max_rank: int
    # Resolution for 2D rendering
    render_resolution: tuple[int, int]
    # Number of runs for each benchmark
    n_warmup: int
    n_runs: int


def create_test_data(size: int, pattern: str = "smooth", device: str = "cuda") -> "torch.Tensor":
    """
    Create test data with known structure for benchmarking.
    
    Patterns:
    - smooth: Low-rank smooth function (compresses well)
    - random: Random noise (compresses poorly)
    - shock: Sharp discontinuity (moderate compression)
    """
    import torch
    
    if pattern == "smooth":
        # Smooth periodic function - should compress to low rank
        x = torch.linspace(0, 4 * np.pi, size, device=device)
        return torch.sin(x) * torch.cos(x * 0.5) + 0.5
    
    elif pattern == "random":
        # Random noise - incompressible
        return torch.randn(size, device=device)
    
    elif pattern == "shock":
        # Step function with smooth transition
        x = torch.linspace(-3, 3, size, device=device)
        return torch.tanh(x * 5)  # Sharp sigmoid
    
    elif pattern == "2d_smooth":
        # 2D smooth function
        n = int(np.sqrt(size))
        x = torch.linspace(0, 2 * np.pi, n, device=device)
        y = torch.linspace(0, 2 * np.pi, n, device=device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        return torch.sin(X) * torch.cos(Y)
    
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def benchmark_dense_to_qtt(config: PipelineConfig):
    """Benchmark dense → QTT compression."""
    import torch
    from tensornet.cfd.pure_qtt_ops import dense_to_qtt
    
    print("\n" + "=" * 70)
    print("STAGE 1: DENSE → QTT COMPRESSION (TT-SVD)")
    print("=" * 70)
    
    results = []
    
    for size in config.grid_sizes:
        print(f"\n  Grid size: 2^{int(np.log2(size))} = {size:,} elements")
        
        # Create test data
        data_smooth = create_test_data(size, "smooth")
        data_random = create_test_data(size, "random")
        
        # Warmup
        for _ in range(config.n_warmup):
            _ = dense_to_qtt(data_smooth, max_bond=config.max_rank)
        torch.cuda.synchronize()
        
        # Benchmark smooth data
        times_smooth = []
        ranks_smooth = []
        for _ in range(config.n_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            qtt = dense_to_qtt(data_smooth, max_bond=config.max_rank)
            torch.cuda.synchronize()
            times_smooth.append((time.perf_counter() - start) * 1000)
            ranks_smooth.append(qtt.max_rank)
        
        # Benchmark random data
        times_random = []
        ranks_random = []
        for _ in range(config.n_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            qtt = dense_to_qtt(data_random, max_bond=config.max_rank)
            torch.cuda.synchronize()
            times_random.append((time.perf_counter() - start) * 1000)
            ranks_random.append(qtt.max_rank)
        
        # Calculate compression ratio
        qtt_bytes = sum(c.numel() * c.element_size() for c in qtt.cores)
        dense_bytes = size * 4  # float32
        
        print(f"    Smooth: {np.median(times_smooth):.2f}ms, rank={np.median(ranks_smooth):.0f}")
        print(f"    Random: {np.median(times_random):.2f}ms, rank={np.median(ranks_random):.0f}")
        print(f"    Compression: {dense_bytes/1024:.1f}KB → {qtt_bytes/1024:.1f}KB ({dense_bytes/qtt_bytes:.1f}x)")
        
        results.append({
            "size": size,
            "n_qubits": int(np.log2(size)),
            "smooth_ms": float(np.median(times_smooth)),
            "smooth_rank": float(np.median(ranks_smooth)),
            "random_ms": float(np.median(times_random)),
            "random_rank": float(np.median(ranks_random)),
            "dense_bytes": dense_bytes,
            "qtt_bytes": qtt_bytes,
            "compression_ratio": dense_bytes / qtt_bytes,
        })
    
    return results


def benchmark_qtt_eval(config: PipelineConfig):
    """Benchmark QTT point evaluation."""
    import torch
    from tensornet.cfd.pure_qtt_ops import dense_to_qtt
    
    print("\n" + "=" * 70)
    print("STAGE 2: QTT POINT EVALUATION")
    print("=" * 70)
    
    results = []
    
    for size in config.grid_sizes:
        n_qubits = int(np.log2(size))
        print(f"\n  Grid size: 2^{n_qubits} = {size:,}")
        
        # Create and compress
        data = create_test_data(size, "smooth")
        qtt = dense_to_qtt(data, max_bond=config.max_rank)
        
        # Prepare cores on GPU
        cores = [c.cuda() for c in qtt.cores]
        
        # Single point evaluation
        def eval_single(cores, index):
            """Evaluate QTT at single index."""
            result = cores[0][:, (index >> (n_qubits - 1)) & 1, :]
            for k in range(1, n_qubits):
                bit = (index >> (n_qubits - 1 - k)) & 1
                result = torch.bmm(result.unsqueeze(0), cores[k][:, bit, :].unsqueeze(0)).squeeze(0)
            return result[0, 0]
        
        # Batch evaluation
        def eval_batch(cores, indices):
            """Evaluate QTT at multiple indices."""
            batch_size = len(indices)
            bits = torch.zeros(batch_size, n_qubits, dtype=torch.long, device=cores[0].device)
            for k in range(n_qubits):
                bits[:, k] = (indices >> (n_qubits - 1 - k)) & 1
            
            result = cores[0][:, bits[:, 0], :]  # (r, batch, r)
            result = result.permute(1, 0, 2)  # (batch, r, r)
            
            for k in range(1, n_qubits):
                core_sliced = cores[k][:, bits[:, k], :]  # (r, batch, r)
                core_sliced = core_sliced.permute(1, 0, 2)  # (batch, r, r)
                result = torch.bmm(result, core_sliced)
            
            return result[:, 0, 0]
        
        # Warmup
        for _ in range(10):
            _ = eval_single(cores, 0)
            _ = eval_batch(cores, torch.arange(1000, device='cuda'))
        torch.cuda.synchronize()
        
        # Benchmark single eval
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times_single = []
        for _ in range(100):
            idx = np.random.randint(0, size)
            start_event.record()
            _ = eval_single(cores, idx)
            end_event.record()
            torch.cuda.synchronize()
            times_single.append(start_event.elapsed_time(end_event))
        
        # Benchmark batch eval
        batch_sizes = [100, 1000, 10000, 100000]
        batch_results = {}
        
        for batch_size in batch_sizes:
            if batch_size > size:
                continue
            indices = torch.randint(0, size, (batch_size,), device='cuda')
            
            times = []
            for _ in range(20):
                start_event.record()
                _ = eval_batch(cores, indices)
                end_event.record()
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event))
            
            per_point_us = np.median(times) * 1000 / batch_size
            batch_results[batch_size] = {
                "total_ms": float(np.median(times)),
                "per_point_us": float(per_point_us),
                "throughput_Mpts_s": float(batch_size / np.median(times) / 1000),
            }
            print(f"    Batch {batch_size:>6}: {np.median(times):.3f}ms ({per_point_us:.3f}μs/pt, {batch_size/np.median(times)/1000:.1f} Mpts/s)")
        
        print(f"    Single eval: {np.median(times_single)*1000:.1f}μs")
        
        results.append({
            "size": size,
            "n_qubits": n_qubits,
            "single_eval_us": float(np.median(times_single) * 1000),
            "batch_eval": batch_results,
        })
    
    return results


def benchmark_qtt_render(config: PipelineConfig):
    """Benchmark QTT 2D rendering using our optimized separable path."""
    import torch
    from tensornet.sim.visualization.tensor_slicer import TensorSlicer
    
    print("\n" + "=" * 70)
    print("STAGE 3: QTT 2D RENDERING (SEPARABLE)")
    print("=" * 70)
    
    results = []
    
    # Test different core configurations
    core_configs = [
        (16, 6),   # 16 cores, rank 6
        (20, 8),   # 20 cores, rank 8 (our main config)
        (24, 8),   # 24 cores, rank 8
        (20, 16),  # 20 cores, rank 16
        (20, 32),  # 20 cores, rank 32
    ]
    
    for n_cores, rank in core_configs:
        print(f"\n  Cores: {n_cores}, Rank: {rank}")
        
        # Create test QTT - make sure cores form valid chain
        cores = []
        for i in range(n_cores):
            r_left = 1 if i == 0 else rank
            r_right = 1 if i == n_cores - 1 else rank
            cores.append(np.random.randn(r_left, 2, r_right).astype(np.float32) * 0.1)
        
        slicer = TensorSlicer(cores)
        # Split cores in half for x and y
        n_half = n_cores // 2
        x_cores = list(range(n_half))
        y_cores = list(range(n_half, n_cores))
        
        # Test at multiple resolutions
        resolutions = [(640, 480), (1920, 1080), (3840, 2160)]
        res_results = {}
        
        for width, height in resolutions:
            # Warmup
            for _ in range(5):
                _ = slicer.render_slice_2d_gpu_tensor(x_cores, y_cores, resolution=(width, height), validate=False)
            torch.cuda.synchronize()
            
            # Clear cache for cold start test
            slicer.invalidate_cache()
            torch.cuda.synchronize()
            
            # First frame (cold)
            start = time.perf_counter()
            _ = slicer.render_slice_2d_gpu_tensor(x_cores, y_cores, resolution=(width, height), validate=False)
            torch.cuda.synchronize()
            first_frame_ms = (time.perf_counter() - start) * 1000
            
            # Cached frames
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            times = []
            for _ in range(50):
                start_event.record()
                _ = slicer.render_slice_2d_gpu_tensor(x_cores, y_cores, resolution=(width, height), validate=False)
                end_event.record()
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event))
            
            median_ms = np.median(times)
            fps = 1000 / median_ms
            
            print(f"    {width}x{height}: {median_ms:.3f}ms ({fps:.0f} FPS), cold: {first_frame_ms:.2f}ms")
            
            res_results[f"{width}x{height}"] = {
                "first_frame_ms": float(first_frame_ms),
                "cached_ms": float(median_ms),
                "fps": float(fps),
            }
        
        # Calculate memory footprint
        core_bytes = sum(c.nbytes for c in cores)
        
        results.append({
            "n_cores": n_cores,
            "rank": rank,
            "core_bytes": core_bytes,
            "resolutions": res_results,
        })
    
    return results


def benchmark_qtt_to_dense(config: PipelineConfig):
    """Benchmark QTT → dense decompression."""
    import torch
    from tensornet.cfd.pure_qtt_ops import dense_to_qtt, qtt_to_dense
    
    print("\n" + "=" * 70)
    print("STAGE 4: QTT → DENSE DECOMPRESSION")
    print("=" * 70)
    
    results = []
    
    for size in config.grid_sizes:
        if size > 2**20:  # Skip very large sizes to avoid OOM
            print(f"\n  Skipping 2^{int(np.log2(size))} (too large for dense)")
            continue
            
        n_qubits = int(np.log2(size))
        print(f"\n  Grid size: 2^{n_qubits} = {size:,}")
        
        # Create and compress
        data = create_test_data(size, "smooth")
        qtt = dense_to_qtt(data, max_bond=config.max_rank)
        
        # Warmup
        for _ in range(3):
            _ = qtt_to_dense(qtt)
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(config.n_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = qtt_to_dense(qtt)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        
        # Verify correctness
        error = (result - data).abs().max().item()
        
        print(f"    Time: {np.median(times):.2f}ms")
        print(f"    Max error: {error:.2e}")
        
        results.append({
            "size": size,
            "n_qubits": n_qubits,
            "decompress_ms": float(np.median(times)),
            "max_error": float(error),
        })
    
    return results


def benchmark_end_to_end(config: PipelineConfig):
    """Benchmark complete end-to-end pipeline."""
    import torch
    from tensornet.cfd.qtt_2d import dense_to_qtt_2d, QTT2DState
    from tensornet.sim.visualization.tensor_slicer import TensorSlicer
    
    print("\n" + "=" * 70)
    print("STAGE 5: END-TO-END PIPELINE")
    print("=" * 70)
    
    # Test 2D workflow: Dense 2D → QTT → Render
    grid_2d_sizes = [64, 128, 256, 512]
    
    results = []
    
    for grid_size in grid_2d_sizes:
        n_bits = int(np.log2(grid_size))
        print(f"\n  2D Grid: {grid_size}×{grid_size} ({grid_size**2:,} pixels)")
        
        # Create 2D test data
        x = torch.linspace(0, 2 * np.pi, grid_size, device='cuda')
        y = torch.linspace(0, 2 * np.pi, grid_size, device='cuda')
        X, Y = torch.meshgrid(x, y, indexing='ij')
        field_2d = torch.sin(X) * torch.cos(Y)
        
        # Stage 1: Dense → QTT (Morton + TT-SVD)
        torch.cuda.synchronize()
        start = time.perf_counter()
        qtt_2d = dense_to_qtt_2d(field_2d, max_bond=config.max_rank)
        torch.cuda.synchronize()
        compress_ms = (time.perf_counter() - start) * 1000
        
        # Get QTT stats
        n_cores = len(qtt_2d.cores)
        max_rank = max(c.shape[0] for c in qtt_2d.cores)
        qtt_bytes = sum(c.numel() * c.element_size() for c in qtt_2d.cores)
        dense_bytes = grid_size * grid_size * 4
        
        print(f"    Compress: {compress_ms:.2f}ms (rank={max_rank}, {n_cores} cores)")
        print(f"    Memory: {dense_bytes/1024:.1f}KB → {qtt_bytes/1024:.1f}KB ({dense_bytes/qtt_bytes:.1f}x)")
        
        # Stage 2: Setup slicer with QTT cores
        # Convert QTT2DState cores to numpy for TensorSlicer
        cores_np = [c.cpu().numpy() for c in qtt_2d.cores]
        slicer = TensorSlicer(cores_np)
        
        # Define x/y core split (Morton interleaved)
        # In Morton layout, even cores are X, odd cores are Y
        x_cores = list(range(0, n_cores, 2))[:n_bits]  # Even indices
        y_cores = list(range(1, n_cores, 2))[:n_bits]  # Odd indices
        
        # If we don't have enough separate x/y cores, just split in half
        if len(x_cores) < n_bits or len(y_cores) < n_bits:
            x_cores = list(range(n_cores // 2))
            y_cores = list(range(n_cores // 2, n_cores))
        
        # Stage 3: Render at multiple resolutions
        render_results = {}
        for width, height in [(grid_size, grid_size), (1920, 1080)]:
            # Warmup
            for _ in range(3):
                try:
                    _ = slicer.render_slice_2d_gpu_tensor(x_cores, y_cores, resolution=(width, height), validate=False)
                except Exception:
                    break
            torch.cuda.synchronize()
            
            # Clear cache
            slicer.invalidate_cache()
            
            # Benchmark
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            times = []
            try:
                for _ in range(20):
                    start_event.record()
                    _ = slicer.render_slice_2d_gpu_tensor(x_cores, y_cores, resolution=(width, height), validate=False)
                    end_event.record()
                    torch.cuda.synchronize()
                    times.append(start_event.elapsed_time(end_event))
                
                median_ms = np.median(times)
                render_results[f"{width}x{height}"] = float(median_ms)
                print(f"    Render {width}x{height}: {median_ms:.3f}ms ({1000/median_ms:.0f} FPS)")
            except Exception as e:
                print(f"    Render {width}x{height}: FAILED ({e})")
                render_results[f"{width}x{height}"] = None
        
        # Total pipeline time
        total_ms = compress_ms + (render_results.get("1920x1080", 0) or 0)
        
        results.append({
            "grid_size": grid_size,
            "n_cores": n_cores,
            "max_rank": max_rank,
            "compress_ms": float(compress_ms),
            "qtt_bytes": qtt_bytes,
            "dense_bytes": dense_bytes,
            "compression_ratio": dense_bytes / qtt_bytes,
            "render_ms": render_results,
            "total_pipeline_ms": float(total_ms),
        })
    
    return results


def run_full_pipeline_benchmark():
    """Run the complete pipeline benchmark suite."""
    if not check_cuda():
        print("ERROR: PyTorch with CUDA is required")
        sys.exit(1)
    
    import torch
    
    print("=" * 70)
    print("FULL QTT PIPELINE BENCHMARK")
    print("=" * 70)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    config = PipelineConfig(
        grid_sizes=[2**10, 2**12, 2**14, 2**16, 2**18, 2**20],  # 1K to 1M
        max_rank=32,
        render_resolution=(1920, 1080),
        n_warmup=3,
        n_runs=10,
    )
    
    print(f"\nConfiguration:")
    print(f"  Grid sizes: {[f'2^{int(np.log2(s))}' for s in config.grid_sizes]}")
    print(f"  Max rank: {config.max_rank}")
    print(f"  Render resolution: {config.render_resolution}")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "config": {
            "grid_sizes": config.grid_sizes,
            "max_rank": config.max_rank,
            "render_resolution": config.render_resolution,
        },
    }
    
    # Run all benchmarks
    results["dense_to_qtt"] = benchmark_dense_to_qtt(config)
    results["qtt_eval"] = benchmark_qtt_eval(config)
    results["qtt_render"] = benchmark_qtt_render(config)
    results["qtt_to_dense"] = benchmark_qtt_to_dense(config)
    results["end_to_end"] = benchmark_end_to_end(config)
    
    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│ STAGE                    │ TIME      │ THROUGHPUT              │")
    print("├─────────────────────────────────────────────────────────────────┤")
    
    # Compression (1M elements)
    if results["dense_to_qtt"]:
        last = results["dense_to_qtt"][-1]
        print(f"│ Dense→QTT (2^20)         │ {last['smooth_ms']:>6.2f}ms  │ {last['size']/last['smooth_ms']/1000:.1f} Melem/s          │")
    
    # Evaluation (batch 100K)
    if results["qtt_eval"]:
        for r in results["qtt_eval"]:
            if 100000 in r.get("batch_eval", {}):
                batch = r["batch_eval"][100000]
                print(f"│ QTT Eval (batch 100K)    │ {batch['total_ms']:>6.2f}ms  │ {batch['throughput_Mpts_s']:.1f} Mpts/s             │")
                break
    
    # Rendering (1080p)
    if results["qtt_render"]:
        for r in results["qtt_render"]:
            if r["n_cores"] == 20 and r["rank"] == 8:
                res = r["resolutions"].get("1920x1080", {})
                if res:
                    print(f"│ QTT Render (1080p)       │ {res['cached_ms']:>6.3f}ms │ {res['fps']:.0f} FPS                  │")
                break
    
    # Decompression
    if results["qtt_to_dense"]:
        last = results["qtt_to_dense"][-1]
        print(f"│ QTT→Dense (2^{last['n_qubits']})         │ {last['decompress_ms']:>6.2f}ms  │ {last['size']/last['decompress_ms']/1000:.1f} Melem/s          │")
    
    print("└─────────────────────────────────────────────────────────────────┘")
    
    # Save results
    results_file = PROJECT_ROOT / "tests" / "benchmarks" / "qtt_full_pipeline_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    run_full_pipeline_benchmark()
