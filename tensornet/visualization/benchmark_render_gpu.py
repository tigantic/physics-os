#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                          ║
║             G P U   R E N D E R   B E N C H M A R K   -   6 0   F P S   T E S T         ║
║                                                                                          ║
║   Validates the fix for the render bottleneck (Python loops → GPU batched bmm)          ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

Target: < 16.67ms for 1080p (1920×1080 = 2,073,600 pixels) = 60 FPS

This script benchmarks:
1. CPU vectorized render (baseline - the bottleneck)
2. GPU batched render (the fix)
3. Compares speedup and validates 60 FPS target
"""

import sys
import time
import numpy as np

# Add parent for imports
sys.path.insert(0, str(__file__).rsplit("/", 3)[0])

from tensornet.visualization.tensor_slicer import (
    TensorSlicer,
    create_test_qtt,
    create_sine_qtt,
    _check_torch,
    _check_triton,
)


def format_ms(ms: float) -> str:
    """Format milliseconds with color coding for 60 FPS target."""
    if ms < 16.67:
        return f"\033[92m{ms:.2f}ms\033[0m"  # Green - meets 60 FPS
    elif ms < 33.33:
        return f"\033[93m{ms:.2f}ms\033[0m"  # Yellow - 30 FPS
    else:
        return f"\033[91m{ms:.2f}ms\033[0m"  # Red - below 30 FPS


def run_benchmark():
    """Run comprehensive render benchmark."""
    print()
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                    GPU RENDER BENCHMARK - 60 FPS VALIDATION                  ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print()
    
    # Check GPU availability
    if not _check_torch():
        print("❌ PyTorch CUDA not available - cannot benchmark GPU render")
        print("   Install: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return False
    
    import torch
    props = torch.cuda.get_device_properties(0)
    print(f"✓ GPU: {props.name}")
    print(f"✓ VRAM: {props.total_memory / 1e9:.1f} GB")
    print(f"✓ Triton: {'Available' if _check_triton() else 'Not available (fallback to bmm)'}")
    print()
    
    # Test configurations - only test ranks that Triton supports (≤8)
    configs = [
        ("Sine QTT (analytic, r=2)", create_sine_qtt, {"n_cores": 20, "frequency": 3.0}),
        ("Random QTT (rank=2)", create_test_qtt, {"n_cores": 20, "rank": 2}),
        ("Random QTT (rank=4)", create_test_qtt, {"n_cores": 20, "rank": 4}),
        ("Random QTT (rank=8)", create_test_qtt, {"n_cores": 20, "rank": 8}),
    ]
    
    resolutions = [
        ("720p", (1280, 720)),
        ("1080p", (1920, 1080)),
    ]
    
    results = []
    use_triton = _check_triton()
    
    for config_name, factory, kwargs in configs:
        print(f"═══════════════════════════════════════════════════════════════════════")
        print(f"  {config_name}")
        print(f"═══════════════════════════════════════════════════════════════════════")
        
        slicer = factory(**kwargs)
        n_cores = slicer.n_cores
        n_x = n_cores // 2
        x_cores = list(range(n_x))
        y_cores = list(range(n_x, n_cores))
        
        # Get rank from first middle core
        rank = slicer.cores[1].shape[0] if n_cores > 1 else 1
        print(f"  Cores: {n_cores}, Rank: {rank}, Grid: 2^{n_cores} = {2**n_cores:,} points")
        print()
        
        for res_name, (width, height) in resolutions:
            n_pixels = width * height
            
            # Choose render method
            if use_triton:
                render_fn = lambda: slicer.render_slice_2d_triton(x_cores, y_cores, {}, (width, height))
                method_name = "Triton"
            else:
                render_fn = lambda: slicer.render_slice_2d_gpu(x_cores, y_cores, {}, (width, height))
                method_name = "GPU-bmm"
            
            # Warmup
            for _ in range(3):
                render_fn()
                torch.cuda.synchronize()
            
            # Benchmark
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            n_iters = 10
            for _ in range(n_iters):
                output = render_fn()
                torch.cuda.synchronize()
            gpu_ms = (time.perf_counter() - t0) * 1000 / n_iters  # ms per frame
            
            gpu_fps = 1000 / gpu_ms
            pixels_per_sec = n_pixels / (gpu_ms / 1000)
            
            # Determine if target met
            target_met = "✓ 60 FPS" if gpu_ms < 16.67 else ("≈ 30 FPS" if gpu_ms < 33.33 else "✗ < 30 FPS")
            
            print(f"  {res_name:>5} ({width}×{height}, {n_pixels/1e6:.1f}M pixels):")
            print(f"         {method_name}: {format_ms(gpu_ms)} → {gpu_fps:.1f} FPS  {target_met}")
            print(f"         Throughput: {pixels_per_sec/1e6:.1f}M pixels/sec")
            
            results.append({
                "config": config_name,
                "resolution": res_name,
                "width": width,
                "height": height,
                "gpu_ms": gpu_ms,
                "fps": gpu_fps,
                "rank": rank,
                "target_met": gpu_ms < 16.67,
            })
        
        print()
    
    # Summary
    print("═══════════════════════════════════════════════════════════════════════")
    print("  SUMMARY")
    print("═══════════════════════════════════════════════════════════════════════")
    
    # Check 1080p target
    results_1080p = [r for r in results if r["resolution"] == "1080p"]
    all_met = all(r["target_met"] for r in results_1080p)
    
    if all_met:
        print("  ✓ ALL 1080p configurations meet 60 FPS target!")
    else:
        failures = [r for r in results_1080p if not r["target_met"]]
        print(f"  ✗ {len(failures)}/{len(results_1080p)} 1080p configs failed 60 FPS target:")
        for r in failures:
            print(f"     - {r['config']}: {r['gpu_ms']:.2f}ms ({r['fps']:.1f} FPS)")
    
    print()
    
    # Performance table
    print("  Performance by rank (1080p):")
    print("  " + "-" * 50)
    print(f"  {'Rank':<8} {'Time':<12} {'FPS':<10} {'Target':<10}")
    print("  " + "-" * 50)
    for r in results_1080p:
        rank_str = f"r={r['rank']}"
        time_str = f"{r['gpu_ms']:.2f}ms"
        fps_str = f"{r['fps']:.1f}"
        target_str = "✓ MET" if r["target_met"] else "✗ MISS"
        print(f"  {rank_str:<8} {time_str:<12} {fps_str:<10} {target_str:<10}")
    
    print()
    print("═══════════════════════════════════════════════════════════════════════")
    
    return all_met


def verify_correctness():
    """Verify GPU output matches CPU output."""
    print()
    print("═══════════════════════════════════════════════════════════════════════")
    print("  CORRECTNESS VERIFICATION")
    print("═══════════════════════════════════════════════════════════════════════")
    
    if not _check_torch():
        print("  ✗ GPU not available")
        return False
    
    # Use small resolution for correctness check
    slicer = create_sine_qtt(n_cores=10, frequency=2.0)
    n_x = 5
    x_cores = list(range(n_x))
    y_cores = list(range(n_x, 10))
    
    resolution = (64, 64)
    
    # CPU render
    cpu_output = slicer.render_slice_2d_vectorized(x_cores, y_cores, {}, resolution)
    
    # GPU render
    gpu_output = slicer.render_slice_2d_gpu(x_cores, y_cores, {}, resolution)
    
    # Compare
    max_diff = np.max(np.abs(cpu_output - gpu_output))
    mean_diff = np.mean(np.abs(cpu_output - gpu_output))
    
    print(f"  Resolution: {resolution}")
    print(f"  Max diff:   {max_diff:.2e}")
    print(f"  Mean diff:  {mean_diff:.2e}")
    
    if max_diff < 1e-5:
        print("  ✓ GPU output matches CPU output (within float32 precision)")
        return True
    else:
        print("  ✗ GPU output differs from CPU output!")
        return False


if __name__ == "__main__":
    correct = verify_correctness()
    if correct:
        success = run_benchmark()
        sys.exit(0 if success else 1)
    else:
        print("\n❌ Correctness check failed - benchmark aborted")
        sys.exit(1)
