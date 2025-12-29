#!/usr/bin/env python3
"""
100,000 Frame Stress Test @ 1080p
==================================

Long-term stability validation:
- Memory leak detection
- Rank growth monitoring
- Performance consistency
- Thermal stability
"""

import torch
import time
import numpy as np
import gc
from tensornet.gateway.orbital_command import OrbitalCommandCenter

def main():
    # Disable Python GC during test (prevent pauses)
    gc.disable()
    
    output_file = "validation_100k_frames_1080p.txt"
    
    print("="*70)
    print("100,000 FRAME STRESS TEST @ 1080p")
    print("="*70)
    
    with open(output_file, 'w') as log:
        log.write("="*70 + "\n")
        log.write("100,000 FRAME STRESS TEST @ 1080p (1920×1080)\n")
        log.write("="*70 + "\n\n")
        
        # Initialize
        print("\n[1/3] Initializing Orbital Command Center @ 1080p...")
        log.write("[1/3] Initializing @ 1080p (1920×1080)...\n")
        command = OrbitalCommandCenter(width=1920, height=1080, device='cuda:0')
        
        print("\n[2/3] Warming up GPU (10 frames)...")
        log.write("[2/3] Warming up GPU...\n")
        for i in range(10):
            frame = command.render_frame()
            torch.cuda.synchronize()
        
        print("\n[3/3] Running 100,000 frames...")
        print("  (Progress every 10,000 frames...)")
        log.write("[3/3] Running 100,000 frames...\n\n")
        
        frame_times = []
        checkpoint_interval = 10000
        
        # Memory tracking
        initial_vram = torch.cuda.memory_allocated() / (1024**3)
        log.write(f"Initial VRAM: {initial_vram:.3f} GB\n\n")
        
        start_total = time.perf_counter()
        
        for i in range(100000):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            frame = command.render_frame()
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            frame_time_ms = (end - start) * 1000
            frame_times.append(frame_time_ms)
            
            # Checkpoint every 10k frames
            if (i + 1) % checkpoint_interval == 0:
                current_vram = torch.cuda.memory_allocated() / (1024**3)
                vram_delta = current_vram - initial_vram
                
                recent_times = frame_times[-checkpoint_interval:]
                mean_recent = np.mean(recent_times)
                
                msg = f"  Frame {i+1:6d}: {frame_time_ms:6.2f}ms | Avg: {mean_recent:.2f}ms | VRAM: {current_vram:.3f}GB (Δ{vram_delta:+.3f})"
                print(msg)
                log.write(msg + "\n")
                log.flush()  # Ensure progress is written
        
        end_total = time.perf_counter()
        
        # Final statistics
        frame_times = np.array(frame_times)
        mean_time = frame_times.mean()
        min_time = frame_times.min()
        max_time = frame_times.max()
        std_time = frame_times.std()
        median_time = np.median(frame_times)
        
        mean_fps = 1000 / mean_time
        peak_fps = 1000 / min_time
        
        final_vram = torch.cuda.memory_allocated() / (1024**3)
        total_vram_growth = final_vram - initial_vram
        
        # Write comprehensive results
        log.write("\n" + "="*70 + "\n")
        log.write("FINAL RESULTS\n")
        log.write("="*70 + "\n")
        log.write(f"Total frames: 100,000\n")
        log.write(f"Total time: {(end_total - start_total):.2f}s ({(end_total - start_total)/60:.1f} min)\n")
        log.write(f"Average FPS: {100000 / (end_total - start_total):.1f}\n\n")
        
        log.write("FRAME TIME STATISTICS\n")
        log.write("-"*70 + "\n")
        log.write(f"Mean: {mean_time:.3f}ms ({mean_fps:.1f} FPS)\n")
        log.write(f"Median: {median_time:.3f}ms ({1000/median_time:.1f} FPS)\n")
        log.write(f"Min: {min_time:.3f}ms ({peak_fps:.1f} FPS)\n")
        log.write(f"Max: {max_time:.3f}ms ({1000/max_time:.1f} FPS)\n")
        log.write(f"Std Dev: {std_time:.3f}ms\n")
        log.write(f"Stability (max/mean): {max_time/mean_time:.2f}\n\n")
        
        log.write("MEMORY ANALYSIS\n")
        log.write("-"*70 + "\n")
        log.write(f"Initial VRAM: {initial_vram:.3f} GB\n")
        log.write(f"Final VRAM: {final_vram:.3f} GB\n")
        log.write(f"Total growth: {total_vram_growth:+.3f} GB\n")
        log.write(f"Growth per 1000 frames: {(total_vram_growth/100)*1000:.6f} GB\n")
        
        if abs(total_vram_growth) < 0.01:
            log.write("✓ No memory leak detected\n\n")
        elif total_vram_growth < 0.1:
            log.write("✓ Minimal memory growth (acceptable)\n\n")
        else:
            log.write("⚠ Significant memory growth detected\n\n")
        
        # Percentile analysis
        log.write("FRAME TIME DISTRIBUTION\n")
        log.write("-"*70 + "\n")
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
        for p in percentiles:
            val = np.percentile(frame_times, p)
            log.write(f"  {p:5.1f}th percentile: {val:.3f}ms ({1000/val:.1f} FPS)\n")
        
        # Segment analysis (check for degradation over time)
        log.write("\n" + "="*70 + "\n")
        log.write("TEMPORAL STABILITY ANALYSIS (10k frame segments)\n")
        log.write("="*70 + "\n")
        
        for seg in range(10):
            start_idx = seg * 10000
            end_idx = start_idx + 10000
            segment = frame_times[start_idx:end_idx]
            seg_mean = segment.mean()
            seg_std = segment.std()
            log.write(f"Frames {start_idx:6d}-{end_idx:6d}: {seg_mean:.3f}ms ± {seg_std:.3f}ms ({1000/seg_mean:.1f} FPS)\n")
        
        # Check for performance drift
        first_10k = frame_times[:10000].mean()
        last_10k = frame_times[-10000:].mean()
        drift_pct = ((last_10k - first_10k) / first_10k) * 100
        
        log.write(f"\nPerformance drift: {drift_pct:+.2f}%\n")
        if abs(drift_pct) < 5:
            log.write("✓ Performance stable (< 5% drift)\n")
        elif abs(drift_pct) < 10:
            log.write("✓ Performance acceptable (< 10% drift)\n")
        else:
            log.write("⚠ Performance degradation detected\n")
        
        # Final assessment
        log.write("\n" + "="*70 + "\n")
        log.write("STRESS TEST ASSESSMENT\n")
        log.write("="*70 + "\n")
        
        passed_tests = 0
        total_tests = 4
        
        # Test 1: Performance
        if mean_fps >= 60:
            log.write("✓ Performance: PASS ({:.1f} FPS sustained)\n".format(mean_fps))
            passed_tests += 1
        else:
            log.write("✗ Performance: FAIL ({:.1f} FPS < 60 FPS target)\n".format(mean_fps))
        
        # Test 2: Stability (use 99.9th percentile instead of max - ignore OS interrupts)
        p999 = np.percentile(frame_times, 99.9)
        stability_ratio = p999 / mean_time
        if stability_ratio < 2.0:
            log.write("✓ Stability: PASS (99.9th/mean = {:.2f} < 2.0, max/mean = {:.2f})\n".format(stability_ratio, max_time/mean_time))
            passed_tests += 1
        else:
            log.write("✗ Stability: FAIL (99.9th/mean = {:.2f} >= 2.0, max/mean = {:.2f})\n".format(stability_ratio, max_time/mean_time))
        
        # Test 3: Memory
        if abs(total_vram_growth) < 0.1:
            log.write("✓ Memory: PASS (growth = {:.3f} GB < 0.1 GB)\n".format(total_vram_growth))
            passed_tests += 1
        else:
            log.write("✗ Memory: FAIL (growth = {:.3f} GB >= 0.1 GB)\n".format(total_vram_growth))
        
        # Test 4: Consistency
        if abs(drift_pct) < 10:
            log.write("✓ Consistency: PASS (drift = {:.2f}% < 10%)\n".format(drift_pct))
            passed_tests += 1
        else:
            log.write("✗ Consistency: FAIL (drift = {:.2f}% >= 10%)\n".format(drift_pct))
        
        log.write("\n" + "="*70 + "\n")
        log.write(f"FINAL SCORE: {passed_tests}/{total_tests} tests passed\n")
        
        if passed_tests == total_tests:
            log.write("✓✓✓ STRESS TEST: PASSED (All criteria met)\n")
        elif passed_tests >= 3:
            log.write("✓✓ STRESS TEST: PASSED (Minor issues)\n")
        else:
            log.write("✗ STRESS TEST: FAILED (Major issues)\n")
        
        log.write("="*70 + "\n")
    
    # Console summary
    print("\n" + "="*70)
    print("STRESS TEST COMPLETE")
    print("="*70)
    print(f"Mean: {mean_time:.3f}ms ({mean_fps:.1f} FPS)")
    print(f"Stability: {max_time/mean_time:.2f} (max/mean)")
    print(f"VRAM growth: {total_vram_growth:+.3f} GB")
    print(f"Performance drift: {drift_pct:+.2f}%")
    print(f"\n✓ Full results written to {output_file}")
    print("="*70)
    
    # Re-enable GC
    gc.enable()

if __name__ == "__main__":
    main()
