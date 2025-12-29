#!/usr/bin/env python3
"""
Test Event-Driven Optimizations
================================

Measures frame time improvements from:
- Grid caching (event-driven)
- HUD throttling (5 Hz)
- Scissor compositing (dirty rects)
- Colormap bounds (no sync)

Expected: 24ms → 17ms (42 FPS → 58+ FPS)
"""

import torch
import time
import numpy as np
from tensornet.gateway.orbital_command import OrbitalCommandCenter

def main():
    print("="*70)
    print("EVENT-DRIVEN OPTIMIZATION TEST")
    print("="*70)
    
    # Initialize
    print("\n[1/3] Initializing Orbital Command Center...")
    command = OrbitalCommandCenter(width=1920, height=1080, device='cuda:0')
    
    print("\n[2/3] Warming up GPU (5 frames)...")
    for i in range(5):
        frame = command.render_frame()
        torch.cuda.synchronize()
    
    print("\n[3/3] Performance Test (1000 frames)...")
    print("  (Progress every 100 frames...)")
    frame_times = []
    
    start_total = time.perf_counter()
    
    for i in range(1000):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        frame = command.render_frame()
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        frame_time_ms = (end - start) * 1000
        frame_times.append(frame_time_ms)
        
        if i % 100 == 0:
            print(f"  Frame {i:4d}: {frame_time_ms:6.2f}ms ({1000/frame_time_ms:5.1f} FPS)")
    
    end_total = time.perf_counter()
    
    # Statistics
    frame_times = np.array(frame_times)
    mean_time = frame_times.mean()
    min_time = frame_times.min()
    max_time = frame_times.max()
    std_time = frame_times.std()
    
    mean_fps = 1000 / mean_time
    peak_fps = 1000 / min_time
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Mean frame time: {mean_time:.2f}ms ({mean_fps:.1f} FPS)")
    print(f"Peak frame time: {min_time:.2f}ms ({peak_fps:.1f} FPS)")
    print(f"Worst frame time: {max_time:.2f}ms ({1000/max_time:.1f} FPS)")
    print(f"Std deviation: {std_time:.2f}ms")
    print(f"\nTotal time: {(end_total - start_total):.2f}s")
    print(f"Average FPS: {1000 / (end_total - start_total):.1f}")
    
    # Write results to file
    output_file = "validation_1000_frames.txt"
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("1000-FRAME VALIDATION TEST - OPERATION VALHALLA\n")
        f.write("="*70 + "\n\n")
        
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Mean frame time: {mean_time:.2f}ms ({mean_fps:.1f} FPS)\n")
        f.write(f"Peak frame time: {min_time:.2f}ms ({peak_fps:.1f} FPS)\n")
        f.write(f"Worst frame time: {max_time:.2f}ms ({1000/max_time:.1f} FPS)\n")
        f.write(f"Std deviation: {std_time:.2f}ms\n")
        f.write(f"Total time: {(end_total - start_total):.2f}s\n")
        f.write(f"Average FPS: {1000 / (end_total - start_total):.1f}\n\n")
        
        f.write("FRAME TIME DISTRIBUTION\n")
        f.write("-"*70 + "\n")
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            val = np.percentile(frame_times, p)
            f.write(f"  {p:2d}th percentile: {val:.2f}ms ({1000/val:.1f} FPS)\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("DETAILED FRAME TIMES (first 100 frames)\n")
        f.write("="*70 + "\n")
        for i in range(min(100, len(frame_times))):
            f.write(f"Frame {i:4d}: {frame_times[i]:6.2f}ms ({1000/frame_times[i]:6.1f} FPS)\n")
    
    print(f"\n✓ Results written to {output_file}")
    
    # Performance assessment
    print("\n" + "="*70)
    print("ASSESSMENT")
    print("="*70)
    
    if mean_fps >= 60:
        print(f"✓✓✓ EXCELLENT: {mean_fps:.1f} FPS (TARGET EXCEEDED)")
        print(f"    Gap to 60 FPS: {16.67 - mean_time:.2f}ms (idle time)")
    elif mean_fps >= 55:
        print(f"✓✓ GOOD: {mean_fps:.1f} FPS (within 10% of target)")
        print(f"    Gap to 60 FPS: {16.67 - mean_time:.2f}ms")
    elif mean_fps >= 42:
        print(f"✓ IMPROVED: {mean_fps:.1f} FPS (baseline was 42 FPS)")
        print(f"    Gap to 60 FPS: {16.67 - mean_time:.2f}ms")
    else:
        print(f"⚠ NEEDS WORK: {mean_fps:.1f} FPS (below baseline)")
        print(f"    Gap to 60 FPS: {16.67 - mean_time:.2f}ms")
    
    # Check for stability
    if max_time < mean_time * 1.5:
        print(f"✓ Frame times stable (max/mean ratio: {max_time/mean_time:.2f})")
    else:
        print(f"⚠ Frame time spikes detected (max/mean ratio: {max_time/mean_time:.2f})")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
