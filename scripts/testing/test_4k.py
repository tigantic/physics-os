#!/usr/bin/env python3
"""
4K Resolution Test (3840×2160)
==============================

Tests performance scaling at 4× pixels vs 1080p.
"""

import torch
import time
import numpy as np
from tensornet.gateway.orbital_command import OrbitalCommandCenter

def main():
    output_file = "validation_4k_1000frames.txt"
    
    with open(output_file, 'w') as log:
        log.write("="*70 + "\n")
        log.write("4K RESOLUTION TEST (3840×2160) - 1000 FRAMES\n")
        log.write("="*70 + "\n\n")
        
        print("="*70)
        print("4K RESOLUTION TEST (3840×2160)")
        print("="*70)
        
        # Initialize at 4K
        print("\n[1/3] Initializing Orbital Command Center @ 4K...")
        log.write("[1/3] Initializing @ 4K (3840×2160)...\n")
        command = OrbitalCommandCenter(width=3840, height=2160, device='cuda:0')
        
        print("\n[2/3] Warming up GPU (5 frames)...")
        log.write("[2/3] Warming up GPU...\n")
        for i in range(5):
            frame = command.render_frame()
            torch.cuda.synchronize()
        
        print("\n[3/3] Performance Test (1000 frames)...")
        print("  (Progress every 100 frames...)")
        log.write("[3/3] Running 1000 frames...\n\n")
        
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
                msg = f"  Frame {i:4d}: {frame_time_ms:6.2f}ms ({1000/frame_time_ms:5.1f} FPS)"
                print(msg)
                log.write(msg + "\n")
        
        end_total = time.perf_counter()
        
        # Statistics
        frame_times = np.array(frame_times)
        mean_time = frame_times.mean()
        min_time = frame_times.min()
        max_time = frame_times.max()
        std_time = frame_times.std()
        
        mean_fps = 1000 / mean_time
        peak_fps = 1000 / min_time
        
        # Write summary
        log.write("\n" + "="*70 + "\n")
        log.write("RESULTS\n")
        log.write("="*70 + "\n")
        log.write(f"Resolution: 3840×2160 (8,294,400 pixels = 4× 1080p)\n")
        log.write(f"Mean frame time: {mean_time:.2f}ms ({mean_fps:.1f} FPS)\n")
        log.write(f"Peak frame time: {min_time:.2f}ms ({peak_fps:.1f} FPS)\n")
        log.write(f"Worst frame time: {max_time:.2f}ms ({1000/max_time:.1f} FPS)\n")
        log.write(f"Std deviation: {std_time:.2f}ms\n")
        log.write(f"Total time: {(end_total - start_total):.2f}s\n")
        log.write(f"Average FPS: {1000 / (end_total - start_total):.1f}\n")
        log.write(f"Stability ratio (max/mean): {max_time/mean_time:.2f}\n\n")
        
        # Percentiles
        log.write("FRAME TIME DISTRIBUTION\n")
        log.write("-"*70 + "\n")
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            val = np.percentile(frame_times, p)
            log.write(f"  {p:2d}th percentile: {val:.2f}ms ({1000/val:.1f} FPS)\n")
        
        # Comparison to 1080p
        log.write("\n" + "="*70 + "\n")
        log.write("COMPARISON TO 1080p\n")
        log.write("="*70 + "\n")
        baseline_1080p = 2.48  # From previous test
        ratio = mean_time / baseline_1080p
        log.write(f"1080p baseline: {baseline_1080p:.2f}ms (403.5 FPS)\n")
        log.write(f"4K result: {mean_time:.2f}ms ({mean_fps:.1f} FPS)\n")
        log.write(f"Slowdown factor: {ratio:.2f}× ({ratio*100:.0f}% increase)\n")
        log.write(f"Pixels per ms: {8294400/mean_time:.0f} (4K) vs {2073600/baseline_1080p:.0f} (1080p)\n")
        
        if mean_fps >= 60:
            log.write(f"\n✓✓✓ EXCELLENT: {mean_fps:.1f} FPS @ 4K (TARGET EXCEEDED)\n")
        elif mean_fps >= 30:
            log.write(f"\n✓✓ GOOD: {mean_fps:.1f} FPS @ 4K (Playable)\n")
        else:
            log.write(f"\n✓ ACCEPTABLE: {mean_fps:.1f} FPS @ 4K (Cinematic)\n")
        
        # Detailed frames
        log.write("\n" + "="*70 + "\n")
        log.write("DETAILED FRAME TIMES (first 100 frames)\n")
        log.write("="*70 + "\n")
        for i in range(min(100, len(frame_times))):
            log.write(f"Frame {i:4d}: {frame_times[i]:6.2f}ms ({1000/frame_times[i]:6.1f} FPS)\n")
        
        log.write("\n" + "="*70 + "\n")
        log.write("TEST COMPLETE\n")
        log.write("="*70 + "\n")
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Resolution: 3840×2160 (4× 1080p)")
    print(f"Mean frame time: {mean_time:.2f}ms ({mean_fps:.1f} FPS)")
    print(f"Peak frame time: {min_time:.2f}ms ({peak_fps:.1f} FPS)")
    print(f"Worst frame time: {max_time:.2f}ms ({1000/max_time:.1f} FPS)")
    print(f"Std deviation: {std_time:.2f}ms")
    print(f"\nTotal time: {(end_total - start_total):.2f}s")
    print(f"Average FPS: {1000 / (end_total - start_total):.1f}")
    print(f"\n✓ Full results written to {output_file}")
    print("="*70)

if __name__ == "__main__":
    main()
