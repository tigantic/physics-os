"""
Oracle Kernel Overhead Analysis
===============================
Measures the true cost breakdown:
1. Python loop overhead
2. Triton driver overhead  
3. Actual GPU kernel execution
"""

import torch
import triton
import triton.language as tl
import time

torch.set_default_device("cuda")

# -----------------------------------------------------------------------------
# MINIMAL TRITON KERNEL
# -----------------------------------------------------------------------------

@triton.jit
def minimal_kernel(
    out_ptr,
    BLOCK: tl.constexpr
):
    """Minimal kernel - just writes zeros."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.zeros((BLOCK,), dtype=tl.float32))


@triton.jit
def fused_ingest_kernel(
    tick_ptr, encoder_ptr, template_ptr, output_ptr,
    stride_tick_a,
    stride_enc_a, stride_enc_b,
    stride_tpl_b1, stride_tpl_a, stride_tpl_b2,
    stride_out_b1, stride_out_a, stride_out_b2,
    ASSETS: tl.constexpr,
    BOND: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """Full ingest kernel."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    num_elements = BOND * ASSETS * BOND
    mask = offs < num_elements
    
    b2 = offs % BOND
    rem = offs // BOND
    a = rem % ASSETS
    b1 = rem // ASSETS
    
    tpl_off = b1 * stride_tpl_b1 + a * stride_tpl_a + b2 * stride_tpl_b2
    val = tl.load(template_ptr + tpl_off, mask=mask)
    
    tick_val = tl.load(tick_ptr + a * stride_tick_a, mask=mask)
    enc_val = tl.load(encoder_ptr + a * stride_enc_a + b1 * stride_enc_b, mask=mask)
    
    x = tick_val * enc_val
    exp_2x = tl.exp(2.0 * x)
    signal = (exp_2x - 1.0) / (exp_2x + 1.0)
    
    is_diag = (b1 == b2)
    val = val + tl.where(is_diag, signal, 0.0)
    
    out_off = b1 * stride_out_b1 + a * stride_out_a + b2 * stride_out_b2
    tl.store(output_ptr + out_off, val, mask=mask)


def run_analysis():
    print("=" * 60)
    print("  ORACLE KERNEL OVERHEAD ANALYSIS")
    print("=" * 60)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    ASSETS, BOND = 4, 32
    N = 100000
    
    # Setup
    tick = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32)
    encoder = torch.randn(ASSETS, BOND, dtype=torch.float32) * 0.1
    template = torch.randn(BOND, ASSETS, BOND, dtype=torch.float32) * 0.01
    output = torch.empty_like(template)
    
    total_elements = ASSETS * BOND * BOND
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    # Warmup
    print("  Warming up kernels...")
    for _ in range(100):
        minimal_kernel[(1,)](output.view(-1), BLOCK=1024)
        fused_ingest_kernel[grid](
            tick, encoder, template, output,
            tick.stride(0),
            encoder.stride(0), encoder.stride(1),
            template.stride(0), template.stride(1), template.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            ASSETS=ASSETS, BOND=BOND, BLOCK_SIZE=BLOCK_SIZE
        )
    torch.cuda.synchronize()
    
    print(f"  Running {N:,} iterations per test...")
    print()
    
    # Test 1: Empty Python loop
    start = time.perf_counter()
    for _ in range(N):
        pass
    end = time.perf_counter()
    t_python = ((end - start) / N) * 1e6
    print(f"  1. Empty Python loop:      {t_python:.3f} µs")
    
    # Test 2: Minimal Triton kernel
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(N):
        minimal_kernel[(1,)](output.view(-1), BLOCK=1024)
    torch.cuda.synchronize()
    end = time.perf_counter()
    t_minimal = ((end - start) / N) * 1e6
    print(f"  2. Minimal Triton kernel:  {t_minimal:.2f} µs")
    
    # Test 3: Full ingest kernel (no input update)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(N):
        fused_ingest_kernel[grid](
            tick, encoder, template, output,
            tick.stride(0),
            encoder.stride(0), encoder.stride(1),
            template.stride(0), template.stride(1), template.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            ASSETS=ASSETS, BOND=BOND, BLOCK_SIZE=BLOCK_SIZE
        )
    torch.cuda.synchronize()
    end = time.perf_counter()
    t_full = ((end - start) / N) * 1e6
    print(f"  3. Full ingest kernel:     {t_full:.2f} µs")
    
    # Test 4: With tick.copy_ (realistic)
    tick2 = torch.tensor([0.3, 0.4, 0.5, 0.6], device='cuda', dtype=torch.float32)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(N):
        tick.copy_(tick2)
        fused_ingest_kernel[grid](
            tick, encoder, template, output,
            tick.stride(0),
            encoder.stride(0), encoder.stride(1),
            template.stride(0), template.stride(1), template.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            ASSETS=ASSETS, BOND=BOND, BLOCK_SIZE=BLOCK_SIZE
        )
    torch.cuda.synchronize()
    end = time.perf_counter()
    t_copy = ((end - start) / N) * 1e6
    print(f"  4. Full + tick.copy_:      {t_copy:.2f} µs")
    
    # Test 5: torch.add_ baseline (minimal CUDA op)
    a = torch.ones(4, device='cuda')
    b = torch.ones(4, device='cuda')
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(N):
        a.add_(b)
    torch.cuda.synchronize()
    end = time.perf_counter()
    t_add = ((end - start) / N) * 1e6
    print(f"  5. torch.add_ (4 elem):    {t_add:.2f} µs")
    
    print()
    print("─" * 60)
    print("  BREAKDOWN:")
    print(f"    Python overhead:         {t_python:.3f} µs")
    print(f"    Triton driver overhead:  {t_minimal - t_python:.2f} µs")
    print(f"    Kernel compute:          {t_full - t_minimal:.2f} µs")
    print(f"    Input copy:              {t_copy - t_full:.2f} µs")
    print()
    print(f"  TOTAL REALISTIC LATENCY:   {t_copy:.2f} µs")
    print(f"  THROUGHPUT:                {1e6/t_copy:,.0f} ticks/sec")
    
    target = 15.0
    if t_copy <= target:
        print(f"  ✓ TARGET MET: {t_copy:.2f}µs <= {target}µs")
    else:
        print(f"  ✗ TARGET: {t_copy:.2f}µs > {target}µs (need {t_copy/target:.1f}x speedup)")
    print("─" * 60)
    

if __name__ == "__main__":
    run_analysis()
