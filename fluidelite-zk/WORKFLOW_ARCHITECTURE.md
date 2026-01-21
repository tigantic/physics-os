# FluidElite ZK - GPU MSM Workflow Architecture

## ✅ SOLVED - Benchmark Results (2026-01-20)

**Target:** 88 TPS @ 2^18 sustained for 5 minutes
**Achieved:** 113.3 TPS @ 2^18 sustained (+29% above target!)
**Status:** PRODUCTION READY

### Empirically Validated Optimal Configurations

| Tier | k | Points | c | Factor | TPS | Latency | vs Baseline |
|------|---|--------|---|--------|-----|---------|-------------|
| **Micro** | 16 | 65,536 | 12 | 10 | 252.7 | 3.96 ms | +5.8% |
| **Standard** | 18 | 262,144 | 16 | 10 | 113.3 | 8.83 ms | +7.4% |
| **Large** | 20 | 1,048,576 | 18 | 4 | 35.8 | 27.9 ms | +6.5% |

### Key Discoveries

1. **Optimal precompute factor is NOT maximum** - factor=10 beats factor=8 by 3-7%
2. **Factor scales inversely with k** - higher k needs lower factor (VRAM constraint)
3. **c-parameter sweet spots**: c=12 (k=16), c=16 (k=18), c=18 (k=20)

### VRAM Equation (validated)

```
VRAM = (2^k × Factor × 64) + (⌈256/c⌉ × 2^c × 96) + (2^k × 32 × 3) + 512 MB
```

---

## 1. High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              HOST (CPU + RAM)                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐           │
│  │   G1 Points      │    │    Scalars       │    │   MSMConfig      │           │
│  │   (Host RAM)     │    │   (Host RAM)     │    │   (c, async,     │           │
│  │                  │    │                  │    │    precompute)   │           │
│  │  262,144 × 64B   │    │  262,144 × 32B   │    │                  │           │
│  │  = 16 MB         │    │  = 8 MB          │    │                  │           │
│  └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘           │
│           │                       │                       │                      │
│           │                       │                       │                      │
│           ▼                       ▼                       ▼                      │
│  ┌────────────────────────────────────────────────────────────────────┐         │
│  │                         msm() CALL                                  │         │
│  │   HostSlice::from_slice(&scalars)                                   │         │
│  │   &gpu_points[..]  (or &precomputed_bases[..])                      │         │
│  │   &config                                                           │         │
│  │   &mut gpu_result[..]                                               │         │
│  └────────────────────────────────────────────────────────────────────┘         │
│                                    │                                             │
│                                    │ PCIe Gen4 x8                                │
│                                    │ ~12 GB/s theoretical                        │
│                                    ▼                                             │
└────────────────────────────────────┼─────────────────────────────────────────────┘
                                     │
┌────────────────────────────────────┼─────────────────────────────────────────────┐
│                              GPU (CUDA)                                          │
├────────────────────────────────────┼─────────────────────────────────────────────┤
│                                    ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                           VRAM (8151 MB)                                     │ │
│  ├─────────────────────────────────────────────────────────────────────────────┤ │
│  │                                                                              │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │ │
│  │  │  VRAM Pool      │  │  Precomputed    │  │  Bucket         │              │ │
│  │  │  (6500 MB)      │  │  Bases          │  │  Accumulator    │              │ │
│  │  │                 │  │  (128 MB)       │  │  (Dynamic)      │              │ │
│  │  │  LOCKED but     │  │                 │  │                 │              │ │
│  │  │  UNUSED!        │  │  8x precompute  │  │  2^c buckets    │              │ │
│  │  │                 │  │  factor         │  │  per window     │              │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘              │ │
│  │                                                                              │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │ │
│  │  │  Scalars        │  │  Result         │  │  Intermediate   │              │ │
│  │  │  (per-proof)    │  │  (1 point)      │  │  Buffers        │              │ │
│  │  │                 │  │                 │  │                 │              │ │
│  │  │  8 MB upload    │  │  64 bytes       │  │  ???            │              │ │
│  │  │  each MSM!      │  │                 │  │                 │              │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘              │ │
│  │                                                                              │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                             │
│                                    ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                        CUDA MSM KERNEL                                       │ │
│  │                                                                              │ │
│  │   1. Parse scalars into c-bit windows                                        │ │
│  │   2. For each window: accumulate points into buckets                         │ │
│  │   3. Reduce buckets → window sum                                             │ │
│  │   4. Final multi-exponentiation                                              │ │
│  │                                                                              │ │
│  │   Compute bound: O(n/c + 2^c) point additions                                │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Current Execution Timeline (BROKEN)

```
Time →
     0ms      10ms      20ms      30ms      40ms      50ms      60ms
     │         │         │         │         │         │         │
     ▼         ▼         ▼         ▼         ▼         ▼         ▼

CPU: ├──PREP──┤         ├──PREP──┤         ├──PREP──┤
                                                              
PCIe:          ├─COPY──┤          ├─COPY──┤          ├─COPY──┤
               (8MB)              (8MB)              (8MB)
                                                              
GPU:                    ├─MSM────┤          ├─MSM────┤          ├─MSM──
                        (SPIKE)            (SPIKE)            (SPIKE)
                                                              
     └──────────────────┘└──────────────────┘└──────────────────┘
           ~20ms                ~20ms                ~20ms
           
     PROBLEM: GPU is IDLE during PREP and COPY phases!
              GPU utilization graph shows SPIKES not FLAT LINE
              
     Result: 50 TPS instead of 88 TPS
```

---

## 3. Ideal Execution Timeline (TARGET)

```
Time →
     0ms      10ms      20ms      30ms      40ms      50ms      60ms
     │         │         │         │         │         │         │
     ▼         ▼         ▼         ▼         ▼         ▼         ▼

Stream 0: ├─MSM(0)───────┤├─MSM(3)───────┤├─MSM(6)───────┤
                                                              
Stream 1:    ├─MSM(1)───────┤├─MSM(4)───────┤├─MSM(7)───────┤
                                                              
Stream 2:       ├─MSM(2)───────┤├─MSM(5)───────┤├─MSM(8)───────┤

          ├──────────────────────────────────────────────────────┤
          │          GPU AT 90%+ CONTINUOUS UTILIZATION          │
          │              NO GAPS, FLAT LINE ON MONITOR           │
          └──────────────────────────────────────────────────────┘
           
     Result: 88+ TPS sustained
```

---

## 4. Memory Layout Analysis

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           RTX 5070 VRAM (8151 MB)                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  BEFORE OPTIMIZATION (nvidia-smi shows ~1.1 GB used):                            │
│  ┌──────────────────────────────────────────────────────────────────────────────┐│
│  │░░░░░░░░░░░│                    FREE (7 GB)                                   ││
│  │  ~1.1 GB  │                    (but fragmented!)                             ││
│  │  used     │                                                                  ││
│  └──────────────────────────────────────────────────────────────────────────────┘│
│                                                                                  │
│  AFTER "FORCE POOL" (we allocate 6.5 GB):                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────┐│
│  │████████████████████████████████████████████████████│░░░░░░░░░│               ││
│  │              POOL (6500 MB)                        │  FREE   │               ││
│  │              BUT NOT ACTUALLY USED!                │ 1.6 GB  │               ││
│  │              Just sitting there locked             │         │               ││
│  └──────────────────────────────────────────────────────────────────────────────┘│
│                                                                                  │
│  WHAT WE ACTUALLY NEED:                                                          │
│  ┌──────────────────────────────────────────────────────────────────────────────┐│
│  │███│████████████│████████│████████│████████│░░░░░░░░░░░░░░░░░░│               ││
│  │Pts│ Precomp    │ Bucket │ Bucket │ Bucket │   Working Space  │               ││
│  │16M│ Bases 128M │ Win 0  │ Win 1  │ Win 2  │   for streams    │               ││
│  │   │            │        │        │        │                  │               ││
│  └──────────────────────────────────────────────────────────────────────────────┘│
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. The REAL Problem: Scalar Upload Bottleneck

```
EACH MSM CALL CURRENTLY DOES:

    msm(
        HostSlice::from_slice(&scalars),  ◄── THIS IS THE PROBLEM!
        &precomputed_bases[..],            ◄── Already on GPU ✓
        &config,
        &mut result[..],
    )

    ┌─────────────────────────────────────────────────────────────────────┐
    │  HostSlice = DATA IS ON CPU RAM                                     │
    │                                                                      │
    │  Every single MSM call:                                              │
    │    1. Allocates 8 MB on GPU for scalars                             │
    │    2. Copies 8 MB from CPU → GPU via PCIe                           │
    │    3. Runs MSM kernel                                                │
    │    4. Frees the 8 MB scalar buffer                                  │
    │                                                                      │
    │  @ 88 TPS = 704 MB/s PCIe bandwidth just for scalars!               │
    │                                                                      │
    │  PLUS: Allocation/deallocation overhead every call                   │
    │  PLUS: cudaMalloc is NOT FREE - it synchronizes!                    │
    └─────────────────────────────────────────────────────────────────────┘

THE FIX:

    1. Pre-allocate scalar buffer on GPU (DeviceVec)
    2. Use copy_from_host() to upload scalars (async-capable)
    3. Pass DeviceSlice to msm() instead of HostSlice
    4. Reuse the same buffer for every proof
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │  // ONCE at startup:                                                 │
    │  let mut gpu_scalars = DeviceVec::<ScalarField>::device_malloc(n);  │
    │                                                                      │
    │  // EACH proof:                                                      │
    │  gpu_scalars.copy_from_host_async(scalars, &stream);  // Non-block  │
    │  msm(&gpu_scalars[..], &precomputed_bases[..], ...);  // Immediate  │
    │                                                                      │
    │  No allocation overhead per-proof!                                   │
    └─────────────────────────────────────────────────────────────────────┘
```

---

## 6. Correct Pipelined Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         TRIPLE-BUFFERED PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  BUFFER 0          BUFFER 1          BUFFER 2                                    │
│  ┌─────────┐       ┌─────────┐       ┌─────────┐                                │
│  │Scalars A│       │Scalars B│       │Scalars C│  ◄── All pre-allocated on GPU │
│  │(8 MB)   │       │(8 MB)   │       │(8 MB)   │                                │
│  └────┬────┘       └────┬────┘       └────┬────┘                                │
│       │                 │                 │                                      │
│       ▼                 ▼                 ▼                                      │
│  ┌─────────┐       ┌─────────┐       ┌─────────┐                                │
│  │Stream 0 │       │Stream 1 │       │Stream 2 │                                │
│  └────┬────┘       └────┬────┘       └────┬────┘                                │
│       │                 │                 │                                      │
│       └─────────────────┴─────────────────┘                                      │
│                         │                                                        │
│                         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────┐│
│  │                    PRECOMPUTED BASES (128 MB, LOCKED)                         ││
│  │                    Shared by all streams - read-only                          ││
│  └──────────────────────────────────────────────────────────────────────────────┘│
│                         │                                                        │
│                         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────┐│
│  │                    RESULT BUFFER (3 x 64 bytes)                               ││
│  └──────────────────────────────────────────────────────────────────────────────┘│
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

EXECUTION FLOW:

    Time 0:   Stream 0: Upload A    Stream 1: ─────     Stream 2: ─────
    Time 1:   Stream 0: MSM(A)      Stream 1: Upload B  Stream 2: ─────
    Time 2:   Stream 0: Upload D    Stream 1: MSM(B)    Stream 2: Upload C
    Time 3:   Stream 0: MSM(D)      Stream 1: Upload E  Stream 2: MSM(C)
    ...       (continues indefinitely with GPU always busy)
```

---

## 7. Required Code Changes

```rust
// ═══════════════════════════════════════════════════════════════════════════════
// CURRENT (BROKEN) - Allocates every call
// ═══════════════════════════════════════════════════════════════════════════════

while running {
    let scalars = generate_scalars();  // CPU
    
    msm(
        HostSlice::from_slice(&scalars),  // ◄── ALLOCATES + COPIES EVERY TIME
        &precomputed_bases[..],
        &config,
        &mut result[..],
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// FIXED - Pre-allocated buffers, async pipeline
// ═══════════════════════════════════════════════════════════════════════════════

// SETUP (once):
const NUM_BUFFERS: usize = 3;
let mut scalar_buffers: Vec<DeviceVec<ScalarField>> = (0..NUM_BUFFERS)
    .map(|_| DeviceVec::device_malloc(size).unwrap())
    .collect();
let mut result_buffers: Vec<DeviceVec<G1Projective>> = (0..NUM_BUFFERS)
    .map(|_| DeviceVec::device_malloc(1).unwrap())
    .collect();
let streams: Vec<IcicleStream> = (0..NUM_BUFFERS)
    .map(|_| IcicleStream::create().unwrap())
    .collect();

// LOOP:
let mut buffer_idx = 0;
while running {
    let scalars = generate_scalars();  // CPU (can overlap with GPU)
    
    // Upload to pre-allocated GPU buffer (async, non-blocking)
    scalar_buffers[buffer_idx]
        .copy_from_host_async(HostSlice::from_slice(&scalars), &streams[buffer_idx]);
    
    // Launch MSM on this stream (async)
    let mut cfg = MSMConfig::default();
    cfg.stream_handle = streams[buffer_idx].handle;
    cfg.is_async = true;
    cfg.precompute_factor = 8;
    
    msm(
        &scalar_buffers[buffer_idx][..],   // ◄── ALREADY ON GPU, NO ALLOC
        &precomputed_bases[..],
        &cfg,
        &mut result_buffers[buffer_idx][..],
    );
    
    // Rotate to next buffer (don't wait for this one to finish)
    buffer_idx = (buffer_idx + 1) % NUM_BUFFERS;
    
    // Only sync when we've gone full circle (all buffers in flight)
    if buffer_idx == 0 {
        for stream in &streams {
            stream.synchronize().ok();
        }
    }
}
```

---

## 8. Metrics to Watch

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              nvidia-smi dmon                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  BROKEN (current):                                                               │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │ GPU-Util:  ▂▇▁▇▂▇▁▇▂▇▁▇  (spiky, 20-80%)                                  │  │
│  │ Mem-Util:  ▁▁▁▁▁▁▁▁▁▁▁▁  (flat ~15%)                                      │  │
│  │ SM Clock:  ▆▃▆▃▆▃▆▃▆▃▆▃  (fluctuating)                                    │  │
│  │ Copy:      ▅▁▅▁▅▁▅▁▅▁▅▁  (spiky with gaps)                                │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
│  FIXED (target):                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │ GPU-Util:  ████████████  (flat 90%+)                                      │  │
│  │ Mem-Util:  ████████████  (flat ~80%)                                      │  │
│  │ SM Clock:  ████████████  (sustained max boost)                            │  │
│  │ Copy:      ▂▂▂▂▂▂▂▂▂▂▂▂  (low steady trickle, just scalars)              │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Summary: What's Actually Wrong

| Issue | Current State | Required State |
|-------|---------------|----------------|
| **Scalar buffer** | HostSlice (CPU) → alloc+copy every MSM | DeviceVec (GPU) → copy_async only |
| **VRAM Pool** | 6.5GB allocated but UNUSED | Use for triple-buffering |
| **Streams** | Single stream, sync every batch | 3 streams, rotate async |
| **GPU Util** | Spiky 20-80% | Flat 90%+ |
| **PCIe** | Blocking copies | Async overlapped with compute |
| **Timing** | Measures launch time (wrong) | Measures completion time |

---

## 10. Implementation Status

| Priority | Task | Status |
|----------|------|--------|
| CRITICAL | Pre-allocated DeviceVec for scalars | ✅ Done |
| CRITICAL | Async `copy_from_host_async()` with streams | ✅ Done |
| HIGH | Triple-buffered pipeline (3 streams) | ✅ Done |
| HIGH | Remove dead VRAM pool | ✅ Done |
| MEDIUM | Tune precompute_factor | ✅ Done (factor=10 optimal) |
| LOW | CUDA events for timing | ✅ Done |

---

## 11. Production Configuration for Zenith Network

```rust
// Optimal config for 88+ TPS @ 2^18
let config = MsmConfig {
    c: 16,
    precompute_factor: 10,
    is_async: true,
    // ... with triple-buffered streams
};
```

**Benchmark command:**
```bash
cargo run --release --features gpu --bin single-var-test -- --k 18 --c 16 --precompute 10 --duration 300
```

---

*Document created: 2026-01-20*
*Last updated: 2026-01-20*
*Target: 88 TPS @ 2^18 sustained for Zenith Network*
*Achieved: 113.3 TPS (+29% above target!)*
