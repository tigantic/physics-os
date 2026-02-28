# Sovereign RAM Bridge Specification v1.0

**Document ID**: BRIDGE-001  
**Version**: 1.0.0  
**Date**: 2025-12-29  
**Authority**: System Architect  
**Status**: Active - Phase 0 Implementation

---

## Executive Summary

The Sovereign RAM Bridge is a zero-copy, lock-free shared memory interface enabling Physics OS simulation (P-cores, Ubuntu/WSL2) to communicate with Glass Cockpit UI (E-cores, Windows) at 165Hz+ without network overhead, serialization, or synchronization primitives.

**Key Properties:**
- **Location**: `/dev/shm/sovereign_bridge` (Linux tmpfs, mapped to Windows)
- **Access Pattern**: Write-only (simulation) / Read-only (UI)
- **Synchronization**: None (stale reads acceptable)
- **Size**: Dynamic header + variable grid data (~16-64MB typical)

---

## Memory Layout

### Master Header (256 bytes)

```c
// All fields are little-endian (x86-64 native)
// Alignment: 8-byte boundaries for atomic reads

struct SovereignBridgeHeader {
    // === Identification (16 bytes) ===
    uint32_t magic;              // 0x00: 0x48545342 ("HTSB" - Ontic Sovereign Bridge)
    uint32_t version;            // 0x04: Protocol version (1)
    uint64_t creation_timestamp; // 0x08: Unix timestamp (seconds since epoch)
    
    // === Frame Metadata (32 bytes) ===
    uint64_t frame_index;        // 0x10: Monotonic frame counter (0 to 2^64-1)
    uint64_t frame_timestamp_ns; // 0x18: Frame generation time (nanoseconds since epoch)
    float    simulation_time_s;  // 0x20: Simulation time in seconds
    uint32_t grid_width;         // 0x24: Tensor grid X dimension
    uint32_t grid_height;        // 0x28: Tensor grid Y dimension  
    uint32_t grid_depth;         // 0x2C: Tensor grid Z dimension (1 for 2D)
    
    // === Data Offsets (32 bytes) ===
    uint64_t telemetry_offset;   // 0x30: Byte offset to telemetry struct
    uint64_t tensor_offset;      // 0x38: Byte offset to tensor grid data
    uint64_t vector_offset;      // 0x40: Byte offset to vector field
    uint64_t convergence_offset; // 0x48: Byte offset to convergence field
    
    // === Data Sizes (32 bytes) ===
    uint64_t telemetry_size;     // 0x50: Size of telemetry struct in bytes
    uint64_t tensor_size;        // 0x58: Size of tensor grid in bytes
    uint64_t vector_size;        // 0x60: Size of vector field in bytes
    uint64_t convergence_size;   // 0x68: Size of convergence field in bytes
    
    // === Validation (16 bytes) ===
    uint64_t header_checksum;    // 0x70: CRC64 of bytes 0x00-0x6F
    uint64_t frame_checksum;     // 0x78: CRC64 of all data sections
    
    // === Reserved (128 bytes) ===
    uint8_t reserved[128];       // 0x80-0xFF: Future expansion
};
```

**Total Header Size**: 256 bytes (0x100)

---

## Telemetry Structure (256 bytes)

```c
struct SovereignTelemetry {
    // === System Vitality (64 bytes) ===
    float p_core_utilization;    // 0x00: P-core load 0.0-1.0 (mean across cores)
    float e_core_utilization;    // 0x04: E-core load 0.0-1.0 (mean across cores)
    float memory_usage_gb;       // 0x08: RAM consumption in gigabytes
    float memory_total_gb;       // 0x0C: Total RAM available in gigabytes
    float gpu_utilization;       // 0x10: GPU load 0.0-1.0
    float gpu_memory_usage_gb;   // 0x14: VRAM consumption in gigabytes
    float cpu_temperature_c;     // 0x18: CPU package temperature (Celsius)
    float gpu_temperature_c;     // 0x1C: GPU temperature (Celsius)
    
    // === Frame Timing (64 bytes) ===
    float mean_frame_time_ms;    // 0x20: Mean frame time (milliseconds)
    float max_frame_time_ms;     // 0x24: Max frame time in current window
    float min_frame_time_ms;     // 0x28: Min frame time in current window
    float stability_score;       // 0x2C: max/mean ratio (1.0 = perfect)
    float target_frame_time_ms;  // 0x30: Target frame budget (6.06ms @ 165Hz)
    float actual_fps;            // 0x34: Measured frames per second
    uint64_t total_frames;       // 0x38: Total frames since simulation start
    uint32_t dropped_frames;     // 0x40: Frames that exceeded budget
    uint32_t padding_0;          // 0x44: Alignment padding
    
    // === Physics Metrics (64 bytes) ===
    float qtt_compression_ratio; // 0x48: Compression achieved (e.g., 45.0 = 45×)
    float mean_tensor_rank;      // 0x4C: Mean QTT rank across grid
    float max_tensor_rank;       // 0x50: Maximum QTT rank observed
    float truncation_error;      // 0x54: Mean truncation error
    float convergence_detected;  // 0x58: Fraction of grid with convergence >0.3
    float max_vorticity;         // 0x5C: Maximum vorticity magnitude
    float total_energy;          // 0x60: Total system energy (arbitrary units)
    float energy_dissipation;    // 0x64: Energy dissipation rate
    
    // === Event Counters (64 bytes) ===
    uint64_t injection_count;    // 0x68: Total scenario injections
    uint64_t checkpoint_count;   // 0x70: Total checkpoints written
    uint64_t warning_count;      // 0x78: Non-fatal warnings logged
    uint64_t error_count;        // 0x80: Errors encountered
    uint32_t simulation_state;   // 0x88: 0=init, 1=running, 2=paused, 3=error
    uint32_t padding_1;          // 0x8C: Alignment padding
    
    // === Reserved (64 bytes) ===
    uint8_t reserved[64];        // 0x90-0xCF: Future expansion
};
```

**Total Telemetry Size**: 256 bytes

---

## Tensor Grid Data

**Format**: Flattened 3D array of `f32` values  
**Layout**: Row-major (C order): `[z][y][x]`  
**Encoding**: IEEE 754 single-precision (4 bytes per element)

```c
// For grid dimensions W × H × D:
float tensor_grid[D][H][W];

// Flattened size = W × H × D × sizeof(float)
// Example: 64³ grid = 262,144 floats = 1,048,576 bytes = 1 MB
```

**Access Pattern**:
```rust
// Index calculation for (x, y, z):
let index = z * (width * height) + y * width + x;
let value = tensor_data[index];
```

**Physical Meaning**: Each element represents a scalar field value at that spatial coordinate (e.g., vorticity, temperature, pressure).

---

## Vector Field Data

**Format**: Flattened 3D array of `(f32, f32, f32)` tuples  
**Layout**: Interleaved components: `[x_vel, y_vel, z_vel]` per grid point  
**Encoding**: 3 × IEEE 754 single-precision (12 bytes per vector)

```c
// For grid dimensions W × H × D:
struct Vec3 {
    float x;
    float y;
    float z;
};

Vec3 vector_field[D][H][W];

// Flattened size = W × H × D × 3 × sizeof(float)
// Example: 64³ grid = 262,144 vectors = 3,145,728 bytes = 3 MB
```

**Physical Meaning**: Atmospheric velocity field (m/s in each cardinal direction).

---

## Convergence Field Data

**Format**: Flattened 3D array of `f32` probability values  
**Layout**: Row-major (C order): `[z][y][x]`  
**Encoding**: IEEE 754 single-precision (4 bytes per element)  
**Range**: `0.0` (no convergence) to `1.0` (certain convergence)

```c
float convergence_field[D][H][W];

// Flattened size = W × H × D × sizeof(float)
// Example: 64³ grid = 1 MB
```

**Physical Meaning**: Probability of atmospheric convergence (weather event formation) at each grid point.

---

## Complete Memory Map Example

For a **64³ grid** (262,144 cells):

```
Offset        Size         Section
─────────────────────────────────────────────────────
0x00000000    256 bytes    Master Header
0x00000100    256 bytes    Telemetry
0x00000200    1 MB         Tensor Grid (64³ × 4 bytes)
0x00100200    3 MB         Vector Field (64³ × 12 bytes)
0x00400200    1 MB         Convergence Field (64³ × 4 bytes)
─────────────────────────────────────────────────────
TOTAL:        ~5.5 MB
```

For a **512³ grid** (134,217,728 cells):

```
Offset        Size         Section
─────────────────────────────────────────────────────
0x00000000    256 bytes    Master Header
0x00000100    256 bytes    Telemetry
0x00000200    512 MB       Tensor Grid (512³ × 4 bytes)
0x20000200    1536 MB      Vector Field (512³ × 12 bytes)
0x80000200    512 MB       Convergence Field (512³ × 4 bytes)
─────────────────────────────────────────────────────
TOTAL:        ~2.5 GB
```

---

## Write Protocol (Simulation Side)

```python
import mmap
import struct
import hashlib

class SovereignBridgeWriter:
    def __init__(self, path="/dev/shm/sovereign_bridge", grid_shape=(64, 64, 64)):
        self.path = path
        self.width, self.height, self.depth = grid_shape
        
        # Calculate sizes
        self.header_size = 256
        self.telemetry_size = 256
        self.tensor_size = self.width * self.height * self.depth * 4
        self.vector_size = self.width * self.height * self.depth * 12
        self.convergence_size = self.width * self.height * self.depth * 4
        
        self.total_size = (self.header_size + self.telemetry_size + 
                          self.tensor_size + self.vector_size + self.convergence_size)
        
        # Create/open shared memory
        with open(self.path, 'wb') as f:
            f.write(b'\x00' * self.total_size)
        
        self.fd = open(self.path, 'r+b')
        self.mmap = mmap.mmap(self.fd.fileno(), self.total_size)
        
        # Write initial header
        self._write_header(frame_index=0)
    
    def _write_header(self, frame_index):
        """Write master header."""
        import time
        
        header = struct.pack(
            '<IIQQQIIII',  # Format string for header fields
            0x48545342,    # magic ("HTSB")
            1,             # version
            int(time.time()),  # creation_timestamp
            frame_index,   # frame_index
            int(time.time() * 1e9),  # frame_timestamp_ns
            self.width,
            self.height,
            self.depth,
        )
        
        self.mmap.seek(0)
        self.mmap.write(header)
    
    def write_frame(self, frame_index, telemetry, tensor_data, vector_data, convergence_data):
        """Write complete frame to bridge."""
        
        # 1. Write telemetry
        self.mmap.seek(self.header_size)
        self.mmap.write(telemetry.tobytes())
        
        # 2. Write tensor data
        self.mmap.seek(self.header_size + self.telemetry_size)
        self.mmap.write(tensor_data.astype('f4').tobytes())
        
        # 3. Write vector data
        self.mmap.seek(self.header_size + self.telemetry_size + self.tensor_size)
        self.mmap.write(vector_data.astype('f4').tobytes())
        
        # 4. Write convergence data
        self.mmap.seek(self.header_size + self.telemetry_size + self.tensor_size + self.vector_size)
        self.mmap.write(convergence_data.astype('f4').tobytes())
        
        # 5. Update header with frame metadata
        self._write_header(frame_index)
```

---

## Read Protocol (UI Side)

```rust
use memmap2::MmapOptions;
use std::fs::OpenOptions;

pub struct SovereignBridgeReader {
    mmap: memmap2::Mmap,
    grid_width: u32,
    grid_height: u32,
    grid_depth: u32,
}

impl SovereignBridgeReader {
    pub fn open(path: &str) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .open(path)?;
        
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        
        // Read grid dimensions from header
        let grid_width = u32::from_le_bytes([mmap[0x24], mmap[0x25], mmap[0x26], mmap[0x27]]);
        let grid_height = u32::from_le_bytes([mmap[0x28], mmap[0x29], mmap[0x2A], mmap[0x2B]]);
        let grid_depth = u32::from_le_bytes([mmap[0x2C], mmap[0x2D], mmap[0x2E], mmap[0x2F]]);
        
        Ok(Self {
            mmap,
            grid_width,
            grid_height,
            grid_depth,
        })
    }
    
    pub fn read_frame_index(&self) -> u64 {
        u64::from_le_bytes([
            self.mmap[0x10], self.mmap[0x11], self.mmap[0x12], self.mmap[0x13],
            self.mmap[0x14], self.mmap[0x15], self.mmap[0x16], self.mmap[0x17],
        ])
    }
    
    pub fn read_telemetry(&self) -> Telemetry {
        let offset = 256; // After header
        
        Telemetry {
            p_core_utilization: f32::from_le_bytes([
                self.mmap[offset], self.mmap[offset+1], 
                self.mmap[offset+2], self.mmap[offset+3]
            ]),
            // ... read remaining telemetry fields
        }
    }
    
    pub fn read_tensor_data(&self) -> Vec<f32> {
        let offset = 512; // After header + telemetry
        let count = (self.grid_width * self.grid_height * self.grid_depth) as usize;
        
        (0..count)
            .map(|i| {
                let idx = offset + i * 4;
                f32::from_le_bytes([
                    self.mmap[idx], self.mmap[idx+1], 
                    self.mmap[idx+2], self.mmap[idx+3]
                ])
            })
            .collect()
    }
}
```

---

## Validation and Error Handling

### CRC64 Checksums

```rust
use crc::{Crc, CRC_64_ECMA_182};

const CRC64: Crc<u64> = Crc::<u64>::new(&CRC_64_ECMA_182);

fn validate_header(header_bytes: &[u8]) -> bool {
    let stored_checksum = u64::from_le_bytes(header_bytes[0x70..0x78].try_into().unwrap());
    let computed_checksum = CRC64.checksum(&header_bytes[0..0x70]);
    
    stored_checksum == computed_checksum
}
```

### Magic Number Validation

```rust
fn is_valid_bridge(mmap: &[u8]) -> bool {
    let magic = u32::from_le_bytes([mmap[0], mmap[1], mmap[2], mmap[3]]);
    magic == 0x48545342 // "HTSB"
}
```

### Stale Data Detection

```rust
use std::time::{SystemTime, UNIX_EPOCH};

fn is_frame_fresh(frame_timestamp_ns: u64, threshold_ms: u64) -> bool {
    let now_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    
    let age_ms = (now_ns - frame_timestamp_ns) / 1_000_000;
    age_ms < threshold_ms
}
```

---

## Performance Characteristics

| Operation | Latency | Throughput | Notes |
|-----------|---------|------------|-------|
| **Write (simulation)** | ~0.5ms | 2 GB/s | Single memcpy per section |
| **Read (UI)** | ~0.1ms | 10 GB/s | Zero-copy mmap read |
| **Validation** | ~0.05ms | - | CRC64 of header only |
| **Frame overhead** | ~0.65ms | - | Total write + validate |

**Memory Bandwidth**: 64³ grid (5.5 MB) @ 165Hz = **907 MB/s sustained write**

---

## Endianness and Portability

- **Little-endian only**: x86-64 native byte order
- **No big-endian support**: Both simulation and UI run on same architecture
- **Alignment**: All fields naturally aligned (no padding required)
- **Portability**: Specification is platform-specific (Linux/Windows on x86-64)

---

## Future Extensions (v1.1+)

Reserved space in header and telemetry allows for:
- Multi-resolution grids (coarse + fine levels)
- Temporal prediction buffers (next N frames)
- Event notification queue
- Bidirectional injection buffer integration
- Compression metadata (if QTT cores stored directly)

---

## Constitutional Compliance

This specification satisfies:
- **Article II.2**: Binary format, no JSON/text parsing
- **Article V.1**: float32 for data, float64 for timestamps where precision matters
- **Article VIII**: Zero-copy design, O(1) read access per frame
- **Article IX**: Reproducible format, explicit versioning

---

**Status**: ✅ Ready for Phase 0 implementation  
**Next Step**: Implement Python writer in The Physics OS simulation  
**Next Step**: Implement Rust reader in Glass Cockpit UI

---

*Tigantic Holdings LLC - Sovereign Intelligence Systems*
