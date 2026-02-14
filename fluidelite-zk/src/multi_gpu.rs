//! Multi-GPU Support
//!
//! Device selection via `CUDA_VISIBLE_DEVICES` and round-robin dispatch
//! for batched proofs across multiple GPUs.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │                  MultiGpuDispatcher                          │
//! │                                                             │
//! │  ┌───────────┐  ┌───────────┐  ┌───────────┐               │
//! │  │  GPU 0    │  │  GPU 1    │  │  GPU 2    │  ...           │
//! │  │ (active)  │  │ (active)  │  │ (active)  │               │
//! │  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘               │
//! │        │              │              │                      │
//! │  ┌─────▼──────────────▼──────────────▼─────┐               │
//! │  │        Round-Robin Job Queue             │               │
//! │  │  Job₀→GPU₀  Job₁→GPU₁  Job₂→GPU₂  ...  │               │
//! │  └──────────────────────────────────────────┘               │
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,no_run
//! use fluidelite_zk::multi_gpu::{MultiGpuDispatcher, GpuDeviceInfo};
//!
//! let dispatcher = MultiGpuDispatcher::detect().unwrap();
//! println!("Found {} GPUs", dispatcher.device_count());
//!
//! // Dispatch work round-robin.
//! let device = dispatcher.next_device();
//! println!("Dispatching to GPU {}: {}", device.index, device.name);
//! ```

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Information about a single GPU device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDeviceInfo {
    /// Device index (0-based).
    pub index: usize,
    /// Device name (e.g., "NVIDIA GeForce RTX 5070").
    pub name: String,
    /// Total VRAM in MiB.
    pub vram_mib: usize,
    /// CUDA compute capability (major, minor).
    pub compute_capability: (u32, u32),
    /// Whether this device is available (i.e., not in exclusive mode).
    pub available: bool,
}

/// Statistics for a single GPU device.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeviceStats {
    /// Total jobs dispatched to this device.
    pub total_jobs: u64,
    /// Total computation time (milliseconds).
    pub total_compute_ms: u64,
    /// Average job latency (milliseconds).
    pub avg_latency_ms: f64,
}

/// Round-robin dispatcher across multiple GPU devices.
///
/// Thread-safe: uses atomic counter for round-robin index.
pub struct MultiGpuDispatcher {
    /// Detected GPU devices.
    devices: Vec<GpuDeviceInfo>,
    /// Round-robin counter (atomic for thread safety).
    next_index: AtomicUsize,
    /// Per-device statistics.
    device_stats: Vec<std::sync::Mutex<DeviceStats>>,
}

impl MultiGpuDispatcher {
    /// Detect available GPU devices.
    ///
    /// Reads `CUDA_VISIBLE_DEVICES` environment variable to determine
    /// which GPUs are available. Falls back to detecting all GPUs if
    /// the env var is not set.
    pub fn detect() -> Result<Self, String> {
        let devices = detect_gpu_devices()?;

        if devices.is_empty() {
            return Err("no GPU devices detected".into());
        }

        let device_stats = devices
            .iter()
            .map(|_| std::sync::Mutex::new(DeviceStats::default()))
            .collect();

        tracing::info!(
            gpu_count = devices.len(),
            devices = ?devices.iter().map(|d| &d.name).collect::<Vec<_>>(),
            "multi-GPU dispatcher initialized"
        );

        Ok(Self {
            devices,
            next_index: AtomicUsize::new(0),
            device_stats,
        })
    }

    /// Create a dispatcher from a pre-configured set of devices.
    pub fn from_devices(devices: Vec<GpuDeviceInfo>) -> Result<Self, String> {
        if devices.is_empty() {
            return Err("no devices provided".into());
        }

        let device_stats = devices
            .iter()
            .map(|_| std::sync::Mutex::new(DeviceStats::default()))
            .collect();

        Ok(Self {
            devices,
            next_index: AtomicUsize::new(0),
            device_stats,
        })
    }

    /// Number of available GPU devices.
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Get the next device via round-robin selection.
    ///
    /// Thread-safe and lock-free.
    pub fn next_device(&self) -> &GpuDeviceInfo {
        let idx = self.next_index.fetch_add(1, Ordering::Relaxed) % self.devices.len();
        &self.devices[idx]
    }

    /// Get a specific device by index.
    pub fn device(&self, index: usize) -> Option<&GpuDeviceInfo> {
        self.devices.get(index)
    }

    /// All detected devices.
    pub fn devices(&self) -> &[GpuDeviceInfo] {
        &self.devices
    }

    /// Record a completed job on a device.
    pub fn record_job(&self, device_index: usize, latency_ms: u64) {
        if let Some(stats_lock) = self.device_stats.get(device_index) {
            if let Ok(mut stats) = stats_lock.lock() {
                stats.total_jobs += 1;
                stats.total_compute_ms += latency_ms;
                stats.avg_latency_ms = if stats.total_jobs > 0 {
                    stats.total_compute_ms as f64 / stats.total_jobs as f64
                } else {
                    0.0
                };
            }
        }
    }

    /// Get statistics for all devices.
    pub fn all_stats(&self) -> Vec<(GpuDeviceInfo, DeviceStats)> {
        self.devices
            .iter()
            .zip(self.device_stats.iter())
            .map(|(dev, stats_lock)| {
                let stats = stats_lock.lock().map(|s| s.clone()).unwrap_or_default();
                (dev.clone(), stats)
            })
            .collect()
    }

    /// Total throughput across all GPUs (jobs per second).
    pub fn total_throughput(&self) -> f64 {
        let total_compute_ms: u64 = self
            .device_stats
            .iter()
            .filter_map(|s| s.lock().ok())
            .map(|s| s.total_compute_ms)
            .sum();

        let total_jobs: u64 = self
            .device_stats
            .iter()
            .filter_map(|s| s.lock().ok())
            .map(|s| s.total_jobs)
            .sum();

        if total_compute_ms == 0 {
            0.0
        } else {
            // Throughput assuming GPUs run in parallel: total_jobs / max_device_time
            let max_device_ms = self
                .device_stats
                .iter()
                .filter_map(|s| s.lock().ok())
                .map(|s| s.total_compute_ms)
                .max()
                .unwrap_or(0);

            if max_device_ms == 0 {
                0.0
            } else {
                (total_jobs as f64) / (max_device_ms as f64 / 1000.0)
            }
        }
    }

    /// Dispatch a batch of work items across GPUs round-robin.
    ///
    /// Returns a mapping of (device_index, work_items) for the caller
    /// to execute on each device.
    pub fn dispatch_batch<T: Clone>(&self, items: &[T]) -> Vec<(usize, Vec<T>)> {
        let n = self.devices.len();
        let mut buckets: Vec<Vec<T>> = vec![Vec::new(); n];

        for (i, item) in items.iter().enumerate() {
            buckets[i % n].push(item.clone());
        }

        buckets
            .into_iter()
            .enumerate()
            .filter(|(_, b)| !b.is_empty())
            .collect()
    }
}

/// Detect GPU devices from the system.
///
/// Respects `CUDA_VISIBLE_DEVICES` environment variable.
fn detect_gpu_devices() -> Result<Vec<GpuDeviceInfo>, String> {
    // Parse CUDA_VISIBLE_DEVICES if set.
    let visible_devices = std::env::var("CUDA_VISIBLE_DEVICES").ok();

    let allowed_indices: Option<Vec<usize>> = visible_devices.as_ref().map(|val| {
        val.split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect()
    });

    // Use ICICLE runtime for actual device detection when GPU feature is enabled.
    #[cfg(feature = "gpu")]
    {
        detect_icicle_devices(allowed_indices)
    }

    #[cfg(not(feature = "gpu"))]
    {
        detect_from_nvidia_smi(allowed_indices)
    }
}

/// Detect devices using ICICLE runtime.
#[cfg(feature = "gpu")]
fn detect_icicle_devices(allowed_indices: Option<Vec<usize>>) -> Result<Vec<GpuDeviceInfo>, String> {
    use icicle_runtime::Device;

    // ICICLE uses a device abstraction. We detect via the CUDA backend.
    // For now, detect via nvidia-smi fallback since ICICLE doesn't expose
    // per-device metadata directly.
    detect_from_nvidia_smi(allowed_indices)
}

/// Detect devices by parsing `nvidia-smi` output.
fn detect_from_nvidia_smi(allowed_indices: Option<Vec<usize>>) -> Result<Vec<GpuDeviceInfo>, String> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .map_err(|e| format!("nvidia-smi not found or failed: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "nvidia-smi failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut devices = Vec::new();

    for line in stdout.lines() {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 4 {
            continue;
        }

        let index = parts[0].parse::<usize>().unwrap_or(0);

        // Filter by CUDA_VISIBLE_DEVICES if set.
        if let Some(ref allowed) = allowed_indices {
            if !allowed.contains(&index) {
                continue;
            }
        }

        let name = parts[1].to_string();
        let vram_mib = parts[2].parse::<usize>().unwrap_or(0);

        // Parse compute capability (e.g., "8.9" → (8, 9)).
        let compute_cap = parts[3];
        let cc_parts: Vec<u32> = compute_cap
            .split('.')
            .filter_map(|s| s.parse().ok())
            .collect();
        let compute_capability = if cc_parts.len() >= 2 {
            (cc_parts[0], cc_parts[1])
        } else {
            (0, 0)
        };

        devices.push(GpuDeviceInfo {
            index,
            name,
            vram_mib,
            compute_capability,
            available: true,
        });
    }

    Ok(devices)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_devices(n: usize) -> Vec<GpuDeviceInfo> {
        (0..n)
            .map(|i| GpuDeviceInfo {
                index: i,
                name: format!("Test GPU {}", i),
                vram_mib: 8192,
                compute_capability: (8, 9),
                available: true,
            })
            .collect()
    }

    #[test]
    fn test_dispatcher_creation() {
        let devices = make_test_devices(4);
        let dispatcher = MultiGpuDispatcher::from_devices(devices).unwrap();
        assert_eq!(dispatcher.device_count(), 4);
    }

    #[test]
    fn test_empty_devices_rejected() {
        let result = MultiGpuDispatcher::from_devices(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_round_robin() {
        let devices = make_test_devices(3);
        let dispatcher = MultiGpuDispatcher::from_devices(devices).unwrap();

        // Round-robin should cycle: 0, 1, 2, 0, 1, 2, ...
        assert_eq!(dispatcher.next_device().index, 0);
        assert_eq!(dispatcher.next_device().index, 1);
        assert_eq!(dispatcher.next_device().index, 2);
        assert_eq!(dispatcher.next_device().index, 0);
        assert_eq!(dispatcher.next_device().index, 1);
    }

    #[test]
    fn test_single_device_round_robin() {
        let devices = make_test_devices(1);
        let dispatcher = MultiGpuDispatcher::from_devices(devices).unwrap();

        for _ in 0..10 {
            assert_eq!(dispatcher.next_device().index, 0);
        }
    }

    #[test]
    fn test_batch_dispatch() {
        let devices = make_test_devices(3);
        let dispatcher = MultiGpuDispatcher::from_devices(devices).unwrap();

        let items: Vec<u32> = (0..10).collect();
        let dispatched = dispatcher.dispatch_batch(&items);

        // 10 items across 3 GPUs: GPU0=4, GPU1=3, GPU2=3
        assert_eq!(dispatched.len(), 3);

        let total: usize = dispatched.iter().map(|(_, v)| v.len()).sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn test_batch_dispatch_fewer_items_than_gpus() {
        let devices = make_test_devices(4);
        let dispatcher = MultiGpuDispatcher::from_devices(devices).unwrap();

        let items: Vec<u32> = vec![1, 2];
        let dispatched = dispatcher.dispatch_batch(&items);

        // Only 2 items → 2 GPUs get work, 2 get nothing.
        assert_eq!(dispatched.len(), 2);
    }

    #[test]
    fn test_record_job_stats() {
        let devices = make_test_devices(2);
        let dispatcher = MultiGpuDispatcher::from_devices(devices).unwrap();

        dispatcher.record_job(0, 100);
        dispatcher.record_job(0, 200);
        dispatcher.record_job(1, 150);

        let stats = dispatcher.all_stats();
        assert_eq!(stats[0].1.total_jobs, 2);
        assert_eq!(stats[0].1.total_compute_ms, 300);
        assert_eq!(stats[1].1.total_jobs, 1);
        assert_eq!(stats[1].1.total_compute_ms, 150);
    }

    #[test]
    fn test_throughput_calculation() {
        let devices = make_test_devices(2);
        let dispatcher = MultiGpuDispatcher::from_devices(devices).unwrap();

        // GPU 0: 10 jobs in 1000ms → 10 jobs/s
        // GPU 1: 10 jobs in 500ms → 20 jobs/s
        // Total: 20 jobs / max(1000, 500)ms = 20 jobs/s
        for _ in 0..10 {
            dispatcher.record_job(0, 100);
            dispatcher.record_job(1, 50);
        }

        let throughput = dispatcher.total_throughput();
        assert!(throughput > 15.0, "expected >15 TPS, got {}", throughput);
    }

    #[test]
    fn test_device_access() {
        let devices = make_test_devices(3);
        let dispatcher = MultiGpuDispatcher::from_devices(devices).unwrap();

        assert!(dispatcher.device(0).is_some());
        assert!(dispatcher.device(2).is_some());
        assert!(dispatcher.device(3).is_none());
    }

    #[test]
    fn test_concurrent_round_robin() {
        use std::sync::Arc;
        use std::thread;

        let devices = make_test_devices(4);
        let dispatcher = Arc::new(MultiGpuDispatcher::from_devices(devices).unwrap());

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let disp = Arc::clone(&dispatcher);
                thread::spawn(move || {
                    let mut indices = Vec::new();
                    for _ in 0..100 {
                        indices.push(disp.next_device().index);
                    }
                    indices
                })
            })
            .collect();

        let mut all_indices = Vec::new();
        for h in handles {
            all_indices.extend(h.join().unwrap());
        }

        // 800 total dispatches should be roughly evenly distributed.
        assert_eq!(all_indices.len(), 800);

        let mut counts = [0u32; 4];
        for idx in &all_indices {
            counts[*idx] += 1;
        }
        // Each GPU should get ~200 (+/- some for concurrency).
        for (i, &count) in counts.iter().enumerate() {
            assert!(
                count >= 150 && count <= 250,
                "GPU {} got {} jobs (expected ~200)",
                i,
                count
            );
        }
    }

    #[test]
    fn test_detect_from_nvidia_smi() {
        // This test may fail on machines without nvidia-smi.
        match detect_from_nvidia_smi(None) {
            Ok(devices) => {
                // If we have GPUs, verify basic structure.
                for dev in &devices {
                    assert!(!dev.name.is_empty());
                    assert!(dev.vram_mib > 0);
                }
            }
            Err(_) => {
                // No GPU — that's fine for CI.
            }
        }
    }
}
