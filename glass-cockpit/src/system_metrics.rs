/*!
 * Phase 7: System Metrics Collector
 * 
 * Cross-platform CPU, memory, and GPU metrics collection.
 * Designed for minimal overhead (<0.1ms per sample).
 */
#![allow(dead_code)] // Metrics API ready for integration

use std::time::{Duration, Instant};

#[cfg(target_os = "windows")]
use std::mem::MaybeUninit;

/// CPU utilization sample
#[derive(Clone, Copy, Default)]
pub struct CpuSample {
    /// Total CPU usage 0.0-1.0
    pub total: f32,
    /// Per-core usage (if available)
    pub per_core: [f32; 32],
    /// Number of cores
    pub core_count: usize,
}

/// Memory usage sample
#[derive(Clone, Copy, Default)]
pub struct MemorySample {
    /// Used physical memory in bytes
    pub used: u64,
    /// Total physical memory in bytes
    pub total: u64,
    /// Available memory in bytes
    pub available: u64,
    /// Used percentage 0.0-1.0
    pub usage: f32,
}

/// GPU metrics (if available)
#[derive(Clone, Copy, Default)]
pub struct GpuSample {
    /// GPU utilization 0.0-1.0
    pub utilization: f32,
    /// VRAM used in bytes
    pub memory_used: u64,
    /// VRAM total in bytes
    pub memory_total: u64,
    /// Temperature in Celsius (if available)
    pub temperature: f32,
    /// Power draw in Watts (if available)
    pub power_draw: f32,
    /// Whether GPU metrics are available
    pub available: bool,
}

/// System metrics collector
pub struct MetricsCollector {
    last_sample_time: Instant,
    sample_interval: Duration,
    
    // Cached samples
    cpu: CpuSample,
    memory: MemorySample,
    gpu: GpuSample,
    
    // For CPU delta calculation
    #[cfg(target_os = "windows")]
    last_idle: u64,
    #[cfg(target_os = "windows")]
    last_kernel: u64,
    #[cfg(target_os = "windows")]
    last_user: u64,
    
    #[cfg(target_os = "linux")]
    last_cpu_stats: (u64, u64), // (idle, total)
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            last_sample_time: Instant::now(),
            sample_interval: Duration::from_millis(100), // 10Hz sampling
            cpu: CpuSample::default(),
            memory: MemorySample::default(),
            gpu: GpuSample::default(),
            
            #[cfg(target_os = "windows")]
            last_idle: 0,
            #[cfg(target_os = "windows")]
            last_kernel: 0,
            #[cfg(target_os = "windows")]
            last_user: 0,
            
            #[cfg(target_os = "linux")]
            last_cpu_stats: (0, 0),
        }
    }
    
    /// Sample all metrics (rate-limited)
    pub fn sample(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.last_sample_time) < self.sample_interval {
            return;
        }
        self.last_sample_time = now;
        
        self.sample_cpu();
        self.sample_memory();
        self.sample_gpu();
    }
    
    /// Force immediate sample
    pub fn sample_now(&mut self) {
        self.sample_cpu();
        self.sample_memory();
        self.sample_gpu();
    }
    
    pub fn cpu(&self) -> &CpuSample {
        &self.cpu
    }
    
    pub fn memory(&self) -> &MemorySample {
        &self.memory
    }
    
    pub fn gpu(&self) -> &GpuSample {
        &self.gpu
    }
    
    // ========================================================================
    // Platform-specific implementations
    // ========================================================================
    
    #[cfg(target_os = "windows")]
    fn sample_cpu(&mut self) {
        use std::ptr;
        
        #[repr(C)]
        struct FILETIME {
            dwLowDateTime: u32,
            dwHighDateTime: u32,
        }
        
        impl FILETIME {
            fn to_u64(&self) -> u64 {
                ((self.dwHighDateTime as u64) << 32) | (self.dwLowDateTime as u64)
            }
        }
        
        #[link(name = "kernel32")]
        extern "system" {
            fn GetSystemTimes(
                lpIdleTime: *mut FILETIME,
                lpKernelTime: *mut FILETIME,
                lpUserTime: *mut FILETIME,
            ) -> i32;
        }
        
        unsafe {
            let mut idle: MaybeUninit<FILETIME> = MaybeUninit::uninit();
            let mut kernel: MaybeUninit<FILETIME> = MaybeUninit::uninit();
            let mut user: MaybeUninit<FILETIME> = MaybeUninit::uninit();
            
            if GetSystemTimes(
                idle.as_mut_ptr(),
                kernel.as_mut_ptr(),
                user.as_mut_ptr(),
            ) != 0 {
                let idle = idle.assume_init().to_u64();
                let kernel = kernel.assume_init().to_u64();
                let user = user.assume_init().to_u64();
                
                let idle_delta = idle.saturating_sub(self.last_idle);
                let kernel_delta = kernel.saturating_sub(self.last_kernel);
                let user_delta = user.saturating_sub(self.last_user);
                
                let total = kernel_delta + user_delta;
                let active = total - idle_delta;
                
                if total > 0 {
                    self.cpu.total = (active as f32 / total as f32).clamp(0.0, 1.0);
                }
                
                self.last_idle = idle;
                self.last_kernel = kernel;
                self.last_user = user;
            }
        }
        
        // Get core count
        self.cpu.core_count = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
    }
    
    #[cfg(target_os = "linux")]
    fn sample_cpu(&mut self) {
        if let Ok(content) = std::fs::read_to_string("/proc/stat") {
            if let Some(line) = content.lines().next() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 5 && parts[0] == "cpu" {
                    let user: u64 = parts[1].parse().unwrap_or(0);
                    let nice: u64 = parts[2].parse().unwrap_or(0);
                    let system: u64 = parts[3].parse().unwrap_or(0);
                    let idle: u64 = parts[4].parse().unwrap_or(0);
                    
                    let total = user + nice + system + idle;
                    let (last_idle, last_total) = self.last_cpu_stats;
                    
                    let idle_delta = idle.saturating_sub(last_idle);
                    let total_delta = total.saturating_sub(last_total);
                    
                    if total_delta > 0 {
                        self.cpu.total = 1.0 - (idle_delta as f32 / total_delta as f32);
                        self.cpu.total = self.cpu.total.clamp(0.0, 1.0);
                    }
                    
                    self.last_cpu_stats = (idle, total);
                }
            }
        }
        
        self.cpu.core_count = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
    }
    
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    fn sample_cpu(&mut self) {
        self.cpu.core_count = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
    }
    
    #[cfg(target_os = "windows")]
    fn sample_memory(&mut self) {
        #[repr(C)]
        struct MEMORYSTATUSEX {
            dwLength: u32,
            dwMemoryLoad: u32,
            ullTotalPhys: u64,
            ullAvailPhys: u64,
            ullTotalPageFile: u64,
            ullAvailPageFile: u64,
            ullTotalVirtual: u64,
            ullAvailVirtual: u64,
            ullAvailExtendedVirtual: u64,
        }
        
        #[link(name = "kernel32")]
        extern "system" {
            fn GlobalMemoryStatusEx(lpBuffer: *mut MEMORYSTATUSEX) -> i32;
        }
        
        unsafe {
            let mut status: MaybeUninit<MEMORYSTATUSEX> = MaybeUninit::uninit();
            let ptr = status.as_mut_ptr();
            (*ptr).dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
            
            if GlobalMemoryStatusEx(ptr) != 0 {
                let status = status.assume_init();
                self.memory.total = status.ullTotalPhys;
                self.memory.available = status.ullAvailPhys;
                self.memory.used = status.ullTotalPhys - status.ullAvailPhys;
                self.memory.usage = status.dwMemoryLoad as f32 / 100.0;
            }
        }
    }
    
    #[cfg(target_os = "linux")]
    fn sample_memory(&mut self) {
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            let mut mem_total: u64 = 0;
            let mut mem_available: u64 = 0;
            
            for line in content.lines() {
                if line.starts_with("MemTotal:") {
                    mem_total = parse_meminfo_value(line);
                } else if line.starts_with("MemAvailable:") {
                    mem_available = parse_meminfo_value(line);
                }
            }
            
            self.memory.total = mem_total * 1024; // Convert KB to bytes
            self.memory.available = mem_available * 1024;
            self.memory.used = self.memory.total.saturating_sub(self.memory.available);
            
            if self.memory.total > 0 {
                self.memory.usage = self.memory.used as f32 / self.memory.total as f32;
            }
        }
    }
    
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    fn sample_memory(&mut self) {
        // Fallback: no memory info
    }
    
    fn sample_gpu(&mut self) {
        // GPU metrics require NVML or vendor-specific APIs
        // For now, mark as unavailable
        self.gpu.available = false;
        
        // Could integrate with:
        // - NVML (NVIDIA)
        // - AMD GPU Profiler
        // - Intel GPU Tools
    }
}

#[cfg(target_os = "linux")]
fn parse_meminfo_value(line: &str) -> u64 {
    line.split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

/// Format bytes as human-readable string
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    
    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metrics_collector() {
        let mut collector = MetricsCollector::new();
        collector.sample_now();
        
        // Should have some reasonable values
        assert!(collector.cpu().core_count > 0);
        
        // Memory should be non-zero on any system
        #[cfg(any(target_os = "windows", target_os = "linux"))]
        assert!(collector.memory().total > 0);
    }
    
    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1536), "1.5 KB");
        assert_eq!(format_bytes(1048576), "1.0 MB");
        assert_eq!(format_bytes(1073741824), "1.0 GB");
    }
}
