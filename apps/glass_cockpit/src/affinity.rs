/*!
 * E-Core Affinity Enforcement (Windows-specific)
 * 
 * Doctrine 1: Computational Sovereignty
 * 
 * Pins the UI process to E-cores (logical processors 16-31 on i9-14900HX)
 * to prevent interference with P-core physics computation.
 */

use anyhow::Result;

// Phase 1-2 scaffolding: E-core affinity enforcement for Windows UI process
#[cfg(target_os = "windows")]
#[allow(dead_code)]
pub fn enforce_e_core_affinity() -> Result<()> {
    use windows::Win32::System::Threading::{
        GetCurrentProcess, SetProcessAffinityMask,
    };
    
    unsafe {
        let process = GetCurrentProcess();
        
        // Affinity mask for E-cores (logical processors 16-31)
        // Binary: 0xFFFF0000 (bits 16-31 set)
        const E_CORE_MASK: usize = 0xFFFF_0000;
        
        SetProcessAffinityMask(process, E_CORE_MASK)
            .context("Failed to set process affinity mask")?;
    }
    
    Ok(())
}

// Phase 1-2 scaffolding: E-core affinity for Linux (uses sched_setaffinity)
#[cfg(target_os = "linux")]
#[allow(dead_code)]
pub fn enforce_e_core_affinity() -> Result<()> {
    use std::fs;
    use nix::libc;
    
    // Detect CPU topology to find E-cores
    // On Intel hybrid CPUs, E-cores are typically higher-numbered
    // For i9-14900HX: 8 P-cores (0-7 + HT 8-15) + 16 E-cores (16-31)
    
    // Read number of CPUs
    let cpu_count = match fs::read_to_string("/sys/devices/system/cpu/online") {
        Ok(content) => {
            // Parse "0-31" format
            if let Some(max) = content.trim().split('-').next_back() {
                max.parse::<usize>().unwrap_or(31) + 1
            } else {
                32
            }
        }
        Err(_) => 32, // Default to 32 cores
    };
    
    // If we have 32+ cores, assume hybrid architecture with E-cores at 16-31
    if cpu_count >= 32 {
        // Use libc sched_setaffinity
        let mut cpu_set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        
        // Set affinity to cores 16-31 (E-cores on hybrid Intel)
        for cpu in 16..32 {
            unsafe { libc::CPU_SET(cpu, &mut cpu_set) };
        }
        
        let result = unsafe {
            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &cpu_set)
        };
        
        if result == 0 {
            println!("  ✓ Linux: Pinned to E-cores (16-31)");
            Ok(())
        } else {
            // Non-fatal - may not have permission
            println!("  ⚠ Linux: Could not set E-core affinity (permission denied or not hybrid CPU)");
            Ok(())
        }
    } else {
        // Non-hybrid CPU - run on any core
        println!("  ℹ Linux: Non-hybrid CPU detected ({} cores), skipping E-core pinning", cpu_count);
        Ok(())
    }
}

// Fallback for other platforms (macOS, etc.)
#[cfg(not(any(target_os = "windows", target_os = "linux")))]
#[allow(dead_code)]
pub fn enforce_e_core_affinity() -> Result<()> {
    // E-core affinity only enforced on Windows/Linux
    println!("  ℹ E-core affinity not available on this platform");
    Ok(())
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    
    #[test]
    #[cfg(target_os = "windows")]
    fn test_affinity_enforcement() {
        // This test validates affinity can be set
        // Actual core verification requires Task Manager inspection
        assert!(enforce_e_core_affinity().is_ok());
    }
}
