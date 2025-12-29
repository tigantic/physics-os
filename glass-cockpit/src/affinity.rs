/*!
 * E-Core Affinity Enforcement (Windows-specific)
 * 
 * Doctrine 1: Computational Sovereignty
 * 
 * Pins the UI process to E-cores (logical processors 16-31 on i9-14900HX)
 * to prevent interference with P-core physics computation.
 */

use anyhow::Result;

#[cfg(target_os = "windows")]
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

#[cfg(not(target_os = "windows"))]
pub fn enforce_e_core_affinity() -> Result<()> {
    // E-core affinity only enforced on Windows
    // Linux/macOS can use taskset externally if needed
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    #[cfg(target_os = "windows")]
    fn test_affinity_enforcement() {
        // This test validates affinity can be set
        // Actual core verification requires Task Manager inspection
        assert!(enforce_e_core_affinity().is_ok());
    }
}
