"""
GPU Memory Management
=====================

OPERATION VALHALLA - Phase 2.5: Memory Management Strategy

VRAM pool management for RTX 5070 (8GB).
Implements allocation strategy with defragmentation.

Author: OPERATION VALHALLA
Date: 2025-12-28
"""

import torch
from typing import Dict, Optional
from dataclasses import dataclass
import gc


@dataclass
class MemoryStats:
    """GPU memory statistics."""
    allocated_gb: float
    reserved_gb: float
    free_gb: float
    total_gb: float
    utilization: float


class VRAMManager:
    """
    Memory pool manager for GPU tensors.
    
    Features:
        - Allocation tracking
        - Defragmentation on demand
        - Memory pressure detection
        - Emergency cleanup protocols
    """
    
    def __init__(self, device: str = 'cuda:0', max_gb: float = 7.0):
        """
        Initialize VRAM manager.
        
        Args:
            device: CUDA device
            max_gb: Maximum VRAM budget (leave headroom)
        """
        self.device = device
        self.max_bytes = int(max_gb * 1024**3)
        
        # Get total VRAM
        props = torch.cuda.get_device_properties(0)
        self.total_gb = props.total_memory / 1024**3
        
        print(f"✓ VRAMManager: {self.total_gb:.2f} GB total, {max_gb:.2f} GB budget")
        
    def get_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory
        free = total - reserved
        
        return MemoryStats(
            allocated_gb=allocated / 1024**3,
            reserved_gb=reserved / 1024**3,
            free_gb=free / 1024**3,
            total_gb=total / 1024**3,
            utilization=allocated / self.max_bytes
        )
    
    def check_available(self, required_gb: float) -> bool:
        """Check if required memory is available."""
        stats = self.get_stats()
        return stats.free_gb >= required_gb
    
    def defragment(self):
        """
        Defragment GPU memory.
        
        Forces PyTorch to release unused cached memory.
        """
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    def emergency_cleanup(self):
        """Emergency memory cleanup when OOM imminent."""
        print("⚠ VRAM PRESSURE - Emergency cleanup")
        self.defragment()
        gc.collect()
        
    def allocate_field(self, shape: tuple, dtype=torch.float32) -> Optional[torch.Tensor]:
        """
        Safe tensor allocation with OOM protection.
        
        Args:
            shape: Tensor shape
            dtype: Data type
            
        Returns:
            Allocated tensor or None if OOM
        """
        required_bytes = torch.finfo(dtype).bits // 8
        for dim in shape:
            required_bytes *= dim
        required_gb = required_bytes / 1024**3
        
        # Check budget
        stats = self.get_stats()
        if stats.allocated_gb + required_gb > self.max_bytes / 1024**3:
            print(f"⚠ Allocation denied: {required_gb:.3f} GB exceeds budget")
            return None
        
        try:
            tensor = torch.empty(shape, dtype=dtype, device=self.device)
            return tensor
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"⚠ OOM during allocation: {required_gb:.3f} GB")
                self.emergency_cleanup()
                return None
            raise
    
    def print_summary(self):
        """Print memory usage summary."""
        stats = self.get_stats()
        print(f"\n{'='*60}")
        print("VRAM SUMMARY")
        print(f"{'='*60}")
        print(f"Allocated:   {stats.allocated_gb:.3f} GB")
        print(f"Reserved:    {stats.reserved_gb:.3f} GB")
        print(f"Free:        {stats.free_gb:.3f} GB")
        print(f"Total:       {stats.total_gb:.3f} GB")
        print(f"Utilization: {stats.utilization*100:.1f}%")
        print(f"{'='*60}\n")


# Global memory manager instance
_vram_manager: Optional[VRAMManager] = None


def get_vram_manager() -> VRAMManager:
    """Get global VRAM manager instance."""
    global _vram_manager
    if _vram_manager is None:
        _vram_manager = VRAMManager()
    return _vram_manager
