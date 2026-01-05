"""
Digital Twin example: Compressed state storage and querying.

This example demonstrates how QTT-SDK enables digital twins to maintain
high-fidelity state histories with minimal storage, enabling:
- Long-term state archival
- Fast similarity search
- Efficient state interpolation
"""

import torch
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json

from qtt_sdk import (
    QTTState,
    dense_to_qtt,
    qtt_to_dense,
    qtt_add,
    qtt_scale,
    qtt_norm,
    qtt_inner_product,
    truncate_qtt,
)


@dataclass
class TwinState:
    """A timestamped digital twin state."""
    timestamp: float
    qtt: QTTState
    metadata: Dict


class DigitalTwinStateManager:
    """
    Manages compressed state history for a digital twin.
    
    Stores simulation snapshots in QTT format, enabling:
    - 100-1000x compression vs. dense storage
    - O(log N) similarity queries
    - Efficient state interpolation
    """
    
    def __init__(self, max_bond: int = 64, grid_size: int = 2**16):
        self.max_bond = max_bond
        self.grid_size = grid_size
        self.num_qubits = int(math.log2(grid_size))
        self.states: Dict[float, TwinState] = {}
        self.timestamps: List[float] = []
    
    def store_state(
        self, 
        timestamp: float, 
        state: torch.Tensor,
        metadata: Optional[Dict] = None
    ):
        """
        Store a simulation state snapshot.
        
        Args:
            timestamp: Simulation time
            state: State vector (length must equal grid_size)
            metadata: Optional metadata (operating conditions, etc.)
        """
        if len(state) != self.grid_size:
            raise ValueError(f"State length {len(state)} != grid_size {self.grid_size}")
        
        qtt = dense_to_qtt(state, max_bond=self.max_bond)
        
        twin_state = TwinState(
            timestamp=timestamp,
            qtt=qtt,
            metadata=metadata or {}
        )
        
        self.states[timestamp] = twin_state
        self.timestamps.append(timestamp)
        self.timestamps.sort()
    
    def get_state(self, timestamp: float) -> Optional[torch.Tensor]:
        """Retrieve and decompress a state at exact timestamp."""
        if timestamp in self.states:
            return qtt_to_dense(self.states[timestamp].qtt)
        return None
    
    def get_compressed_state(self, timestamp: float) -> Optional[QTTState]:
        """Retrieve compressed state (no decompression)."""
        if timestamp in self.states:
            return self.states[timestamp].qtt
        return None
    
    def similarity(self, t1: float, t2: float) -> float:
        """
        Compute cosine similarity between two states.
        
        This is computed directly in QTT format without decompression.
        
        Returns:
            Cosine similarity in [-1, 1]
        """
        if t1 not in self.states or t2 not in self.states:
            raise KeyError(f"State not found at t={t1} or t={t2}")
        
        qtt1 = self.states[t1].qtt
        qtt2 = self.states[t2].qtt
        
        inner = qtt_inner_product(qtt1, qtt2)
        norm1 = qtt_norm(qtt1)
        norm2 = qtt_norm(qtt2)
        
        return inner / (norm1 * norm2)
    
    def find_similar_states(
        self, 
        reference_time: float, 
        threshold: float = 0.95,
        max_results: int = 10
    ) -> List[Tuple[float, float]]:
        """
        Find states similar to a reference state.
        
        Args:
            reference_time: Timestamp of reference state
            threshold: Minimum cosine similarity
            max_results: Maximum results to return
        
        Returns:
            List of (timestamp, similarity) tuples, sorted by similarity
        """
        if reference_time not in self.states:
            raise KeyError(f"Reference state not found at t={reference_time}")
        
        ref_qtt = self.states[reference_time].qtt
        ref_norm = qtt_norm(ref_qtt)
        
        results = []
        for t, state in self.states.items():
            if t == reference_time:
                continue
            
            inner = qtt_inner_product(ref_qtt, state.qtt)
            similarity = inner / (ref_norm * qtt_norm(state.qtt))
            
            if similarity >= threshold:
                results.append((t, similarity))
        
        results.sort(key=lambda x: -x[1])
        return results[:max_results]
    
    def interpolate_state(self, t: float) -> torch.Tensor:
        """
        Interpolate state at arbitrary time using QTT operations.
        
        Uses linear interpolation between nearest states.
        """
        if t in self.states:
            return qtt_to_dense(self.states[t].qtt)
        
        # Find bracketing timestamps
        t_before = None
        t_after = None
        
        for ts in self.timestamps:
            if ts <= t:
                t_before = ts
            if ts > t and t_after is None:
                t_after = ts
                break
        
        if t_before is None or t_after is None:
            raise ValueError(f"Cannot interpolate: t={t} outside range [{self.timestamps[0]}, {self.timestamps[-1]}]")
        
        # Linear interpolation weight
        alpha = (t - t_before) / (t_after - t_before)
        
        # Interpolate in QTT format
        qtt_before = self.states[t_before].qtt
        qtt_after = self.states[t_after].qtt
        
        # result = (1-alpha) * before + alpha * after
        scaled_before = qtt_scale(qtt_before, 1.0 - alpha)
        scaled_after = qtt_scale(qtt_after, alpha)
        interpolated = qtt_add(scaled_before, scaled_after, max_bond=self.max_bond)
        
        return qtt_to_dense(interpolated)
    
    def memory_usage(self) -> int:
        """Total memory used for all states (bytes)."""
        return sum(state.qtt.memory_bytes for state in self.states.values())
    
    def compression_ratio(self) -> float:
        """Compression ratio vs. dense storage."""
        dense_size = len(self.states) * self.grid_size * 8
        return dense_size / self.memory_usage() if self.states else 1.0
    
    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"DigitalTwinStateManager:\n"
            f"  States: {len(self.states)}\n"
            f"  Grid size: {self.grid_size:,}\n"
            f"  Time range: [{min(self.timestamps):.1f}, {max(self.timestamps):.1f}]\n"
            f"  Memory: {self.memory_usage() / 1e6:.2f} MB\n"
            f"  Compression: {self.compression_ratio():.0f}x"
        )


def thermal_digital_twin_demo():
    """
    Demonstrate a thermal simulation digital twin.
    
    Simulates a heat transfer system and stores state history
    for later analysis and similarity search.
    """
    print("=" * 60)
    print("Thermal Digital Twin with QTT State Compression")
    print("=" * 60)
    
    # Configuration
    grid_size = 2**16  # 65,536 spatial points
    num_snapshots = 100
    dt = 0.1  # Time step
    
    print(f"\nConfiguration:")
    print(f"  Spatial grid: {grid_size:,} points")
    print(f"  Time steps: {num_snapshots}")
    print(f"  Dense storage would need: {num_snapshots * grid_size * 8 / 1e6:.1f} MB")
    
    # Create digital twin manager
    twin = DigitalTwinStateManager(max_bond=64, grid_size=grid_size)
    
    # Simulate thermal evolution
    print("\nSimulating thermal evolution...")
    start = time.perf_counter()
    
    x = torch.linspace(0, 1, grid_size, dtype=torch.float64)
    temperature = 20 + 10 * torch.sin(2 * math.pi * x)  # Initial condition
    
    for step in range(num_snapshots):
        t = step * dt
        
        # Simple heat equation evolution (for demo purposes)
        # In practice, this would be a real simulation
        decay = math.exp(-0.1 * t)
        noise = 0.01 * torch.randn(grid_size, dtype=torch.float64)
        temperature = 20 + 10 * decay * torch.sin(2 * math.pi * x) + noise
        
        # Add time-varying boundary conditions
        if step > 50:
            temperature += 5 * torch.exp(-((x - 0.5) ** 2) / 0.01)  # Heat source
        
        # Store state with metadata
        metadata = {
            "step": step,
            "boundary_temp": 20.0,
            "heat_source_active": step > 50
        }
        twin.store_state(t, temperature, metadata)
    
    sim_time = time.perf_counter() - start
    print(f"Simulation complete in {sim_time:.2f}s")
    print(f"\n{twin.summary()}")
    
    # Demonstrate similarity search
    print("\n" + "-" * 40)
    print("Similarity Analysis")
    print("-" * 40)
    
    reference_time = 8.0  # Near end, with heat source active
    print(f"\nFinding states similar to t={reference_time}...")
    
    similar = twin.find_similar_states(reference_time, threshold=0.99, max_results=5)
    print(f"Most similar states (cosine similarity > 0.99):")
    for t, sim in similar:
        heat_source = twin.states[t].metadata.get("heat_source_active", False)
        print(f"  t={t:.1f}: similarity={sim:.4f}, heat_source={heat_source}")
    
    # Demonstrate state interpolation
    print("\n" + "-" * 40)
    print("State Interpolation")
    print("-" * 40)
    
    # Interpolate at a time between snapshots
    t_interp = 5.25
    start = time.perf_counter()
    interpolated = twin.interpolate_state(t_interp)
    interp_time = time.perf_counter() - start
    
    print(f"Interpolated state at t={t_interp} in {interp_time*1000:.1f}ms")
    print(f"  Mean temperature: {interpolated.mean():.2f}")
    print(f"  Max temperature: {interpolated.max():.2f}")
    
    # Demonstrate change detection
    print("\n" + "-" * 40)
    print("Change Detection")
    print("-" * 40)
    
    print("Similarity between adjacent time steps:")
    for i in range(0, 80, 10):
        t1 = i * dt
        t2 = (i + 10) * dt
        sim = twin.similarity(t1, t2)
        change = 1.0 - sim
        print(f"  t={t1:.1f} -> t={t2:.1f}: similarity={sim:.4f}, change={change:.4f}")
    
    print(f"\nNote: Large change at t=5.0 indicates heat source activation")


def fleet_digital_twin_demo():
    """
    Demonstrate fleet-level digital twin management.
    
    Manages compressed states for multiple assets (vehicles, machines)
    and enables cross-asset similarity analysis.
    """
    print("\n" + "=" * 60)
    print("Fleet Digital Twin Management")
    print("=" * 60)
    
    num_assets = 10
    grid_size = 2**14  # 16,384 points per asset
    snapshots_per_asset = 50
    
    print(f"\nConfiguration:")
    print(f"  Assets: {num_assets}")
    print(f"  State size per asset: {grid_size:,} points")
    print(f"  Snapshots per asset: {snapshots_per_asset}")
    print(f"  Total dense storage: {num_assets * snapshots_per_asset * grid_size * 8 / 1e6:.1f} MB")
    
    # Create digital twin for each asset
    fleet: Dict[str, DigitalTwinStateManager] = {}
    
    start = time.perf_counter()
    for asset_id in range(num_assets):
        asset_name = f"vehicle_{asset_id:02d}"
        twin = DigitalTwinStateManager(max_bond=32, grid_size=grid_size)
        
        # Simulate different operating conditions per asset
        x = torch.linspace(0, 1, grid_size, dtype=torch.float64)
        base_pattern = torch.sin(2 * math.pi * x * (asset_id + 1))  # Asset-specific
        
        for step in range(snapshots_per_asset):
            t = step * 0.1
            noise = 0.05 * torch.randn(grid_size, dtype=torch.float64)
            state = base_pattern * math.exp(-0.02 * t) + noise
            twin.store_state(t, state, {"asset_id": asset_id})
        
        fleet[asset_name] = twin
    
    setup_time = time.perf_counter() - start
    
    total_memory = sum(twin.memory_usage() for twin in fleet.values())
    print(f"\nFleet initialized in {setup_time:.2f}s")
    print(f"Total memory: {total_memory / 1e6:.2f} MB")
    print(f"Compression ratio: {num_assets * snapshots_per_asset * grid_size * 8 / total_memory:.0f}x")
    
    # Cross-asset similarity analysis
    print("\n" + "-" * 40)
    print("Cross-Asset Similarity Analysis (at t=2.0)")
    print("-" * 40)
    
    reference_asset = "vehicle_00"
    reference_state = fleet[reference_asset].get_compressed_state(2.0)
    ref_norm = qtt_norm(reference_state)
    
    similarities = []
    for asset_name, twin in fleet.items():
        if asset_name == reference_asset:
            continue
        other_state = twin.get_compressed_state(2.0)
        sim = qtt_inner_product(reference_state, other_state) / (ref_norm * qtt_norm(other_state))
        similarities.append((asset_name, sim))
    
    similarities.sort(key=lambda x: -x[1])
    
    print(f"Assets most similar to {reference_asset}:")
    for name, sim in similarities[:5]:
        print(f"  {name}: {sim:.4f}")


if __name__ == "__main__":
    thermal_digital_twin_demo()
    fleet_digital_twin_demo()
    
    print("\n" + "=" * 60)
    print("Demo complete. QTT-SDK enables digital twins to maintain")
    print("100-1000x more state history with the same memory budget.")
    print("=" * 60)
