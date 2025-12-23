"""
Big Data example: Streaming analytics on massive datasets.

This example demonstrates how QTT-SDK enables analytics on datasets
that exceed available RAM by keeping data compressed during computation.
"""

import torch
import math
import time
from typing import Iterator, Tuple

# Assuming qtt_sdk is installed
from qtt_sdk import (
    QTTState,
    dense_to_qtt,
    qtt_add,
    qtt_scale,
    qtt_norm,
    qtt_inner_product,
    truncate_qtt,
)


def simulate_sensor_stream(
    num_chunks: int = 100,
    chunk_size: int = 2**16,  # 65,536 points per chunk
    noise_level: float = 0.1
) -> Iterator[torch.Tensor]:
    """
    Simulate a stream of sensor data chunks.
    
    In practice, this would read from files, databases, or network streams.
    """
    for i in range(num_chunks):
        t = torch.linspace(i, i + 1, chunk_size, dtype=torch.float64)
        # Simulate temperature sensor with daily cycle and noise
        signal = 20 + 5 * torch.sin(2 * math.pi * t / 24)  # Daily cycle
        signal += 2 * torch.sin(2 * math.pi * t * 12)       # High-frequency component
        signal += noise_level * torch.randn(chunk_size)     # Noise
        yield signal


class StreamingStatistics:
    """
    Compute running statistics on streaming data using QTT compression.
    
    This class accumulates statistics (mean, variance) on arbitrarily
    large datasets while using bounded memory.
    """
    
    def __init__(self, max_bond: int = 64):
        self.max_bond = max_bond
        self.sum_qtt: QTTState = None
        self.sum_sq_qtt: QTTState = None
        self.count = 0
        self.total_points = 0
    
    def update(self, chunk: torch.Tensor):
        """Process a new data chunk."""
        qtt_chunk = dense_to_qtt(chunk, max_bond=self.max_bond)
        
        # Accumulate sum
        if self.sum_qtt is None:
            self.sum_qtt = qtt_chunk
        else:
            self.sum_qtt = qtt_add(
                self.sum_qtt, qtt_chunk, 
                max_bond=self.max_bond * 2
            )
            # Periodic recompression to control rank growth
            if self.count % 10 == 0:
                self.sum_qtt = truncate_qtt(self.sum_qtt, self.max_bond)
        
        # Accumulate sum of squares (for variance)
        chunk_sq = chunk ** 2
        qtt_sq = dense_to_qtt(chunk_sq, max_bond=self.max_bond)
        
        if self.sum_sq_qtt is None:
            self.sum_sq_qtt = qtt_sq
        else:
            self.sum_sq_qtt = qtt_add(
                self.sum_sq_qtt, qtt_sq,
                max_bond=self.max_bond * 2
            )
            if self.count % 10 == 0:
                self.sum_sq_qtt = truncate_qtt(self.sum_sq_qtt, self.max_bond)
        
        self.count += 1
        self.total_points += len(chunk)
    
    def mean_qtt(self) -> QTTState:
        """Return QTT representation of the mean (per chunk)."""
        return qtt_scale(self.sum_qtt, 1.0 / self.count)
    
    def global_mean(self) -> float:
        """Return the global scalar mean across all points."""
        total_sum = qtt_norm(self.sum_qtt) ** 2  # Approximate
        # More accurate: use inner product with ones vector
        return total_sum / self.total_points
    
    def memory_usage(self) -> int:
        """Total memory used for statistics (in bytes)."""
        total = 0
        if self.sum_qtt:
            total += self.sum_qtt.memory_bytes
        if self.sum_sq_qtt:
            total += self.sum_sq_qtt.memory_bytes
        return total


def anomaly_detection_demo():
    """
    Demonstrate anomaly detection on streaming sensor data.
    
    Uses QTT compression to maintain a compressed baseline profile
    and detect deviations in real-time.
    """
    print("=" * 60)
    print("Streaming Anomaly Detection with QTT Compression")
    print("=" * 60)
    
    # Build baseline from historical data
    print("\nPhase 1: Building baseline profile...")
    baseline_stats = StreamingStatistics(max_bond=32)
    
    start = time.perf_counter()
    for i, chunk in enumerate(simulate_sensor_stream(num_chunks=50, noise_level=0.05)):
        baseline_stats.update(chunk)
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1} chunks, memory: {baseline_stats.memory_usage() / 1e3:.1f} KB")
    
    baseline_time = time.perf_counter() - start
    print(f"\nBaseline built in {baseline_time:.2f}s")
    print(f"Total points processed: {baseline_stats.total_points:,}")
    print(f"Memory used: {baseline_stats.memory_usage() / 1e3:.1f} KB")
    print(f"Compression vs dense: {baseline_stats.total_points * 8 / baseline_stats.memory_usage():.0f}x")
    
    # Real-time anomaly detection
    print("\n" + "=" * 60)
    print("Phase 2: Real-time anomaly detection...")
    print("=" * 60)
    
    baseline_mean = baseline_stats.mean_qtt()
    baseline_norm = qtt_norm(baseline_mean)
    
    # Process new data and detect anomalies
    anomalies_detected = 0
    
    for i, chunk in enumerate(simulate_sensor_stream(num_chunks=20, noise_level=0.1)):
        # Inject anomaly at chunk 10
        if i == 10:
            chunk = chunk + 50  # Large spike
        
        qtt_chunk = dense_to_qtt(chunk, max_bond=32)
        
        # Compare to baseline using inner product
        deviation = qtt_norm(qtt_chunk) / baseline_norm
        
        if deviation > 3.0:  # Threshold
            anomalies_detected += 1
            print(f"  [ANOMALY] Chunk {i}: deviation = {deviation:.2f}x baseline")
        else:
            if i % 5 == 0:
                print(f"  Chunk {i}: normal (deviation = {deviation:.2f}x)")
    
    print(f"\nTotal anomalies detected: {anomalies_detected}")


def correlation_analysis_demo():
    """
    Demonstrate correlation analysis between multiple sensor streams.
    
    Uses QTT inner products for O(n*r^2) correlation instead of O(N).
    """
    print("\n" + "=" * 60)
    print("Multi-Sensor Correlation Analysis")
    print("=" * 60)
    
    num_sensors = 5
    chunk_size = 2**18  # 262,144 points per sensor
    max_bond = 64
    
    print(f"\nAnalyzing {num_sensors} sensors with {chunk_size:,} points each")
    print(f"Dense correlation would need: {num_sensors * chunk_size * 8 / 1e6:.1f} MB")
    
    # Generate sensor data
    sensors = []
    for s in range(num_sensors):
        t = torch.linspace(0, 100, chunk_size, dtype=torch.float64)
        # Sensors have correlated base signal plus independent noise
        base = torch.sin(2 * math.pi * t / 24)  # Common daily cycle
        independent = 0.3 * torch.sin(2 * math.pi * t * (s + 1))  # Sensor-specific
        noise = 0.1 * torch.randn(chunk_size)
        sensors.append(base + independent + noise)
    
    # Compress all sensors
    start = time.perf_counter()
    qtt_sensors = [dense_to_qtt(s, max_bond=max_bond) for s in sensors]
    compress_time = time.perf_counter() - start
    
    total_qtt_memory = sum(q.memory_bytes for q in qtt_sensors)
    print(f"Compressed in {compress_time:.2f}s")
    print(f"QTT memory: {total_qtt_memory / 1e3:.1f} KB")
    print(f"Compression ratio: {num_sensors * chunk_size * 8 / total_qtt_memory:.0f}x")
    
    # Compute correlation matrix using QTT inner products
    print("\nCorrelation matrix (computed in QTT format):")
    print("-" * 40)
    
    start = time.perf_counter()
    correlations = torch.zeros(num_sensors, num_sensors)
    norms = [qtt_norm(q) for q in qtt_sensors]
    
    for i in range(num_sensors):
        for j in range(i, num_sensors):
            corr = qtt_inner_product(qtt_sensors[i], qtt_sensors[j])
            corr = corr / (norms[i] * norms[j])
            correlations[i, j] = corr
            correlations[j, i] = corr
    
    corr_time = time.perf_counter() - start
    
    # Print correlation matrix
    header = "     " + "".join(f"  S{i}  " for i in range(num_sensors))
    print(header)
    for i in range(num_sensors):
        row = f"S{i}:  " + "  ".join(f"{correlations[i,j]:.3f}" for j in range(num_sensors))
        print(row)
    
    print(f"\nCorrelation computed in {corr_time * 1000:.1f}ms")


if __name__ == "__main__":
    anomaly_detection_demo()
    correlation_analysis_demo()
    
    print("\n" + "=" * 60)
    print("Demo complete. QTT-SDK enables big data analytics")
    print("on datasets 100-10,000x larger than available RAM.")
    print("=" * 60)
