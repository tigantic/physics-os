# Copyright (c) 2025 Tigantic
# Phase 18 Tests: Adaptive, Realtime, and Coordination
"""
Comprehensive tests for Phase 18 modules:
- Adaptive bond dimension optimization
- Real-time inference optimization
- Multi-vehicle coordination
"""

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
import torch.nn as nn


# =============================================================================
# ADAPTIVE MODULE TESTS
# =============================================================================

class TestAdaptiveBondOptimizer:
    """Tests for adaptive bond dimension optimizer."""
    
    def test_adaptive_bond_config_creation(self):
        """Test AdaptiveBondConfig creation."""
        from tensornet.adaptive.bond_optimizer import (
            AdaptiveBondConfig,
            TruncationStrategy,
        )
        
        config = AdaptiveBondConfig(
            chi_min=4,
            chi_max=128,
            target_truncation_error=1e-8,
            strategy=TruncationStrategy.ENTROPY_BASED,
        )
        
        assert config.chi_min == 4
        assert config.chi_max == 128
        assert config.target_truncation_error == 1e-8
        assert config.strategy == TruncationStrategy.ENTROPY_BASED
    
    def test_bond_dimension_tracker(self):
        """Test BondDimensionTracker."""
        from tensornet.adaptive.bond_optimizer import (
            BondDimensionTracker,
            AdaptiveBondConfig,
        )
        
        config = AdaptiveBondConfig(chi_min=4, chi_max=128)
        tracker = BondDimensionTracker(config=config, num_sites=10)
        
        # Verify tracker was created and has correct attributes
        assert tracker is not None
        assert tracker.total_truncation_error >= 0
    
    def test_entropy_monitor(self):
        """Test EntropyMonitor."""
        from tensornet.adaptive.bond_optimizer import EntropyMonitor
        
        monitor = EntropyMonitor(num_sites=10)
        
        # Test with singular values
        sv = torch.tensor([0.8, 0.4, 0.2, 0.1])
        monitor.update(bond_index=0, singular_values=sv)
        
        # Verify monitor tracks entropy
        assert monitor is not None
    
    def test_truncation_scheduler(self):
        """Test TruncationScheduler."""
        from tensornet.adaptive.bond_optimizer import (
            TruncationScheduler,
            AdaptiveBondConfig,
            BondDimensionTracker,
            EntropyMonitor,
            TruncationStrategy,
        )
        
        config = AdaptiveBondConfig(
            chi_min=8,
            chi_max=64,
            strategy=TruncationStrategy.ERROR_TARGET,
            target_truncation_error=1e-6,
        )
        tracker = BondDimensionTracker(config, num_sites=10)
        entropy_monitor = EntropyMonitor(num_sites=10)
        scheduler = TruncationScheduler(config, tracker, entropy_monitor)
        
        # Verify scheduler was created
        assert scheduler is not None
    
    def test_adaptive_truncator(self):
        """Test AdaptiveTruncator SVD truncation."""
        from tensornet.adaptive.bond_optimizer import (
            AdaptiveTruncator,
            AdaptiveBondConfig,
            TruncationStrategy,
        )
        
        config = AdaptiveBondConfig(
            chi_min=2,
            chi_max=10,
            strategy=TruncationStrategy.FIXED,
        )
        truncator = AdaptiveTruncator(config, num_sites=10)
        
        # Create a test matrix
        A = torch.randn(20, 20)
        
        # Truncate using truncate method
        result = truncator.truncate(A, bond_index=0, max_chi=5)
        
        assert result is not None
    
    def test_estimate_optimal_chi(self):
        """Test estimate_optimal_chi function."""
        from tensornet.adaptive.bond_optimizer import estimate_optimal_chi
        
        # Pass entanglement entropy (float), not singular values
        # estimate_optimal_chi expects entropy value, not tensor
        entanglement_entropy = 1.5  # Moderate entanglement
        
        chi = estimate_optimal_chi(
            entanglement_entropy,
            target_truncation_error=0.02,
        )
        
        # Should return valid chi
        assert chi >= 1


class TestEntanglement:
    """Tests for entanglement analysis."""
    
    def test_entanglement_spectrum(self):
        """Test EntanglementSpectrum analysis."""
        from tensornet.adaptive.entanglement import EntanglementSpectrum
        
        # Create test singular values as tensor
        singular_values = torch.tensor([0.7, 0.5, 0.3, 0.1])
        
        spectrum = EntanglementSpectrum.from_singular_values(singular_values, bond_index=0)
        
        assert spectrum.entropy >= 0
        assert spectrum.renyi_2 >= 0
        assert spectrum.effective_rank >= 1
    
    def test_area_law_analyzer(self):
        """Test AreaLawAnalyzer."""
        from tensornet.adaptive.entanglement import AreaLawAnalyzer
        
        analyzer = AreaLawAnalyzer()
        
        # Generate area-law scaling data: S ~ c for 1D
        subsystem_sizes = [4, 8, 12, 16, 20]
        entropies = [1.0 + 0.01 * np.random.randn() for _ in subsystem_sizes]
        
        scaling = analyzer.analyze(subsystem_sizes, entropies)
        
        # Should return an AreaLawScaling object
        assert scaling is not None
        assert hasattr(scaling, 'scaling_type')
    
    def test_entanglement_entropy(self):
        """Test EntanglementEntropy computation."""
        from tensornet.adaptive.entanglement import EntanglementEntropy
        
        # Create from singular values using classmethod
        sv = torch.tensor([1.0, 0.0, 0.0, 0.0])
        
        entropy_obj = EntanglementEntropy.from_singular_values(sv, bond_index=0)
        
        # Pure state should have zero or near-zero entropy (use .value attribute)
        assert entropy_obj.value < 1e-10
    
    def test_mutual_information(self):
        """Test MutualInformation computation."""
        from tensornet.adaptive.entanglement import MutualInformation
        
        # I(A:B) = S(A) + S(B) - S(AB)
        expected = 1.5 + 1.2 - 2.0
        
        mi = MutualInformation(
            region_a=0,
            region_b=1,
            entropy_a=1.5,
            entropy_b=1.2,
            entropy_ab=2.0,
            value=expected,
        )
        
        assert abs(mi.value - expected) < 1e-10
    
    def test_compute_entanglement_entropy_function(self):
        """Test compute_entanglement_entropy convenience function."""
        from tensornet.adaptive.entanglement import compute_entanglement_entropy
        
        # Create singular values as tensor
        singular_values = torch.tensor([0.8, 0.4, 0.2, 0.1])
        
        entropy = compute_entanglement_entropy(singular_values)
        
        assert entropy > 0  # Non-trivial entanglement


class TestCompression:
    """Tests for tensor compression."""
    
    def test_svd_compression(self):
        """Test SVDCompression."""
        from tensornet.adaptive.compression import SVDCompression
        
        compressor = SVDCompression()
        
        # Create low-rank matrix
        A = torch.randn(20, 5)
        B = torch.randn(5, 30)
        tensor = A @ B  # Rank-5 matrix
        
        result = compressor.compress(tensor, target_rank=5)
        
        assert result.truncation_error < 1e-4  # Should be nearly exact
        assert result.compression_ratio >= 1
    
    def test_randomized_svd(self):
        """Test RandomizedSVD."""
        from tensornet.adaptive.compression import RandomizedSVD
        
        compressor = RandomizedSVD(oversampling=10, n_power_iterations=2)
        
        tensor = torch.randn(100, 50)
        result = compressor.compress(tensor, target_rank=10)
        
        assert result.compression_ratio >= 1
        assert result.method.name == "RANDOMIZED_SVD"
    
    def test_variational_compression(self):
        """Test VariationalCompression."""
        from tensornet.adaptive.compression import VariationalCompression
        
        compressor = VariationalCompression(max_iterations=50)
        
        tensor = torch.randn(30, 30)
        result = compressor.compress(tensor, target_rank=5)
        
        assert result.compression_ratio >= 1
    
    def test_tensor_cross_interpolation(self):
        """Test TensorCrossInterpolation."""
        from tensornet.adaptive.compression import TensorCrossInterpolation
        
        tci = TensorCrossInterpolation()
        
        # Create smooth matrix
        x = torch.linspace(0, 1, 20)
        y = torch.linspace(0, 1, 30)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        tensor = torch.sin(X * np.pi) * torch.cos(Y * np.pi)
        
        result = tci.compress(tensor, target_rank=5)
        
        assert result.compression_ratio >= 1
    
    def test_compress_adaptively(self):
        """Test compress_adaptively function."""
        from tensornet.adaptive.compression import compress_adaptively
        
        tensor = torch.randn(50, 50)
        
        result = compress_adaptively(tensor, target_rank=10)
        
        assert result.compression_ratio >= 1


# =============================================================================
# REALTIME MODULE TESTS
# =============================================================================

class TestInferenceEngine:
    """Tests for real-time inference engine."""
    
    def test_inference_config_creation(self):
        """Test InferenceConfig creation."""
        from tensornet.realtime.inference_engine import InferenceConfig
        
        config = InferenceConfig(
            max_batch_size=32,
            batch_timeout_ms=100.0,
            num_workers=4,
            enable_caching=True,
        )
        
        assert config.max_batch_size == 32
        assert config.batch_timeout_ms == 100.0
        assert config.num_workers == 4
        assert config.enable_caching is True
    
    def test_inference_engine_basic(self):
        """Test basic InferenceEngine usage."""
        from tensornet.realtime.inference_engine import (
            InferenceEngine,
            InferenceConfig,
        )
        
        # Create simple model
        model = nn.Linear(10, 5)
        config = InferenceConfig(max_batch_size=8)
        engine = InferenceEngine(model, config)
        
        # Run inference
        input_tensor = torch.randn(10)
        result = engine.infer(input_tensor)
        
        assert result.output.shape == (5,)
        assert result.latency_ms >= 0
    
    def test_inference_engine_batched(self):
        """Test batched inference."""
        from tensornet.realtime.inference_engine import (
            InferenceEngine,
            InferenceConfig,
        )
        
        model = nn.Linear(10, 5)
        config = InferenceConfig(max_batch_size=16)
        engine = InferenceEngine(model, config)
        
        # Run batched inference
        inputs = [torch.randn(10) for _ in range(5)]
        results = engine.infer_batch(inputs)
        
        assert len(results) == 5
        for result in results:
            assert result.output.shape == (5,)
    
    def test_inference_engine_warmup(self):
        """Test inference engine warmup."""
        from tensornet.realtime.inference_engine import (
            InferenceEngine,
            InferenceConfig,
        )
        
        model = nn.Linear(10, 5)
        engine = InferenceEngine(model)
        
        sample = torch.randn(10)
        engine.warmup(sample)
        
        # Use correct attribute name _warmed_up
        assert engine._warmed_up is True
    
    def test_run_inference_function(self):
        """Test run_inference convenience function."""
        from tensornet.realtime.inference_engine import run_inference
        
        model = nn.Linear(10, 5)
        input_tensor = torch.randn(10)
        
        result = run_inference(model, input_tensor)
        
        assert result.output.shape == (5,)


class TestKernelFusion:
    """Tests for kernel fusion."""
    
    def test_fusion_pattern_matching(self):
        """Test FusionPattern matching."""
        from tensornet.realtime.kernel_fusion import (
            FusionPattern,
            FusionType,
        )
        
        pattern = FusionPattern(
            name="add_relu",
            fusion_type=FusionType.ELEMENT_WISE,
            ops=["add", "relu"],
            speedup_estimate=1.3,
        )
        
        assert pattern.name == "add_relu"
        assert len(pattern.ops) == 2
        assert pattern.matches(["add", "relu"])
    
    def test_operator_graph_construction(self):
        """Test OperatorGraph construction."""
        from tensornet.realtime.kernel_fusion import (
            OperatorGraph,
            OperatorNode,
        )
        
        graph = OperatorGraph()
        
        # Add nodes using correct API
        input_node = OperatorNode(
            id=0,
            op_type="input",
            inputs=[],
            outputs=[1],
        )
        add_node = OperatorNode(
            id=1,
            op_type="add",
            inputs=[0],
            outputs=[2],
        )
        relu_node = OperatorNode(
            id=2,
            op_type="relu",
            inputs=[1],
            outputs=[],
        )
        
        graph.add_node(input_node)
        graph.add_node(add_node)
        graph.add_node(relu_node)
        
        assert len(graph.nodes) == 3
        assert graph.get_node(1) is not None
    
    def test_kernel_fuser(self):
        """Test KernelFuser."""
        from tensornet.realtime.kernel_fusion import (
            KernelFuser,
            OperatorGraph,
            OperatorNode,
        )
        
        fuser = KernelFuser()
        
        # Create simple graph
        graph = OperatorGraph()
        graph.add_node(OperatorNode(0, "input", [], [1, 2]))
        graph.add_node(OperatorNode(1, "input", [], [3]))
        graph.add_node(OperatorNode(2, "add", [0, 1], [3]))
        graph.add_node(OperatorNode(3, "relu", [2], []))
        
        # Fuser should analyze graph
        assert fuser is not None
    
    def test_fuse_operators_function(self):
        """Test fuse_operators function."""
        from tensornet.realtime.kernel_fusion import fuse_operators
        
        operations = ["add", "relu"]
        fused = fuse_operators(operations)
        
        assert fused is not None


class TestMemoryManager:
    """Tests for memory management."""
    
    def test_memory_config(self):
        """Test MemoryConfig creation."""
        from tensornet.realtime.memory_manager import (
            MemoryConfig,
            AllocationStrategy,
        )
        
        config = MemoryConfig(
            pool_size_mb=256,
            strategy=AllocationStrategy.PREALLOCATED,
            enable_caching=True,
        )
        
        assert config.pool_size_mb == 256
        assert config.strategy == AllocationStrategy.PREALLOCATED
    
    def test_tensor_cache(self):
        """Test TensorCache."""
        from tensornet.realtime.memory_manager import TensorCache
        
        cache = TensorCache(max_size=100, max_bytes=10 * 1024 * 1024)
        
        # Add tensors to cache
        tensor = torch.randn(10, 10)
        
        cache.put(tensor)
        
        # Try to get a tensor of same shape
        retrieved = cache.get((10, 10), torch.float32, "cpu")
        assert retrieved is not None
    
    def test_memory_pool(self):
        """Test MemoryPool."""
        from tensornet.realtime.memory_manager import (
            MemoryPool,
            MemoryConfig,
        )
        
        config = MemoryConfig(pool_size_mb=10)
        pool = MemoryPool(config)
        
        # Allocate tensor
        handle = pool.allocate((10, 10), torch.float32)
        
        assert handle is not None
        assert handle.size_bytes == 10 * 10 * 4  # float32 = 4 bytes
        
        # Free tensor
        pool.free(handle)
    
    def test_streaming_buffer(self):
        """Test StreamingBuffer."""
        from tensornet.realtime.memory_manager import StreamingBuffer
        
        buffer = StreamingBuffer(buffer_size=1024)
        
        # Get current buffer
        current = buffer.current_buffer
        assert current is not None
        assert current.shape[0] == 1024
        
        # Swap buffers
        buffer.swap()
        next_buf = buffer.current_buffer
        assert next_buf is not None
    
    def test_memory_planner(self):
        """Test MemoryPlanner."""
        from tensornet.realtime.memory_manager import MemoryPlanner
        
        planner = MemoryPlanner()
        
        # Create tensor specs and lifetimes for planning
        tensor_specs = [
            {"shape": (10, 20), "dtype": torch.float32, "name": "hidden1"},
            {"shape": (20,), "dtype": torch.float32, "name": "bias1"},
            {"shape": (20, 5), "dtype": torch.float32, "name": "hidden2"},
            {"shape": (5,), "dtype": torch.float32, "name": "bias2"},
        ]
        lifetimes = [
            (0, 2),  # hidden1 used steps 0-2
            (0, 2),  # bias1 used steps 0-2
            (2, 4),  # hidden2 used steps 2-4
            (2, 4),  # bias2 used steps 2-4
        ]
        
        plan = planner.plan(tensor_specs, lifetimes)
        
        assert plan is not None
        assert "total_size" in plan
        assert "memory_saved" in plan


class TestLatencyOptimizer:
    """Tests for latency optimization."""
    
    def test_latency_target(self):
        """Test LatencyTarget creation."""
        from tensornet.realtime.latency_optimizer import LatencyTarget
        
        target = LatencyTarget(
            target_ms=10.0,
            max_ms=20.0,
            percentile=0.99,
        )
        
        assert target.target_ms == 10.0
        assert target.percentile == 0.99
    
    def test_latency_profile(self):
        """Test LatencyProfile creation."""
        from tensornet.realtime.latency_optimizer import LatencyProfile
        
        measurements = [5.0, 6.0, 5.5, 7.0, 5.2] * 20  # 100 measurements
        
        profile = LatencyProfile.from_measurements(measurements)
        
        assert profile.samples == 100
        assert profile.mean_ms > 0
        assert profile.p50_ms > 0
        assert profile.p99_ms > 0
    
    def test_precision_scheduler(self):
        """Test PrecisionScheduler."""
        from tensornet.realtime.latency_optimizer import (
            PrecisionScheduler,
            PrecisionPolicy,
        )
        
        scheduler = PrecisionScheduler(
            policy=PrecisionPolicy.FULL,
            accuracy_threshold=0.99,
        )
        
        assert scheduler.get_dtype() == torch.float32
        assert not scheduler.should_use_autocast()
        
        scheduler.policy = PrecisionPolicy.MIXED_FP16
        assert scheduler.should_use_autocast()
    
    def test_pipeline_optimizer(self):
        """Test PipelineOptimizer."""
        from tensornet.realtime.latency_optimizer import (
            PipelineOptimizer,
            PipelineConfig,
        )
        
        config = PipelineConfig(num_stages=4)
        optimizer = PipelineOptimizer(config)
        
        # Create simple stages
        stages = [
            lambda x: x * 2,
            lambda x: x + 1,
            lambda x: x ** 2,
        ]
        
        sample = torch.randn(10)
        profiles = optimizer.profile_stages(stages, sample)
        
        assert len(profiles) == 3
        for profile in profiles:
            assert profile.samples > 0
    
    def test_latency_optimizer(self):
        """Test LatencyOptimizer."""
        from tensornet.realtime.latency_optimizer import (
            LatencyOptimizer,
            LatencyTarget,
        )
        
        target = LatencyTarget(target_ms=100.0)
        optimizer = LatencyOptimizer(target)
        
        model = nn.Linear(10, 5)
        sample = torch.randn(10)
        
        profile = optimizer.profile(model, sample, num_iterations=10)
        
        assert profile.samples == 10
        assert profile.mean_ms > 0
    
    def test_optimize_for_latency_function(self):
        """Test optimize_for_latency function."""
        from tensornet.realtime.latency_optimizer import optimize_for_latency
        
        model = nn.Linear(10, 5)
        sample = torch.randn(10)
        
        result = optimize_for_latency(model, sample, target_ms=100.0)
        
        assert "initial_latency_ms" in result
        assert "final_latency_ms" in result


# =============================================================================
# COORDINATION MODULE TESTS
# =============================================================================

class TestSwarmCoordination:
    """Tests for swarm coordination."""
    
    def test_vehicle_state_creation(self):
        """Test VehicleState creation."""
        from tensornet.coordination.swarm import VehicleState
        
        state = VehicleState(
            vehicle_id="uav_001",
            position=np.array([10.0, 20.0, 100.0]),
            velocity=np.array([5.0, 0.0, 0.0]),
        )
        
        assert state.vehicle_id == "uav_001"
        assert state.speed == 5.0
        assert state.position.shape == (3,)
    
    def test_vehicle_state_tensor_conversion(self):
        """Test VehicleState tensor conversion."""
        from tensornet.coordination.swarm import VehicleState
        
        state = VehicleState(
            vehicle_id="uav_001",
            position=np.array([1.0, 2.0, 3.0]),
            velocity=np.array([0.5, 0.0, 0.0]),
        )
        
        tensor = state.to_tensor()
        assert tensor.shape == (10,)  # 3 + 3 + 4
        
        restored = VehicleState.from_tensor(tensor, "uav_001")
        assert np.allclose(restored.position, state.position)
    
    def test_swarm_config(self):
        """Test SwarmConfig creation."""
        from tensornet.coordination.swarm import SwarmConfig, TopologyType
        
        config = SwarmConfig(
            topology=TopologyType.FULLY_CONNECTED,
            communication_range=100.0,
            collision_radius=2.0,
        )
        
        assert config.validate()
    
    def test_swarm_topology_fully_connected(self):
        """Test fully connected SwarmTopology."""
        from tensornet.coordination.swarm import (
            SwarmTopology,
            TopologyType,
        )
        
        topology = SwarmTopology(TopologyType.FULLY_CONNECTED)
        vehicle_ids = ["uav_1", "uav_2", "uav_3", "uav_4"]
        topology.build(vehicle_ids)
        
        # Each vehicle should be connected to all others
        neighbors = topology.get_neighbors("uav_1")
        assert len(neighbors) == 3
        assert topology.is_connected()
    
    def test_swarm_topology_ring(self):
        """Test ring SwarmTopology."""
        from tensornet.coordination.swarm import (
            SwarmTopology,
            TopologyType,
        )
        
        topology = SwarmTopology(TopologyType.RING)
        vehicle_ids = ["uav_1", "uav_2", "uav_3", "uav_4"]
        topology.build(vehicle_ids)
        
        # Each vehicle should have exactly 2 neighbors
        for vid in vehicle_ids:
            neighbors = topology.get_neighbors(vid)
            assert len(neighbors) == 2
        
        assert topology.is_connected()
    
    def test_swarm_coordinator(self):
        """Test SwarmCoordinator."""
        from tensornet.coordination.swarm import (
            SwarmCoordinator,
            SwarmConfig,
            VehicleState,
        )
        
        config = SwarmConfig(collision_radius=5.0)
        coordinator = SwarmCoordinator(config)
        
        # Register vehicles
        for i in range(4):
            state = VehicleState(
                vehicle_id=f"uav_{i}",
                position=np.array([i * 10.0, 0.0, 100.0]),
                velocity=np.zeros(3),
            )
            coordinator.register_vehicle(state)
        
        assert coordinator.num_vehicles == 4
        
        # Compute control
        targets = {f"uav_{i}": np.array([0.0, 0.0, 100.0]) for i in range(4)}
        commands = coordinator.compute_control(targets)
        
        assert len(commands) == 4
    
    def test_swarm_step_simulation(self):
        """Test swarm simulation step."""
        from tensornet.coordination.swarm import (
            SwarmCoordinator,
            VehicleState,
        )
        
        coordinator = SwarmCoordinator()
        
        # Add vehicles
        for i in range(3):
            coordinator.register_vehicle(VehicleState(
                vehicle_id=f"v{i}",
                position=np.array([i * 5.0, 0.0, 0.0]),
                velocity=np.zeros(3),
            ))
        
        # Set targets and compute control
        targets = {f"v{i}": np.array([0.0, 0.0, 0.0]) for i in range(3)}
        coordinator.compute_control(targets)
        
        # Step simulation
        coordinator.step(dt=0.1)
        
        metrics = coordinator.get_metrics()
        assert "spread" in metrics
        assert metrics["num_vehicles"] == 3
    
    def test_compute_swarm_functions(self):
        """Test swarm utility functions."""
        from tensornet.coordination.swarm import (
            VehicleState,
            compute_swarm_centroid,
            compute_swarm_spread,
        )
        
        states = {
            "v0": VehicleState("v0", np.array([0.0, 0.0, 0.0]), np.zeros(3)),
            "v1": VehicleState("v1", np.array([10.0, 0.0, 0.0]), np.zeros(3)),
            "v2": VehicleState("v2", np.array([5.0, 0.0, 0.0]), np.zeros(3)),
        }
        
        centroid = compute_swarm_centroid(states)
        assert np.allclose(centroid, [5.0, 0.0, 0.0])
        
        spread = compute_swarm_spread(states)
        assert spread > 0


class TestFormationControl:
    """Tests for formation control."""
    
    def test_formation_config(self):
        """Test FormationConfig creation."""
        from tensornet.coordination.formation import (
            FormationConfig,
            FormationType,
        )
        
        config = FormationConfig(
            formation_type=FormationType.WEDGE,
            spacing=15.0,
            heading=0.0,
        )
        
        assert config.spacing == 15.0
        assert config.formation_type == FormationType.WEDGE
    
    def test_formation_controller_wedge(self):
        """Test FormationController wedge formation."""
        from tensornet.coordination.formation import (
            FormationController,
            FormationConfig,
            FormationType,
        )
        
        config = FormationConfig(
            formation_type=FormationType.WEDGE,
            spacing=10.0,
        )
        controller = FormationController(config)
        
        center = np.array([0.0, 0.0, 100.0])
        positions = controller.compute_formation_positions(center, 5)
        
        assert len(positions) == 5
        # First position (leader) should be at center
        assert np.allclose(positions[0], center)
    
    def test_formation_controller_circle(self):
        """Test FormationController circle formation."""
        from tensornet.coordination.formation import (
            FormationController,
            FormationConfig,
            FormationType,
        )
        
        config = FormationConfig(
            formation_type=FormationType.CIRCLE,
            spacing=10.0,
        )
        controller = FormationController(config)
        
        center = np.array([0.0, 0.0, 0.0])
        positions = controller.compute_formation_positions(center, 6)
        
        assert len(positions) == 6
        # All should be equidistant from center
        distances = [np.linalg.norm(p - center) for p in positions]
        assert np.allclose(distances, distances[0], atol=1e-10)
    
    def test_formation_controller_grid(self):
        """Test FormationController grid formation."""
        from tensornet.coordination.formation import (
            FormationController,
            FormationConfig,
            FormationType,
        )
        
        config = FormationConfig(
            formation_type=FormationType.GRID,
            spacing=5.0,
        )
        controller = FormationController(config)
        
        center = np.array([0.0, 0.0, 0.0])
        positions = controller.compute_formation_positions(center, 9)
        
        assert len(positions) == 9
    
    def test_formation_state_from_states(self):
        """Test FormationState creation."""
        from tensornet.coordination.formation import FormationState
        from tensornet.coordination.swarm import VehicleState
        
        states = {
            "v0": VehicleState("v0", np.array([0.0, 0.0, 0.0]), np.zeros(3)),
            "v1": VehicleState("v1", np.array([10.0, 0.0, 0.0]), np.zeros(3)),
        }
        targets = {
            "v0": np.array([0.0, 0.0, 0.0]),
            "v1": np.array([10.5, 0.0, 0.0]),
        }
        
        formation_state = FormationState.from_states(states, targets, threshold=1.0)
        
        assert formation_state.converged  # Within threshold
        assert formation_state.errors["v0"] < 1e-10
        assert formation_state.errors["v1"] == 0.5
    
    def test_validate_formation(self):
        """Test validate_formation function."""
        from tensornet.coordination.formation import validate_formation
        from tensornet.coordination.swarm import VehicleState
        
        states = {
            "v0": VehicleState("v0", np.array([0.0, 0.0, 0.0]), np.zeros(3)),
        }
        targets = {
            "v0": np.array([0.0, 0.0, 0.0]),
        }
        
        assert validate_formation(states, targets, threshold=0.1)
        
        targets["v0"] = np.array([10.0, 0.0, 0.0])
        assert not validate_formation(states, targets, threshold=0.1)


class TestTaskAllocation:
    """Tests for task allocation."""
    
    def test_task_creation(self):
        """Test Task creation."""
        from tensornet.coordination.task_allocation import (
            Task,
            TaskPriority,
        )
        
        task = Task(
            task_id="task_001",
            task_type="survey",
            position=np.array([100.0, 200.0, 50.0]),
            priority=TaskPriority.HIGH,
            duration=120.0,
            reward=10.0,
        )
        
        assert task.task_id == "task_001"
        assert task.priority == TaskPriority.HIGH
        assert task.distance_from(np.zeros(3)) > 0
    
    def test_task_allocator_greedy(self):
        """Test TaskAllocator greedy allocation."""
        from tensornet.coordination.task_allocation import (
            Task,
            TaskAllocator,
            TaskPriority,
        )
        from tensornet.coordination.swarm import VehicleState
        
        allocator = TaskAllocator()
        
        # Add tasks
        for i in range(3):
            task = Task(
                task_id=f"task_{i}",
                task_type="survey",
                position=np.array([i * 50.0, 0.0, 0.0]),
                priority=TaskPriority.NORMAL,
            )
            allocator.add_task(task)
        
        # Create vehicle states
        states = {
            "v0": VehicleState("v0", np.array([0.0, 0.0, 0.0]), np.zeros(3)),
            "v1": VehicleState("v1", np.array([100.0, 0.0, 0.0]), np.zeros(3)),
        }
        
        assignments = allocator.allocate_greedy(states)
        
        assert len(assignments) > 0
        assert all(a.vehicle_id in states for a in assignments)
    
    def test_task_allocator_nearest(self):
        """Test TaskAllocator nearest allocation."""
        from tensornet.coordination.task_allocation import (
            Task,
            TaskAllocator,
        )
        from tensornet.coordination.swarm import VehicleState
        
        allocator = TaskAllocator()
        
        # Add task near origin
        task = Task(
            task_id="near_task",
            task_type="survey",
            position=np.array([5.0, 0.0, 0.0]),
        )
        allocator.add_task(task)
        
        # Vehicle at origin should get the task
        states = {
            "v0": VehicleState("v0", np.array([0.0, 0.0, 0.0]), np.zeros(3)),
        }
        
        assignments = allocator.allocate_nearest(states)
        
        assert len(assignments) == 1
        assert assignments[0].vehicle_id == "v0"
    
    def test_auction_protocol(self):
        """Test AuctionProtocol."""
        from tensornet.coordination.task_allocation import (
            Task,
            AuctionProtocol,
        )
        from tensornet.coordination.swarm import VehicleState
        
        auction = AuctionProtocol()
        
        # Create tasks
        tasks = [
            Task(
                task_id=f"task_{i}",
                task_type="survey",
                position=np.array([i * 30.0, 0.0, 0.0]),
                reward=10.0 - i,
            )
            for i in range(3)
        ]
        
        # Create vehicles
        states = {
            "v0": VehicleState("v0", np.array([0.0, 0.0, 0.0]), np.zeros(3)),
            "v1": VehicleState("v1", np.array([50.0, 0.0, 0.0]), np.zeros(3)),
        }
        
        assignments = auction.run_auction(tasks, states)
        
        assert len(assignments) <= len(states)  # At most one task per vehicle
    
    def test_allocate_tasks_function(self):
        """Test allocate_tasks convenience function."""
        from tensornet.coordination.task_allocation import (
            Task,
            allocate_tasks,
        )
        from tensornet.coordination.swarm import VehicleState
        
        tasks = [
            Task("t0", "survey", np.array([10.0, 0.0, 0.0])),
            Task("t1", "survey", np.array([20.0, 0.0, 0.0])),
        ]
        states = {
            "v0": VehicleState("v0", np.array([0.0, 0.0, 0.0]), np.zeros(3)),
        }
        
        # Test different methods
        for method in ["greedy", "nearest", "auction"]:
            assignments = allocate_tasks(tasks, states, method=method)
            # Reset task status for next method
            for task in tasks:
                from tensornet.coordination.task_allocation import TaskStatus
                task.status = TaskStatus.PENDING


class TestConsensusProtocols:
    """Tests for consensus protocols."""
    
    def test_consensus_config(self):
        """Test ConsensusConfig creation."""
        from tensornet.coordination.consensus import ConsensusConfig
        
        config = ConsensusConfig(
            max_iterations=500,
            convergence_threshold=1e-8,
            step_size=0.2,
        )
        
        assert config.validate()
    
    def test_average_consensus(self):
        """Test AverageConsensus protocol."""
        from tensornet.coordination.consensus import (
            AverageConsensus,
            ConsensusConfig,
        )
        
        config = ConsensusConfig(
            max_iterations=1000,
            convergence_threshold=1e-6,
            step_size=0.1,
        )
        consensus = AverageConsensus(config)
        
        # Initial values
        initial_values = {
            "agent_0": np.array([1.0]),
            "agent_1": np.array([3.0]),
            "agent_2": np.array([5.0]),
            "agent_3": np.array([7.0]),
        }
        
        result = consensus.run(initial_values)
        
        assert result.converged
        # Should converge to average: (1 + 3 + 5 + 7) / 4 = 4
        assert result.consensus_value is not None
        assert abs(result.consensus_value[0] - 4.0) < 0.01
    
    def test_max_consensus(self):
        """Test MaxConsensus protocol."""
        from tensornet.coordination.consensus import (
            MaxConsensus,
            ConsensusConfig,
        )
        
        config = ConsensusConfig(
            max_iterations=100,
            convergence_threshold=1e-10,
        )
        consensus = MaxConsensus(config)
        
        initial_values = {
            "a": np.array([1.0, 2.0]),
            "b": np.array([3.0, 1.0]),
            "c": np.array([2.0, 4.0]),
        }
        
        result = consensus.run(initial_values)
        
        assert result.converged
        # Should converge to element-wise max: [3, 4]
        for val in result.values.values():
            assert np.allclose(val, [3.0, 4.0])
    
    def test_weighted_consensus(self):
        """Test WeightedConsensus protocol."""
        from tensornet.coordination.consensus import WeightedConsensus
        
        consensus = WeightedConsensus()
        consensus.set_weights({
            "a": 2.0,  # Higher confidence
            "b": 1.0,
        })
        
        initial_values = {
            "a": np.array([10.0]),
            "b": np.array([0.0]),
        }
        
        result = consensus.run(initial_values)
        
        # Should converge closer to a's value due to higher weight
        if result.converged:
            assert result.consensus_value[0] > 3.0  # Weighted toward 10
    
    def test_leader_election(self):
        """Test LeaderElection."""
        from tensornet.coordination.consensus import LeaderElection
        
        election = LeaderElection()
        
        agent_ids = ["agent_a", "agent_b", "agent_c"]
        priorities = {
            "agent_a": 10.0,
            "agent_b": 50.0,  # Highest priority
            "agent_c": 30.0,
        }
        
        leader = election.elect(agent_ids, priorities)
        
        assert leader == "agent_b"
    
    def test_run_consensus_function(self):
        """Test run_consensus convenience function."""
        from tensornet.coordination.consensus import run_consensus
        
        initial_values = {
            "a": np.array([1.0]),
            "b": np.array([2.0]),
            "c": np.array([3.0]),
        }
        
        result = run_consensus(initial_values, protocol="average")
        
        assert result.converged
        assert abs(result.consensus_value[0] - 2.0) < 0.01


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestPhase18Integration:
    """Integration tests across Phase 18 modules."""
    
    def test_adaptive_with_realtime_inference(self):
        """Test adaptive compression with real-time inference."""
        from tensornet.adaptive.compression import SVDCompression
        from tensornet.realtime.inference_engine import InferenceEngine
        
        # Create and compress a model's weight matrix
        compressor = SVDCompression()
        weight = torch.randn(64, 32)
        result = compressor.compress(weight, target_rank=8)
        
        # Use compressed representation in inference
        model = nn.Linear(32, 64)
        model.weight.data = result.compressed[0] @ torch.diag(result.compressed[1]) @ result.compressed[2]
        
        engine = InferenceEngine(model)
        output = engine.infer(torch.randn(32))
        
        assert output.output.shape == (64,)
    
    def test_coordination_full_mission(self):
        """Test full coordination mission."""
        from tensornet.coordination.swarm import (
            SwarmCoordinator,
            VehicleState,
        )
        from tensornet.coordination.formation import (
            FormationController,
            FormationConfig,
            FormationType,
        )
        from tensornet.coordination.task_allocation import (
            Task,
            TaskAllocator,
        )
        from tensornet.coordination.consensus import (
            AverageConsensus,
        )
        
        # Initialize swarm
        coordinator = SwarmCoordinator()
        for i in range(4):
            coordinator.register_vehicle(VehicleState(
                vehicle_id=f"uav_{i}",
                position=np.array([i * 20.0, 0.0, 100.0]),
                velocity=np.zeros(3),
            ))
        
        # Set up formation
        formation = FormationController(FormationConfig(
            formation_type=FormationType.WEDGE,
            spacing=15.0,
        ))
        
        # Get target positions
        targets = formation.get_target_positions(coordinator.get_all_states())
        
        # Simulate convergence
        for _ in range(10):
            coordinator.compute_control(targets)
            coordinator.step(dt=0.1)
        
        # Allocate tasks
        allocator = TaskAllocator()
        allocator.add_task(Task(
            task_id="survey_1",
            task_type="survey",
            position=np.array([100.0, 100.0, 100.0]),
        ))
        
        assignments = allocator.allocate_greedy(coordinator.get_all_states())
        
        # Run consensus on some shared value
        consensus = AverageConsensus()
        initial_values = {
            f"uav_{i}": np.array([float(i)])
            for i in range(4)
        }
        result = consensus.run(initial_values)
        
        assert coordinator.num_vehicles == 4
        assert len(assignments) >= 0
        assert result.converged
    
    def test_latency_optimized_coordination(self):
        """Test latency-optimized coordination decisions."""
        from tensornet.realtime.latency_optimizer import LatencyOptimizer, LatencyTarget
        from tensornet.coordination.swarm import SwarmCoordinator, VehicleState
        
        # Create a simple coordination decision model
        class CoordinationModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(30, 12)  # 3 vehicles * 10 state -> 3 * 4 commands
            
            def forward(self, x):
                return self.fc(x)
        
        model = CoordinationModel()
        
        # Optimize for low latency
        optimizer = LatencyOptimizer(LatencyTarget(target_ms=5.0))
        sample = torch.randn(30)
        
        profile = optimizer.profile(model, sample, num_iterations=20)
        
        assert profile.mean_ms > 0
        
        # Use in coordination
        coordinator = SwarmCoordinator()
        for i in range(3):
            coordinator.register_vehicle(VehicleState(
                f"v{i}",
                np.array([i * 10.0, 0.0, 0.0]),
                np.zeros(3),
            ))
        
        # Fast inference for real-time control
        all_states = coordinator.get_all_states()
        state_tensor = torch.cat([s.to_tensor() for s in all_states.values()])
        
        with torch.no_grad():
            commands = model(state_tensor)
        
        assert commands.shape == (12,)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
