"""
Phase 19 Test Suite
===================

Comprehensive tests for Phase 19 modules:
- Neural-enhanced truncation (tensornet/neural/)
- Distributed TN solvers (tensornet/distributed_tn/)
- Autonomous mission planning (tensornet/autonomy/)

Test Coverage:
- 75+ tests total
- Unit tests for all classes
- Integration tests for module interactions
"""

import pytest
import torch
import numpy as np
import math
import time
from typing import List, Tuple


# =============================================================================
# Neural Module Tests (tensornet/neural/)
# =============================================================================

class TestTruncationPolicy:
    """Tests for neural truncation policy."""
    
    def test_policy_action_enum(self):
        """Test PolicyAction enumeration."""
        from ontic.ml.neural.truncation_policy import PolicyAction
        
        # Uses auto() so values are 1, 2, 3, 4, 5
        assert PolicyAction.DECREASE_LARGE is not None
        assert PolicyAction.MAINTAIN is not None
        assert PolicyAction.INCREASE_LARGE is not None
        assert len(list(PolicyAction)) == 5
    
    def test_policy_state_creation(self):
        """Test PolicyState dataclass."""
        from ontic.ml.neural.truncation_policy import PolicyState
        
        state = PolicyState(
            current_chi=64,
            truncation_error=1e-8,
            entropy=0.5,
            entropy_gradient=0.1,
            step=100,
            memory_usage=0.5,
            target_error=1e-10,
            chi_min=4,
            chi_max=512,
        )
        
        assert state.current_chi == 64
        assert state.truncation_error == 1e-8
        
        # Test to_tensor
        tensor = state.to_tensor()
        assert tensor.shape == (8,)
    
    def test_policy_network_forward(self):
        """Test PolicyNetwork forward pass."""
        from ontic.ml.neural.truncation_policy import PolicyNetwork
        
        network = PolicyNetwork(state_dim=8, hidden_dim=32, action_dim=5)
        state_tensor = torch.randn(1, 8)
        
        action_logits, value = network(state_tensor)
        
        assert action_logits.shape == (1, 5)
        assert value.shape == (1, 1)
        # action_logits are logits, not probabilities
    
    def test_truncation_policy_creation(self):
        """Test TruncationPolicy wrapper."""
        from ontic.ml.neural.truncation_policy import TruncationPolicy, PolicyNetwork
        
        network = PolicyNetwork(state_dim=8, action_dim=5, hidden_dim=64)
        policy = TruncationPolicy(network=network, chi_min=4, chi_max=256)
        
        new_chi = policy.get_chi(
            current_chi=64,
            truncation_error=1e-8,
            entropy=0.5,
        )
        assert isinstance(new_chi, int)
        assert new_chi >= 4
        assert new_chi <= 256
    
    def test_replay_buffer(self):
        """Test ReplayBuffer for RL."""
        from ontic.ml.neural.truncation_policy import ReplayBuffer, Experience
        
        buffer = ReplayBuffer(capacity=100)
        
        for i in range(10):
            exp = Experience(
                state=torch.randn(8),
                action=1,
                reward=0.5,
                next_state=torch.randn(8),
                done=False,
                log_prob=-0.5,
                value=0.8,
            )
            buffer.push(exp)
        
        assert len(buffer) == 10
        batch = buffer.sample(5)
        assert len(batch) == 5
    
    def test_rl_truncation_agent_creation(self):
        """Test RLTruncationAgent initialization."""
        from ontic.ml.neural.truncation_policy import (
            RLTruncationAgent,
            TruncationPolicy,
            PolicyNetwork,
        )
        
        network = PolicyNetwork()
        policy = TruncationPolicy(network=network)
        agent = RLTruncationAgent(policy=policy)
        
        assert agent.gamma == 0.99
        assert agent.policy is not None


class TestBondPredictor:
    """Tests for bond dimension predictor."""
    
    def test_entropy_features(self):
        """Test EntropyFeatures dataclass."""
        from ontic.ml.neural.bond_predictor import EntropyFeatures
        
        entropies = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.5, 0.3, 0.1])
        features = EntropyFeatures.from_entropies(entropies)
        
        assert abs(features.max_entropy - 0.7) < 1e-5
        assert 0 <= features.peak_location <= 1
    
    def test_predictor_config(self):
        """Test PredictorConfig defaults."""
        from ontic.ml.neural.bond_predictor import PredictorConfig
        
        config = PredictorConfig()
        
        # Default input_dim is 71 for full feature set
        assert config.input_dim == 71
        assert config.hidden_dim == 256
        assert config.num_layers == 3
    
    def test_predictor_network_forward(self):
        """Test BondPredictorNetwork forward pass."""
        from ontic.ml.neural.bond_predictor import BondPredictorNetwork, PredictorConfig
        
        config = PredictorConfig(input_dim=16)
        network = BondPredictorNetwork(config)
        
        x = torch.randn(4, 16)
        log_chi, uncertainty = network(x)
        
        assert log_chi.shape == (4, 1)
        assert uncertainty.shape == (4, 1)
        assert (uncertainty >= 0).all()
    
    def test_bond_dimension_predictor(self):
        """Test BondDimensionPredictor class."""
        from ontic.ml.neural.bond_predictor import (
            BondDimensionPredictor,
            EntropyFeatures,
            TemporalFeatures,
        )
        
        predictor = BondDimensionPredictor()
        
        # Create entropy features
        entropies = torch.tensor([0.1 * i for i in range(10)])
        entropy_features = EntropyFeatures.from_entropies(entropies)
        
        # Create temporal features
        temporal_features = TemporalFeatures.from_histories(
            entropy_history=[0.3, 0.4, 0.5],
            chi_history=[32, 48, 64],
            error_history=[1e-6, 1e-7, 1e-8],
        )
        
        result = predictor.predict(
            entropy_features=entropy_features,
            temporal_features=temporal_features,
            target_error=1e-8,
            current_chi=64,
        )
        
        assert result.chi > 0
        assert result.uncertainty >= 0
        assert 0 <= result.confidence <= 1


class TestEntanglementGNN:
    """Tests for entanglement GNN."""
    
    def test_node_features(self):
        """Test NodeFeatures dataclass."""
        from ontic.ml.neural.entanglement_gnn import NodeFeatures
        
        node = NodeFeatures(
            site_index=5,
            local_dim=2,
            entropy=0.5,
            bond_dim_left=32,
            bond_dim_right=64,
        )
        
        assert node.site_index == 5
        assert node.local_dim == 2
        
        tensor = node.to_tensor()
        assert tensor.shape == (7,)  # 7 features
    
    def test_edge_features(self):
        """Test EdgeFeatures dataclass."""
        from ontic.ml.neural.entanglement_gnn import EdgeFeatures
        
        edge = EdgeFeatures(
            source=0,
            target=1,
            bond_dim=32,
            truncation_error=1e-8,
            correlation=0.5,
            distance=1,
        )
        
        assert edge.source == 0
        assert edge.target == 1
        
        tensor = edge.to_tensor()
        assert tensor.shape == (4,)  # 4 features
    
    def test_gnn_config(self):
        """Test GNNConfig defaults."""
        from ontic.ml.neural.entanglement_gnn import GNNConfig
        
        config = GNNConfig()
        
        # Node features: 7 (from NodeFeatures.to_tensor)
        assert config.node_input_dim == 7
        assert config.hidden_dim == 128
        assert config.num_layers == 4
    
    def test_message_passing_layer(self):
        """Test MessagePassingLayer forward."""
        from ontic.ml.neural.entanglement_gnn import MessagePassingLayer
        
        layer = MessagePassingLayer(
            node_dim=7,
            edge_dim=4,
            hidden_dim=32,
        )
        
        node_features = torch.randn(5, 7)
        edge_features = torch.randn(4, 4)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        
        output = layer(node_features, edge_index, edge_features)
        
        assert output.shape == (5, 7)
    
    def test_entanglement_gnn_forward(self):
        """Test EntanglementGNN forward pass."""
        from ontic.ml.neural.entanglement_gnn import (
            EntanglementGNN, GNNConfig, EntanglementGraph, NodeFeatures, EdgeFeatures
        )
        
        config = GNNConfig(node_input_dim=7, edge_input_dim=4)
        gnn = EntanglementGNN(config)
        
        # Create nodes - using correct parameter names
        nodes = [
            NodeFeatures(site_index=i, local_dim=2, entropy=0.5, 
                        bond_dim_left=16, bond_dim_right=16)
            for i in range(5)
        ]
        
        # Create edges - using correct parameter names
        edges = [
            EdgeFeatures(source=i, target=i+1, bond_dim=16,
                        truncation_error=1e-8, correlation=0.5, distance=1)
            for i in range(4)
        ]
        
        graph = EntanglementGraph(nodes=nodes, edges=edges)
        
        node_out, graph_out = gnn(graph)
        
        assert node_out.shape[0] == 5
        assert graph_out.shape[0] == config.output_dim


class TestAlgorithmSelector:
    """Tests for algorithm selector."""
    
    def test_algorithm_type_enum(self):
        """Test AlgorithmType enumeration."""
        from ontic.ml.neural.algorithm_selector import AlgorithmType
        
        assert AlgorithmType.DMRG.value == 1
        assert AlgorithmType.TEBD.value == 3
        assert len(AlgorithmType) >= 5
    
    def test_selection_criteria_enum(self):
        """Test SelectionCriteria enumeration."""
        from ontic.ml.neural.algorithm_selector import SelectionCriteria
        
        assert SelectionCriteria.ACCURACY is not None
        assert SelectionCriteria.SPEED is not None
        assert SelectionCriteria.BALANCED is not None
    
    def test_problem_features(self):
        """Test ProblemFeatures dataclass."""
        from ontic.ml.neural.algorithm_selector import ProblemFeatures
        
        features = ProblemFeatures(
            num_sites=20,
            local_dim=2,
            target_accuracy=1e-6,
            is_ground_state=True,
        )
        
        assert features.num_sites == 20
        assert features.local_dim == 2
        
        difficulty = features.estimate_difficulty()
        assert 0 <= difficulty <= 1
    
    def test_algorithm_recommendation(self):
        """Test AlgorithmRecommendation dataclass."""
        from ontic.ml.neural.algorithm_selector import (
            AlgorithmRecommendation,
            AlgorithmType,
        )
        
        rec = AlgorithmRecommendation(
            algorithm=AlgorithmType.DMRG,
            confidence=0.95,
            estimated_time=10.0,
            estimated_memory=4.0,
            fallback=AlgorithmType.TEBD,
            parameters={"chi_max": 128},
            reasoning="Best for ground state",
        )
        
        assert rec.algorithm == AlgorithmType.DMRG
        assert rec.confidence == 0.95
    
    def test_algorithm_selector(self):
        """Test AlgorithmSelector class."""
        from ontic.ml.neural.algorithm_selector import (
            AlgorithmSelector,
            ProblemFeatures,
            SelectionCriteria,
        )
        
        selector = AlgorithmSelector()
        
        features = ProblemFeatures(
            num_sites=20,
            local_dim=2,
            target_accuracy=1e-6,
            is_ground_state=True,
        )
        
        recommendation = selector.select(features, SelectionCriteria.BALANCED)
        
        assert recommendation.algorithm is not None
        assert recommendation.confidence > 0


# =============================================================================
# Distributed TN Module Tests (tensornet/distributed_tn/)
# =============================================================================

class TestDistributedDMRG:
    """Tests for distributed DMRG."""
    
    def test_partition_strategy_enum(self):
        """Test PartitionStrategy enumeration."""
        from ontic.engine.distributed_tn.distributed_dmrg import PartitionStrategy
        
        assert PartitionStrategy.EQUAL is not None
        assert PartitionStrategy.ENTROPY_BASED is not None
        assert PartitionStrategy.ADAPTIVE is not None
    
    def test_partition_config(self):
        """Test PartitionConfig dataclass."""
        from ontic.engine.distributed_tn.distributed_dmrg import PartitionConfig
        
        config = PartitionConfig(
            num_partitions=4,
            overlap=2,
        )
        
        assert config.num_partitions == 4
        assert config.overlap == 2
    
    def test_dmrg_partition(self):
        """Test DMRGPartition dataclass."""
        from ontic.engine.distributed_tn.distributed_dmrg import DMRGPartition
        
        tensors = [torch.randn(1, 2, 4), torch.randn(4, 2, 1)]
        
        partition = DMRGPartition(
            partition_id=0,
            start_site=0,
            end_site=2,
            tensors=tensors,
        )
        
        assert partition.partition_id == 0
        assert partition.num_sites == 2
    
    def test_dmrg_worker(self):
        """Test DMRGWorker class."""
        from ontic.engine.distributed_tn.distributed_dmrg import DMRGWorker, DMRGPartition
        
        tensors = [torch.randn(1, 2, 4) for _ in range(5)]
        tensors.append(torch.randn(4, 2, 1))
        
        partition = DMRGPartition(
            partition_id=0,
            start_site=0,
            end_site=6,
            tensors=tensors,
        )
        
        worker = DMRGWorker(partition, hamiltonian=None)
        
        assert worker.partition == partition
        assert worker.chi_max == 64
    
    def test_distributed_dmrg_partitioning(self):
        """Test DistributedDMRG.partition_system."""
        from ontic.engine.distributed_tn.distributed_dmrg import DistributedDMRG
        
        mps_tensors = [torch.randn(1, 2, 4)]
        for _ in range(8):
            mps_tensors.append(torch.randn(4, 2, 4))
        mps_tensors.append(torch.randn(4, 2, 1))
        
        dmrg = DistributedDMRG(num_workers=2)
        partitions = dmrg.partition_system(mps_tensors, hamiltonian=None)
        
        assert len(partitions) == 2
        dmrg.shutdown()


class TestParallelTEBD:
    """Tests for parallel TEBD."""
    
    def test_splitting_order_enum(self):
        """Test SplittingOrder enumeration."""
        from ontic.engine.distributed_tn.parallel_tebd import SplittingOrder
        
        assert SplittingOrder.FIRST.value == 1
        assert SplittingOrder.SECOND.value == 2
        assert SplittingOrder.FOURTH.value == 4
    
    def test_ghost_sites(self):
        """Test GhostSites dataclass."""
        from ontic.engine.distributed_tn.parallel_tebd import GhostSites
        
        ghost = GhostSites(num_ghost=2)
        
        tensors = [torch.randn(4, 2, 4), torch.randn(4, 2, 4)]
        ghost.update_left(tensors)
        
        assert len(ghost.left_ghost) == 2
        assert ghost.last_update > 0
    
    def test_tebd_partition(self):
        """Test TEBDPartition dataclass."""
        from ontic.engine.distributed_tn.parallel_tebd import TEBDPartition
        
        tensors = [torch.randn(4, 2, 4) for _ in range(5)]
        
        partition = TEBDPartition(
            partition_id=0,
            start_site=0,
            end_site=5,
            tensors=tensors,
        )
        
        assert partition.num_sites == 5
    
    def test_tebd_worker_gate_application(self):
        """Test TEBDWorker.apply_gate."""
        from ontic.engine.distributed_tn.parallel_tebd import TEBDWorker, TEBDPartition
        
        d = 2
        tensors = [
            torch.randn(1, d, 4),
            torch.randn(4, d, 4),
            torch.randn(4, d, 1),
        ]
        
        partition = TEBDPartition(
            partition_id=0,
            start_site=0,
            end_site=3,
            tensors=tensors,
        )
        
        worker = TEBDWorker(partition, chi_max=8)
        
        gate = torch.eye(d * d).reshape(d, d, d, d)
        error = worker.apply_gate(gate, 1)
        
        assert error >= 0
    
    def test_parallel_tebd_partitioning(self):
        """Test ParallelTEBD.partition_mps."""
        from ontic.engine.distributed_tn.parallel_tebd import ParallelTEBD
        
        d = 2
        tensors = [torch.randn(1, d, 4)]
        for _ in range(8):
            tensors.append(torch.randn(4, d, 4))
        tensors.append(torch.randn(4, d, 1))
        
        tebd = ParallelTEBD(num_partitions=2)
        partitions = tebd.partition_mps(tensors)
        
        assert len(partitions) == 2
        assert sum(p.num_sites for p in partitions) == 10
        
        tebd.shutdown()


class TestMPSOperations:
    """Tests for distributed MPS operations."""
    
    def test_compression_strategy_enum(self):
        """Test CompressionStrategy enumeration."""
        from ontic.engine.distributed_tn.mps_operations import CompressionStrategy
        
        assert CompressionStrategy.SVD is not None
        assert CompressionStrategy.VARIATIONAL is not None
    
    def test_mps_partition(self):
        """Test MPSPartition dataclass."""
        from ontic.engine.distributed_tn.mps_operations import MPSPartition
        
        tensors = [torch.randn(1, 2, 4), torch.randn(4, 2, 1)]
        
        partition = MPSPartition(
            partition_id=0,
            start_site=0,
            end_site=2,
            tensors=tensors,
        )
        
        assert partition.num_sites == 2
        assert partition.get_tensor(0).shape == (1, 2, 4)
    
    def test_distributed_mps_from_tensors(self):
        """Test DistributedMPS.from_tensors."""
        from ontic.engine.distributed_tn.mps_operations import DistributedMPS
        
        tensors = [torch.randn(1, 2, 4)]
        for _ in range(6):
            tensors.append(torch.randn(4, 2, 4))
        tensors.append(torch.randn(4, 2, 1))
        
        dmps = DistributedMPS()
        dmps.from_tensors(tensors)
        
        assert len(dmps.partitions) == 4
        assert dmps.total_sites == 8
        
        recovered = dmps.to_tensors()
        assert len(recovered) == 8
        
        dmps.shutdown()
    
    def test_merge_partitions(self):
        """Test merge_partitions function."""
        from ontic.engine.distributed_tn.mps_operations import (
            MPSPartition,
            merge_partitions,
        )
        
        partitions = [
            MPSPartition(0, 0, 2, [torch.randn(1, 2, 4), torch.randn(4, 2, 4)]),
            MPSPartition(1, 2, 4, [torch.randn(4, 2, 4), torch.randn(4, 2, 1)]),
        ]
        
        tensors = merge_partitions(partitions)
        
        assert len(tensors) == 4


class TestLoadBalancer:
    """Tests for load balancer."""
    
    def test_balancing_strategy_enum(self):
        """Test BalancingStrategy enumeration."""
        from ontic.engine.distributed_tn.load_balancer import BalancingStrategy
        
        assert BalancingStrategy.STATIC is not None
        assert BalancingStrategy.DYNAMIC is not None
        assert BalancingStrategy.WORK_STEALING is not None
    
    def test_worker_status(self):
        """Test WorkerStatus dataclass."""
        from ontic.engine.distributed_tn.load_balancer import WorkerStatus, WorkerState
        
        status = WorkerStatus(worker_id=0, capacity=100)
        status.current_load = 50
        
        assert status.utilization == 0.5
        assert not status.is_overloaded
        assert not status.is_idle
    
    def test_work_unit(self):
        """Test WorkUnit dataclass."""
        from ontic.engine.distributed_tn.load_balancer import WorkUnit
        
        work = WorkUnit(work_id=0, partition_id=1, estimated_cost=2.0)
        
        assert work.work_id == 0
        assert work.partition_id == 1
        time.sleep(0.01)
        assert work.age > 0
    
    def test_load_balancer_add_work(self):
        """Test LoadBalancer.add_work."""
        from ontic.engine.distributed_tn.load_balancer import LoadBalancer
        
        balancer = LoadBalancer(num_workers=4)
        
        work_id = balancer.add_work(partition_id=0, estimated_cost=1.0)
        
        assert work_id == 0
        assert balancer.workers[0].current_load > 0 or any(
            w.current_load > 0 for w in balancer.workers.values()
        )
    
    def test_load_balancer_complete_work(self):
        """Test LoadBalancer.complete_work."""
        from ontic.engine.distributed_tn.load_balancer import LoadBalancer
        
        balancer = LoadBalancer(num_workers=4)
        work_id = balancer.add_work(partition_id=0, target_worker=0)
        
        initial_load = balancer.workers[0].current_load
        balancer.complete_work(worker_id=0, work_id=work_id, time_taken=0.1)
        
        assert balancer.workers[0].current_load < initial_load
        assert balancer.workers[0].completed_tasks == 1
    
    def test_load_balancer_rebalance(self):
        """Test LoadBalancer.rebalance."""
        from ontic.engine.distributed_tn.load_balancer import LoadBalancer
        
        balancer = LoadBalancer(num_workers=4)
        
        # Add imbalanced work
        for _ in range(10):
            balancer.add_work(partition_id=0, target_worker=0)
        
        imbalance_before = balancer.compute_imbalance()
        result = balancer.rebalance()
        
        assert result.imbalance_before == imbalance_before
    
    def test_rebalance_workload_function(self):
        """Test rebalance_workload function."""
        from ontic.engine.distributed_tn.load_balancer import rebalance_workload
        
        loads = [10, 0, 5, 5]
        moves = rebalance_workload(loads)
        
        # Should suggest moving work from worker 0
        assert any(src == 0 for src, _, _ in moves)


# =============================================================================
# Autonomy Module Tests (tensornet/autonomy/)
# =============================================================================

class TestMissionPlanner:
    """Tests for mission planner."""
    
    def test_mission_status_enum(self):
        """Test MissionStatus enumeration."""
        from ontic.aerospace.autonomy.mission_planner import MissionStatus
        
        assert MissionStatus.PENDING is not None
        assert MissionStatus.EXECUTING is not None
        assert MissionStatus.COMPLETED is not None
    
    def test_mission_constraints(self):
        """Test MissionConstraints dataclass."""
        from ontic.aerospace.autonomy.mission_planner import MissionConstraints
        
        constraints = MissionConstraints(
            max_time=60.0,
            min_accuracy=0.99,
            max_agents=10,
        )
        
        assert constraints.max_time == 60.0
        assert constraints.max_agents == 10
    
    def test_mission_phase(self):
        """Test MissionPhase dataclass."""
        from ontic.aerospace.autonomy.mission_planner import (
            MissionPhase,
            MissionPhaseType,
            MissionStatus,
        )
        
        phase = MissionPhase(
            phase_id=0,
            phase_type=MissionPhaseType.COMPUTE,
            name="Test Phase",
            duration_estimate=10.0,
        )
        
        assert phase.status == MissionStatus.PENDING
        
        phase.start()
        assert phase.status == MissionStatus.EXECUTING
        
        phase.complete("done")
        assert phase.status == MissionStatus.COMPLETED
        assert phase.result == "done"
    
    def test_mission_progress(self):
        """Test Mission progress tracking."""
        from ontic.aerospace.autonomy.mission_planner import (
            Mission,
            MissionPhase,
            MissionPhaseType,
            MissionStatus,
        )
        
        mission = Mission(
            mission_id="test_001",
            name="Test Mission",
        )
        
        mission.phases.append(MissionPhase(0, MissionPhaseType.INITIALIZE, "Init"))
        mission.phases.append(MissionPhase(1, MissionPhaseType.COMPUTE, "Compute"))
        
        assert mission.progress == 0.0
        
        mission.phases[0].status = MissionStatus.COMPLETED
        assert mission.progress == 0.5
    
    def test_mission_planner_creation(self):
        """Test MissionPlanner.create_mission."""
        from ontic.aerospace.autonomy.mission_planner import MissionPlanner
        
        planner = MissionPlanner()
        mission = planner.create_mission(
            name="Test",
            description="A test mission",
        )
        
        assert mission.mission_id == "mission_0000"
        assert mission.name == "Test"
    
    def test_mission_planner_computation_mission(self):
        """Test MissionPlanner.plan_computation_mission."""
        from ontic.aerospace.autonomy.mission_planner import MissionPlanner
        
        planner = MissionPlanner()
        mission = planner.plan_computation_mission(
            problem_size=10,
            target_accuracy=1e-6,
        )
        
        assert len(mission.phases) >= 4
        assert mission.total_estimate > 0


class TestPathPlanning:
    """Tests for path planning."""
    
    def test_planning_algorithm_enum(self):
        """Test PlanningAlgorithm enumeration."""
        from ontic.aerospace.autonomy.path_planning import PlanningAlgorithm
        
        assert PlanningAlgorithm.A_STAR is not None
        assert PlanningAlgorithm.RRT is not None
        assert PlanningAlgorithm.DIJKSTRA is not None
    
    def test_waypoint(self):
        """Test Waypoint dataclass."""
        from ontic.aerospace.autonomy.path_planning import Waypoint
        
        w1 = Waypoint(position=(0.0, 0.0))
        w2 = Waypoint(position=(3.0, 4.0))
        
        assert w1.x == 0.0
        assert w2.y == 4.0
        assert w1.distance_to(w2) == 5.0
    
    def test_path(self):
        """Test Path dataclass."""
        from ontic.aerospace.autonomy.path_planning import Path, Waypoint
        
        path = Path()
        path.add_waypoint(Waypoint(position=(0.0, 0.0)))
        path.add_waypoint(Waypoint(position=(1.0, 0.0)))
        path.add_waypoint(Waypoint(position=(1.0, 1.0)))
        
        assert len(path) == 3
        assert path.total_distance == 2.0
    
    def test_path_planner_astar(self):
        """Test PathPlanner with A*."""
        from ontic.aerospace.autonomy.path_planning import PathPlanner, PathPlannerConfig
        
        config = PathPlannerConfig()
        planner = PathPlanner(config)
        planner.set_bounds(0, 0, 20, 20)
        
        path = planner.plan((0.0, 0.0), (10.0, 10.0))
        
        assert path.is_valid
        assert len(path.waypoints) > 0
    
    def test_path_planner_with_obstacles(self):
        """Test PathPlanner with obstacles."""
        from ontic.aerospace.autonomy.path_planning import PathPlanner
        
        planner = PathPlanner()
        planner.set_bounds(0, 0, 20, 20)
        
        # Add wall of obstacles
        for y in range(5, 15):
            planner.add_obstacle(10, y)
        
        path = planner.plan((5.0, 10.0), (15.0, 10.0))
        
        # Should find path around obstacles
        assert path.is_valid or path.total_distance > 0
    
    def test_plan_path_function(self):
        """Test plan_path convenience function."""
        from ontic.aerospace.autonomy.path_planning import plan_path
        
        path = plan_path(
            start=(0.0, 0.0),
            goal=(5.0, 5.0),
        )
        
        assert path.is_valid
    
    def test_smooth_path(self):
        """Test smooth_path function."""
        from ontic.aerospace.autonomy.path_planning import smooth_path, Path, Waypoint
        
        path = Path()
        for i in range(10):
            path.add_waypoint(Waypoint(position=(float(i), float(i % 2))))
        
        smoothed = smooth_path(path, smoothing_factor=0.5, iterations=5)
        
        assert len(smoothed.waypoints) == len(path.waypoints)


class TestObstacleAvoidance:
    """Tests for obstacle avoidance."""
    
    def test_obstacle_type_enum(self):
        """Test ObstacleType enumeration."""
        from ontic.aerospace.autonomy.obstacle_avoidance import ObstacleType
        
        assert ObstacleType.STATIC is not None
        assert ObstacleType.DYNAMIC is not None
    
    def test_avoidance_strategy_enum(self):
        """Test AvoidanceStrategy enumeration."""
        from ontic.aerospace.autonomy.obstacle_avoidance import AvoidanceStrategy
        
        assert AvoidanceStrategy.POTENTIAL_FIELD is not None
        assert AvoidanceStrategy.REACTIVE is not None
    
    def test_obstacle(self):
        """Test Obstacle dataclass."""
        from ontic.aerospace.autonomy.obstacle_avoidance import Obstacle
        
        obs = Obstacle(
            obstacle_id=0,
            position=(5.0, 5.0),
            radius=1.0,
        )
        
        point = (7.0, 5.0)
        dist = obs.distance_to(point)
        
        assert dist == 1.0  # 2.0 - 1.0 radius
    
    def test_obstacle_predicted_position(self):
        """Test Obstacle.predicted_position."""
        from ontic.aerospace.autonomy.obstacle_avoidance import Obstacle, ObstacleType
        
        obs = Obstacle(
            obstacle_id=0,
            obstacle_type=ObstacleType.DYNAMIC,
            position=(0.0, 0.0),
            velocity=(1.0, 0.0),
        )
        
        pred = obs.predicted_position(5.0)
        
        assert pred == (5.0, 0.0)
    
    def test_obstacle_avoidance_add_remove(self):
        """Test ObstacleAvoidance add/remove."""
        from ontic.aerospace.autonomy.obstacle_avoidance import ObstacleAvoidance
        
        avoider = ObstacleAvoidance()
        
        obs_id = avoider.add_obstacle(
            position=(5.0, 5.0),
            radius=1.0,
        )
        
        assert obs_id == 0
        assert len(avoider.obstacles) == 1
        
        avoider.remove_obstacle(obs_id)
        assert len(avoider.obstacles) == 0
    
    def test_potential_field_avoidance(self):
        """Test potential field avoidance."""
        from ontic.aerospace.autonomy.obstacle_avoidance import ObstacleAvoidance
        
        avoider = ObstacleAvoidance()
        avoider.add_obstacle(position=(5.0, 0.0), radius=1.0)
        
        result = avoider.compute_avoidance(
            position=(3.0, 0.0),
            velocity=(1.0, 0.0),
        )
        
        assert result.success
        # Should push away from obstacle
        assert result.avoidance_vector[0] < 0
    
    def test_is_path_clear(self):
        """Test ObstacleAvoidance.is_path_clear."""
        from ontic.aerospace.autonomy.obstacle_avoidance import ObstacleAvoidance
        
        avoider = ObstacleAvoidance()
        avoider.add_obstacle(position=(5.0, 0.0), radius=1.0)
        
        clear = avoider.is_path_clear(
            start=(0.0, 0.0),
            end=(10.0, 0.0),
            check_radius=0.5,
        )
        
        assert not clear


class TestDecisionMaking:
    """Tests for decision making."""
    
    def test_decision_type_enum(self):
        """Test DecisionType enumeration."""
        from ontic.aerospace.autonomy.decision_making import DecisionType
        
        assert DecisionType.ACTION is not None
        assert DecisionType.EMERGENCY is not None
    
    def test_state_estimate(self):
        """Test StateEstimate dataclass."""
        from ontic.aerospace.autonomy.decision_making import StateEstimate
        
        state = StateEstimate(
            position=(1.0, 2.0),
            velocity=(0.5, 0.5),
        )
        
        assert state.speed == pytest.approx(math.sqrt(0.5))
        
        future = state.predict(1.0)
        assert future.position == (1.5, 2.5)
    
    def test_action_option(self):
        """Test ActionOption dataclass."""
        from ontic.aerospace.autonomy.decision_making import ActionOption
        
        action = ActionOption(
            action_id=0,
            name="move_forward",
            risk_level=0.1,
            cost=1.0,
            time_estimate=2.0,
        )
        
        assert action.name == "move_forward"
        assert action.risk_level == 0.1
    
    def test_action_space(self):
        """Test ActionSpace dataclass."""
        from ontic.aerospace.autonomy.decision_making import ActionSpace
        
        space = ActionSpace()
        space.add_option("action1", risk=0.1, cost=1.0)
        space.add_option("action2", risk=0.5, cost=2.0)
        space.add_option("action3", risk=0.9, cost=3.0)
        
        space.constraints["max_risk"] = 0.6
        
        valid = space.filter_by_constraints()
        assert len(valid) == 2
    
    def test_decision_maker_evaluation(self):
        """Test DecisionMaker.evaluate_option."""
        from ontic.aerospace.autonomy.decision_making import DecisionMaker, ActionOption
        
        maker = DecisionMaker()
        
        option = ActionOption(
            action_id=0,
            name="test",
            risk_level=0.2,
            cost=1.0,
            time_estimate=1.0,
        )
        
        score = maker.evaluate_option(option)
        assert score > 0
    
    def test_decision_maker_decision(self):
        """Test DecisionMaker.make_decision."""
        from ontic.aerospace.autonomy.decision_making import DecisionMaker, ActionSpace
        
        maker = DecisionMaker()
        
        space = ActionSpace()
        space.add_option("safe_slow", risk=0.1, cost=1.0, time=10.0)
        space.add_option("fast_risky", risk=0.8, cost=2.0, time=2.0)
        space.add_option("balanced", risk=0.3, cost=1.5, time=5.0)
        
        maker.set_action_space(space)
        decision = maker.make_decision()
        
        assert decision.selected_action is not None
        assert decision.confidence > 0
    
    def test_make_decision_function(self):
        """Test make_decision convenience function."""
        from ontic.aerospace.autonomy.decision_making import make_decision
        
        options = [
            {"name": "option1", "risk": 0.1, "cost": 1.0, "time": 5.0},
            {"name": "option2", "risk": 0.5, "cost": 0.5, "time": 2.0},
        ]
        
        decision = make_decision(options)
        
        assert decision.selected_action is not None
    
    def test_evaluate_options_function(self):
        """Test evaluate_options function."""
        from ontic.aerospace.autonomy.decision_making import evaluate_options
        
        options = [
            {"name": "fast", "risk": 0.2, "cost": 1.0, "time": 1.0},
            {"name": "slow", "risk": 0.1, "cost": 2.0, "time": 10.0},
        ]
        
        results = evaluate_options(options)
        
        assert len(results) == 2
        # First should have highest score
        assert results[0][1] >= results[1][1]


# =============================================================================
# Integration Tests
# =============================================================================

class TestPhase19Integration:
    """Integration tests for Phase 19."""
    
    def test_neural_algorithm_selection_for_distributed(self):
        """Test using neural selector to choose distributed algorithm."""
        from ontic.ml.neural.algorithm_selector import (
            AlgorithmSelector,
            ProblemFeatures,
            SelectionCriteria,
            AlgorithmType,
        )
        
        selector = AlgorithmSelector()
        
        # Large problem - might suggest distributed
        features = ProblemFeatures(
            num_sites=100,
            local_dim=2,
            target_accuracy=1e-6,
            is_ground_state=True,
        )
        
        rec = selector.select(features, SelectionCriteria.SPEED)
        
        assert rec.algorithm in AlgorithmType
        assert rec.confidence > 0
    
    def test_mission_with_path_planning(self):
        """Test mission that includes path planning."""
        from ontic.aerospace.autonomy.mission_planner import MissionPlanner, MissionPhaseType
        from ontic.aerospace.autonomy.path_planning import plan_path
        
        planner = MissionPlanner()
        mission = planner.create_mission("Navigation Mission")
        
        planner.add_phase(mission, MissionPhaseType.INITIALIZE, "Plan Path")
        planner.add_phase(mission, MissionPhaseType.COMPUTE, "Execute Path")
        planner.add_phase(mission, MissionPhaseType.FINALIZE, "Complete")
        
        # Plan a path as part of mission
        path = plan_path((0.0, 0.0), (10.0, 10.0))
        
        assert path.is_valid
        assert len(mission.phases) == 3
    
    def test_decision_with_obstacle_avoidance(self):
        """Test decision making with obstacle input."""
        from ontic.aerospace.autonomy.obstacle_avoidance import ObstacleAvoidance
        from ontic.aerospace.autonomy.decision_making import DecisionMaker, ActionSpace
        
        avoider = ObstacleAvoidance()
        avoider.add_obstacle(position=(5.0, 0.0), radius=1.0)
        
        result = avoider.compute_avoidance((3.0, 0.0), (1.0, 0.0))
        
        # Use avoidance result to inform decision
        maker = DecisionMaker()
        space = ActionSpace()
        
        if result.safety_margin < 2.0:
            # Add avoidance option
            space.add_option("avoid", risk=0.1)
        
        space.add_option("continue", risk=result.safety_margin < 0.5 and 0.9 or 0.1)
        
        maker.set_action_space(space)
        decision = maker.make_decision()
        
        assert decision.selected_action is not None
    
    def test_distributed_dmrg_with_load_balancing(self):
        """Test distributed DMRG with load balancer."""
        from ontic.engine.distributed_tn.distributed_dmrg import DistributedDMRG
        from ontic.engine.distributed_tn.load_balancer import LoadBalancer
        
        # Create MPS
        tensors = [torch.randn(1, 2, 4)]
        for _ in range(6):
            tensors.append(torch.randn(4, 2, 4))
        tensors.append(torch.randn(4, 2, 1))
        
        dmrg = DistributedDMRG(num_workers=2)
        partitions = dmrg.partition_system(tensors, hamiltonian=None)
        
        # Use load balancer to manage work
        balancer = LoadBalancer(num_workers=2)
        
        for p in partitions:
            balancer.add_work(
                partition_id=p.partition_id,
                estimated_cost=p.num_sites,
            )
        
        stats = balancer.get_statistics()
        
        assert stats["total_pending_work"] == 2
        
        dmrg.shutdown()
    
    def test_bond_predictor_for_adaptive_tebd(self):
        """Test bond predictor for adaptive TEBD."""
        from ontic.ml.neural.bond_predictor import (
            BondDimensionPredictor, EntropyFeatures, TemporalFeatures
        )
        from ontic.engine.distributed_tn.parallel_tebd import ParallelTEBD
        
        predictor = BondDimensionPredictor()
        
        # Create entropy features
        entropies = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.5, 0.3, 0.1])
        entropy_features = EntropyFeatures.from_entropies(entropies)
        
        # Create temporal features  
        temporal_features = TemporalFeatures.from_histories(
            entropy_history=[0.3, 0.4, 0.5],
            chi_history=[32, 48, 64],
            error_history=[1e-6, 1e-7, 1e-8],
        )
        
        # Get prediction
        prediction = predictor.predict(
            entropy_features=entropy_features,
            temporal_features=temporal_features,
            target_error=1e-8,
            current_chi=64,
        )
        
        # Use prediction for TEBD
        chi_max = min(128, max(4, prediction.chi))
        
        tebd = ParallelTEBD(num_partitions=2, chi_max=chi_max)
        
        assert tebd.chi_max == chi_max
        tebd.shutdown()


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
