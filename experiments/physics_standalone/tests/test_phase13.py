"""
Integration tests for Phase 13 modules.

Tests Digital Twin, ML Surrogates, and Distributed Computing.
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any


# ==============================================================================
# Digital Twin Tests
# ==============================================================================

class TestDigitalTwin:
    """Tests for digital twin framework."""
    
    def test_state_sync_imports(self):
        """Test state synchronization imports."""
        from tensornet.infra.digital_twin import (
            StateSync,
            StateSynchronizer,
            SyncConfig,
            StateVector,
        )
        
        # Test that imports work
        assert SyncConfig is not None
        assert StateSync is not None
    
    def test_reduced_order_models(self):
        """Test reduced-order models."""
        from tensornet.infra.digital_twin import (
            PODModel,
            DMDModel,
            AutoencoderROM,
        )
        from tensornet.infra.digital_twin.reduced_order import ROMConfig
        
        # Create snapshot data
        n_snapshots = 50
        n_dofs = 100
        snapshots = torch.randn(n_snapshots, n_dofs)
        
        # Test POD - use train_from_snapshots
        config = ROMConfig(n_modes=10)
        pod = PODModel(config)
        pod.train_from_snapshots(snapshots)
        
        # Test basic functionality
        assert pod.basis is not None
    
    def test_health_monitoring(self):
        """Test health monitoring systems."""
        from tensornet.infra.digital_twin import (
            HealthMonitor,
            StructuralHealth,
            ThermalHealth,
            AnomalyDetector,
        )
        from tensornet.infra.digital_twin.health_monitor import HealthConfig
        
        # Test health monitor creation
        config = HealthConfig()
        monitor = HealthMonitor(config)
        
        # Test that object is created
        assert monitor is not None
    
    def test_predictive_maintenance(self):
        """Test predictive maintenance."""
        from tensornet.infra.digital_twin import (
            RULEstimator,
            MaintenanceScheduler,
            PredictiveMaintenance,
        )
        from tensornet.infra.digital_twin.predictive import MaintenanceConfig
        
        # Test RUL estimator creation
        config = MaintenanceConfig()
        rul = RULEstimator(config)
        
        # Test that object is created
        assert rul is not None
    
    def test_digital_twin_creation(self):
        """Test digital twin orchestration."""
        from tensornet.infra.digital_twin import (
            DigitalTwin,
            TwinMode,
            create_vehicle_twin,
        )
        
        # Test that factory function works
        twin = create_vehicle_twin(vehicle_id='test_vehicle')
        
        assert twin is not None


# ==============================================================================
# ML Surrogates Tests
# ==============================================================================

class TestMLSurrogates:
    """Tests for ML surrogate models."""
    
    def test_base_surrogates(self):
        """Test base surrogate models."""
        from tensornet.ml.ml_surrogates import (
            SurrogateConfig,
            MLPSurrogate,
            ResNetSurrogate,
            evaluate_surrogate,
        )
        
        config = SurrogateConfig(
            input_dim=4,
            output_dim=2,
            hidden_dims=[32, 32],
        )
        
        # Test MLP
        mlp = MLPSurrogate(config)
        x = torch.randn(10, 4)
        y = mlp(x)
        assert y.shape == (10, 2)
        
        # Test ResNet
        resnet = ResNetSurrogate(config)
        y_res = resnet(x)
        assert y_res.shape == (10, 2)
    
    def test_physics_informed_nets(self):
        """Test physics-informed neural networks."""
        from tensornet.ml.ml_surrogates import (
            PINNConfig,
            PhysicsInformedNet,
            NavierStokesPINN,
            EulerPINN,
        )
        
        config = PINNConfig(
            input_dim=3,  # x, y, t
            output_dim=4,  # rho, u, v, p
            hidden_dims=[32, 32],
        )
        
        # Test base PINN
        pinn = PhysicsInformedNet(config)
        x = torch.randn(10, 3, requires_grad=True)
        y = pinn(x)
        assert y.shape == (10, 4)
        
        # Test Euler PINN
        euler_pinn = EulerPINN(config)
        y_euler = euler_pinn(x)
        assert y_euler.shape == (10, 4)
    
    def test_deeponet(self):
        """Test Deep Operator Networks."""
        from tensornet.ml.ml_surrogates import (
            DeepONetConfig,
            DeepONet,
        )
        
        config = DeepONetConfig(
            branch_input_dim=50,  # Discretized function
            trunk_input_dim=2,    # Query coordinates
            n_outputs=1,
        )
        
        # Test DeepONet
        deeponet = DeepONet(config)
        
        u = torch.randn(10, 50)  # Input function
        y = torch.randn(10, 2)   # Query coordinates
        
        output = deeponet(u, y)
        assert output.shape[0] == 10  # Batch size preserved
    
    def test_fourier_neural_operator(self):
        """Test Fourier Neural Operators."""
        from tensornet.ml.ml_surrogates import (
            FNOConfig,
            FNO2d,
            FNO3d,
        )
        
        # Test 2D FNO with proper config
        config = FNOConfig(
            input_dim=3,
            output_dim=1,
            modes1=8,
            modes2=8,
            width=16,
            n_layers=2,
            in_channels=3,
            out_channels=1,
        )
        
        fno2d = FNO2d(config)
        
        x_2d = torch.randn(2, 3, 32, 32)
        y_2d = fno2d(x_2d)
        # Output should have 1 channel since out_channels=1
        assert y_2d.shape == (2, 1, 32, 32)
    
    def test_uncertainty_quantification(self):
        """Test uncertainty quantification methods."""
        from tensornet.ml.ml_surrogates import (
            UncertaintyConfig,
            EnsembleUQ,
            MCDropoutUQ,
            BayesianUQ,
            compute_prediction_interval,
        )
        
        # Test MC Dropout - use UncertaintyConfig
        from tensornet.ml.ml_surrogates import SurrogateConfig, MLPSurrogate
        
        config = SurrogateConfig(
            input_dim=4,
            output_dim=1,
            hidden_dims=[16, 16],
            dropout=0.2,
        )
        
        model = MLPSurrogate(config)
        
        uq_config = UncertaintyConfig(n_samples=10)
        mc_uq = MCDropoutUQ(uq_config, model)
        
        x = torch.randn(5, 4)
        estimate = mc_uq.predict_with_uncertainty(x)
        
        assert estimate.mean.shape[0] == 5
        assert estimate.std.shape[0] == 5
    
    def test_training_utilities(self):
        """Test training utilities."""
        from tensornet.ml.ml_surrogates import (
            TrainingConfig,
            SurrogateTrainer,
            DataAugmentor,
            ActiveLearner,
            train_surrogate,
        )
        from tensornet.ml.ml_surrogates import SurrogateConfig, MLPSurrogate
        
        # Create model and data
        config = SurrogateConfig(
            input_dim=2,
            output_dim=1,
            hidden_dims=[16, 16],
        )
        model = MLPSurrogate(config)
        
        x = torch.randn(100, 2)
        y = torch.sin(x[:, 0:1]) + torch.cos(x[:, 1:2])
        
        # Test training
        train_config = TrainingConfig(
            n_epochs=10,
            batch_size=32,
            verbose=False,
        )
        
        trainer = SurrogateTrainer(model, train_config)
        history = trainer.train(x, y)
        
        assert 'train_loss' in history
        assert len(history['train_loss']) > 0
        
        # Test data augmentation
        augmentor = DataAugmentor(noise_std=0.01)
        x_aug, y_aug = augmentor.augment(x[:10], y[:10])
        assert x_aug.shape == (10, 2)


# ==============================================================================
# Distributed Computing Tests
# ==============================================================================

class TestDistributedComputing:
    """Tests for distributed computing module."""
    
    def test_domain_decomposition(self):
        """Test domain decomposition."""
        from tensornet.engine.distributed import (
            DomainConfig,
            DomainDecomposition,
            decompose_domain,
            compute_ghost_zones,
        )
        
        # 2D decomposition
        config = DomainConfig(
            nx=64, ny=64, nz=1,
            n_procs=4,
            n_ghost=2,
        )
        
        decomp = decompose_domain(config)
        
        # Check all subdomains created
        assert len(decomp.subdomains) == 4
        
        # Check processor grid
        assert decomp.proc_dims[0] * decomp.proc_dims[1] == 4
        
        # Check subdomain sizes
        for rank in range(4):
            sub = decomp.get_subdomain(rank)
            assert sub.local_nx > 0
            assert sub.local_ny > 0
    
    def test_gpu_manager(self):
        """Test GPU resource management."""
        from tensornet.engine.distributed import (
            GPUConfig,
            GPUManager,
            get_available_gpus,
            distribute_workload,
        )
        
        # Test workload distribution
        workloads = distribute_workload(1000, 4, 'equal')
        
        assert len(workloads) == 4
        total = sum(end - start for start, end in workloads.values())
        assert total == 1000
        
        # Test GPU manager (works even without GPUs)
        config = GPUConfig()
        manager = GPUManager(config)
        manager.initialize()
        
        device = manager.get_device()
        assert device is not None
    
    def test_communication(self):
        """Test communication patterns."""
        from tensornet.engine.distributed import (
            Communicator,
            AllReduceOp,
            all_reduce,
            broadcast,
        )
        
        # Create single-process communicator
        comm = Communicator(n_procs=4, rank=0)
        
        # Test all-reduce
        data = torch.tensor([1.0, 2.0, 3.0])
        result = all_reduce(comm, data, AllReduceOp.SUM)
        
        assert result.shape == data.shape
        
        # Test broadcast
        result = broadcast(comm, data, root=0)
        assert torch.allclose(result, data)
    
    def test_task_scheduler(self):
        """Test distributed task scheduling."""
        from tensornet.engine.distributed import (
            TaskConfig,
            TaskGraph,
            DistributedScheduler,
            schedule_dependency_graph,
            execute_parallel,
        )
        
        # Test task graph
        graph = TaskGraph()
        
        t1 = graph.add_task("init", lambda: 1)
        t2 = graph.add_task("process", lambda: 2, deps=[t1])
        t3 = graph.add_task("finalize", lambda: 3, deps=[t2])
        
        assert len(graph.tasks) == 3
        
        # Check topological order
        order = graph.topological_order()
        assert order.index(t1) < order.index(t2)
        assert order.index(t2) < order.index(t3)
        
        # Test parallel execution
        results = execute_parallel([
            lambda: 1 ** 2,
            lambda: 2 ** 2,
            lambda: 3 ** 2,
        ])
        
        assert results == [1, 4, 9]
    
    def test_parallel_solvers(self):
        """Test parallel iterative solvers."""
        from tensornet.engine.distributed import (
            ParallelConfig,
            ParallelCGSolver,
            parallel_solve,
        )
        
        # Create test problem
        n = 50
        A_mat = torch.eye(n) * 2.0 + torch.randn(n, n) * 0.1
        A_mat = (A_mat + A_mat.t()) / 2  # Symmetrize
        A_mat = A_mat + n * torch.eye(n)  # Make positive definite
        
        x_true = torch.randn(n)
        b = A_mat @ x_true
        
        def matvec(x):
            return A_mat @ x
        
        # Solve
        x_sol = parallel_solve(matvec, b, method='cg')
        
        # Check solution quality
        residual = torch.norm(b - matvec(x_sol)) / torch.norm(b)
        assert residual < 1e-4


# ==============================================================================
# Cross-Module Integration Tests
# ==============================================================================

class TestPhase13Integration:
    """Integration tests across Phase 13 modules."""
    
    def test_surrogate_with_twin(self):
        """Test using ML surrogate within digital twin."""
        from tensornet.ml.ml_surrogates import (
            SurrogateConfig,
            MLPSurrogate,
            train_surrogate,
        )
        from tensornet.infra.digital_twin import (
            DigitalTwin,
            create_vehicle_twin,
        )
        
        # Create surrogate for aerodynamics
        config = SurrogateConfig(
            input_dim=3,  # Mach, alpha, beta
            output_dim=3,  # CL, CD, CM
            hidden_dims=[32, 32],
        )
        
        aero_surrogate = MLPSurrogate(config)
        
        # Train with synthetic data
        x_train = torch.rand(100, 3) * torch.tensor([10.0, 20.0, 10.0])
        y_train = torch.randn(100, 3)  # Random coefficients
        
        train_surrogate(aero_surrogate, x_train, y_train, n_epochs=10, verbose=False)
        
        # Use in twin
        twin = create_vehicle_twin('test')
        
        # Query surrogate
        query = torch.tensor([[5.0, 10.0, 0.0]])
        coeffs = aero_surrogate.predict(query)
        
        assert coeffs.shape == (1, 3)
    
    def test_distributed_surrogate_training(self):
        """Test distributed training of surrogates."""
        from tensornet.ml.ml_surrogates import (
            SurrogateConfig,
            MLPSurrogate,
            TrainingConfig,
            SurrogateTrainer,
        )
        from tensornet.engine.distributed import (
            distribute_workload,
            execute_parallel,
        )
        
        # Create multiple surrogates for ensemble
        configs = [
            SurrogateConfig(input_dim=4, output_dim=1, hidden_dims=[16, 16])
            for _ in range(3)
        ]
        
        models = [MLPSurrogate(c) for c in configs]
        
        # Training data
        x = torch.randn(200, 4)
        y = torch.sin(x.sum(dim=1, keepdim=True))
        
        # Train in parallel using execute_parallel
        def train_model(model):
            config = TrainingConfig(n_epochs=10, verbose=False)
            trainer = SurrogateTrainer(model, config)
            trainer.train(x, y)
            return model.trained
        
        # Execute parallel training
        results = execute_parallel([lambda m=m: train_model(m) for m in models])
        
        assert all(results)
    
    def test_full_phase13_imports(self):
        """Test all Phase 13 exports from main package."""
        from tensornet import (
            # Digital Twin
            StateSync,
            StateSynchronizer,
            SyncConfig,
            StateVector,
            PODModel,
            DMDModel,
            AutoencoderROM,
            HealthMonitor,
            StructuralHealth,
            ThermalHealth,
            AnomalyDetector,
            RULEstimator,
            MaintenanceScheduler,
            PredictiveMaintenance,
            DigitalTwin,
            TwinMode,
            TwinStatus,
            create_vehicle_twin,
            connect_twins,
            validate_twin_fidelity,
            # ML Surrogates
            SurrogateConfig,
            CFDSurrogate,
            MLPSurrogate,
            ResNetSurrogate,
            PINNConfig,
            PhysicsInformedNet,
            NavierStokesPINN,
            EulerPINN,
            DeepONetConfig,
            DeepONet,
            MultiInputDeepONet,
            FNOConfig,
            FNO2d,
            FNO3d,
            TFNO2d,
            UncertaintyConfig,
            EnsembleUQ,
            MCDropoutUQ,
            BayesianUQ,
            TrainingConfig,
            SurrogateTrainer,
            DataAugmentor,
            ActiveLearner,
            evaluate_surrogate,
            train_surrogate,
            cross_validate,
            # Distributed Computing
            DomainConfig,
            DomainDecomposition,
            SubdomainInfo,
            decompose_domain,
            compute_ghost_zones,
            exchange_ghost_data,
            GPUConfig,
            GPUDevice,
            GPUManager,
            MemoryPool,
            get_available_gpus,
            select_optimal_device,
            distribute_workload,
            CommPattern,
            Communicator,
            AllReduceOp,
            async_send,
            async_recv,
            barrier,
            all_reduce,
            broadcast,
            scatter,
            gather,
            TaskConfig,
            Task,
            TaskGraph,
            DistributedScheduler,
            schedule_dependency_graph,
            execute_parallel,
            ParallelConfig,
            DomainSolver,
            ParallelCGSolver,
            ParallelGMRESSolver,
            SchwarzPreconditioner,
            parallel_solve,
        )
        
        # All imports successful
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
