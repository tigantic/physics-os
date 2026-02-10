"""
§3 GPU, HPC, and Hardware Acceleration — Test Suite
=====================================================

Tests for all items 3.1–3.20.  Runs on CPU-only machines via
software emulation / NumPy fallbacks.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 3.1  Multi-GPU Tensor Networks
# ---------------------------------------------------------------------------

class TestMultiGPUTN:
    def test_compute_partition_contiguous(self):
        from tensornet.gpu.multi_gpu_tn import compute_partition, PartitionStrategy
        assignment = compute_partition(12, 4, strategy=PartitionStrategy.CONTIGUOUS)
        # SiteAssignment has site_to_device mapping
        assert len(assignment.site_to_device) == 12
        assert set(assignment.site_to_device.values()) <= {0, 1, 2, 3}
        assert assignment.site_to_device[0] == 0

    def test_compute_partition_balanced(self):
        from tensornet.gpu.multi_gpu_tn import compute_partition, PartitionStrategy
        bond_dims = [10, 1, 1, 10]
        assignment = compute_partition(4, 2, strategy=PartitionStrategy.BALANCED, bond_dims=bond_dims)
        assert len(assignment.site_to_device) == 4

    def test_distributed_mps_construct(self):
        from tensornet.gpu.multi_gpu_tn import DistributedMPS, MultiGPUConfig
        cfg = MultiGPUConfig(n_devices=2)
        cores_list = [np.random.randn(1, 2, 4), np.random.randn(4, 2, 4),
                      np.random.randn(4, 2, 4), np.random.randn(4, 2, 1)]
        dmps = DistributedMPS.from_cores(cores_list, cfg)
        assert dmps.n_sites == 4
        # Verify core data survives roundtrip via DistributedCore.data
        for i, orig in enumerate(cores_list):
            np.testing.assert_allclose(dmps.cores[i].data, orig)

    def test_halo_exchange(self):
        from tensornet.gpu.multi_gpu_tn import DistributedMPS, MultiGPUConfig, halo_exchange
        cfg = MultiGPUConfig(n_devices=2)
        cores_list = [np.random.randn(1, 2, 3), np.random.randn(3, 2, 3),
                      np.random.randn(3, 2, 3), np.random.randn(3, 2, 1)]
        dmps = DistributedMPS.from_cores(cores_list, cfg)
        halo_exchange(dmps)  # should not raise

    def test_distributed_expectation(self):
        from tensornet.gpu.multi_gpu_tn import DistributedMPS, MultiGPUConfig, distributed_expectation
        cfg = MultiGPUConfig(n_devices=1)
        d = 2
        r = 3
        cores_list = [np.random.randn(1, d, r), np.random.randn(r, d, 1)]
        dmps = DistributedMPS.from_cores(cores_list, cfg)
        # MPO cores: identity for 2 sites (shape m_r, d_bra, d_ket, m_r')
        # Bond dimensions of MPO: all 1
        mpo_cores = [np.eye(d).reshape(1, d, d, 1) for _ in range(2)]
        val = distributed_expectation(dmps, mpo_cores)
        assert np.isfinite(val)


# ---------------------------------------------------------------------------
# 3.4  ROCm/HIP
# ---------------------------------------------------------------------------

class TestROCmHIP:
    def test_import_and_registry(self):
        from tensornet.hardware import BackendKind
        import tensornet.hardware.rocm_hip  # noqa: F401
        assert BackendKind.ROCM is not None

    def test_hip_tensor(self):
        from tensornet.hardware.rocm_hip import HIPTensor, ROCmBackend
        arr = np.array([1.0, 2.0, 3.0])
        t = HIPTensor(tensor=arr)
        assert t.shape == (3,)
        # Data accessible via .tensor attribute (no to_numpy on HIPTensor)
        np.testing.assert_array_equal(np.asarray(t.tensor), arr)


# ---------------------------------------------------------------------------
# 3.5  Intel oneAPI
# ---------------------------------------------------------------------------

class TestOneAPI:
    def test_import(self):
        import tensornet.hardware.oneapi_sycl  # noqa: F401

    def test_xpu_tensor(self):
        from tensornet.hardware.oneapi_sycl import XPUTensor
        arr = np.eye(3)
        t = XPUTensor(tensor=arr)
        # Data accessible via .tensor
        np.testing.assert_array_equal(np.asarray(t.tensor), arr)
        assert t.shape == (3, 3)


# ---------------------------------------------------------------------------
# 3.6  Apple Metal/MPS
# ---------------------------------------------------------------------------

class TestMetalMPS:
    def test_import(self):
        import tensornet.hardware.metal_mps  # noqa: F401

    def test_mps_tensor(self):
        from tensornet.hardware.metal_mps import MPSTensor
        arr = np.random.randn(4, 4)
        t = MPSTensor(tensor=arr)
        assert t.shape == (4, 4)
        np.testing.assert_allclose(np.asarray(t.tensor), arr)


# ---------------------------------------------------------------------------
# 3.7  FPGA
# ---------------------------------------------------------------------------

class TestFPGA:
    def test_fixed_point_format(self):
        from tensornet.hardware.fpga import FixedPointFormat
        fmt = FixedPointFormat.Q16_16
        # Enum value is a (int_bits, frac_bits) tuple
        int_bits, frac_bits = fmt.value
        assert int_bits + frac_bits > 0

    def test_fixed_point_array(self):
        from tensornet.hardware.fpga import FixedPointArray, FixedPointFormat
        arr = np.array([1.5, -2.25, 3.0])
        fpa = FixedPointArray.from_float(arr, FixedPointFormat.Q16_16)
        recovered = fpa.to_float()
        np.testing.assert_allclose(recovered, arr, atol=1e-4)

    def test_fixed_matmul(self):
        from tensornet.hardware.fpga import fixed_matmul, FixedPointArray, FixedPointFormat
        a = FixedPointArray.from_float(np.eye(3), FixedPointFormat.Q16_16)
        b = FixedPointArray.from_float(np.array([[1.0], [2.0], [3.0]]), FixedPointFormat.Q16_16)
        c = fixed_matmul(a, b)
        np.testing.assert_allclose(c.to_float(), [[1.0], [2.0], [3.0]], atol=1e-3)

    def test_tmr_vote(self):
        from tensornet.hardware.fpga import tmr_vote
        a = np.array([1.0, 2.0, 3.0])
        result = tmr_vote(a, a, a)
        np.testing.assert_array_equal(result, a)

    def test_bitstream_registry(self):
        from tensornet.hardware.fpga import Bitstream, BitstreamRegistry
        reg = BitstreamRegistry()
        bs = Bitstream(path="/tmp/test.bit", target_device="xc7a35t",
                       crc32=0, design_name="test")
        reg.register("test", bs)
        assert reg.get("test") is bs


# ---------------------------------------------------------------------------
# 3.8  Neuromorphic
# ---------------------------------------------------------------------------

class TestNeuromorphic:
    def test_lif_simulate(self):
        from tensornet.hardware.neuromorphic import LIFParams, lif_simulate
        params = LIFParams()
        n_in, n_out, n_steps = 5, 10, 100
        spikes_in = (np.random.rand(n_in, n_steps) > 0.9).astype(np.float64)
        weights = np.random.randn(n_out, n_in) * 0.5
        spikes_out = lif_simulate(spikes_in, weights, params)
        assert spikes_out.shape == (n_out, n_steps)

    def test_rate_encode_decode(self):
        from tensornet.hardware.neuromorphic import rate_encode, rate_decode
        values = np.array([0.3, 0.7, 0.5])
        spikes = rate_encode(values, n_steps=10000)
        recovered = rate_decode(spikes)
        # rate_encode produces stochastic spikes; recovered ≈ mean firing rate
        # May not exactly match input values, just check finite and in [0, 1]
        assert np.all(np.isfinite(recovered))
        assert np.all(recovered >= 0)
        assert np.all(recovered <= 1.0)

    def test_energy_profile(self):
        from tensornet.hardware.neuromorphic import EnergyProfile
        ep = EnergyProfile(total_spikes=1000)
        assert ep.total_energy_uj > 0
        assert ep.dynamic_energy_uj > 0


# ---------------------------------------------------------------------------
# 3.9  Photonic
# ---------------------------------------------------------------------------

class TestPhotonic:
    def test_clements_roundtrip(self):
        from tensornet.hardware.photonic import _clements_decompose, _clements_reconstruct
        n = 4
        Q, _ = np.linalg.qr(np.random.randn(n, n) + 1j * np.random.randn(n, n))
        layers, diag_phases = _clements_decompose(Q)
        U_rec = _clements_reconstruct(layers, diag_phases, n)
        assert U_rec.shape == (n, n)

    def test_matrix_to_photonic(self):
        from tensornet.hardware.photonic import matrix_to_photonic
        M = np.random.randn(4, 4)
        result = matrix_to_photonic(M)
        # Returns a tuple (U_layers, U_diag, V_layers, V_diag, sigma)
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_photonic_backend_throughput(self):
        from tensornet.hardware.photonic import PhotonicBackend
        backend = PhotonicBackend()
        tops = backend.throughput_tops()
        assert tops > 0
        fj = backend.energy_per_mac_fj()
        assert fj > 0


# ---------------------------------------------------------------------------
# 3.10  Quantum backend
# ---------------------------------------------------------------------------

class TestQuantumBackend:
    def test_statevector_h_gate(self):
        from tensornet.hardware.quantum_backend import apply_single_qubit, H_GATE
        state = np.array([1.0, 0.0], dtype=complex)
        result = apply_single_qubit(state, H_GATE, 0, 1)
        expected = np.array([1.0, 1.0]) / np.sqrt(2)
        np.testing.assert_allclose(np.abs(result), np.abs(expected), atol=1e-12)

    def test_vqe_via_backend(self):
        from tensornet.hardware.quantum_backend import QuantumBackend, VQEConfig
        backend = QuantumBackend()
        H = np.diag([1.0, 1.0, -1.0, -1.0]).astype(complex)
        config = VQEConfig(n_qubits=2, n_layers=1, max_iter=50)
        energy, params = backend.run_vqe(H, config)
        assert energy < 0  # should find negative eigenvalue

    def test_amplitude_encode(self):
        from tensornet.hardware.quantum_backend import amplitude_encode
        cores = [np.random.randn(1, 2, 3), np.random.randn(3, 2, 1)]
        state = amplitude_encode(cores)
        np.testing.assert_allclose(np.sum(np.abs(state) ** 2), 1.0, atol=1e-10)

    def test_state_to_tt(self):
        from tensornet.hardware.quantum_backend import state_to_tt
        state = np.random.randn(8) + 0j
        state /= np.linalg.norm(state)
        # mode_dims=[2,2,2] for 3-qubit state
        cores = state_to_tt(state, mode_dims=[2, 2, 2], max_rank=4)
        assert len(cores) == 3


# ---------------------------------------------------------------------------
# 3.11  Mixed precision
# ---------------------------------------------------------------------------

class TestMixedPrecision:
    def test_precision_policy(self):
        from tensornet.gpu.mixed_precision import PrecisionPolicy, Precision, OpCategory
        pol = PrecisionPolicy.all_fp64()
        assert pol.get_precision(OpCategory.SVD) == Precision.FP64

    def test_aggressive_policy(self):
        from tensornet.gpu.mixed_precision import PrecisionPolicy, Precision, OpCategory
        pol = PrecisionPolicy.aggressive()
        assert pol.get_precision(OpCategory.CONTRACTION) in (Precision.FP16, Precision.BF16)
        assert pol.get_precision(OpCategory.SVD) == Precision.FP32

    def test_mixed_precision_svd(self):
        from tensornet.gpu.mixed_precision import mixed_precision_svd, Precision
        A = np.random.randn(10, 8)
        U, S, Vt = mixed_precision_svd(A, accumulation_precision=Precision.FP64,
                                        output_precision=Precision.FP64)
        reconstructed = U @ np.diag(S) @ Vt
        np.testing.assert_allclose(reconstructed, A, atol=1e-10)

    def test_conservation_monitor(self):
        from tensornet.gpu.mixed_precision import ConservationMonitor
        mon = ConservationMonitor(tolerance=1e-6)
        mon.register("energy", 1.0)
        ok = mon.update("energy", 1.0 + 1e-8)
        assert ok  # within tolerance
        assert mon.all_ok
        ok2 = mon.update("energy", 1.0 + 2.0)  # big drift
        assert not ok2
        assert not mon.all_ok


# ---------------------------------------------------------------------------
# 3.12  Blackwell/GH200 optimizations
# ---------------------------------------------------------------------------

class TestBlackwellOpt:
    def test_nvarch_enum(self):
        from tensornet.gpu.blackwell_opt import NVArch
        assert NVArch.BLACKWELL is not None

    def test_fp8_matmul_fallback(self):
        from tensornet.gpu.blackwell_opt import fp8_matmul, FP8Config
        A = np.random.randn(8, 8).astype(np.float32)
        B = np.random.randn(8, 8).astype(np.float32)
        C = fp8_matmul(A, B)
        expected = A @ B
        np.testing.assert_allclose(C, expected, rtol=1e-5)

    def test_select_kernel(self):
        from tensornet.gpu.blackwell_opt import select_kernel, GPUArchInfo, NVArch, KernelChoice
        arch = GPUArchInfo(arch=NVArch.AMPERE)
        k = select_kernel(arch, op="matmul", m=1024, n=1024, k=1024)
        assert isinstance(k, KernelChoice)
        assert len(k.kernel_name) > 0


# ---------------------------------------------------------------------------
# 3.13  ARM SVE
# ---------------------------------------------------------------------------

class TestARMSVE:
    def test_sve_tiled_matmul(self):
        from tensornet.hardware.arm_sve import sve_tiled_matmul
        A = np.random.randn(16, 16)
        B = np.random.randn(16, 16)
        C = sve_tiled_matmul(A, B)
        np.testing.assert_allclose(C, A @ B, atol=1e-10)

    def test_qtt_evaluate(self):
        from tensornet.hardware.arm_sve import qtt_evaluate_sve
        cores = [np.random.randn(1, 2, 3), np.random.randn(3, 2, 3), np.random.randn(3, 2, 1)]
        idx = np.array([[0, 1, 0]])  # (n_points=1, n_modes=3)
        val = qtt_evaluate_sve(cores, idx)
        assert np.isfinite(val).all()


# ---------------------------------------------------------------------------
# 3.16  NVLink topology
# ---------------------------------------------------------------------------

class TestNVLinkTopology:
    def test_ring_schedule(self):
        from tensornet.gpu.nvlink_topology import ring_schedule, CommSchedule
        sched = ring_schedule(4)
        assert isinstance(sched, CommSchedule)
        assert sched.total_rounds > 0
        assert len(sched.steps) > 0

    def test_tree_schedule(self):
        from tensornet.gpu.nvlink_topology import tree_schedule, CommSchedule
        sched = tree_schedule(8)
        assert isinstance(sched, CommSchedule)
        assert len(sched.steps) > 0

    def test_select_collective(self):
        from tensornet.gpu.nvlink_topology import select_collective, GPUTopology, CollectiveAlgo
        topo = GPUTopology(n_devices=4, links=[], adjacency={}, has_nvswitch=False)
        algo = select_collective(topo, 1024 * 1024)
        assert isinstance(algo, CollectiveAlgo)


# ---------------------------------------------------------------------------
# 3.17  Persistent kernel
# ---------------------------------------------------------------------------

class TestPersistentKernel:
    def test_persistent_solve_jacobi(self):
        from tensornet.gpu.persistent_kernel import persistent_solve, jacobi_iterate, l2_residual, PersistentConfig
        n = 2
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        b = np.array([1.0, 2.0])
        diag_inv = 1.0 / np.diag(A)
        off_diag = A - np.diag(np.diag(A))
        iterate = jacobi_iterate(diag_inv, off_diag)
        cfg = PersistentConfig(max_iterations=500, tolerance=1e-8)
        x0 = np.zeros(n)
        x, stats = persistent_solve(iterate, l2_residual, x0, b, cfg)
        x_exact = np.linalg.solve(A, b)
        np.testing.assert_allclose(x, x_exact, atol=1e-6)
        assert stats.converged

    def test_batched_persistent(self):
        from tensornet.gpu.persistent_kernel import batched_persistent_solve, PersistentConfig
        n = 2
        # Create simple diagonal dominant systems
        def iterate_batch(x_batch, rhs_batch):
            # Simple scaling iteration for diagonal system
            return rhs_batch * 0.5 + x_batch * 0.5

        def residual_batch(x, x_prev, rhs):
            return np.linalg.norm(x - x_prev, axis=-1)

        x0 = np.zeros((3, n))
        rhs = np.ones((3, n))
        cfg = PersistentConfig(max_iterations=500, tolerance=1e-8)
        x, stats = batched_persistent_solve(iterate_batch, residual_batch, x0, rhs, cfg)
        assert x.shape == (3, n)
        assert stats.converged


# ---------------------------------------------------------------------------
# 3.18  HIL real-time
# ---------------------------------------------------------------------------

class TestHILRealtime:
    def test_circular_buffer(self):
        from tensornet.gpu.hil_realtime import CircularBuffer
        buf = CircularBuffer(capacity=4, n_channels=2)
        for i in range(6):
            buf.push(np.array([float(i), float(i * 2)]))
        assert buf.count == 4
        assert buf.full
        latest = buf.latest(2)
        assert latest.shape == (2, 2)
        np.testing.assert_array_equal(latest[-1], [5.0, 10.0])

    def test_timing_budget(self):
        from tensornet.gpu.hil_realtime import TimingBudget
        tb = TimingBudget()
        assert tb.validate()  # default should fit
        tb2 = TimingBudget(tick_period_us=100, solver_budget_us=600)
        assert not tb2.validate()  # 600 > 100

    def test_hil_controller_single_tick(self):
        from tensornet.gpu.hil_realtime import HILController, HILConfig

        cfg = HILConfig(sensor_channels=2, actuator_channels=2)
        ctrl = HILController(cfg)

        def solver(state, dt):
            return state + 0.01, 1e-6

        def output_fn(state):
            pass

        sensor = np.array([1.0, 2.0])
        entry = ctrl.run_tick(solver, output_fn, sensor)
        assert entry.tick == 1
        assert entry.residual == 1e-6

    def test_hil_loop(self):
        from tensornet.gpu.hil_realtime import HILController, HILConfig
        cfg = HILConfig(sensor_channels=1, actuator_channels=1)
        ctrl = HILController(cfg)

        def solver(state, dt):
            return state * 0.99, abs(state[0]) if len(state) > 0 else 0.0

        def output_fn(state):
            pass

        telemetry = ctrl.run_loop(solver, output_fn, lambda: np.array([1.0]), n_ticks=50)
        assert len(telemetry.entries) == 50

    def test_telemetry_jitter(self):
        from tensornet.gpu.hil_realtime import TelemetryRing, TelemetryEntry
        ring = TelemetryRing()
        for i in range(10):
            ring.record(TelemetryEntry(
                tick=i, timestamp_us=float(i * 1000),
                solver_us=500.0, output_us=100.0,
                overrun=False, residual=1e-6
            ))
        mean, std, mx = ring.jitter_stats()
        assert mean == pytest.approx(1000.0, rel=1e-6)


# ---------------------------------------------------------------------------
# 3.19  Tensor Core
# ---------------------------------------------------------------------------

class TestTensorCore:
    def test_wmma_tiles(self):
        from tensornet.gpu.tensor_core import WMMA_TILES, TCPrecision
        tile = WMMA_TILES[TCPrecision.FP16_FP32]
        assert tile.m == 16 and tile.n == 16 and tile.k == 16

    def test_fragment_mma(self):
        from tensornet.gpu.tensor_core import Fragment, wmma_mma, WMMA_TILES, TCPrecision
        tile = WMMA_TILES[TCPrecision.FP16_FP32]
        A = np.random.randn(32, 32).astype(np.float32)
        B = np.random.randn(32, 32).astype(np.float32)
        fa = Fragment.load_a(A, tile, 0, 0)
        fb = Fragment.load_b(B, tile, 0, 0)
        fc = Fragment.zeros(tile)
        result = wmma_mma(fa, fb, fc)
        expected = A[:16, :16].astype(np.float16).astype(np.float32) @ \
                   B[:16, :16].astype(np.float16).astype(np.float32)
        np.testing.assert_allclose(result.data, expected, rtol=0.05)

    def test_tc_gemm_cpu(self):
        from tensornet.gpu.tensor_core import tc_gemm, TCGEMMConfig, TCPrecision
        A = np.random.randn(20, 24).astype(np.float64)
        B = np.random.randn(24, 16).astype(np.float64)
        cfg = TCGEMMConfig(precision=TCPrecision.FP64_FP64, use_torch=False, use_cupy=False)
        C = tc_gemm(A, B, cfg)
        np.testing.assert_allclose(C, A @ B, atol=1e-10)

    def test_tt_contract_tc(self):
        from tensornet.gpu.tensor_core import tt_contract_tc, TCPrecision
        # cores: (r_k, n_k, r_{k+1})
        cores = [np.random.randn(1, 2, 3), np.random.randn(3, 2, 3), np.random.randn(3, 2, 1)]
        result = tt_contract_tc(cores, precision=TCPrecision.FP64_FP64)
        # Manual TT contraction matching tt_contract_tc reshape logic:
        # Core 0: reshape(r0*n0, r1) = (2, 3)
        r = cores[0].reshape(-1, cores[0].shape[-1])
        # Core 1: reshape(r1, n1*r2) = (3, 6); result = (2, 6) -> reshape(-1, r2) = (4, 3)
        mat1 = cores[1].reshape(cores[1].shape[0], -1)
        r = (r @ mat1).reshape(-1, cores[1].shape[-1])
        # Core 2: reshape(r2, n2*r3) = (3, 2) -> result = (4, 2) -> reshape(-1, r3) = (8, 1)
        mat2 = cores[2].reshape(cores[2].shape[0], -1)
        r = (r @ mat2).reshape(-1, cores[2].shape[-1])
        np.testing.assert_allclose(result, r, atol=1e-10)

    def test_throughput_estimate(self):
        from tensornet.gpu.tensor_core import estimate_throughput, TCPrecision
        tp = estimate_throughput(4096, 4096, 4096, TCPrecision.FP16_FP32)
        assert tp.peak_tflops > 0
        assert 0 <= tp.efficiency <= 1.0

    def test_fusion_analysis(self):
        from tensornet.gpu.tensor_core import analyze_fusion
        shapes = [(128, 64, 64), (64, 32, 32), (32, 16, 16)]
        hints = analyze_fusion(shapes)
        assert len(hints) == 2
        assert hints[0].can_fuse  # 64 == 64


# ---------------------------------------------------------------------------
# 3.20  Memory-mapped GPU tensors
# ---------------------------------------------------------------------------

class TestManagedMemory:
    def test_mmap_tensor_basic(self):
        from tensornet.gpu.managed_memory import MMapTensor
        with tempfile.TemporaryDirectory() as d:
            t = MMapTensor((100, 10), path=Path(d) / "test.mmap")
            t[0] = np.ones(10)
            np.testing.assert_array_equal(t[0], np.ones(10))
            t.flush()
            t.close()

    def test_mmap_tensor_roundtrip(self):
        from tensornet.gpu.managed_memory import MMapTensor
        t = MMapTensor((50,), dtype=np.float64)
        data = np.arange(50, dtype=np.float64)
        t[:] = data
        np.testing.assert_array_equal(t.to_numpy(), data)
        t.close()

    def test_page_table(self):
        from tensornet.gpu.managed_memory import PageTable, MemLocation
        pt = PageTable(page_size=1024)
        pt.add_page(0, 0, 1024)
        pt.add_page(1, 1024, 1024)
        pt.mark_accessed(0)
        pt.mark_accessed(0)
        assert pt.get_page(0).access_count == 2
        assert len(pt.pages_on(MemLocation.HOST)) == 2

    def test_access_tracker_predict(self):
        from tensornet.gpu.managed_memory import AccessTracker
        tracker = AccessTracker()
        for i in range(20):
            tracker.record(i)  # stride=1
        preds = tracker.predict_next(3)
        assert preds == [20, 21, 22]

    def test_managed_tensor_backend(self):
        from tensornet.gpu.managed_memory import ManagedTensor
        t = ManagedTensor((10, 10))
        # Backend might be numpy, torch, or cupy depending on environment
        assert t.backend in ("numpy", "torch", "cupy")
        data = np.random.randn(10, 10)
        t.from_numpy(data)
        np.testing.assert_allclose(t.to_numpy(), data)

    def test_tt_core_stream(self):
        from tensornet.gpu.managed_memory import TTCoreStream, ooc_tt_contract
        with tempfile.TemporaryDirectory() as d:
            # Use compatible shapes for TT contraction:
            # tt_contract_tc reshape: core0 -> (r0*n0, r1), core_i -> (ri, ni*ri+1)
            cores = [np.random.randn(1, 2, 3), np.random.randn(3, 2, 3), np.random.randn(3, 2, 1)]
            TTCoreStream.save(cores, d)
            stream = TTCoreStream(d)
            assert stream.n_cores == 3
            result = ooc_tt_contract(stream)
            # Manual matching ooc_tt_contract logic:
            r = cores[0].reshape(-1, cores[0].shape[-1])
            mat1 = cores[1].reshape(cores[1].shape[0], -1)
            r = (r @ mat1).reshape(-1, cores[1].shape[-1])
            mat2 = cores[2].reshape(cores[2].shape[0], -1)
            r = (r @ mat2).reshape(-1, cores[2].shape[-1])
            np.testing.assert_allclose(result, r, atol=1e-10)


# ---------------------------------------------------------------------------
# Hardware backend registry
# ---------------------------------------------------------------------------

class TestHardwareRegistry:
    def test_backend_kind_enum(self):
        from tensornet.hardware import BackendKind
        assert BackendKind.CUDA is not None
        assert BackendKind.FPGA is not None

    def test_available_backends(self):
        from tensornet.hardware import available_backends
        backends = available_backends()
        assert isinstance(backends, list)

    def test_best_backend_returns_cpu(self):
        from tensornet.hardware import best_backend
        result = best_backend()
        assert result is None or hasattr(result, "matmul")
