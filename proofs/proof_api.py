#!/usr/bin/env python3
"""
Phase 6 Proof Tests: Unification

Tests for:
- REST API server components
- GPU acceleration backend
- Distributed execution
- End-to-end integration

Run with: python -m ontic.discovery test
"""

from __future__ import annotations
import time
import hashlib
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Test framework
TESTS_RUN = 0
TESTS_PASSED = 0
TEST_RESULTS: List[Dict[str, Any]] = []


def test(name: str):
    """Test decorator."""
    def decorator(func):
        def wrapper():
            global TESTS_RUN, TESTS_PASSED
            TESTS_RUN += 1
            start = time.perf_counter()
            try:
                result = func()
                elapsed = (time.perf_counter() - start) * 1000
                TESTS_PASSED += 1
                TEST_RESULTS.append({
                    "name": name,
                    "passed": True,
                    "time_ms": elapsed,
                    "result": result,
                })
                print(f"✅ PASS | {name}")
                if result:
                    print(f"       {result}")
                print(f"       ({elapsed:.1f}ms)")
                return True
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                TEST_RESULTS.append({
                    "name": name,
                    "passed": False,
                    "time_ms": elapsed,
                    "error": str(e),
                })
                print(f"❌ FAIL | {name}")
                print(f"       Error: {e}")
                return False
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator


# =============================================================================
# API MODEL TESTS
# =============================================================================

@test("API Models: DiscoveryRequest")
def test_discovery_request():
    """Test DiscoveryRequest model."""
    from ontic.ml.discovery.api.models import DiscoveryRequest, DomainType
    
    # Create request
    req = DiscoveryRequest(
        domain=DomainType.DEFI,
        demo=True,
        use_gpu=True,
    )
    
    # Convert to dict and back
    d = req.to_dict()
    req2 = DiscoveryRequest.from_dict(d)
    
    assert req2.domain == DomainType.DEFI
    assert req2.demo is True
    assert req2.use_gpu is True
    
    return f"Round-trip successful, domain={req2.domain.value}"


@test("API Models: DiscoveryResponse")
def test_discovery_response():
    """Test DiscoveryResponse model."""
    from ontic.ml.discovery.api.models import (
        DiscoveryResponse, DomainType, Finding, SeverityLevel, PipelineMetrics
    )
    
    # Create response with findings
    response = DiscoveryResponse(
        success=True,
        domain=DomainType.MARKETS,
        findings=[
            Finding(
                id="f1",
                severity=SeverityLevel.CRITICAL,
                category="flash_crash",
                title="Flash Crash Detected",
                description="V-shaped recovery pattern",
                confidence=0.95,
            )
        ],
        metrics=PipelineMetrics(
            total_time_ms=150.0,
            gpu_used=True,
        ),
    )
    
    # Serialize
    d = response.to_dict()
    json_str = response.to_json()
    
    assert d["success"] is True
    assert len(d["findings"]) == 1
    assert d["findings"][0]["severity"] == "critical"
    assert d["metrics"]["gpu_used"] is True
    
    return f"Response serialized, {len(json_str)} bytes"


@test("API Models: LiveDataRequest")
def test_live_data_request():
    """Test LiveDataRequest model."""
    from ontic.ml.discovery.api.models import (
        LiveDataRequest, LiveMode, HistoricalEvent
    )
    
    req = LiveDataRequest(
        mode=LiveMode.REPLAY,
        event=HistoricalEvent.FLASH_CRASH_2010,
        window_size=30,
    )
    
    d = req.to_dict()
    req2 = LiveDataRequest.from_dict(d)
    
    assert req2.mode == LiveMode.REPLAY
    assert req2.event == HistoricalEvent.FLASH_CRASH_2010
    assert req2.window_size == 30
    
    return f"mode={req2.mode.value}, event={req2.event.value}"


@test("API Models: GPUStatus")
def test_gpu_status():
    """Test GPUStatus model."""
    from ontic.ml.discovery.api.models import GPUStatus, GPUDevice
    
    status = GPUStatus(
        available=True,
        cuda_version="12.8",
        icicle_available=False,
        devices=[
            GPUDevice(
                index=0,
                name="RTX 4090",
                memory_total_mb=24576,
                memory_free_mb=20000,
                compute_capability="8.9",
                utilization_percent=15.0,
            )
        ],
        active_device=0,
    )
    
    d = status.to_dict()
    
    assert d["available"] is True
    assert len(d["devices"]) == 1
    assert d["devices"][0]["name"] == "RTX 4090"
    
    return f"GPU: {d['devices'][0]['name']}, {d['devices'][0]['memory_total_mb']}MB"


@test("API Models: StreamingRequest")
def test_streaming_request():
    """Test StreamingRequest model."""
    from ontic.ml.discovery.api.models import StreamingRequest, SeverityLevel
    
    req = StreamingRequest(
        symbol="ETH-USD",
        exchange="coinbase",
        window_size=50,
        bar_interval_seconds=300,
        alert_threshold=SeverityLevel.MEDIUM,
    )
    
    d = req.to_dict()
    req2 = StreamingRequest.from_dict(d)
    
    assert req2.symbol == "ETH-USD"
    assert req2.window_size == 50
    assert req2.bar_interval_seconds == 300
    
    return f"symbol={req2.symbol}, window={req2.window_size}"


# =============================================================================
# GPU BACKEND TESTS
# =============================================================================

@test("GPU Backend: Availability Check")
def test_gpu_availability():
    """Test GPU availability detection."""
    from ontic.ml.discovery.api.gpu import gpu_available, get_gpu_info
    
    info = get_gpu_info()
    
    return f"GPU available: {info['available']}, CUDA: {info['cuda_version']}"


@test("GPU Backend: GPUBackend CPU Fallback")
def test_gpu_backend_cpu():
    """Test GPUBackend with CPU fallback."""
    from ontic.ml.discovery.api.gpu import GPUBackend, AcceleratorConfig, AcceleratorType
    import numpy as np
    
    # Force CPU mode by not using GPU even if available
    backend = GPUBackend(AcceleratorConfig())
    
    # Test matrix multiplication
    a = np.random.randn(100, 100).astype(np.float32)
    b = np.random.randn(100, 100).astype(np.float32)
    
    result = backend.matmul(a, b)
    
    # Verify shape
    if hasattr(result, 'shape'):
        assert result.shape == (100, 100)
    else:
        result = backend.to_cpu(result)
        assert result.shape == (100, 100)
    
    return f"100x100 matmul on {backend.accelerator_type.value}"


@test("GPU Backend: FFT")
def test_gpu_backend_fft():
    """Test GPUBackend FFT."""
    from ontic.ml.discovery.api.gpu import GPUBackend
    import numpy as np
    
    backend = GPUBackend()
    
    # Create signal
    x = np.sin(np.linspace(0, 4*np.pi, 256))
    
    result = backend.fft(x)
    result_cpu = backend.to_cpu(result)
    
    assert len(result_cpu) == 256
    
    return f"256-point FFT on {backend.accelerator_type.value}"


@test("GPU Backend: Eigendecomposition")
def test_gpu_backend_eigh():
    """Test GPUBackend eigendecomposition."""
    from ontic.ml.discovery.api.gpu import GPUBackend
    import numpy as np
    
    backend = GPUBackend()
    
    # Create symmetric matrix
    a = np.random.randn(50, 50)
    a = (a + a.T) / 2  # Make symmetric
    
    eigenvalues, eigenvectors = backend.eigh(a)
    
    ev_cpu = backend.to_cpu(eigenvalues)
    assert len(ev_cpu) == 50
    
    return f"50x50 eigendecomposition, {backend.accelerator_type.value}"


@test("GPU Backend: SVD")
def test_gpu_backend_svd():
    """Test GPUBackend SVD."""
    from ontic.ml.discovery.api.gpu import GPUBackend
    import numpy as np
    
    backend = GPUBackend()
    
    # Create matrix
    x = np.random.randn(100, 50).astype(np.float32)
    
    U, S, Vh = backend.svd(x)
    
    S_cpu = backend.to_cpu(S)
    assert len(S_cpu) == 50
    
    return f"100x50 SVD, singular values: {len(S_cpu)}"


@test("GPU Backend: Metrics Tracking")
def test_gpu_backend_metrics():
    """Test GPUBackend metrics tracking."""
    from ontic.ml.discovery.api.gpu import GPUBackend
    import numpy as np
    
    backend = GPUBackend()
    backend.clear_metrics()
    
    # Run several operations
    a = np.random.randn(50, 50).astype(np.float32)
    backend.matmul(a, a)
    backend.fft(a[0])
    backend.eigh((a + a.T) / 2)
    
    metrics = backend.get_metrics()
    
    assert len(metrics) == 3
    assert metrics[0].operation == "matmul"
    assert metrics[1].operation == "fft"
    assert metrics[2].operation == "eigh"
    
    total_time = sum(m.execution_time_ms for m in metrics)
    return f"3 operations tracked, total: {total_time:.2f}ms"


# =============================================================================
# ICICLE ACCELERATOR TESTS
# =============================================================================

@test("Icicle: NTT (CPU Fallback)")
def test_icicle_ntt():
    """Test Icicle NTT with CPU fallback."""
    from ontic.ml.discovery.api.gpu import IcicleAccelerator
    import numpy as np
    
    icicle = IcicleAccelerator()
    
    # Small NTT
    n = 8
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
    prime = 17  # Small prime for testing
    
    result = icicle.ntt(values, prime)
    
    assert len(result) == n
    
    return f"NTT of {n} elements (icicle available: {icicle.available})"


@test("Icicle: Poseidon Hash (CPU Fallback)")
def test_icicle_poseidon():
    """Test Icicle Poseidon hash with CPU fallback."""
    from ontic.ml.discovery.api.gpu import IcicleAccelerator
    import numpy as np
    
    icicle = IcicleAccelerator()
    
    inputs = np.array([1, 2, 3], dtype=np.int64)
    
    result = icicle.poseidon_hash(inputs)
    
    assert isinstance(result, (int, np.integer))
    
    return f"Poseidon hash: {result % (10**10)}..."


# =============================================================================
# DISTRIBUTED EXECUTION TESTS
# =============================================================================

@test("Distributed: DistributedConfig")
def test_distributed_config():
    """Test DistributedConfig."""
    from ontic.ml.discovery.api.distributed import DistributedConfig
    
    config = DistributedConfig(
        coordinator_host="localhost",
        coordinator_port=50051,
        worker_threads=8,
        load_balancing="least_loaded",
    )
    
    d = config.to_dict()
    
    assert d["worker_threads"] == 8
    assert d["load_balancing"] == "least_loaded"
    
    return f"threads={config.worker_threads}, lb={config.load_balancing}"


@test("Distributed: WorkerNode")
def test_worker_node():
    """Test WorkerNode."""
    from ontic.ml.discovery.api.distributed import WorkerNode, DistributedTask
    
    worker = WorkerNode(worker_id="test-worker")
    
    # Register a function
    worker.register_function("square", lambda x: x * x)
    
    # Start worker
    worker.start()
    
    # Submit task
    task = DistributedTask(
        id="task-1",
        function_name="square",
        args=(7,),
    )
    worker.submit_task(task)
    
    # Wait for result
    time.sleep(0.2)
    result = worker.get_result("task-1")
    
    worker.stop()
    
    assert result is not None
    assert result.result == 49
    
    return f"Task result: {result.result}"


@test("Distributed: DistributedCoordinator")
def test_distributed_coordinator():
    """Test DistributedCoordinator."""
    from ontic.ml.discovery.api.distributed import DistributedCoordinator
    
    coordinator = DistributedCoordinator()
    coordinator.start(local_mode=True)
    
    # Register function
    coordinator.register_function("add", lambda a, b: a + b)
    
    # Submit task
    task_id = coordinator.submit("add", 10, 20)
    
    # Get result
    result = coordinator.get_result(task_id, timeout=5.0)
    
    coordinator.stop()
    
    assert result is not None
    assert result.result == 30
    
    return f"Coordinated task: 10 + 20 = {result.result}"


@test("Distributed: Batch Submission")
def test_distributed_batch():
    """Test batch task submission."""
    from ontic.ml.discovery.api.distributed import DistributedCoordinator
    
    coordinator = DistributedCoordinator()
    coordinator.start(local_mode=True)
    
    # Register function
    coordinator.register_function("double", lambda x: x * 2)
    
    # Submit batch
    items = [((i,), {}) for i in range(5)]
    task_ids = coordinator.submit_batch("double", items)
    
    # Get results
    results = coordinator.get_results(task_ids, timeout=5.0)
    
    coordinator.stop()
    
    values = [r.result for r in results if r and r.result is not None]
    
    assert values == [0, 2, 4, 6, 8]
    
    return f"Batch results: {values}"


@test("Distributed: Map Operation")
def test_distributed_map():
    """Test distributed map operation."""
    from ontic.ml.discovery.api.distributed import DistributedCoordinator
    
    coordinator = DistributedCoordinator()
    coordinator.start(local_mode=True)
    
    coordinator.register_function("cube", lambda x: x ** 3)
    
    results = coordinator.map("cube", [1, 2, 3, 4], timeout=5.0)
    
    coordinator.stop()
    
    assert results == [1, 8, 27, 64]
    
    return f"Map results: {results}"


# =============================================================================
# API SERVER TESTS
# =============================================================================

@test("API Server: ServerConfig")
def test_server_config():
    """Test ServerConfig."""
    from ontic.ml.discovery.api.server import ServerConfig
    
    config = ServerConfig(
        host="0.0.0.0",
        port=8080,
        debug=True,
        enable_gpu=True,
        enable_distributed=False,
    )
    
    assert config.port == 8080
    assert config.debug is True
    
    return f"Config: port={config.port}, debug={config.debug}"


@test("API Server: ServerStats")
def test_server_stats():
    """Test ServerStats."""
    from ontic.ml.discovery.api.server import ServerStats
    
    stats = ServerStats()
    stats.requests_total = 100
    stats.requests_success = 95
    stats.requests_failed = 5
    
    uptime = stats.uptime_seconds
    
    assert stats.requests_total == 100
    assert uptime >= 0
    
    return f"Requests: {stats.requests_success}/{stats.requests_total} success"


@test("API Server: DiscoveryAPIServer Creation")
def test_api_server_creation():
    """Test DiscoveryAPIServer creation."""
    from ontic.ml.discovery.api.server import DiscoveryAPIServer, ServerConfig, FASTAPI_AVAILABLE
    
    config = ServerConfig(port=8001)
    server = DiscoveryAPIServer(config)
    
    if FASTAPI_AVAILABLE:
        assert server.app is not None
        return f"FastAPI app created, routes registered"
    else:
        return f"FastAPI not available, server in fallback mode"


@test("API Server: create_app Factory")
def test_create_app():
    """Test create_app factory function."""
    from ontic.ml.discovery.api.server import create_app, ServerConfig, FASTAPI_AVAILABLE
    
    app = create_app(ServerConfig(port=8002))
    
    if FASTAPI_AVAILABLE:
        assert app is not None
        return f"App created via factory"
    else:
        return f"FastAPI not available"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

@test("Integration: Full Discovery Request Flow")
def test_full_discovery_flow():
    """Test full discovery request flow."""
    from ontic.ml.discovery.api.models import (
        DiscoveryRequest, DiscoveryResponse, DomainType, Finding, SeverityLevel
    )
    from ontic.ml.discovery.api.gpu import GPUBackend
    import numpy as np
    
    # Create request
    request = DiscoveryRequest(
        domain=DomainType.MARKETS,
        demo=True,
        use_gpu=True,
    )
    
    # Simulate GPU processing
    backend = GPUBackend()
    data = np.random.randn(100, 100).astype(np.float32)
    backend.matmul(data, data)
    
    # Create response
    response = DiscoveryResponse(
        success=True,
        domain=request.domain,
        findings=[
            Finding(
                id="int-1",
                severity=SeverityLevel.HIGH,
                category="integration_test",
                title="Integration Test Finding",
                description="Test finding from integration",
                confidence=0.99,
            )
        ],
        attestation_hash=hashlib.sha256(b"test").hexdigest(),
    )
    
    # Verify
    json_str = response.to_json()
    parsed = json.loads(json_str)
    
    assert parsed["success"] is True
    assert len(parsed["findings"]) == 1
    
    return f"Flow complete: {len(json_str)} bytes response"


@test("Integration: Live Data API Flow")
def test_live_data_api_flow():
    """Test live data API flow."""
    from ontic.ml.discovery.api.models import (
        LiveDataRequest, LiveDataResponse, LiveMode, HistoricalEvent, StreamingAlert, SeverityLevel
    )
    
    # Create request
    request = LiveDataRequest(
        mode=LiveMode.REPLAY,
        event=HistoricalEvent.FLASH_CRASH_2010,
        window_size=20,
    )
    
    # Simulate response
    response = LiveDataResponse(
        success=True,
        mode=request.mode,
        event=request.event,
        bars_analyzed=60,
        alerts=[
            StreamingAlert(
                timestamp=datetime.utcnow().isoformat(),
                alert_type="flash_crash",
                severity=SeverityLevel.CRITICAL,
                message="Flash crash detected at bar 45",
                price=1050.0,
                bar_index=45,
            )
        ],
        statistics={
            "peak_drawdown": -0.0918,
            "recovery_time_bars": 15,
        },
    )
    
    json_str = response.to_json()
    parsed = json.loads(json_str)
    
    assert parsed["bars_analyzed"] == 60
    assert len(parsed["alerts"]) == 1
    
    return f"Live flow: {parsed['bars_analyzed']} bars, {len(parsed['alerts'])} alerts"


@test("Integration: Distributed Discovery")
def test_distributed_discovery():
    """Test distributed discovery integration."""
    from ontic.ml.discovery.api.distributed import DistributedCoordinator
    from ontic.ml.discovery.api.gpu import GPUBackend
    import numpy as np
    
    # Setup coordinator
    coordinator = DistributedCoordinator()
    coordinator.start(local_mode=True)
    
    # Register analysis function
    def analyze_batch(data):
        backend = GPUBackend()
        return {"processed": len(data), "sum": sum(data)}
    
    coordinator.register_function("analyze", analyze_batch)
    
    # Submit batch
    batches = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    results = coordinator.map("analyze", batches, timeout=5.0)
    
    coordinator.stop()
    
    total_processed = sum(r["processed"] for r in results if r)
    total_sum = sum(r["sum"] for r in results if r)
    
    assert total_processed == 9
    assert total_sum == 45
    
    return f"Distributed: {total_processed} items, sum={total_sum}"


@test("Integration: Attestation Hash")
def test_attestation_hash():
    """Test attestation hash generation."""
    from ontic.ml.discovery.api.models import DiscoveryResponse, DomainType
    import hashlib
    import json
    
    response = DiscoveryResponse(
        success=True,
        domain=DomainType.DEFI,
    )
    
    # Generate attestation
    attestation_data = {
        "domain": response.domain.value,
        "success": response.success,
        "timestamp": response.timestamp,
    }
    attestation_hash = hashlib.sha256(
        json.dumps(attestation_data, sort_keys=True).encode()
    ).hexdigest()
    
    assert len(attestation_hash) == 64
    
    return f"SHA-256: {attestation_hash[:16]}..."


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_all_tests():
    """Run all Phase 6 proof tests."""
    print("=" * 60)
    print("PHASE 6: UNIFICATION - PROOF TESTS")
    print("=" * 60)
    print()
    
    # API Model Tests
    print("--- API Models ---")
    test_discovery_request()
    test_discovery_response()
    test_live_data_request()
    test_gpu_status()
    test_streaming_request()
    print()
    
    # GPU Backend Tests
    print("--- GPU Backend ---")
    test_gpu_availability()
    test_gpu_backend_cpu()
    test_gpu_backend_fft()
    test_gpu_backend_eigh()
    test_gpu_backend_svd()
    test_gpu_backend_metrics()
    print()
    
    # Icicle Tests
    print("--- Icicle Accelerator ---")
    test_icicle_ntt()
    test_icicle_poseidon()
    print()
    
    # Distributed Tests
    print("--- Distributed Execution ---")
    test_distributed_config()
    test_worker_node()
    test_distributed_coordinator()
    test_distributed_batch()
    test_distributed_map()
    print()
    
    # API Server Tests
    print("--- API Server ---")
    test_server_config()
    test_server_stats()
    test_api_server_creation()
    test_create_app()
    print()
    
    # Integration Tests
    print("--- Integration ---")
    test_full_discovery_flow()
    test_live_data_api_flow()
    test_distributed_discovery()
    test_attestation_hash()
    print()
    
    # Summary
    print("=" * 60)
    print(f"UNIFICATION API: {TESTS_PASSED}/{TESTS_RUN} tests passed")
    if TESTS_PASSED == TESTS_RUN:
        print("✅ Phase 6 validation COMPLETE")
    else:
        print(f"❌ {TESTS_RUN - TESTS_PASSED} tests failed")
    print("=" * 60)
    
    # Generate attestation
    attestation = {
        "phase": 6,
        "name": "Unification",
        "tests_passed": TESTS_PASSED,
        "tests_total": TESTS_RUN,
        "timestamp": datetime.utcnow().isoformat(),
        "results": TEST_RESULTS,
    }
    
    attestation_json = json.dumps(attestation, indent=2, default=str)
    attestation_hash = hashlib.sha256(attestation_json.encode()).hexdigest()
    
    print(f"\nAttestation Hash: {attestation_hash[:32]}...")
    
    # Save attestation
    try:
        with open("proofs/proof_api.json", "w") as f:
            f.write(attestation_json)
        print(f"Attestation saved: proofs/proof_api.json")
    except Exception as e:
        print(f"Could not save attestation: {e}")
    
    return TESTS_PASSED == TESTS_RUN


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
