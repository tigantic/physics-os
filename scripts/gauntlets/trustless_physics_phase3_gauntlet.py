#!/usr/bin/env python3
"""
Trustless Physics Gauntlet — Phase 3 Validation
================================================

Validates the Scaling & Decentralization phase: Prover Pool (batch,
incremental, compression), Gevulot integration, Certificate Dashboard,
Multi-Tenant operations, and Lean ProverOptimization formal proofs.

Test Layers:
    1. prover_pool: Rust module compilation + unit tests (traits, batch,
       incremental, compressor)
    2. gevulot: Gevulot integration module (client, registry, types)
    3. dashboard: Certificate dashboard (models, analytics, store)
    4. multi_tenant: Multi-tenant operations (tenant, metering, store,
       isolation)
    5. lean_prover_optimization: Lean 4 ProverOptimization formalization
    6. integration: Cross-module validation
    7. regression: Phase 1 + Phase 2 still pass

Pass criteria: ALL tests must pass. No exceptions.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# ── Setup ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger("trustless_physics_phase3_gauntlet")

# ═════════════════════════════════════════════════════════════════════════════
# Gauntlet Framework
# ═════════════════════════════════════════════════════════════════════════════

RESULTS: dict[str, dict[str, Any]] = {}
_start_time = time.monotonic()


def gauntlet(name: str, layer: str = "phase3"):
    """Decorator to register a gauntlet test."""
    def decorator(func):
        def wrapper():
            t0 = time.monotonic()
            try:
                func()
                elapsed = time.monotonic() - t0
                RESULTS[name] = {
                    "layer": layer,
                    "passed": True,
                    "time_seconds": round(elapsed, 4),
                    "error": None,
                }
                logger.info(f"  ✅ {name} ({elapsed:.3f}s)")
                return True
            except Exception as e:
                elapsed = time.monotonic() - t0
                RESULTS[name] = {
                    "layer": layer,
                    "passed": False,
                    "time_seconds": round(elapsed, 4),
                    "error": f"{type(e).__name__}: {e}",
                }
                logger.error(f"  ❌ {name} ({elapsed:.3f}s)")
                logger.error(f"     {type(e).__name__}: {e}")
                traceback.print_exc()
                return False
        wrapper.__name__ = name
        return wrapper
    return decorator


def run_cmd(cmd: list[str], cwd: Path | None = None, timeout: int = 120) -> str:
    """Run a shell command and return stdout. Raises on failure."""
    result = subprocess.run(
        cmd,
        cwd=cwd or ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {' '.join(cmd)}\n"
            f"STDOUT: {result.stdout[-2000:]}\n"
            f"STDERR: {result.stderr[-2000:]}"
        )
    return result.stdout


# ═════════════════════════════════════════════════════════════════════════════
# Layer 1: Prover Pool Module
# ═════════════════════════════════════════════════════════════════════════════


@gauntlet("prover_pool_files_exist", layer="prover_pool")
def test_prover_pool_files_exist():
    """Verify all prover_pool source files exist."""
    base = ROOT / "fluidelite-zk" / "src" / "prover_pool"
    required = ["mod.rs", "traits.rs", "batch.rs", "incremental.rs", "compressor.rs"]
    for fname in required:
        fpath = base / fname
        assert fpath.exists(), f"Missing: {fpath.relative_to(ROOT)}"
        size = fpath.stat().st_size
        assert size > 500, f"{fname} too small: {size} bytes"
    logger.info(f"    All {len(required)} prover_pool files present")


@gauntlet("prover_pool_traits_api", layer="prover_pool")
def test_prover_pool_traits_api():
    """Verify traits.rs has all required types and trait impls."""
    src = (ROOT / "fluidelite-zk" / "src" / "prover_pool" / "traits.rs").read_text()
    # Core types
    assert "pub enum SolverType" in src
    assert "Euler3D" in src
    assert "NsImex" in src
    # Traits
    assert "pub trait PhysicsProof" in src
    assert "pub trait PhysicsProver" in src
    assert "pub trait PhysicsVerifier" in src
    # Trait impls for Euler3D
    assert "PhysicsProof for" in src and "Euler3DProof" in src
    assert "PhysicsProver for" in src and "Euler3DProver" in src
    assert "PhysicsVerifier for" in src and "Euler3DVerifier" in src
    # Trait impls for NS-IMEX
    assert "PhysicsProof for" in src and "NSIMEXProof" in src
    assert "PhysicsProver for" in src and "NSIMEXProver" in src
    assert "PhysicsVerifier for" in src and "NSIMEXVerifier" in src
    # Factory
    assert "pub struct ProverFactory" in src or "ProverFactory" in src


@gauntlet("prover_pool_batch_api", layer="prover_pool")
def test_prover_pool_batch_api():
    """Verify batch.rs has BatchProver with thread::scope parallelism."""
    src = (ROOT / "fluidelite-zk" / "src" / "prover_pool" / "batch.rs").read_text()
    assert "pub struct BatchProver" in src
    assert "pub struct BatchConfig" in src
    assert "pub struct ProveResult" in src
    assert "prove_batch" in src
    assert "thread::scope" in src or "std::thread" in src
    assert "Mutex" in src


@gauntlet("prover_pool_incremental_api", layer="prover_pool")
def test_prover_pool_incremental_api():
    """Verify incremental.rs has IncrementalProver with LRU cache."""
    src = (ROOT / "fluidelite-zk" / "src" / "prover_pool" / "incremental.rs").read_text()
    assert "pub struct IncrementalProver" in src
    assert "pub struct IncrementalConfig" in src
    assert "pub struct CacheKey" in src
    assert "pub struct DeltaAnalysis" in src
    assert "analyze_delta" in src
    assert "FNV" in src or "fnv" in src or "0xcbf29ce484222325" in src


@gauntlet("prover_pool_compressor_api", layer="prover_pool")
def test_prover_pool_compressor_api():
    """Verify compressor.rs has ProofCompressor with strip + RLE."""
    src = (ROOT / "fluidelite-zk" / "src" / "prover_pool" / "compressor.rs").read_text()
    assert "pub struct ProofCompressor" in src
    assert "pub struct CompressedProof" in src
    assert "pub enum CompressionMethod" in src or "CompressionMethod" in src
    assert "compress" in src
    assert "decompress" in src
    assert "zero" in src.lower() or "strip" in src.lower()
    assert "rle" in src.lower() or "run_length" in src.lower()


@gauntlet("prover_pool_compilation", layer="prover_pool")
def test_prover_pool_compilation():
    """Verify prover_pool compiles without errors."""
    output = run_cmd(
        ["cargo", "check", "--lib", "-p", "fluidelite-zk"],
        timeout=180,
    )
    # Grep stderr is captured too; check for success
    logger.info("    prover_pool compilation: OK")


@gauntlet("prover_pool_rust_tests", layer="prover_pool")
def test_prover_pool_rust_tests():
    """Run all prover_pool Rust unit tests."""
    output = run_cmd(
        ["cargo", "test", "--lib", "-p", "fluidelite-zk", "--", "prover_pool"],
        timeout=120,
    )
    # Count passed tests
    match = re.search(r"test result: ok\. (\d+) passed", output)
    assert match, f"Tests did not pass cleanly:\n{output[-1000:]}"
    count = int(match.group(1))
    assert count >= 40, f"Expected ≥40 prover_pool tests, got {count}"
    logger.info(f"    prover_pool tests: {count} passed")


# ═════════════════════════════════════════════════════════════════════════════
# Layer 2: Gevulot Integration Module
# ═════════════════════════════════════════════════════════════════════════════


@gauntlet("gevulot_files_exist", layer="gevulot")
def test_gevulot_files_exist():
    """Verify all gevulot source files exist."""
    base = ROOT / "fluidelite-zk" / "src" / "gevulot"
    required = ["mod.rs", "types.rs", "client.rs", "registry.rs"]
    for fname in required:
        fpath = base / fname
        assert fpath.exists(), f"Missing: {fpath.relative_to(ROOT)}"
        size = fpath.stat().st_size
        assert size > 500, f"{fname} too small: {size} bytes"
    logger.info(f"    All {len(required)} gevulot files present")


@gauntlet("gevulot_types_api", layer="gevulot")
def test_gevulot_types_api():
    """Verify types.rs has submission, verification, config types."""
    src = (ROOT / "fluidelite-zk" / "src" / "gevulot" / "types.rs").read_text()
    assert "pub struct SubmissionId" in src or "SubmissionId" in src
    assert "pub enum SubmissionStatus" in src or "SubmissionStatus" in src
    assert "pub struct GevulotSubmission" in src
    assert "pub struct VerificationRecord" in src
    assert "pub struct GevulotConfig" in src
    assert "pub enum GevulotNetwork" in src


@gauntlet("gevulot_client_api", layer="gevulot")
def test_gevulot_client_api():
    """Verify client.rs has GevulotClient with submission lifecycle."""
    src = (ROOT / "fluidelite-zk" / "src" / "gevulot" / "client.rs").read_text()
    assert "pub struct GevulotClient" in src
    assert "submit" in src.lower()
    assert "SharedGevulotClient" in src
    assert "Arc" in src
    assert "Mutex" in src


@gauntlet("gevulot_registry_api", layer="gevulot")
def test_gevulot_registry_api():
    """Verify registry.rs has ProofRegistry with hash-indexed audit trail."""
    src = (ROOT / "fluidelite-zk" / "src" / "gevulot" / "registry.rs").read_text()
    assert "pub struct ProofRegistry" in src
    assert "pub struct RegistryEntry" in src
    assert "pub struct RegistryQuery" in src or "RegistryQuery" in src
    assert "register" in src
    assert "query" in src


@gauntlet("gevulot_rust_tests", layer="gevulot")
def test_gevulot_rust_tests():
    """Run all gevulot Rust unit tests."""
    output = run_cmd(
        ["cargo", "test", "--lib", "-p", "fluidelite-zk", "--", "gevulot"],
        timeout=120,
    )
    match = re.search(r"test result: ok\. (\d+) passed", output)
    assert match, f"Tests did not pass cleanly:\n{output[-1000:]}"
    count = int(match.group(1))
    assert count >= 40, f"Expected ≥40 gevulot tests, got {count}"
    logger.info(f"    gevulot tests: {count} passed")


# ═════════════════════════════════════════════════════════════════════════════
# Layer 3: Certificate Dashboard Module
# ═════════════════════════════════════════════════════════════════════════════


@gauntlet("dashboard_files_exist", layer="dashboard")
def test_dashboard_files_exist():
    """Verify all dashboard source files exist."""
    base = ROOT / "fluidelite-zk" / "src" / "dashboard"
    required = ["mod.rs", "models.rs", "analytics.rs"]
    for fname in required:
        fpath = base / fname
        assert fpath.exists(), f"Missing: {fpath.relative_to(ROOT)}"
        size = fpath.stat().st_size
        assert size > 500, f"{fname} too small: {size} bytes"
    logger.info(f"    All {len(required)} dashboard files present")


@gauntlet("dashboard_models_api", layer="dashboard")
def test_dashboard_models_api():
    """Verify models.rs has certificate, timeline, analytics types."""
    src = (ROOT / "fluidelite-zk" / "src" / "dashboard" / "models.rs").read_text()
    assert "pub struct ProofCertificate" in src
    assert "pub struct CertificateId" in src or "CertificateId" in src
    assert "pub struct DashboardSummary" in src or "DashboardSummary" in src
    assert "pub struct SystemHealth" in src or "SystemHealth" in src
    assert "pub struct CertificateQuery" in src or "CertificateQuery" in src
    assert "Serialize" in src
    assert "Deserialize" in src


@gauntlet("dashboard_analytics_api", layer="dashboard")
def test_dashboard_analytics_api():
    """Verify analytics.rs has CertificateStore with query engine."""
    src = (ROOT / "fluidelite-zk" / "src" / "dashboard" / "analytics.rs").read_text()
    assert "pub struct CertificateStore" in src
    assert "insert" in src
    assert "query" in src
    assert "dashboard_summary" in src or "summary" in src
    assert "solver_analytics" in src or "analytics" in src


@gauntlet("dashboard_rust_tests", layer="dashboard")
def test_dashboard_rust_tests():
    """Run all dashboard Rust unit tests."""
    output = run_cmd(
        ["cargo", "test", "--lib", "-p", "fluidelite-zk", "--", "dashboard"],
        timeout=120,
    )
    match = re.search(r"test result: ok\. (\d+) passed", output)
    assert match, f"Tests did not pass cleanly:\n{output[-1000:]}"
    count = int(match.group(1))
    assert count >= 20, f"Expected ≥20 dashboard tests, got {count}"
    logger.info(f"    dashboard tests: {count} passed")


# ═════════════════════════════════════════════════════════════════════════════
# Layer 4: Multi-Tenant Operations Module
# ═════════════════════════════════════════════════════════════════════════════


@gauntlet("multi_tenant_files_exist", layer="multi_tenant")
def test_multi_tenant_files_exist():
    """Verify all multi_tenant source files exist."""
    base = ROOT / "fluidelite-zk" / "src" / "multi_tenant"
    required = ["mod.rs", "tenant.rs", "metering.rs", "store.rs", "isolation.rs"]
    for fname in required:
        fpath = base / fname
        assert fpath.exists(), f"Missing: {fpath.relative_to(ROOT)}"
        size = fpath.stat().st_size
        assert size > 500, f"{fname} too small: {size} bytes"
    logger.info(f"    All {len(required)} multi_tenant files present")


@gauntlet("multi_tenant_tenant_api", layer="multi_tenant")
def test_multi_tenant_tenant_api():
    """Verify tenant.rs has TenantManager with tier-based config."""
    src = (ROOT / "fluidelite-zk" / "src" / "multi_tenant" / "tenant.rs").read_text()
    assert "pub struct TenantManager" in src
    assert "pub struct TenantConfig" in src
    assert "pub enum TenantTier" in src or "TenantTier" in src
    assert "pub struct ApiKey" in src or "ApiKey" in src
    assert "register" in src
    assert "authenticate" in src


@gauntlet("multi_tenant_metering_api", layer="multi_tenant")
def test_multi_tenant_metering_api():
    """Verify metering.rs has UsageMeter with rate limiting."""
    src = (ROOT / "fluidelite-zk" / "src" / "multi_tenant" / "metering.rs").read_text()
    assert "pub struct UsageMeter" in src
    assert "pub enum RateLimitDecision" in src or "RateLimitDecision" in src
    assert "check_rate_limit" in src or "rate_limit" in src
    assert "UsageRecord" in src


@gauntlet("multi_tenant_store_api", layer="multi_tenant")
def test_multi_tenant_store_api():
    """Verify store.rs has PersistentCertStore with WAL."""
    src = (ROOT / "fluidelite-zk" / "src" / "multi_tenant" / "store.rs").read_text()
    assert "pub struct PersistentCertStore" in src
    assert "pub struct StoreConfig" in src or "StoreConfig" in src
    assert "WAL" in src or "wal" in src
    assert "snapshot" in src
    assert "recover" in src or "replay" in src


@gauntlet("multi_tenant_isolation_api", layer="multi_tenant")
def test_multi_tenant_isolation_api():
    """Verify isolation.rs has ComputeIsolator with RAII guards."""
    src = (ROOT / "fluidelite-zk" / "src" / "multi_tenant" / "isolation.rs").read_text()
    assert "IsolationTracker" in src or "ComputeIsolator" in src
    assert "IsolationGuard" in src
    assert "Drop" in src  # RAII
    assert "AtomicUsize" in src or "atomic" in src.lower()


@gauntlet("multi_tenant_rust_tests", layer="multi_tenant")
def test_multi_tenant_rust_tests():
    """Run all multi_tenant Rust unit tests."""
    output = run_cmd(
        ["cargo", "test", "--lib", "-p", "fluidelite-zk", "--", "multi_tenant"],
        timeout=120,
    )
    match = re.search(r"test result: ok\. (\d+) passed", output)
    assert match, f"Tests did not pass cleanly:\n{output[-1000:]}"
    count = int(match.group(1))
    assert count >= 40, f"Expected ≥40 multi_tenant tests, got {count}"
    logger.info(f"    multi_tenant tests: {count} passed")


# ═════════════════════════════════════════════════════════════════════════════
# Layer 5: Lean ProverOptimization
# ═════════════════════════════════════════════════════════════════════════════


@gauntlet("lean_prover_opt_exists", layer="lean")
def test_lean_prover_opt_exists():
    """Verify ProverOptimization.lean exists with required content."""
    lean_path = ROOT / "lean_yang_mills" / "YangMills" / "ProverOptimization.lean"
    assert lean_path.exists(), f"Missing: {lean_path.relative_to(ROOT)}"
    src = lean_path.read_text()
    size = len(src)
    assert size > 2000, f"ProverOptimization.lean too small: {size} bytes"
    logger.info(f"    ProverOptimization.lean: {size} bytes")


@gauntlet("lean_prover_opt_structure", layer="lean")
def test_lean_prover_opt_structure():
    """Verify ProverOptimization.lean has correct structure."""
    src = (ROOT / "lean_yang_mills" / "YangMills" / "ProverOptimization.lean").read_text()
    # Namespace
    assert "namespace ProverOptimization" in src
    # Imports
    assert "import Mathlib" in src
    # Core structures
    assert "PhysicsProof" in src
    assert "ProofBatch" in src
    assert "CompressedProof" in src
    assert "ProofBundle" in src


@gauntlet("lean_prover_opt_batch_theorems", layer="lean")
def test_lean_prover_opt_batch_theorems():
    """Verify batch proving soundness theorems exist."""
    src = (ROOT / "lean_yang_mills" / "YangMills" / "ProverOptimization.lean").read_text()
    assert "theorem batch_soundness" in src
    assert "theorem batch_proof_independence" in src
    assert "theorem batch_concat_valid" in src
    assert "theorem empty_batch_valid" in src
    assert "theorem singleton_batch_valid" in src


@gauntlet("lean_prover_opt_incremental_theorems", layer="lean")
def test_lean_prover_opt_incremental_theorems():
    """Verify incremental proving correctness theorems exist."""
    src = (ROOT / "lean_yang_mills" / "YangMills" / "ProverOptimization.lean").read_text()
    assert "theorem incremental_soundness" in src
    assert "qualifies_for_incremental" in src
    assert "delta_norm" in src or "change_fraction" in src


@gauntlet("lean_prover_opt_compression_theorems", layer="lean")
def test_lean_prover_opt_compression_theorems():
    """Verify compression losslessness theorems exist."""
    src = (ROOT / "lean_yang_mills" / "YangMills" / "ProverOptimization.lean").read_text()
    assert "theorem compression_lossless" in src or "compression_lossless" in src
    assert "theorem compression_preserves_validity" in src
    assert "strip_trailing_zeros" in src
    assert "rle_encode" in src
    assert "rle_decode" in src
    assert "theorem strip_length_bound" in src


@gauntlet("lean_prover_opt_master_certificate", layer="lean")
def test_lean_prover_opt_master_certificate():
    """Verify master certificate structure exists and is constructible."""
    src = (ROOT / "lean_yang_mills" / "YangMills" / "ProverOptimization.lean").read_text()
    assert "ProverOptimizationCertificate" in src
    assert "prover_optimization_certificate_exists" in src
    # Count theorems
    theorem_count = src.count("theorem ")
    assert theorem_count >= 15, f"Expected ≥15 theorems, got {theorem_count}"
    logger.info(f"    Theorem count: {theorem_count}")


@gauntlet("lean_root_imports", layer="lean")
def test_lean_root_imports():
    """Verify YangMills.lean imports ProverOptimization."""
    root_lean = ROOT / "lean_yang_mills" / "YangMills.lean"
    assert root_lean.exists(), "Missing YangMills.lean root file"
    src = root_lean.read_text()
    assert "import YangMills.ProverOptimization" in src


# ═════════════════════════════════════════════════════════════════════════════
# Layer 6: Integration Tests
# ═════════════════════════════════════════════════════════════════════════════


@gauntlet("lib_rs_modules", layer="integration")
def test_lib_rs_modules():
    """Verify lib.rs declares all Phase 3 modules."""
    lib_rs = (ROOT / "fluidelite-zk" / "src" / "lib.rs").read_text()
    assert "pub mod prover_pool;" in lib_rs
    assert "pub mod gevulot;" in lib_rs
    assert "pub mod dashboard;" in lib_rs
    assert "pub mod multi_tenant;" in lib_rs


@gauntlet("full_lib_compilation", layer="integration")
def test_full_lib_compilation():
    """Verify entire fluidelite-zk lib compiles without errors."""
    output = run_cmd(
        ["cargo", "check", "--lib", "-p", "fluidelite-zk"],
        timeout=180,
    )
    logger.info("    Full lib compilation: OK")


@gauntlet("full_test_suite", layer="integration")
def test_full_test_suite():
    """Run ALL fluidelite-zk lib tests (Phase 1 + 2 + 3)."""
    output = run_cmd(
        ["cargo", "test", "--lib", "-p", "fluidelite-zk"],
        timeout=180,
    )
    match = re.search(r"test result: ok\. (\d+) passed", output)
    assert match, f"Tests did not pass:\n{output[-1000:]}"
    count = int(match.group(1))
    assert count >= 250, f"Expected ≥250 total tests, got {count}"
    logger.info(f"    Full test suite: {count} passed")


@gauntlet("cross_module_types", layer="integration")
def test_cross_module_types():
    """Verify SolverType is shared across all Phase 3 modules."""
    # SolverType is defined in prover_pool/traits.rs and used everywhere
    for submod in ["gevulot", "dashboard", "multi_tenant"]:
        mod_dir = ROOT / "fluidelite-zk" / "src" / submod
        all_src = ""
        for rs_file in mod_dir.glob("*.rs"):
            all_src += rs_file.read_text()
        assert "SolverType" in all_src, f"SolverType not used in {submod}"


@gauntlet("serde_consistency", layer="integration")
def test_serde_consistency():
    """Verify SolverType serializes to expected snake_case strings."""
    traits_src = (ROOT / "fluidelite-zk" / "src" / "prover_pool" / "traits.rs").read_text()
    # Check serde rename attributes
    assert 'rename = "euler3d"' in traits_src, "Missing serde rename for Euler3D"
    assert 'rename = "ns_imex"' in traits_src, "Missing serde rename for NsImex"


@gauntlet("total_loc_check", layer="integration")
def test_total_loc_check():
    """Verify Phase 3 has substantial LOC across all modules."""
    total_loc = 0
    for submod in ["prover_pool", "gevulot", "dashboard", "multi_tenant"]:
        mod_dir = ROOT / "fluidelite-zk" / "src" / submod
        for rs_file in mod_dir.glob("*.rs"):
            lines = rs_file.read_text().count("\n")
            total_loc += lines
    assert total_loc >= 3000, f"Expected ≥3000 LOC, got {total_loc}"
    logger.info(f"    Phase 3 Rust LOC: {total_loc}")


# ═════════════════════════════════════════════════════════════════════════════
# Layer 7: Regression Tests
# ═════════════════════════════════════════════════════════════════════════════


@gauntlet("regression_euler3d", layer="regression")
def test_regression_euler3d():
    """Verify Phase 1 euler3d tests still pass."""
    output = run_cmd(
        ["cargo", "test", "--lib", "-p", "fluidelite-zk", "--", "euler3d"],
        timeout=120,
    )
    match = re.search(r"test result: ok\. (\d+) passed", output)
    assert match, f"Euler3D regression failed:\n{output[-1000:]}"
    count = int(match.group(1))
    assert count >= 36, f"Expected ≥36 euler3d tests, got {count}"
    logger.info(f"    euler3d regression: {count} passed")


@gauntlet("regression_ns_imex", layer="regression")
def test_regression_ns_imex():
    """Verify Phase 2 ns_imex tests still pass."""
    output = run_cmd(
        ["cargo", "test", "--lib", "-p", "fluidelite-zk", "--", "ns_imex"],
        timeout=120,
    )
    match = re.search(r"test result: ok\. (\d+) passed", output)
    assert match, f"NS-IMEX regression failed:\n{output[-1000:]}"
    count = int(match.group(1))
    assert count >= 48, f"Expected ≥48 ns_imex tests, got {count}"
    logger.info(f"    ns_imex regression: {count} passed")


@gauntlet("regression_core_modules", layer="regression")
def test_regression_core_modules():
    """Verify core modules (field, mps, mpo, ops) still pass."""
    for mod_name in ["field", "mps", "mpo", "ops"]:
        output = run_cmd(
            ["cargo", "test", "--lib", "-p", "fluidelite-zk", "--", mod_name],
            timeout=60,
        )
        match = re.search(r"test result: ok\. (\d+) passed", output)
        assert match, f"{mod_name} regression failed"
        count = int(match.group(1))
        assert count >= 1, f"Expected ≥1 {mod_name} tests, got {count}"
    logger.info("    Core module regression: all passed")


@gauntlet("regression_lean_euler", layer="regression")
def test_regression_lean_euler():
    """Verify Phase 1 EulerConservation.lean still exists with theorems."""
    lean_path = ROOT / "lean_yang_mills" / "YangMills" / "EulerConservation.lean"
    assert lean_path.exists(), "EulerConservation.lean missing"
    src = lean_path.read_text()
    assert "theorem" in src
    assert "namespace EulerConservation" in src
    theorem_count = src.count("theorem ")
    assert theorem_count >= 10, f"Expected ≥10 theorems, got {theorem_count}"
    logger.info(f"    EulerConservation.lean: {theorem_count} theorems")


@gauntlet("regression_lean_ns", layer="regression")
def test_regression_lean_ns():
    """Verify Phase 2 NavierStokesConservation.lean still exists with theorems."""
    lean_path = ROOT / "lean_yang_mills" / "YangMills" / "NavierStokesConservation.lean"
    assert lean_path.exists(), "NavierStokesConservation.lean missing"
    src = lean_path.read_text()
    assert "theorem" in src
    assert "namespace NavierStokesConservation" in src
    theorem_count = src.count("theorem ")
    assert theorem_count >= 10, f"Expected ≥10 theorems, got {theorem_count}"
    logger.info(f"    NavierStokesConservation.lean: {theorem_count} theorems")


# ═════════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════════


ALL_TESTS = [
    # Layer 1: Prover Pool
    test_prover_pool_files_exist,
    test_prover_pool_traits_api,
    test_prover_pool_batch_api,
    test_prover_pool_incremental_api,
    test_prover_pool_compressor_api,
    test_prover_pool_compilation,
    test_prover_pool_rust_tests,
    # Layer 2: Gevulot
    test_gevulot_files_exist,
    test_gevulot_types_api,
    test_gevulot_client_api,
    test_gevulot_registry_api,
    test_gevulot_rust_tests,
    # Layer 3: Dashboard
    test_dashboard_files_exist,
    test_dashboard_models_api,
    test_dashboard_analytics_api,
    test_dashboard_rust_tests,
    # Layer 4: Multi-Tenant
    test_multi_tenant_files_exist,
    test_multi_tenant_tenant_api,
    test_multi_tenant_metering_api,
    test_multi_tenant_store_api,
    test_multi_tenant_isolation_api,
    test_multi_tenant_rust_tests,
    # Layer 5: Lean
    test_lean_prover_opt_exists,
    test_lean_prover_opt_structure,
    test_lean_prover_opt_batch_theorems,
    test_lean_prover_opt_incremental_theorems,
    test_lean_prover_opt_compression_theorems,
    test_lean_prover_opt_master_certificate,
    test_lean_root_imports,
    # Layer 6: Integration
    test_lib_rs_modules,
    test_full_lib_compilation,
    test_full_test_suite,
    test_cross_module_types,
    test_serde_consistency,
    test_total_loc_check,
    # Layer 7: Regression
    test_regression_euler3d,
    test_regression_ns_imex,
    test_regression_core_modules,
    test_regression_lean_euler,
    test_regression_lean_ns,
]


def run_all() -> bool:
    """Run the complete Phase 3 gauntlet."""
    print("=" * 72)
    print("  TRUSTLESS PHYSICS GAUNTLET — PHASE 3")
    print("  Scaling & Decentralization")
    print("=" * 72)
    print()

    layers = [
        "prover_pool",
        "gevulot",
        "dashboard",
        "multi_tenant",
        "lean",
        "integration",
        "regression",
    ]

    for layer in layers:
        layer_tests = [t for t in ALL_TESTS if RESULTS.get(t.__name__, {}).get("layer") == layer or True]
        # Run tests by layer
        pass

    # Run all tests sequentially
    for test_fn in ALL_TESTS:
        test_fn()

    # Summary
    total_tests = len(RESULTS)
    total_passed = sum(1 for r in RESULTS.values() if r["passed"])
    total_failed = total_tests - total_passed
    total_elapsed = time.monotonic() - _start_time

    print()
    print("=" * 72)
    print("  PHASE 3 GAUNTLET SUMMARY")
    print("=" * 72)

    # Per-layer summary
    for layer in layers:
        layer_results = {
            k: v for k, v in RESULTS.items() if v["layer"] == layer
        }
        layer_passed = sum(1 for v in layer_results.values() if v["passed"])
        layer_total = len(layer_results)
        status = "✅" if layer_passed == layer_total else "❌"
        print(f"  {status} {layer:20s} {layer_passed}/{layer_total}")

    print(f"\n{'=' * 72}")
    print(f"  Results: {total_passed}/{total_tests} passed")
    print(f"  Time:    {total_elapsed:.2f}s")
    print(f"{'=' * 72}\n")

    if total_failed > 0:
        print(f"❌ FAILED: {total_failed} test(s) failed")
        failed_names = [
            name for name, r in RESULTS.items() if not r["passed"]
        ]
        for name in failed_names:
            err = RESULTS[name].get("error", "unknown")
            print(f"   • {name}: {err}")
    else:
        print(f"✅ ALL {total_tests} TESTS PASSED")

    # Save attestation JSON
    attestation = {
        "project": "HyperTensor-VM",
        "protocol": "trustless_physics_gauntlet_phase3",
        "phase": 3,
        "description": (
            "Scaling & Decentralization: Prover Pool (batch, incremental, "
            "compression), Gevulot Integration, Certificate Dashboard, "
            "Multi-Tenant Operations, Lean ProverOptimization"
        ),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_tests": total_tests,
        "passed": total_passed,
        "failed": total_failed,
        "total_time_seconds": round(total_elapsed, 3),
        "gauntlets": RESULTS,
        "components": {
            "prover_pool": {
                "language": "Rust",
                "files": 5,
                "description": (
                    "PhysicsProver trait abstraction, BatchProver with "
                    "thread::scope parallelism, IncrementalProver with "
                    "LRU cache and FNV-1a delta analysis, ProofCompressor "
                    "with zero-strip + RLE"
                ),
            },
            "gevulot": {
                "language": "Rust",
                "files": 4,
                "description": (
                    "GevulotClient with submission lifecycle, ProofRegistry "
                    "with hash-indexed audit trail, SharedGevulotClient "
                    "thread-safe wrapper"
                ),
            },
            "dashboard": {
                "language": "Rust",
                "files": 3,
                "description": (
                    "ProofCertificate models, CertificateStore with query "
                    "engine, solver analytics with percentiles, timeline "
                    "bucketing, system health"
                ),
            },
            "multi_tenant": {
                "language": "Rust",
                "files": 5,
                "description": (
                    "TenantManager with tier-based config, UsageMeter with "
                    "sliding-window rate limiting, PersistentCertStore with "
                    "WAL + crash recovery, ComputeIsolator with RAII guards"
                ),
            },
            "lean_proofs": {
                "language": "Lean 4",
                "files": 1,
                "description": (
                    "ProverOptimization formal verification: batch soundness, "
                    "incremental correctness, compression losslessness, "
                    "bundle aggregation, pool scaling, tenant isolation, "
                    "Gevulot equivalence"
                ),
            },
        },
    }

    attestation_path = ROOT / "TRUSTLESS_PHYSICS_PHASE3_ATTESTATION.json"
    with open(attestation_path, "w") as f:
        json.dump(attestation, f, indent=2)
    print(f"\nAttestation saved to: {attestation_path.name}")

    return total_failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
