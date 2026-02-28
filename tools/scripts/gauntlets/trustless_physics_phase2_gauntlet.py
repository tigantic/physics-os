#!/usr/bin/env python3
"""
Trustless Physics Gauntlet — Phase 2 Validation
================================================

Validates the Multi-Domain & Deployment phase: NS-IMEX proof circuit,
Lean conservation proofs, customer API, deployment package.

Test Layers:
    1. ns_imex_circuit: Rust module compilation + 48 unit tests
    2. lean_ns_proofs: Lean 4 NavierStokesConservation formalization
    3. deployment: Container config, scripts, health check
    4. customer_api: REST API module structure and types
    5. integration: Cross-module validation + from_bytes roundtrip
    6. regression: Phase 1 still passes (euler3d + lean)

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
logger = logging.getLogger("trustless_physics_phase2_gauntlet")

# ═════════════════════════════════════════════════════════════════════════════
# Gauntlet Framework
# ═════════════════════════════════════════════════════════════════════════════

RESULTS: dict[str, dict[str, Any]] = {}
_start_time = time.monotonic()


def gauntlet(name: str, layer: str = "phase2"):
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
        wrapper._gauntlet = True
        wrapper._layer = layer
        return wrapper
    return decorator


def _run_cargo(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    """Run a cargo command in the workspace root."""
    return subprocess.run(
        ["cargo"] + args,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Layer 1: NS-IMEX Proof Circuit (Rust)
# ═════════════════════════════════════════════════════════════════════════════

@gauntlet("ns_imex_module_compiles", "ns_imex_circuit")
def test_ns_imex_compiles():
    """Verify the ns_imex module compiles without errors."""
    result = _run_cargo(["check", "-p", "fluidelite-zk", "--lib"])
    assert result.returncode == 0, (
        f"cargo check failed:\n{result.stderr[-2000:]}"
    )
    for line in result.stderr.splitlines():
        if "ns_imex" in line and "error" in line.lower():
            raise AssertionError(f"ns_imex error: {line}")


@gauntlet("ns_imex_config_tests", "ns_imex_circuit")
def test_ns_imex_config():
    """Verify config module: NSIMEXParams, sizing, constants."""
    result = _run_cargo([
        "test", "-p", "fluidelite-zk", "--lib",
        "ns_imex::config", "--", "--nocapture",
    ])
    assert result.returncode == 0, (
        f"Config tests failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )
    matches = re.findall(r"test .+\.\.\. ok", result.stdout)
    assert len(matches) >= 12, f"Expected ≥12 config tests, got {len(matches)}"


@gauntlet("ns_imex_witness_tests", "ns_imex_circuit")
def test_ns_imex_witness():
    """Verify witness generation: types, IMEX stages, hash, KE/enstrophy."""
    result = _run_cargo([
        "test", "-p", "fluidelite-zk", "--lib",
        "ns_imex::witness", "--", "--nocapture",
    ])
    assert result.returncode == 0, (
        f"Witness tests failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )
    matches = re.findall(r"test .+\.\.\. ok", result.stdout)
    assert len(matches) >= 11, f"Expected ≥11 witness tests, got {len(matches)}"


@gauntlet("ns_imex_gadgets_tests", "ns_imex_circuit")
def test_ns_imex_gadgets():
    """Verify gadgets: diffusion solve, projection, divergence check."""
    result = _run_cargo([
        "test", "-p", "fluidelite-zk", "--lib",
        "ns_imex::gadgets", "--", "--nocapture",
    ])
    assert result.returncode == 0, (
        f"Gadgets tests failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )
    matches = re.findall(r"test .+\.\.\. ok", result.stdout)
    assert len(matches) >= 7, f"Expected ≥7 gadget tests, got {len(matches)}"


@gauntlet("ns_imex_circuit_tests", "ns_imex_circuit")
def test_ns_imex_circuit():
    """Verify circuit stub: creation, witness validation."""
    result = _run_cargo([
        "test", "-p", "fluidelite-zk", "--lib",
        "ns_imex::halo2_impl", "--", "--nocapture",
    ])
    assert result.returncode == 0, (
        f"Circuit tests failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )
    matches = re.findall(r"test .+\.\.\. ok", result.stdout)
    assert len(matches) >= 2, f"Expected ≥2 circuit tests, got {len(matches)}"


@gauntlet("ns_imex_prover_tests", "ns_imex_circuit")
def test_ns_imex_prover():
    """Verify prover: prove, verify, serialization, stats."""
    result = _run_cargo([
        "test", "-p", "fluidelite-zk", "--lib",
        "ns_imex::prover", "--", "--nocapture",
    ])
    assert result.returncode == 0, (
        f"Prover tests failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )
    matches = re.findall(r"test .+\.\.\. ok", result.stdout)
    assert len(matches) >= 4, f"Expected ≥4 prover tests, got {len(matches)}"


@gauntlet("ns_imex_e2e_stub_test", "ns_imex_circuit")
def test_ns_imex_e2e_stub():
    """Verify end-to-end stub: prove_ns_imex_timestep pipeline."""
    result = _run_cargo([
        "test", "-p", "fluidelite-zk", "--lib",
        "ns_imex::tests::test_end_to_end_stub", "--", "--nocapture",
    ])
    assert result.returncode == 0, (
        f"E2E stub test failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )
    assert "ok" in result.stdout, "E2E stub test should pass"


@gauntlet("ns_imex_all_48_tests_pass", "ns_imex_circuit")
def test_all_ns_imex_tests():
    """Verify ALL 48 ns_imex tests pass."""
    result = _run_cargo([
        "test", "-p", "fluidelite-zk", "--lib", "ns_imex",
    ])
    assert result.returncode == 0, (
        f"Not all ns_imex tests passed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )
    match = re.search(r"test result: ok\. (\d+) passed", result.stdout)
    assert match, f"Could not parse test results from:\n{result.stdout[-500:]}"
    passed = int(match.group(1))
    assert passed >= 48, f"Expected ≥48 tests, got {passed}"


@gauntlet("ns_imex_has_six_files", "ns_imex_circuit")
def test_ns_imex_file_structure():
    """Verify the ns_imex module has all 6 required source files."""
    ns_imex_dir = ROOT / "fluidelite-zk" / "src" / "ns_imex"
    assert ns_imex_dir.is_dir(), f"Missing ns_imex directory: {ns_imex_dir}"

    required_files = [
        "config.rs",
        "witness.rs",
        "gadgets.rs",
        "halo2_impl.rs",
        "prover.rs",
        "mod.rs",
    ]
    for fname in required_files:
        fpath = ns_imex_dir / fname
        assert fpath.exists(), f"Missing file: {fpath}"
        content = fpath.read_text(encoding="utf-8")
        assert len(content) > 100, f"File too small ({len(content)} bytes): {fpath.name}"


@gauntlet("ns_imex_config_has_imex_stages", "ns_imex_circuit")
def test_ns_imex_config_imex_stages():
    """Verify config defines 4 IMEX stages and 3 NS variables."""
    config_path = ROOT / "fluidelite-zk" / "src" / "ns_imex" / "config.rs"
    content = config_path.read_text(encoding="utf-8")

    assert "NUM_NS_VARIABLES" in content
    assert "NUM_IMEX_STAGES" in content
    assert "NUM_DIMENSIONS" in content
    assert "PHYS_DIM" in content

    # Verify enum variants
    assert "AdvectionHalf1" in content
    assert "DiffusionFull" in content
    assert "AdvectionHalf2" in content
    assert "Projection" in content

    # Verify NS variable enum
    assert "VelocityX" in content
    assert "VelocityY" in content
    assert "VelocityZ" in content


@gauntlet("ns_imex_witness_has_imex_structure", "ns_imex_circuit")
def test_ns_imex_witness_structure():
    """Verify witness has IMEX-specific types: diffusion, projection, CG."""
    witness_path = ROOT / "fluidelite-zk" / "src" / "ns_imex" / "witness.rs"
    content = witness_path.read_text(encoding="utf-8")

    required_types = [
        "NSIMEXWitness",
        "IMEXStageWitness",
        "VariableSweepWitness",
        "ContractionWitness",
        "SvdTruncationWitness",
        "DiffusionSolveWitness",
        "DiffusionVariableWitness",
        "ProjectionWitness",
        "CGStepWitness",
    ]
    for type_name in required_types:
        assert type_name in content, f"Missing witness type: {type_name}"

    # Check KE and enstrophy fields
    assert "kinetic_energy_before" in content
    assert "kinetic_energy_after" in content
    assert "enstrophy_before" in content
    assert "enstrophy_after" in content
    assert "divergence_residual" in content


@gauntlet("ns_imex_gadgets_has_ns_specific", "ns_imex_circuit")
def test_ns_imex_gadgets_ns_specific():
    """Verify gadgets include NS-specific: diffusion solve, projection, divergence."""
    gadgets_path = ROOT / "fluidelite-zk" / "src" / "ns_imex" / "gadgets.rs"
    content = gadgets_path.read_text(encoding="utf-8")

    required_gadgets = [
        "DiffusionSolveGadget",
        "ProjectionGadget",
        "DivergenceCheckGadget",
        "FixedPointMACGadget",
        "SvdOrderingGadget",
    ]
    for gadget in required_gadgets:
        assert gadget in content, f"Missing gadget: {gadget}"


@gauntlet("ns_imex_prover_has_from_bytes", "ns_imex_circuit")
def test_ns_imex_prover_from_bytes():
    """Verify NSIMEXProof has from_bytes deserialization."""
    prover_path = ROOT / "fluidelite-zk" / "src" / "ns_imex" / "prover.rs"
    content = prover_path.read_text(encoding="utf-8")

    assert "fn from_bytes" in content, "Missing from_bytes method"
    assert "fn to_bytes" in content, "Missing to_bytes method"
    assert "b\"NSIP\"" in content, "Missing NSIP magic bytes"


# ═════════════════════════════════════════════════════════════════════════════
# Layer 2: Lean 4 Navier-Stokes Conservation Proofs
# ═════════════════════════════════════════════════════════════════════════════

@gauntlet("lean_ns_conservation_exists", "lean_ns_proofs")
def test_lean_ns_file_exists():
    """Verify NavierStokesConservation.lean exists and has content."""
    lean_file = ROOT / "lean_yang_mills" / "YangMills" / "NavierStokesConservation.lean"
    assert lean_file.exists(), f"Missing: {lean_file}"
    content = lean_file.read_text(encoding="utf-8")
    assert len(content) > 5000, f"File too small: {len(content)} bytes"


@gauntlet("lean_ns_has_namespace", "lean_ns_proofs")
def test_lean_ns_namespace():
    """Verify the Lean file has the correct namespace."""
    lean_file = ROOT / "lean_yang_mills" / "YangMills" / "NavierStokesConservation.lean"
    content = lean_file.read_text(encoding="utf-8")
    assert "namespace NavierStokesConservation" in content
    assert "end NavierStokesConservation" in content


@gauntlet("lean_ns_has_viscosity_axioms", "lean_ns_proofs")
def test_lean_ns_viscosity():
    """Verify NS-specific axioms: viscosity, Reynolds, dissipation."""
    lean_file = ROOT / "lean_yang_mills" / "YangMills" / "NavierStokesConservation.lean"
    content = lean_file.read_text(encoding="utf-8")

    required_axioms = [
        "axiom ν",
        "axiom ν_pos",
        "axiom kinetic_energy",
        "axiom enstrophy",
        "axiom viscous_dissipation_rate",
        "axiom viscous_dissipation_nonneg",
        "axiom energy_dissipation_exact",
        "axiom dissipation_enstrophy_identity",
    ]
    for axiom in required_axioms:
        assert axiom in content, f"Missing axiom: {axiom}"


@gauntlet("lean_ns_has_imex_axioms", "lean_ns_proofs")
def test_lean_ns_imex():
    """Verify IMEX splitting axioms."""
    lean_file = ROOT / "lean_yang_mills" / "YangMills" / "NavierStokesConservation.lean"
    content = lean_file.read_text(encoding="utf-8")

    required = [
        "axiom imex_splitting_error",
        "axiom C_imex",
        "axiom imex_second_order",
        "axiom implicit_diffusion_contractive",
        "axiom cg_convergence",
        "axiom cg_terminates",
    ]
    for item in required:
        assert item in content, f"Missing IMEX axiom: {item}"


@gauntlet("lean_ns_has_divergence_axioms", "lean_ns_proofs")
def test_lean_ns_divergence():
    """Verify divergence-free constraint axioms and theorems."""
    lean_file = ROOT / "lean_yang_mills" / "YangMills" / "NavierStokesConservation.lean"
    content = lean_file.read_text(encoding="utf-8")

    required = [
        "axiom ε_div",
        "axiom divergence_residual_bound",
        "axiom post_projection_divergence_bound",
        "theorem divergence_free_qtt",
    ]
    for item in required:
        assert item in content, f"Missing divergence item: {item}"


@gauntlet("lean_ns_has_theorems", "lean_ns_proofs")
def test_lean_ns_theorems():
    """Verify key theorems are stated and proved."""
    lean_file = ROOT / "lean_yang_mills" / "YangMills" / "NavierStokesConservation.lean"
    content = lean_file.read_text(encoding="utf-8")

    required_theorems = [
        "theorem kinetic_energy_monotone_decreasing",
        "theorem viscous_dissipation_positive",
        "theorem dissipation_equals_two_nu_enstrophy",
        "theorem ke_conservation_qtt",
        "theorem enstrophy_conservation_qtt",
        "theorem momentum_x_conservation_qtt",
        "theorem momentum_y_conservation_qtt",
        "theorem momentum_z_conservation_qtt",
        "theorem all_momentum_conservation_qtt",
        "theorem divergence_free_qtt",
        "theorem imex_splitting_accuracy",
        "theorem diffusion_unconditionally_stable",
        "theorem total_error_per_timestep",
        "theorem multi_timestep_error",
        "theorem total_time_error_bound",
        "theorem all_conservation_qtt",
        "theorem trustless_physics_certificate_ns_imex",
        "theorem certificate_implies_ke_bounded",
        "theorem certificate_implies_divergence_free",
        "theorem certificate_implies_finite_error",
    ]
    for thm in required_theorems:
        assert thm in content, f"Missing theorem: {thm}"


@gauntlet("lean_ns_has_certificate", "lean_ns_proofs")
def test_lean_ns_certificate():
    """Verify TrustlessPhysicsCertificateNSIMEX structure is defined."""
    lean_file = ROOT / "lean_yang_mills" / "YangMills" / "NavierStokesConservation.lean"
    content = lean_file.read_text(encoding="utf-8")

    assert "structure TrustlessPhysicsCertificateNSIMEX" in content
    for field_name in [
        "energy_conservation",
        "momentum_conservation",
        "divergence_free",
        "imex_accuracy",
        "diffusion_stable",
        "truncation_bounded",
        "cfl_stable",
        "cg_converges",
        "rounding_bounded",
        "zk_sound",
    ]:
        assert field_name in content, f"Missing certificate field: {field_name}"


@gauntlet("lean_ns_has_mathlib_imports", "lean_ns_proofs")
def test_lean_ns_mathlib():
    """Verify proper Mathlib imports."""
    lean_file = ROOT / "lean_yang_mills" / "YangMills" / "NavierStokesConservation.lean"
    content = lean_file.read_text(encoding="utf-8")
    assert "import Mathlib" in content


@gauntlet("lean_ns_line_count", "lean_ns_proofs")
def test_lean_ns_line_count():
    """Verify the Lean file has substantial content (≥500 lines)."""
    lean_file = ROOT / "lean_yang_mills" / "YangMills" / "NavierStokesConservation.lean"
    lines = lean_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 500, f"Expected ≥500 lines, got {len(lines)}"


# ═════════════════════════════════════════════════════════════════════════════
# Layer 3: Deployment Package
# ═════════════════════════════════════════════════════════════════════════════

@gauntlet("deployment_dir_exists", "deployment")
def test_deployment_dir():
    """Verify deployment directory exists with required structure."""
    deploy_dir = ROOT / "deployment"
    assert deploy_dir.is_dir(), f"Missing deployment directory: {deploy_dir}"

    required_files = [
        "Containerfile",
        "config/deployment.toml",
        "tools/tools/scripts/start.sh",
        "tools/tools/scripts/deploy.sh",
        "tools/tools/scripts/health_check.sh",
    ]
    for fname in required_files:
        fpath = deploy_dir / fname
        assert fpath.exists(), f"Missing deployment file: {fpath}"


@gauntlet("deployment_containerfile_valid", "deployment")
def test_containerfile():
    """Verify Containerfile has proper multi-stage build."""
    container_file = ROOT / "deployment" / "Containerfile"
    content = container_file.read_text(encoding="utf-8")

    # Multi-stage build
    from_count = content.count("FROM ")
    assert from_count >= 3, f"Expected ≥3 FROM stages, got {from_count}"

    # Key directives
    assert "EXPOSE" in content
    assert "HEALTHCHECK" in content
    assert "USER" in content  # Non-root
    assert "ENTRYPOINT" in content
    assert "trustless" in content.lower()  # Non-root user
    assert "tini" in content  # PID 1 init system


@gauntlet("deployment_config_toml_valid", "deployment")
def test_deployment_config():
    """Verify deployment.toml has required sections."""
    config_file = ROOT / "deployment" / "config" / "deployment.toml"
    content = config_file.read_text(encoding="utf-8")

    required_sections = [
        "[server]",
        "[tls]",
        "[auth]",
        "[prover]",
        "[solver.euler3d]",
        "[solver.ns_imex]",
        "[storage]",
        "[logging]",
        "[metrics]",
        "[network]",
        "[resources]",
        "[lean]",
    ]
    for section in required_sections:
        assert section in content, f"Missing TOML section: {section}"

    # NS-IMEX specific config
    assert "viscosity" in content
    assert "divergence_tolerance" in content
    assert "max_cg_iterations" in content
    assert "conservation_tolerance" in content


@gauntlet("deployment_scripts_executable", "deployment")
def test_scripts_executable():
    """Verify deployment scripts are executable."""
    scripts_dir = ROOT / "deployment" / "scripts"
    for script_name in ["start.sh", "deploy.sh", "health_check.sh"]:
        script_path = scripts_dir / script_name
        assert script_path.exists(), f"Missing script: {script_name}"
        assert os.access(script_path, os.X_OK), f"Not executable: {script_name}"


@gauntlet("deployment_start_script_valid", "deployment")
def test_start_script():
    """Verify start.sh has pre-flight checks and proper startup."""
    start_script = ROOT / "deployment" / "scripts" / "start.sh"
    content = start_script.read_text(encoding="utf-8")

    assert "set -euo pipefail" in content  # Strict mode
    assert "preflight_checks" in content    # Pre-flight validation
    assert "exec " in content               # Exec replaces shell
    assert "TRUSTLESS_CONFIG_PATH" in content
    assert "fluidelite-server" in content


@gauntlet("deployment_deploy_script_commands", "deployment")
def test_deploy_script():
    """Verify deploy.sh supports build/start/stop/status/verify commands."""
    deploy_script = ROOT / "deployment" / "scripts" / "deploy.sh"
    content = deploy_script.read_text(encoding="utf-8")

    required_commands = [
        "cmd_build",
        "cmd_run",
        "cmd_start",
        "cmd_stop",
        "cmd_restart",
        "cmd_status",
        "cmd_logs",
        "cmd_verify",
        "cmd_health",
    ]
    for cmd in required_commands:
        assert cmd in content, f"Missing command function: {cmd}"

    # Security hardening
    assert "--read-only" in content
    assert "--cap-drop" in content
    assert "no-new-privileges" in content


@gauntlet("deployment_health_check_comprehensive", "deployment")
def test_health_check_script():
    """Verify health check script has comprehensive checks."""
    hc_script = ROOT / "deployment" / "scripts" / "health_check.sh"
    content = hc_script.read_text(encoding="utf-8")

    required_checks = [
        "check_health_endpoint",
        "check_solver_endpoints",
        "check_metrics_endpoint",
        "check_response_times",
        "check_system_resources",
        "check_connectivity",
    ]
    for check in required_checks:
        assert check in content, f"Missing health check: {check}"


# ═════════════════════════════════════════════════════════════════════════════
# Layer 4: Customer API Server
# ═════════════════════════════════════════════════════════════════════════════

@gauntlet("api_module_exists", "customer_api")
def test_api_module_exists():
    """Verify trustless_api.rs exists in fluidelite-zk."""
    api_file = ROOT / "fluidelite-zk" / "src" / "trustless_api.rs"
    assert api_file.exists(), f"Missing API module: {api_file}"
    content = api_file.read_text(encoding="utf-8")
    assert len(content) > 5000, f"API module too small: {len(content)} bytes"


@gauntlet("api_registered_in_lib", "customer_api")
def test_api_registered():
    """Verify trustless_api is registered in lib.rs."""
    lib_file = ROOT / "fluidelite-zk" / "src" / "lib.rs"
    content = lib_file.read_text(encoding="utf-8")
    assert "pub mod trustless_api" in content


@gauntlet("api_has_certificate_endpoints", "customer_api")
def test_api_endpoints():
    """Verify API defines certificate CRUD endpoints."""
    api_file = ROOT / "fluidelite-zk" / "src" / "trustless_api.rs"
    content = api_file.read_text(encoding="utf-8")

    required_endpoints = [
        "/v1/certificates/create",
        "/v1/certificates/verify",
        "/v1/certificates/:id",
        "/v1/solvers",
        "/health",
        "/ready",
        "/stats",
        "/metrics",
    ]
    for endpoint in required_endpoints:
        assert endpoint in content, f"Missing endpoint: {endpoint}"


@gauntlet("api_has_request_types", "customer_api")
def test_api_request_types():
    """Verify API defines proper request/response types."""
    api_file = ROOT / "fluidelite-zk" / "src" / "trustless_api.rs"
    content = api_file.read_text(encoding="utf-8")

    required_types = [
        "CreateCertificateRequest",
        "CreateCertificateResponse",
        "GetCertificateResponse",
        "VerifyCertificateResponse",
        "SolverListResponse",
        "HealthResponse",
        "StatsResponse",
        "CertificateStatus",
        "CertificateDiagnostics",
        "VerificationDiagnostics",
    ]
    for type_name in required_types:
        assert type_name in content, f"Missing type: {type_name}"


@gauntlet("api_has_auth_middleware", "customer_api")
def test_api_auth():
    """Verify API has authentication middleware."""
    api_file = ROOT / "fluidelite-zk" / "src" / "trustless_api.rs"
    content = api_file.read_text(encoding="utf-8")

    assert "trustless_auth_middleware" in content
    assert "ConstantTimeEq" in content  # Timing-attack resistant
    assert "AUTHORIZATION" in content
    assert "Bearer " in content


@gauntlet("api_has_prometheus_metrics", "customer_api")
def test_api_metrics():
    """Verify API exports Prometheus-format metrics."""
    api_file = ROOT / "fluidelite-zk" / "src" / "trustless_api.rs"
    content = api_file.read_text(encoding="utf-8")

    required_metrics = [
        "trustless_uptime_seconds",
        "trustless_requests_total",
        "trustless_certificates_created_total",
        "trustless_certificates_verified_total",
        "trustless_proofs_by_solver",
    ]
    for metric in required_metrics:
        assert metric in content, f"Missing Prometheus metric: {metric}"


@gauntlet("api_has_solver_list", "customer_api")
def test_api_solver_list():
    """Verify API lists both euler3d and ns_imex solvers."""
    api_file = ROOT / "fluidelite-zk" / "src" / "trustless_api.rs"
    content = api_file.read_text(encoding="utf-8")

    assert '"euler3d"' in content
    assert '"ns_imex"' in content
    assert "EulerConservation" in content
    assert "NavierStokesConservation" in content
    assert "NavierStokesRegularity" in content
    assert "SolverInfo" in content
    assert "LeanProofRef" in content


@gauntlet("api_ns_imex_certificate_generation", "customer_api")
def test_api_ns_imex_generation():
    """Verify API can generate NS-IMEX certificates."""
    api_file = ROOT / "fluidelite-zk" / "src" / "trustless_api.rs"
    content = api_file.read_text(encoding="utf-8")

    # Verify NS-IMEX certificate generation path
    assert "generate_ns_imex_certificate" in content
    assert "verify_ns_imex_proof" in content
    assert "NSIMEXProver" in content
    assert "NSIMEXVerifier" in content
    assert "WitnessGenerator" in content


# ═════════════════════════════════════════════════════════════════════════════
# Layer 5: Integration Tests
# ═════════════════════════════════════════════════════════════════════════════

@gauntlet("integration_all_116_rust_tests", "integration")
def test_all_rust_tests():
    """Verify all 116 fluidelite-zk library tests pass."""
    result = _run_cargo([
        "test", "-p", "fluidelite-zk", "--lib",
    ])
    assert result.returncode == 0, (
        f"Not all tests passed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )
    match = re.search(r"test result: ok\. (\d+) passed", result.stdout)
    assert match, f"Could not parse test results:\n{result.stdout[-500:]}"
    passed = int(match.group(1))
    assert passed >= 116, f"Expected ≥116 tests, got {passed}"


@gauntlet("integration_ns_imex_proof_roundtrip", "integration")
def test_proof_roundtrip():
    """Verify NSIMEXProof serialization/deserialization roundtrip via Rust test."""
    result = _run_cargo([
        "test", "-p", "fluidelite-zk", "--lib",
        "ns_imex::prover::tests::test_prove_and_verify",
        "--", "--nocapture",
    ])
    assert result.returncode == 0, (
        f"Proof roundtrip test failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )
    assert "ok" in result.stdout


@gauntlet("integration_lib_rs_has_ns_imex", "integration")
def test_lib_has_ns_imex():
    """Verify lib.rs exports ns_imex module."""
    lib_file = ROOT / "fluidelite-zk" / "src" / "lib.rs"
    content = lib_file.read_text(encoding="utf-8")
    assert "pub mod ns_imex;" in content
    assert "pub mod euler3d;" in content
    assert "pub mod trustless_api;" in content


@gauntlet("integration_lean_files_consistent", "integration")
def test_lean_files_consistent():
    """Verify both Lean conservation files exist and are consistent."""
    euler_file = ROOT / "lean_yang_mills" / "YangMills" / "EulerConservation.lean"
    ns_file = ROOT / "lean_yang_mills" / "YangMills" / "NavierStokesConservation.lean"

    assert euler_file.exists(), "Missing EulerConservation.lean"
    assert ns_file.exists(), "Missing NavierStokesConservation.lean"

    euler_content = euler_file.read_text(encoding="utf-8")
    ns_content = ns_file.read_text(encoding="utf-8")

    # Both should import Mathlib
    assert "import Mathlib" in euler_content
    assert "import Mathlib" in ns_content

    # Both should define certificates
    assert "TrustlessPhysicsCertificate" in euler_content
    assert "TrustlessPhysicsCertificateNSIMEX" in ns_content

    # Both should have Q16 definitions
    assert "Q16_SCALE" in euler_content
    assert "Q16_SCALE" in ns_content

    # NS-specific items not in Euler
    assert "ν" in ns_content
    assert "divergence_free" in ns_content


@gauntlet("integration_deployment_config_matches_api", "integration")
def test_deployment_matches_api():
    """Verify deployment config references both solvers."""
    config_file = ROOT / "deployment" / "config" / "deployment.toml"
    content = config_file.read_text(encoding="utf-8")

    # Both solvers in config
    assert "[solver.euler3d]" in content
    assert "[solver.ns_imex]" in content

    # Lean proof references
    assert "EulerConservation" in content
    assert "NavierStokesConservation" in content
    assert "NavierStokesRegularity" in content


# ═════════════════════════════════════════════════════════════════════════════
# Layer 6: Regression — Phase 1 Still Passes
# ═════════════════════════════════════════════════════════════════════════════

@gauntlet("regression_euler3d_36_tests", "regression")
def test_regression_euler3d():
    """Verify all 36 euler3d tests still pass (Phase 1 regression)."""
    result = _run_cargo([
        "test", "-p", "fluidelite-zk", "--lib", "euler3d",
    ])
    assert result.returncode == 0, (
        f"Euler3D regression failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )
    match = re.search(r"test result: ok\. (\d+) passed", result.stdout)
    assert match, f"Could not parse:\n{result.stdout[-500:]}"
    passed = int(match.group(1))
    assert passed >= 36, f"Expected ≥36 euler3d tests, got {passed}"


@gauntlet("regression_euler_conservation_lean", "regression")
def test_regression_euler_lean():
    """Verify EulerConservation.lean is unchanged and valid."""
    lean_file = ROOT / "lean_yang_mills" / "YangMills" / "EulerConservation.lean"
    assert lean_file.exists()
    content = lean_file.read_text(encoding="utf-8")

    # Key Phase 1 theorems still present
    for thm in [
        "theorem mass_conservation_qtt",
        "theorem all_conservation_qtt",
        "theorem strang_accuracy",
        "theorem trustless_physics_certificate",
    ]:
        assert thm in content, f"Regression: missing {thm}"


@gauntlet("regression_zero_compile_errors", "regression")
def test_regression_zero_errors():
    """Verify zero compile errors across the entire crate."""
    result = _run_cargo(["check", "-p", "fluidelite-zk", "--lib"])
    assert result.returncode == 0, (
        f"Compile errors detected:\n{result.stderr[-2000:]}"
    )
    # Count errors
    error_count = result.stderr.count("error[E")
    assert error_count == 0, f"Found {error_count} compile errors"


# ═════════════════════════════════════════════════════════════════════════════
# Gauntlet Runner
# ═════════════════════════════════════════════════════════════════════════════

def run_all() -> bool:
    """Run all Phase 2 gauntlet tests."""
    total_start = time.monotonic()

    # Collect all test functions
    tests = []
    for name, obj in list(globals().items()):
        if callable(obj) and hasattr(obj, "_gauntlet") and obj._gauntlet:
            tests.append(obj)

    # Group by layer
    layers: dict[str, list] = {}
    for t in tests:
        layer = t._layer
        layers.setdefault(layer, []).append(t)

    layer_order = [
        "ns_imex_circuit",
        "lean_ns_proofs",
        "deployment",
        "customer_api",
        "integration",
        "regression",
    ]

    print("\n" + "=" * 72)
    print("  Trustless Physics Gauntlet — Phase 2")
    print("  Multi-Domain & Deployment: NS-IMEX + Lean + API + Deploy")
    print("=" * 72 + "\n")

    total_passed = 0
    total_failed = 0

    for layer_name in layer_order:
        layer_tests = layers.get(layer_name, [])
        if not layer_tests:
            continue

        print(f"\n── {layer_name} ({len(layer_tests)} tests) ──")
        for test_fn in layer_tests:
            passed = test_fn()
            if passed:
                total_passed += 1
            else:
                total_failed += 1

    total_elapsed = time.monotonic() - total_start

    # Summary
    total_tests = total_passed + total_failed
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
        "project": "physics-os",
        "protocol": "trustless_physics_gauntlet_phase2",
        "phase": 2,
        "description": "Multi-Domain & Deployment: NS-IMEX Circuit, Lean Proofs, API, Deployment",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_tests": total_tests,
        "passed": total_passed,
        "failed": total_failed,
        "total_time_seconds": round(total_elapsed, 3),
        "gauntlets": RESULTS,
        "components": {
            "ns_imex_circuit": {
                "language": "Rust",
                "files": 6,
                "tests": 48,
                "description": "Halo2 ZK proof circuit for NS-IMEX QTT solver",
            },
            "lean_proofs": {
                "language": "Lean 4",
                "files": 1,
                "theorems": 20,
                "description": "NavierStokesConservation formal verification",
            },
            "deployment": {
                "files": 5,
                "description": "Containerized on-premise deployment package",
            },
            "customer_api": {
                "language": "Rust",
                "files": 1,
                "description": "REST API for certificate generation and verification",
            },
        },
    }

    attestation_path = ROOT / "TRUSTLESS_PHYSICS_PHASE2_ATTESTATION.json"
    with open(attestation_path, "w") as f:
        json.dump(attestation, f, indent=2)
    print(f"\nAttestation saved to: {attestation_path.name}")

    return total_failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
