"""Ontic CLI — Command-line interface.

Usage::

    python -m physics_os.cli run     --domain burgers --n-bits 8
    python -m physics_os.cli validate envelope.json
    python -m physics_os.cli attest  envelope.json
    python -m physics_os.cli verify  certificate.json
    python -m physics_os.cli serve   --host 0.0.0.0 --port 8000

Every command writes structured JSON to stdout.  Human-readable
summaries go to stderr.  This makes it easy to pipe results::

    physics_os run --domain burgers | jq '.conservation'
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

import physics_os

logger = logging.getLogger(__name__)


# ── Utilities ───────────────────────────────────────────────────────


def _stderr(*args: object) -> None:
    """Print to stderr for human-readable output."""
    print(*args, file=sys.stderr, flush=True)


def _json_out(data: Any) -> None:
    """Write data as pretty JSON to stdout."""
    print(json.dumps(data, indent=2, default=str), flush=True)


def _load_json(path: str) -> dict[str, Any]:
    """Load a JSON file.  Exit with code 1 on failure."""
    p = Path(path)
    if not p.exists():
        _stderr(f"Error: file not found: {p}")
        sys.exit(1)
    try:
        with open(p) as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        _stderr(f"Error: invalid JSON in {p}: {exc}")
        sys.exit(1)


# ── Commands ────────────────────────────────────────────────────────


def cmd_run(args: argparse.Namespace) -> None:
    """Execute a physics simulation locally."""
    from physics_os.core.executor import ExecutionConfig, execute
    from physics_os.core.sanitizer import sanitize_result
    from physics_os.core.evidence import generate_validation_report, generate_claims
    from physics_os.core.certificates import issue_certificate
    from physics_os.core.hasher import content_hash
    from physics_os.core.registry import DOMAINS

    domain = args.domain
    if domain not in DOMAINS:
        _stderr(f"Error: unknown domain '{domain}'.  Available: {', '.join(sorted(DOMAINS))}")
        sys.exit(1)

    # Parse extra parameters
    parameters: dict[str, Any] = {}
    if args.param:
        for kv in args.param:
            if "=" not in kv:
                _stderr(f"Error: --param must be key=value, got '{kv}'")
                sys.exit(1)
            k, v = kv.split("=", 1)
            try:
                parameters[k] = json.loads(v)
            except json.JSONDecodeError:
                parameters[k] = v

    config = ExecutionConfig(
        domain=domain,
        n_bits=args.n_bits,
        n_steps=args.n_steps,
        dt=args.dt,
        max_rank=args.max_rank,
        truncation_tol=args.truncation_tol,
        parameters=parameters or None,
    )

    _stderr(f"Running: domain={domain}  n_bits={args.n_bits}  n_steps={args.n_steps}")

    result = execute(config)
    if not result.success:
        _stderr(f"Execution failed: {result.error}")
        _json_out({"success": False, "error": str(result.error)})
        sys.exit(2)

    # Sanitize
    sanitized = sanitize_result(
        result,
        domain,
        precision=args.precision,
        include_fields=not args.no_fields,
    )

    _stderr(f"Completed in {sanitized['performance']['wall_time_s']:.4f}s")

    output: dict[str, Any] = {
        "domain": domain,
        "result": sanitized,
    }

    # Validate
    if not args.no_validate:
        report = generate_validation_report(sanitized, domain)
        output["validation"] = report
        status = "PASS" if report["valid"] else "FAIL"
        _stderr(f"Validation: {status}")

    # Attest
    if args.attest:
        claims = generate_claims(sanitized, domain)
        result_hash = content_hash(sanitized)
        config_hash = content_hash({
            "domain": domain, "n_bits": args.n_bits,
            "n_steps": args.n_steps, "parameters": parameters,
        })
        cert = issue_certificate(
            job_id=f"cli-{content_hash({'t': __import__('time').time()})[:12]}",
            claims=claims,
            input_manifest_hash=config_hash,
            result_hash=result_hash,
            config_hash=config_hash,
            runtime_version=physics_os.RUNTIME_VERSION,
        )
        output["certificate"] = cert
        satisfied = sum(1 for c in claims if c["satisfied"])
        _stderr(f"Certificate: {satisfied}/{len(claims)} claims satisfied")

    # Write output
    if args.output:
        Path(args.output).write_text(json.dumps(output, indent=2, default=str))
        _stderr(f"Written to {args.output}")
    else:
        _json_out(output)


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate a result artifact or envelope."""
    from physics_os.core.evidence import generate_validation_report
    from physics_os.core.hasher import content_hash, verify_hash
    from physics_os.core.certificates import verify_certificate

    data = _load_json(args.file)

    output: dict[str, Any] = {"file": args.file}

    # Full envelope
    if "envelope_version" in data:
        _stderr("Validating envelope...")
        output["type"] = "envelope"

        # Verify artifact hashes
        hashes = data.get("artifact_hashes", {})
        payload_result = data.get("result")
        if payload_result and hashes.get("result"):
            actual = content_hash(payload_result)
            match = actual == hashes["result"]
            output["result_hash_valid"] = match
            _stderr(f"  Result hash: {'MATCH' if match else 'MISMATCH'}")

        # Verify certificate
        cert = data.get("certificate")
        if cert:
            sig_valid = verify_certificate(cert)
            output["certificate_valid"] = sig_valid
            _stderr(f"  Certificate signature: {'VALID' if sig_valid else 'INVALID'}")

        # Validate result payload
        if payload_result:
            domain = payload_result.get("domain", data.get("domain", "unknown"))
            report = generate_validation_report(payload_result, domain)
            output["validation"] = report
            _stderr(f"  Validation: {'PASS' if report['valid'] else 'FAIL'}")

    # Raw result
    elif "grid" in data or "conservation" in data or "fields" in data:
        _stderr("Validating raw result...")
        output["type"] = "raw_result"
        domain = data.get("domain", "unknown")
        report = generate_validation_report(data, domain)
        output["validation"] = report
        output["content_hash"] = content_hash(data)
        _stderr(f"  Validation: {'PASS' if report['valid'] else 'FAIL'}")

    else:
        _stderr("Error: unrecognized artifact format")
        sys.exit(1)

    _json_out(output)


def cmd_attest(args: argparse.Namespace) -> None:
    """Issue a trust certificate for a validated result."""
    from physics_os.core.evidence import generate_claims
    from physics_os.core.certificates import issue_certificate
    from physics_os.core.hasher import content_hash

    data = _load_json(args.file)

    # Extract result payload
    if "result" in data and isinstance(data["result"], dict):
        result_payload = data["result"]
    elif "grid" in data or "conservation" in data:
        result_payload = data
    else:
        _stderr("Error: no result payload found to attest")
        sys.exit(1)

    domain = data.get("domain", result_payload.get("domain", "unknown"))
    claims = generate_claims(result_payload, domain)
    result_hash = content_hash(result_payload)
    config_hash = content_hash(data.get("config", {"domain": domain}))

    cert = issue_certificate(
        job_id=data.get("job_id", f"attest-{result_hash[:12]}"),
        claims=claims,
        input_manifest_hash=config_hash,
        result_hash=result_hash,
        config_hash=config_hash,
        runtime_version=physics_os.RUNTIME_VERSION,
    )

    satisfied = sum(1 for c in claims if c["satisfied"])
    total = len(claims)
    _stderr(f"Certificate issued: {satisfied}/{total} claims satisfied")

    if args.output:
        Path(args.output).write_text(json.dumps(cert, indent=2, default=str))
        _stderr(f"Written to {args.output}")
    else:
        _json_out(cert)


def cmd_verify(args: argparse.Namespace) -> None:
    """Verify a trust certificate's signature."""
    from physics_os.core.certificates import verify_certificate

    cert = _load_json(args.file)

    if "signature" not in cert:
        _stderr("Error: no 'signature' field found — not a certificate")
        sys.exit(1)

    valid = verify_certificate(cert)
    _stderr(f"Signature: {'VALID' if valid else 'INVALID'}")
    _stderr(f"Algorithm: {cert.get('signature', '').split(':')[0]}")
    _stderr(f"Job ID: {cert.get('job_id', 'unknown')}")
    _stderr(f"Issued: {cert.get('issued_at', 'unknown')}")

    claims = cert.get("claims", [])
    for claim in claims:
        tag = claim.get("tag", "?")
        sat = claim.get("satisfied", False)
        _stderr(f"  [{tag}] {'PASS' if sat else 'FAIL'}: {claim.get('claim', '')}")

    _json_out({
        "signature_valid": valid,
        "job_id": cert.get("job_id"),
        "issued_at": cert.get("issued_at"),
        "claims_satisfied": sum(1 for c in claims if c.get("satisfied")),
        "claims_total": len(claims),
    })

    if not valid:
        sys.exit(3)


def cmd_capabilities(args: argparse.Namespace) -> None:
    """List available physics domains."""
    from physics_os.core.registry import list_domains

    domains = list_domains()
    _stderr(f"{len(domains)} domains available:")
    for d in domains:
        _stderr(f"  {d['key']:25s}  {d['label']}")

    _json_out({"domains": domains, "count": len(domains)})


def cmd_problem(args: argparse.Namespace) -> None:
    """Compile and run a high-level physics problem."""
    from physics_os.templates.compiler import compile_problem
    from physics_os.templates.models import (
        BoundarySpec,
        FlowConditions,
        GeometrySpec,
        GeometryType,
        ProblemClass,
        ProblemSpec,
    )
    from physics_os.core.executor import ExecutionConfig, execute
    from physics_os.core.sanitizer import sanitize_result
    from physics_os.core.evidence import generate_validation_report, generate_claims
    from physics_os.core.certificates import issue_certificate
    from physics_os.core.hasher import content_hash

    # ── Parse problem class ─────────────────────────────────────────
    try:
        pc = ProblemClass(args.problem_class)
    except ValueError:
        available = [p.value for p in ProblemClass]
        _stderr(f"Error: unknown problem class '{args.problem_class}'.  Available: {', '.join(available)}")
        sys.exit(1)

    # ── Parse geometry ──────────────────────────────────────────────
    try:
        geo_type = GeometryType(args.shape)
    except ValueError:
        available = [g.value for g in GeometryType]
        _stderr(f"Error: unknown shape '{args.shape}'.  Available: {', '.join(available)}")
        sys.exit(1)

    geo_params: dict[str, Any] = {}
    if args.geo_param:
        for kv in args.geo_param:
            if "=" not in kv:
                _stderr(f"Error: --geo-param must be key=value, got '{kv}'")
                sys.exit(1)
            k, v = kv.split("=", 1)
            try:
                geo_params[k] = json.loads(v)
            except json.JSONDecodeError:
                geo_params[k] = v

    geometry = GeometrySpec(shape=geo_type, params=geo_params)
    flow = FlowConditions(
        velocity=args.velocity,
        fluid=args.fluid,
        temperature=args.temperature,
        pressure=args.pressure,
    )

    boundaries = None
    if args.boundary:
        bd: dict[str, str] = {}
        for kv in args.boundary:
            if "=" not in kv:
                _stderr(f"Error: --boundary must be key=value, got '{kv}'")
                sys.exit(1)
            k, v = kv.split("=", 1)
            bd[k] = v
        boundaries = BoundarySpec(**bd)

    spec = ProblemSpec(
        problem_class=pc,
        geometry=geometry,
        flow=flow,
        boundaries=boundaries,
        quality=args.quality,
        t_end=args.t_end,
        domain_multiplier=args.domain_multiplier,
        max_rank=args.max_rank,
    )

    # ── Compile ─────────────────────────────────────────────────────
    _stderr(f"Compiling: {args.problem_class} / {args.shape} / {args.fluid}")
    try:
        compiled = compile_problem(spec)
    except (ValueError, KeyError) as exc:
        _stderr(f"Compilation error: {exc}")
        sys.exit(1)

    _stderr(
        f"  → domain={compiled.domain}  n_bits={compiled.n_bits}  "
        f"n_steps={compiled.n_steps}  Re={compiled.reynolds_number:.1f}"
    )

    # ── Execute ─────────────────────────────────────────────────────
    config = ExecutionConfig(
        domain=compiled.domain,
        n_bits=compiled.n_bits,
        n_steps=compiled.n_steps,
        dt=compiled.dt,
        max_rank=compiled.max_rank,
        truncation_tol=1e-10,
        parameters=compiled.parameters,
    )

    _stderr(f"Running simulation...")
    result = execute(config)
    if not result.success:
        _stderr(f"Execution failed: {result.error}")
        _json_out({"success": False, "error": str(result.error), "warnings": compiled.warnings})
        sys.exit(2)

    sanitized = sanitize_result(
        result,
        compiled.domain,
        precision=args.precision,
        include_fields=not args.no_fields,
    )

    _stderr(f"Completed in {sanitized['performance']['wall_time_s']:.4f}s")

    output: dict[str, Any] = {
        "problem_class": args.problem_class,
        "domain": compiled.domain,
        "reynolds_number": compiled.reynolds_number,
        "mach_number": compiled.mach_number,
        "characteristic_length": compiled.characteristic_length,
        "fluid_name": compiled.fluid_name,
        "quality_tier": compiled.quality_tier,
        "warnings": compiled.warnings,
        "result": sanitized,
    }

    # Validate
    if not args.no_validate:
        report = generate_validation_report(sanitized, compiled.domain)
        output["validation"] = report
        valid_status = "PASS" if report["valid"] else "FAIL"
        _stderr(f"Validation: {valid_status}")

    # Attest
    if args.attest:
        claims = generate_claims(sanitized, compiled.domain)
        result_hash = content_hash(sanitized)
        config_hash = content_hash(config.merged_parameters)
        cert = issue_certificate(
            job_id=f"cli-problem-{content_hash({'t': __import__('time').time()})[:12]}",
            claims=claims,
            input_manifest_hash=content_hash(spec.model_dump()),
            result_hash=result_hash,
            config_hash=config_hash,
            runtime_version=physics_os.RUNTIME_VERSION,
        )
        output["certificate"] = cert
        satisfied = sum(1 for c in claims if c["satisfied"])
        _stderr(f"Certificate: {satisfied}/{len(claims)} claims satisfied")

    # Write output
    if args.output:
        Path(args.output).write_text(json.dumps(output, indent=2, default=str))
        _stderr(f"Written to {args.output}")
    else:
        _json_out(output)


def cmd_templates(args: argparse.Namespace) -> None:
    """List available problem templates."""
    from physics_os.templates.models import ProblemClass
    from physics_os.templates.registry import TemplateRegistry

    registry = TemplateRegistry()
    templates: list[dict[str, Any]] = []
    for pc in ProblemClass:
        info = registry.get(pc)
        if info is None:
            continue
        templates.append({
            "problem_class": info.problem_class.value,
            "label": info.label,
            "supported_geometries": [g.value for g in info.supported_geometries],
            "default_geometry": info.default_geometry.value,
        })

    _stderr(f"{len(templates)} problem templates available:")
    for t in templates:
        geos = ", ".join(t["supported_geometries"])
        _stderr(f"  {t['problem_class']:25s}  {t['label']:35s}  [{geos}]")

    _json_out({"templates": templates, "count": len(templates)})


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the HTTP API server."""
    import uvicorn

    _stderr(f"Starting Ontic API on {args.host}:{args.port}")
    uvicorn.run(
        "physics_os.api.app:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload,
        workers=1,
    )


# ── Parser ──────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="physics_os",
        description="Ontic Runtime — licensed execution fabric for compression-native compute",
    )
    parser.add_argument(
        "--version", action="version",
        version=f"physics_os {physics_os.__version__}",
    )

    subs = parser.add_subparsers(dest="command", required=True)

    # ── run ──────────────────────────────────────────────────────
    p_run = subs.add_parser("run", help="Execute a physics simulation")
    p_run.add_argument("--domain", required=True, help="Physics domain key")
    p_run.add_argument("--n-bits", type=int, default=8, help="Grid resolution (default: 8)")
    p_run.add_argument("--n-steps", type=int, default=100, help="Time steps (default: 100)")
    p_run.add_argument("--dt", type=float, default=None, help="Time step size")
    p_run.add_argument("--max-rank", type=int, default=64, help="Max TT rank (default: 64)")
    p_run.add_argument("--truncation-tol", type=float, default=1e-10, help="Truncation tolerance")
    p_run.add_argument("--param", action="append", help="Extra param as key=value (repeatable)")
    p_run.add_argument("--precision", type=int, default=8, help="Output decimal precision")
    p_run.add_argument("--no-fields", action="store_true", help="Omit field arrays from output")
    p_run.add_argument("--no-validate", action="store_true", help="Skip validation")
    p_run.add_argument("--attest", action="store_true", help="Issue trust certificate")
    p_run.add_argument("-o", "--output", help="Write result to file instead of stdout")

    # ── validate ─────────────────────────────────────────────────
    p_val = subs.add_parser("validate", help="Validate a result artifact or envelope")
    p_val.add_argument("file", help="Path to JSON artifact or envelope")

    # ── attest ───────────────────────────────────────────────────
    p_att = subs.add_parser("attest", help="Issue a trust certificate")
    p_att.add_argument("file", help="Path to JSON result or envelope")
    p_att.add_argument("-o", "--output", help="Write certificate to file")

    # ── verify ───────────────────────────────────────────────────
    p_ver = subs.add_parser("verify", help="Verify a trust certificate")
    p_ver.add_argument("file", help="Path to JSON certificate")

    # ── capabilities ─────────────────────────────────────────────
    subs.add_parser("capabilities", help="List available physics domains")

    # ── templates ────────────────────────────────────────────────
    subs.add_parser("templates", help="List available problem templates")

    # ── problem ──────────────────────────────────────────────────
    p_prob = subs.add_parser("problem", help="Compile and run a high-level physics problem")
    p_prob.add_argument("--problem-class", required=True, help="Problem type (e.g. external_flow)")
    p_prob.add_argument("--shape", required=True, help="Geometry shape (e.g. circle, naca4)")
    p_prob.add_argument("--velocity", type=float, required=True, help="Free-stream velocity (m/s)")
    p_prob.add_argument("--fluid", default="air", help="Fluid name (default: air)")
    p_prob.add_argument("--temperature", type=float, default=None, help="Temperature (K)")
    p_prob.add_argument("--pressure", type=float, default=None, help="Pressure (Pa)")
    p_prob.add_argument("--quality", default="standard", help="Quality tier: quick, standard, high, maximum")
    p_prob.add_argument("--t-end", type=float, default=None, help="Simulation end time (s)")
    p_prob.add_argument("--domain-multiplier", type=float, default=10.0, help="Domain size multiplier")
    p_prob.add_argument("--max-rank", type=int, default=64, help="Max TT rank (default: 64)")
    p_prob.add_argument("--geo-param", action="append", help="Geometry param as key=value (repeatable)")
    p_prob.add_argument("--boundary", action="append", help="Boundary as key=value (repeatable)")
    p_prob.add_argument("--precision", type=int, default=8, help="Output decimal precision")
    p_prob.add_argument("--no-fields", action="store_true", help="Omit field arrays from output")
    p_prob.add_argument("--no-validate", action="store_true", help="Skip validation")
    p_prob.add_argument("--attest", action="store_true", help="Issue trust certificate")
    p_prob.add_argument("-o", "--output", help="Write result to file instead of stdout")

    # ── serve ────────────────────────────────────────────────────
    p_serve = subs.add_parser("serve", help="Start the HTTP API server")
    p_serve.add_argument("--host", default="0.0.0.0", help="Bind host")
    p_serve.add_argument("--port", type=int, default=8000, help="Bind port")
    p_serve.add_argument("--log-level", default="info", help="Log level")
    p_serve.add_argument("--reload", action="store_true", help="Enable auto-reload")

    return parser


_DISPATCH: dict[str, Any] = {
    "run": cmd_run,
    "validate": cmd_validate,
    "attest": cmd_attest,
    "verify": cmd_verify,
    "capabilities": cmd_capabilities,
    "templates": cmd_templates,
    "problem": cmd_problem,
    "serve": cmd_serve,
}


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = _DISPATCH.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)
    handler(args)


if __name__ == "__main__":
    main()
