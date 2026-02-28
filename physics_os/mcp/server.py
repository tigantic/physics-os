"""Ontic MCP — Model Context Protocol tool server.

Exposes the Ontic Engine Runtime as MCP tools for agent-native
workflows.  Agents call these tools to:

1. Discover available physics domains
2. Submit simulation, validation, and attestation jobs
3. Poll job status
4. Retrieve results, validation reports, and trust certificates

Install the MCP SDK: ``pip install mcp``

Run::

    python -m physics_os.mcp.server

Or configure in ``mcp.json``::

    {
      "mcpServers": {
        "physics_os": {
          "command": "python",
          "args": ["-m", "physics_os.mcp.server"]
        }
      }
    }
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool

    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False

# ── Tool implementations (framework-agnostic) ──────────────────────


def _list_domains() -> dict[str, Any]:
    """List available physics domains."""
    from physics_os.core.registry import list_domains

    domains = list_domains()
    return {"domains": domains, "count": len(domains)}


def _submit_job(
    domain: str,
    job_type: str = "full_pipeline",
    n_bits: int = 8,
    n_steps: int = 100,
    dt: float | None = None,
    max_rank: int = 64,
    truncation_tol: float = 1e-10,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Submit a physics simulation job and run it synchronously."""
    from physics_os.core.executor import ExecutionConfig, execute
    from physics_os.core.sanitizer import sanitize_result
    from physics_os.core.evidence import generate_validation_report, generate_claims
    from physics_os.core.certificates import issue_certificate
    from physics_os.core.hasher import content_hash
    from physics_os.core.registry import DOMAINS

    import physics_os
    import time

    if domain not in DOMAINS:
        return {
            "error": f"Unknown domain '{domain}'.  Available: {', '.join(sorted(DOMAINS))}",
        }

    config = ExecutionConfig(
        domain=domain,
        n_bits=n_bits,
        n_steps=n_steps,
        dt=dt,
        max_rank=max_rank,
        truncation_tol=truncation_tol,
        parameters=parameters,
    )

    result = execute(config)
    if not result.success:
        return {"error": str(result.error), "success": False}

    sanitized = sanitize_result(result, domain)
    output: dict[str, Any] = {
        "domain": domain,
        "success": True,
        "result": sanitized,
    }

    if job_type in ("full_pipeline", "physics_simulation"):
        report = generate_validation_report(sanitized, domain)
        output["validation"] = report

    if job_type == "full_pipeline":
        claims = generate_claims(sanitized, domain)
        result_hash = content_hash(sanitized)
        config_hash = content_hash({
            "domain": domain, "n_bits": n_bits,
            "n_steps": n_steps, "parameters": parameters,
        })
        cert = issue_certificate(
            job_id=f"mcp-{content_hash({'t': time.time()})[:12]}",
            claims=claims,
            input_manifest_hash=config_hash,
            result_hash=result_hash,
            config_hash=config_hash,
            runtime_version=physics_os.RUNTIME_VERSION,
        )
        output["certificate"] = cert

    return output


def _validate_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    """Validate a result artifact or envelope."""
    from physics_os.core.evidence import generate_validation_report
    from physics_os.core.hasher import content_hash
    from physics_os.core.certificates import verify_certificate

    output: dict[str, Any] = {}

    if "envelope_version" in artifact:
        output["type"] = "envelope"
        cert = artifact.get("certificate")
        if cert:
            output["certificate_valid"] = verify_certificate(cert)
        payload = artifact.get("result")
        if payload:
            domain = payload.get("domain", artifact.get("domain", "unknown"))
            output["validation"] = generate_validation_report(payload, domain)
    elif "grid" in artifact or "conservation" in artifact:
        output["type"] = "raw_result"
        domain = artifact.get("domain", "unknown")
        output["validation"] = generate_validation_report(artifact, domain)
        output["content_hash"] = content_hash(artifact)
    else:
        output["error"] = "Unrecognized artifact format"

    return output


def _verify_certificate(certificate: dict[str, Any]) -> dict[str, Any]:
    """Verify a trust certificate's signature."""
    from physics_os.core.certificates import verify_certificate

    valid = verify_certificate(certificate)
    claims = certificate.get("claims", [])
    return {
        "signature_valid": valid,
        "job_id": certificate.get("job_id"),
        "issued_at": certificate.get("issued_at"),
        "claims_satisfied": sum(1 for c in claims if c.get("satisfied")),
        "claims_total": len(claims),
    }


# ── Tool metadata ──────────────────────────────────────────────────

TOOLS: list[dict[str, Any]] = [
    {
        "name": "ontic_list_domains",
        "description": (
            "List all available physics simulation domains supported by "
            "the Ontic Engine runtime.  Returns domain keys, labels, and "
            "accepted parameters."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "ontic_run_simulation",
        "description": (
            "Submit a physics simulation job to the Ontic Engine runtime.  "
            "Runs a QTT-compressed PDE solver for the specified domain.  "
            "Returns the result payload, validation report, and trust "
            "certificate (for full_pipeline jobs)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": (
                        "Physics domain key.  Use ontic_list_domains "
                        "to see available options."
                    ),
                },
                "job_type": {
                    "type": "string",
                    "enum": [
                        "full_pipeline",
                        "physics_simulation",
                        "validation_only",
                        "attestation_only",
                    ],
                    "default": "full_pipeline",
                    "description": "Type of job to run.",
                },
                "n_bits": {
                    "type": "integer",
                    "default": 8,
                    "description": "Grid resolution in bits (2^n points per dimension).",
                },
                "n_steps": {
                    "type": "integer",
                    "default": 100,
                    "description": "Number of time steps.",
                },
                "dt": {
                    "type": "number",
                    "description": "Time step size (optional, auto-computed if omitted).",
                },
                "max_rank": {
                    "type": "integer",
                    "default": 64,
                    "description": "Maximum tensor-train rank.",
                },
                "parameters": {
                    "type": "object",
                    "description": "Domain-specific extra parameters.",
                },
            },
            "required": ["domain"],
        },
    },
    {
        "name": "ontic_validate",
        "description": (
            "Validate a physics simulation result artifact or envelope.  "
            "Checks field integrity, conservation laws, numerical stability, "
            "and certificate signatures."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "artifact": {
                    "type": "object",
                    "description": "The result payload or full envelope to validate.",
                },
            },
            "required": ["artifact"],
        },
    },
    {
        "name": "ontic_verify_certificate",
        "description": (
            "Verify the cryptographic signature on a trust certificate.  "
            "Returns whether the signature is valid and a summary of claims."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "certificate": {
                    "type": "object",
                    "description": "The trust certificate JSON object.",
                },
            },
            "required": ["certificate"],
        },
    },
]


_TOOL_DISPATCH: dict[str, Any] = {
    "ontic_list_domains": lambda args: _list_domains(),
    "ontic_run_simulation": lambda args: _submit_job(**args),
    "ontic_validate": lambda args: _validate_artifact(args["artifact"]),
    "ontic_verify_certificate": lambda args: _verify_certificate(args["certificate"]),
}


def handle_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a tool call by name.  Used by both MCP and fallback modes."""
    handler = _TOOL_DISPATCH.get(name)
    if handler is None:
        return {"error": f"Unknown tool: {name}"}
    return handler(arguments)


# ── MCP Server (uses official SDK when available) ───────────────────


def create_mcp_server() -> Any:
    """Create and configure the MCP server instance.

    Raises ``ImportError`` if the ``mcp`` package is not installed.
    """
    if not _MCP_AVAILABLE:
        raise ImportError(
            "The 'mcp' package is required for the MCP server.  "
            "Install it with: pip install mcp"
        )

    server = Server("physics_os")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"],
            )
            for t in TOOLS
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        result = handle_tool_call(name, arguments)
        return [
            TextContent(
                type="text",
                text=json.dumps(result, indent=2, default=str),
            )
        ]

    return server


async def run_server() -> None:
    """Run the MCP server over stdio."""
    server = create_mcp_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    """Entry point for ``python -m physics_os.mcp.server``."""
    if not _MCP_AVAILABLE:
        print(
            "Error: The 'mcp' package is required.  Install with: pip install mcp",
            file=sys.stderr,
        )
        sys.exit(1)

    import asyncio
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
