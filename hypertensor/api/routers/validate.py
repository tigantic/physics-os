"""HyperTensor API — Validate endpoint.

POST /v1/validate — Stateless validation of an artifact bundle.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from ...core.evidence import generate_validation_report
from ...core.hasher import content_hash, verify_hash
from ...core.certificates import verify_certificate

router = APIRouter(prefix="/v1", tags=["validate"])


@router.post(
    "/validate",
    summary="Validate an artifact bundle",
    description=(
        "Stateless validation of a result artifact.  Does not require "
        "authentication.  Accepts a result payload and returns a "
        "structured validation report."
    ),
)
async def validate_artifact(body: dict[str, Any]) -> dict[str, Any]:
    """Validate a result or envelope."""

    result: dict[str, Any] = {}

    # If it's a full envelope, validate structure + hashes
    if "envelope_version" in body:
        result["envelope_valid"] = True
        errors: list[str] = []

        # Check required envelope fields
        for field in ["job_id", "job_type", "status", "input_manifest_hash",
                       "artifact_hashes", "versions"]:
            if field not in body:
                errors.append(f"Missing required envelope field: {field}")
                result["envelope_valid"] = False

        # Verify artifact hashes if present
        hashes = body.get("artifact_hashes", {})
        payload_result = body.get("result")
        if payload_result and hashes.get("result"):
            actual = content_hash(payload_result)
            if actual != hashes["result"]:
                errors.append(
                    f"Result hash mismatch: expected {hashes['result']}, got {actual}"
                )
                result["envelope_valid"] = False
            else:
                result["result_hash_verified"] = True

        # Verify certificate signature if present
        certificate = body.get("certificate")
        if certificate:
            sig_valid = verify_certificate(certificate)
            result["certificate_signature_valid"] = sig_valid
            if not sig_valid:
                errors.append("Certificate signature verification failed")

        result["envelope_errors"] = errors

        # Also validate the result payload
        if payload_result:
            domain = payload_result.get("domain", body.get("domain", "unknown"))
            result["validation"] = generate_validation_report(payload_result, domain)

    # If it's a raw result payload, just validate it
    elif "grid" in body or "conservation" in body or "fields" in body:
        domain = body.get("domain", "unknown")
        result["validation"] = generate_validation_report(body, domain)
        result["result_hash"] = content_hash(body)

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "E009",
                "message": (
                    "Unrecognized artifact format.  Expected either a full "
                    "envelope (with 'envelope_version') or a result payload "
                    "(with 'grid', 'conservation', or 'fields')."
                ),
                "retryable": False,
            },
        )

    return result
