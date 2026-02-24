"""HyperTensor API — Capabilities endpoint.

GET /v1/capabilities — Enumerate available physics domains and parameters.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from hypertensor.core.registry import list_domains

router = APIRouter(prefix="/v1", tags=["capabilities"])


@router.get(
    "/capabilities",
    summary="List capabilities",
    description=(
        "Returns available physics domains, accepted parameters, "
        "and result field descriptions.  No execution occurs."
    ),
)
async def get_capabilities() -> dict[str, Any]:
    domains = list_domains()
    return {
        "domain_count": len(domains),
        "domains": domains,
        "job_types": [
            {
                "type": "full_pipeline",
                "description": "Run simulation + validation + attestation in one job.",
            },
            {
                "type": "physics_vm_execution",
                "description": "Run a full physics simulation on the QTT runtime.",
            },
            {
                "type": "rank_atlas_benchmark",
                "description": "Execute a Rank Atlas accuracy benchmark sweep.",
            },
            {
                "type": "rank_atlas_diagnostic",
                "description": "Run Rank Atlas diagnostic analysis.",
            },
            {
                "type": "validation",
                "description": "Validate a previously computed result without re-executing.",
            },
            {
                "type": "attestation",
                "description": "Issue a trust certificate for a validated result.",
            },
        ],
        "contract_version": "v1",
    }
