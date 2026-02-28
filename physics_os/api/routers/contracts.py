"""Ontic API — Contract schema endpoint.

GET /v1/contracts/{version} — Return versioned contract schemas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, status

router = APIRouter(prefix="/v1", tags=["contracts"])

_CONTRACTS_DIR = Path(__file__).resolve().parents[3] / "contracts"


@router.get(
    "/contracts/{version}",
    summary="Get contract schemas",
    description="Returns the versioned contract specification and JSON schemas.",
)
async def get_contract(version: str) -> dict[str, Any]:
    version_dir = _CONTRACTS_DIR / version
    if not version_dir.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Contract version '{version}' not found.  Available: {_list_versions()}",
        )

    schemas: dict[str, Any] = {}
    for schema_file in sorted(version_dir.glob("*.schema.json")):
        with open(schema_file) as f:
            schemas[schema_file.stem] = json.load(f)

    spec_text = ""
    spec_file = version_dir / "SPEC.md"
    if spec_file.exists():
        spec_text = spec_file.read_text()

    return {
        "version": version,
        "schemas": schemas,
        "spec": spec_text,
        "available_versions": _list_versions(),
    }


@router.get(
    "/contracts",
    summary="List contract versions",
    description="Returns available contract versions.",
)
async def list_contracts() -> dict[str, Any]:
    return {
        "versions": _list_versions(),
        "current": "v1",
    }


def _list_versions() -> list[str]:
    if not _CONTRACTS_DIR.is_dir():
        return []
    return sorted(
        d.name for d in _CONTRACTS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
