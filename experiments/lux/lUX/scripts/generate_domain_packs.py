#!/usr/bin/env python3
"""
Generate DomainPack JSON fixtures for all 140 TPC-certified domains.

Reads from certificates/index.json and produces one DomainPack JSON per domain
into the lUX fixture directory. Each pack has domain-specific metrics,
gate packs, viewer bindings, and template configurations derived from the
domain's physics category and conservation laws.

Usage:
    python3 scripts/generate_domain_packs.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
MONOREPO_ROOT = SCRIPT_DIR.parent  # lUX/lUX
REPO_ROOT = MONOREPO_ROOT.parents[1]  # physics-os-main
INDEX_PATH = REPO_ROOT / "certificates" / "index.json"
OUTPUT_DIR = MONOREPO_ROOT / "packages" / "core" / "tests" / "fixtures" / "domain-packs"

# ── Category → metric definitions ────────────────────────────────────────────

_COMMON_METRICS: dict[str, dict[str, Any]] = {
    "conservation_residual": {
        "label": "Conservation Residual",
        "symbol_latex": "r",
        "unit": "dimensionless",
        "format": "scientific",
        "precision": 3,
        "validity_range": [0, 1],
    },
    "l2_drift": {
        "label": "L2 Drift",
        "symbol_latex": "\\|\\Delta\\|_2",
        "unit": "dimensionless",
        "format": "scientific",
        "precision": 3,
        "validity_range": [0, 1],
    },
}

_CONSERVATION_METRICS: dict[str, dict[str, Any]] = {
    "mass": {
        "label": "Total Mass",
        "symbol_latex": "M",
        "unit": "kg",
        "format": "scientific",
        "precision": 6,
    },
    "energy": {
        "label": "Total Energy",
        "symbol_latex": "E",
        "unit": "J",
        "format": "scientific",
        "precision": 4,
    },
    "momentum": {
        "label": "Total Momentum",
        "symbol_latex": "p",
        "unit": "kg·m/s",
        "format": "scientific",
        "precision": 4,
    },
    "charge": {
        "label": "Total Charge",
        "symbol_latex": "Q",
        "unit": "C",
        "format": "scientific",
        "precision": 6,
    },
    "angular_momentum": {
        "label": "Angular Momentum",
        "symbol_latex": "L",
        "unit": "kg·m²/s",
        "format": "scientific",
        "precision": 4,
    },
    "probability": {
        "label": "Probability Norm",
        "symbol_latex": "\\|\\psi\\|^2",
        "unit": "dimensionless",
        "format": "fixed",
        "precision": 8,
        "validity_range": [0.999, 1.001],
    },
    "particle_number": {
        "label": "Particle Number",
        "symbol_latex": "N",
        "unit": "dimensionless",
        "format": "fixed",
        "precision": 4,
    },
    "divergence-free": {
        "label": "Divergence Residual",
        "symbol_latex": "\\nabla \\cdot \\mathbf{v}",
        "unit": "1/s",
        "format": "scientific",
        "precision": 3,
        "validity_range": [0, 0.01],
    },
    "unitarity": {
        "label": "Unitarity Error",
        "symbol_latex": "\\|UU^\\dagger - I\\|",
        "unit": "dimensionless",
        "format": "scientific",
        "precision": 4,
        "validity_range": [0, 1e-8],
    },
    "maximum principle": {
        "label": "Maximum Principle",
        "symbol_latex": "T_{max}",
        "unit": "K",
        "format": "scientific",
        "precision": 4,
    },
    "entropy": {
        "label": "Entropy",
        "symbol_latex": "S",
        "unit": "J/K",
        "format": "scientific",
        "precision": 4,
    },
    "fidelity": {
        "label": "Fidelity",
        "symbol_latex": "F",
        "unit": "dimensionless",
        "format": "fixed",
        "precision": 6,
        "validity_range": [0, 1],
    },
    "trace": {
        "label": "Trace Norm",
        "symbol_latex": "\\mathrm{Tr}(\\rho)",
        "unit": "dimensionless",
        "format": "fixed",
        "precision": 8,
        "validity_range": [0.999, 1.001],
    },
}

# Extra metrics per physics category
_CATEGORY_EXTRAS: dict[str, dict[str, dict[str, Any]]] = {
    "Fluid Dynamics": {
        "velocity_max": {
            "label": "Max Velocity",
            "symbol_latex": "v_{max}",
            "unit": "m/s",
            "format": "scientific",
            "precision": 3,
        },
        "cfl_number": {
            "label": "CFL Number",
            "symbol_latex": "\\mathrm{CFL}",
            "unit": "dimensionless",
            "format": "fixed",
            "precision": 3,
            "validity_range": [0, 1],
        },
    },
    "Thermal Physics": {
        "temperature_range": {
            "label": "Temperature Range",
            "symbol_latex": "\\Delta T",
            "unit": "K",
            "format": "scientific",
            "precision": 3,
        },
    },
    "Electromagnetism": {
        "field_energy": {
            "label": "EM Field Energy",
            "symbol_latex": "U_{EM}",
            "unit": "J",
            "format": "scientific",
            "precision": 4,
        },
    },
    "Plasma Physics": {
        "divB_residual": {
            "label": "∇·B Residual",
            "symbol_latex": "\\nabla \\cdot \\mathbf{B}",
            "unit": "T/m",
            "format": "scientific",
            "precision": 4,
            "validity_range": [0, 1e-10],
        },
    },
    "Quantum Mechanics": {
        "expectation_energy": {
            "label": "⟨H⟩",
            "symbol_latex": "\\langle H \\rangle",
            "unit": "eV",
            "format": "scientific",
            "precision": 6,
        },
    },
    "Quantum Many-Body": {
        "entanglement_entropy": {
            "label": "Entanglement Entropy",
            "symbol_latex": "S_{vN}",
            "unit": "dimensionless",
            "format": "fixed",
            "precision": 4,
        },
    },
    "Electronic Structure": {
        "electron_count": {
            "label": "Electron Count",
            "symbol_latex": "N_e",
            "unit": "dimensionless",
            "format": "fixed",
            "precision": 6,
        },
    },
    "Solid State": {
        "lattice_constant": {
            "label": "Lattice Constant",
            "symbol_latex": "a",
            "unit": "Å",
            "format": "fixed",
            "precision": 4,
        },
    },
    "Nuclear and Particle": {
        "cross_section": {
            "label": "Cross Section",
            "symbol_latex": "\\sigma",
            "unit": "barn",
            "format": "scientific",
            "precision": 4,
        },
    },
    "Astrophysics": {
        "luminosity": {
            "label": "Luminosity",
            "symbol_latex": "L",
            "unit": "L☉",
            "format": "scientific",
            "precision": 4,
        },
    },
    "Geophysics": {
        "nusselt_number": {
            "label": "Nusselt Number",
            "symbol_latex": "\\mathrm{Nu}",
            "unit": "dimensionless",
            "format": "fixed",
            "precision": 2,
        },
    },
    "Optics": {
        "intensity": {
            "label": "Intensity",
            "symbol_latex": "I",
            "unit": "W/m²",
            "format": "scientific",
            "precision": 4,
        },
    },
    "Classical Mechanics": {
        "kinetic_energy": {
            "label": "Kinetic Energy",
            "symbol_latex": "T",
            "unit": "J",
            "format": "scientific",
            "precision": 4,
        },
    },
    "Materials Science": {
        "stress_tensor_norm": {
            "label": "Stress Tensor Norm",
            "symbol_latex": "\\|\\sigma\\|",
            "unit": "Pa",
            "format": "scientific",
            "precision": 4,
        },
    },
    "Chemical Physics": {
        "reaction_rate": {
            "label": "Reaction Rate",
            "symbol_latex": "k",
            "unit": "1/s",
            "format": "scientific",
            "precision": 4,
        },
    },
    "Biophysics": {
        "free_energy": {
            "label": "Free Energy",
            "symbol_latex": "\\Delta G",
            "unit": "kcal/mol",
            "format": "scientific",
            "precision": 3,
        },
    },
    "Quantum Information": {
        "circuit_fidelity": {
            "label": "Circuit Fidelity",
            "symbol_latex": "F_c",
            "unit": "dimensionless",
            "format": "fixed",
            "precision": 6,
            "validity_range": [0, 1],
        },
    },
    "Coupled Multi-Physics": {
        "coupling_residual": {
            "label": "Coupling Residual",
            "symbol_latex": "r_c",
            "unit": "dimensionless",
            "format": "scientific",
            "precision": 3,
        },
    },
    "Statistical Mechanics": {
        "partition_function": {
            "label": "Partition Function",
            "symbol_latex": "Z",
            "unit": "dimensionless",
            "format": "scientific",
            "precision": 4,
        },
    },
    "Computational Methods": {
        "objective_value": {
            "label": "Objective Value",
            "symbol_latex": "f(x)",
            "unit": "dimensionless",
            "format": "scientific",
            "precision": 6,
        },
    },
    "Relativity": {
        "hamiltonian_constraint": {
            "label": "Hamiltonian Constraint",
            "symbol_latex": "\\mathcal{H}",
            "unit": "dimensionless",
            "format": "scientific",
            "precision": 6,
            "validity_range": [0, 1e-8],
        },
    },
    "Special Applied": {
        "application_metric": {
            "label": "Primary Metric",
            "symbol_latex": "\\phi",
            "unit": "dimensionless",
            "format": "scientific",
            "precision": 4,
        },
    },
}

# Standard viewer set — all domains get these
_STANDARD_VIEWERS = [
    {"when": {"artifact_type": "time_series"}, "component": "TimeSeriesViewer", "default_config": {}, "priority": 10},
    {"when": {"artifact_type": "log"}, "component": "LogViewer", "default_config": {}, "priority": 10},
    {"when": {"artifact_type": "table"}, "component": "TableViewer", "default_config": {}, "priority": 10},
]


def _domain_id_to_pack_id(domain_id: str, name: str) -> str:
    """Convert 'II.2' + 'Euler 3D' → 'com.physics.euler_3d'."""
    slug = (
        name.lower()
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace("'", "")
        .replace(".", "")
        .replace("+", "plus")
    )
    # Remove multiple underscores and trailing underscores
    while "__" in slug:
        slug = slug.replace("__", "_")
    slug = slug.strip("_")
    return f"com.physics.{slug}"


def _build_metrics(cert: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build metric definitions from conservation laws + category."""
    metrics: dict[str, dict[str, Any]] = {}
    # Always include common metrics
    metrics.update(_COMMON_METRICS)
    # Add conservation-law-based metrics
    for law in cert.get("conservation_laws", []):
        key = law.lower().replace(" ", "_").replace("-", "_")
        if key in _CONSERVATION_METRICS:
            metrics[key] = _CONSERVATION_METRICS[key]
    # Add category-specific extras
    cat = cert.get("category", "")
    if cat in _CATEGORY_EXTRAS:
        metrics.update(_CATEGORY_EXTRAS[cat])
    return metrics


def _build_domain_pack(cert: dict[str, Any]) -> dict[str, Any]:
    """Build a complete DomainPack JSON for a single certificate entry."""
    metrics = _build_metrics(cert)
    metric_ids = list(metrics.keys())

    # Headline metrics: conservation_residual + l2_drift + first 1-2 domain-specific
    headline = ["conservation_residual", "l2_drift"]
    for mid in metric_ids:
        if mid not in headline and len(headline) < 4:
            headline.append(mid)

    pack: dict[str, Any] = {
        "id": _domain_id_to_pack_id(cert["domain_id"], cert["name"]),
        "version": "1.0.0",
        "metrics": metrics,
        "gate_packs": {
            "audit": {
                "label": "Audit",
                "manifest_ref": "audit-v1",
                "highlight_metrics": headline,
            }
        },
        "viewers": _STANDARD_VIEWERS[:],
        "templates": {
            "executive_summary_metric_ids": headline,
            "publication_sections": ["Abstract", "Methods", "Results", "Integrity"],
            "citation_format": "bibtex",
        },
    }
    return pack


def main() -> None:
    with open(INDEX_PATH) as f:
        data = json.load(f)

    certs = data["certificates"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generated = 0
    for cert in certs:
        pack = _build_domain_pack(cert)
        pack_id = pack["id"]
        out_path = OUTPUT_DIR / f"{pack_id}.json"
        with open(out_path, "w") as f:
            json.dump(pack, f, indent=2)
            f.write("\n")
        generated += 1

    print(f"Generated {generated} domain packs in {OUTPUT_DIR.relative_to(REPO_ROOT)}")

    # Also generate a manifest of all pack IDs
    manifest: dict[str, str] = {}
    for cert in certs:
        pack_id = _domain_id_to_pack_id(cert["domain_id"], cert["name"])
        manifest[cert["domain_id"]] = pack_id

    manifest_path = OUTPUT_DIR / "_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"Manifest written to {manifest_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
