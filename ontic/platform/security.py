"""
Security & Supply-Chain Auditing — SBOM generation and dependency scanning.

Provides
--------
* ``generate_sbom`` — produce a CycloneDX-lite JSON SBOM of the project's
  Python dependency tree.
* ``audit_dependencies`` — check installed packages against known-vulnerable
  version ranges (offline heuristic; for production use pipe into Trivy/Grype).
* ``license_audit`` — detect potentially incompatible licenses in the dependency
  tree (GPL contamination guard).
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import platform
import sys
import time
from dataclasses import asdict, dataclass, field as dc_field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

__all__ = [
    "generate_sbom",
    "audit_dependencies",
    "license_audit",
    "SBOMEntry",
    "AuditResult",
    "LicenseReport",
]

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SBOM Data Structures
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SBOMEntry:
    """One component in the Software Bill of Materials."""

    name: str
    version: str
    license: str
    author: str
    homepage: str
    purl: str  # Package URL (pkg:pypi/name@version)


@dataclass
class SBOM:
    """CycloneDX-lite SBOM container."""

    bom_format: str = "CycloneDX"
    spec_version: str = "1.5"
    version: int = 1
    metadata: Dict[str, Any] = dc_field(default_factory=dict)
    components: List[SBOMEntry] = dc_field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bomFormat": self.bom_format,
            "specVersion": self.spec_version,
            "version": self.version,
            "metadata": self.metadata,
            "components": [asdict(c) for c in self.components],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SBOM Generation
# ═══════════════════════════════════════════════════════════════════════════════


def generate_sbom(
    *,
    output_path: Optional[Union[str, Path]] = None,
    include_extras: bool = True,
) -> SBOM:
    """
    Generate a Software Bill of Materials from installed Python packages.

    Scans ``importlib.metadata.distributions()`` for all installed packages
    and records name, version, license, author, homepage, and Package URL.

    Parameters
    ----------
    output_path : if set, write the SBOM as JSON to this file.
    include_extras : include all installed packages, not just ontic deps.

    Returns
    -------
    SBOM object.
    """
    entries: List[SBOMEntry] = []

    for dist in sorted(
        importlib.metadata.distributions(),
        key=lambda d: (d.metadata.get("Name") or "").lower(),
    ):
        name = dist.metadata.get("Name", "unknown")
        version = dist.metadata.get("Version", "0.0.0")
        lic = dist.metadata.get("License", "")
        if not lic:
            # try classifier
            classifiers = dist.metadata.get_all("Classifier") or []
            for c in classifiers:
                if "License" in str(c):
                    lic = str(c).split(" :: ")[-1]
                    break
        if not lic:
            lic = "UNKNOWN"

        author = dist.metadata.get("Author", "") or dist.metadata.get("Author-email", "") or ""
        homepage = dist.metadata.get("Home-page", "") or ""
        purl = f"pkg:pypi/{name.lower()}@{version}"

        entries.append(SBOMEntry(
            name=name,
            version=version,
            license=lic,
            author=author,
            homepage=homepage,
            purl=purl,
        ))

    sbom = SBOM(
        metadata={
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tools": [{"vendor": "HyperTensor", "name": "ontic-sbom", "version": "1.0.0"}],
            "component": {
                "name": "HyperTensor-VM",
                "version": "2.0.0",
                "type": "application",
            },
            "python_version": sys.version,
            "platform": platform.platform(),
        },
        components=entries,
    )

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(sbom.to_dict(), indent=2))
        logger.info("SBOM written: %s (%d components)", p, len(entries))

    return sbom


# ═══════════════════════════════════════════════════════════════════════════════
# Dependency Audit
# ═══════════════════════════════════════════════════════════════════════════════

# Known-vulnerable ranges (simplified offline heuristic).
# Production code should integrate with Trivy, Grype, or pip-audit.
_KNOWN_VULNS: Dict[str, List[Dict[str, str]]] = {
    "requests": [
        {"cve": "CVE-2023-32681", "affected": "<2.31.0", "severity": "Medium"},
    ],
    "certifi": [
        {"cve": "CVE-2023-37920", "affected": "<2023.7.22", "severity": "High"},
    ],
    "urllib3": [
        {"cve": "CVE-2023-43804", "affected": "<2.0.6", "severity": "Medium"},
    ],
    "setuptools": [
        {"cve": "CVE-2024-6345", "affected": "<70.0.0", "severity": "High"},
    ],
}


@dataclass(frozen=True)
class VulnFinding:
    """A single vulnerability finding."""

    package: str
    installed_version: str
    cve: str
    affected_range: str
    severity: str


@dataclass
class AuditResult:
    """Result of a dependency audit scan."""

    findings: List[VulnFinding] = dc_field(default_factory=list)
    packages_scanned: int = 0

    @property
    def clean(self) -> bool:
        return len(self.findings) == 0

    def summary(self) -> str:
        if self.clean:
            return f"Audit PASS: {self.packages_scanned} packages scanned, 0 findings."
        lines = [f"Audit FAIL: {len(self.findings)} vulnerability finding(s):"]
        for f in self.findings:
            lines.append(
                f"  {f.package}=={f.installed_version} — {f.cve} "
                f"(affected {f.affected_range}, severity {f.severity})"
            )
        return "\n".join(lines)


def _parse_simple_range(v_str: str, installed: str) -> bool:
    """
    Extremely simplified version-range check.  Only handles ``<X.Y.Z``.
    For production, use ``packaging.version``.
    """
    v_str = v_str.strip()
    if v_str.startswith("<"):
        threshold = v_str[1:]
        inst_parts = [int(p) for p in installed.split(".")[:3]]
        thresh_parts = [int(p) for p in threshold.split(".")[:3]]
        while len(inst_parts) < 3:
            inst_parts.append(0)
        while len(thresh_parts) < 3:
            thresh_parts.append(0)
        return tuple(inst_parts) < tuple(thresh_parts)
    return False


def audit_dependencies() -> AuditResult:
    """
    Check installed packages against a built-in list of known-vulnerable
    version ranges.

    This is an **offline heuristic only**.  For production vulnerability
    management, integrate ``pip-audit``, Trivy, or Grype into CI.

    Returns
    -------
    AuditResult
    """
    result = AuditResult()
    count = 0

    for dist in importlib.metadata.distributions():
        name = (dist.metadata.get("Name") or "").lower()
        version = dist.metadata.get("Version", "0.0.0")
        count += 1

        vulns = _KNOWN_VULNS.get(name, [])
        for v in vulns:
            try:
                if _parse_simple_range(v["affected"], version):
                    result.findings.append(VulnFinding(
                        package=name,
                        installed_version=version,
                        cve=v["cve"],
                        affected_range=v["affected"],
                        severity=v["severity"],
                    ))
            except (ValueError, IndexError):
                pass

    result.packages_scanned = count
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# License Audit
# ═══════════════════════════════════════════════════════════════════════════════

_GPL_MARKERS = {"GPL", "AGPL", "LGPL", "GNU General Public"}
_PERMISSIVE_MARKERS = {"MIT", "Apache", "BSD", "ISC", "PSF", "Python", "MPL"}


@dataclass(frozen=True)
class LicenseEntry:
    """License information for one package."""

    name: str
    version: str
    license: str
    category: str  # 'permissive', 'copyleft', 'unknown'


@dataclass
class LicenseReport:
    """Result of a license compatibility audit."""

    entries: List[LicenseEntry] = dc_field(default_factory=list)
    copyleft: List[LicenseEntry] = dc_field(default_factory=list)
    unknown: List[LicenseEntry] = dc_field(default_factory=list)

    @property
    def clean(self) -> bool:
        return len(self.copyleft) == 0

    def summary(self) -> str:
        lines = [
            f"License audit: {len(self.entries)} packages, "
            f"{len(self.copyleft)} copyleft, {len(self.unknown)} unknown."
        ]
        if self.copyleft:
            lines.append("Copyleft packages (review required):")
            for e in self.copyleft:
                lines.append(f"  {e.name}=={e.version}: {e.license}")
        return "\n".join(lines)


def license_audit() -> LicenseReport:
    """
    Scan installed packages for license compatibility.

    Flags any package with GPL/AGPL/LGPL as ``copyleft`` (potential
    contamination risk).

    Returns
    -------
    LicenseReport
    """
    report = LicenseReport()

    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name", "unknown")
        version = dist.metadata.get("Version", "0.0.0")
        lic = dist.metadata.get("License", "UNKNOWN") or "UNKNOWN"

        # Classify
        lic_upper = lic.upper()
        if any(m.upper() in lic_upper for m in _GPL_MARKERS):
            cat = "copyleft"
        elif any(m.upper() in lic_upper for m in _PERMISSIVE_MARKERS):
            cat = "permissive"
        elif lic_upper in ("UNKNOWN", ""):
            cat = "unknown"
        else:
            cat = "permissive"  # Assume permissive if unrecognised

        entry = LicenseEntry(name=name, version=version, license=lic, category=cat)
        report.entries.append(entry)
        if cat == "copyleft":
            report.copyleft.append(entry)
        elif cat == "unknown":
            report.unknown.append(entry)

    return report
