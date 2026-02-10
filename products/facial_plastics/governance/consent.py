"""Informed consent workflow management.

Tracks informed consent for simulation-assisted surgical planning:
  - Consent scope (data use, simulation, reporting)
  - Patient acknowledgements (limitations, not FDA-cleared, etc.)
  - Consent versioning and revocation
  - Regulatory compliance tracking
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ConsentScope(Enum):
    """Types of consent that may be granted."""
    DATA_COLLECTION = "data_collection"
    CT_IMAGING = "ct_imaging"
    PHOTO_CAPTURE = "photo_capture"
    SIMULATION_USE = "simulation_use"
    REPORT_GENERATION = "report_generation"
    OUTCOME_TRACKING = "outcome_tracking"
    RESEARCH_USE = "research_use"
    ANONYMIZED_SHARING = "anonymized_sharing"
    AI_ANALYSIS = "ai_analysis"


class ConsentStatus(Enum):
    """Current state of a consent record."""
    PENDING = "pending"
    GRANTED = "granted"
    REVOKED = "revoked"
    EXPIRED = "expired"


@dataclass
class ConsentRecord:
    """A single consent record."""
    consent_id: str
    case_id: str
    patient_id: str
    scope: ConsentScope
    status: ConsentStatus = ConsentStatus.PENDING
    granted_at: Optional[float] = None
    revoked_at: Optional[float] = None
    expires_at: Optional[float] = None
    version: int = 1
    document_hash: str = ""       # hash of the consent form shown
    witness: str = ""
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        if self.status != ConsentStatus.GRANTED:
            return False
        if self.expires_at is not None and time.time() > self.expires_at:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "consent_id": self.consent_id,
            "case_id": self.case_id,
            "patient_id": self.patient_id,
            "scope": self.scope.value,
            "status": self.status.value,
            "granted_at": self.granted_at,
            "revoked_at": self.revoked_at,
            "expires_at": self.expires_at,
            "version": self.version,
            "document_hash": self.document_hash,
            "witness": self.witness,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ConsentRecord:
        return cls(
            consent_id=d["consent_id"],
            case_id=d["case_id"],
            patient_id=d["patient_id"],
            scope=ConsentScope(d["scope"]),
            status=ConsentStatus(d.get("status", "pending")),
            granted_at=d.get("granted_at"),
            revoked_at=d.get("revoked_at"),
            expires_at=d.get("expires_at"),
            version=d.get("version", 1),
            document_hash=d.get("document_hash", ""),
            witness=d.get("witness", ""),
            notes=d.get("notes", ""),
        )


# ── Standard disclaimers ─────────────────────────────────────────

STANDARD_DISCLAIMERS = [
    "This simulation tool is not FDA-cleared or CE-marked for clinical use.",
    "Simulation results are predictive estimates, not guarantees of surgical outcomes.",
    "Material properties are based on population averages and may differ from individual tissue properties.",
    "Healing predictions are probabilistic and subject to individual biological variation.",
    "The surgeon retains full clinical responsibility for surgical decisions.",
    "This tool is intended as a planning aid only, not as a diagnostic device.",
    "Patient data is processed locally and not transmitted externally unless explicitly consented.",
]


# ── Consent manager ──────────────────────────────────────────────

class ConsentManager:
    """Manage informed consent for simulation-assisted planning.

    Tracks all consent scopes required for a case, ensures
    all required consents are obtained before allowing operations.
    """

    REQUIRED_SCOPES_FOR_SIMULATION = frozenset({
        ConsentScope.DATA_COLLECTION,
        ConsentScope.SIMULATION_USE,
        ConsentScope.REPORT_GENERATION,
    })

    REQUIRED_SCOPES_FOR_OUTCOME_TRACKING = frozenset({
        ConsentScope.OUTCOME_TRACKING,
    })

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        self._records: Dict[str, ConsentRecord] = {}  # consent_id → record
        self._storage_path = storage_path

        if storage_path and storage_path.exists():
            self._load()

    def record_consent(
        self,
        case_id: str,
        patient_id: str,
        scope: ConsentScope,
        *,
        document_text: str = "",
        witness: str = "",
        notes: str = "",
        expires_days: Optional[int] = None,
    ) -> ConsentRecord:
        """Record a new consent grant."""
        now = time.time()
        consent_id = hashlib.sha256(
            f"{case_id}:{patient_id}:{scope.value}:{now}".encode()
        ).hexdigest()[:16]

        expires_at = None
        if expires_days is not None:
            expires_at = now + expires_days * 86400

        doc_hash = ""
        if document_text:
            doc_hash = hashlib.sha256(document_text.encode()).hexdigest()

        record = ConsentRecord(
            consent_id=consent_id,
            case_id=case_id,
            patient_id=patient_id,
            scope=scope,
            status=ConsentStatus.GRANTED,
            granted_at=now,
            expires_at=expires_at,
            document_hash=doc_hash,
            witness=witness,
            notes=notes,
        )

        self._records[consent_id] = record
        self._save()

        logger.info(
            "Consent %s granted: %s for case %s, scope=%s",
            consent_id, patient_id, case_id, scope.value,
        )
        return record

    def revoke_consent(self, consent_id: str, reason: str = "") -> bool:
        """Revoke a previously granted consent."""
        record = self._records.get(consent_id)
        if record is None:
            logger.warning("Consent %s not found for revocation", consent_id)
            return False

        record.status = ConsentStatus.REVOKED
        record.revoked_at = time.time()
        record.notes += f"\nRevoked: {reason}" if reason else "\nRevoked"
        self._save()

        logger.info("Consent %s revoked: %s", consent_id, reason)
        return True

    def check_consent(
        self,
        case_id: str,
        required_scopes: frozenset[ConsentScope],
    ) -> Tuple[bool, List[ConsentScope]]:
        """Check if all required consents are active for a case.

        Returns (all_granted, list_of_missing_scopes).
        """
        active_scopes: set[ConsentScope] = set()
        for record in self._records.values():
            if record.case_id == case_id and record.is_active:
                active_scopes.add(record.scope)

        missing = [s for s in required_scopes if s not in active_scopes]
        return (len(missing) == 0, missing)

    def can_simulate(self, case_id: str) -> Tuple[bool, List[ConsentScope]]:
        """Check if simulation consent requirements are met."""
        return self.check_consent(case_id, self.REQUIRED_SCOPES_FOR_SIMULATION)

    def can_track_outcomes(self, case_id: str) -> Tuple[bool, List[ConsentScope]]:
        """Check if outcome tracking consent is granted."""
        return self.check_consent(
            case_id,
            self.REQUIRED_SCOPES_FOR_SIMULATION | self.REQUIRED_SCOPES_FOR_OUTCOME_TRACKING,
        )

    def get_case_consents(self, case_id: str) -> List[ConsentRecord]:
        """Get all consent records for a case."""
        return [
            r for r in self._records.values()
            if r.case_id == case_id
        ]

    def get_disclaimers(self) -> List[str]:
        """Return standard platform disclaimers."""
        return list(STANDARD_DISCLAIMERS)

    def _save(self) -> None:
        """Persist all records."""
        if self._storage_path is None:
            return
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = [r.to_dict() for r in self._records.values()]
        with open(self._storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def _load(self) -> None:
        """Load records from storage."""
        if self._storage_path is None or not self._storage_path.exists():
            return
        with open(self._storage_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for d in data:
            try:
                record = ConsentRecord.from_dict(d)
                self._records[record.consent_id] = record
            except (KeyError, ValueError) as exc:
                logger.warning("Skipping invalid consent record: %s", exc)
