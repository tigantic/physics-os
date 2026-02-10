"""Post-operative outcome data ingestion.

Imports actual surgical outcomes for comparison with predictions:
  - Post-op CT scans (DICOM)
  - Post-op surface scans (OBJ/STL/PLY)
  - Post-op photographs (standardized views)
  - Clinical measurements (landmarks, distances, angles)
  - Patient-reported outcomes (NOSE score, satisfaction)

Timepoints: 1 week, 1 month, 3 months, 6 months, 12 months.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.provenance import hash_file
from ..core.types import (
    ClinicalMeasurement,
    DicomMetadata,
    Landmark,
    LandmarkType,
    Modality,
    SurfaceMesh,
    Vec3,
)

logger = logging.getLogger(__name__)


class OutcomeTimepoint:
    """Standard post-op follow-up timepoints."""
    WEEK_1 = "1_week"
    MONTH_1 = "1_month"
    MONTH_3 = "3_months"
    MONTH_6 = "6_months"
    MONTH_12 = "12_months"
    MONTH_24 = "24_months"

    ALL = [WEEK_1, MONTH_1, MONTH_3, MONTH_6, MONTH_12, MONTH_24]


@dataclass
class PatientReportedOutcome:
    """Patient-reported outcome measurement."""
    instrument: str             # "NOSE", "ROE", "VAS_satisfaction"
    score: float
    max_score: float
    timepoint: str
    timestamp: float = 0.0
    raw_responses: Dict[str, int] = field(default_factory=dict)

    @property
    def normalized_score(self) -> float:
        """Score normalized to 0–100."""
        if self.max_score <= 0:
            return 0.0
        return self.score / self.max_score * 100.0


@dataclass
class OutcomeRecord:
    """Complete outcome record for one follow-up timepoint."""
    case_id: str
    timepoint: str
    collected_at: float = 0.0

    # Imaging data paths
    ct_path: Optional[Path] = None
    surface_scan_path: Optional[Path] = None
    photo_paths: List[Path] = field(default_factory=list)

    # Computed data
    landmarks: Dict[LandmarkType, Vec3] = field(default_factory=dict)
    measurements: List[ClinicalMeasurement] = field(default_factory=list)
    surface_mesh: Optional[SurfaceMesh] = None

    # Patient-reported outcomes
    pro_scores: List[PatientReportedOutcome] = field(default_factory=list)

    # Provenance
    file_hashes: Dict[str, str] = field(default_factory=dict)

    # Metadata
    clinician_notes: str = ""
    complications: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "timepoint": self.timepoint,
            "collected_at": self.collected_at,
            "n_landmarks": len(self.landmarks),
            "n_measurements": len(self.measurements),
            "has_surface_scan": self.surface_mesh is not None,
            "has_ct": self.ct_path is not None,
            "n_photos": len(self.photo_paths),
            "pro_scores": [
                {
                    "instrument": p.instrument,
                    "score": p.score,
                    "normalized": p.normalized_score,
                }
                for p in self.pro_scores
            ],
            "complications": self.complications,
            "file_hashes": self.file_hashes,
        }


class OutcomeIngester:
    """Ingest post-operative outcome data into CaseBundle.

    Handles:
      - DICOM import with hash verification
      - Surface scan import (OBJ/STL/PLY)
      - Photograph import
      - Manual measurement entry
      - Patient-reported outcome questionnaires
    """

    def __init__(self, case_dir: Path) -> None:
        self._case_dir = case_dir
        self._outcomes_dir = case_dir / "outcomes"
        self._outcomes_dir.mkdir(parents=True, exist_ok=True)

    def ingest_surface_scan(
        self,
        scan_path: Path,
        timepoint: str,
    ) -> OutcomeRecord:
        """Import a post-op surface scan."""
        if not scan_path.exists():
            raise FileNotFoundError(f"Surface scan not found: {scan_path}")

        # Compute file hash
        file_hash = hash_file(scan_path)

        # Copy to outcomes directory
        tp_dir = self._outcomes_dir / timepoint
        tp_dir.mkdir(parents=True, exist_ok=True)
        dest = tp_dir / scan_path.name

        if not dest.exists():
            dest.write_bytes(scan_path.read_bytes())

        record = OutcomeRecord(
            case_id=self._case_dir.name,
            timepoint=timepoint,
            collected_at=time.time(),
            surface_scan_path=dest,
            file_hashes={"surface_scan": file_hash},
        )

        # Parse surface mesh
        suffix = scan_path.suffix.lower()
        if suffix in (".obj", ".stl", ".ply"):
            from ..data.surface_ingest import SurfaceIngester
            ingester = SurfaceIngester()
            mesh = ingester.ingest(scan_path)
            record.surface_mesh = mesh

        self._save_record(record)
        logger.info(
            "Ingested surface scan for case %s at %s",
            record.case_id, timepoint,
        )
        return record

    def ingest_photographs(
        self,
        photo_paths: List[Path],
        timepoint: str,
    ) -> OutcomeRecord:
        """Import post-op photographs."""
        tp_dir = self._outcomes_dir / timepoint / "photos"
        tp_dir.mkdir(parents=True, exist_ok=True)

        dest_paths: List[Path] = []
        file_hashes: Dict[str, str] = {}

        for photo in photo_paths:
            if not photo.exists():
                logger.warning("Photo not found: %s", photo)
                continue
            dest = tp_dir / photo.name
            if not dest.exists():
                dest.write_bytes(photo.read_bytes())
            dest_paths.append(dest)
            file_hashes[photo.name] = hash_file(photo)

        record = OutcomeRecord(
            case_id=self._case_dir.name,
            timepoint=timepoint,
            collected_at=time.time(),
            photo_paths=dest_paths,
            file_hashes=file_hashes,
        )

        self._save_record(record)
        logger.info(
            "Ingested %d photos for case %s at %s",
            len(dest_paths), record.case_id, timepoint,
        )
        return record

    def record_measurements(
        self,
        measurements: List[ClinicalMeasurement],
        timepoint: str,
    ) -> OutcomeRecord:
        """Record clinical measurements from a follow-up visit."""
        record = OutcomeRecord(
            case_id=self._case_dir.name,
            timepoint=timepoint,
            collected_at=time.time(),
            measurements=measurements,
        )
        self._save_record(record)
        return record

    def record_landmarks(
        self,
        landmarks: Dict[LandmarkType, Vec3],
        timepoint: str,
    ) -> OutcomeRecord:
        """Record detected or manually placed landmarks."""
        record = OutcomeRecord(
            case_id=self._case_dir.name,
            timepoint=timepoint,
            collected_at=time.time(),
            landmarks=landmarks,
        )
        self._save_record(record)
        return record

    def record_pro(
        self,
        instrument: str,
        score: float,
        max_score: float,
        timepoint: str,
        *,
        raw_responses: Optional[Dict[str, int]] = None,
    ) -> OutcomeRecord:
        """Record a patient-reported outcome score."""
        pro = PatientReportedOutcome(
            instrument=instrument,
            score=score,
            max_score=max_score,
            timepoint=timepoint,
            timestamp=time.time(),
            raw_responses=raw_responses or {},
        )

        record = OutcomeRecord(
            case_id=self._case_dir.name,
            timepoint=timepoint,
            collected_at=time.time(),
            pro_scores=[pro],
        )
        self._save_record(record)
        return record

    def list_outcomes(self) -> List[Dict[str, Any]]:
        """List all outcome records for the case."""
        results: List[Dict[str, Any]] = []
        index_file = self._outcomes_dir / "index.json"
        if index_file.exists():
            with open(index_file, "r", encoding="utf-8") as f:
                results = json.load(f)
        return results

    def _save_record(self, record: OutcomeRecord) -> None:
        """Persist an outcome record to the index."""
        index_file = self._outcomes_dir / "index.json"
        records: List[Dict[str, Any]] = []
        if index_file.exists():
            with open(index_file, "r", encoding="utf-8") as f:
                records = json.load(f)

        records.append(record.to_dict())

        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, default=str)
