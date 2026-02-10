"""Case library management — discovery, indexing, and querying."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..core.case_bundle import CaseBundle
from ..core.types import Modality, ProcedureType

logger = logging.getLogger(__name__)


@dataclass
class CaseIndex:
    """Lightweight index entry for a case in the library."""
    case_id: str
    procedure_type: Optional[str] = None
    modalities: List[str] = field(default_factory=list)
    twin_complete: bool = False
    n_runs: int = 0
    quality_level: str = "draft"
    created_utc: str = ""
    updated_utc: str = ""
    root: str = ""


class CaseLibrary:
    """Manages a directory of CaseBundles with indexing and querying.

    The library maintains a JSON index file that caches summary
    information for all cases, enabling fast queries without
    loading each bundle.
    """

    def __init__(self, library_root: str | Path) -> None:
        self._root = Path(library_root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._index_path = self._root / "_library_index.json"
        self._index: Dict[str, CaseIndex] = {}
        if self._index_path.exists():
            self._load_index()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def case_count(self) -> int:
        return len(self._index)

    # ── Index management ──────────────────────────────────────

    def rebuild_index(self) -> int:
        """Scan the library root and rebuild the index from manifests.

        Returns the number of cases indexed.
        """
        self._index.clear()
        count = 0
        for subdir in sorted(self._root.iterdir()):
            if not subdir.is_dir() or subdir.name.startswith("_"):
                continue
            manifest_path = subdir / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                bundle = CaseBundle.load(subdir)
                m = bundle.manifest
                self._index[m.case_id] = CaseIndex(
                    case_id=m.case_id,
                    procedure_type=m.procedure_type,
                    modalities=[
                        a.get("modality", "") for a in m.acquisitions
                    ],
                    twin_complete=m.twin_complete,
                    n_runs=len(m.runs),
                    quality_level=m.quality_level,
                    created_utc=m.created_utc,
                    updated_utc=m.updated_utc,
                    root=str(subdir),
                )
                count += 1
            except Exception as e:
                logger.warning("Failed to index %s: %s", subdir.name, e)

        self._save_index()
        logger.info("Indexed %d cases in %s", count, self._root)
        return count

    def _load_index(self) -> None:
        with open(self._index_path) as f:
            raw = json.load(f)
        self._index = {
            cid: CaseIndex(**entry) for cid, entry in raw.items()
        }

    def _save_index(self) -> None:
        raw = {}
        for cid, entry in self._index.items():
            raw[cid] = {
                "case_id": entry.case_id,
                "procedure_type": entry.procedure_type,
                "modalities": entry.modalities,
                "twin_complete": entry.twin_complete,
                "n_runs": entry.n_runs,
                "quality_level": entry.quality_level,
                "created_utc": entry.created_utc,
                "updated_utc": entry.updated_utc,
                "root": entry.root,
            }
        with open(self._index_path, "w") as f:
            json.dump(raw, f, indent=2)

    def refresh_case(self, case_id: str) -> None:
        """Refresh the index entry for a single case."""
        bundle = self.load_bundle(case_id)
        m = bundle.manifest
        self._index[case_id] = CaseIndex(
            case_id=m.case_id,
            procedure_type=m.procedure_type,
            modalities=[a.get("modality", "") for a in m.acquisitions],
            twin_complete=m.twin_complete,
            n_runs=len(m.runs),
            quality_level=m.quality_level,
            created_utc=m.created_utc,
            updated_utc=m.updated_utc,
            root=str(bundle.root),
        )
        self._save_index()

    # ── CRUD ──────────────────────────────────────────────────

    def create_case(
        self,
        procedure: Optional[ProcedureType] = None,
        case_id: Optional[str] = None,
    ) -> CaseBundle:
        """Create a new case in the library."""
        bundle = CaseBundle.create(
            self._root,
            procedure=procedure,
            case_id=case_id,
        )
        # Update index
        m = bundle.manifest
        self._index[m.case_id] = CaseIndex(
            case_id=m.case_id,
            procedure_type=m.procedure_type,
            twin_complete=False,
            created_utc=m.created_utc,
            updated_utc=m.updated_utc,
            root=str(bundle.root),
        )
        self._save_index()
        logger.info("Created case %s (procedure=%s)", m.case_id, m.procedure_type)
        return bundle

    def load_bundle(self, case_id: str) -> CaseBundle:
        """Load a CaseBundle by ID."""
        if case_id in self._index:
            root = Path(self._index[case_id].root)
        else:
            root = self._root / case_id

        if not root.exists():
            raise FileNotFoundError(f"Case not found: {case_id}")

        return CaseBundle.load(root)

    def list_cases(self) -> List[CaseIndex]:
        """Return all indexed cases."""
        return list(self._index.values())

    def delete_case(self, case_id: str, *, confirm: bool = False) -> None:
        """Delete a case from the library.

        Requires confirm=True to prevent accidental deletion.
        """
        if not confirm:
            raise ValueError("Must pass confirm=True to delete a case")

        entry = self._index.get(case_id)
        if entry is None:
            raise FileNotFoundError(f"Case not in index: {case_id}")

        import shutil
        root = Path(entry.root)
        if root.exists():
            shutil.rmtree(root)

        del self._index[case_id]
        self._save_index()
        logger.info("Deleted case %s", case_id)

    # ── Querying ──────────────────────────────────────────────

    def query(
        self,
        *,
        procedure: Optional[ProcedureType] = None,
        modality: Optional[Modality] = None,
        twin_complete: Optional[bool] = None,
        has_runs: Optional[bool] = None,
        quality_level: Optional[str] = None,
    ) -> List[CaseIndex]:
        """Query cases matching criteria."""
        results = []
        for entry in self._index.values():
            if procedure is not None and entry.procedure_type != procedure.value:
                continue
            if modality is not None and modality.value not in entry.modalities:
                continue
            if twin_complete is not None and entry.twin_complete != twin_complete:
                continue
            if has_runs is not None:
                if has_runs and entry.n_runs == 0:
                    continue
                if not has_runs and entry.n_runs > 0:
                    continue
            if quality_level is not None and entry.quality_level != quality_level:
                continue
            results.append(entry)
        return results

    def statistics(self) -> Dict[str, int]:
        """Return summary statistics about the library."""
        stats: Dict[str, int] = {
            "total_cases": len(self._index),
            "twin_complete": sum(1 for e in self._index.values() if e.twin_complete),
            "with_runs": sum(1 for e in self._index.values() if e.n_runs > 0),
        }
        # Count by procedure
        proc_counts: Dict[str, int] = {}
        for entry in self._index.values():
            p = entry.procedure_type or "unspecified"
            proc_counts[p] = proc_counts.get(p, 0) + 1
        stats.update({f"procedure_{k}": v for k, v in proc_counts.items()})
        return stats

    def __repr__(self) -> str:
        return f"CaseLibrary({self._root}, n_cases={len(self._index)})"
