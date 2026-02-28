"""
6.7 — Experiment Tracking
=========================

MLflow / Weights-&-Biases style experiment tracker implemented purely
in-process with JSON persistence.  Tracks runs, metrics, parameters,
artifacts and supports comparison / leaderboard queries.

Components:
    * Run           — single experiment run with metric time-series
    * Experiment    — group of runs sharing a hypothesis
    * ExperimentTracker — top-level API (create / log / query / compare)
"""

from __future__ import annotations

import copy
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ── Run / Experiment data model ──────────────────────────────────

class RunStatus(Enum):
    CREATED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class MetricEntry:
    """One metric value at a given step."""
    name: str
    value: float
    step: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class Run:
    """Single experiment run."""
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    experiment_id: str = ""
    name: str = ""
    status: RunStatus = RunStatus.CREATED
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, List[MetricEntry]] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    def log_param(self, key: str, value: Any) -> None:
        self.params[key] = value

    def log_metric(self, name: str, value: float, step: int = 0) -> None:
        entry = MetricEntry(name=name, value=value, step=step)
        self.metrics.setdefault(name, []).append(entry)

    def log_artifact(self, path: str) -> None:
        self.artifacts.append(path)

    def get_metric_history(self, name: str) -> List[MetricEntry]:
        return self.metrics.get(name, [])

    def best_metric(self, name: str, mode: str = "min") -> Optional[MetricEntry]:
        history = self.metrics.get(name, [])
        if not history:
            return None
        if mode == "min":
            return min(history, key=lambda e: e.value)
        return max(history, key=lambda e: e.value)

    def duration(self) -> float:
        if self.end_time > 0:
            return self.end_time - self.start_time
        if self.start_time > 0:
            return time.time() - self.start_time
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "name": self.name,
            "status": self.status.name,
            "params": self.params,
            "metrics": {
                k: [{"name": e.name, "value": e.value,
                      "step": e.step, "timestamp": e.timestamp}
                     for e in v]
                for k, v in self.metrics.items()
            },
            "tags": self.tags,
            "artifacts": self.artifacts,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Run":
        run = cls(
            run_id=d["run_id"],
            experiment_id=d.get("experiment_id", ""),
            name=d.get("name", ""),
            status=RunStatus[d["status"]],
            params=d.get("params", {}),
            tags=d.get("tags", {}),
            artifacts=d.get("artifacts", []),
            start_time=d.get("start_time", 0.0),
            end_time=d.get("end_time", 0.0),
        )
        for k, entries in d.get("metrics", {}).items():
            run.metrics[k] = [
                MetricEntry(
                    name=e["name"], value=e["value"],
                    step=e["step"], timestamp=e.get("timestamp", 0.0),
                )
                for e in entries
            ]
        return run


@dataclass
class Experiment:
    """Group of runs sharing a hypothesis."""
    experiment_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    description: str = ""
    runs: Dict[str, Run] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def add_run(self, run: Run) -> None:
        run.experiment_id = self.experiment_id
        self.runs[run.run_id] = run

    def list_runs(
        self,
        status: Optional[RunStatus] = None,
    ) -> List[Run]:
        results = list(self.runs.values())
        if status is not None:
            results = [r for r in results if r.status == status]
        return results

    def leaderboard(
        self,
        metric: str,
        mode: str = "min",
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Rank runs by best metric value."""
        scores: List[Tuple[str, float]] = []
        for run in self.runs.values():
            best = run.best_metric(metric, mode)
            if best is not None:
                scores.append((run.run_id, best.value))
        reverse = mode == "max"
        scores.sort(key=lambda x: x[1], reverse=reverse)
        return scores[:top_k]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "created_at": self.created_at,
            "runs": {k: v.to_dict() for k, v in self.runs.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Experiment":
        exp = cls(
            experiment_id=d["experiment_id"],
            name=d.get("name", ""),
            description=d.get("description", ""),
            tags=d.get("tags", {}),
            created_at=d.get("created_at", 0.0),
        )
        for k, v in d.get("runs", {}).items():
            exp.runs[k] = Run.from_dict(v)
        return exp


# ── Experiment tracker ───────────────────────────────────────────

class ExperimentTracker:
    """Top-level API for experiment management."""

    def __init__(self, store_dir: Optional[Path] = None) -> None:
        self._experiments: Dict[str, Experiment] = {}
        self._active_run: Optional[Run] = None
        self.store_dir = Path(store_dir) if store_dir else None

    # ── Experiment management ─────────────────────────────────

    def create_experiment(
        self,
        name: str,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> Experiment:
        exp = Experiment(name=name, description=description, tags=tags or {})
        self._experiments[exp.experiment_id] = exp
        return exp

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        return self._experiments.get(experiment_id)

    def list_experiments(self) -> List[Experiment]:
        return list(self._experiments.values())

    # ── Run management ────────────────────────────────────────

    def start_run(
        self,
        experiment_id: str,
        name: str = "",
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Run:
        exp = self._experiments.get(experiment_id)
        if exp is None:
            raise KeyError(f"Experiment {experiment_id!r} not found")
        run = Run(name=name, params=params or {}, tags=tags or {})
        run.status = RunStatus.RUNNING
        run.start_time = time.time()
        exp.add_run(run)
        self._active_run = run
        return run

    def end_run(
        self,
        run: Optional[Run] = None,
        status: RunStatus = RunStatus.COMPLETED,
    ) -> None:
        r = run or self._active_run
        if r is None:
            return
        r.status = status
        r.end_time = time.time()
        if r is self._active_run:
            self._active_run = None

    @property
    def active_run(self) -> Optional[Run]:
        return self._active_run

    # ── Logging shortcuts ─────────────────────────────────────

    def log_param(self, key: str, value: Any, run: Optional[Run] = None) -> None:
        r = run or self._active_run
        if r:
            r.log_param(key, value)

    def log_metric(
        self, name: str, value: float, step: int = 0,
        run: Optional[Run] = None,
    ) -> None:
        r = run or self._active_run
        if r:
            r.log_metric(name, value, step)

    def log_artifact(self, path: str, run: Optional[Run] = None) -> None:
        r = run or self._active_run
        if r:
            r.log_artifact(path)

    # ── Comparison ────────────────────────────────────────────

    def compare_runs(
        self,
        run_ids: List[str],
        metric: str,
    ) -> List[Tuple[str, Optional[float]]]:
        """Return (run_id, best_value) for each run."""
        results: List[Tuple[str, Optional[float]]] = []
        for rid in run_ids:
            found = self._find_run(rid)
            if found:
                best = found.best_metric(metric)
                results.append((rid, best.value if best else None))
            else:
                results.append((rid, None))
        return results

    def global_leaderboard(
        self,
        metric: str,
        mode: str = "min",
        top_k: int = 10,
    ) -> List[Tuple[str, str, float]]:
        """Rank all runs across all experiments."""
        scores: List[Tuple[str, str, float]] = []
        for exp in self._experiments.values():
            for run in exp.runs.values():
                best = run.best_metric(metric, mode)
                if best:
                    scores.append((exp.experiment_id, run.run_id, best.value))
        reverse = mode == "max"
        scores.sort(key=lambda x: x[2], reverse=reverse)
        return scores[:top_k]

    # ── Persistence ───────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> None:
        p = path or (self.store_dir / "tracker.json" if self.store_dir else None)
        if p is None:
            return
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        data = {k: v.to_dict() for k, v in self._experiments.items()}
        Path(p).write_text(json.dumps(data, indent=2))

    def load(self, path: Optional[Path] = None) -> int:
        p = path or (self.store_dir / "tracker.json" if self.store_dir else None)
        if p is None or not Path(p).exists():
            return 0
        data = json.loads(Path(p).read_text())
        for k, v in data.items():
            self._experiments[k] = Experiment.from_dict(v)
        return len(data)

    # ── Internal ──────────────────────────────────────────────

    def _find_run(self, run_id: str) -> Optional[Run]:
        for exp in self._experiments.values():
            if run_id in exp.runs:
                return exp.runs[run_id]
        return None


__all__ = [
    "RunStatus",
    "MetricEntry",
    "Run",
    "Experiment",
    "ExperimentTracker",
]
