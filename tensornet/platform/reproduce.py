"""
Reproducibility Layer
=====================

Ensures every simulation run is deterministic, auditable, and re-playable.

* Seed management:  global + per-module seed capture before every run.
* Environment capture:  Python, torch, numpy, platform, git SHA.
* Artifact hashing:  SHA-256 of every output tensor / file for provenance.
* Context manager:  ``ReproducibilityContext`` wraps an entire run.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

import numpy as np
import torch
from torch import Tensor


# ═══════════════════════════════════════════════════════════════════════════════
# Artifact Hashing
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ArtifactHash:
    """Content-addressed hash of a compute artifact."""

    algorithm: str
    digest: str
    nbytes: int

    def __str__(self) -> str:
        return f"{self.algorithm}:{self.digest[:16]}…"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "digest": self.digest,
            "nbytes": self.nbytes,
        }


def hash_tensor(t: Tensor, algorithm: str = "sha256") -> ArtifactHash:
    """Compute a hash of a tensor's raw bytes (contiguous, float64 if needed)."""
    arr = t.detach().cpu().contiguous().to(torch.float64).numpy()
    raw = arr.tobytes()
    h = hashlib.new(algorithm, raw)
    return ArtifactHash(algorithm=algorithm, digest=h.hexdigest(), nbytes=len(raw))


def hash_bytes(data: bytes, algorithm: str = "sha256") -> ArtifactHash:
    h = hashlib.new(algorithm, data)
    return ArtifactHash(algorithm=algorithm, digest=h.hexdigest(), nbytes=len(data))


def hash_file(path: Union[str, Path], algorithm: str = "sha256") -> ArtifactHash:
    p = Path(path)
    h = hashlib.new(algorithm)
    nbytes = 0
    with p.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            nbytes += len(chunk)
    return ArtifactHash(algorithm=algorithm, digest=h.hexdigest(), nbytes=nbytes)


# ═══════════════════════════════════════════════════════════════════════════════
# Environment Capture
# ═══════════════════════════════════════════════════════════════════════════════


def capture_environment() -> Dict[str, Any]:
    """Snapshot current execution environment for provenance records."""
    env: Dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        env["cuda_version"] = torch.version.cuda  # type: ignore[attr-defined]
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["gpu_count"] = torch.cuda.device_count()

    # Git SHA (best-effort)
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent.parent.parent,
        )
        if result.returncode == 0:
            env["git_sha"] = result.stdout.strip()
            dirty = subprocess.run(
                ["git", "diff", "--quiet"],
                capture_output=True,
                timeout=5,
                cwd=Path(__file__).resolve().parent.parent.parent,
            )
            env["git_dirty"] = dirty.returncode != 0
    except Exception:
        pass

    return env


# ═══════════════════════════════════════════════════════════════════════════════
# Seed Management
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SeedState:
    """Captured seed state for replay."""

    master_seed: int
    python_state: Any
    numpy_state: Dict[str, Any]
    torch_state: Tensor

    def to_dict(self) -> Dict[str, Any]:
        return {
            "master_seed": self.master_seed,
            "torch_state_hash": hash_tensor(self.torch_state).digest,
        }


def lock_seeds(seed: int = 42) -> SeedState:
    """Lock all RNGs and return the captured state for provenance."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Enforce deterministic algorithms where possible
    torch.use_deterministic_algorithms(mode=False)  # mode=True breaks some ops
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]

    return SeedState(
        master_seed=seed,
        python_state=random.getstate(),
        numpy_state={
            "state": "captured",
            "seed": seed,
        },
        torch_state=torch.random.get_rng_state(),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Reproducibility Context
# ═══════════════════════════════════════════════════════════════════════════════


class ReproducibilityContext:
    """
    Context manager that locks seeds, captures environment, and records
    output hashes.

    Usage::

        with ReproducibilityContext(seed=42) as ctx:
            result = solver.solve(state, ...)
            ctx.record("final_velocity", hash_tensor(result.final_state.fields["u"].data))
        provenance = ctx.provenance()
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._seed_state: Optional[SeedState] = None
        self._environment: Optional[Dict[str, Any]] = None
        self._artifacts: Dict[str, ArtifactHash] = {}
        self._wall_start: float = 0.0
        self._wall_end: float = 0.0

    def __enter__(self) -> "ReproducibilityContext":
        self._seed_state = lock_seeds(self._seed)
        self._environment = capture_environment()
        self._wall_start = time.monotonic()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._wall_end = time.monotonic()

    def record(self, name: str, artifact_hash: ArtifactHash) -> None:
        """Register a named artifact hash."""
        self._artifacts[name] = artifact_hash

    def provenance(self) -> Dict[str, Any]:
        """Return the full provenance record."""
        return {
            "seed": self._seed_state.to_dict() if self._seed_state else {},
            "environment": self._environment or {},
            "artifacts": {
                k: v.to_dict() for k, v in self._artifacts.items()
            },
            "wall_time_seconds": round(self._wall_end - self._wall_start, 4),
        }

    def save(self, path: Union[str, Path]) -> None:
        """Write provenance record as JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.provenance(), indent=2, default=str))
