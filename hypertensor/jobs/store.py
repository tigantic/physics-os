"""In-memory job store.

Production deployments would swap this for PostgreSQL or Redis.
The interface is deliberately simple so the swap is trivial.
"""

from __future__ import annotations

import threading
from typing import Any

from .models import Job, JobState


class JobStore:
    """Thread-safe in-memory job store.

    Keyed by ``job_id``.  Supports lookup by idempotency key
    to prevent duplicate submissions.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._idempotency: dict[str, str] = {}  # key → job_id
        self._lock = threading.Lock()

    def create(self, job: Job) -> Job:
        """Store a new job.  Returns the stored job."""
        with self._lock:
            self._jobs[job.job_id] = job
            if job.idempotency_key:
                self._idempotency[job.idempotency_key] = job.job_id
        return job

    def get(self, job_id: str) -> Job | None:
        """Get a job by ID.  Returns None if not found."""
        with self._lock:
            return self._jobs.get(job_id)

    def get_by_idempotency_key(self, key: str) -> Job | None:
        """Get a job by idempotency key.  Returns None if not found."""
        with self._lock:
            job_id = self._idempotency.get(key)
            if job_id:
                return self._jobs.get(job_id)
        return None

    def update(self, job: Job) -> None:
        """Update an existing job in the store."""
        with self._lock:
            if job.job_id not in self._jobs:
                raise KeyError(f"Job {job.job_id} not found")
            self._jobs[job.job_id] = job

    def list_jobs(
        self,
        *,
        state: JobState | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Job]:
        """List jobs with optional state filter."""
        with self._lock:
            jobs = list(self._jobs.values())
        if state is not None:
            jobs = [j for j in jobs if j.state == state]
        # Sort by created_at descending
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[offset : offset + limit]

    def count(self) -> int:
        """Total number of jobs in the store."""
        with self._lock:
            return len(self._jobs)

    def clear(self) -> None:
        """Clear all jobs.  For testing only."""
        with self._lock:
            self._jobs.clear()
            self._idempotency.clear()


# Singleton store — shared across all API workers in a single process.
# For multi-process deployments, swap for a shared backend.
store = JobStore()
