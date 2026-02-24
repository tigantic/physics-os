"""Concurrent burst test — G4.5.

Verifies that submitting many jobs simultaneously produces no
data-corruption, deadlocks, or lost writes in the ``JobStore``.

Per QUEUE_BEHAVIOR_SPEC.md §4.3, the ``JobStore`` is thread-safe
for concurrent access from async handlers within a single process.

This test exercises:
    • Parallel ``create()`` calls — no duplicate job IDs
    • Parallel ``update()`` calls — no lost writes
    • Parallel ``get()`` calls under write pressure — no exceptions
    • Idempotency key lookup under contention — correct mapping
    • Count consistency after burst — ``store.count() == N``
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pytest

from hypertensor.jobs.models import Job, JobInput, JobState, JobType
from hypertensor.jobs.store import JobStore

# ── Helpers ─────────────────────────────────────────────────────────

BURST_SIZE = 50  # Number of concurrent jobs
MAX_WORKERS = 16  # Thread pool width


def _make_job(
    *,
    idempotency_key: str | None = None,
    domain: str = "burgers",
) -> Job:
    """Create a minimal Job for testing."""
    return Job(
        job_id=str(uuid.uuid4()),
        job_type=JobType.FULL_PIPELINE,
        input=JobInput(
            job_type=JobType.FULL_PIPELINE,
            domain=domain,
            n_bits=8,
            n_steps=100,
        ),
        idempotency_key=idempotency_key,
    )


# ════════════════════════════════════════════════════════════════════
# Tests
# ════════════════════════════════════════════════════════════════════


class TestConcurrentBurst:
    """G4.5 — concurrent burst causes no corruption."""

    def setup_method(self) -> None:
        self.store = JobStore()

    # ── Parallel creates ────────────────────────────────────────────

    def test_parallel_creates_no_duplicates(self) -> None:
        """All jobs created in a burst must appear in the store with
        distinct IDs and correct count."""
        jobs = [_make_job() for _ in range(BURST_SIZE)]
        errors: list[str] = []

        def _create(j: Job) -> str:
            self.store.create(j)
            return j.job_id

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_create, j): j for j in jobs}
            ids = set()
            for f in as_completed(futures):
                try:
                    ids.add(f.result())
                except Exception as exc:
                    errors.append(str(exc))

        assert not errors, f"Errors during parallel create: {errors}"
        assert len(ids) == BURST_SIZE, (
            f"Expected {BURST_SIZE} unique IDs, got {len(ids)}"
        )
        assert self.store.count() == BURST_SIZE

    # ── Parallel transitions ────────────────────────────────────────

    def test_parallel_state_transitions(self) -> None:
        """Each job transitions QUEUED → RUNNING → SUCCEEDED
        under concurrent pressure without corruption."""
        jobs = [_make_job() for _ in range(BURST_SIZE)]
        for j in jobs:
            self.store.create(j)

        errors: list[str] = []
        barrier = threading.Barrier(MAX_WORKERS, timeout=5)

        def _transition(j: Job) -> None:
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                pass
            try:
                j.transition(JobState.RUNNING)
                self.store.update(j)
                j.transition(JobState.SUCCEEDED)
                self.store.update(j)
            except Exception as exc:
                errors.append(f"{j.job_id}: {exc}")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            list(pool.map(_transition, jobs))

        assert not errors, f"Transition errors: {errors}"
        for j in jobs:
            stored = self.store.get(j.job_id)
            assert stored is not None
            assert stored.state == JobState.SUCCEEDED, (
                f"Job {j.job_id} in unexpected state {stored.state}"
            )

    # ── Parallel reads under writes ─────────────────────────────────

    def test_reads_under_write_pressure(self) -> None:
        """Reading existing jobs while other threads create new ones
        must not raise or return None for existing IDs."""
        seed_jobs = [_make_job() for _ in range(BURST_SIZE)]
        for j in seed_jobs:
            self.store.create(j)

        read_results: list[bool] = []
        write_errors: list[str] = []

        def _reader() -> None:
            for j in seed_jobs:
                got = self.store.get(j.job_id)
                read_results.append(got is not None)

        def _writer() -> None:
            try:
                for _ in range(BURST_SIZE):
                    self.store.create(_make_job())
            except Exception as exc:
                write_errors.append(str(exc))

        threads = [
            threading.Thread(target=_reader) for _ in range(4)
        ] + [
            threading.Thread(target=_writer) for _ in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not write_errors, f"Write errors: {write_errors}"
        assert all(read_results), "Some reads returned None for existing jobs"

    # ── Idempotency under contention ────────────────────────────────

    def test_idempotency_key_consistency(self) -> None:
        """Multiple threads looking up the same idempotency key
        must always get the same job_id back."""
        idem_key = "burst-test-unique"
        original = _make_job(idempotency_key=idem_key)
        self.store.create(original)

        results: list[str | None] = []

        def _lookup() -> None:
            got = self.store.get_by_idempotency_key(idem_key)
            results.append(got.job_id if got else None)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            list(pool.map(lambda _: _lookup(), range(BURST_SIZE)))

        assert all(r == original.job_id for r in results), (
            f"Idempotency lookup returned inconsistent results: "
            f"{set(results)}"
        )

    # ── No deadlocks ────────────────────────────────────────────────

    def test_no_deadlock_mixed_operations(self) -> None:
        """Mixed create + get + update + count + list operations
        run concurrently without deadlock (completes within 5s)."""
        seed = [_make_job() for _ in range(20)]
        for j in seed:
            self.store.create(j)

        done_flag = threading.Event()
        errors: list[str] = []

        def _create_loop() -> None:
            for _ in range(50):
                if done_flag.is_set():
                    return
                try:
                    self.store.create(_make_job())
                except Exception as exc:
                    errors.append(f"create: {exc}")

        def _get_loop() -> None:
            for j in seed * 3:
                if done_flag.is_set():
                    return
                try:
                    self.store.get(j.job_id)
                except Exception as exc:
                    errors.append(f"get: {exc}")

        def _list_loop() -> None:
            for _ in range(20):
                if done_flag.is_set():
                    return
                try:
                    self.store.list_jobs(limit=10)
                except Exception as exc:
                    errors.append(f"list: {exc}")

        def _count_loop() -> None:
            for _ in range(50):
                if done_flag.is_set():
                    return
                try:
                    self.store.count()
                except Exception as exc:
                    errors.append(f"count: {exc}")

        threads = [
            threading.Thread(target=_create_loop),
            threading.Thread(target=_create_loop),
            threading.Thread(target=_get_loop),
            threading.Thread(target=_get_loop),
            threading.Thread(target=_list_loop),
            threading.Thread(target=_count_loop),
        ]

        for t in threads:
            t.start()

        # Wait with a hard timeout to detect deadlocks
        for t in threads:
            t.join(timeout=5)
            if t.is_alive():
                done_flag.set()
                pytest.fail("Deadlock detected — thread did not complete within 5s")

        assert not errors, f"Mixed-op errors: {errors}"

    # ── Count consistency ───────────────────────────────────────────

    def test_count_consistency_after_burst(self) -> None:
        """After a burst of creates, count exactly matches."""
        jobs = [_make_job() for _ in range(BURST_SIZE)]

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            list(pool.map(self.store.create, jobs))

        assert self.store.count() == BURST_SIZE
