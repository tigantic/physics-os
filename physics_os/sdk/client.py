"""Ontic SDK — Python client library.

Provides a typed, synchronous + async client for the Ontic Engine
Runtime API.  Handles authentication, job submission, polling,
result retrieval, and local validation.

Usage::

    from physics_os.sdk.client import OnticClient

    client = OnticClient(
        base_url="https://api.physics_os.io",
        api_key="sk-...",
    )

    job = client.run("burgers", n_bits=8, n_steps=100)
    print(job.result["conservation"])
    print(job.certificate["claims"])
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterator
from urllib.error import HTTPError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


# ── Data classes ────────────────────────────────────────────────────


@dataclass
class JobResult:
    """Represents a completed The Ontic Engine job with all artifacts."""

    job_id: str
    status: str
    domain: str
    result: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    certificate: dict[str, Any] | None = None
    envelope: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

    @property
    def succeeded(self) -> bool:
        return self.status in ("succeeded", "validated", "attested")

    @property
    def conservation(self) -> dict[str, Any] | None:
        if self.result:
            return self.result.get("conservation")
        return None

    @property
    def fields(self) -> dict[str, Any] | None:
        if self.result:
            return self.result.get("fields")
        return None

    @property
    def claims(self) -> list[dict[str, Any]]:
        if self.certificate:
            return self.certificate.get("claims", [])
        return []


@dataclass
class Domain:
    """Available physics domain descriptor."""

    key: str
    label: str
    parameters: list[dict[str, Any]] = field(default_factory=list)


# ── Client ──────────────────────────────────────────────────────────


class OnticClient:
    """Synchronous client for the Ontic Engine Runtime API.

    Parameters
    ----------
    base_url : str
        API base URL (e.g. ``http://localhost:8000`` or ``https://api.physics_os.io``).
    api_key : str
        Bearer API key for authentication.
    timeout_s : float
        HTTP request timeout in seconds.
    poll_interval_s : float
        Interval between job status polls.
    max_poll_s : float
        Maximum time to poll before raising ``TimeoutError``.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = "",
        timeout_s: float = 30.0,
        poll_interval_s: float = 1.0,
        max_poll_s: float = 300.0,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_s
        self._poll_interval = poll_interval_s
        self._max_poll = max_poll_s

    # ── Core methods ────────────────────────────────────────────────

    def run(
        self,
        domain: str,
        *,
        n_bits: int = 8,
        n_steps: int = 100,
        dt: float | None = None,
        max_rank: int = 64,
        truncation_tol: float = 1e-10,
        parameters: dict[str, Any] | None = None,
        job_type: str = "full_pipeline",
        idempotency_key: str | None = None,
        wait: bool = True,
    ) -> JobResult:
        """Submit a job and optionally wait for completion.

        Parameters
        ----------
        domain : str
            Physics domain key (e.g. ``"burgers"``, ``"maxwell"``).
        n_bits, n_steps, dt, max_rank, truncation_tol :
            Simulation parameters.
        parameters : dict, optional
            Domain-specific extra parameters.
        job_type : str
            Job type (``"full_pipeline"``, ``"physics_simulation"``, etc.).
        idempotency_key : str, optional
            Prevents duplicate submissions.
        wait : bool
            If True (default), polls until the job reaches a terminal state.

        Returns
        -------
        JobResult
            The completed (or submitted) job with all available artifacts.
        """
        body: dict[str, Any] = {
            "domain": domain,
            "job_type": job_type,
            "n_bits": n_bits,
            "n_steps": n_steps,
            "max_rank": max_rank,
            "truncation_tol": truncation_tol,
        }
        if dt is not None:
            body["dt"] = dt
        if parameters:
            body["parameters"] = parameters

        headers: dict[str, str] = {}
        if idempotency_key:
            headers["X-Idempotency-Key"] = idempotency_key

        resp = self._post("/v1/jobs", body, extra_headers=headers)
        job_id = resp["job_id"]
        logger.info("Job submitted: %s (type=%s, domain=%s)", job_id, job_type, domain)

        if not wait:
            return JobResult(
                job_id=job_id,
                status=resp.get("status", "queued"),
                domain=domain,
            )

        return self.wait_for(job_id, domain=domain)

    def wait_for(self, job_id: str, domain: str = "unknown") -> JobResult:
        """Poll a job until it reaches a terminal state.

        Raises ``TimeoutError`` if ``max_poll_s`` is exceeded.
        """
        terminal = {"succeeded", "validated", "attested", "failed"}
        start = time.monotonic()

        while True:
            status_resp = self.get_status(job_id)
            current = status_resp.get("status", "unknown")
            logger.debug("Job %s: %s", job_id, current)

            if current in terminal:
                return self._build_result(job_id, domain, status_resp)

            elapsed = time.monotonic() - start
            if elapsed > self._max_poll:
                raise TimeoutError(
                    f"Job {job_id} did not complete within {self._max_poll}s "
                    f"(last status: {current})"
                )
            time.sleep(self._poll_interval)

    def get_status(self, job_id: str) -> dict[str, Any]:
        """Get current job status."""
        return self._get(f"/v1/jobs/{job_id}")

    def get_result(self, job_id: str) -> dict[str, Any]:
        """Get the result payload for a completed job."""
        return self._get(f"/v1/jobs/{job_id}/result")

    def get_validation(self, job_id: str) -> dict[str, Any]:
        """Get the validation report for a validated job."""
        return self._get(f"/v1/jobs/{job_id}/validation")

    def get_certificate(self, job_id: str) -> dict[str, Any]:
        """Get the trust certificate for an attested job."""
        return self._get(f"/v1/jobs/{job_id}/certificate")

    def validate(self, artifact: dict[str, Any]) -> dict[str, Any]:
        """Validate an artifact bundle (stateless, no auth required)."""
        return self._post("/v1/validate", artifact, auth=False)

    def capabilities(self) -> list[Domain]:
        """List available physics domains."""
        resp = self._get("/v1/capabilities")
        return [
            Domain(
                key=d["key"],
                label=d["label"],
                parameters=d.get("parameters", []),
            )
            for d in resp.get("domains", [])
        ]

    def health(self) -> dict[str, Any]:
        """Check API health."""
        return self._get("/v1/health")

    def contracts(self, version: str = "v1") -> dict[str, Any]:
        """Get contract schemas for a version."""
        return self._get(f"/v1/contracts/{version}")

    # ── Problem templates ───────────────────────────────────────────

    def list_templates(self) -> list[dict[str, Any]]:
        """List available problem templates.

        Returns
        -------
        list[dict]
            Template descriptors with problem_class, label,
            supported_geometries, and example_params.
        """
        resp = self._get("/v1/templates")
        return resp.get("templates", [])

    def get_template(self, problem_class: str) -> dict[str, Any]:
        """Get details for a specific problem template.

        Parameters
        ----------
        problem_class : str
            Template key (e.g. ``"external_flow"``).
        """
        return self._get(f"/v1/templates/{problem_class}")

    def solve_problem(
        self,
        problem_class: str,
        geometry: dict[str, Any],
        flow: dict[str, Any],
        *,
        boundaries: dict[str, str] | None = None,
        quality: str = "standard",
        t_end: float | None = None,
        domain_multiplier: float = 10.0,
        max_rank: int = 64,
        idempotency_key: str | None = None,
        wait: bool = True,
    ) -> JobResult:
        """Submit a high-level physics problem.

        The Problem Compiler resolves geometry, fluid properties,
        dimensionless numbers, and optimal resolution automatically.

        Parameters
        ----------
        problem_class : str
            One of: external_flow, internal_flow, heat_transfer,
            wave_propagation, natural_convection, boundary_layer,
            vortex_dynamics, channel_flow.
        geometry : dict
            ``{"shape": "circle", "params": {"radius": 0.01}}``.
        flow : dict
            ``{"velocity": 10.0, "fluid": "air"}``.
        boundaries : dict, optional
            Boundary conditions (inlet, outlet, walls, top, bottom).
        quality : str
            Quality tier: quick, standard, high, maximum.
        t_end : float, optional
            Simulation end time in seconds.
        domain_multiplier : float
            Domain size as multiple of characteristic length.
        max_rank : int
            Maximum tensor-train rank.
        idempotency_key : str, optional
            Prevents duplicate submissions.
        wait : bool
            If True (default), polls until job completes.

        Returns
        -------
        JobResult
            Completed job with result, validation, and certificate.
        """
        body: dict[str, Any] = {
            "problem_class": problem_class,
            "geometry": geometry,
            "flow": flow,
            "quality": quality,
            "domain_multiplier": domain_multiplier,
            "max_rank": max_rank,
        }
        if boundaries:
            body["boundaries"] = boundaries
        if t_end is not None:
            body["t_end"] = t_end

        headers: dict[str, str] = {}
        if idempotency_key:
            headers["X-Idempotency-Key"] = idempotency_key

        resp = self._post("/v1/problems", body, extra_headers=headers)
        job_id = resp["job_id"]
        domain = resp.get("compilation", {}).get("domain", "unknown")
        logger.info(
            "Problem submitted: %s (class=%s, domain=%s)",
            job_id, problem_class, domain,
        )

        if not wait:
            return JobResult(
                job_id=job_id,
                status=resp.get("status", "queued"),
                domain=domain,
            )

        return self.wait_for(job_id, domain=domain)

    # ── Batch / convenience ─────────────────────────────────────────

    def run_batch(
        self,
        jobs: list[dict[str, Any]],
        wait: bool = True,
    ) -> list[JobResult]:
        """Submit multiple jobs and optionally wait for all to complete.

        Parameters
        ----------
        jobs : list[dict]
            Each dict must contain ``"domain"`` and may contain any
            parameter accepted by ``run()``.
        wait : bool
            If True, polls all jobs to completion.
        """
        submitted: list[tuple[str, str]] = []  # (job_id, domain)
        for spec in jobs:
            domain = spec.pop("domain")
            resp = self.run(domain, wait=False, **spec)
            submitted.append((resp.job_id, domain))

        if not wait:
            return [
                JobResult(job_id=jid, status="queued", domain=d)
                for jid, d in submitted
            ]

        return [self.wait_for(jid, domain=d) for jid, d in submitted]

    # ── Internal HTTP ───────────────────────────────────────────────

    def _build_result(
        self,
        job_id: str,
        domain: str,
        status_resp: dict[str, Any],
    ) -> JobResult:
        """Assemble a full JobResult from individual artifact endpoints."""
        current = status_resp.get("status", "unknown")
        jr = JobResult(job_id=job_id, status=current, domain=domain)

        if current == "failed":
            jr.error = status_resp.get("error")
            return jr

        # Fetch artifacts
        try:
            jr.result = self.get_result(job_id)
        except APIError:
            pass

        try:
            jr.validation = self.get_validation(job_id)
        except APIError:
            pass

        try:
            jr.certificate = self.get_certificate(job_id)
        except APIError:
            pass

        # Full envelope is the status response itself if it has envelope fields
        if "artifact_hashes" in status_resp:
            jr.envelope = status_resp

        return jr

    def _headers(self, auth: bool = True) -> dict[str, str]:
        h: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if auth and self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def _get(self, path: str, auth: bool = True) -> dict[str, Any]:
        url = f"{self._base}{path}"
        req = Request(url, headers=self._headers(auth), method="GET")
        return self._do(req)

    def _post(
        self,
        path: str,
        body: dict[str, Any],
        auth: bool = True,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        url = f"{self._base}{path}"
        headers = self._headers(auth)
        if extra_headers:
            headers.update(extra_headers)
        data = json.dumps(body).encode("utf-8")
        req = Request(url, data=data, headers=headers, method="POST")
        return self._do(req)

    def _do(self, req: Request) -> dict[str, Any]:
        try:
            with urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8")
            except Exception:
                pass
            raise APIError(
                status_code=exc.code,
                message=body or str(exc),
                url=req.full_url,
            ) from exc
        except Exception as exc:
            raise APIError(
                status_code=0,
                message=str(exc),
                url=req.full_url,
            ) from exc


# ── Exceptions ──────────────────────────────────────────────────────


class APIError(Exception):
    """Raised when the API returns an error response."""

    def __init__(self, status_code: int, message: str, url: str = "") -> None:
        self.status_code = status_code
        self.message = message
        self.url = url
        super().__init__(f"HTTP {status_code} from {url}: {message}")

    @property
    def retryable(self) -> bool:
        return self.status_code in (429, 500, 502, 503, 504)

    @property
    def detail(self) -> dict[str, Any]:
        try:
            return json.loads(self.message)
        except (json.JSONDecodeError, TypeError):
            return {"message": self.message}
