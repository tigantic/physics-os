"""
TPC Certificate Authority HTTP client.

Wraps the CA REST API for certificate issuance, retrieval, and verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import requests

from fluidelite_verify.certificate import Certificate
from fluidelite_verify.errors import CertificateNotFound, TPCError


@dataclass
class IssueResult:
    """Result of certificate issuance via CA."""
    certificate_id: str
    content_hash: str
    signer_pubkey: str
    domain: str
    size_bytes: int
    issued_at: str
    on_chain_status: str


class TPCClient:
    """
    HTTP client for the TPC Certificate Authority.

    Usage:
        client = TPCClient(
            ca_url="https://ca.fluidelite.io",
            api_key="YOUR_API_KEY"
        )

        # Issue a certificate
        result = client.issue(
            domain="thermal",
            proof=b"\\xde\\xad\\xbe\\xef",
            public_inputs=["0x01"]
        )

        # Retrieve and verify
        cert = client.get_certificate(result.certificate_id)
        assert cert.verify().valid
    """

    def __init__(
        self,
        ca_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize the CA client.

        Args:
            ca_url: Base URL of the certificate authority (e.g., http://localhost:8444).
            api_key: API key for authentication.
            timeout: Request timeout in seconds.
        """
        self._base_url = ca_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()

        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"

        self._session.headers["Content-Type"] = "application/json"

    def issue(
        self,
        domain: str,
        proof: bytes,
        public_inputs: Optional[list[str]] = None,
        solver_hash: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> IssueResult:
        """
        Issue a new TPC certificate.

        Args:
            domain: Physics domain (thermal, euler3d, ns_imex, fluidelite).
            proof: Proof bytes.
            public_inputs: Public input values (hex strings).
            solver_hash: SHA-256 hash of the solver binary (hex).
            metadata: Additional metadata key-value pairs.

        Returns:
            IssueResult with certificate ID and metadata.

        Raises:
            TPCError: If issuance fails.
        """
        payload: dict[str, Any] = {
            "domain": domain,
            "proof": proof.hex(),
            "public_inputs": public_inputs or [],
        }

        if solver_hash:
            payload["solver_hash"] = solver_hash
        if metadata:
            payload["metadata"] = metadata

        resp = self._session.post(
            f"{self._base_url}/v1/certificates/issue",
            json=payload,
            timeout=self._timeout,
        )

        if resp.status_code != 201:
            raise TPCError(
                f"Certificate issuance failed (HTTP {resp.status_code}): "
                f"{resp.text}"
            )

        data = resp.json()
        return IssueResult(
            certificate_id=data["certificate_id"],
            content_hash=data["content_hash"],
            signer_pubkey=data["signer_pubkey"],
            domain=data["domain"],
            size_bytes=data["size_bytes"],
            issued_at=data["issued_at"],
            on_chain_status=data["on_chain_status"],
        )

    def get_certificate(self, certificate_id: str) -> Certificate:
        """
        Retrieve a certificate by ID.

        Args:
            certificate_id: UUID of the certificate.

        Returns:
            Parsed Certificate object.

        Raises:
            CertificateNotFound: If the certificate doesn't exist.
        """
        resp = self._session.get(
            f"{self._base_url}/v1/certificates/{certificate_id}",
            timeout=self._timeout,
        )

        if resp.status_code == 404:
            raise CertificateNotFound(
                f"Certificate {certificate_id} not found"
            )

        if resp.status_code != 200:
            raise TPCError(
                f"Failed to retrieve certificate (HTTP {resp.status_code}): "
                f"{resp.text}"
            )

        return Certificate.from_bytes(resp.content)

    def verify(
        self,
        certificate: Optional[Certificate] = None,
        certificate_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Verify a certificate via the CA.

        Args:
            certificate: Certificate object to verify.
            certificate_id: Certificate UUID to look up and verify.

        Returns:
            Verification response dict from the CA.

        Raises:
            TPCError: If verification request fails.
        """
        payload: dict[str, Any] = {}

        if certificate:
            payload["certificate"] = certificate.raw.hex()
        elif certificate_id:
            payload["certificate_id"] = certificate_id
        else:
            raise TPCError("Provide either certificate or certificate_id")

        resp = self._session.post(
            f"{self._base_url}/v1/certificates/verify",
            json=payload,
            timeout=self._timeout,
        )

        if resp.status_code not in (200, 422):
            raise TPCError(
                f"Verification failed (HTTP {resp.status_code}): {resp.text}"
            )

        return resp.json()

    def stats(self) -> dict[str, Any]:
        """
        Get CA statistics.

        Returns:
            Stats dict with issuance counts, uptime, etc.
        """
        resp = self._session.get(
            f"{self._base_url}/v1/certificates/stats",
            timeout=self._timeout,
        )

        if resp.status_code != 200:
            raise TPCError(
                f"Failed to get stats (HTTP {resp.status_code}): {resp.text}"
            )

        return resp.json()

    def health(self) -> dict[str, Any]:
        """
        Check CA health.

        Returns:
            Health status dict.
        """
        resp = self._session.get(
            f"{self._base_url}/health",
            timeout=self._timeout,
        )

        return resp.json()
