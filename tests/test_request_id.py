"""G7.1 — Request-ID middleware tests.

Verifies that every HTTP request receives a unique ``X-Request-ID``
header, and that client-supplied request IDs are preserved.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from physics_os.api.app import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Test client for the Ontic Engine API."""
    app = create_app()
    return TestClient(app)


class TestRequestIDMiddleware:
    """G7.1 — Every request has a request_id."""

    def test_response_contains_request_id_header(self, client: TestClient) -> None:
        """Response must include X-Request-ID even when client omits it."""
        resp = client.get("/health")
        assert "X-Request-ID" in resp.headers, (
            "Response missing X-Request-ID header"
        )
        # Must be a valid UUID-4
        rid = resp.headers["X-Request-ID"]
        parsed = uuid.UUID(rid, version=4)
        assert str(parsed) == rid

    def test_client_supplied_id_preserved(self, client: TestClient) -> None:
        """When client sends X-Request-ID, the server echoes it back."""
        custom_id = "test-req-" + uuid.uuid4().hex[:8]
        resp = client.get("/health", headers={"X-Request-ID": custom_id})
        assert resp.headers["X-Request-ID"] == custom_id

    def test_each_request_gets_unique_id(self, client: TestClient) -> None:
        """Two requests without client ID must receive different IDs."""
        resp1 = client.get("/health")
        resp2 = client.get("/health")
        id1 = resp1.headers["X-Request-ID"]
        id2 = resp2.headers["X-Request-ID"]
        assert id1 != id2, "Request IDs must be unique per request"

    def test_request_id_on_non_health_endpoint(self, client: TestClient) -> None:
        """Request ID must be present on all endpoints, not just /health."""
        resp = client.get("/v1/capabilities")
        assert "X-Request-ID" in resp.headers

    def test_request_id_on_error_response(self, client: TestClient) -> None:
        """Even 404 responses must include X-Request-ID."""
        resp = client.get("/nonexistent-endpoint")
        assert "X-Request-ID" in resp.headers
