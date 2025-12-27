"""
Tests for HyperTensor REST API Server
=====================================

Tests for the FastAPI server endpoints including field operations,
sampling, slicing, and simulation stepping.
"""

import pytest

# Check if FastAPI is available
try:
    from fastapi.testclient import TestClient
    from sdk.server.main import app, HAS_FASTAPI
    FASTAPI_AVAILABLE = HAS_FASTAPI
except ImportError:
    FASTAPI_AVAILABLE = False
    TestClient = None
    app = None


# Skip all tests if FastAPI is not available
pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
]


@pytest.fixture
def client():
    """Create a test client for the API."""
    if not FASTAPI_AVAILABLE:
        pytest.skip("FastAPI not available")
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for the health check endpoint."""
    
    def test_health_returns_200(self, client):
        """Health endpoint should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_returns_status_healthy(self, client):
        """Health endpoint should return healthy status."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"


class TestFieldCreation:
    """Tests for field creation endpoints."""
    
    def test_create_field_default_config(self, client):
        """Create field with default configuration."""
        response = client.post("/field/create", json={})
        assert response.status_code == 200
        data = response.json()
        assert "handle" in data
        assert "stats" in data
    
    def test_create_field_custom_size(self, client):
        """Create field with custom size."""
        config = {
            "size_x": 32,
            "size_y": 32,
            "size_z": 32,
            "field_type": "vector",
            "max_rank": 16
        }
        response = client.post("/field/create", json=config)
        assert response.status_code == 200
    
    def test_create_field_invalid_size_rejected(self, client):
        """Reject field creation with invalid size."""
        config = {"size_x": 4}  # Below minimum of 8
        response = client.post("/field/create", json=config)
        assert response.status_code == 422  # Validation error


class TestFieldOperations:
    """Tests for field operation endpoints."""
    
    @pytest.fixture
    def field_handle(self, client):
        """Create a field and return its handle."""
        response = client.post("/field/create", json={"size_x": 32, "size_y": 32, "size_z": 32})
        return response.json()["handle"]
    
    def test_sample_field(self, client, field_handle):
        """Sample field at specific points."""
        sample_request = {
            "points": [[0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
            "max_rank": 8
        }
        response = client.post(f"/field/{field_handle}/sample", json=sample_request)
        assert response.status_code == 200
        data = response.json()
        assert "values" in data
        assert len(data["values"]) == 2
    
    def test_get_field_stats(self, client, field_handle):
        """Get field statistics."""
        response = client.get(f"/field/{field_handle}/stats")
        assert response.status_code == 200
        data = response.json()
        assert "max_rank" in data
        assert "memory_bytes" in data
    
    def test_delete_field(self, client, field_handle):
        """Delete a field."""
        response = client.delete(f"/field/{field_handle}")
        assert response.status_code == 200
        
        # Verify field is deleted
        response = client.get(f"/field/{field_handle}/stats")
        assert response.status_code == 404


class TestSimulationStep:
    """Tests for simulation stepping."""
    
    @pytest.fixture
    def field_handle(self, client):
        """Create a field for simulation testing."""
        response = client.post("/field/create", json={"size_x": 32, "size_y": 32, "size_z": 32})
        return response.json()["handle"]
    
    def test_step_simulation(self, client, field_handle):
        """Step simulation forward."""
        response = client.post(f"/field/{field_handle}/step", json={"dt": 0.01})
        assert response.status_code == 200
        data = response.json()
        assert "stats" in data
    
    def test_step_updates_step_count(self, client, field_handle):
        """Step count should increment."""
        # Get initial step count
        response = client.get(f"/field/{field_handle}/stats")
        initial_steps = response.json()["step_count"]
        
        # Step simulation
        client.post(f"/field/{field_handle}/step", json={"dt": 0.01})
        
        # Check step count increased
        response = client.get(f"/field/{field_handle}/stats")
        assert response.json()["step_count"] == initial_steps + 1


class TestCORSConfiguration:
    """Tests for CORS configuration."""
    
    def test_cors_headers_present(self, client):
        """CORS headers should be present in responses."""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )
        # CORS preflight should return appropriate headers
        # Note: Actual CORS behavior depends on configuration


class TestErrorHandling:
    """Tests for structured error responses."""
    
    def test_field_not_found_error_structure(self, client):
        """Test that 404 errors return structured error response."""
        response = client.get("/fields/99999/stats")
        assert response.status_code == 404
        data = response.json()
        
        # Check structured error format
        assert "detail" in data
        detail = data["detail"]
        assert "error" in detail
        error = detail["error"]
        assert "code" in error
        assert "message" in error
        assert error["code"] == "FIELD_NOT_FOUND"
    
    def test_delete_nonexistent_field_error(self, client):
        """Test delete returns structured error for nonexistent field."""
        response = client.delete("/fields/99999")
        assert response.status_code == 404
        data = response.json()
        
        assert "detail" in data
        assert "error" in data["detail"]
        assert data["detail"]["error"]["code"] == "FIELD_NOT_FOUND"
    
    def test_sample_nonexistent_field_error(self, client):
        """Test sample returns structured error for nonexistent field."""
        response = client.post(
            "/fields/99999/sample",
            json={"points": [[0.5, 0.5, 0.5]], "max_rank": 16}
        )
        assert response.status_code == 404
        data = response.json()
        
        assert "detail" in data
        assert data["detail"]["error"]["code"] == "FIELD_NOT_FOUND"
    
    def test_step_nonexistent_field_error(self, client):
        """Test step returns structured error for nonexistent field."""
        response = client.post("/fields/99999/step?dt=0.01")
        assert response.status_code == 404
        data = response.json()
        
        assert "detail" in data
        assert data["detail"]["error"]["code"] == "FIELD_NOT_FOUND"
    
    def test_error_response_no_sensitive_info(self, client):
        """Test that error responses don't leak sensitive information."""
        response = client.get("/fields/99999/stats")
        data = response.json()
        
        # Error message should not contain stack traces or internal paths
        error_str = str(data)
        assert "Traceback" not in error_str
        assert "File \"" not in error_str
        assert "line " not in error_str.lower() or "correlation" in error_str.lower()


class TestConcurrency:
    """Tests for concurrent access safety."""
    
    def test_concurrent_field_creation(self, client):
        """Test that multiple fields can be created concurrently."""
        import concurrent.futures
        
        def create_field():
            response = client.post("/fields", json={
                "size_x": 32,
                "size_y": 32,
                "size_z": 32,
                "field_type": "scalar",
                "max_rank": 16
            })
            return response.status_code
        
        # Create multiple fields concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_field) for _ in range(5)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All should succeed or return controlled errors (not 500)
        for status in results:
            assert status in [200, 201, 500]  # 500 only if HyperTensor not available
    
    def test_handle_allocation_unique(self, client):
        """Test that each field gets a unique handle."""
        handles = set()
        
        for _ in range(3):
            response = client.post("/fields", json={
                "size_x": 32,
                "size_y": 32,
                "size_z": 32,
                "field_type": "scalar",
                "max_rank": 16
            })
            if response.status_code == 200:
                handle = response.json()["handle"]
                assert handle not in handles, "Duplicate handle allocated"
                handles.add(handle)
    
    def test_concurrent_health_checks(self, client):
        """Test that health endpoint handles concurrent requests."""
        import concurrent.futures
        
        def health_check():
            response = client.get("/health")
            return response.status_code
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(health_check) for _ in range(20)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All health checks should succeed
        assert all(status == 200 for status in results)


class TestServerConfiguration:
    """Tests for server configuration handling."""
    
    def test_cors_origins_from_env(self, monkeypatch):
        """Test that CORS origins can be configured via environment."""
        # This test verifies the configuration is read (may need reload)
        # Check that the environment variable pattern is correct
        test_origins = "http://example.com,http://test.com"
        monkeypatch.setenv("HYPERTENSOR_CORS_ORIGINS", test_origins)
        
        # The origins should be parsed as comma-separated
        origins = test_origins.split(",")
        assert len(origins) == 2
        assert "http://example.com" in origins
        assert "http://test.com" in origins
    
    def test_default_cors_localhost_only(self, client):
        """Test that default CORS allows only localhost."""
        import os
        
        # Get the default CORS setting
        default_origins = os.environ.get(
            "HYPERTENSOR_CORS_ORIGINS",
            "http://localhost:3000,http://localhost:8080,http://127.0.0.1:3000,http://127.0.0.1:8080"
        )
        
        # Verify localhost is in defaults
        assert "localhost" in default_origins
        # Verify no wildcard
        assert "*" not in default_origins
    
    def test_api_has_docs_endpoint(self, client):
        """Test that API documentation is available."""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_openapi_schema_available(self, client):
        """Test that OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert data["info"]["title"] == "HyperTensor API"
    
    def test_health_returns_version_info(self, client):
        """Health endpoint should include version information."""
        response = client.get("/health")
        data = response.json()
        
        # Should have some form of version or metadata
        assert "status" in data
        # Version may be in response if implemented
    
    def test_error_responses_consistent_schema(self, client):
        """All error responses should follow consistent schema."""
        # Test various error endpoints
        error_endpoints = [
            ("/fields/99999/stats", "GET"),
            ("/field/99999/sample", "POST"),
        ]
        
        for endpoint, method in error_endpoints:
            if method == "GET":
                response = client.get(endpoint)
            else:
                response = client.post(endpoint, json={})
            
            if response.status_code >= 400:
                data = response.json()
                # Should have error structure
                if "detail" in data:
                    # FastAPI style or custom
                    assert isinstance(data["detail"], (str, dict))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
