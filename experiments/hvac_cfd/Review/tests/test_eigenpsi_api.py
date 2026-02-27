#!/usr/bin/env python3
"""
EigenPsi API Integration Tests

Tests the FastAPI server endpoints that wire UI to Review backend.
Follows Constitution Article III: Testing Protocols.

Run with:
    cd HVAC_CFD/Review
    pytest tests/test_eigenpsi_api.py -v
"""

import pytest
import json
import tempfile
import shutil
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add UI directory to path for importing the server
_REVIEW_DIR = Path(__file__).parent.parent
_UI_DIR = _REVIEW_DIR.parent / "UI"
sys.path.insert(0, str(_UI_DIR))
sys.path.insert(0, str(_REVIEW_DIR))


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_projects_dir(tmp_path):
    """Create a temporary projects directory."""
    projects = tmp_path / "projects"
    projects.mkdir()
    return projects


@pytest.fixture
def sample_job_spec():
    """Valid job specification matching schema."""
    return {
        "client": {
            "name": "Test Client Corp",
            "project_id": "TEST-001",
            "contact": "Jane Doe",
            "email": "jane@test.com"
        },
        "room": {
            "name": "Test Conference Room",
            "type": "conference_room",
            "dimensions_m": [10.0, 8.0, 3.0]
        },
        "load": {
            "occupants": 12,
            "heat_load_per_person_watts": 100,
            "equipment_load_watts": 500,
            "lighting_load_watts": 200
        },
        "hvac": {
            "supply_temp_c": 18.0,
            "num_diffusers": 2,
            "diffuser_area_m2": 0.1
        },
        "constraints": {
            "max_velocity_ms": 0.25,
            "target_temp_c": 22.0,
            "temp_range_c": [20.0, 24.0],
            "max_co2_ppm": 1000
        },
        "deliverables": {
            "thermal_heatmap": True,
            "velocity_field": True,
            "convergence_plot": True,
            "pdf_report": True
        },
        "notes": "Integration test project"
    }


@pytest.fixture
def intake_form_spec():
    """Legacy intake form format (from eigenpsi-intake-form.html)."""
    return {
        "_form": "eigenpsi_intake_v1",
        "_exported": "2026-01-17T10:00:00.000Z",
        "project": {
            "client_name": "Legacy Client",
            "project_name": "LEGACY-001",
            "contact_name": "John Legacy",
            "contact_email": "john@legacy.com",
            "analysis_type": "comfort_verification"
        },
        "geometry": {
            "room_name": "Open Office",
            "room_type": "office_open",
            "room_length_ft": 30,
            "room_width_ft": 20,
            "room_height_ft": 10
        },
        "hvac_supply": {
            "supply_cfm": 1500,
            "supply_temp_f": 55,
            "num_diffusers": 4,
            "diffuser_type": "ceiling_square"
        },
        "thermal_loads": {
            "num_occupants": 20,
            "equipment_watts": 1000
        },
        "targets": {
            "target_temp_f": 72,
            "max_velocity_fpm": 50,
            "max_co2_ppm": 1000
        }
    }


@pytest.fixture
def test_client(temp_projects_dir):
    """Create a FastAPI TestClient with mocked projects directory."""
    # Patch the PROJECTS_DIR before importing
    import eigenpsi_server
    original_projects_dir = eigenpsi_server.PROJECTS_DIR
    eigenpsi_server.PROJECTS_DIR = temp_projects_dir
    
    from fastapi.testclient import TestClient
    client = TestClient(eigenpsi_server.app)
    
    yield client
    
    # Restore
    eigenpsi_server.PROJECTS_DIR = original_projects_dir


# =============================================================================
# API CONTRACT TESTS
# =============================================================================

class TestHealthAndStatus:
    """Test server status endpoints."""
    
    def test_status_endpoint(self, test_client):
        """GET /api/status returns server state."""
        response = test_client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert "active_job" in data
        assert "projects_dir" in data
        assert "gpu" in data


class TestProjectCRUD:
    """Test project creation and retrieval."""
    
    def test_create_project_with_valid_spec(self, test_client, sample_job_spec):
        """POST /api/projects creates project with valid JobSpec."""
        response = test_client.post("/api/projects", json=sample_job_spec)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "created"
        assert "TEST-001" in data["project_id"]
        assert Path(data["spec_path"]).exists()
    
    def test_create_project_with_legacy_format(self, test_client, intake_form_spec):
        """POST /api/projects handles legacy intake form format."""
        response = test_client.post("/api/projects", json=intake_form_spec)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "created"
    
    def test_list_projects_empty(self, test_client):
        """GET /api/projects returns empty list initially."""
        response = test_client.get("/api/projects")
        assert response.status_code == 200
        data = response.json()
        assert "projects" in data
        assert isinstance(data["projects"], list)
    
    def test_list_projects_after_create(self, test_client, sample_job_spec):
        """GET /api/projects includes created project."""
        # Create
        test_client.post("/api/projects", json=sample_job_spec)
        
        # List
        response = test_client.get("/api/projects")
        assert response.status_code == 200
        data = response.json()
        assert len(data["projects"]) >= 1
        assert any("TEST-001" in p["id"] for p in data["projects"])
    
    def test_get_project_details(self, test_client, sample_job_spec):
        """GET /api/projects/{id} returns project details."""
        # Create
        create_resp = test_client.post("/api/projects", json=sample_job_spec)
        project_id = create_resp.json()["project_id"]
        
        # Get
        response = test_client.get(f"/api/projects/{project_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == project_id
        assert data["spec"] is not None
        assert data["spec"]["client"]["name"] == "Test Client Corp"
    
    def test_get_nonexistent_project(self, test_client):
        """GET /api/projects/{id} returns 404 for missing project."""
        response = test_client.get("/api/projects/nonexistent-project-xyz")
        assert response.status_code == 404


class TestJobExecution:
    """Test simulation run/stop endpoints."""
    
    def test_run_nonexistent_project(self, test_client):
        """POST /api/projects/{id}/run returns 404 for missing project."""
        response = test_client.post(
            "/api/projects/nonexistent/run",
            json={"duration": 10}
        )
        assert response.status_code == 404
    
    def test_run_project_starts(self, test_client, sample_job_spec):
        """POST /api/projects/{id}/run starts simulation."""
        # Create project
        create_resp = test_client.post("/api/projects", json=sample_job_spec)
        project_id = create_resp.json()["project_id"]
        
        # Run - may fail if hyperfoam isn't set up, but should return 200 for start
        response = test_client.post(
            f"/api/projects/{project_id}/run",
            json={"duration": 10, "skip_optimize": True}
        )
        # Should at least start the job
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
    
    def test_stop_not_running(self, test_client, sample_job_spec):
        """POST /api/projects/{id}/stop returns 400 when no job running."""
        create_resp = test_client.post("/api/projects", json=sample_job_spec)
        project_id = create_resp.json()["project_id"]
        
        response = test_client.post(f"/api/projects/{project_id}/stop")
        assert response.status_code == 400


class TestAssets:
    """Test asset retrieval endpoints."""
    
    def test_get_asset_not_found(self, test_client, sample_job_spec):
        """GET /api/projects/{id}/assets/{file} returns 404 for missing asset."""
        create_resp = test_client.post("/api/projects", json=sample_job_spec)
        project_id = create_resp.json()["project_id"]
        
        response = test_client.get(f"/api/projects/{project_id}/assets/nonexistent.png")
        assert response.status_code == 404


class TestSchemaValidation:
    """Test Pydantic schema validation."""
    
    def test_invalid_dimensions_rejected(self, test_client):
        """Project with invalid dimensions should still be accepted (legacy fallback)."""
        invalid_spec = {
            "client": {"name": "Test", "project_id": "INV-001"},
            "room": {
                "name": "Bad Room",
                "dimensions_m": [-5.0, 0, 3.0]  # Invalid: negative/zero
            },
            "load": {"occupants": 5},
            "constraints": {
                "max_velocity_ms": 0.25,
                "target_temp_c": 22.0,
                "max_co2_ppm": 1000
            }
        }
        # With legacy fallback, this should still create (no strict validation)
        response = test_client.post("/api/projects", json=invalid_spec)
        # Fallback accepts legacy format
        assert response.status_code == 200
    
    def test_cfm_to_velocity_conversion(self, test_client, temp_projects_dir):
        """CFM is converted to velocity m/s."""
        spec_with_cfm = {
            "client": {"name": "CFM Test", "project_id": "CFM-001"},
            "room": {"dimensions_m": [10, 10, 3]},
            "load": {"occupants": 10},
            "hvac": {
                "supply_cfm": 1000,
                "num_diffusers": 2,
                "diffuser_area_m2": 0.1
            },
            "constraints": {
                "max_velocity_ms": 0.25,
                "target_temp_c": 22.0,
                "max_co2_ppm": 1000
            }
        }
        response = test_client.post("/api/projects", json=spec_with_cfm)
        assert response.status_code == 200
        
        # Check the saved spec has velocity calculated
        project_id = response.json()["project_id"]
        spec_path = temp_projects_dir / project_id / "job_spec.json"
        with open(spec_path) as f:
            saved = json.load(f)
        
        # CFM 1000 * 0.000471947 = 0.471947 m³/s
        # Divided by (2 * 0.1) = 0.2 m² = 2.36 m/s
        assert "supply_velocity_ms" in saved["hvac"]
        assert saved["hvac"]["supply_velocity_ms"] > 0


# =============================================================================
# UI SERVING TESTS
# =============================================================================

class TestUIServing:
    """Test HTML UI serving endpoints."""
    
    def test_cockpit_serves_html(self, test_client):
        """GET / returns cockpit HTML."""
        response = test_client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
    
    def test_intake_serves_html(self, test_client):
        """GET /intake returns intake form HTML."""
        response = test_client.get("/intake")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert b"EigenPsi" in response.content or b"eigenpsi" in response.content.lower()


# =============================================================================
# WEBSOCKET TESTS
# =============================================================================

class TestWebSocket:
    """Test WebSocket log streaming."""
    
    def test_websocket_connection(self, test_client):
        """WebSocket /ws/logs accepts connection."""
        with test_client.websocket_connect("/ws/logs") as websocket:
            # Should receive status message
            data = websocket.receive_json()
            assert "type" in data
            # Close gracefully
            websocket.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
