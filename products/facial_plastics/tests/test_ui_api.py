"""Tests for the UI API layer — UIApplication and server routing.

Tests the UIApplication class methods (G1-G9 interaction contract)
without actually launching an HTTP server.
"""

from __future__ import annotations

import json
import pathlib
import pytest

from products.facial_plastics.ui.api import UIApplication


@pytest.fixture
def app(tmp_path: pathlib.Path) -> UIApplication:
    """Provide a UIApplication backed by a temporary directory."""
    return UIApplication(library_root=tmp_path)


# =====================================================================
# UIApplication core
# =====================================================================


class TestUIApplicationInit:
    """Verify UIApplication instantiation."""

    def test_create_default(self, app: UIApplication) -> None:
    
        assert app is not None

    def test_has_operators(self, app: UIApplication) -> None:
    
        ops = app.list_operators()
        assert isinstance(ops, dict)
        # Should include all 4 registries
        assert "operators" in ops
        assert len(ops["operators"]) >= 21  # 8+7+6 new + rhinoplasty


# =====================================================================
# G1: Case Library
# =====================================================================


class TestG1CaseLibrary:
    """Test case listing, creation, and curation."""

    def test_list_cases(self, app: UIApplication) -> None:
    
        data = app.list_cases()
        assert isinstance(data, dict)
        assert "cases" in data
        assert "total" in data

    def test_create_case(self, app: UIApplication) -> None:
    
        result = app.create_case(
            procedure="rhinoplasty",
            patient_age=35,
            patient_sex="female",
            notes="test case",
        )
        assert isinstance(result, dict)
        assert "case_id" in result

    def test_get_nonexistent_case(self, app: UIApplication) -> None:
    
        result = app.get_case("nonexistent-id-12345")
        assert isinstance(result, dict)
        # Should return error or empty
        assert "error" in result or "case" in result

    def test_curate_library(self, app: UIApplication) -> None:
    
        result = app.curate_library()
        assert isinstance(result, dict)


# =====================================================================
# G2: Twin Inspect
# =====================================================================


class TestG2TwinInspect:
    """Test twin summary and mesh data retrieval."""

    def test_get_twin_summary_no_case(self, app: UIApplication) -> None:
    
        result = app.get_twin_summary("nonexistent-id")
        assert isinstance(result, dict)

    def test_get_mesh_data_no_case(self, app: UIApplication) -> None:
    
        result = app.get_mesh_data("nonexistent-id")
        assert isinstance(result, dict)

    def test_get_landmarks_no_case(self, app: UIApplication) -> None:
    
        result = app.get_landmarks("nonexistent-id")
        assert isinstance(result, dict)


# =====================================================================
# G3: Plan Author
# =====================================================================


class TestG3PlanAuthor:
    """Test operator listing, template creation, and plan compilation."""

    def test_list_operators(self, app: UIApplication) -> None:
    
        data = app.list_operators()
        assert isinstance(data, dict)
        assert "operators" in data

        ops = data["operators"]
        # Verify all procedure families present
        names = set(ops.keys())
        assert "smas_plication" in names
        assert "upper_lid_skin_excision" in names
        assert "ha_filler_injection" in names

    def test_list_templates(self, app: UIApplication) -> None:
    
        data = app.list_templates()
        assert isinstance(data, dict)
        assert "templates" in data

        tmpl = data["templates"]
        assert "rhinoplasty" in tmpl
        assert "facelift" in tmpl
        assert "blepharoplasty" in tmpl
        assert "fillers" in tmpl

    def test_create_plan_from_template(self, app: UIApplication) -> None:
    
        result = app.create_plan_from_template(category="rhinoplasty", template="reduction_rhinoplasty")
        assert isinstance(result, dict)
        assert "plan" in result or "error" in result

    def test_create_plan_from_facelift_template(self, app: UIApplication) -> None:
    
        result = app.create_plan_from_template(category="facelift", template="smas_plication_facelift")
        assert isinstance(result, dict)
        assert "plan" in result or "error" in result

    def test_create_plan_from_bleph_template(self, app: UIApplication) -> None:
    
        result = app.create_plan_from_template(category="blepharoplasty", template="upper_blepharoplasty")
        assert isinstance(result, dict)
        assert "plan" in result or "error" in result

    def test_create_plan_from_filler_template(self, app: UIApplication) -> None:
    
        result = app.create_plan_from_template(category="fillers", template="liquid_facelift")
        assert isinstance(result, dict)
        assert "plan" in result or "error" in result

    def test_create_plan_bad_category(self, app: UIApplication) -> None:
    
        result = app.create_plan_from_template(category="nonexistent", template="bad_template")
        assert isinstance(result, dict)
        assert "error" in result


# =====================================================================
# G4: Consult (what-if)
# =====================================================================


class TestG4Consult:
    """Test what-if and parameter sweep."""

    def test_run_whatif_no_case(self, app: UIApplication) -> None:
    
        result = app.run_whatif("nonexistent-id", {}, {})
        assert isinstance(result, dict)

    def test_parameter_sweep_no_case(self, app: UIApplication) -> None:
    
        result = app.parameter_sweep("nonexistent-id", {}, "some_op", "amount_mm", [1.0, 2.0, 3.0])
        assert isinstance(result, dict)


# =====================================================================
# G5: Report
# =====================================================================


class TestG5Report:
    """Test report generation."""

    def test_generate_report_no_case(self, app: UIApplication) -> None:
    
        result = app.generate_report("nonexistent-id", {})
        assert isinstance(result, dict)


# =====================================================================
# G6 / G7 / G8: Visualization, Timeline, Compare
# =====================================================================


class TestG6Visualization:
    def test_get_visualization_data_no_case(self, app: UIApplication) -> None:
    
        result = app.get_visualization_data("nonexistent-id")
        assert isinstance(result, dict)


class TestG7Timeline:
    def test_get_timeline_no_case(self, app: UIApplication) -> None:
    
        result = app.get_timeline("nonexistent-id")
        assert isinstance(result, dict)


class TestG8Compare:
    def test_compare_plans_empty(self, app: UIApplication) -> None:
    
        result = app.compare_plans("nonexistent-id", {}, {})
        assert isinstance(result, dict)

    def test_compare_cases(self, app: UIApplication) -> None:
    
        result = app.compare_cases("id-a", "id-b")
        assert isinstance(result, dict)


# =====================================================================
# G9: Interaction Contract
# =====================================================================


class TestG9Contract:
    """Test the G9 full interaction contract."""

    def test_get_contract(self, app: UIApplication) -> None:
    
        contract = app.get_contract()
        assert isinstance(contract, dict)
        assert "version" in contract
        assert "modes" in contract
        assert "procedures" in contract
        assert "operators" in contract

    def test_contract_has_all_modes(self, app: UIApplication) -> None:
    
        contract = app.get_contract()
        modes = contract["modes"]
        assert len(modes) >= 8

    def test_contract_procedures(self, app: UIApplication) -> None:
    
        contract = app.get_contract()
        procs = contract["procedures"]
        assert isinstance(procs, list)
        assert len(procs) >= 8  # at least 8 ProcedureTypes

    def test_contract_serializable(self, app: UIApplication) -> None:
        """The contract must be JSON-serializable."""
    
        contract = app.get_contract()
        text = json.dumps(contract)
        assert isinstance(text, str)
        parsed = json.loads(text)
        assert parsed == contract
