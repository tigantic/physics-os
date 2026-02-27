"""
Integration tests for HVACDocumentParser with real files.

These tests use actual fixture files to verify end-to-end parsing.
Per Article II (Test Discipline) - test with real-world inputs.
"""
import pytest
from pathlib import Path
from staging.ingestor import HVACDocumentParser


@pytest.fixture
def parser():
    return HVACDocumentParser()


@pytest.fixture
def fixtures_dir():
    return Path(__file__).parent / "fixtures"


class TestTextFileIntegration:
    """Integration tests with .txt files."""
    
    def test_parse_sample_hvac_spec(self, parser, fixtures_dir):
        """Parse the sample HVAC spec text file."""
        spec_path = fixtures_dir / "sample_hvac_spec.txt"
        
        if not spec_path.exists():
            pytest.skip("sample_hvac_spec.txt not found")
        
        result = parser.parse(str(spec_path))
        
        # Should succeed
        assert result.get("success") is True
        assert "error" not in result
        
        fields = result["fields"]
        
        # Should extract key dimensions
        assert fields["room_width"]["value"] == 40.0
        assert fields["room_length"]["value"] == 30.0
        assert fields["room_height"]["value"] == 12.0
        
        # Should extract HVAC parameters
        assert fields["inlet_cfm"]["value"] == 2500.0
        assert fields["supply_temp"]["value"] == 55.0
        
        # Should extract diffuser count
        assert fields["vent_count"]["value"] == 4
        
        # Should be ready to submit
        assert result["summary"]["ready_to_submit"] is True


class TestExcelIntegration:
    """Integration tests with .xlsx files."""
    
    def test_parse_hvac_schedule(self, parser, fixtures_dir):
        """Parse the HVAC schedule Excel file."""
        schedule_path = fixtures_dir / "hvac_schedule.xlsx"
        
        if not schedule_path.exists():
            pytest.skip("hvac_schedule.xlsx not found")
        
        result = parser.parse(str(schedule_path))
        
        # Should succeed
        assert result.get("success") is True
        assert "error" not in result
        
        fields = result["fields"]
        
        # Should extract dimensions (first row: Conference A 30x25x10)
        assert fields["room_width"]["value"] == 30.0
        assert fields["room_length"]["value"] == 25.0
        assert fields["room_height"]["value"] == 10.0
        
        # Should extract CFM (first row: 1500)
        assert fields["inlet_cfm"]["value"] == 1500.0
    
    def test_excel_has_raw_text(self, parser, fixtures_dir):
        """Excel parsing should capture raw text for debugging."""
        schedule_path = fixtures_dir / "hvac_schedule.xlsx"
        
        if not schedule_path.exists():
            pytest.skip("hvac_schedule.xlsx not found")
        
        result = parser.parse(str(schedule_path))
        
        # Should have raw text representation
        assert "raw_text" in result
        assert len(result["raw_text"]) > 0
        
        # Raw text should contain sheet data
        assert "Conference A" in result["raw_text"] or "30" in result["raw_text"]


class TestPDFIntegration:
    """Integration tests with .pdf files."""
    
    def test_parse_cfd_report_if_exists(self, parser):
        """Parse the CFD report PDF if available."""
        pdf_path = Path(__file__).parent.parent.parent.parent / "Review" / "Apex_Architecture_Group_CR-2026-B_CFD_Report.pdf"
        
        if not pdf_path.exists():
            pytest.skip("CFD report PDF not found")
        
        result = parser.parse(str(pdf_path))
        
        # Should succeed (even if extraction is partial)
        assert result.get("success") is True
        assert "error" not in result
        
        # Should have raw text
        assert "raw_text" in result
        assert len(result["raw_text"]) > 100


class TestEdgeCasesIntegration:
    """Edge case integration tests."""
    
    def test_nonexistent_file(self, parser):
        """Nonexistent file returns error."""
        result = parser.parse("/path/that/does/not/exist.pdf")
        
        assert "error" in result
        assert "not found" in result["error"].lower()
    
    def test_empty_text_file(self, parser, fixtures_dir, tmp_path):
        """Empty text file returns defaults."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")
        
        result = parser.parse(str(empty_file))
        
        # Should still succeed with defaults
        assert result.get("success") is True
        
        # Should have default values
        assert result["fields"]["room_height"]["value"] == 9.0  # Default
        assert result["fields"]["supply_temp"]["value"] == 55.0  # Default
    
    def test_unknown_extension_treated_as_text(self, parser, tmp_path):
        """Unknown file extensions are treated as text (graceful handling)."""
        weird_file = tmp_path / "test.xyz"
        weird_file.write_text("Room: 20x30x10, CFM: 500")
        
        result = parser.parse(str(weird_file))
        
        # Should succeed by treating as text
        assert result.get("success") is True
        
        # Should extract what it can from content
        # (parser is graceful with unknown extensions)


class TestSubmitterIntegration:
    """End-to-end tests: ingest → submit."""
    
    def test_full_pipeline_text_to_payload(self, parser, fixtures_dir):
        """Full pipeline: parse text file → generate solver payload."""
        from staging.submitter import SimulationSubmitter
        
        spec_path = fixtures_dir / "sample_hvac_spec.txt"
        if not spec_path.exists():
            pytest.skip("sample_hvac_spec.txt not found")
        
        # Step 1: Parse
        parse_result = parser.parse(str(spec_path))
        assert parse_result.get("success") is True
        
        # Step 2: Build UI state
        ui_state = {}
        for field_name, field_data in parse_result["fields"].items():
            ui_state[field_name] = field_data["value"]
        
        # Step 3: Submit
        submitter = SimulationSubmitter()
        payload = submitter.submit_job(ui_state)
        
        # Should have valid payload
        assert "case_id" in payload
        assert "domain" in payload
        assert "boundary_conditions" in payload
        
        # Top-level units field is "SI"
        assert payload["units"] == "SI"
        
        # Domain should be in meters
        domain = payload["domain"]
        assert "width_x_m" in domain
        
        # 40 ft ≈ 12.19 m
        assert 12.0 < domain["width_x_m"] < 12.5
    
    def test_full_pipeline_excel_to_payload(self, parser, fixtures_dir):
        """Full pipeline: parse Excel → generate solver payload."""
        from staging.submitter import SimulationSubmitter
        
        schedule_path = fixtures_dir / "hvac_schedule.xlsx"
        if not schedule_path.exists():
            pytest.skip("hvac_schedule.xlsx not found")
        
        # Step 1: Parse
        parse_result = parser.parse(str(schedule_path))
        assert parse_result.get("success") is True
        
        # Step 2: Build UI state
        ui_state = {}
        for field_name, field_data in parse_result["fields"].items():
            ui_state[field_name] = field_data["value"]
        
        # Step 3: Submit
        submitter = SimulationSubmitter()
        payload = submitter.submit_job(ui_state)
        
        # Should have valid payload
        assert "case_id" in payload
        assert "boundary_conditions" in payload
        
        # Inlet should have velocity vector
        inlet = payload["boundary_conditions"]["inlet"]
        assert "velocity_vector_ms" in inlet
        
        # Velocity should be non-zero (downward Y component)
        velocity = inlet["velocity_vector_ms"]
        assert velocity[1] < 0  # Negative Y = downward


# =============================================================================
# SESSION ISOLATION TESTS
# =============================================================================

class TestSessionIsolation:
    """
    Tests verifying concurrent user isolation.
    
    Article III §3.2: Multiple users should not interfere.
    """
    
    def test_parser_instances_isolated(self, parser):
        """Each parser instance should be independent."""
        parser1 = HVACDocumentParser()
        parser2 = HVACDocumentParser()
        
        # Parse different content
        result1 = parser1.parse_text_content("Room 10x15, 300 CFM")
        result2 = parser2.parse_text_content("Room 20x25, 600 CFM")
        
        # Results should be different
        assert result1["fields"]["room_width"]["value"] == 10.0
        assert result2["fields"]["room_width"]["value"] == 20.0
        
        # Original parsers should still return same results
        result1_again = parser1.parse_text_content("Room 10x15, 300 CFM")
        assert result1_again["fields"]["room_width"]["value"] == 10.0
    
    def test_submitter_instances_isolated(self):
        """Each submitter instance should generate unique IDs."""
        from staging.submitter import SimulationSubmitter
        
        sub1 = SimulationSubmitter()
        sub2 = SimulationSubmitter()
        
        data = {"room_width": 10, "room_length": 15, "inlet_cfm": 300}
        
        payload1 = sub1.submit_job(data)
        payload2 = sub2.submit_job(data)
        
        # Case IDs must be unique
        assert payload1["case_id"] != payload2["case_id"]
    
    def test_multi_room_state_isolated(self, parser, fixtures_dir):
        """Multi-room selection should not affect other instances."""
        schedule_path = fixtures_dir / "hvac_schedule.xlsx"
        if not schedule_path.exists():
            pytest.skip("hvac_schedule.xlsx not found")
        
        parser1 = HVACDocumentParser()
        parser2 = HVACDocumentParser()
        
        result1 = parser1.parse(str(schedule_path))
        result2 = parser2.parse(str(schedule_path))
        
        # Select different rooms
        if result1.get("multi_room", {}).get("room_count", 0) >= 2:
            parser1.select_room(result1, 0)
            parser2.select_room(result2, 1)
            
            # They should now point to different rooms
            assert result1["multi_room"]["selected_index"] == 0
            assert result2["multi_room"]["selected_index"] == 1
