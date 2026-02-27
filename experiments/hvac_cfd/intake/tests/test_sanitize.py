"""
Tests for input sanitization utilities.

Per Article II (Test Discipline) - comprehensive security testing.
"""
import pytest
from staging.sanitize import (
    sanitize_filename,
    sanitize_project_name,
    sanitize_room_name,
    sanitize_path,
    sanitize_numeric,
    is_safe_extension,
)


class TestSanitizeFilename:
    """Tests for filename sanitization."""
    
    def test_path_traversal_blocked(self):
        """Path traversal attacks are blocked."""
        result = sanitize_filename("../../../etc/passwd")
        assert "../" not in result
        assert "\\" not in result
    
    def test_null_bytes_removed(self):
        """Null bytes are stripped."""
        result = sanitize_filename("file\x00.txt")
        assert "\x00" not in result
        assert result.endswith(".txt")
    
    def test_spaces_converted_to_underscores(self):
        """Spaces become underscores."""
        result = sanitize_filename("My File Name.pdf")
        assert " " not in result
        assert "_" in result
    
    def test_parentheses_removed(self):
        """Special characters removed."""
        result = sanitize_filename("file (1).pdf")
        assert "(" not in result
        assert ")" not in result
    
    def test_empty_returns_unnamed(self):
        """Empty string returns 'unnamed'."""
        assert sanitize_filename("") == "unnamed"
        assert sanitize_filename(None) == "unnamed"
    
    def test_length_limit_preserves_extension(self):
        """Long names truncated but extension preserved."""
        long_name = "a" * 200 + ".pdf"
        result = sanitize_filename(long_name)
        assert len(result) <= 100
        assert result.endswith(".pdf")
    
    def test_keeps_alphanumeric(self):
        """Normal alphanumeric names preserved."""
        result = sanitize_filename("project123.txt")
        assert result == "project123.txt"


class TestSanitizeProjectName:
    """Tests for project name sanitization."""
    
    def test_xss_blocked(self):
        """XSS attempts are sanitized."""
        result = sanitize_project_name("<script>alert(1)</script>")
        assert "<script>" not in result
        assert "</script>" not in result
    
    def test_shell_injection_blocked(self):
        """Shell metacharacters removed."""
        result = sanitize_project_name("Office; rm -rf /")
        assert ";" not in result
        assert "|" not in sanitize_project_name("test|command")
        assert "`" not in sanitize_project_name("test`whoami`")
    
    def test_empty_returns_default(self):
        """Empty string returns default name."""
        assert sanitize_project_name("") == "Unnamed Project"
        assert sanitize_project_name(None) == "Unnamed Project"
    
    def test_apostrophe_allowed(self):
        """Apostrophes in names are allowed."""
        result = sanitize_project_name("Bob's Office")
        assert "'" in result
    
    def test_normal_name_preserved(self):
        """Normal project names preserved."""
        result = sanitize_project_name("North Tower Conference Room")
        assert result == "North Tower Conference Room"


class TestSanitizeRoomName:
    """Tests for room name sanitization."""
    
    def test_empty_returns_default(self):
        """Empty room name returns default."""
        assert sanitize_room_name("") == "Main Room"
    
    def test_normal_name_preserved(self):
        """Normal room names preserved."""
        result = sanitize_room_name("Conference Room B")
        assert result == "Conference Room B"
    
    def test_length_limited(self):
        """Room names are length-limited."""
        long_name = "x" * 100
        result = sanitize_room_name(long_name)
        assert len(result) <= 50


class TestSanitizePath:
    """Tests for path sanitization."""
    
    def test_empty_returns_none(self):
        """Empty path returns None."""
        assert sanitize_path("") is None
        assert sanitize_path(None) is None
    
    def test_valid_path_returned(self):
        """Valid paths are resolved."""
        result = sanitize_path("/tmp")
        assert result is not None
        assert result.is_absolute()
    
    def test_relative_paths_resolved(self):
        """Relative paths are resolved to absolute."""
        result = sanitize_path(".")
        assert result is not None
        assert result.is_absolute()


class TestIsExtensionSafe:
    """Tests for extension validation."""
    
    def test_allowed_extensions(self):
        """Allowed extensions return True."""
        assert is_safe_extension("file.pdf") is True
        assert is_safe_extension("file.xlsx") is True
        assert is_safe_extension("file.txt") is True
    
    def test_blocked_extensions(self):
        """Dangerous extensions return False."""
        assert is_safe_extension("file.exe") is False
        assert is_safe_extension("file.sh") is False
        assert is_safe_extension("file.bat") is False
    
    def test_case_insensitive(self):
        """Extension check is case-insensitive."""
        assert is_safe_extension("file.PDF") is True
        assert is_safe_extension("file.XLSX") is True


class TestSanitizeNumeric:
    """Tests for numeric sanitization."""
    
    def test_valid_number(self):
        """Valid numbers are parsed."""
        assert sanitize_numeric("12.5") == 12.5
        assert sanitize_numeric("100") == 100.0
    
    def test_invalid_returns_default(self):
        """Truly invalid input returns default (when no numbers extractable)."""
        assert sanitize_numeric("invalid") == 0.0
        assert sanitize_numeric("no_numbers_here", default=10.0) == 10.0
    
    def test_min_clamping(self):
        """Values below min are clamped."""
        assert sanitize_numeric("-10", min_val=0) == 0.0
    
    def test_max_clamping(self):
        """Values above max are clamped."""
        assert sanitize_numeric("100", max_val=50) == 50.0
    
    def test_none_returns_default(self):
        """None input returns default."""
        assert sanitize_numeric(None, default=5.0) == 5.0
    
    def test_strips_non_numeric(self):
        """Non-numeric characters stripped."""
        assert sanitize_numeric("$12.50 USD") == 12.50


class TestIntegrationWithSubmitter:
    """Integration tests: sanitization in submitter."""
    
    def test_submitter_sanitizes_project_name(self):
        """Submitter sanitizes project name."""
        from staging.submitter import SimulationSubmitter
        
        submitter = SimulationSubmitter()
        payload = submitter.submit_job({
            "project_name": "<script>hack</script>",
            "room_width": 20.0,
            "room_length": 30.0,
            "room_height": 10.0,
            "inlet_cfm": 500.0,
        })
        
        assert "<script>" not in payload["project_name"]
    
    def test_submitter_sanitizes_room_name(self):
        """Submitter sanitizes room name."""
        from staging.submitter import SimulationSubmitter
        
        submitter = SimulationSubmitter()
        payload = submitter.submit_job({
            "room_name": "Room; DROP TABLE users;",
            "room_width": 20.0,
            "room_length": 30.0,
            "room_height": 10.0,
            "inlet_cfm": 500.0,
        })
        
        assert ";" not in payload["room_name"]
