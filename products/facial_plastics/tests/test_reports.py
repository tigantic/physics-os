"""Tests for reports module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from products.facial_plastics.reports import (
    ReportBuilder,
    ReportMetadata,
    ReportSection,
)


class TestReportSection:
    """Test report section data structure."""

    def test_construction(self):
        section = ReportSection(
            title="Patient Demographics",
            content={"name": "John Doe", "age": 35},
        )
        assert section.title == "Patient Demographics"
        assert section.content["age"] == 35

    def test_to_dict(self):
        section = ReportSection(title="Summary", content={"key": "val"})
        d = section.to_dict()
        assert d["title"] == "Summary"
        assert d["content"]["key"] == "val"


class TestReportBuilder:
    """Test report builder fluent API."""

    _DISCLAIMERS = [
        "Simulation results are for educational purposes only.",
        "Not a substitute for clinical judgement.",
    ]

    def test_empty_report(self):
        builder = ReportBuilder(case_id="test001")
        result = builder.build_json()
        assert isinstance(result, dict)

    def test_add_disclaimers(self):
        builder = ReportBuilder(case_id="test001")
        builder.add_disclaimers(self._DISCLAIMERS)
        result = builder.build_json()
        raw = str(result)
        assert "Simulation" in raw or "disclaim" in raw.lower()

    def test_json_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            builder = ReportBuilder(case_id="test001")
            builder.add_disclaimers(self._DISCLAIMERS)
            path = Path(td) / "report.json"
            builder.save_json(path)
            assert path.exists()
            assert path.stat().st_size > 0

    def test_markdown_output(self):
        builder = ReportBuilder(case_id="test001")
        builder.add_disclaimers(self._DISCLAIMERS)
        md = builder.build_markdown()
        assert isinstance(md, str)
        assert len(md) > 0

    def test_html_output(self):
        builder = ReportBuilder(case_id="test001")
        builder.add_disclaimers(self._DISCLAIMERS)
        html = builder.build_html()
        assert "<!doctype" in html.lower() or "<html" in html.lower()

    def test_save_markdown(self):
        with tempfile.TemporaryDirectory() as td:
            builder = ReportBuilder(case_id="test_md")
            builder.add_disclaimers(self._DISCLAIMERS)
            path = Path(td) / "report.md"
            builder.save_markdown(path)
            assert path.exists()

    def test_save_html(self):
        with tempfile.TemporaryDirectory() as td:
            builder = ReportBuilder(case_id="test_html")
            builder.add_disclaimers(self._DISCLAIMERS)
            path = Path(td) / "report.html"
            builder.save_html(path)
            assert path.exists()

    def test_fluent_api_chaining(self):
        """ReportBuilder methods should return self for chaining."""
        builder = ReportBuilder(case_id="chain_test")
        result = builder.add_disclaimers(self._DISCLAIMERS)
        assert result is builder
