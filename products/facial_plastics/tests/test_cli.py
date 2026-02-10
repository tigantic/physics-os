"""Tests for the CLI entry point — argument parsing and command dispatch.

These tests validate argument parsing and help output without requiring
live data or network connections.
"""

from __future__ import annotations

import pytest

from products.facial_plastics.cli import build_parser, main


# =====================================================================
# Argument parser construction
# =====================================================================


class TestBuildParser:
    """Test top-level parser construction."""

    def test_parser_creates(self):
        parser = build_parser()
        assert parser is not None
        assert parser.prog == "facial_plastics"

    def test_parser_has_subcommands(self):
        parser = build_parser()
        # Parsing with no args should raise (required subcommand)
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_help_flag(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0


# =====================================================================
# Ingest sub-commands
# =====================================================================


class TestIngestArgs:
    def test_ingest_dicom(self):
        parser = build_parser()
        args = parser.parse_args(["ingest", "dicom", "/path/to/dicom"])
        assert args.command == "ingest"
        assert args.ingest_type == "dicom"
        assert args.path == "/path/to/dicom"
        assert args.case_id is None

    def test_ingest_dicom_with_case_id(self):
        parser = build_parser()
        args = parser.parse_args(["ingest", "dicom", "/path/to/dicom", "--case-id", "abc123"])
        assert args.case_id == "abc123"

    def test_ingest_photo(self):
        parser = build_parser()
        args = parser.parse_args(["ingest", "photo", "a.jpg", "b.jpg"])
        assert args.command == "ingest"
        assert args.ingest_type == "photo"
        assert args.paths == ["a.jpg", "b.jpg"]

    def test_ingest_surface(self):
        parser = build_parser()
        args = parser.parse_args(["ingest", "surface", "scan.stl"])
        assert args.ingest_type == "surface"
        assert args.path == "scan.stl"

    def test_ingest_no_subtype(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["ingest"])


# =====================================================================
# Twin sub-commands
# =====================================================================


class TestTwinArgs:
    def test_twin_build(self):
        parser = build_parser()
        args = parser.parse_args(["twin", "build", "case-001"])
        assert args.command == "twin"
        assert args.twin_action == "build"
        assert args.case_id == "case-001"
        assert args.max_elements is None

    def test_twin_build_with_elements(self):
        parser = build_parser()
        args = parser.parse_args(["twin", "build", "case-001", "--max-elements", "50000"])
        assert args.max_elements == 50000

    def test_twin_inspect(self):
        parser = build_parser()
        args = parser.parse_args(["twin", "inspect", "case-001"])
        assert args.twin_action == "inspect"


# =====================================================================
# Plan sub-commands
# =====================================================================


class TestPlanArgs:
    def test_plan_create(self):
        parser = build_parser()
        args = parser.parse_args(["plan", "create", "--template", "rhinoplasty:dorsal_hump_reduction"])
        assert args.command == "plan"
        assert args.plan_action == "create"
        assert args.template == "rhinoplasty:dorsal_hump_reduction"
        assert args.output is None

    def test_plan_create_with_output(self):
        parser = build_parser()
        args = parser.parse_args(["plan", "create", "--template", "facelift:necklift", "-o", "plan.json"])
        assert args.output == "plan.json"

    def test_plan_compile(self):
        parser = build_parser()
        args = parser.parse_args(["plan", "compile", "--case-id", "abc", "--plan", "p.json"])
        assert args.plan_action == "compile"
        assert args.case_id == "abc"
        assert args.plan == "p.json"

    def test_plan_list_operators(self):
        parser = build_parser()
        args = parser.parse_args(["plan", "list-operators"])
        assert args.plan_action == "list-operators"

    def test_plan_list_templates(self):
        parser = build_parser()
        args = parser.parse_args(["plan", "list-templates"])
        assert args.plan_action == "list-templates"


# =====================================================================
# Simulate sub-command
# =====================================================================


class TestSimulateArgs:
    def test_simulate(self):
        parser = build_parser()
        args = parser.parse_args(["simulate", "case-001", "--plan", "p.json"])
        assert args.command == "simulate"
        assert args.case_id == "case-001"
        assert args.plan == "p.json"
        assert args.heal_days is None

    def test_simulate_with_heal(self):
        parser = build_parser()
        args = parser.parse_args(["simulate", "case-001", "--plan", "p.json", "--heal-days", "90"])
        assert args.heal_days == 90


# =====================================================================
# Report sub-command
# =====================================================================


class TestReportArgs:
    def test_report_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["report", "case-001"])
        assert args.command == "report"
        assert args.case_id == "case-001"
        assert args.format == "json"

    def test_report_html(self):
        parser = build_parser()
        args = parser.parse_args(["report", "case-001", "--format", "html", "-o", "r.html"])
        assert args.format == "html"
        assert args.output == "r.html"


# =====================================================================
# Serve sub-command
# =====================================================================


class TestServeArgs:
    def test_serve_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["serve"])
        assert args.command == "serve"
        assert args.port == 8420
        assert args.host == "127.0.0.1"

    def test_serve_custom_port(self):
        parser = build_parser()
        args = parser.parse_args(["serve", "--port", "9000", "--host", "0.0.0.0"])
        assert args.port == 9000
        assert args.host == "0.0.0.0"


# =====================================================================
# Library sub-commands
# =====================================================================


class TestLibraryArgs:
    def test_library_list(self):
        parser = build_parser()
        args = parser.parse_args(["library", "list"])
        assert args.command == "library"
        assert args.lib_action == "list"
        assert args.limit == 50

    def test_library_list_filtered(self):
        parser = build_parser()
        args = parser.parse_args(["library", "list", "--procedure", "facelift", "--limit", "10"])
        assert args.procedure == "facelift"
        assert args.limit == 10

    def test_library_curate(self):
        parser = build_parser()
        args = parser.parse_args(["library", "curate"])
        assert args.lib_action == "curate"

    def test_library_augment(self):
        parser = build_parser()
        args = parser.parse_args(["library", "augment", "--count", "20", "--procedure", "blepharoplasty"])
        assert args.lib_action == "augment"
        assert args.count == 20
        assert args.procedure == "blepharoplasty"


# =====================================================================
# Info sub-command
# =====================================================================


class TestInfoArgs:
    def test_info(self):
        parser = build_parser()
        args = parser.parse_args(["info"])
        assert args.command == "info"


# =====================================================================
# Global options
# =====================================================================


class TestGlobalOptions:
    def test_verbose(self):
        parser = build_parser()
        args = parser.parse_args(["-v", "info"])
        assert args.verbose is True

    def test_config(self):
        parser = build_parser()
        args = parser.parse_args(["--config", "/etc/fp.json", "info"])
        assert args.config == "/etc/fp.json"

    def test_data_root(self):
        parser = build_parser()
        args = parser.parse_args(["--data-root", "/data", "info"])
        assert args.data_root == "/data"
