"""
Phase 7 — Productization & Ecosystem Hardening tests.

Covers:
• Export (VTU write/parse, CSV, JSON)
• Mesh import (raw arrays, GMSH format detection)
• Post-processing (probe, slice, integrate, FFT, gradient, histogram, stats)
• Deprecation (@deprecated, @since, VersionInfo, check_version_gate)
• Security (SBOM generation, license audit)
• SDK surface (imports, WorkflowBuilder config, recipes)
• Visualization (smoke test — backend-agnostic)

Run:  pytest tests/test_productization.py -v
"""

from __future__ import annotations

import json
import math
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
import torch
from torch import Tensor

from ontic.platform.data_model import (
    BCType,
    BoundaryCondition,
    FieldData,
    InitialCondition,
    Mesh,
    SimulationState,
    StructuredMesh,
    UnstructuredMesh,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _make_structured_1d(n: int = 64) -> StructuredMesh:
    """Create a 1-D structured mesh [0, 1]."""
    return StructuredMesh(shape=(n,), domain=((0.0, 1.0),))


def _make_structured_2d(nx: int = 16, ny: int = 16) -> StructuredMesh:
    """Create a 2-D structured mesh [0,1]²."""
    return StructuredMesh(shape=(nx, ny), domain=((0.0, 1.0), (0.0, 1.0)))


def _make_field_1d(name: str = "u", n: int = 64) -> FieldData:
    mesh = _make_structured_1d(n)
    return FieldData(
        name=name,
        data=torch.sin(torch.linspace(0, 2 * math.pi, n)),
        mesh=mesh,
    )


def _make_field_2d(name: str = "u", nx: int = 16, ny: int = 16) -> FieldData:
    mesh = _make_structured_2d(nx, ny)
    x = torch.linspace(0, 1, nx)
    y = torch.linspace(0, 1, ny)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    data = torch.sin(2 * math.pi * xx) * torch.cos(2 * math.pi * yy)
    return FieldData(
        name=name,
        data=data.reshape(-1),
        mesh=mesh,
    )


def _make_state_1d(n: int = 64) -> SimulationState:
    mesh = _make_structured_1d(n)
    field = _make_field_1d("u", n)
    return SimulationState(t=0.0, fields={"u": field}, mesh=mesh)


def _make_state_2d(nx: int = 16, ny: int = 16) -> SimulationState:
    mesh = _make_structured_2d(nx, ny)
    field = _make_field_2d("u", nx, ny)
    return SimulationState(t=0.0, fields={"u": field}, mesh=mesh)


# ═══════════════════════════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════════════════════════


class TestExportVTU:
    """VTU (VTK XML) export."""

    def test_export_vtu_1d(self, tmp_path: Path) -> None:
        from ontic.platform.export import export_vtu

        state = _make_state_1d(32)
        out = export_vtu(state, tmp_path / "test.vtu")
        assert out.exists()
        text = out.read_text()
        assert "<VTKFile" in text
        assert "VTU" in text.upper() or "UnstructuredGrid" in text

    def test_export_vtu_2d(self, tmp_path: Path) -> None:
        from ontic.platform.export import export_vtu

        state = _make_state_2d(8, 8)
        out = export_vtu(state, tmp_path / "test2d.vtu")
        assert out.exists()
        content = out.read_text()
        assert "NumberOfPoints" in content
        assert "NumberOfCells" in content

    def test_export_vtu_fields_filter(self, tmp_path: Path) -> None:
        from ontic.platform.export import export_vtu

        mesh = _make_structured_1d(16)
        fa = FieldData(name="a", data=torch.ones(16), mesh=mesh)
        fb = FieldData(name="b", data=torch.zeros(16), mesh=mesh)
        state = SimulationState(t=0.0, fields={"a": fa, "b": fb}, mesh=mesh)
        out = export_vtu(state, tmp_path / "filter.vtu", fields=["a"])
        text = out.read_text()
        assert 'Name="a"' in text
        # 'b' should not appear as a Data array name
        assert 'Name="b"' not in text


class TestExportCSV:
    """CSV export."""

    def test_export_csv_roundtrip(self, tmp_path: Path) -> None:
        from ontic.platform.export import export_csv

        data = {"energy": [1.0, 0.9, 0.8], "step": [0.0, 1.0, 2.0]}
        out = export_csv(data, tmp_path / "obs.csv")
        assert out.exists()
        lines = out.read_text().strip().split("\n")
        header = lines[0]
        assert "energy" in header
        assert "step" in header
        assert len(lines) == 4  # header + 3 data rows


class TestExportJSON:
    """JSON export."""

    def test_export_json(self, tmp_path: Path) -> None:
        from ontic.platform.export import export_json

        payload = {"solver": "RK4", "steps": 100, "converged": True}
        out = export_json(payload, tmp_path / "meta.json")
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded["solver"] == "RK4"
        assert loaded["steps"] == 100


class TestExportBundle:
    """ExportBundle convenience wrapper."""

    def test_bundle_multi_format(self, tmp_path: Path) -> None:
        from ontic.platform.export import ExportBundle

        state = _make_state_1d(32)
        bundle = ExportBundle(state, output_dir=tmp_path)
        outputs = bundle.all("test")
        assert len(outputs) >= 2
        for p in outputs:
            assert p.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# Mesh import
# ═══════════════════════════════════════════════════════════════════════════════


class TestMeshImport:
    """Mesh import (raw arrays, GMSH format detection)."""

    def test_import_raw_numpy_like(self) -> None:
        from ontic.platform.mesh_import import import_raw

        nodes = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        elements = torch.tensor([[0, 1, 2]], dtype=torch.long)
        mesh = import_raw(nodes, elements)
        assert isinstance(mesh, UnstructuredMesh)
        assert mesh.n_cells == 1
        assert mesh.ndim == 2

    def test_import_raw_tet(self) -> None:
        from ontic.platform.mesh_import import import_raw

        nodes = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        elements = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        mesh = import_raw(nodes, elements)
        assert mesh.n_cells == 1
        assert mesh.ndim == 3

    def test_detect_format_gmsh(self, tmp_path: Path) -> None:
        from ontic.platform.mesh_import import detect_mesh_format

        gmsh_v2 = "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        f = tmp_path / "test.msh"
        f.write_text(gmsh_v2)
        fmt = detect_mesh_format(f)
        assert fmt.startswith("gmsh")  # returns 'gmsh2' or 'gmsh4'

    def test_detect_format_unknown(self, tmp_path: Path) -> None:
        from ontic.platform.mesh_import import detect_mesh_format

        f = tmp_path / "test.xyz"
        f.write_text("some random content")
        fmt = detect_mesh_format(f)
        assert fmt == "unknown"

    def test_import_gmsh_v2(self, tmp_path: Path) -> None:
        from ontic.platform.mesh_import import import_gmsh

        gmsh_content = (
            "$MeshFormat\n"
            "2.2 0 8\n"
            "$EndMeshFormat\n"
            "$Nodes\n"
            "3\n"
            "1 0.0 0.0 0.0\n"
            "2 1.0 0.0 0.0\n"
            "3 0.5 1.0 0.0\n"
            "$EndNodes\n"
            "$Elements\n"
            "1\n"
            "1 2 0 1 2 3\n"
            "$EndElements\n"
        )
        f = tmp_path / "tri.msh"
        f.write_text(gmsh_content)
        mesh = import_gmsh(f)
        assert isinstance(mesh, UnstructuredMesh)
        assert mesh.n_cells == 1

    def test_import_gmsh_v4(self, tmp_path: Path) -> None:
        from ontic.platform.mesh_import import import_gmsh

        gmsh_content = (
            "$MeshFormat\n"
            "4.1 0 8\n"
            "$EndMeshFormat\n"
            "$Nodes\n"
            "1 3 1 3\n"
            "2 1 0 3\n"
            "1\n"
            "2\n"
            "3\n"
            "0.0 0.0 0.0\n"
            "1.0 0.0 0.0\n"
            "0.5 1.0 0.0\n"
            "$EndNodes\n"
            "$Elements\n"
            "1 1 1 1\n"
            "2 1 2 1\n"
            "1 1 2 3\n"
            "$EndElements\n"
        )
        f = tmp_path / "tri_v4.msh"
        f.write_text(gmsh_content)
        mesh = import_gmsh(f)
        assert isinstance(mesh, UnstructuredMesh)
        assert mesh.n_cells == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Post-processing
# ═══════════════════════════════════════════════════════════════════════════════


class TestPostProcess:
    """Post-processing operations on field data."""

    def test_probe_1d(self) -> None:
        from ontic.platform.postprocess import probe

        field = _make_field_1d("u", 64)
        val = probe(field, torch.tensor([0.5]))
        assert isinstance(val, Tensor)
        assert val.numel() == 1

    def test_probe_2d(self) -> None:
        from ontic.platform.postprocess import probe

        field = _make_field_2d("u", 16, 16)
        val = probe(field, torch.tensor([0.5, 0.5]))
        assert isinstance(val, Tensor)

    def test_slice_field(self) -> None:
        from ontic.platform.postprocess import slice_field

        field = _make_field_2d("T", 16, 16)
        coords, values = slice_field(field, axis=0, index=8)
        assert isinstance(coords, Tensor)
        assert isinstance(values, Tensor)
        assert values.shape[0] == 16

    def test_integrate_constant(self) -> None:
        from ontic.platform.postprocess import integrate

        mesh = _make_structured_1d(100)
        # Constant field = 1.0 on [0, 1] should integrate to ~1.0
        field = FieldData(name="c", data=torch.ones(100), mesh=mesh)
        val = integrate(field)
        assert abs(val.item() - 1.0) < 0.02  # Within 2%

    def test_field_statistics(self) -> None:
        from ontic.platform.postprocess import field_statistics, FieldStats

        field = _make_field_1d("u", 128)
        stats = field_statistics(field)
        assert isinstance(stats, FieldStats)
        assert stats.min <= stats.mean <= stats.max
        assert stats.std >= 0.0
        assert stats.l2_norm >= 0.0
        d = stats.to_dict()
        assert "min" in d
        assert "percentiles" in d

    def test_fft_field_1d(self) -> None:
        from ontic.platform.postprocess import fft_field

        field = _make_field_1d("u", 128)
        freqs, power = fft_field(field)
        assert freqs.shape[0] > 0
        assert power.shape[0] == freqs.shape[0]
        assert (power >= 0).all()

    def test_gradient_field_1d(self) -> None:
        from ontic.platform.postprocess import gradient_field

        # Linear field: u = x  should have gradient ≈ 1
        mesh = _make_structured_1d(64)
        data = torch.linspace(0, 1, 64)
        field = FieldData(name="u", data=data, mesh=mesh)
        grad = gradient_field(field)
        assert isinstance(grad, FieldData)
        # Interior points should be ~1.0
        interior = grad.data[2:-2]
        assert torch.allclose(interior, torch.ones_like(interior), atol=0.1)

    def test_histogram(self) -> None:
        from ontic.platform.postprocess import histogram

        field = _make_field_1d("u", 256)
        edges, counts = histogram(field, n_bins=10)
        assert counts.numel() > 0
        assert edges.numel() > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Deprecation / Versioning
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeprecation:
    """SemVer enforcement and API lifecycle."""

    def test_version_info_parse(self) -> None:
        from ontic.platform.deprecation import VersionInfo

        v = VersionInfo.parse("1.2.3")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3
        assert str(v) == "1.2.3"

    def test_version_info_comparison(self) -> None:
        from ontic.platform.deprecation import VersionInfo

        v1 = VersionInfo(1, 0, 0)
        v2 = VersionInfo(2, 0, 0)
        assert v1 < v2
        assert v2 > v1
        assert v1 == VersionInfo(1, 0, 0)

    def test_deprecated_warning(self) -> None:
        from ontic.platform.deprecation import deprecated, VersionInfo, PLATFORM_VERSION

        # Create a function deprecated at a future version
        future = VersionInfo(PLATFORM_VERSION.major + 1, 0, 0)

        @deprecated(removal_version=str(future), alternative="new_func")
        def old_func() -> int:
            return 42

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_func()
            assert result == 42
            assert len(w) == 1
            assert "deprecated" in str(w[0].message).lower()

    def test_deprecated_raises_when_overdue(self) -> None:
        from ontic.platform.deprecation import deprecated, PLATFORM_VERSION

        # Removal version == current → should raise
        @deprecated(removal_version=str(PLATFORM_VERSION), alternative="new_func")
        def dead_func() -> int:
            return 0

        with pytest.raises(RuntimeError, match="removed"):
            dead_func()

    def test_since_decorator(self) -> None:
        from ontic.platform.deprecation import since

        @since("1.5.0")
        def new_func() -> str:
            return "hello"

        assert new_func() == "hello"
        assert hasattr(new_func, "__since__")
        assert new_func.__since__ == "1.5.0"

    def test_check_version_gate(self) -> None:
        from ontic.platform.deprecation import check_version_gate

        # Should not raise if no overdue removals in loaded modules
        violations = check_version_gate()
        # Returns list of violations (may be empty — that's fine for clean code)
        assert isinstance(violations, list)


# ═══════════════════════════════════════════════════════════════════════════════
# Security
# ═══════════════════════════════════════════════════════════════════════════════


class TestSecurity:
    """SBOM generation, dependency audit, license compliance."""

    def test_generate_sbom(self, tmp_path: Path) -> None:
        from ontic.platform.security import generate_sbom, SBOM

        sbom = generate_sbom(output_path=tmp_path / "sbom.json")
        assert isinstance(sbom, SBOM)
        assert sbom.bom_format == "CycloneDX"
        assert len(sbom.components) > 0

        # Check file was written
        assert (tmp_path / "sbom.json").exists()
        loaded = json.loads((tmp_path / "sbom.json").read_text())
        assert loaded["bomFormat"] == "CycloneDX"

    def test_audit_dependencies(self) -> None:
        from ontic.platform.security import audit_dependencies

        result = audit_dependencies()
        assert hasattr(result, "findings")
        assert hasattr(result, "packages_scanned")
        assert result.packages_scanned > 0

    def test_license_audit(self) -> None:
        from ontic.platform.security import license_audit

        report = license_audit()
        assert hasattr(report, "entries")
        assert hasattr(report, "copyleft")
        assert hasattr(report, "unknown")
        summary = report.summary()
        assert isinstance(summary, str)
        assert "audit" in summary.lower() or "License" in summary


# ═══════════════════════════════════════════════════════════════════════════════
# SDK Surface
# ═══════════════════════════════════════════════════════════════════════════════


class TestSDKImports:
    """SDK public API surface: all re-exports are importable."""

    def test_sdk_version(self) -> None:
        from ontic.infra.sdk import __sdk_version__
        assert __sdk_version__ == "2.0.0"

    def test_workflow_builder_import(self) -> None:
        from ontic.infra.sdk import WorkflowBuilder
        assert WorkflowBuilder is not None

    def test_data_model_reexports(self) -> None:
        from ontic.infra.sdk import (
            Mesh,
            StructuredMesh,
            UnstructuredMesh,
            FieldData,
            SimulationState,
        )
        assert all(cls is not None for cls in [
            Mesh, StructuredMesh, UnstructuredMesh, FieldData, SimulationState,
        ])

    def test_protocol_reexports(self) -> None:
        from ontic.infra.sdk import (
            ProblemSpec,
            Solver,
            Observable,
            SolveResult,
        )
        assert all(cls is not None for cls in [
            ProblemSpec, Solver, Observable, SolveResult,
        ])

    def test_export_reexports(self) -> None:
        from ontic.infra.sdk import export_vtu, export_csv, export_json
        assert all(fn is not None for fn in [export_vtu, export_csv, export_json])

    def test_postprocess_reexports(self) -> None:
        from ontic.infra.sdk import (
            probe, slice_field, integrate, field_statistics,
            fft_field, gradient_field, histogram,
        )
        assert all(fn is not None for fn in [
            probe, slice_field, integrate, field_statistics,
            fft_field, gradient_field, histogram,
        ])

    def test_deprecation_reexports(self) -> None:
        from ontic.infra.sdk import PLATFORM_VERSION, VersionInfo, deprecated, since
        assert PLATFORM_VERSION is not None
        assert VersionInfo is not None

    def test_security_reexports(self) -> None:
        from ontic.infra.sdk import generate_sbom, audit_dependencies, license_audit
        assert all(fn is not None for fn in [
            generate_sbom, audit_dependencies, license_audit,
        ])


class TestWorkflowBuilderConfig:
    """WorkflowBuilder fluent API configuration."""

    def test_builder_domain(self) -> None:
        from ontic.infra.sdk import WorkflowBuilder

        b = WorkflowBuilder("test")
        b.domain(shape=(64,), extent=((0.0, 1.0),))
        b.field("u", ic="uniform", value=0.0)
        b.solver("PHY-I.1")
        wf = b.build()
        assert wf.name == "test"

    def test_builder_validation_no_mesh(self) -> None:
        from ontic.infra.sdk import WorkflowBuilder

        b = WorkflowBuilder("bad")
        b.field("u")
        b.solver("PHY-I.1")
        with pytest.raises(ValueError, match="mesh"):
            b.build()

    def test_builder_validation_no_solver(self) -> None:
        from ontic.infra.sdk import WorkflowBuilder

        b = WorkflowBuilder("bad")
        b.domain(shape=(64,), extent=((0.0, 1.0),))
        b.field("u")
        with pytest.raises(ValueError, match="solver"):
            b.build()

    def test_builder_validation_no_fields(self) -> None:
        from ontic.infra.sdk import WorkflowBuilder

        b = WorkflowBuilder("bad")
        b.domain(shape=(64,), extent=((0.0, 1.0),))
        b.solver("PHY-I.1")
        with pytest.raises(ValueError, match="field"):
            b.build()

    def test_builder_chaining(self) -> None:
        from ontic.infra.sdk import WorkflowBuilder

        wf = (
            WorkflowBuilder("chain_test")
            .domain(shape=(32,), extent=((0.0, 1.0),))
            .field("u", ic="uniform", value=1.0)
            .field("v", ic="uniform", value=0.0)
            .bc("u", "left", "dirichlet", value=0.0)
            .solver("PHY-II.1")
            .time(0.0, 1.0, dt=0.01)
            .observe("energy")
            .export("vtu", path="out")
            .export("csv", path="out")
            .seed(123)
            .meta(author="test")
            .build()
        )
        assert wf.name == "chain_test"


# ═══════════════════════════════════════════════════════════════════════════════
# Recipes
# ═══════════════════════════════════════════════════════════════════════════════


class TestRecipes:
    """Recipe registration and retrieval."""

    def test_list_recipes(self) -> None:
        from ontic.infra.sdk.recipes import list_recipes

        recipes = list_recipes()
        assert isinstance(recipes, list)
        assert len(recipes) >= 8  # We registered 8 built-in recipes

    def test_list_recipes_by_domain(self) -> None:
        from ontic.infra.sdk.recipes import list_recipes

        fluid = list_recipes(domain="fluid_dynamics")
        assert len(fluid) >= 2  # burgers_1d, sod_shock_tube

    def test_get_recipe_burgers(self) -> None:
        from ontic.infra.sdk.recipes import get_recipe

        builder = get_recipe("burgers_1d")
        assert builder._config.name == "burgers_1d"
        wf = builder.build()
        assert wf.name == "burgers_1d"

    def test_get_recipe_harmonic(self) -> None:
        from ontic.infra.sdk.recipes import get_recipe

        builder = get_recipe("harmonic_oscillator")
        assert builder._config.solver_id == "PHY-I.4"

    def test_get_recipe_with_overrides(self) -> None:
        from ontic.infra.sdk.recipes import get_recipe

        builder = get_recipe("burgers_1d", n_cells=512, reynolds=200.0)
        assert builder._config.mesh_shape == (512,)

    def test_get_recipe_unknown_raises(self) -> None:
        from ontic.infra.sdk.recipes import get_recipe

        with pytest.raises(KeyError, match="Unknown recipe"):
            get_recipe("nonexistent_solver_9999")

    def test_recipe_info_repr(self) -> None:
        from ontic.infra.sdk.recipes import list_recipes

        recipes = list_recipes()
        for r in recipes:
            assert "Recipe" in repr(r)


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization (smoke tests — may skip if matplotlib not available)
# ═══════════════════════════════════════════════════════════════════════════════


class TestVisualization:
    """Matplotlib-based plotting (smoke tests)."""

    def test_ensure_matplotlib(self) -> None:
        from ontic.platform.visualize import ensure_matplotlib

        # Just returns True/False — should not raise
        result = ensure_matplotlib()
        assert isinstance(result, bool)

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("matplotlib"),
        reason="matplotlib not installed",
    )
    def test_plot_field_1d(self, tmp_path: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        from ontic.platform.visualize import plot_field_1d

        field = _make_field_1d("u", 64)
        fig, ax = plot_field_1d(field, save_path=tmp_path / "field1d.png")
        assert (tmp_path / "field1d.png").exists()
        import matplotlib.pyplot as plt
        plt.close(fig)

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("matplotlib"),
        reason="matplotlib not installed",
    )
    def test_plot_field_2d(self, tmp_path: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        from ontic.platform.visualize import plot_field_2d

        field = _make_field_2d("T", 16, 16)
        fig, ax = plot_field_2d(field, save_path=tmp_path / "field2d.png")
        assert (tmp_path / "field2d.png").exists()
        import matplotlib.pyplot as plt
        plt.close(fig)

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("matplotlib"),
        reason="matplotlib not installed",
    )
    def test_plot_convergence(self, tmp_path: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        from ontic.platform.visualize import plot_convergence

        resolutions = [16, 32, 64, 128]
        errors = [0.01, 0.0025, 0.000625, 0.00015625]
        fig, ax = plot_convergence(
            resolutions, errors, save_path=tmp_path / "conv.png"
        )
        assert (tmp_path / "conv.png").exists()
        import matplotlib.pyplot as plt
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Platform __init__ surface
# ═══════════════════════════════════════════════════════════════════════════════


class TestPlatformInit:
    """Platform __init__ exports Phase 7 modules."""

    def test_version_updated(self) -> None:
        from ontic.platform import __version__

        assert __version__ == "2.0.0"

    def test_new_exports(self) -> None:
        import ontic.platform as plat

        # Export
        assert hasattr(plat, "export_vtu")
        assert hasattr(plat, "export_csv")
        assert hasattr(plat, "export_json")
        assert hasattr(plat, "ExportBundle")

        # Import
        assert hasattr(plat, "import_gmsh")
        assert hasattr(plat, "import_raw")
        assert hasattr(plat, "detect_mesh_format")

        # Post-processing
        assert hasattr(plat, "probe")
        assert hasattr(plat, "slice_field")
        assert hasattr(plat, "integrate")
        assert hasattr(plat, "field_statistics")
        assert hasattr(plat, "fft_field")
        assert hasattr(plat, "gradient_field")
        assert hasattr(plat, "histogram")

        # Deprecation
        assert hasattr(plat, "PLATFORM_VERSION")
        assert hasattr(plat, "VersionInfo")
        assert hasattr(plat, "deprecated")
        assert hasattr(plat, "since")
        assert hasattr(plat, "check_version_gate")

        # Security
        assert hasattr(plat, "generate_sbom")
        assert hasattr(plat, "audit_dependencies")
        assert hasattr(plat, "license_audit")

        # Visualization
        assert hasattr(plat, "plot_field_1d")
        assert hasattr(plat, "plot_field_2d")
        assert hasattr(plat, "plot_convergence")
        assert hasattr(plat, "ensure_matplotlib")
