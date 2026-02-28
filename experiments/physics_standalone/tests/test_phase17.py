"""
Phase 17 Test Suite: Documentation Site, TensorRT Benchmarks, Flight Validation

Tests for static documentation generator, TensorRT integration benchmarking,
and flight data validation campaign infrastructure.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
import torch


# =============================================================================
# SITE GENERATOR TESTS
# =============================================================================

class TestSiteGenerator:
    """Tests for static documentation site generator."""
    
    def test_site_config_creation(self):
        """Test SiteConfig creation with defaults."""
        from ontic.infra.site.generator import SiteConfig
        
        config = SiteConfig()
        
        assert config.title == "Ontic Documentation"
        assert config.base_url == "/"
        assert config.theme == "physics_os"
    
    def test_site_config_with_options(self):
        """Test SiteConfig with custom options."""
        from ontic.infra.site.generator import SiteConfig
        
        config = SiteConfig(
            title="Custom Docs",
            base_url="/docs/",
            theme="custom",
            minify_html=False,
            generate_search_index=True,
        )
        
        assert config.title == "Custom Docs"
        assert config.base_url == "/docs/"
        assert config.generate_search_index is True
    
    def test_markdown_renderer_basic(self):
        """Test basic Markdown rendering."""
        from ontic.infra.site.generator import MarkdownRenderer, SiteConfig
        
        config = SiteConfig()
        renderer = MarkdownRenderer(config)
        
        md = "# Hello World"
        html = renderer.render(md)
        
        assert "<h1>" in html
        assert "Hello World" in html
    
    def test_markdown_renderer_code_blocks(self):
        """Test code block rendering."""
        from ontic.infra.site.generator import MarkdownRenderer, SiteConfig
        
        config = SiteConfig()
        renderer = MarkdownRenderer(config)
        
        md = "```python\nprint('hello')\n```"
        html = renderer.render(md)
        
        assert "<code" in html
        assert "print" in html
    
    def test_navigation_creation(self):
        """Test Navigation structure creation."""
        from ontic.infra.site.generator import Navigation, NavItem
        
        nav = Navigation()
        nav.add_item(NavItem(title="Home", path="/"))
        nav.add_item(NavItem(title="Docs", path="/docs/"))
        
        assert len(nav.items) == 2
        assert nav.items[0].title == "Home"
    
    def test_nav_item_to_dict(self):
        """Test NavItem dictionary conversion."""
        from ontic.infra.site.generator import NavItem
        
        item = NavItem(title="API", path="/api/", icon="📚")
        d = item.to_dict()
        
        assert d['title'] == "API"
        assert d['path'] == "/api/"
    
    def test_page_creation(self):
        """Test Page creation."""
        from ontic.infra.site.generator import Page, PageType
        
        page = Page(
            path="/index.html",
            title="Home",
            content="# Welcome",
            page_type=PageType.INDEX,
        )
        
        assert page.title == "Home"
        assert page.page_type == PageType.INDEX
    
    def test_page_toc_extraction(self):
        """Test automatic TOC extraction from page content."""
        from ontic.infra.site.generator import Page
        
        content = """# Title
        
## Section 1

Some content.

### Subsection

More content.

## Section 2
"""
        page = Page(path="/test.html", title="Test", content=content)
        
        # Should extract headers as TOC
        assert len(page.toc) > 0
    
    def test_site_builder_initialization(self):
        """Test SiteBuilder initialization."""
        from ontic.infra.site.generator import SiteBuilder, SiteConfig
        
        config = SiteConfig(title="Test")
        builder = SiteBuilder(config)
        
        assert builder.config.title == "Test"
    
    def test_build_result(self):
        """Test BuildResult structure."""
        from ontic.infra.site.generator import BuildResult
        
        result = BuildResult(
            success=True,
            pages_built=10,
            assets_processed=25,
            build_time=1.5,
            output_dir="_site",
        )
        
        assert result.success is True
        assert result.pages_built == 10


class TestThemes:
    """Tests for documentation themes."""
    
    def test_theme_colors(self):
        """Test ThemeColors structure."""
        from ontic.infra.site.themes import ThemeColors
        
        colors = ThemeColors(
            primary="#1a1a2e",
            secondary="#4a4e69",
        )
        
        assert colors.primary == "#1a1a2e"
    
    def test_theme_colors_to_css_vars(self):
        """Test CSS variable generation from colors."""
        from ontic.infra.site.themes import ThemeColors
        
        colors = ThemeColors()
        css = colors.to_css_vars()
        
        assert ":root" in css
        assert "--color-primary" in css
    
    def test_theme_typography(self):
        """Test ThemeTypography structure."""
        from ontic.infra.site.themes import ThemeTypography
        
        typo = ThemeTypography()
        
        assert "16px" in typo.font_size_base
        assert typo.line_height == 1.6
    
    def test_theme_typography_to_css(self):
        """Test CSS generation from typography."""
        from ontic.infra.site.themes import ThemeTypography
        
        typo = ThemeTypography()
        css = typo.to_css()
        
        assert "font-family" in css
        assert "font-size" in css
    
    def test_get_theme(self):
        """Test theme retrieval."""
        from ontic.infra.site.themes import get_theme
        
        theme = get_theme("default")
        
        assert theme is not None
    
    def test_list_themes(self):
        """Test listing available themes."""
        from ontic.infra.site.themes import list_themes
        
        themes = list_themes()
        
        assert isinstance(themes, list)
        assert "default" in themes


class TestSearch:
    """Tests for search functionality."""
    
    def test_tokenizer_basic(self):
        """Test basic tokenization."""
        from ontic.infra.site.search import Tokenizer
        
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("Hello World Example")
        
        assert len(tokens) >= 2
    
    def test_tokenizer_stop_words(self):
        """Test stop word removal."""
        from ontic.infra.site.search import Tokenizer
        
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("the quick brown fox")
        
        # "the" should be removed
        assert "the" not in tokens
    
    def test_search_index_add_document(self):
        """Test adding document to search index."""
        from ontic.infra.site.search import SearchIndex
        
        index = SearchIndex()
        index.add_document(
            doc_id="test1",
            content="This is a test document about tensors",
            title="Test Document",
        )
        
        assert len(index.documents) == 1
    
    def test_search_basic_query(self):
        """Test basic search query."""
        from ontic.infra.site.search import SearchIndex
        
        index = SearchIndex()
        index.add_document("doc1", "tensor networks quantum physics", "Tensor Intro")
        index.add_document("doc2", "classical mechanics physics formulas", "Physics")
        
        results = index.search("tensor")
        
        # Should find at least doc1
        assert len(results) >= 1
    
    def test_search_relevance_ranking(self):
        """Test search relevance ranking."""
        from ontic.infra.site.search import SearchIndex
        
        index = SearchIndex()
        index.add_document("doc1", "tensor", "Tensor")
        index.add_document("doc2", "tensor tensor tensor", "Many Tensors")
        
        results = index.search("tensor")
        
        assert len(results) >= 1
    
    def test_search_no_results(self):
        """Test search with no matches."""
        from ontic.infra.site.search import SearchIndex
        
        index = SearchIndex()
        index.add_document("doc1", "tensor networks", "Tensors")
        
        results = index.search("zzzznonexistent")
        
        assert len(results) == 0


class TestAssets:
    """Tests for asset management."""
    
    def test_asset_creation(self):
        """Test Asset dataclass creation."""
        from ontic.infra.site.assets import Asset, AssetType
        
        asset = Asset(
            path=Path("style.css"),
            asset_type=AssetType.CSS,
            content=b"body { color: black; }",
        )
        
        assert asset.asset_type == AssetType.CSS
        assert len(asset.content) > 0
    
    def test_css_minifier(self):
        """Test CSS minification."""
        from ontic.infra.site.assets import CSSMinifier
        
        minifier = CSSMinifier()
        
        css = """
        body {
            color: black;
            margin: 0;
        }
        """
        
        minified = minifier.minify(css)
        
        assert len(minified) <= len(css)
        assert "color" in minified
    
    def test_js_minifier(self):
        """Test JavaScript minification."""
        from ontic.infra.site.assets import JSMinifier
        
        minifier = JSMinifier()
        
        js = """
        function hello() {
            console.log("Hello");
        }
        """
        
        minified = minifier.minify(js)
        
        assert len(minified) <= len(js)


# =============================================================================
# TENSORRT BENCHMARK TESTS
# =============================================================================

class TestBenchmarkSuite:
    """Tests for TensorRT benchmark suite."""
    
    def test_benchmark_config_creation(self):
        """Test BenchmarkConfig creation."""
        from ontic.sim.benchmarks.benchmark_suite import BenchmarkConfig
        
        config = BenchmarkConfig()
        
        assert config.warmup_runs == 10
        assert config.benchmark_runs == 100
    
    def test_benchmark_config_custom(self):
        """Test BenchmarkConfig with custom values."""
        from ontic.sim.benchmarks.benchmark_suite import BenchmarkConfig
        
        config = BenchmarkConfig(
            warmup_runs=5,
            benchmark_runs=50,
            device="cpu",
        )
        
        assert config.warmup_runs == 5
        assert config.device == "cpu"
    
    def test_latency_stats(self):
        """Test LatencyStats calculation."""
        from ontic.sim.benchmarks.benchmark_suite import LatencyStats
        
        latencies = [1.0, 1.5, 1.2, 1.3, 1.1, 1.4, 1.25, 1.35, 1.15, 1.05]
        stats = LatencyStats.from_measurements(latencies)
        
        assert stats.mean_ms > 0
        assert stats.min_ms == 1.0
        assert stats.max_ms == 1.5
        assert stats.std_ms >= 0
    
    def test_latency_stats_to_dict(self):
        """Test LatencyStats dictionary conversion."""
        from ontic.sim.benchmarks.benchmark_suite import LatencyStats
        
        stats = LatencyStats.from_measurements([1.0, 1.1, 1.2])
        d = stats.to_dict()
        
        assert 'mean_ms' in d
        assert 'std_ms' in d
        assert 'p95_ms' in d
    
    def test_memory_stats(self):
        """Test MemoryStats structure."""
        from ontic.sim.benchmarks.benchmark_suite import MemoryStats
        
        stats = MemoryStats(
            peak_memory_mb=512.0,
            allocated_memory_mb=256.0,
            reserved_memory_mb=1024.0,
            model_size_mb=100.0,
        )
        
        assert stats.peak_memory_mb == 512.0
        assert stats.model_size_mb == 100.0
    
    def test_throughput_stats(self):
        """Test ThroughputStats structure."""
        from ontic.sim.benchmarks.benchmark_suite import ThroughputStats
        
        stats = ThroughputStats(
            samples_per_second=1000.0,
            batches_per_second=100.0,
            effective_batch_size=10,
        )
        
        assert stats.samples_per_second == 1000.0
        assert stats.effective_batch_size == 10
    
    def test_benchmark_result(self):
        """Test BenchmarkResult creation."""
        from ontic.sim.benchmarks.benchmark_suite import (
            BenchmarkResult, PrecisionMode, LatencyStats
        )
        
        result = BenchmarkResult(
            name="test",
            precision=PrecisionMode.FP32,
            batch_size=1,
            latency=LatencyStats.from_measurements([1.0, 1.1]),
        )
        
        assert result.name == "test"
        assert result.precision == PrecisionMode.FP32
        assert result.latency is not None
    
    def test_latency_benchmark(self):
        """Test LatencyBenchmark execution."""
        from ontic.sim.benchmarks.benchmark_suite import LatencyBenchmark, BenchmarkConfig
        
        config = BenchmarkConfig(
            warmup_runs=2,
            benchmark_runs=5,
            device="cpu",
        )
        
        model = torch.nn.Linear(10, 5)
        input_tensor = torch.randn(1, 10)
        
        benchmark = LatencyBenchmark(config)
        stats = benchmark.run(model, input_tensor)
        
        assert stats.mean_ms > 0


class TestProfiler:
    """Tests for TensorRT profiler."""
    
    def test_profile_config(self):
        """Test ProfileConfig creation."""
        from ontic.sim.benchmarks.profiler import ProfileConfig
        
        config = ProfileConfig()
        
        assert config is not None
    
    def test_tensorrt_profiler_initialization(self):
        """Test TensorRTProfiler initialization."""
        from ontic.sim.benchmarks.profiler import TensorRTProfiler, ProfileConfig
        
        config = ProfileConfig()
        profiler = TensorRTProfiler(config)
        
        assert profiler.config is not None
    
    def test_profile_result_structure(self):
        """Test ProfileResult structure."""
        from ontic.sim.benchmarks.profiler import ProfileResult
        
        result = ProfileResult(
            model_name="test_model",
            total_time_ms=10.0,
            layer_profiles=[],
        )
        
        assert result.model_name == "test_model"
        assert result.total_time_ms == 10.0


class TestBenchmarkReports:
    """Tests for benchmark reports."""
    
    def test_report_format_enum(self):
        """Test ReportFormat enum."""
        from ontic.sim.benchmarks.reports import ReportFormat
        
        assert ReportFormat.MARKDOWN.value == "md"
        assert ReportFormat.JSON.value == "json"
        assert ReportFormat.CSV.value == "csv"
    
    def test_benchmark_report_creation(self):
        """Test BenchmarkReport creation."""
        from ontic.sim.benchmarks.reports import BenchmarkReport
        
        report = BenchmarkReport(title="Test Report")
        
        assert report.title == "Test Report"


class TestPerformanceAnalysis:
    """Tests for performance analysis."""
    
    def test_optimization_recommendation(self):
        """Test OptimizationRecommendation structure."""
        from ontic.sim.benchmarks.analysis import (
            OptimizationRecommendation,
            OptimizationCategory,
            ImpactLevel,
            EffortLevel,
        )
        
        rec = OptimizationRecommendation(
            title="Use FP16 precision",
            category=OptimizationCategory.PRECISION,
            impact=ImpactLevel.HIGH,
            effort=EffortLevel.LOW,
            description="Switch to FP16 for 2x speedup",
        )
        
        assert rec.impact == ImpactLevel.HIGH
        assert rec.priority_score > 0
    
    def test_performance_analyzer_initialization(self):
        """Test PerformanceAnalyzer initialization."""
        from ontic.sim.benchmarks.analysis import PerformanceAnalyzer
        
        analyzer = PerformanceAnalyzer()
        
        assert analyzer is not None


# =============================================================================
# FLIGHT VALIDATION TESTS
# =============================================================================

class TestFlightDataLoader:
    """Tests for flight data loading."""
    
    def test_flight_data_source_enum(self):
        """Test FlightDataSource enum."""
        from ontic.sim.flight_validation.data_loader import FlightDataSource
        
        assert FlightDataSource.FLIGHT_TEST is not None
        assert FlightDataSource.WIND_TUNNEL is not None
    
    def test_flight_data_format_enum(self):
        """Test FlightDataFormat enum."""
        from ontic.sim.flight_validation.data_loader import FlightDataFormat
        
        assert FlightDataFormat.CSV.value == "csv"
        assert FlightDataFormat.JSON.value == "json"
    
    def test_flight_condition_creation(self):
        """Test FlightCondition creation."""
        from ontic.sim.flight_validation.data_loader import FlightCondition
        
        condition = FlightCondition(
            timestamp=0.0,
            altitude_m=10000.0,
            mach_number=0.8,
            velocity_m_s=250.0,
            angle_of_attack_deg=5.0,
        )
        
        assert condition.mach_number == 0.8
        assert condition.angle_of_attack_deg == 5.0
    
    def test_flight_condition_dynamic_pressure(self):
        """Test dynamic pressure calculation."""
        from ontic.sim.flight_validation.data_loader import FlightCondition
        
        condition = FlightCondition(
            timestamp=0.0,
            velocity_m_s=100.0,
            density_kg_m3=1.225,
        )
        
        q = condition.dynamic_pressure_pa()
        
        assert abs(q - 6125.0) < 1.0  # 0.5 * 1.225 * 100^2
    
    def test_aerodynamic_data_creation(self):
        """Test AerodynamicData creation."""
        from ontic.sim.flight_validation.data_loader import AerodynamicData
        
        aero = AerodynamicData(
            timestamp=0.0,
            cl=0.5,
            cd=0.02,
            cm=-0.1,
        )
        
        assert aero.cl == 0.5
        assert aero.cd == 0.02
    
    def test_flight_record_creation(self):
        """Test FlightRecord creation."""
        from ontic.sim.flight_validation.data_loader import (
            FlightRecord, FlightDataSource, FlightCondition
        )
        
        record = FlightRecord(
            record_id="test_flight",
            source=FlightDataSource.FLIGHT_TEST,
            vehicle_name="Test Vehicle",
        )
        
        record.conditions.append(FlightCondition(timestamp=0.0, mach_number=0.5))
        record.conditions.append(FlightCondition(timestamp=1.0, mach_number=0.6))
        
        assert len(record.conditions) == 2
    
    def test_flight_record_interpolation(self):
        """Test flight record time interpolation."""
        from ontic.sim.flight_validation.data_loader import (
            FlightRecord, FlightDataSource, FlightCondition
        )
        
        record = FlightRecord(
            record_id="test",
            source=FlightDataSource.FLIGHT_TEST,
        )
        
        record.conditions = [
            FlightCondition(timestamp=0.0, mach_number=0.5),
            FlightCondition(timestamp=1.0, mach_number=0.7),
        ]
        
        interp = record.get_condition_at_time(0.5)
        
        assert interp is not None
        assert abs(interp.mach_number - 0.6) < 0.01
    
    def test_flight_data_loader_csv(self):
        """Test CSV flight data loading."""
        from ontic.sim.flight_validation.data_loader import FlightDataLoader
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("time,mach,aoa,cl,cd\n")
            f.write("0.0,0.5,5.0,0.5,0.02\n")
            f.write("1.0,0.6,6.0,0.6,0.025\n")
            csv_path = f.name
        
        try:
            loader = FlightDataLoader()
            record = loader.load(csv_path)
            
            assert len(record.conditions) == 2
            assert len(record.aero_data) == 2
        finally:
            Path(csv_path).unlink()
    
    def test_flight_data_loader_json(self):
        """Test JSON flight data loading."""
        from ontic.sim.flight_validation.data_loader import FlightDataLoader
        
        data = {
            "record_id": "test_json",
            "vehicle_name": "Test",
            "conditions": [
                {"timestamp": 0.0, "mach_number": 0.5, "angle_of_attack_deg": 5.0},
                {"timestamp": 1.0, "mach_number": 0.6, "angle_of_attack_deg": 6.0},
            ],
            "aero_data": [
                {"timestamp": 0.0, "cl": 0.5, "cd": 0.02},
            ],
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            json_path = f.name
        
        try:
            loader = FlightDataLoader()
            record = loader.load(json_path)
            
            assert record.vehicle_name == "Test"
            assert len(record.conditions) == 2
        finally:
            Path(json_path).unlink()
    
    def test_parse_telemetry(self):
        """Test telemetry parsing."""
        from ontic.sim.flight_validation.data_loader import parse_telemetry
        
        telemetry = {
            "id": "telem_001",
            "frames": [
                {"time": 0.0, "altitude": 10000, "mach": 0.8, "aoa": 5.0},
                {"time": 0.1, "altitude": 10050, "mach": 0.81, "aoa": 5.1},
            ],
        }
        
        record = parse_telemetry(telemetry)
        
        assert len(record.conditions) == 2


class TestFlightComparison:
    """Tests for flight data comparison."""
    
    def test_validation_metric_enum(self):
        """Test ValidationMetric enum."""
        from ontic.sim.flight_validation.comparison import ValidationMetric
        
        assert ValidationMetric.MEAN_ERROR is not None
        assert ValidationMetric.RMS_ERROR is not None
    
    def test_comparison_status_enum(self):
        """Test ComparisonStatus enum."""
        from ontic.sim.flight_validation.comparison import ComparisonStatus
        
        assert ComparisonStatus.EXCELLENT is not None
        assert ComparisonStatus.POOR is not None
    
    def test_field_comparison_metrics(self):
        """Test FieldComparison metric calculations."""
        from ontic.sim.flight_validation.comparison import FieldComparison
        
        flight = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sim = np.array([1.1, 2.05, 3.0, 4.1, 5.05])
        
        comparison = FieldComparison(
            field_name="test",
            flight_data=flight,
            simulation_data=sim,
        )
        
        assert comparison.mean_error != 0
        assert comparison.rms_error > 0
        assert comparison.correlation > 0.99  # Very high correlation
    
    def test_field_comparison_status(self):
        """Test FieldComparison status determination."""
        from ontic.sim.flight_validation.comparison import FieldComparison, ComparisonStatus
        
        # Excellent case (< 2% error)
        flight = np.array([100.0, 100.0, 100.0])
        sim = np.array([101.0, 101.0, 101.0])  # 1% error
        
        comparison = FieldComparison(
            field_name="excellent",
            flight_data=flight,
            simulation_data=sim,
        )
        
        assert comparison.status == ComparisonStatus.EXCELLENT
    
    def test_temporal_comparison(self):
        """Test TemporalComparison."""
        from ontic.sim.flight_validation.comparison import TemporalComparison
        
        timestamps = np.array([0.0, 1.0, 2.0, 3.0])
        flight = np.array([1.0, 2.0, 3.0, 4.0])
        sim = np.array([1.1, 2.1, 3.1, 4.1])
        
        temporal = TemporalComparison(
            field_name="cl",
            timestamps=timestamps,
            flight_values=flight,
            simulation_values=sim,
        )
        
        assert len(temporal.instantaneous_errors) == 4
        assert temporal.field_comparison is not None
    
    def test_comparison_result(self):
        """Test ComparisonResult."""
        from ontic.sim.flight_validation.comparison import (
            ComparisonResult, FieldComparison
        )
        
        result = ComparisonResult(
            flight_record_id="flight_001",
            simulation_id="sim_001",
        )
        
        # Add field comparison
        comparison = FieldComparison(
            field_name="cl",
            flight_data=np.array([0.5, 0.5]),
            simulation_data=np.array([0.51, 0.51]),
        )
        
        result.add_field_comparison(comparison)
        
        assert len(result.field_comparisons) == 1
        assert result.validation_passed is True
    
    def test_flight_data_validator(self):
        """Test FlightDataValidator."""
        from ontic.sim.flight_validation.comparison import FlightDataValidator
        from ontic.sim.flight_validation.data_loader import (
            FlightRecord, FlightDataSource, AerodynamicData
        )
        
        # Create flight record
        record = FlightRecord(
            record_id="test",
            source=FlightDataSource.FLIGHT_TEST,
        )
        record.aero_data = [
            AerodynamicData(timestamp=0.0, cl=0.5, cd=0.02),
            AerodynamicData(timestamp=1.0, cl=0.6, cd=0.025),
        ]
        
        # Simulation data
        sim_data = {
            'cl': np.array([0.51, 0.61]),
            'cd': np.array([0.021, 0.026]),
        }
        
        validator = FlightDataValidator()
        result = validator.validate(record, sim_data)
        
        assert result.flight_record_id == "test"
        assert 'cl' in result.field_comparisons
    
    def test_compare_flight_data_function(self):
        """Test compare_flight_data convenience function."""
        from ontic.sim.flight_validation.comparison import compare_flight_data
        from ontic.sim.flight_validation.data_loader import (
            FlightRecord, FlightDataSource, AerodynamicData
        )
        
        record = FlightRecord(
            record_id="test",
            source=FlightDataSource.FLIGHT_TEST,
        )
        record.aero_data = [
            AerodynamicData(timestamp=0.0, cl=0.5, cd=0.02),
        ]
        
        sim_data = {'cl': np.array([0.5]), 'cd': np.array([0.02])}
        
        result = compare_flight_data(record, sim_data)
        
        assert result is not None


class TestUncertainty:
    """Tests for uncertainty quantification."""
    
    def test_uncertainty_source_enum(self):
        """Test UncertaintySource enum."""
        from ontic.sim.flight_validation.uncertainty import UncertaintySource
        
        assert UncertaintySource.SENSOR_ACCURACY is not None
        assert UncertaintySource.TURBULENCE_MODEL is not None
    
    def test_uncertainty_component(self):
        """Test UncertaintyComponent."""
        from ontic.sim.flight_validation.uncertainty import (
            UncertaintyComponent, UncertaintySource, UncertaintyType
        )
        
        component = UncertaintyComponent(
            name="sensor_accuracy",
            source=UncertaintySource.SENSOR_ACCURACY,
            uncertainty_type=UncertaintyType.EPISTEMIC,
            value=0.01,
        )
        
        assert component.value == 0.01
    
    def test_measurement_uncertainty(self):
        """Test MeasurementUncertainty."""
        from ontic.sim.flight_validation.uncertainty import (
            MeasurementUncertainty, UncertaintyComponent,
            UncertaintySource, UncertaintyType
        )
        
        unc = MeasurementUncertainty(
            parameter_name="cl",
            measured_value=0.5,
        )
        
        unc.add_component(UncertaintyComponent(
            name="sensor",
            source=UncertaintySource.SENSOR_ACCURACY,
            uncertainty_type=UncertaintyType.ALEATORY,
            value=0.01,
        ))
        
        assert unc.combined_standard_uncertainty > 0
        assert unc.expanded_uncertainty > 0
    
    def test_model_uncertainty(self):
        """Test ModelUncertainty."""
        from ontic.sim.flight_validation.uncertainty import ModelUncertainty
        
        unc = ModelUncertainty(
            parameter_name="cl",
            nominal_value=0.5,
            grid_uncertainty=0.02,
            turbulence_model_uncertainty=0.03,
        )
        
        assert unc.total_model_uncertainty > 0
        # Should be sqrt(0.02^2 + 0.03^2)
        expected = np.sqrt(0.02**2 + 0.03**2)
        assert abs(unc.total_model_uncertainty - expected) < 1e-10
    
    def test_validation_uncertainty(self):
        """Test ValidationUncertainty."""
        from ontic.sim.flight_validation.uncertainty import (
            ValidationUncertainty, MeasurementUncertainty, ModelUncertainty
        )
        
        meas = MeasurementUncertainty(
            parameter_name="cl",
            measured_value=0.5,
        )
        meas.expanded_uncertainty = 0.02
        
        model = ModelUncertainty(
            parameter_name="cl",
            nominal_value=0.51,
            grid_uncertainty=0.01,
        )
        
        val_unc = ValidationUncertainty(
            parameter_name="cl",
            measurement_uncertainty=meas,
            model_uncertainty=model,
            comparison_error=0.01,
        )
        
        assert val_unc.combined_uncertainty > 0
    
    def test_uncertainty_budget(self):
        """Test UncertaintyBudget."""
        from ontic.sim.flight_validation.uncertainty import (
            UncertaintyBudget, MeasurementUncertainty, ModelUncertainty
        )
        
        budget = UncertaintyBudget(
            name="Test Budget",
            description="Test uncertainty budget",
        )
        
        meas = MeasurementUncertainty(parameter_name="cl", measured_value=0.5)
        model = ModelUncertainty(parameter_name="cl", nominal_value=0.5)
        
        budget.add_measurement_uncertainty(meas)
        budget.add_model_uncertainty(model)
        
        assert len(budget.measurement_uncertainties) == 1
        assert len(budget.model_uncertainties) == 1
        assert 'cl' in budget.validation_uncertainties
    
    def test_uncertainty_propagation_linear(self):
        """Test linear uncertainty propagation."""
        from ontic.sim.flight_validation.uncertainty import UncertaintyPropagation
        
        prop = UncertaintyPropagation(method="linear")
        
        # Simple function: f(x, y) = x + y
        def func(x, y):
            return x + y
        
        inputs = {'x': 1.0, 'y': 2.0}
        uncertainties = {'x': 0.1, 'y': 0.2}
        
        value, unc = prop.propagate_linear(func, inputs, uncertainties)
        
        assert abs(value - 3.0) < 1e-6
        # Uncertainty should be sqrt(0.1^2 + 0.2^2) ≈ 0.224
        expected_unc = np.sqrt(0.1**2 + 0.2**2)
        assert abs(unc - expected_unc) < 0.01
    
    def test_uncertainty_propagation_monte_carlo(self):
        """Test Monte Carlo uncertainty propagation."""
        from ontic.sim.flight_validation.uncertainty import UncertaintyPropagation
        
        prop = UncertaintyPropagation(method="monte_carlo")
        
        def func(x, y):
            return x * y
        
        inputs = {'x': 2.0, 'y': 3.0}
        uncertainties = {'x': 0.1, 'y': 0.1}
        
        mean, std, samples = prop.propagate_monte_carlo(
            func, inputs, uncertainties, n_samples=1000, seed=42
        )
        
        assert abs(mean - 6.0) < 0.2  # Mean should be ~6
        assert std > 0
        assert len(samples) == 1000
    
    def test_grid_convergence_index(self):
        """Test Grid Convergence Index calculation."""
        from ontic.sim.flight_validation.uncertainty import GridConvergenceIndex
        
        gci = GridConvergenceIndex(refinement_ratio=2.0)
        
        # Test with known second-order convergence
        coarse = 1.1
        medium = 1.025
        fine = 1.00625  # Perfect second-order convergence
        
        result = gci.calculate_gci(coarse, medium, fine)
        
        assert 'fine_value' in result
        assert 'richardson_extrapolation' in result
        assert 'gci_fine' in result
    
    def test_calculate_measurement_uncertainty_function(self):
        """Test calculate_measurement_uncertainty convenience function."""
        from ontic.sim.flight_validation.uncertainty import calculate_measurement_uncertainty
        
        unc = calculate_measurement_uncertainty(
            measured_value=0.5,
            sensor_accuracy=0.01,
            calibration_uncertainty=0.005,
            repeatability=0.002,
        )
        
        assert len(unc.components) == 3
        assert unc.expanded_uncertainty > 0


class TestValidationReports:
    """Tests for validation reports."""
    
    def test_report_format_enum(self):
        """Test ReportFormat enum."""
        from ontic.sim.flight_validation.reports import ReportFormat
        
        assert ReportFormat.MARKDOWN.value == "md"
        assert ReportFormat.HTML.value == "html"
    
    def test_validation_level_enum(self):
        """Test ValidationLevel enum."""
        from ontic.sim.flight_validation.reports import ValidationLevel
        
        assert ValidationLevel.UNIT is not None
        assert ValidationLevel.SYSTEM is not None
    
    def test_validation_case_creation(self):
        """Test ValidationCase creation."""
        from ontic.sim.flight_validation.reports import ValidationCase
        
        case = ValidationCase(
            case_id="case_001",
            description="Test case",
            passed=True,
        )
        
        assert case.case_id == "case_001"
        assert case.passed is True
    
    def test_validation_campaign(self):
        """Test ValidationCampaign."""
        from ontic.sim.flight_validation.reports import (
            ValidationCampaign, ValidationCase, ValidationLevel
        )
        
        campaign = ValidationCampaign(
            campaign_id="campaign_001",
            name="Test Campaign",
            level=ValidationLevel.SYSTEM,
        )
        
        campaign.add_case(ValidationCase("c1", "Case 1", passed=True))
        campaign.add_case(ValidationCase("c2", "Case 2", passed=True))
        campaign.add_case(ValidationCase("c3", "Case 3", passed=False))
        
        assert campaign.total_cases == 3
        assert campaign.passed_cases == 2
        assert campaign.failed_cases == 1
    
    def test_validation_campaign_pass_rate(self):
        """Test campaign pass rate calculation."""
        from ontic.sim.flight_validation.reports import (
            ValidationCampaign, ValidationCase
        )
        
        campaign = ValidationCampaign(
            campaign_id="test",
            name="Test",
            pass_threshold=0.8,
        )
        
        # 4 out of 5 pass = 80%
        for i in range(4):
            campaign.add_case(ValidationCase(f"c{i}", f"Case {i}", passed=True))
        campaign.add_case(ValidationCase("c4", "Case 4", passed=False))
        
        assert campaign.get_pass_rate() == 0.8
        assert campaign.is_successful() is True  # 80% >= 80%
    
    def test_validation_report_markdown(self):
        """Test Markdown report generation."""
        from ontic.sim.flight_validation.reports import (
            ValidationCampaign, ValidationCase, ValidationReport, ReportFormat
        )
        
        campaign = ValidationCampaign(
            campaign_id="test",
            name="Test Campaign",
        )
        campaign.add_case(ValidationCase("c1", "Case 1", passed=True))
        
        report = ValidationReport(campaign)
        md = report.generate(ReportFormat.MARKDOWN)
        
        assert "Test Campaign" in md
        assert "Case 1" in md
    
    def test_validation_report_html(self):
        """Test HTML report generation."""
        from ontic.sim.flight_validation.reports import (
            ValidationCampaign, ValidationCase, ValidationReport, ReportFormat
        )
        
        campaign = ValidationCampaign(
            campaign_id="test",
            name="Test",
        )
        campaign.add_case(ValidationCase("c1", "Case 1", passed=True))
        
        report = ValidationReport(campaign)
        html = report.generate(ReportFormat.HTML)
        
        assert "<html>" in html
        assert "Test" in html
    
    def test_validation_report_json(self):
        """Test JSON report generation."""
        from ontic.sim.flight_validation.reports import (
            ValidationCampaign, ValidationCase, ValidationReport, ReportFormat
        )
        
        campaign = ValidationCampaign(
            campaign_id="test",
            name="Test",
        )
        campaign.add_case(ValidationCase("c1", "Case 1", passed=True))
        
        report = ValidationReport(campaign)
        json_str = report.generate(ReportFormat.JSON)
        
        data = json.loads(json_str)
        assert 'campaign' in data
        assert 'cases' in data
    
    def test_generate_validation_report_function(self):
        """Test generate_validation_report convenience function."""
        from ontic.sim.flight_validation.reports import (
            generate_validation_report, ValidationCampaign, ValidationCase
        )
        
        campaign = ValidationCampaign(campaign_id="test", name="Test")
        campaign.add_case(ValidationCase("c1", "Case 1", passed=True))
        
        md = generate_validation_report(campaign)
        
        assert "Test" in md


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
