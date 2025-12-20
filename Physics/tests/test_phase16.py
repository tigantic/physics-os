"""
Phase 16 Integration Tests: Integration & Deployment Hardening.

Tests for:
- Workflow orchestration
- Configuration management
- Monitoring and logging
- System diagnostics
"""

import pytest
import torch
import numpy as np
import sys
import tempfile
from pathlib import Path
import time
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestWorkflows:
    """Tests for workflow orchestration."""
    
    def test_workflow_step(self):
        """Test WorkflowStep creation and execution."""
        from tensornet.integration import WorkflowStep
        
        def add_step(ctx):
            return {'result': ctx['a'] + ctx['b']}
        
        step = WorkflowStep(
            name="add",
            executor=add_step,
            description="Add two numbers",
            required_inputs=['a', 'b'],
            outputs=['result'],
        )
        
        assert step.name == "add"
        assert step.validate_inputs({'a': 1, 'b': 2})
        assert not step.validate_inputs({'a': 1})
        
        outputs = step.execute({'a': 5, 'b': 3})
        assert outputs['result'] == 8
    
    def test_workflow_stage(self):
        """Test WorkflowStage."""
        from tensornet.integration import WorkflowStep, WorkflowStage
        
        stage = WorkflowStage(
            name="compute",
            description="Computation stage",
        )
        
        stage.add_step(WorkflowStep(
            name="step1",
            executor=lambda ctx: {'x': 1},
        ))
        stage.add_step(WorkflowStep(
            name="step2",
            executor=lambda ctx: {'y': 2},
        ))
        
        assert len(stage.steps) == 2
    
    def test_workflow_result(self):
        """Test WorkflowResult dataclass."""
        from tensornet.integration import WorkflowResult, WorkflowStatus
        
        result = WorkflowResult(
            workflow_name="test",
            status=WorkflowStatus.COMPLETED,
            context={'output': 42},
            step_results={'step1': {'status': 'completed'}},
            duration=1.5,
        )
        
        assert result.success is True
        assert result.get_output('output') == 42
        
        summary = result.summary()
        assert "test" in summary
    
    def test_workflow_engine(self):
        """Test WorkflowEngine execution."""
        from tensornet.integration import (
            Workflow, WorkflowStage, WorkflowStep, WorkflowEngine, WorkflowStatus
        )
        
        workflow = Workflow(name="test_workflow")
        
        stage = WorkflowStage(name="main")
        stage.add_step(WorkflowStep(
            name="init",
            executor=lambda ctx: {'initialized': True},
            outputs=['initialized'],
        ))
        stage.add_step(WorkflowStep(
            name="compute",
            executor=lambda ctx: {'result': 42},
            required_inputs=['initialized'],
            outputs=['result'],
        ))
        workflow.add_stage(stage)
        
        engine = WorkflowEngine(verbose=False)
        result = engine.run(workflow)
        
        assert result.status == WorkflowStatus.COMPLETED
        assert result.get_output('result') == 42
    
    def test_cfd_simulation_workflow(self):
        """Test CFDSimulationWorkflow."""
        from tensornet.integration import CFDSimulationWorkflow, WorkflowEngine
        
        workflow = CFDSimulationWorkflow({
            'nx': 20,
            'ny': 10,
            'n_steps': 5,
        })
        
        assert workflow.name == "CFD_Simulation"
        assert len(workflow.stages) == 4
        
        engine = WorkflowEngine(verbose=False)
        result = engine.run(workflow)
        
        assert result.success
        assert 'results' in result.context
        assert 'max_mach' in result.context['results']
    
    def test_guidance_workflow(self):
        """Test GuidanceWorkflow."""
        from tensornet.integration import GuidanceWorkflow, WorkflowEngine
        
        workflow = GuidanceWorkflow({
            'target': [50000.0, 0.0, 0.0],
        })
        
        assert workflow.name == "Guidance_Loop"
        
        engine = WorkflowEngine(verbose=False)
        result = engine.run(workflow)
        
        assert result.success
        assert 'control_command' in result.context
    
    def test_digital_twin_workflow(self):
        """Test DigitalTwinWorkflow."""
        from tensornet.integration import DigitalTwinWorkflow, WorkflowEngine
        
        workflow = DigitalTwinWorkflow()
        
        engine = WorkflowEngine(verbose=False)
        result = engine.run(workflow)
        
        assert result.success
        assert 'health_status' in result.context
    
    def test_run_workflow(self):
        """Test run_workflow convenience function."""
        from tensornet.integration import create_cfd_workflow, run_workflow
        
        workflow = create_cfd_workflow(nx=10, ny=5, n_steps=2)
        result = run_workflow(workflow, verbose=False)
        
        assert result.success
    
    def test_workflow_with_failure(self):
        """Test workflow handling of step failures."""
        from tensornet.integration import (
            Workflow, WorkflowStage, WorkflowStep, WorkflowEngine, WorkflowStatus
        )
        
        def failing_step(ctx):
            raise ValueError("Intentional failure")
        
        workflow = Workflow(name="failing_workflow")
        stage = WorkflowStage(name="main")
        stage.add_step(WorkflowStep(
            name="fail",
            executor=failing_step,
        ))
        workflow.add_stage(stage)
        
        engine = WorkflowEngine(verbose=False)
        result = engine.run(workflow)
        
        assert result.status == WorkflowStatus.FAILED
        assert result.error is not None


class TestConfig:
    """Tests for configuration management."""
    
    def test_config_value(self):
        """Test ConfigValue dataclass."""
        from tensornet.integration import ConfigValue, ConfigSource
        
        value = ConfigValue(
            key="cfl",
            value=0.5,
            default=0.5,
            dtype=float,
            description="CFL number",
        )
        
        assert value.validate() is True
        assert value.get() == 0.5
    
    def test_config_section(self):
        """Test ConfigSection."""
        from tensornet.integration import ConfigSection
        
        section = ConfigSection(name="solver")
        section.define("tolerance", 1e-10, float, "Convergence tolerance")
        section.define("max_iter", 1000, int, "Maximum iterations")
        
        assert section.get("tolerance") == 1e-10
        
        section.set("tolerance", 1e-8)
        assert section.get("tolerance") == 1e-8
        
        d = section.to_dict()
        assert 'tolerance' in d
    
    def test_configuration(self):
        """Test Configuration."""
        from tensornet.integration import Configuration, ConfigSection
        
        config = Configuration(name="test")
        
        cfd = ConfigSection(name="cfd")
        cfd.define("cfl", 0.5, float)
        config.add_section(cfd)
        
        assert config.get("cfd.cfl") == 0.5
        
        config.set("cfd.gamma", 1.4)
        assert config.get("cfd.gamma") == 1.4
    
    def test_config_manager(self):
        """Test ConfigManager."""
        from tensornet.integration import ConfigManager
        
        # Create new instance
        manager = ConfigManager()
        
        # Check defaults are loaded
        assert manager.get("cfd.cfl") == 0.5
        assert manager.get("solver.tolerance") == 1e-10
        
        # Override
        manager.set("cfd.cfl", 0.3)
        assert manager.get("cfd.cfl") == 0.3
    
    def test_config_file(self):
        """Test configuration file loading/saving."""
        from tensornet.integration import ConfigManager
        import json
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            
            # Save config
            manager = ConfigManager()
            manager.set("test.value", 42)
            manager.save(config_path)
            
            assert config_path.exists()
            
            # Load in new manager
            manager2 = ConfigManager()
            assert manager2.load_file(config_path) is True
            assert manager2.get("test.value") == 42
    
    def test_environment_config(self):
        """Test EnvironmentConfig."""
        from tensornet.integration import EnvironmentConfig
        
        # Set environment variable
        os.environ["HYPERTENSOR_CFD_CFL"] = "0.7"
        
        try:
            config = EnvironmentConfig.load()
            assert config.get("cfd.cfl") == 0.7
        finally:
            del os.environ["HYPERTENSOR_CFD_CFL"]
    
    def test_config_validation(self):
        """Test configuration validation."""
        from tensornet.integration import Configuration, ConfigSection, validate_config
        
        config = Configuration()
        cfd = ConfigSection(name="cfd")
        cfd.define("cfl", 1.5, float)  # Invalid - > 1
        config.add_section(cfd)
        
        deployment = ConfigSection(name="deployment")
        deployment.define("precision", "float128", str)  # Invalid
        config.add_section(deployment)
        
        errors = validate_config(config)
        
        assert len(errors) > 0
    
    def test_merge_configs(self):
        """Test merge_configs function."""
        from tensornet.integration import Configuration, ConfigSection, merge_configs
        
        base = Configuration(name="base")
        section1 = ConfigSection(name="section1")
        section1.set("key", "base_value")
        base.add_section(section1)
        
        override = Configuration(name="override")
        section1_override = ConfigSection(name="section1")
        section1_override.set("key", "override_value")
        override.add_section(section1_override)
        
        merged = merge_configs(base, override)
        
        assert merged.get("section1.key") == "override_value"


class TestMonitoring:
    """Tests for monitoring and logging."""
    
    def test_log_entry(self):
        """Test LogEntry dataclass."""
        from tensornet.integration import LogEntry, LogLevel
        
        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            message="Test message",
        )
        
        d = entry.to_dict()
        assert d['level'] == 'INFO'
        assert d['message'] == 'Test message'
        
        json_str = entry.to_json()
        assert 'INFO' in json_str
    
    def test_structured_logger(self):
        """Test StructuredLogger."""
        from tensornet.integration import StructuredLogger, LogLevel
        
        logger = StructuredLogger("test", level=LogLevel.DEBUG)
        
        # Replace handler to capture output
        captured = []
        logger.handlers = [lambda e: captured.append(e)]
        
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        
        assert len(captured) == 3
        assert captured[1].level == LogLevel.INFO
    
    def test_log_level_filtering(self):
        """Test log level filtering."""
        from tensornet.integration import StructuredLogger, LogLevel
        
        logger = StructuredLogger("test", level=LogLevel.WARNING)
        
        captured = []
        logger.handlers = [lambda e: captured.append(e)]
        
        logger.debug("Debug")
        logger.info("Info")
        logger.warning("Warning")
        logger.error("Error")
        
        assert len(captured) == 2  # Only WARNING and ERROR
    
    def test_metric_collector(self):
        """Test MetricCollector."""
        from tensornet.integration import MetricCollector, MetricType
        
        collector = MetricCollector("test")
        
        collector.record("accuracy", 0.95)
        collector.record("accuracy", 0.96)
        collector.record("accuracy", 0.97)
        
        latest = collector.get_latest("accuracy")
        assert latest is not None
        assert latest.value == 0.97
        
        all_values = collector.get_all("accuracy")
        assert len(all_values) == 3
        
        summary = collector.summary()
        assert 'accuracy' in summary
        assert summary['accuracy']['mean'] == pytest.approx(0.96, abs=0.01)
    
    def test_counter_metric(self):
        """Test counter metrics."""
        from tensornet.integration import MetricCollector
        
        collector = MetricCollector()
        
        collector.increment("requests")
        collector.increment("requests")
        collector.increment("requests", 5)
        
        latest = collector.get_latest("requests")
        assert latest.value == 7
    
    def test_timing_metric(self):
        """Test timing metrics."""
        from tensornet.integration import MetricCollector, MetricType
        
        collector = MetricCollector()
        
        collector.timing("operation", 0.5)
        collector.timing("operation", 0.6)
        
        all_timings = collector.get_all("operation")
        assert len(all_timings) == 2
        assert all_timings[0].metric_type == MetricType.TIMER
    
    def test_metrics_registry(self):
        """Test MetricsRegistry."""
        from tensornet.integration import MetricsRegistry
        
        MetricsRegistry.record("test_metric", 42, collector="test")
        
        summary = MetricsRegistry.summary()
        assert 'test' in summary
    
    def test_telemetry_event(self):
        """Test TelemetryEvent."""
        from tensornet.integration import TelemetryEvent
        
        event = TelemetryEvent(
            name="operation",
            start_time=time.time(),
        )
        
        time.sleep(0.01)
        event.finish()
        
        assert event.duration is not None
        assert event.duration >= 0.01
    
    def test_telemetry_collector(self):
        """Test TelemetryCollector."""
        from tensornet.integration import TelemetryCollector
        
        collector = TelemetryCollector()
        
        event = collector.start_event("test_op", key="value")
        time.sleep(0.01)
        completed = collector.end_event(event.event_id)
        
        assert completed is not None
        assert completed.duration >= 0.01
        
        events = collector.get_events("test_op")
        assert len(events) == 1
    
    def test_alert_manager(self):
        """Test AlertManager."""
        from tensornet.integration import AlertManager, AlertSeverity
        
        manager = AlertManager()
        
        alert = manager.raise_alert(
            "high_memory",
            AlertSeverity.WARNING,
            "Memory usage above 80%",
            usage=85,
        )
        
        assert alert.severity == AlertSeverity.WARNING
        
        active = manager.get_active()
        assert len(active) == 1
        
        manager.resolve("high_memory")
        
        active = manager.get_active()
        assert len(active) == 0
    
    def test_convenience_functions(self):
        """Test logging convenience functions."""
        from tensornet.integration import get_logger, log_info, log_warning
        
        logger = get_logger("test_convenience")
        assert logger is not None
        
        # These should not raise
        log_info("Test info")
        log_warning("Test warning")


class TestDiagnostics:
    """Tests for system diagnostics."""
    
    def test_memory_info(self):
        """Test MemoryInfo dataclass."""
        from tensornet.integration import MemoryInfo
        
        info = MemoryInfo(
            total=16 * 1024**3,
            available=8 * 1024**3,
            used=8 * 1024**3,
            percent=50.0,
        )
        
        d = info.to_dict()
        assert d['total_gb'] == 16.0
        assert d['percent'] == 50.0
    
    def test_gpu_info(self):
        """Test GPUInfo dataclass."""
        from tensornet.integration import GPUInfo
        
        info = GPUInfo(
            device_id=0,
            name="Test GPU",
            memory_total=8 * 1024**3,
            memory_used=4 * 1024**3,
            memory_free=4 * 1024**3,
        )
        
        assert info.memory_utilization == 50.0
        
        d = info.to_dict()
        assert d['name'] == "Test GPU"
    
    def test_get_system_info(self):
        """Test get_system_info function."""
        from tensornet.integration import get_system_info
        
        info = get_system_info()
        
        assert info.platform != ""
        assert info.python_version != ""
        assert info.cpu_count > 0
        
        summary = info.summary()
        assert "Python:" in summary
    
    def test_health_check(self):
        """Test HealthCheck."""
        from tensornet.integration import HealthCheck, HealthStatus
        
        def check_fn():
            return {
                'status': HealthStatus.HEALTHY,
                'message': 'All good',
            }
        
        check = HealthCheck("test_check", check_fn)
        result = check.run()
        
        assert result.status == HealthStatus.HEALTHY
        assert result.duration >= 0  # Can be 0 on fast systems
    
    def test_health_check_failure(self):
        """Test HealthCheck handling failures."""
        from tensornet.integration import HealthCheck, HealthStatus
        
        def failing_check():
            raise RuntimeError("Check failed")
        
        check = HealthCheck("failing", failing_check)
        result = check.run()
        
        assert result.status == HealthStatus.UNHEALTHY
        assert "Check failed" in result.message
    
    def test_system_health_monitor(self):
        """Test SystemHealthMonitor."""
        from tensornet.integration import SystemHealthMonitor, HealthStatus
        
        monitor = SystemHealthMonitor()
        
        # Check default checks are registered
        assert "memory" in monitor.checks
        assert "imports" in monitor.checks
        
        # Run all checks
        results = monitor.run_all()
        
        assert len(results) > 0
        
        # Get summary
        summary = monitor.summary()
        assert "Status:" in summary
    
    def test_diagnostics_report(self):
        """Test DiagnosticsReport."""
        from tensornet.integration import (
            DiagnosticsReport, SystemInfo, MemoryInfo, HealthCheckResult, HealthStatus
        )
        
        sys_info = SystemInfo(
            platform="Test",
            python_version="3.11",
            pytorch_version="2.0",
            numpy_version="1.24",
            cpu_count=4,
            memory=MemoryInfo(total=16e9, available=8e9, used=8e9, percent=50),
        )
        
        health_results = {
            'test': HealthCheckResult("test", HealthStatus.HEALTHY, "OK"),
        }
        
        report = DiagnosticsReport(
            system_info=sys_info,
            health_results=health_results,
            issues=["Test issue"],
            recommendations=["Test recommendation"],
        )
        
        md = report.to_markdown()
        assert "# Diagnostics Report" in md
        assert "Test issue" in md
    
    def test_run_diagnostics(self):
        """Test run_diagnostics function."""
        from tensornet.integration import run_diagnostics
        
        report = run_diagnostics()
        
        assert report.system_info is not None
        assert len(report.health_results) > 0
    
    def test_check_system_health(self):
        """Test check_system_health function."""
        from tensornet.integration import check_system_health
        
        # Should return True on a healthy system
        result = check_system_health()
        assert isinstance(result, bool)
    
    def test_debug_context(self):
        """Test DebugContext."""
        from tensornet.integration import DebugContext
        
        ctx = DebugContext.capture("test", x=1, y="hello")
        
        assert ctx.name == "test"
        assert ctx.variables['x'] == 1
        
        d = ctx.to_dict()
        assert 'variables' in d
    
    def test_profiler(self):
        """Test Profiler."""
        from tensornet.integration import Profiler
        
        profiler = Profiler()
        
        profiler.start("operation")
        time.sleep(0.01)
        duration = profiler.stop("operation")
        
        assert duration >= 0.01
        
        summary = profiler.summary()
        assert 'operation' in summary
        assert summary['operation']['count'] == 1
    
    def test_profiler_context_manager(self):
        """Test Profiler context manager."""
        from tensornet.integration import Profiler
        
        profiler = Profiler()
        
        with profiler.profile("section"):
            time.sleep(0.01)
        
        summary = profiler.summary()
        assert 'section' in summary
    
    def test_tracing_span(self):
        """Test TracingSpan."""
        from tensornet.integration import TracingSpan
        
        span = TracingSpan(name="test_span", trace_id="trace123")
        
        span.tag("key", "value")
        span.log("event", data=42)
        
        time.sleep(0.01)
        span.finish()
        
        assert span.duration >= 0.01
        assert span.tags['key'] == 'value'
        assert len(span.logs) == 1


class TestIntegrationImports:
    """Test that all integration module exports work."""
    
    def test_workflow_imports(self):
        """Test workflow imports."""
        from tensornet.integration import (
            WorkflowStep,
            WorkflowStage,
            Workflow,
            WorkflowResult,
            WorkflowStatus,
            CFDSimulationWorkflow,
            GuidanceWorkflow,
            DigitalTwinWorkflow,
            ValidationWorkflow,
            WorkflowEngine,
            run_workflow,
            create_cfd_workflow,
            create_guidance_workflow,
        )
        
        assert WorkflowEngine is not None
        assert CFDSimulationWorkflow is not None
    
    def test_config_imports(self):
        """Test config imports."""
        from tensornet.integration import (
            ConfigSource,
            ConfigValue,
            ConfigSection,
            Configuration,
            ConfigManager,
            EnvironmentConfig,
            get_config,
            load_config,
            save_config,
            merge_configs,
            ConfigValidator,
            validate_config,
        )
        
        assert ConfigManager is not None
    
    def test_monitoring_imports(self):
        """Test monitoring imports."""
        from tensornet.integration import (
            LogLevel,
            LogEntry,
            LogFormatter,
            StructuredLogger,
            MetricType,
            Metric,
            MetricCollector,
            MetricsRegistry,
            TelemetryEvent,
            TelemetryCollector,
            AlertSeverity,
            Alert,
            AlertManager,
            get_logger,
            log_info,
            log_warning,
            log_error,
            record_metric,
        )
        
        assert StructuredLogger is not None
        assert MetricsRegistry is not None
    
    def test_diagnostics_imports(self):
        """Test diagnostics imports."""
        from tensornet.integration import (
            SystemInfo,
            GPUInfo,
            MemoryInfo,
            DiagnosticsReport,
            HealthStatus,
            HealthCheck,
            HealthCheckResult,
            SystemHealthMonitor,
            DebugContext,
            Profiler,
            TracingSpan,
            get_system_info,
            run_diagnostics,
            check_system_health,
        )
        
        assert SystemHealthMonitor is not None
        assert Profiler is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
