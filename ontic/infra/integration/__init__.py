"""
Integration Module for Project The Ontic Engine.

Phase 16: System integration and deployment hardening including:
- End-to-end workflow orchestration
- Configuration management
- Logging and monitoring infrastructure
- Health checks and diagnostics

Components:
    - workflows: End-to-end simulation workflows
    - config: Configuration management system
    - monitoring: Logging and metrics collection
    - diagnostics: System health and debugging
"""

from .config import (  # Configuration management; Environment handling; Validation
    ConfigManager,
    ConfigSection,
    ConfigSource,
    Configuration,
    ConfigValidator,
    ConfigValue,
    EnvironmentConfig,
    get_config,
    load_config,
    merge_configs,
    save_config,
    validate_config,
)
from .diagnostics import (  # System diagnostics; Health checks; Debugging; Utilities
    DebugContext,
    DiagnosticsReport,
    GPUInfo,
    HealthCheck,
    HealthCheckResult,
    HealthStatus,
    MemoryInfo,
    Profiler,
    SystemHealthMonitor,
    SystemInfo,
    TracingSpan,
    check_system_health,
    get_system_info,
    run_diagnostics,
)
from .monitoring import (  # Logging; Metrics; Telemetry; Alerting; Convenience
    Alert,
    AlertManager,
    AlertSeverity,
    LogEntry,
    LogFormatter,
    LogLevel,
    Metric,
    MetricCollector,
    MetricsRegistry,
    MetricType,
    StructuredLogger,
    TelemetryCollector,
    TelemetryEvent,
    get_logger,
    log_error,
    log_info,
    log_warning,
    record_metric,
)
from .workflows import (  # Workflow definitions; Pre-built workflows; Workflow execution
    CFDSimulationWorkflow,
    DigitalTwinWorkflow,
    GuidanceWorkflow,
    ValidationWorkflow,
    Workflow,
    WorkflowEngine,
    WorkflowResult,
    WorkflowStage,
    WorkflowStatus,
    WorkflowStep,
    create_cfd_workflow,
    create_guidance_workflow,
    run_workflow,
)

__all__ = [
    # Workflows
    "WorkflowStep",
    "WorkflowStage",
    "Workflow",
    "WorkflowResult",
    "WorkflowStatus",
    "CFDSimulationWorkflow",
    "GuidanceWorkflow",
    "DigitalTwinWorkflow",
    "ValidationWorkflow",
    "WorkflowEngine",
    "run_workflow",
    "create_cfd_workflow",
    "create_guidance_workflow",
    # Config
    "ConfigSource",
    "ConfigValue",
    "ConfigSection",
    "Configuration",
    "ConfigManager",
    "EnvironmentConfig",
    "get_config",
    "load_config",
    "save_config",
    "merge_configs",
    "ConfigValidator",
    "validate_config",
    # Monitoring
    "LogLevel",
    "LogEntry",
    "LogFormatter",
    "StructuredLogger",
    "MetricType",
    "Metric",
    "MetricCollector",
    "MetricsRegistry",
    "TelemetryEvent",
    "TelemetryCollector",
    "AlertSeverity",
    "Alert",
    "AlertManager",
    "get_logger",
    "log_info",
    "log_warning",
    "log_error",
    "record_metric",
    # Diagnostics
    "SystemInfo",
    "GPUInfo",
    "MemoryInfo",
    "DiagnosticsReport",
    "HealthStatus",
    "HealthCheck",
    "HealthCheckResult",
    "SystemHealthMonitor",
    "DebugContext",
    "Profiler",
    "TracingSpan",
    "get_system_info",
    "run_diagnostics",
    "check_system_health",
]
