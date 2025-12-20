"""
Integration Module for Project HyperTensor.

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

from .workflows import (
    # Workflow definitions
    WorkflowStep,
    WorkflowStage,
    Workflow,
    WorkflowResult,
    WorkflowStatus,
    # Pre-built workflows
    CFDSimulationWorkflow,
    GuidanceWorkflow,
    DigitalTwinWorkflow,
    ValidationWorkflow,
    # Workflow execution
    WorkflowEngine,
    run_workflow,
    create_cfd_workflow,
    create_guidance_workflow,
)

from .config import (
    # Configuration management
    ConfigSource,
    ConfigValue,
    ConfigSection,
    Configuration,
    ConfigManager,
    # Environment handling
    EnvironmentConfig,
    get_config,
    load_config,
    save_config,
    merge_configs,
    # Validation
    ConfigValidator,
    validate_config,
)

from .monitoring import (
    # Logging
    LogLevel,
    LogEntry,
    LogFormatter,
    StructuredLogger,
    # Metrics
    MetricType,
    Metric,
    MetricCollector,
    MetricsRegistry,
    # Telemetry
    TelemetryEvent,
    TelemetryCollector,
    # Alerting
    AlertSeverity,
    Alert,
    AlertManager,
    # Convenience
    get_logger,
    log_info,
    log_warning,
    log_error,
    record_metric,
)

from .diagnostics import (
    # System diagnostics
    SystemInfo,
    GPUInfo,
    MemoryInfo,
    DiagnosticsReport,
    # Health checks
    HealthStatus,
    HealthCheck,
    HealthCheckResult,
    SystemHealthMonitor,
    # Debugging
    DebugContext,
    Profiler,
    TracingSpan,
    # Utilities
    get_system_info,
    run_diagnostics,
    check_system_health,
)

__all__ = [
    # Workflows
    'WorkflowStep',
    'WorkflowStage',
    'Workflow',
    'WorkflowResult',
    'WorkflowStatus',
    'CFDSimulationWorkflow',
    'GuidanceWorkflow',
    'DigitalTwinWorkflow',
    'ValidationWorkflow',
    'WorkflowEngine',
    'run_workflow',
    'create_cfd_workflow',
    'create_guidance_workflow',
    # Config
    'ConfigSource',
    'ConfigValue',
    'ConfigSection',
    'Configuration',
    'ConfigManager',
    'EnvironmentConfig',
    'get_config',
    'load_config',
    'save_config',
    'merge_configs',
    'ConfigValidator',
    'validate_config',
    # Monitoring
    'LogLevel',
    'LogEntry',
    'LogFormatter',
    'StructuredLogger',
    'MetricType',
    'Metric',
    'MetricCollector',
    'MetricsRegistry',
    'TelemetryEvent',
    'TelemetryCollector',
    'AlertSeverity',
    'Alert',
    'AlertManager',
    'get_logger',
    'log_info',
    'log_warning',
    'log_error',
    'record_metric',
    # Diagnostics
    'SystemInfo',
    'GPUInfo',
    'MemoryInfo',
    'DiagnosticsReport',
    'HealthStatus',
    'HealthCheck',
    'HealthCheckResult',
    'SystemHealthMonitor',
    'DebugContext',
    'Profiler',
    'TracingSpan',
    'get_system_info',
    'run_diagnostics',
    'check_system_health',
]
