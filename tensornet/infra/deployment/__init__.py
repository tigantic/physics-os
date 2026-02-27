"""
Deployment Module
=================

Tools for deploying tensor network CFD models to embedded hardware.

Key Components:
    - TensorRT export for NVIDIA Jetson
    - ONNX model conversion
    - Memory optimization for embedded systems
    - Hardware abstraction layer
"""

from tensornet.infra.deployment.embedded import (
                                           EmbeddedRuntime,
                                           JetsonConfig,
                                           MemoryProfile,
                                           PowerMode,
                                           configure_jetson_power,
                                           create_inference_pipeline,
                                           optimize_memory_layout,
)
from tensornet.infra.deployment.tensorrt_export import (
                                           ExportConfig,
                                           ExportResult,
                                           TensorRTExporter,
                                           benchmark_inference,
                                           export_to_onnx,
                                           optimize_for_tensorrt,
                                           validate_exported_model,
)

__all__ = [
    # TensorRT Export
    "ExportConfig",
    "ExportResult",
    "TensorRTExporter",
    "export_to_onnx",
    "optimize_for_tensorrt",
    "validate_exported_model",
    "benchmark_inference",
    # Embedded Deployment
    "JetsonConfig",
    "MemoryProfile",
    "PowerMode",
    "EmbeddedRuntime",
    "optimize_memory_layout",
    "configure_jetson_power",
    "create_inference_pipeline",
]
