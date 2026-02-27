"""HyperTensor — Licensed execution fabric for compression-native compute.

This package is the product.  It provides three surfaces (API, SDK, MCP)
over a single closed execution core.  No source code, compiler
implementations, or operator kernels are ever distributed to clients.

Surfaces
--------
- ``hypertensor.api``  — FastAPI REST/WebSocket server
- ``hypertensor.sdk``  — Python client library
- ``hypertensor.mcp``  — MCP tool server for agent-native workflows
- ``hypertensor.cli``  — Command-line interface

Core (never distributed)
------------------------
- ``hypertensor.core``      — Execution engine, evidence, certificates
- ``hypertensor.contracts``  — Versioned Pydantic schemas
- ``hypertensor.jobs``       — Job state machine and store
"""

__version__ = "40.0.1"
API_VERSION = "2.0.0"
SCHEMA_VERSION = "1.0.0"
RUNTIME_VERSION = "1.0.0"
