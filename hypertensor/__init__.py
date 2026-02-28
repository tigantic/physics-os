"""The Physics OS — Platform Shell for The Ontic Engine.

This package is the platform surface layer.  It provides four client surfaces
(API, SDK, MCP, CLI) over a single closed execution core powered by The Ontic
Engine.  No source code, compiler implementations, or operator kernels are
ever distributed to clients.

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

Brand Hierarchy
---------------
Tigantic Holdings LLC · DBA HolonomiX · The Physics OS · The Ontic Engine
"""

__version__ = "40.0.1"
API_VERSION = "2.0.0"
SCHEMA_VERSION = "1.0.0"
RUNTIME_VERSION = "1.0.0"
