"""
Field OS - Complete Orchestration Layer
=========================================

Layer 7 of the HyperTensor platform.

Brings together all layers into a unified operating system for reality:
- Unified Field representation and lifecycle management
- Composable pipelines for data flow
- Plugin system for extensibility
- Session management and persistence
- Observable state with event streams
- Complete REPL for interactive field manipulation

Components:
    FieldOS         - Main orchestrator / kernel
    Field           - Unified field representation
    Pipeline        - Composable processing stages
    Plugin          - Extension interface
    Session         - State persistence and history
    Observable      - Reactive state streams

Example:
    from tensornet.fieldos import FieldOS

    os = FieldOS()

    # Load and evolve fields
    field = os.create_field("velocity", shape=(128, 128, 3))
    os.evolve(field, timesteps=100)

    # Query with natural language
    result = os.query("show maximum vorticity")

    # Save session
    os.save_session("sim_001")
"""

from __future__ import annotations

from .field import Field, FieldMetadata, FieldType

# Core
from .kernel import FieldOS, FieldOSConfig
from .observable import Event, EventType, Observable, Observer
from .pipeline import Pipeline, Stage, StageResult
from .plugin import Plugin, PluginInfo, PluginManager
from .session import Checkpoint, Session, SessionState

__all__ = [
    # Core
    "FieldOS",
    "FieldOSConfig",
    # Field
    "Field",
    "FieldType",
    "FieldMetadata",
    # Pipeline
    "Pipeline",
    "Stage",
    "StageResult",
    # Plugin
    "Plugin",
    "PluginManager",
    "PluginInfo",
    # Session
    "Session",
    "SessionState",
    "Checkpoint",
    # Observable
    "Observable",
    "Observer",
    "Event",
    "EventType",
]

__version__ = "0.1.0"
