"""
Plugin System
==============

Extensibility through plugins.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# PLUGIN INFO
# =============================================================================


class PluginState(Enum):
    """Plugin lifecycle state."""

    UNLOADED = "unloaded"
    LOADED = "loaded"
    ENABLED = "enabled"
    DISABLED = "disabled"
    ERROR = "error"


@dataclass
class PluginInfo:
    """
    Metadata about a plugin.
    """

    id: str
    name: str
    version: str = "0.0.0"
    description: str = ""
    author: str = ""
    dependencies: list[str] = field(default_factory=list)
    state: PluginState = PluginState.UNLOADED
    error: str | None = None
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "dependencies": self.dependencies,
            "state": self.state.value,
            "error": self.error,
            "path": self.path,
        }


# =============================================================================
# PLUGIN BASE
# =============================================================================


class Plugin(ABC):
    """
    Abstract base for FieldOS plugins.

    Example:
        class MyPlugin(Plugin):
            @property
            def info(self) -> PluginInfo:
                return PluginInfo(
                    id="my-plugin",
                    name="My Plugin",
                    version="1.0.0",
                )

            def on_enable(self, kernel):
                kernel.register_pipeline("my_pipe", ...)

            def on_disable(self, kernel):
                pass
    """

    @property
    @abstractmethod
    def info(self) -> PluginInfo:
        """Plugin metadata."""
        pass

    def on_load(self, kernel: Any):
        """Called when plugin is loaded."""
        pass

    def on_enable(self, kernel: Any):
        """Called when plugin is enabled."""
        pass

    def on_disable(self, kernel: Any):
        """Called when plugin is disabled."""
        pass

    def on_unload(self, kernel: Any):
        """Called when plugin is unloaded."""
        pass


# =============================================================================
# PLUGIN HOOKS
# =============================================================================


class PluginHook:
    """
    Hook point for plugin extensions.
    """

    def __init__(self, name: str):
        self.name = name
        self._handlers: list[Callable] = []

    def register(self, handler: Callable):
        """Register a handler."""
        self._handlers.append(handler)

    def unregister(self, handler: Callable):
        """Unregister a handler."""
        self._handlers = [h for h in self._handlers if h != handler]

    def call(self, *args, **kwargs) -> list[Any]:
        """Call all handlers and collect results."""
        results = []
        for handler in self._handlers:
            try:
                result = handler(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Hook handler error: {e}")
        return results

    def call_until(
        self, predicate: Callable[[Any], bool], *args, **kwargs
    ) -> Any | None:
        """Call handlers until predicate returns True."""
        for handler in self._handlers:
            try:
                result = handler(*args, **kwargs)
                if predicate(result):
                    return result
            except Exception as e:
                logger.error(f"Hook handler error: {e}")
        return None


# =============================================================================
# PLUGIN MANAGER
# =============================================================================


class PluginManager:
    """
    Manages plugin lifecycle.

    Example:
        manager = PluginManager(kernel)
        manager.discover("./plugins")
        manager.enable("my-plugin")
    """

    def __init__(self, kernel: Any = None):
        self._kernel = kernel
        self._plugins: dict[str, Plugin] = {}
        self._plugin_info: dict[str, PluginInfo] = {}
        self._hooks: dict[str, PluginHook] = {}

    # -------------------------------------------------------------------------
    # Discovery
    # -------------------------------------------------------------------------

    def discover(self, path: str) -> list[PluginInfo]:
        """
        Discover plugins in directory.

        Args:
            path: Directory to search

        Returns:
            List of discovered plugin info
        """
        discovered = []
        path = Path(path)

        if not path.exists():
            logger.warning(f"Plugin directory not found: {path}")
            return discovered

        for item in path.iterdir():
            if item.is_dir() and (item / "__init__.py").exists():
                # Package plugin
                info = self._discover_package(item)
                if info:
                    discovered.append(info)
            elif item.suffix == ".py" and item.stem != "__init__":
                # Single-file plugin
                info = self._discover_file(item)
                if info:
                    discovered.append(info)

        return discovered

    def _discover_package(self, path: Path) -> PluginInfo | None:
        """Discover plugin from package directory."""
        try:
            spec = importlib.util.spec_from_file_location(
                path.stem,
                path / "__init__.py",
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Look for Plugin subclass
                plugin_class = self._find_plugin_class(module)
                if plugin_class:
                    plugin = plugin_class()
                    info = plugin.info
                    info.path = str(path)
                    self._plugins[info.id] = plugin
                    self._plugin_info[info.id] = info
                    return info
        except Exception as e:
            logger.error(f"Error discovering plugin at {path}: {e}")
        return None

    def _discover_file(self, path: Path) -> PluginInfo | None:
        """Discover plugin from single file."""
        try:
            spec = importlib.util.spec_from_file_location(path.stem, path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                plugin_class = self._find_plugin_class(module)
                if plugin_class:
                    plugin = plugin_class()
                    info = plugin.info
                    info.path = str(path)
                    self._plugins[info.id] = plugin
                    self._plugin_info[info.id] = info
                    return info
        except Exception as e:
            logger.error(f"Error discovering plugin at {path}: {e}")
        return None

    def _find_plugin_class(self, module) -> type[Plugin] | None:
        """Find Plugin subclass in module."""
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and issubclass(obj, Plugin) and obj is not Plugin:
                return obj
        return None

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    def register(self, plugin: Plugin) -> PluginInfo:
        """
        Register a plugin instance.

        Args:
            plugin: Plugin to register

        Returns:
            Plugin info
        """
        info = plugin.info
        self._plugins[info.id] = plugin
        self._plugin_info[info.id] = info
        info.state = PluginState.LOADED
        return info

    def unregister(self, plugin_id: str):
        """Unregister a plugin."""
        if plugin_id in self._plugins:
            plugin = self._plugins[plugin_id]
            if self._plugin_info[plugin_id].state == PluginState.ENABLED:
                self.disable(plugin_id)
            plugin.on_unload(self._kernel)
            del self._plugins[plugin_id]
            del self._plugin_info[plugin_id]

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def enable(self, plugin_id: str) -> bool:
        """
        Enable a plugin.

        Args:
            plugin_id: Plugin ID

        Returns:
            True if enabled successfully
        """
        if plugin_id not in self._plugins:
            logger.error(f"Plugin not found: {plugin_id}")
            return False

        plugin = self._plugins[plugin_id]
        info = self._plugin_info[plugin_id]

        # Check dependencies
        for dep in info.dependencies:
            if dep not in self._plugin_info:
                info.error = f"Missing dependency: {dep}"
                info.state = PluginState.ERROR
                return False
            if self._plugin_info[dep].state != PluginState.ENABLED:
                # Enable dependency first
                if not self.enable(dep):
                    info.error = f"Failed to enable dependency: {dep}"
                    info.state = PluginState.ERROR
                    return False

        try:
            plugin.on_load(self._kernel)
            plugin.on_enable(self._kernel)
            info.state = PluginState.ENABLED
            logger.info(f"Plugin enabled: {plugin_id}")
            return True
        except Exception as e:
            info.error = str(e)
            info.state = PluginState.ERROR
            logger.error(f"Error enabling plugin {plugin_id}: {e}")
            return False

    def disable(self, plugin_id: str) -> bool:
        """
        Disable a plugin.

        Args:
            plugin_id: Plugin ID

        Returns:
            True if disabled successfully
        """
        if plugin_id not in self._plugins:
            return False

        plugin = self._plugins[plugin_id]
        info = self._plugin_info[plugin_id]

        # Check dependents
        for other_id, other_info in self._plugin_info.items():
            if (
                plugin_id in other_info.dependencies
                and other_info.state == PluginState.ENABLED
            ):
                # Disable dependent first
                self.disable(other_id)

        try:
            plugin.on_disable(self._kernel)
            info.state = PluginState.DISABLED
            logger.info(f"Plugin disabled: {plugin_id}")
            return True
        except Exception as e:
            info.error = str(e)
            info.state = PluginState.ERROR
            logger.error(f"Error disabling plugin {plugin_id}: {e}")
            return False

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def get(self, plugin_id: str) -> Plugin | None:
        """Get plugin by ID."""
        return self._plugins.get(plugin_id)

    def get_info(self, plugin_id: str) -> PluginInfo | None:
        """Get plugin info by ID."""
        return self._plugin_info.get(plugin_id)

    def list_plugins(self) -> list[PluginInfo]:
        """List all registered plugins."""
        return list(self._plugin_info.values())

    def list_enabled(self) -> list[str]:
        """List enabled plugin IDs."""
        return [
            pid
            for pid, info in self._plugin_info.items()
            if info.state == PluginState.ENABLED
        ]

    # -------------------------------------------------------------------------
    # Hooks
    # -------------------------------------------------------------------------

    def create_hook(self, name: str) -> PluginHook:
        """Create a hook point."""
        if name not in self._hooks:
            self._hooks[name] = PluginHook(name)
        return self._hooks[name]

    def get_hook(self, name: str) -> PluginHook | None:
        """Get existing hook."""
        return self._hooks.get(name)


# =============================================================================
# DECORATOR HELPERS
# =============================================================================


def plugin_info(
    id: str,
    name: str,
    version: str = "0.0.0",
    **kwargs,
) -> Callable[[type[Plugin]], type[Plugin]]:
    """
    Decorator to define plugin info.

    Example:
        @plugin_info("my-plugin", "My Plugin", version="1.0.0")
        class MyPlugin(Plugin):
            pass
    """

    def decorator(cls: type[Plugin]) -> type[Plugin]:
        original_info = getattr(cls, "info", None)

        @property
        def info(self) -> PluginInfo:
            return PluginInfo(id=id, name=name, version=version, **kwargs)

        cls.info = info
        return cls

    return decorator
