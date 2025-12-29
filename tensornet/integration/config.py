"""
Configuration Management for Project HyperTensor.

Provides hierarchical configuration with:
- Multiple sources (files, environment, defaults)
- Type validation
- Environment-specific overrides
- Secure credential handling
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Type, Callable
from enum import Enum, auto
from pathlib import Path
import json
import os
import copy


class ConfigSource(Enum):
    """Configuration value source."""
    DEFAULT = auto()
    FILE = auto()
    ENVIRONMENT = auto()
    OVERRIDE = auto()


@dataclass
class ConfigValue:
    """
    Single configuration value with metadata.
    
    Attributes:
        key: Configuration key
        value: Current value
        default: Default value
        source: Where value came from
        dtype: Expected data type
        description: Human-readable description
        required: Whether value is required
        sensitive: Whether value is sensitive (passwords, keys)
    """
    key: str
    value: Any
    default: Any = None
    source: ConfigSource = ConfigSource.DEFAULT
    dtype: Optional[Type] = None
    description: str = ""
    required: bool = False
    sensitive: bool = False
    
    def validate(self) -> bool:
        """Validate the value against its type."""
        if self.value is None:
            return not self.required
        
        if self.dtype is not None:
            return isinstance(self.value, self.dtype)
        
        return True
    
    def get(self) -> Any:
        """Get the value, returning default if None."""
        return self.value if self.value is not None else self.default
    
    def __repr__(self) -> str:
        if self.sensitive:
            return f"ConfigValue(key={self.key}, value=***SENSITIVE***)"
        return f"ConfigValue(key={self.key}, value={self.value})"


@dataclass
class ConfigSection:
    """
    Section of related configuration values.
    
    Attributes:
        name: Section name
        values: Configuration values in this section
        description: Section description
    """
    name: str
    values: Dict[str, ConfigValue] = field(default_factory=dict)
    description: str = ""
    
    def set(self, key: str, value: Any, source: ConfigSource = ConfigSource.OVERRIDE):
        """Set a configuration value."""
        if key in self.values:
            self.values[key].value = value
            self.values[key].source = source
        else:
            self.values[key] = ConfigValue(
                key=key,
                value=value,
                source=source,
            )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        if key in self.values:
            return self.values[key].get()
        return default
    
    def define(
        self,
        key: str,
        default: Any = None,
        dtype: Optional[Type] = None,
        description: str = "",
        required: bool = False,
        sensitive: bool = False,
    ):
        """Define a configuration value with metadata."""
        self.values[key] = ConfigValue(
            key=key,
            value=default,
            default=default,
            dtype=dtype,
            description=description,
            required=required,
            sensitive=sensitive,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert section to dictionary."""
        return {k: v.get() for k, v in self.values.items()}
    
    def validate(self) -> List[str]:
        """Validate all values, returning list of errors."""
        errors = []
        for key, value in self.values.items():
            if not value.validate():
                errors.append(f"{self.name}.{key}: Invalid value type")
            if value.required and value.value is None:
                errors.append(f"{self.name}.{key}: Required value missing")
        return errors


@dataclass
class Configuration:
    """
    Complete configuration with multiple sections.
    
    Attributes:
        name: Configuration name
        version: Configuration version
        sections: Configuration sections
    """
    name: str = "hypertensor"
    version: str = "1.0"
    sections: Dict[str, ConfigSection] = field(default_factory=dict)
    
    def add_section(self, section: ConfigSection):
        """Add a configuration section."""
        self.sections[section.name] = section
    
    def get_section(self, name: str) -> Optional[ConfigSection]:
        """Get a configuration section."""
        return self.sections.get(name)
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get a value by dot-separated path.
        
        Args:
            path: Path like "section.key"
            default: Default if not found
        """
        parts = path.split('.', 1)
        if len(parts) == 1:
            return default
        
        section = self.sections.get(parts[0])
        if section is None:
            return default
        
        return section.get(parts[1], default)
    
    def set(self, path: str, value: Any, source: ConfigSource = ConfigSource.OVERRIDE):
        """
        Set a value by dot-separated path.
        
        Args:
            path: Path like "section.key"
            value: Value to set
            source: Source of the value
        """
        parts = path.split('.', 1)
        if len(parts) == 1:
            return
        
        section_name, key = parts
        if section_name not in self.sections:
            self.sections[section_name] = ConfigSection(name=section_name)
        
        self.sections[section_name].set(key, value, source)
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert to nested dictionary."""
        return {name: section.to_dict() for name, section in self.sections.items()}
    
    def validate(self) -> List[str]:
        """Validate all sections, returning list of errors."""
        errors = []
        for section in self.sections.values():
            errors.extend(section.validate())
        return errors
    
    def merge(self, other: 'Configuration', source: ConfigSource = ConfigSource.OVERRIDE):
        """Merge another configuration into this one."""
        for section_name, section in other.sections.items():
            if section_name not in self.sections:
                self.sections[section_name] = copy.deepcopy(section)
            else:
                for key, value in section.values.items():
                    self.sections[section_name].set(key, value.value, source)


class EnvironmentConfig:
    """
    Configuration from environment variables.
    
    Follows convention: HYPERTENSOR_SECTION_KEY = value
    """
    
    PREFIX = "HYPERTENSOR"
    
    @classmethod
    def load(cls, prefix: Optional[str] = None) -> Configuration:
        """
        Load configuration from environment.
        
        Args:
            prefix: Environment variable prefix
            
        Returns:
            Configuration from environment
        """
        prefix = prefix or cls.PREFIX
        config = Configuration(name=f"{prefix}_env")
        
        for key, value in os.environ.items():
            if key.startswith(f"{prefix}_"):
                # Parse key: PREFIX_SECTION_KEY
                parts = key[len(prefix) + 1:].split('_', 1)
                if len(parts) == 2:
                    section_name = parts[0].lower()
                    config_key = parts[1].lower()
                    
                    config.set(
                        f"{section_name}.{config_key}",
                        cls._parse_value(value),
                        ConfigSource.ENVIRONMENT,
                    )
        
        return config
    
    @classmethod
    def _parse_value(cls, value: str) -> Any:
        """Parse string value to appropriate type."""
        # Try boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Try JSON
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # Return as string
        return value


class ConfigManager:
    """
    Central configuration manager.
    
    Handles loading, merging, and accessing configuration from
    multiple sources with proper precedence.
    """
    
    _instance: Optional['ConfigManager'] = None
    
    def __init__(self):
        """Initialize config manager."""
        self.config = Configuration()
        self._watchers: List[Callable[[str, Any], None]] = []
        self._loaded_files: List[Path] = []
        
        # Load defaults
        self._load_defaults()
    
    @classmethod
    def get_instance(cls) -> 'ConfigManager':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _load_defaults(self):
        """Load default configuration values."""
        # CFD defaults
        cfd = ConfigSection(name="cfd", description="CFD solver configuration")
        cfd.define("cfl", 0.5, float, "CFL number")
        cfd.define("gamma", 1.4, float, "Ratio of specific heats")
        cfd.define("flux_scheme", "roe", str, "Numerical flux scheme")
        cfd.define("limiter", "minmod", str, "Slope limiter")
        cfd.define("max_iterations", 10000, int, "Maximum solver iterations")
        self.config.add_section(cfd)
        
        # Solver defaults
        solver = ConfigSection(name="solver", description="Solver configuration")
        solver.define("tolerance", 1e-10, float, "Convergence tolerance")
        solver.define("max_iter", 1000, int, "Maximum iterations")
        solver.define("verbose", False, bool, "Verbose output")
        self.config.add_section(solver)
        
        # GPU defaults
        gpu = ConfigSection(name="gpu", description="GPU configuration")
        gpu.define("enabled", True, bool, "Enable GPU acceleration")
        gpu.define("device", 0, int, "GPU device index")
        gpu.define("memory_fraction", 0.9, float, "GPU memory fraction")
        self.config.add_section(gpu)
        
        # Logging defaults
        logging = ConfigSection(name="logging", description="Logging configuration")
        logging.define("level", "INFO", str, "Log level")
        logging.define("format", "structured", str, "Log format")
        logging.define("file", None, str, "Log file path")
        self.config.add_section(logging)
        
        # Deployment defaults
        deploy = ConfigSection(name="deployment", description="Deployment configuration")
        deploy.define("mode", "development", str, "Deployment mode")
        deploy.define("target_platform", "desktop", str, "Target platform")
        deploy.define("precision", "float32", str, "Numerical precision")
        self.config.add_section(deploy)
    
    def load_file(
        self,
        path: Union[str, Path],
        format: str = "auto",
    ) -> bool:
        """
        Load configuration from file.
        
        Args:
            path: Path to config file
            format: File format (json, yaml, auto)
            
        Returns:
            Whether load was successful
        """
        path = Path(path)
        if not path.exists():
            return False
        
        if format == "auto":
            format = path.suffix.lstrip('.')
        
        try:
            with open(path) as f:
                if format == "json":
                    data = json.load(f)
                else:
                    # Default to JSON
                    data = json.load(f)
            
            self._apply_dict(data, ConfigSource.FILE)
            self._loaded_files.append(path)
            return True
            
        except Exception:
            return False
    
    def _apply_dict(self, data: Dict, source: ConfigSource):
        """Apply dictionary to configuration."""
        for section_name, section_data in data.items():
            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    self.config.set(f"{section_name}.{key}", value, source)
    
    def load_environment(self, prefix: str = "HYPERTENSOR"):
        """Load configuration from environment variables."""
        env_config = EnvironmentConfig.load(prefix)
        self.config.merge(env_config, ConfigSource.ENVIRONMENT)
    
    def get(self, path: str, default: Any = None) -> Any:
        """Get configuration value by path."""
        return self.config.get(path, default)
    
    def set(self, path: str, value: Any):
        """Set configuration value."""
        old_value = self.config.get(path)
        self.config.set(path, value, ConfigSource.OVERRIDE)
        
        # Notify watchers
        for watcher in self._watchers:
            try:
                watcher(path, value)
            except Exception:
                pass
    
    def watch(self, callback: Callable[[str, Any], None]):
        """Register a configuration change watcher."""
        self._watchers.append(callback)
    
    def to_dict(self) -> Dict:
        """Export configuration to dictionary."""
        return self.config.to_dict()
    
    def save(self, path: Union[str, Path], format: str = "json"):
        """
        Save configuration to file.
        
        Args:
            path: Output path
            format: File format
        """
        path = Path(path)
        data = self.config.to_dict()
        
        with open(path, 'w') as f:
            if format == "json":
                json.dump(data, f, indent=2)


class ConfigValidator:
    """
    Configuration validation with custom rules.
    """
    
    def __init__(self):
        """Initialize validator."""
        self.rules: List[Callable[[Configuration], List[str]]] = []
    
    def add_rule(self, rule: Callable[[Configuration], List[str]]):
        """Add a validation rule."""
        self.rules.append(rule)
    
    def validate(self, config: Configuration) -> List[str]:
        """
        Validate configuration.
        
        Returns:
            List of error messages
        """
        errors = config.validate()
        
        for rule in self.rules:
            try:
                errors.extend(rule(config))
            except Exception as e:
                errors.append(f"Validation rule error: {e}")
        
        return errors


# =============================================================================
# Convenience Functions
# =============================================================================


def get_config() -> ConfigManager:
    """Get the global configuration manager."""
    return ConfigManager.get_instance()


def load_config(path: Union[str, Path]) -> bool:
    """
    Load configuration from file.
    
    Args:
        path: Config file path
        
    Returns:
        Whether load was successful
    """
    return get_config().load_file(path)


def save_config(path: Union[str, Path]):
    """
    Save configuration to file.
    
    Args:
        path: Output path
    """
    get_config().save(path)


def merge_configs(
    base: Configuration,
    override: Configuration,
) -> Configuration:
    """
    Merge two configurations.
    
    Args:
        base: Base configuration
        override: Override configuration
        
    Returns:
        Merged configuration
    """
    result = copy.deepcopy(base)
    result.merge(override)
    return result


def validate_config(config: Optional[Configuration] = None) -> List[str]:
    """
    Validate configuration.
    
    Args:
        config: Configuration to validate (uses global if None)
        
    Returns:
        List of validation errors
    """
    if config is None:
        config = get_config().config
    
    validator = ConfigValidator()
    
    # Add standard rules
    def check_cfl(cfg: Configuration) -> List[str]:
        cfl = cfg.get("cfd.cfl", 0.5)
        if cfl <= 0 or cfl > 1:
            return ["cfd.cfl must be between 0 and 1"]
        return []
    
    def check_precision(cfg: Configuration) -> List[str]:
        valid = {"float16", "float32", "float64"}
        precision = cfg.get("deployment.precision", "float32")
        if precision not in valid:
            return [f"deployment.precision must be one of {valid}"]
        return []
    
    validator.add_rule(check_cfl)
    validator.add_rule(check_precision)
    
    return validator.validate(config)
