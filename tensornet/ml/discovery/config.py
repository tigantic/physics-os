"""
Discovery Engine Configuration

Centralized configuration for all magic numbers and tunable parameters.
All values can be overridden via environment variables or constructor arguments.

Usage:
    from tensornet.ml.discovery.config import DiscoveryConfig, get_config
    
    # Get global config (singleton)
    config = get_config()
    
    # Access values
    bins = config.histogram_bins
    
    # Override via environment
    export DISCOVERY_HISTOGRAM_BINS=2048
"""

from dataclasses import dataclass, field
from typing import Optional
import os


def _env_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    value = os.environ.get(f"DISCOVERY_{key}")
    return int(value) if value else default


def _env_float(key: str, default: float) -> float:
    """Get float from environment variable."""
    value = os.environ.get(f"DISCOVERY_{key}")
    return float(value) if value else default


@dataclass
class IngestionConfig:
    """Configuration for data ingestion."""
    
    # Histogram/binning
    histogram_bins: int = field(default_factory=lambda: _env_int("HISTOGRAM_BINS", 1024))
    max_histogram_bins: int = field(default_factory=lambda: _env_int("MAX_HISTOGRAM_BINS", 2048))
    
    # Grid sizes
    plasma_grid_size: int = field(default_factory=lambda: _env_int("PLASMA_GRID_SIZE", 256))
    molecular_grid_size: int = field(default_factory=lambda: _env_int("MOLECULAR_GRID_SIZE", 128))
    
    # Persistence algorithm limits
    max_persistence_points: int = field(default_factory=lambda: _env_int("MAX_PERSISTENCE_POINTS", 256))


@dataclass
class ConnectorConfig:
    """Configuration for data connectors."""
    
    # Queue sizes
    queue_size: int = field(default_factory=lambda: _env_int("QUEUE_SIZE", 10000))
    result_queue_size: int = field(default_factory=lambda: _env_int("RESULT_QUEUE_SIZE", 1000))
    
    # Simulated connector defaults
    default_btc_price: float = field(default_factory=lambda: _env_float("DEFAULT_BTC_PRICE", 50000.0))
    default_volatility: float = field(default_factory=lambda: _env_float("DEFAULT_VOLATILITY", 0.0002))
    default_update_rate: float = field(default_factory=lambda: _env_float("DEFAULT_UPDATE_RATE", 10.0))
    
    # Reconnection
    max_reconnect_attempts: int = field(default_factory=lambda: _env_int("MAX_RECONNECT_ATTEMPTS", 5))
    reconnect_delay_seconds: float = field(default_factory=lambda: _env_float("RECONNECT_DELAY_SECONDS", 1.0))


@dataclass
class MarketConfig:
    """Configuration for market analysis."""
    
    # Financial calendar
    trading_days_per_year: int = field(default_factory=lambda: _env_int("TRADING_DAYS_PER_YEAR", 252))
    
    # Lookback windows
    default_lookback: int = field(default_factory=lambda: _env_int("DEFAULT_LOOKBACK", 100))
    volatility_window: int = field(default_factory=lambda: _env_int("VOLATILITY_WINDOW", 60))
    
    # Alert thresholds
    drawdown_alert_threshold: float = field(default_factory=lambda: _env_float("DRAWDOWN_ALERT", 0.05))
    volatility_spike_threshold: float = field(default_factory=lambda: _env_float("VOL_SPIKE", 2.0))


@dataclass
class SecurityConfig:
    """Configuration for security features."""
    
    # Request signing
    max_request_age_seconds: int = field(default_factory=lambda: _env_int("MAX_REQUEST_AGE_SECONDS", 300))
    
    # Audit logging
    max_audit_events: int = field(default_factory=lambda: _env_int("MAX_AUDIT_EVENTS", 10000))
    
    # API key expiration
    default_key_expiry_days: int = field(default_factory=lambda: _env_int("DEFAULT_KEY_EXPIRY_DAYS", 365))


@dataclass
class CryptoConfig:
    """Configuration for cryptographic operations."""
    
    # BN254 prime (standard, should not change)
    bn254_prime: int = 21888242871839275222246405745257275088696311157297823662689037894645226208583
    
    # BLS12-381 prime (standard, should not change)
    bls12_381_prime: int = 52435875175126190479447740508185965837690552500527637822603658699938581184513
    
    # Poseidon rounds
    poseidon_full_rounds: int = field(default_factory=lambda: _env_int("POSEIDON_FULL_ROUNDS", 8))
    poseidon_partial_rounds: int = field(default_factory=lambda: _env_int("POSEIDON_PARTIAL_ROUNDS", 57))


@dataclass
class DiscoveryConfig:
    """
    Master configuration for Discovery Engine.
    
    All magic numbers and tunable parameters are centralized here.
    Values can be overridden via:
    1. Constructor arguments
    2. Environment variables (DISCOVERY_*)
    3. Programmatic modification
    """
    
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    connector: ConnectorConfig = field(default_factory=ConnectorConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    crypto: CryptoConfig = field(default_factory=CryptoConfig)
    
    # Global flags
    debug: bool = field(default_factory=lambda: os.environ.get("DISCOVERY_DEBUG", "").lower() == "true")
    verbose_logging: bool = field(default_factory=lambda: os.environ.get("DISCOVERY_VERBOSE", "").lower() == "true")
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate configuration values."""
        if self.ingestion.histogram_bins < 16:
            raise ValueError("histogram_bins must be >= 16")
        if self.ingestion.max_persistence_points < 32:
            raise ValueError("max_persistence_points must be >= 32")
        if self.connector.queue_size < 100:
            raise ValueError("queue_size must be >= 100")
        if self.market.trading_days_per_year < 200:
            raise ValueError("trading_days_per_year must be >= 200 (standard is 252)")


# Global singleton
_global_config: Optional[DiscoveryConfig] = None


def get_config() -> DiscoveryConfig:
    """
    Get global configuration singleton.
    
    Creates default configuration on first call.
    Thread-safe for reads after initialization.
    
    Returns:
        Global DiscoveryConfig instance
    """
    global _global_config
    if _global_config is None:
        _global_config = DiscoveryConfig()
    return _global_config


def set_config(config: DiscoveryConfig) -> None:
    """
    Set global configuration.
    
    Use this at application startup to customize configuration.
    
    Args:
        config: New configuration to use globally
    """
    global _global_config
    config._validate()
    _global_config = config


def reset_config() -> None:
    """Reset global configuration to defaults."""
    global _global_config
    _global_config = None


# Convenience exports
__all__ = [
    "DiscoveryConfig",
    "IngestionConfig",
    "ConnectorConfig",
    "MarketConfig",
    "SecurityConfig",
    "CryptoConfig",
    "get_config",
    "set_config",
    "reset_config",
]
