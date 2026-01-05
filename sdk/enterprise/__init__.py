# Copyright 2025 Tigantic Labs. All Rights Reserved.
"""
HyperTensor Enterprise Module

Provides enterprise features including:
- License management
- Telemetry and monitoring
- Priority support integration
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class LicenseTier(Enum):
    """License tier levels."""
    COMMUNITY = "community"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


@dataclass
class LicenseStatus:
    """License status information."""
    tier: LicenseTier
    valid: bool
    expires: Optional[datetime]
    features: Dict[str, bool]
    organization: Optional[str]
    
    @property
    def is_expired(self) -> bool:
        if self.expires is None:
            return False
        return datetime.now() > self.expires


class LicenseManager:
    """
    Enterprise license manager.
    
    Handles license activation, validation, and feature gating.
    """
    
    _instance: Optional['LicenseManager'] = None
    _status: Optional[LicenseStatus] = None
    
    @classmethod
    def activate(cls, license_key: str) -> LicenseStatus:
        """
        Activate an enterprise license.
        
        Args:
            license_key: License key from Tigantic Labs
            
        Returns:
            License status after activation
        """
        # Validate key format
        if not cls._validate_key_format(license_key):
            raise ValueError("Invalid license key format")
        
        # Decode license info (simplified - real implementation would verify signature)
        tier, expires, features, org = cls._decode_license(license_key)
        
        cls._status = LicenseStatus(
            tier=tier,
            valid=True,
            expires=expires,
            features=features,
            organization=org,
        )
        
        # Store license
        cls._store_license(license_key)
        
        return cls._status
    
    @classmethod
    def status(cls) -> LicenseStatus:
        """Get current license status."""
        if cls._status is None:
            # Try to load stored license
            cls._load_stored_license()
        
        if cls._status is None:
            # Default to community tier
            cls._status = LicenseStatus(
                tier=LicenseTier.COMMUNITY,
                valid=True,
                expires=None,
                features={
                    "basic_operations": True,
                    "gpu_acceleration": True,
                    "distributed": False,
                    "priority_support": False,
                    "custom_operators": False,
                },
                organization=None,
            )
        
        return cls._status
    
    @classmethod
    def check_feature(cls, feature: str) -> bool:
        """Check if a feature is enabled."""
        status = cls.status()
        return status.features.get(feature, False)
    
    @classmethod
    def _validate_key_format(cls, key: str) -> bool:
        """Validate license key format."""
        # Format: TIER-XXXXXXXX-XXXXXXXX-XXXX
        parts = key.split("-")
        if len(parts) != 4:
            return False
        if parts[0] not in ["PRO", "ENT", "TRIAL"]:
            return False
        return True
    
    @classmethod
    def _decode_license(cls, key: str) -> tuple:
        """Decode license key (simplified)."""
        parts = key.split("-")
        tier_map = {
            "PRO": LicenseTier.PROFESSIONAL,
            "ENT": LicenseTier.ENTERPRISE,
            "TRIAL": LicenseTier.PROFESSIONAL,
        }
        tier = tier_map.get(parts[0], LicenseTier.COMMUNITY)
        
        # Features by tier
        features = {
            LicenseTier.PROFESSIONAL: {
                "basic_operations": True,
                "gpu_acceleration": True,
                "distributed": True,
                "priority_support": False,
                "custom_operators": True,
            },
            LicenseTier.ENTERPRISE: {
                "basic_operations": True,
                "gpu_acceleration": True,
                "distributed": True,
                "priority_support": True,
                "custom_operators": True,
            },
        }
        
        return tier, None, features.get(tier, {}), None
    
    @classmethod
    def _store_license(cls, key: str) -> None:
        """Store license key."""
        license_dir = Path.home() / ".hypertensor"
        license_dir.mkdir(exist_ok=True)
        
        license_file = license_dir / "license.key"
        license_file.write_text(key)
    
    @classmethod
    def _load_stored_license(cls) -> None:
        """Load stored license key."""
        license_file = Path.home() / ".hypertensor" / "license.key"
        
        if license_file.exists():
            try:
                key = license_file.read_text().strip()
                cls.activate(key)
            except Exception:
                pass


@dataclass
class TelemetryConfig:
    """Telemetry configuration."""
    enabled: bool = False
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    batch_size: int = 100
    flush_interval: float = 60.0


@dataclass
class MetricEvent:
    """Telemetry metric event."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)


class Telemetry:
    """
    Enterprise telemetry and monitoring.
    
    Provides opt-in metrics collection for performance monitoring
    and debugging.
    """
    
    _config: TelemetryConfig = TelemetryConfig()
    _buffer: list = []
    _last_flush: float = 0.0
    
    @classmethod
    def enable(
        cls,
        endpoint: str,
        api_key: str,
        batch_size: int = 100,
        flush_interval: float = 60.0,
    ) -> None:
        """
        Enable telemetry collection.
        
        Args:
            endpoint: Metrics endpoint URL
            api_key: API key for authentication
            batch_size: Number of events to buffer before flush
            flush_interval: Maximum time between flushes (seconds)
        """
        cls._config = TelemetryConfig(
            enabled=True,
            endpoint=endpoint,
            api_key=api_key,
            batch_size=batch_size,
            flush_interval=flush_interval,
        )
        cls._last_flush = time.time()
    
    @classmethod
    def disable(cls) -> None:
        """Disable telemetry collection."""
        cls._config.enabled = False
        cls._buffer.clear()
    
    @classmethod
    def record(
        cls,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record a metric event.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional metric tags
        """
        if not cls._config.enabled:
            return
        
        event = MetricEvent(
            name=name,
            value=value,
            tags=tags or {},
        )
        cls._buffer.append(event)
        
        # Check if flush needed
        if len(cls._buffer) >= cls._config.batch_size:
            cls.flush()
        elif time.time() - cls._last_flush > cls._config.flush_interval:
            cls.flush()
    
    @classmethod
    def flush(cls) -> None:
        """Flush buffered events to endpoint."""
        if not cls._config.enabled or not cls._buffer:
            return
        
        # Prepare payload
        events = [
            {
                "name": e.name,
                "value": e.value,
                "timestamp": e.timestamp,
                "tags": e.tags,
            }
            for e in cls._buffer
        ]
        
        try:
            import urllib.request
            
            data = json.dumps({"events": events}).encode("utf-8")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cls._config.api_key}",
            }
            
            req = urllib.request.Request(
                cls._config.endpoint,
                data=data,
                headers=headers,
                method="POST",
            )
            
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    cls._buffer.clear()
                    
        except Exception as e:
            # Log error but don't fail
            print(f"Telemetry flush failed: {e}")
        
        cls._last_flush = time.time()
    
    @classmethod
    def is_enabled(cls) -> bool:
        """Check if telemetry is enabled."""
        return cls._config.enabled


# Convenience exports
__all__ = [
    "LicenseManager",
    "LicenseStatus",
    "LicenseTier",
    "Telemetry",
    "TelemetryConfig",
]
