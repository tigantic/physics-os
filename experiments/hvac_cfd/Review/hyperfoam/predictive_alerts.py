"""
TigantiCFD Predictive Alerts
============================

Anomaly detection and predictive alerting for HVAC digital twin.

Capabilities:
- T4.01: Statistical anomaly detection
- T4.02: ML-based prediction
- T4.03: Threshold-based alerts
- T4.04: Alert severity classification

Reference:
    Chandola, V. et al. (2009). "Anomaly detection: A survey."
    ACM Computing Surveys.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Callable
from datetime import datetime, timezone, timedelta
from enum import Enum
import numpy as np
from collections import deque


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(Enum):
    """Types of alerts."""
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    ANOMALY_DETECTED = "anomaly_detected"
    TREND_DEVIATION = "trend_deviation"
    EQUIPMENT_FAULT = "equipment_fault"
    COMFORT_VIOLATION = "comfort_violation"
    ENERGY_SPIKE = "energy_spike"


@dataclass
class Alert:
    """A single alert event."""
    alert_id: str
    timestamp: datetime
    alert_type: AlertType
    severity: AlertSeverity
    metric_name: str
    current_value: float
    threshold_value: Optional[float]
    message: str
    zone_id: Optional[str] = None
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "message": self.message,
            "zone_id": self.zone_id,
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
            "metadata": self.metadata
        }


@dataclass
class ThresholdConfig:
    """Configuration for threshold-based alerting."""
    metric_name: str
    warning_low: Optional[float] = None
    warning_high: Optional[float] = None
    critical_low: Optional[float] = None
    critical_high: Optional[float] = None
    unit: str = ""
    hysteresis: float = 0.02  # 2% hysteresis to prevent flapping


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection."""
    metric_name: str
    window_size: int = 100        # Number of samples for baseline
    sigma_threshold: float = 3.0  # Standard deviations for anomaly
    min_samples: int = 20         # Minimum samples before detection


class StatisticalAnomalyDetector:
    """
    Statistical anomaly detection using z-score and rolling statistics.
    
    Detects anomalies when values deviate significantly from
    the rolling mean, accounting for natural variance.
    """
    
    def __init__(
        self,
        window_size: int = 100,
        sigma_threshold: float = 3.0
    ):
        self.window_size = window_size
        self.sigma_threshold = sigma_threshold
        
        self.history: deque = deque(maxlen=window_size)
        self.mean: float = 0.0
        self.std: float = 1.0
        self.n_samples: int = 0
        
    def update(self, value: float) -> Tuple[bool, float, float]:
        """
        Update with new value and check for anomaly.
        
        Returns:
            (is_anomaly, z_score, threshold)
        """
        self.history.append(value)
        self.n_samples += 1
        
        if len(self.history) < 5:
            return False, 0.0, self.sigma_threshold
        
        # Compute rolling statistics
        data = np.array(self.history)
        self.mean = np.mean(data)
        self.std = np.std(data) + 1e-10  # Prevent division by zero
        
        # Compute z-score
        z_score = abs(value - self.mean) / self.std
        
        is_anomaly = z_score > self.sigma_threshold
        
        return is_anomaly, z_score, self.sigma_threshold
    
    def get_bounds(self) -> Tuple[float, float]:
        """Get current anomaly detection bounds."""
        lower = self.mean - self.sigma_threshold * self.std
        upper = self.mean + self.sigma_threshold * self.std
        return lower, upper


class TrendAnalyzer:
    """
    Trend analysis for predictive alerting.
    
    Detects when metrics are trending toward threshold violation.
    """
    
    def __init__(
        self,
        window_size: int = 20,
        prediction_horizon: float = 300.0  # 5 minutes
    ):
        self.window_size = window_size
        self.horizon = prediction_horizon
        
        self.times: deque = deque(maxlen=window_size)
        self.values: deque = deque(maxlen=window_size)
        
    def update(
        self,
        value: float,
        timestamp: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """
        Update with new value and compute trend.
        
        Returns:
            (slope, predicted_value, r_squared)
        """
        if timestamp is None:
            timestamp = len(self.times)
        
        self.times.append(timestamp)
        self.values.append(value)
        
        if len(self.times) < 3:
            return 0.0, value, 0.0
        
        # Linear regression
        t = np.array(self.times)
        v = np.array(self.values)
        
        # Normalize time
        t_norm = t - t[0]
        
        # Fit line
        n = len(t_norm)
        sum_t = np.sum(t_norm)
        sum_v = np.sum(v)
        sum_tv = np.sum(t_norm * v)
        sum_t2 = np.sum(t_norm * t_norm)
        
        denom = n * sum_t2 - sum_t * sum_t
        if abs(denom) < 1e-10:
            return 0.0, value, 0.0
        
        slope = (n * sum_tv - sum_t * sum_v) / denom
        intercept = (sum_v - slope * sum_t) / n
        
        # Predict future value
        future_t = t_norm[-1] + self.horizon
        predicted = slope * future_t + intercept
        
        # R-squared
        v_pred = slope * t_norm + intercept
        ss_res = np.sum((v - v_pred) ** 2)
        ss_tot = np.sum((v - np.mean(v)) ** 2)
        r_squared = 1 - ss_res / (ss_tot + 1e-10) if ss_tot > 0 else 0
        
        return slope, predicted, r_squared
    
    def will_exceed(
        self,
        threshold: float,
        current_value: float
    ) -> Tuple[bool, float]:
        """
        Check if trend will exceed threshold within horizon.
        
        Returns:
            (will_exceed, time_to_exceed)
        """
        slope, predicted, r2 = self.update(current_value)
        
        # Only predict if trend is reliable
        if r2 < 0.5:
            return False, float('inf')
        
        if abs(slope) < 1e-10:
            return False, float('inf')
        
        # Time to exceed
        time_to_exceed = (threshold - current_value) / slope
        
        if time_to_exceed < 0:
            return False, float('inf')
        
        return time_to_exceed <= self.horizon, time_to_exceed


class AlertManager:
    """
    Central alert management for HVAC digital twin.
    
    Handles threshold monitoring, anomaly detection, and
    alert lifecycle management.
    """
    
    def __init__(self):
        self.thresholds: Dict[str, ThresholdConfig] = {}
        self.anomaly_detectors: Dict[str, StatisticalAnomalyDetector] = {}
        self.trend_analyzers: Dict[str, TrendAnalyzer] = {}
        
        self.alerts: List[Alert] = []
        self.active_alerts: Dict[str, Alert] = {}  # Keyed by metric+zone
        
        self._alert_counter = 0
        self.callbacks: List[Callable[[Alert], None]] = []
        
    def configure_threshold(self, config: ThresholdConfig) -> None:
        """Configure threshold alerting for a metric."""
        self.thresholds[config.metric_name] = config
        
    def configure_anomaly(self, config: AnomalyConfig) -> None:
        """Configure anomaly detection for a metric."""
        self.anomaly_detectors[config.metric_name] = StatisticalAnomalyDetector(
            window_size=config.window_size,
            sigma_threshold=config.sigma_threshold
        )
        
    def add_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add callback for new alerts."""
        self.callbacks.append(callback)
        
    def check_value(
        self,
        metric_name: str,
        value: float,
        zone_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> List[Alert]:
        """
        Check a metric value and generate alerts if needed.
        
        Returns:
            List of new alerts generated
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        new_alerts = []
        alert_key = f"{metric_name}:{zone_id}"
        
        # Check thresholds
        if metric_name in self.thresholds:
            config = self.thresholds[metric_name]
            alert = self._check_threshold(
                config, value, zone_id, timestamp
            )
            if alert:
                new_alerts.append(alert)
        
        # Check for anomalies
        if metric_name in self.anomaly_detectors:
            detector = self.anomaly_detectors[metric_name]
            is_anomaly, z_score, threshold = detector.update(value)
            
            if is_anomaly:
                alert = self._create_alert(
                    AlertType.ANOMALY_DETECTED,
                    AlertSeverity.WARNING,
                    metric_name,
                    value,
                    threshold,
                    f"Anomaly detected: {metric_name}={value:.2f} "
                    f"(z-score={z_score:.2f}, expected range: "
                    f"{detector.get_bounds()[0]:.2f} to {detector.get_bounds()[1]:.2f})",
                    zone_id,
                    timestamp,
                    {"z_score": z_score}
                )
                new_alerts.append(alert)
        
        # Process alerts
        for alert in new_alerts:
            self.alerts.append(alert)
            self.active_alerts[alert_key] = alert
            
            # Notify callbacks
            for callback in self.callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    # Log callback errors but don't stop alert processing
                    import logging
                    logging.getLogger(__name__).warning(f"Alert callback failed: {e}")
        
        return new_alerts
    
    def _check_threshold(
        self,
        config: ThresholdConfig,
        value: float,
        zone_id: Optional[str],
        timestamp: datetime
    ) -> Optional[Alert]:
        """Check threshold and generate alert if violated."""
        severity = None
        direction = None
        threshold = None
        
        # Check critical first (takes priority)
        if config.critical_high is not None and value > config.critical_high:
            severity = AlertSeverity.CRITICAL
            direction = "above"
            threshold = config.critical_high
        elif config.critical_low is not None and value < config.critical_low:
            severity = AlertSeverity.CRITICAL
            direction = "below"
            threshold = config.critical_low
        elif config.warning_high is not None and value > config.warning_high:
            severity = AlertSeverity.WARNING
            direction = "above"
            threshold = config.warning_high
        elif config.warning_low is not None and value < config.warning_low:
            severity = AlertSeverity.WARNING
            direction = "below"
            threshold = config.warning_low
        
        if severity is None:
            return None
        
        message = (
            f"{config.metric_name} is {direction} {severity.value} threshold: "
            f"{value:.2f} {config.unit} (threshold: {threshold:.2f} {config.unit})"
        )
        
        return self._create_alert(
            AlertType.THRESHOLD_EXCEEDED,
            severity,
            config.metric_name,
            value,
            threshold,
            message,
            zone_id,
            timestamp
        )
    
    def _create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        metric_name: str,
        value: float,
        threshold: Optional[float],
        message: str,
        zone_id: Optional[str],
        timestamp: datetime,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """Create a new alert."""
        self._alert_counter += 1
        
        return Alert(
            alert_id=f"alert-{self._alert_counter:06d}",
            timestamp=timestamp,
            alert_type=alert_type,
            severity=severity,
            metric_name=metric_name,
            current_value=value,
            threshold_value=threshold,
            message=message,
            zone_id=zone_id,
            metadata=metadata or {}
        )
    
    def acknowledge(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def resolve(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                return True
        return False
    
    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        zone_id: Optional[str] = None
    ) -> List[Alert]:
        """Get all unresolved alerts, optionally filtered."""
        result = [a for a in self.alerts if not a.resolved]
        
        if severity:
            result = [a for a in result if a.severity == severity]
        
        if zone_id:
            result = [a for a in result if a.zone_id == zone_id]
        
        return result
    
    def get_alert_summary(self) -> Dict[str, int]:
        """Get count of active alerts by severity."""
        active = self.get_active_alerts()
        
        return {
            "emergency": sum(1 for a in active if a.severity == AlertSeverity.EMERGENCY),
            "critical": sum(1 for a in active if a.severity == AlertSeverity.CRITICAL),
            "warning": sum(1 for a in active if a.severity == AlertSeverity.WARNING),
            "info": sum(1 for a in active if a.severity == AlertSeverity.INFO),
            "total": len(active)
        }


# Pre-configured thresholds for HVAC monitoring
HVAC_THRESHOLDS = [
    ThresholdConfig(
        metric_name="temperature",
        warning_low=18.0,
        warning_high=26.0,
        critical_low=15.0,
        critical_high=30.0,
        unit="°C"
    ),
    ThresholdConfig(
        metric_name="co2",
        warning_high=1000,
        critical_high=2000,
        unit="ppm"
    ),
    ThresholdConfig(
        metric_name="humidity",
        warning_low=30,
        warning_high=60,
        critical_low=20,
        critical_high=70,
        unit="%RH"
    ),
    ThresholdConfig(
        metric_name="air_velocity",
        warning_high=0.25,
        critical_high=0.35,
        unit="m/s"
    ),
    ThresholdConfig(
        metric_name="pmv",
        warning_low=-1.0,
        warning_high=1.0,
        critical_low=-2.0,
        critical_high=2.0,
        unit=""
    ),
]


def create_hvac_alert_manager() -> AlertManager:
    """Create an AlertManager pre-configured for HVAC monitoring."""
    manager = AlertManager()
    
    for config in HVAC_THRESHOLDS:
        manager.configure_threshold(config)
    
    # Configure anomaly detection for key metrics
    for metric in ["temperature", "co2", "air_velocity"]:
        manager.configure_anomaly(AnomalyConfig(
            metric_name=metric,
            window_size=60,
            sigma_threshold=3.0
        ))
    
    return manager
