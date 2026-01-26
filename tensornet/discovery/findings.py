"""
FINDINGS — Discovery Result Types

Strongly-typed dataclasses for all finding categories:
    - Anomaly: Unexpected patterns, outliers, distribution drifts
    - Invariant: Conservation laws, symmetries, fixed points
    - Bottleneck: Resource constraints, phase transitions, critical points
    - Prediction: Emergent patterns, failure modes, optimization targets

Each finding carries:
    - Source primitive(s) that detected it
    - Severity level (INFO, LOW, MEDIUM, HIGH, CRITICAL)
    - Evidence (numerical measurements, thresholds crossed)
    - Remediation suggestions (when applicable)

Constitutional Reference: CONSTITUTION.md, Article I (Proof Standards)
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


class FindingType(Enum):
    """Categories of discoverable patterns."""
    
    ANOMALY = auto()
    INVARIANT = auto()
    BOTTLENECK = auto()
    PREDICTION = auto()


class Severity(Enum):
    """Finding severity levels aligned with security standards."""
    
    INFO = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5
    
    def __str__(self) -> str:
        return self.name
    
    def __lt__(self, other: "Severity") -> bool:
        return self.value < other.value
    
    def __le__(self, other: "Severity") -> bool:
        return self.value <= other.value


@dataclass
class Finding:
    """
    Base class for all discovery findings.
    
    Every finding is immutable after creation and carries a cryptographic
    hash for attestation purposes.
    
    Attributes:
        id: Unique identifier (UUID)
        type: Finding category (ANOMALY, INVARIANT, BOTTLENECK, PREDICTION)
        severity: Impact level (INFO through CRITICAL)
        summary: One-line human-readable description
        primitives: List of Genesis primitives that contributed to detection
        evidence: Numerical evidence supporting the finding
        timestamp: UTC timestamp of discovery
        metadata: Additional context (optional)
        
    Example:
        >>> finding = Finding(
        ...     type=FindingType.ANOMALY,
        ...     severity=Severity.HIGH,
        ...     summary="Distribution drift detected: Wasserstein distance > threshold",
        ...     primitives=["OT"],
        ...     evidence={"wasserstein_distance": 0.234, "threshold": 0.1},
        ... )
    """
    
    type: FindingType = FindingType.ANOMALY  # Default, subclasses override in __post_init__
    severity: Severity = Severity.INFO
    summary: str = ""
    primitives: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate finding after initialization."""
        if not self.summary:
            raise ValueError("Finding summary cannot be empty")
        if not self.primitives:
            raise ValueError("Finding must have at least one contributing primitive")
    
    @property
    def hash(self) -> str:
        """
        SHA256 hash of finding content for attestation.
        
        The hash is computed from type, severity, summary, primitives, and evidence.
        Excludes id, timestamp, and metadata for reproducibility.
        """
        content = {
            "type": self.type.name,
            "severity": self.severity.name,
            "summary": self.summary,
            "primitives": sorted(self.primitives),
            "evidence": self._serialize_evidence(),
        }
        content_json = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_json.encode()).hexdigest()
    
    def _serialize_evidence(self) -> Dict[str, Any]:
        """Serialize evidence for hashing (handles torch tensors)."""
        result = {}
        for key, value in self.evidence.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.tolist()
            elif isinstance(value, (list, tuple)):
                result[key] = [
                    v.tolist() if isinstance(v, torch.Tensor) else v
                    for v in value
                ]
            else:
                result[key] = value
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert finding to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.name,
            "severity": self.severity.name,
            "summary": self.summary,
            "primitives": self.primitives,
            "evidence": self._serialize_evidence(),
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "hash": self.hash,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Finding":
        """Reconstruct finding from dictionary."""
        return cls(
            id=data["id"],
            type=FindingType[data["type"]],
            severity=Severity[data["severity"]],
            summary=data["summary"],
            primitives=data["primitives"],
            evidence=data["evidence"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )
    
    def __repr__(self) -> str:
        return (
            f"Finding(type={self.type.name}, severity={self.severity.name}, "
            f"summary={self.summary!r})"
        )


@dataclass
class AnomalyFinding(Finding):
    """
    Specialized finding for anomaly detection.
    
    Anomalies include:
        - Distribution drift (OT)
        - Spectral outliers (RMT)
        - Topological defects (PH)
        - Feature space outliers (RKHS)
    
    Attributes:
        anomaly_score: Numerical score indicating anomaly magnitude
        baseline: Reference value for comparison
        deviation: Standard deviations from baseline
        location: Where in the data the anomaly was detected (optional)
    """
    
    # Override type with default
    type: FindingType = field(default=FindingType.ANOMALY)
    anomaly_score: float = 0.0
    baseline: Optional[float] = None
    deviation: Optional[float] = None
    location: Optional[Union[int, Tuple[int, ...], str]] = None
    
    def __post_init__(self) -> None:
        self.type = FindingType.ANOMALY
        super().__post_init__()
        # Add anomaly-specific evidence
        self.evidence["anomaly_score"] = self.anomaly_score
        if self.baseline is not None:
            self.evidence["baseline"] = self.baseline
        if self.deviation is not None:
            self.evidence["deviation_sigma"] = self.deviation
        if self.location is not None:
            self.evidence["location"] = str(self.location)


@dataclass
class InvariantFinding(Finding):
    """
    Specialized finding for discovered invariants.
    
    Invariants include:
        - Conservation laws (energy, mass, charge)
        - Symmetries (rotational, translational)
        - Fixed points in dynamical systems
        - Topological invariants (Betti numbers)
    
    Attributes:
        invariant_name: Human-readable name for the invariant
        value: Current value of the invariant
        tolerance: Acceptable deviation from exact invariance
        drift: Observed drift rate (if any)
    """
    
    # Override type with default
    type: FindingType = field(default=FindingType.INVARIANT)
    invariant_name: str = ""
    value: Optional[float] = None
    tolerance: float = 1e-10
    drift: Optional[float] = None
    
    def __post_init__(self) -> None:
        self.type = FindingType.INVARIANT
        super().__post_init__()
        self.evidence["invariant_name"] = self.invariant_name
        if self.value is not None:
            self.evidence["value"] = self.value
        self.evidence["tolerance"] = self.tolerance
        if self.drift is not None:
            self.evidence["drift_rate"] = self.drift


@dataclass
class BottleneckFinding(Finding):
    """
    Specialized finding for bottleneck detection.
    
    Bottlenecks include:
        - Computational constraints (memory, time)
        - Physical phase transitions
        - Information bottlenecks in neural architectures
        - Transport bottlenecks (OT)
    
    Attributes:
        bottleneck_type: Category of bottleneck (compute, memory, physics, transport)
        capacity: Maximum observed throughput
        utilization: Current utilization fraction
        critical_threshold: Threshold at which bottleneck becomes critical
    """
    
    # Override type with default
    type: FindingType = field(default=FindingType.BOTTLENECK)
    bottleneck_type: str = "compute"
    capacity: Optional[float] = None
    utilization: Optional[float] = None
    critical_threshold: float = 0.9
    
    def __post_init__(self) -> None:
        self.type = FindingType.BOTTLENECK
        super().__post_init__()
        self.evidence["bottleneck_type"] = self.bottleneck_type
        if self.capacity is not None:
            self.evidence["capacity"] = self.capacity
        if self.utilization is not None:
            self.evidence["utilization"] = self.utilization
        self.evidence["critical_threshold"] = self.critical_threshold


@dataclass
class PredictionFinding(Finding):
    """
    Specialized finding for predictive patterns.
    
    Predictions include:
        - Failure mode predictions
        - Trend extrapolations
        - Anomaly precursors
        - Optimization targets
    
    Attributes:
        prediction: The predicted value or state
        confidence: Confidence level (0-1)
        horizon: Time horizon for prediction
        supporting_evidence: Additional data supporting prediction
    """
    
    # Override type with default
    type: FindingType = field(default=FindingType.PREDICTION)
    prediction: Optional[Any] = None
    confidence: float = 0.5
    horizon: Optional[str] = None
    supporting_evidence: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        self.type = FindingType.PREDICTION
        super().__post_init__()
        if self.prediction is not None:
            self.evidence["prediction"] = str(self.prediction)
        self.evidence["confidence"] = self.confidence
        if self.horizon is not None:
            self.evidence["horizon"] = self.horizon
        self.evidence.update(self.supporting_evidence)


@dataclass
class FindingCollection:
    """
    Collection of findings from a discovery run.
    
    Provides filtering, sorting, and aggregation capabilities.
    
    Attributes:
        findings: List of Finding objects
        run_id: Identifier for the discovery run that produced these findings
        created_at: Timestamp when collection was created
    """
    
    findings: List[Finding] = field(default_factory=list)
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def add(self, finding: Finding) -> None:
        """Add a finding to the collection."""
        self.findings.append(finding)
    
    def add_all(self, findings: List[Finding]) -> None:
        """Add multiple findings to the collection."""
        self.findings.extend(findings)
    
    def __len__(self) -> int:
        return len(self.findings)
    
    def __iter__(self):
        return iter(self.findings)
    
    def __getitem__(self, index: int) -> Finding:
        return self.findings[index]
    
    def filter_by_type(self, finding_type: FindingType) -> List[Finding]:
        """Return all findings of a specific type."""
        return [f for f in self.findings if f.type == finding_type]
    
    def filter_by_severity(
        self,
        min_severity: Severity = Severity.INFO,
        max_severity: Severity = Severity.CRITICAL,
    ) -> List[Finding]:
        """Return findings within a severity range."""
        return [
            f for f in self.findings
            if min_severity <= f.severity <= max_severity
        ]
    
    def filter_by_primitive(self, primitive: str) -> List[Finding]:
        """Return findings involving a specific primitive."""
        return [f for f in self.findings if primitive in f.primitives]
    
    def sort_by_severity(self, descending: bool = True) -> List[Finding]:
        """Return findings sorted by severity."""
        return sorted(
            self.findings,
            key=lambda f: f.severity.value,
            reverse=descending,
        )
    
    @property
    def critical_findings(self) -> List[Finding]:
        """Return all CRITICAL severity findings."""
        return self.filter_by_severity(Severity.CRITICAL, Severity.CRITICAL)
    
    @property
    def high_findings(self) -> List[Finding]:
        """Return all HIGH severity findings."""
        return self.filter_by_severity(Severity.HIGH, Severity.HIGH)
    
    @property
    def anomalies(self) -> List[Finding]:
        """Return all anomaly findings."""
        return self.filter_by_type(FindingType.ANOMALY)
    
    @property
    def invariants(self) -> List[Finding]:
        """Return all invariant findings."""
        return self.filter_by_type(FindingType.INVARIANT)
    
    @property
    def bottlenecks(self) -> List[Finding]:
        """Return all bottleneck findings."""
        return self.filter_by_type(FindingType.BOTTLENECK)
    
    @property
    def predictions(self) -> List[Finding]:
        """Return all prediction findings."""
        return self.filter_by_type(FindingType.PREDICTION)
    
    def summary(self) -> Dict[str, Any]:
        """Return a summary of the collection."""
        severity_counts = {s.name: 0 for s in Severity}
        type_counts = {t.name: 0 for t in FindingType}
        primitive_counts: Dict[str, int] = {}
        
        for finding in self.findings:
            severity_counts[finding.severity.name] += 1
            type_counts[finding.type.name] += 1
            for primitive in finding.primitives:
                primitive_counts[primitive] = primitive_counts.get(primitive, 0) + 1
        
        return {
            "total": len(self.findings),
            "by_severity": severity_counts,
            "by_type": type_counts,
            "by_primitive": primitive_counts,
            "run_id": self.run_id,
            "created_at": self.created_at.isoformat(),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert collection to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "created_at": self.created_at.isoformat(),
            "summary": self.summary(),
            "findings": [f.to_dict() for f in self.findings],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FindingCollection":
        """Reconstruct collection from dictionary."""
        collection = cls(
            run_id=data["run_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
        )
        for finding_data in data["findings"]:
            # Determine finding subclass based on type
            finding_type = FindingType[finding_data["type"]]
            if finding_type == FindingType.ANOMALY:
                finding = AnomalyFinding.from_dict(finding_data)
            elif finding_type == FindingType.INVARIANT:
                finding = InvariantFinding.from_dict(finding_data)
            elif finding_type == FindingType.BOTTLENECK:
                finding = BottleneckFinding.from_dict(finding_data)
            elif finding_type == FindingType.PREDICTION:
                finding = PredictionFinding.from_dict(finding_data)
            else:
                finding = Finding.from_dict(finding_data)
            collection.add(finding)
        return collection
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize collection to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> "FindingCollection":
        """Deserialize collection from JSON string."""
        return cls.from_dict(json.loads(json_str))
