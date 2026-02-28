"""
DISCOVERY ENGINE — Autonomous Discovery Orchestrator

The DiscoveryEngine is the central orchestrator for cross-primitive
autonomous discovery. It:
    1. Manages primitive pipelines
    2. Collects and aggregates findings
    3. Generates attestation artifacts
    4. Provides streaming discovery for continuous monitoring

Constitutional Reference: AUTONOMOUS_DISCOVERY_ENGINE.md

Key Features:
    - Multi-primitive orchestration
    - Finding aggregation and deduplication
    - SHA256 attestation for reproducibility
    - Streaming mode for real-time analysis
    - Domain-specific presets (DeFi, Physics, etc.)

Example:
    >>> from ontic.ml.discovery import DiscoveryEngine
    >>> 
    >>> engine = DiscoveryEngine(domain="defi")
    >>> 
    >>> # One-shot discovery
    >>> findings = engine.discover(contract_state)
    >>> 
    >>> # Streaming discovery
    >>> for finding in engine.stream(data_generator):
    ...     if finding.severity >= Severity.HIGH:
    ...         alert(finding)

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Type, Union

import torch

from ontic.ml.discovery.findings import (
    Finding,
    FindingCollection,
    FindingType,
    Severity,
)
from ontic.ml.discovery.pipelines import (
    CrossPrimitivePipeline,
    DeFiPipeline,
    GeometryPipeline,
    Pipeline,
    PipelineResult,
)
from ontic.ml.discovery.protocol import (
    GenesisPrimitive,
    PrimitiveConfig,
    PrimitiveResult,
    PrimitiveType,
)


@dataclass
class DiscoveryConfig:
    """
    Configuration for the DiscoveryEngine.
    
    Attributes:
        domain: Target domain ('defi', 'physics', 'geometry', 'custom')
        pipeline: Custom pipeline (if domain='custom')
        severity_threshold: Minimum severity to report
        dedup_findings: Whether to deduplicate similar findings
        enable_attestation: Whether to generate attestation artifacts
        attestation_dir: Directory for attestation files
        seed: Random seed for reproducibility
    """
    
    domain: str = "custom"
    pipeline: Optional[Pipeline] = None
    severity_threshold: Severity = Severity.INFO
    dedup_findings: bool = True
    enable_attestation: bool = True
    attestation_dir: str = "attestations"
    seed: int = 42
    
    def __post_init__(self) -> None:
        torch.manual_seed(self.seed)


@dataclass
class DiscoveryRun:
    """
    Metadata for a single discovery run.
    
    Attributes:
        run_id: Unique identifier for this run
        started_at: Timestamp when run started
        completed_at: Timestamp when run completed
        config: Configuration used
        input_hash: SHA256 of input data
        findings: Collected findings
        attestation: Attestation data (if enabled)
    """
    
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    config: Optional[DiscoveryConfig] = None
    input_hash: Optional[str] = None
    findings: Optional[FindingCollection] = None
    pipeline_result: Optional[PipelineResult] = None
    attestation: Optional["DiscoveryAttestation"] = None
    
    def duration_seconds(self) -> Optional[float]:
        """Get run duration in seconds."""
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert run to dictionary."""
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds(),
            "input_hash": self.input_hash,
            "n_findings": len(self.findings) if self.findings else 0,
            "findings_summary": self.findings.summary() if self.findings else None,
        }


@dataclass
class DiscoveryAttestation:
    """
    Cryptographic attestation for a discovery run.
    
    Provides SHA256 hashes for reproducibility and verification.
    
    Attributes:
        run_id: Associated run ID
        timestamp: Attestation timestamp
        input_hash: SHA256 of input data
        findings_hash: SHA256 of findings
        pipeline_hash: SHA256 of pipeline configuration
        attestation_hash: SHA256 of entire attestation
    """
    
    run_id: str
    timestamp: datetime
    input_hash: str
    findings_hash: str
    pipeline_hash: str
    attestation_hash: str = ""
    
    def __post_init__(self) -> None:
        if not self.attestation_hash:
            self.attestation_hash = self._compute_attestation_hash()
    
    def _compute_attestation_hash(self) -> str:
        """Compute hash of the attestation itself."""
        content = {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "input_hash": self.input_hash,
            "findings_hash": self.findings_hash,
            "pipeline_hash": self.pipeline_hash,
        }
        content_json = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_json.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert attestation to dictionary."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "input_hash": self.input_hash,
            "findings_hash": self.findings_hash,
            "pipeline_hash": self.pipeline_hash,
            "attestation_hash": self.attestation_hash,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save attestation to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())
    
    @classmethod
    def from_json(cls, json_str: str) -> "DiscoveryAttestation":
        """Load from JSON string."""
        data = json.loads(json_str)
        return cls(
            run_id=data["run_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            input_hash=data["input_hash"],
            findings_hash=data["findings_hash"],
            pipeline_hash=data["pipeline_hash"],
            attestation_hash=data["attestation_hash"],
        )
    
    def verify(self) -> bool:
        """Verify attestation hash matches content."""
        expected_hash = self._compute_attestation_hash()
        return expected_hash == self.attestation_hash


class DiscoveryEngine:
    """
    Autonomous Discovery Engine — the flagship orchestrator.
    
    Chains Genesis primitives to discover patterns that would be
    invisible to any single analysis method.
    
    Example:
        >>> engine = DiscoveryEngine(domain="defi")
        >>> 
        >>> # Discover patterns
        >>> result = engine.discover(data)
        >>> 
        >>> # Check findings
        >>> for finding in result.findings:
        ...     print(f"[{finding.severity}] {finding.summary}")
        >>> 
        >>> # Get attestation
        >>> print(f"Attestation: {result.attestation.attestation_hash}")
    """
    
    # Domain presets
    DOMAIN_PIPELINES: Dict[str, Type[Pipeline]] = {
        "defi": DeFiPipeline,
        "geometry": GeometryPipeline,
        "full": CrossPrimitivePipeline,
    }
    
    def __init__(
        self,
        domain: str = "full",
        config: Optional[DiscoveryConfig] = None,
        **kwargs,
    ) -> None:
        """
        Initialize DiscoveryEngine.
        
        Args:
            domain: Target domain ('defi', 'physics', 'geometry', 'full', 'custom')
            config: Discovery configuration
            **kwargs: Additional arguments passed to pipeline
        """
        # Set up config
        if config is None:
            config = DiscoveryConfig(domain=domain)
        self.config = config
        
        # Create pipeline
        if domain == "custom" and config.pipeline is not None:
            self.pipeline = config.pipeline
        elif domain in self.DOMAIN_PIPELINES:
            pipeline_class = self.DOMAIN_PIPELINES[domain]
            self.pipeline = pipeline_class(**kwargs)
        else:
            # Default to full cross-primitive pipeline
            self.pipeline = CrossPrimitivePipeline(**kwargs)
        
        # Run history
        self.runs: List[DiscoveryRun] = []
        self._current_run: Optional[DiscoveryRun] = None
    
    def discover(self, input_data: Any) -> DiscoveryRun:
        """
        Run discovery on input data.
        
        Args:
            input_data: Data to analyze
            
        Returns:
            DiscoveryRun with findings and attestation
        """
        # Create run
        run = DiscoveryRun(config=self.config)
        run.input_hash = self._hash_input(input_data)
        self._current_run = run
        
        # Execute pipeline
        pipeline_result = self.pipeline.execute(input_data)
        
        # Process findings
        findings = pipeline_result.findings
        
        # Filter by severity
        filtered_findings = FindingCollection()
        for finding in findings:
            if finding.severity >= self.config.severity_threshold:
                filtered_findings.add(finding)
        
        # Deduplicate if enabled
        if self.config.dedup_findings:
            filtered_findings = self._deduplicate_findings(filtered_findings)
        
        # Complete run
        run.completed_at = datetime.now(timezone.utc)
        run.findings = filtered_findings
        run.pipeline_result = pipeline_result
        
        # Generate attestation
        if self.config.enable_attestation:
            run.attestation = self._create_attestation(run)
            
            # Save attestation
            attestation_path = Path(self.config.attestation_dir) / f"{run.run_id}.json"
            run.attestation.save(attestation_path)
        
        # Store run
        self.runs.append(run)
        self._current_run = None
        
        return run
    
    def stream(
        self,
        data_generator: Iterator[Any],
        batch_size: int = 1,
    ) -> Generator[Finding, None, None]:
        """
        Stream findings from a data generator.
        
        Yields findings as they are discovered, enabling
        real-time monitoring and alerting.
        
        Args:
            data_generator: Iterator yielding data samples
            batch_size: Number of samples to batch before analysis
            
        Yields:
            Finding objects as they are discovered
        """
        batch: List[Any] = []
        
        for data in data_generator:
            batch.append(data)
            
            if len(batch) >= batch_size:
                # Run discovery on batch
                # For single items, just process directly
                if batch_size == 1:
                    run = self.discover(batch[0])
                else:
                    # Batch processing (implementation depends on data type)
                    run = self.discover(batch)
                
                # Yield findings
                for finding in run.findings:
                    yield finding
                
                batch.clear()
        
        # Process remaining
        if batch:
            if len(batch) == 1:
                run = self.discover(batch[0])
            else:
                run = self.discover(batch)
            
            for finding in run.findings:
                yield finding
    
    def _hash_input(self, input_data: Any) -> str:
        """Compute SHA256 hash of input data."""
        if isinstance(input_data, torch.Tensor):
            content = input_data.numpy().tobytes()
        elif isinstance(input_data, dict):
            content = json.dumps(
                {k: str(v) for k, v in input_data.items()},
                sort_keys=True,
            ).encode()
        elif isinstance(input_data, (list, tuple)):
            content = str(input_data).encode()
        else:
            content = str(input_data).encode()
        
        return hashlib.sha256(content).hexdigest()
    
    def _deduplicate_findings(
        self, findings: FindingCollection
    ) -> FindingCollection:
        """Remove duplicate findings based on content hash."""
        seen_hashes: set = set()
        deduped = FindingCollection()
        
        for finding in findings:
            if finding.hash not in seen_hashes:
                seen_hashes.add(finding.hash)
                deduped.add(finding)
        
        return deduped
    
    def _create_attestation(self, run: DiscoveryRun) -> DiscoveryAttestation:
        """Create attestation for a discovery run."""
        # Hash findings
        findings_content = run.findings.to_json() if run.findings else ""
        findings_hash = hashlib.sha256(findings_content.encode()).hexdigest()
        
        # Hash pipeline configuration
        pipeline_content = json.dumps({
            "name": self.pipeline.name,
            "stages": self.pipeline.stage_names,
        }, sort_keys=True)
        pipeline_hash = hashlib.sha256(pipeline_content.encode()).hexdigest()
        
        return DiscoveryAttestation(
            run_id=run.run_id,
            timestamp=datetime.now(timezone.utc),
            input_hash=run.input_hash or "",
            findings_hash=findings_hash,
            pipeline_hash=pipeline_hash,
        )
    
    def get_run(self, run_id: str) -> Optional[DiscoveryRun]:
        """Get a run by ID."""
        for run in self.runs:
            if run.run_id == run_id:
                return run
        return None
    
    def summary(self) -> Dict[str, Any]:
        """Get engine summary."""
        total_findings = sum(len(r.findings) for r in self.runs if r.findings)
        critical_findings = sum(
            len(r.findings.critical_findings)
            for r in self.runs
            if r.findings
        )
        
        return {
            "domain": self.config.domain,
            "pipeline": self.pipeline.name,
            "stages": self.pipeline.stage_names,
            "n_runs": len(self.runs),
            "total_findings": total_findings,
            "critical_findings": critical_findings,
            "attestation_enabled": self.config.enable_attestation,
        }
    
    def reset(self) -> None:
        """Reset engine state."""
        self.pipeline.reset_all()
        self.runs.clear()
    
    def __repr__(self) -> str:
        return f"DiscoveryEngine(domain={self.config.domain}, pipeline={self.pipeline.name})"
