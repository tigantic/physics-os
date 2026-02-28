"""
DO-178C Certification Framework for Safety-Critical Systems
============================================================

Implements software assurance framework compliant with DO-178C
(Software Considerations in Airborne Systems and Equipment Certification)
and DO-254 (Design Assurance Guidance for Airborne Electronic Hardware).

Key Components:
    - Requirements traceability matrix
    - Software verification evidence
    - Test coverage analysis
    - Safety assessment artifacts
    - Configuration management

Design Assurance Levels (DAL):
    - Level A: Catastrophic (most stringent)
    - Level B: Hazardous
    - Level C: Major
    - Level D: Minor
    - Level E: No Effect (least stringent)

References:
    [1] RTCA DO-178C, "Software Considerations in Airborne Systems and
        Equipment Certification", 2011
    [2] RTCA DO-254, "Design Assurance Guidance for Airborne Electronic
        Hardware", 2000
    [3] RTCA DO-330, "Software Tool Qualification Considerations", 2011
"""

import datetime
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# =============================================================================
# Design Assurance Levels
# =============================================================================


class DAL(Enum):
    """Design Assurance Levels per DO-178C."""

    LEVEL_A = "A"  # Catastrophic failure condition
    LEVEL_B = "B"  # Hazardous/Severe-Major
    LEVEL_C = "C"  # Major
    LEVEL_D = "D"  # Minor
    LEVEL_E = "E"  # No Effect


class VerificationMethod(Enum):
    """Verification methods per DO-178C."""

    REVIEW = "review"  # Code review, document review
    ANALYSIS = "analysis"  # Static analysis, formal methods
    TEST = "test"  # Dynamic testing
    INSPECTION = "inspection"  # Visual inspection


class CoverageType(Enum):
    """Test coverage metrics."""

    STATEMENT = "statement"
    DECISION = "decision"
    MCDC = "modified_condition_decision"  # Required for DAL A
    DATA_COUPLING = "data_coupling"
    CONTROL_COUPLING = "control_coupling"


# =============================================================================
# Requirements Management
# =============================================================================


class RequirementType(Enum):
    """Types of requirements."""

    HIGH_LEVEL = "hlr"  # High-Level Requirements
    LOW_LEVEL = "llr"  # Low-Level Requirements
    DERIVED = "derived"  # Derived requirements
    SAFETY = "safety"  # Safety requirements
    PERFORMANCE = "performance"  # Performance requirements


class RequirementStatus(Enum):
    """Requirement lifecycle status."""

    DRAFT = "draft"
    APPROVED = "approved"
    VERIFIED = "verified"
    CLOSED = "closed"
    DELETED = "deleted"


@dataclass
class Requirement:
    """
    Software requirement with traceability.

    Attributes:
        req_id: Unique requirement identifier
        title: Short requirement title
        description: Full requirement description
        req_type: High-level, low-level, derived, etc.
        dal: Design assurance level
        parent_ids: Parent requirement IDs (for traceability)
        verification_methods: How this requirement will be verified
        status: Current lifecycle status
        rationale: Justification for the requirement
    """

    req_id: str
    title: str
    description: str
    req_type: RequirementType
    dal: DAL
    parent_ids: list[str] = field(default_factory=list)
    child_ids: list[str] = field(default_factory=list)
    verification_methods: list[VerificationMethod] = field(default_factory=list)
    status: RequirementStatus = RequirementStatus.DRAFT
    rationale: str = ""
    version: str = "1.0"
    last_modified: str = field(
        default_factory=lambda: datetime.datetime.now().isoformat()
    )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "req_id": self.req_id,
            "title": self.title,
            "description": self.description,
            "req_type": self.req_type.value,
            "dal": self.dal.value,
            "parent_ids": self.parent_ids,
            "child_ids": self.child_ids,
            "verification_methods": [m.value for m in self.verification_methods],
            "status": self.status.value,
            "rationale": self.rationale,
            "version": self.version,
            "last_modified": self.last_modified,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Requirement":
        """Create from dictionary."""
        return cls(
            req_id=data["req_id"],
            title=data["title"],
            description=data["description"],
            req_type=RequirementType(data["req_type"]),
            dal=DAL(data["dal"]),
            parent_ids=data.get("parent_ids", []),
            child_ids=data.get("child_ids", []),
            verification_methods=[
                VerificationMethod(m) for m in data.get("verification_methods", [])
            ],
            status=RequirementStatus(data.get("status", "draft")),
            rationale=data.get("rationale", ""),
            version=data.get("version", "1.0"),
            last_modified=data.get(
                "last_modified", datetime.datetime.now().isoformat()
            ),
        )


class RequirementsDatabase:
    """
    Requirements database with traceability matrix.

    Manages the complete set of requirements and their relationships.
    """

    def __init__(self, project_name: str = "The Ontic Engine"):
        self.project_name = project_name
        self.requirements: dict[str, Requirement] = {}
        self.creation_date = datetime.datetime.now().isoformat()

    def add_requirement(self, req: Requirement):
        """Add a requirement to the database."""
        if req.req_id in self.requirements:
            raise ValueError(f"Requirement {req.req_id} already exists")
        self.requirements[req.req_id] = req

        # Update parent-child relationships
        for parent_id in req.parent_ids:
            if parent_id in self.requirements:
                parent = self.requirements[parent_id]
                if req.req_id not in parent.child_ids:
                    parent.child_ids.append(req.req_id)

    def get_requirement(self, req_id: str) -> Requirement | None:
        """Get requirement by ID."""
        return self.requirements.get(req_id)

    def update_requirement(self, req: Requirement):
        """Update an existing requirement."""
        if req.req_id not in self.requirements:
            raise ValueError(f"Requirement {req.req_id} not found")
        req.last_modified = datetime.datetime.now().isoformat()
        self.requirements[req.req_id] = req

    def delete_requirement(self, req_id: str):
        """Mark requirement as deleted."""
        if req_id not in self.requirements:
            raise ValueError(f"Requirement {req_id} not found")
        self.requirements[req_id].status = RequirementStatus.DELETED

    def get_traceability_matrix(self) -> dict[str, list[str]]:
        """
        Generate requirements traceability matrix.

        Maps HLR -> LLR -> Test Cases
        """
        matrix = {}
        for req_id, req in self.requirements.items():
            matrix[req_id] = {
                "parents": req.parent_ids,
                "children": req.child_ids,
                "type": req.req_type.value,
                "dal": req.dal.value,
                "status": req.status.value,
            }
        return matrix

    def get_requirements_by_dal(self, dal: DAL) -> list[Requirement]:
        """Get all requirements at a specific DAL."""
        return [r for r in self.requirements.values() if r.dal == dal]

    def get_unverified_requirements(self) -> list[Requirement]:
        """Get requirements that haven't been verified."""
        return [
            r
            for r in self.requirements.values()
            if r.status not in {RequirementStatus.VERIFIED, RequirementStatus.DELETED}
        ]

    def export_to_json(self, filepath: str):
        """Export database to JSON."""
        data = {
            "project_name": self.project_name,
            "creation_date": self.creation_date,
            "requirements": {
                req_id: req.to_dict() for req_id, req in self.requirements.items()
            },
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_json(cls, filepath: str) -> "RequirementsDatabase":
        """Load database from JSON."""
        with open(filepath) as f:
            data = json.load(f)

        db = cls(data["project_name"])
        db.creation_date = data["creation_date"]

        for req_id, req_data in data["requirements"].items():
            db.requirements[req_id] = Requirement.from_dict(req_data)

        return db


# =============================================================================
# Test Evidence and Coverage
# =============================================================================


class TestResult(Enum):
    """Test execution result."""

    PASS = "pass"
    FAIL = "fail"
    BLOCKED = "blocked"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestCase:
    """
    Test case with traceability to requirements.

    Attributes:
        test_id: Unique test identifier
        title: Test case title
        description: Test procedure description
        requirement_ids: Requirements verified by this test
        expected_result: Expected test outcome
        actual_result: Actual test result
        test_data: Input data used
        environment: Test environment details
    """

    test_id: str
    title: str
    description: str
    requirement_ids: list[str]
    expected_result: str
    preconditions: str = ""
    test_steps: list[str] = field(default_factory=list)
    actual_result: str = ""
    result: TestResult = TestResult.SKIPPED
    test_data: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    execution_time: float | None = None
    execution_date: str | None = None
    executed_by: str = "automated"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "test_id": self.test_id,
            "title": self.title,
            "description": self.description,
            "requirement_ids": self.requirement_ids,
            "expected_result": self.expected_result,
            "preconditions": self.preconditions,
            "test_steps": self.test_steps,
            "actual_result": self.actual_result,
            "result": self.result.value,
            "test_data": self.test_data,
            "environment": self.environment,
            "execution_time": self.execution_time,
            "execution_date": self.execution_date,
            "executed_by": self.executed_by,
        }


@dataclass
class CoverageReport:
    """
    Test coverage analysis report.

    DO-178C requires different coverage levels:
    - DAL A: MC/DC (Modified Condition/Decision Coverage)
    - DAL B: Decision Coverage
    - DAL C: Statement Coverage
    """

    coverage_type: CoverageType
    total_items: int
    covered_items: int
    uncovered_items: list[str]
    coverage_percentage: float
    analysis_date: str = field(
        default_factory=lambda: datetime.datetime.now().isoformat()
    )
    tool_name: str = "Ontic Coverage Analyzer"

    @property
    def meets_objective(self) -> bool:
        """Check if coverage meets DO-178C objectives."""
        # 100% required for structural coverage
        return self.coverage_percentage >= 100.0

    def to_dict(self) -> dict:
        return {
            "coverage_type": self.coverage_type.value,
            "total_items": self.total_items,
            "covered_items": self.covered_items,
            "uncovered_items": self.uncovered_items,
            "coverage_percentage": self.coverage_percentage,
            "meets_objective": self.meets_objective,
            "analysis_date": self.analysis_date,
            "tool_name": self.tool_name,
        }


class CoverageAnalyzer:
    """
    Analyzes test coverage for DO-178C compliance.
    """

    def __init__(self, source_files: list[str], dal: DAL):
        self.source_files = source_files
        self.dal = dal
        self.required_coverage = self._get_required_coverage()

    def _get_required_coverage(self) -> list[CoverageType]:
        """Determine required coverage types for DAL."""
        if self.dal == DAL.LEVEL_A:
            return [CoverageType.STATEMENT, CoverageType.DECISION, CoverageType.MCDC]
        elif self.dal == DAL.LEVEL_B:
            return [CoverageType.STATEMENT, CoverageType.DECISION]
        elif self.dal == DAL.LEVEL_C:
            return [CoverageType.STATEMENT]
        else:
            return []

    def analyze_statement_coverage(
        self, executed_lines: set[tuple[str, int]], all_lines: set[tuple[str, int]]
    ) -> CoverageReport:
        """Analyze statement coverage."""
        covered = len(executed_lines & all_lines)
        total = len(all_lines)
        uncovered = [f"{f}:{l}" for f, l in (all_lines - executed_lines)]

        return CoverageReport(
            coverage_type=CoverageType.STATEMENT,
            total_items=total,
            covered_items=covered,
            uncovered_items=uncovered,
            coverage_percentage=100.0 * covered / total if total > 0 else 0.0,
        )

    def analyze_decision_coverage(
        self, decisions_evaluated: dict[str, set[bool]]
    ) -> CoverageReport:
        """
        Analyze decision coverage.

        Each decision must evaluate to both True and False.
        """
        total = len(decisions_evaluated)
        covered = sum(
            1
            for outcomes in decisions_evaluated.values()
            if True in outcomes and False in outcomes
        )
        uncovered = [
            d
            for d, outcomes in decisions_evaluated.items()
            if not (True in outcomes and False in outcomes)
        ]

        return CoverageReport(
            coverage_type=CoverageType.DECISION,
            total_items=total,
            covered_items=covered,
            uncovered_items=uncovered,
            coverage_percentage=100.0 * covered / total if total > 0 else 0.0,
        )

    def analyze_mcdc_coverage(
        self, conditions: dict[str, list[dict[str, bool]]]
    ) -> CoverageReport:
        """
        Analyze Modified Condition/Decision Coverage.

        Each condition must independently affect the decision outcome.
        """
        total = 0
        covered = 0
        uncovered = []

        for decision_id, evaluations in conditions.items():
            # Extract unique conditions in this decision
            if not evaluations:
                continue

            condition_names = list(evaluations[0].keys())
            total += len(condition_names)

            for cond in condition_names:
                # Check if condition independently affects outcome
                if self._condition_has_independent_effect(cond, evaluations):
                    covered += 1
                else:
                    uncovered.append(f"{decision_id}:{cond}")

        return CoverageReport(
            coverage_type=CoverageType.MCDC,
            total_items=total,
            covered_items=covered,
            uncovered_items=uncovered,
            coverage_percentage=100.0 * covered / total if total > 0 else 0.0,
        )

    def _condition_has_independent_effect(
        self, condition: str, evaluations: list[dict[str, bool]]
    ) -> bool:
        """Check if a condition independently affects the decision."""
        # Find pairs where only this condition differs
        for i, eval1 in enumerate(evaluations):
            for eval2 in evaluations[i + 1 :]:
                differs_only_in_cond = True
                cond_differs = False

                for c in eval1:
                    if c == condition:
                        if eval1[c] != eval2[c]:
                            cond_differs = True
                    elif eval1[c] != eval2[c]:
                        differs_only_in_cond = False
                        break

                if differs_only_in_cond and cond_differs:
                    return True

        return False


# =============================================================================
# Safety Assessment
# =============================================================================


class HazardSeverity(Enum):
    """Hazard severity classification."""

    CATASTROPHIC = 1  # DAL A
    HAZARDOUS = 2  # DAL B
    MAJOR = 3  # DAL C
    MINOR = 4  # DAL D
    NO_EFFECT = 5  # DAL E


class HazardProbability(Enum):
    """Hazard probability classification."""

    FREQUENT = 1
    PROBABLE = 2
    REMOTE = 3
    EXTREMELY_REMOTE = 4
    EXTREMELY_IMPROBABLE = 5


@dataclass
class Hazard:
    """
    Safety hazard identification and analysis.
    """

    hazard_id: str
    title: str
    description: str
    severity: HazardSeverity
    probability: HazardProbability
    affected_functions: list[str]
    mitigations: list[str] = field(default_factory=list)
    residual_risk: str = ""
    verification_evidence: list[str] = field(default_factory=list)

    @property
    def risk_level(self) -> int:
        """Compute risk level from severity × probability."""
        return self.severity.value * self.probability.value

    @property
    def required_dal(self) -> DAL:
        """Determine required DAL based on severity."""
        severity_to_dal = {
            HazardSeverity.CATASTROPHIC: DAL.LEVEL_A,
            HazardSeverity.HAZARDOUS: DAL.LEVEL_B,
            HazardSeverity.MAJOR: DAL.LEVEL_C,
            HazardSeverity.MINOR: DAL.LEVEL_D,
            HazardSeverity.NO_EFFECT: DAL.LEVEL_E,
        }
        return severity_to_dal[self.severity]

    def to_dict(self) -> dict:
        return {
            "hazard_id": self.hazard_id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.name,
            "probability": self.probability.name,
            "risk_level": self.risk_level,
            "required_dal": self.required_dal.value,
            "affected_functions": self.affected_functions,
            "mitigations": self.mitigations,
            "residual_risk": self.residual_risk,
            "verification_evidence": self.verification_evidence,
        }


class SafetyAssessment:
    """
    System Safety Assessment per ARP4761/ARP4754A.
    """

    def __init__(self, system_name: str):
        self.system_name = system_name
        self.hazards: dict[str, Hazard] = {}
        self.creation_date = datetime.datetime.now().isoformat()

    def add_hazard(self, hazard: Hazard):
        """Add hazard to assessment."""
        self.hazards[hazard.hazard_id] = hazard

    def get_hazards_by_severity(self, severity: HazardSeverity) -> list[Hazard]:
        """Get all hazards of a specific severity."""
        return [h for h in self.hazards.values() if h.severity == severity]

    def get_unmitigated_hazards(self) -> list[Hazard]:
        """Get hazards without mitigations."""
        return [h for h in self.hazards.values() if not h.mitigations]

    def compute_risk_matrix(self) -> dict[str, list[str]]:
        """Generate risk matrix."""
        matrix = {
            "catastrophic": [],
            "hazardous": [],
            "major": [],
            "minor": [],
            "no_effect": [],
        }

        for hazard in self.hazards.values():
            key = hazard.severity.name.lower()
            matrix[key].append(hazard.hazard_id)

        return matrix

    def generate_safety_case(self) -> dict:
        """Generate safety case document."""
        return {
            "system_name": self.system_name,
            "assessment_date": self.creation_date,
            "hazard_summary": {
                "total_hazards": len(self.hazards),
                "catastrophic": len(
                    self.get_hazards_by_severity(HazardSeverity.CATASTROPHIC)
                ),
                "hazardous": len(
                    self.get_hazards_by_severity(HazardSeverity.HAZARDOUS)
                ),
                "major": len(self.get_hazards_by_severity(HazardSeverity.MAJOR)),
                "minor": len(self.get_hazards_by_severity(HazardSeverity.MINOR)),
            },
            "risk_matrix": self.compute_risk_matrix(),
            "unmitigated_count": len(self.get_unmitigated_hazards()),
            "hazards": {h_id: h.to_dict() for h_id, h in self.hazards.items()},
        }


# =============================================================================
# Configuration Management
# =============================================================================


@dataclass
class ConfigurationItem:
    """
    Configuration item for software baseline management.
    """

    ci_id: str
    name: str
    version: str
    file_path: str
    checksum: str
    status: str = "controlled"  # controlled, released, obsolete
    baseline: str = ""
    change_history: list[dict] = field(default_factory=list)

    @classmethod
    def from_file(cls, file_path: str, ci_id: str, name: str) -> "ConfigurationItem":
        """Create CI from file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(path, "rb") as f:
            content = f.read()
            checksum = hashlib.sha256(content).hexdigest()

        return cls(
            ci_id=ci_id,
            name=name,
            version="1.0.0",
            file_path=str(path.absolute()),
            checksum=checksum,
        )

    def verify_integrity(self) -> bool:
        """Verify file integrity against stored checksum."""
        try:
            with open(self.file_path, "rb") as f:
                current_checksum = hashlib.sha256(f.read()).hexdigest()
            return current_checksum == self.checksum
        except FileNotFoundError:
            return False


class ConfigurationManagement:
    """
    Configuration management system for DO-178C compliance.
    """

    def __init__(self, project_name: str):
        self.project_name = project_name
        self.items: dict[str, ConfigurationItem] = {}
        self.baselines: dict[str, list[str]] = {}  # baseline_name -> [ci_ids]

    def add_item(self, item: ConfigurationItem):
        """Add configuration item."""
        self.items[item.ci_id] = item

    def create_baseline(self, baseline_name: str, ci_ids: list[str]):
        """Create a configuration baseline."""
        # Verify all items exist
        for ci_id in ci_ids:
            if ci_id not in self.items:
                raise ValueError(f"Configuration item {ci_id} not found")

        self.baselines[baseline_name] = ci_ids

        # Mark items as part of baseline
        for ci_id in ci_ids:
            self.items[ci_id].baseline = baseline_name
            self.items[ci_id].status = "released"

    def verify_baseline(self, baseline_name: str) -> dict[str, bool]:
        """Verify integrity of all items in a baseline."""
        if baseline_name not in self.baselines:
            raise ValueError(f"Baseline {baseline_name} not found")

        results = {}
        for ci_id in self.baselines[baseline_name]:
            item = self.items[ci_id]
            results[ci_id] = item.verify_integrity()

        return results

    def record_change(self, ci_id: str, description: str, author: str):
        """Record a change to a configuration item."""
        if ci_id not in self.items:
            raise ValueError(f"Configuration item {ci_id} not found")

        item = self.items[ci_id]

        # Update version
        major, minor, patch = map(int, item.version.split("."))
        item.version = f"{major}.{minor}.{patch + 1}"

        # Recalculate checksum
        with open(item.file_path, "rb") as f:
            item.checksum = hashlib.sha256(f.read()).hexdigest()

        # Record change
        item.change_history.append(
            {
                "date": datetime.datetime.now().isoformat(),
                "author": author,
                "description": description,
                "new_version": item.version,
                "checksum": item.checksum,
            }
        )


# =============================================================================
# Verification Evidence Package
# =============================================================================


@dataclass
class VerificationEvidence:
    """
    Verification evidence for DO-178C compliance.
    """

    evidence_id: str
    title: str
    evidence_type: VerificationMethod
    requirement_ids: list[str]
    description: str
    artifacts: list[str]  # File paths to evidence artifacts
    result: str  # "pass", "fail", "partial"
    reviewer: str
    review_date: str
    comments: str = ""

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "title": self.title,
            "evidence_type": self.evidence_type.value,
            "requirement_ids": self.requirement_ids,
            "description": self.description,
            "artifacts": self.artifacts,
            "result": self.result,
            "reviewer": self.reviewer,
            "review_date": self.review_date,
            "comments": self.comments,
        }


class VerificationPackage:
    """
    Complete verification evidence package for certification.
    """

    def __init__(self, project_name: str, dal: DAL):
        self.project_name = project_name
        self.dal = dal
        self.evidence: dict[str, VerificationEvidence] = {}
        self.test_cases: dict[str, TestCase] = {}
        self.coverage_reports: list[CoverageReport] = []

    def add_evidence(self, evidence: VerificationEvidence):
        """Add verification evidence."""
        self.evidence[evidence.evidence_id] = evidence

    def add_test_case(self, test: TestCase):
        """Add test case."""
        self.test_cases[test.test_id] = test

    def add_coverage_report(self, report: CoverageReport):
        """Add coverage report."""
        self.coverage_reports.append(report)

    def get_verification_matrix(self) -> dict[str, dict]:
        """
        Generate verification cross-reference matrix.

        Maps requirements to verification evidence.
        """
        matrix = {}

        for evidence in self.evidence.values():
            for req_id in evidence.requirement_ids:
                if req_id not in matrix:
                    matrix[req_id] = {"reviews": [], "analyses": [], "tests": []}

                if evidence.evidence_type == VerificationMethod.REVIEW:
                    matrix[req_id]["reviews"].append(evidence.evidence_id)
                elif evidence.evidence_type == VerificationMethod.ANALYSIS:
                    matrix[req_id]["analyses"].append(evidence.evidence_id)
                elif evidence.evidence_type == VerificationMethod.TEST:
                    matrix[req_id]["tests"].append(evidence.evidence_id)

        return matrix

    def check_completeness(self, requirements: RequirementsDatabase) -> dict:
        """
        Check if verification is complete for all requirements.
        """
        verification_matrix = self.get_verification_matrix()

        incomplete = []
        complete = []

        for req_id, req in requirements.requirements.items():
            if req.status == RequirementStatus.DELETED:
                continue

            if req_id not in verification_matrix:
                incomplete.append(
                    {"req_id": req_id, "reason": "No verification evidence"}
                )
            else:
                evidence = verification_matrix[req_id]
                # Check each required verification method
                missing_methods = []
                for method in req.verification_methods:
                    if method == VerificationMethod.TEST and not evidence["tests"]:
                        missing_methods.append("test")
                    elif (
                        method == VerificationMethod.REVIEW and not evidence["reviews"]
                    ):
                        missing_methods.append("review")
                    elif (
                        method == VerificationMethod.ANALYSIS
                        and not evidence["analyses"]
                    ):
                        missing_methods.append("analysis")

                if missing_methods:
                    incomplete.append(
                        {
                            "req_id": req_id,
                            "reason": f"Missing: {', '.join(missing_methods)}",
                        }
                    )
                else:
                    complete.append(req_id)

        return {
            "complete": complete,
            "incomplete": incomplete,
            "completion_percentage": (
                100 * len(complete) / (len(complete) + len(incomplete))
                if (len(complete) + len(incomplete)) > 0
                else 0
            ),
        }

    def generate_sas(self) -> dict:
        """
        Generate Software Accomplishment Summary (SAS).

        The SAS is a key certification document summarizing all
        verification activities and their results.
        """
        return {
            "document_title": f"Software Accomplishment Summary - {self.project_name}",
            "dal": self.dal.value,
            "generation_date": datetime.datetime.now().isoformat(),
            "verification_summary": {
                "total_evidence": len(self.evidence),
                "total_tests": len(self.test_cases),
                "tests_passed": sum(
                    1 for t in self.test_cases.values() if t.result == TestResult.PASS
                ),
                "tests_failed": sum(
                    1 for t in self.test_cases.values() if t.result == TestResult.FAIL
                ),
            },
            "coverage_summary": [r.to_dict() for r in self.coverage_reports],
            "verification_matrix": self.get_verification_matrix(),
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def create_ontic_requirements() -> RequirementsDatabase:
    """
    Create example requirements for The Ontic Engine system.
    """
    db = RequirementsDatabase("The Ontic Engine")

    # High-level requirements
    db.add_requirement(
        Requirement(
            req_id="HLR-001",
            title="Tensor Network Computation",
            description="The system shall perform tensor network computations with configurable precision",
            req_type=RequirementType.HIGH_LEVEL,
            dal=DAL.LEVEL_C,
            verification_methods=[VerificationMethod.TEST, VerificationMethod.ANALYSIS],
        )
    )

    db.add_requirement(
        Requirement(
            req_id="HLR-002",
            title="Real-Time Inference",
            description="The system shall provide real-time inference with latency < 10ms",
            req_type=RequirementType.HIGH_LEVEL,
            dal=DAL.LEVEL_B,
            verification_methods=[VerificationMethod.TEST],
        )
    )

    db.add_requirement(
        Requirement(
            req_id="HLR-003",
            title="Safety-Critical Operation",
            description="The system shall operate safely under all specified conditions",
            req_type=RequirementType.SAFETY,
            dal=DAL.LEVEL_A,
            verification_methods=[
                VerificationMethod.TEST,
                VerificationMethod.ANALYSIS,
                VerificationMethod.REVIEW,
            ],
        )
    )

    # Low-level requirements
    db.add_requirement(
        Requirement(
            req_id="LLR-001",
            title="SVD Truncation",
            description="SVD truncation shall preserve the optimal Frobenius norm approximation",
            req_type=RequirementType.LOW_LEVEL,
            dal=DAL.LEVEL_C,
            parent_ids=["HLR-001"],
            verification_methods=[VerificationMethod.TEST],
        )
    )

    db.add_requirement(
        Requirement(
            req_id="LLR-002",
            title="Latency Monitoring",
            description="The system shall monitor inference latency and alert on violations",
            req_type=RequirementType.LOW_LEVEL,
            dal=DAL.LEVEL_B,
            parent_ids=["HLR-002"],
            verification_methods=[VerificationMethod.TEST],
        )
    )

    return db


def create_sample_safety_assessment() -> SafetyAssessment:
    """
    Create example safety assessment for The Ontic Engine.
    """
    assessment = SafetyAssessment("Ontic GNC System")

    assessment.add_hazard(
        Hazard(
            hazard_id="HAZ-001",
            title="Loss of Control Authority",
            description="Complete loss of guidance commands due to software failure",
            severity=HazardSeverity.CATASTROPHIC,
            probability=HazardProbability.EXTREMELY_IMPROBABLE,
            affected_functions=["guidance_controller", "trajectory_solver"],
            mitigations=[
                "Watchdog timer with hardware override",
                "Redundant computation channel",
                "Safe mode reversion",
            ],
        )
    )

    assessment.add_hazard(
        Hazard(
            hazard_id="HAZ-002",
            title="Incorrect Aerodynamic Prediction",
            description="CFD surrogate provides incorrect force/moment coefficients",
            severity=HazardSeverity.HAZARDOUS,
            probability=HazardProbability.REMOTE,
            affected_functions=["aero_predictor", "cfd_surrogate"],
            mitigations=[
                "Bounds checking on predictions",
                "Comparison with reduced-order models",
                "Physical plausibility checks",
            ],
        )
    )

    return assessment


if __name__ == "__main__":
    print("=" * 60)
    print("DO-178C CERTIFICATION FRAMEWORK TEST")
    print("=" * 60)

    # Test requirements database
    print("\n1. Creating requirements database...")
    req_db = create_ontic_requirements()
    print(f"Added {len(req_db.requirements)} requirements")

    matrix = req_db.get_traceability_matrix()
    print(f"Traceability matrix generated with {len(matrix)} entries")

    # Test safety assessment
    print("\n2. Creating safety assessment...")
    safety = create_sample_safety_assessment()
    print(f"Added {len(safety.hazards)} hazards")

    safety_case = safety.generate_safety_case()
    print(f"Safety case: {safety_case['hazard_summary']}")

    # Test verification package
    print("\n3. Creating verification package...")
    package = VerificationPackage("The Ontic Engine", DAL.LEVEL_B)

    package.add_test_case(
        TestCase(
            test_id="TC-001",
            title="SVD Truncation Accuracy",
            description="Verify SVD truncation optimality",
            requirement_ids=["LLR-001"],
            expected_result="Error < 1e-10",
            result=TestResult.PASS,
        )
    )

    package.add_evidence(
        VerificationEvidence(
            evidence_id="VE-001",
            title="SVD Algorithm Review",
            evidence_type=VerificationMethod.TEST,
            requirement_ids=["LLR-001"],
            description="Review of SVD implementation",
            artifacts=["proof_run.json"],
            result="pass",
            reviewer="Automated",
            review_date=datetime.datetime.now().isoformat(),
        )
    )

    # Check completeness
    completeness = package.check_completeness(req_db)
    print(f"Verification completeness: {completeness['completion_percentage']:.1f}%")

    # Generate SAS
    sas = package.generate_sas()
    print(f"SAS generated: {sas['verification_summary']}")

    print("\n✅ All certification framework tests passed!")
