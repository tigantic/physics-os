# Module `certification.do178c`

DO-178C Certification Framework for Safety-Critical Systems ============================================================

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

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `ConfigurationItem`

Configuration item for software baseline management.

#### Attributes

- **ci_id** (`<class 'str'>`): 
- **name** (`<class 'str'>`): 
- **version** (`<class 'str'>`): 
- **file_path** (`<class 'str'>`): 
- **checksum** (`<class 'str'>`): 
- **status** (`<class 'str'>`): 
- **baseline** (`<class 'str'>`): 
- **change_history** (`typing.List[typing.Dict]`): 

#### Methods

##### `__init__`

```python
def __init__(self, ci_id: str, name: str, version: str, file_path: str, checksum: str, status: str = 'controlled', baseline: str = '', change_history: List[Dict] = <factory>) -> None
```

##### `from_file`

```python
def from_file(file_path: str, ci_id: str, name: str) -> 'ConfigurationItem'
```

Create CI from file.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:630](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L630)*

##### `verify_integrity`

```python
def verify_integrity(self) -> bool
```

Verify file integrity against stored checksum.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:649](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L649)*

### class `ConfigurationManagement`

Configuration management system for DO-178C compliance.

#### Methods

##### `__init__`

```python
def __init__(self, project_name: str)
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:664](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L664)*

##### `add_item`

```python
def add_item(self, item: do178c.ConfigurationItem)
```

Add configuration item.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:669](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L669)*

##### `create_baseline`

```python
def create_baseline(self, baseline_name: str, ci_ids: List[str])
```

Create a configuration baseline.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:673](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L673)*

##### `record_change`

```python
def record_change(self, ci_id: str, description: str, author: str)
```

Record a change to a configuration item.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:699](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L699)*

##### `verify_baseline`

```python
def verify_baseline(self, baseline_name: str) -> Dict[str, bool]
```

Verify integrity of all items in a baseline.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:687](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L687)*

### class `CoverageAnalyzer`

Analyzes test coverage for DO-178C compliance.

#### Methods

##### `__init__`

```python
def __init__(self, source_files: List[str], dal: do178c.DAL)
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:363](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L363)*

##### `analyze_decision_coverage`

```python
def analyze_decision_coverage(self, decisions_evaluated: Dict[str, Set[bool]]) -> do178c.CoverageReport
```

Analyze decision coverage.

Each decision must evaluate to both True and False.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:397](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L397)*

##### `analyze_mcdc_coverage`

```python
def analyze_mcdc_coverage(self, conditions: Dict[str, List[Dict[str, bool]]]) -> do178c.CoverageReport
```

Analyze Modified Condition/Decision Coverage.

Each condition must independently affect the decision outcome.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:424](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L424)*

##### `analyze_statement_coverage`

```python
def analyze_statement_coverage(self, executed_lines: Set[Tuple[str, int]], all_lines: Set[Tuple[str, int]]) -> do178c.CoverageReport
```

Analyze statement coverage.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:379](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L379)*

### class `CoverageReport`

Test coverage analysis report.

DO-178C requires different coverage levels:
- DAL A: MC/DC (Modified Condition/Decision Coverage)
- DAL B: Decision Coverage
- DAL C: Statement Coverage

#### Attributes

- **coverage_type** (`<enum 'CoverageType'>`): 
- **total_items** (`<class 'int'>`): 
- **covered_items** (`<class 'int'>`): 
- **uncovered_items** (`typing.List[str]`): 
- **coverage_percentage** (`<class 'float'>`): 
- **analysis_date** (`<class 'str'>`): 
- **tool_name** (`<class 'str'>`): 

#### Properties

##### `meets_objective`

```python
def meets_objective(self) -> bool
```

Check if coverage meets DO-178C objectives.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:339](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L339)*

#### Methods

##### `__init__`

```python
def __init__(self, coverage_type: do178c.CoverageType, total_items: int, covered_items: int, uncovered_items: List[str], coverage_percentage: float, analysis_date: str = <factory>, tool_name: str = 'Physics OS Coverage Analyzer') -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:345](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L345)*

### class `CoverageType`(Enum)

Test coverage metrics.

### class `DAL`(Enum)

Design Assurance Levels per DO-178C.

### class `Hazard`

Safety hazard identification and analysis.

#### Attributes

- **hazard_id** (`<class 'str'>`): 
- **title** (`<class 'str'>`): 
- **description** (`<class 'str'>`): 
- **severity** (`<enum 'HazardSeverity'>`): 
- **probability** (`<enum 'HazardProbability'>`): 
- **affected_functions** (`typing.List[str]`): 
- **mitigations** (`typing.List[str]`): 
- **residual_risk** (`<class 'str'>`): 
- **verification_evidence** (`typing.List[str]`): 

#### Properties

##### `required_dal`

```python
def required_dal(self) -> do178c.DAL
```

Determine required DAL based on severity.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:528](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L528)*

##### `risk_level`

```python
def risk_level(self) -> int
```

Compute risk level from severity × probability.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:523](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L523)*

#### Methods

##### `__init__`

```python
def __init__(self, hazard_id: str, title: str, description: str, severity: do178c.HazardSeverity, probability: do178c.HazardProbability, affected_functions: List[str], mitigations: List[str] = <factory>, residual_risk: str = '', verification_evidence: List[str] = <factory>) -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:540](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L540)*

### class `HazardProbability`(Enum)

Hazard probability classification.

### class `HazardSeverity`(Enum)

Hazard severity classification.

### class `Requirement`

Software requirement with traceability.

#### Attributes

- **req_id** (`<class 'str'>`): 
- **title** (`<class 'str'>`): 
- **description** (`<class 'str'>`): 
- **req_type** (`<enum 'RequirementType'>`): 
- **dal** (`<enum 'DAL'>`): 
- **parent_ids** (`typing.List[str]`): 
- **child_ids** (`typing.List[str]`): 
- **verification_methods** (`typing.List[do178c.VerificationMethod]`): 
- **status** (`<enum 'RequirementStatus'>`): 
- **rationale** (`<class 'str'>`): 
- **version** (`<class 'str'>`): 
- **last_modified** (`<class 'str'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, req_id: str, title: str, description: str, req_type: do178c.RequirementType, dal: do178c.DAL, parent_ids: List[str] = <factory>, child_ids: List[str] = <factory>, verification_methods: List[do178c.VerificationMethod] = <factory>, status: do178c.RequirementStatus = <RequirementStatus.DRAFT: 'draft'>, rationale: str = '', version: str = '1.0', last_modified: str = <factory>) -> None
```

##### `from_dict`

```python
def from_dict(data: Dict) -> 'Requirement'
```

Create from dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:141](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L141)*

##### `to_dict`

```python
def to_dict(self) -> Dict
```

Convert to dictionary for serialization.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:124](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L124)*

### class `RequirementStatus`(Enum)

Requirement lifecycle status.

### class `RequirementType`(Enum)

Types of requirements.

### class `RequirementsDatabase`

Requirements database with traceability matrix.

Manages the complete set of requirements and their relationships.

#### Methods

##### `__init__`

```python
def __init__(self, project_name: str = 'The Physics OS')
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:167](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L167)*

##### `add_requirement`

```python
def add_requirement(self, req: do178c.Requirement)
```

Add a requirement to the database.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:172](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L172)*

##### `delete_requirement`

```python
def delete_requirement(self, req_id: str)
```

Mark requirement as deleted.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:196](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L196)*

##### `export_to_json`

```python
def export_to_json(self, filepath: str)
```

Export database to JSON.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:230](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L230)*

##### `get_requirement`

```python
def get_requirement(self, req_id: str) -> Optional[do178c.Requirement]
```

Get requirement by ID.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:185](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L185)*

##### `get_requirements_by_dal`

```python
def get_requirements_by_dal(self, dal: do178c.DAL) -> List[do178c.Requirement]
```

Get all requirements at a specific DAL.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:219](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L219)*

##### `get_traceability_matrix`

```python
def get_traceability_matrix(self) -> Dict[str, List[str]]
```

Generate requirements traceability matrix.

Maps HLR -> LLR -> Test Cases

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:202](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L202)*

##### `get_unverified_requirements`

```python
def get_unverified_requirements(self) -> List[do178c.Requirement]
```

Get requirements that haven't been verified.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:223](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L223)*

##### `load_from_json`

```python
def load_from_json(filepath: str) -> 'RequirementsDatabase'
```

Load database from JSON.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:243](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L243)*

##### `update_requirement`

```python
def update_requirement(self, req: do178c.Requirement)
```

Update an existing requirement.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:189](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L189)*

### class `SafetyAssessment`

System Safety Assessment per ARP4761/ARP4754A.

#### Methods

##### `__init__`

```python
def __init__(self, system_name: str)
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:561](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L561)*

##### `add_hazard`

```python
def add_hazard(self, hazard: do178c.Hazard)
```

Add hazard to assessment.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:566](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L566)*

##### `compute_risk_matrix`

```python
def compute_risk_matrix(self) -> Dict[str, List[str]]
```

Generate risk matrix.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:578](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L578)*

##### `generate_safety_case`

```python
def generate_safety_case(self) -> Dict
```

Generate safety case document.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:594](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L594)*

##### `get_hazards_by_severity`

```python
def get_hazards_by_severity(self, severity: do178c.HazardSeverity) -> List[do178c.Hazard]
```

Get all hazards of a specific severity.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:570](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L570)*

##### `get_unmitigated_hazards`

```python
def get_unmitigated_hazards(self) -> List[do178c.Hazard]
```

Get hazards without mitigations.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:574](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L574)*

### class `TestCase`

Test case with traceability to requirements.

#### Attributes

- **test_id** (`<class 'str'>`): 
- **title** (`<class 'str'>`): 
- **description** (`<class 'str'>`): 
- **requirement_ids** (`typing.List[str]`): 
- **expected_result** (`<class 'str'>`): 
- **preconditions** (`<class 'str'>`): 
- **test_steps** (`typing.List[str]`): 
- **actual_result** (`<class 'str'>`): 
- **result** (`<enum 'TestResult'>`): 
- **test_data** (`typing.Dict[str, typing.Any]`): 
- **environment** (`typing.Dict[str, str]`): 
- **execution_time** (`typing.Optional[float]`): 
- **execution_date** (`typing.Optional[str]`): 
- **executed_by** (`<class 'str'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, test_id: str, title: str, description: str, requirement_ids: List[str], expected_result: str, preconditions: str = '', test_steps: List[str] = <factory>, actual_result: str = '', result: do178c.TestResult = <TestResult.SKIPPED: 'skipped'>, test_data: Dict[str, Any] = <factory>, environment: Dict[str, str] = <factory>, execution_time: Optional[float] = None, execution_date: Optional[str] = None, executed_by: str = 'automated') -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:301](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L301)*

### class `TestResult`(Enum)

Test execution result.

### class `VerificationEvidence`

Verification evidence for DO-178C compliance.

#### Attributes

- **evidence_id** (`<class 'str'>`): 
- **title** (`<class 'str'>`): 
- **evidence_type** (`<enum 'VerificationMethod'>`): 
- **requirement_ids** (`typing.List[str]`): 
- **description** (`<class 'str'>`): 
- **artifacts** (`typing.List[str]`): 
- **result** (`<class 'str'>`): 
- **reviewer** (`<class 'str'>`): 
- **review_date** (`<class 'str'>`): 
- **comments** (`<class 'str'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, evidence_id: str, title: str, evidence_type: do178c.VerificationMethod, requirement_ids: List[str], description: str, artifacts: List[str], result: str, reviewer: str, review_date: str, comments: str = '') -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:744](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L744)*

### class `VerificationMethod`(Enum)

Verification methods per DO-178C.

### class `VerificationPackage`

Complete verification evidence package for certification.

#### Methods

##### `__init__`

```python
def __init__(self, project_name: str, dal: do178c.DAL)
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:764](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L764)*

##### `add_coverage_report`

```python
def add_coverage_report(self, report: do178c.CoverageReport)
```

Add coverage report.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:779](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L779)*

##### `add_evidence`

```python
def add_evidence(self, evidence: do178c.VerificationEvidence)
```

Add verification evidence.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:771](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L771)*

##### `add_test_case`

```python
def add_test_case(self, test: do178c.TestCase)
```

Add test case.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:775](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L775)*

##### `check_completeness`

```python
def check_completeness(self, requirements: do178c.RequirementsDatabase) -> Dict
```

Check if verification is complete for all requirements.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:809](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L809)*

##### `generate_sas`

```python
def generate_sas(self) -> Dict
```

Generate Software Accomplishment Summary (SAS).

The SAS is a key certification document summarizing all
verification activities and their results.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:854](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L854)*

##### `get_verification_matrix`

```python
def get_verification_matrix(self) -> Dict[str, Dict]
```

Generate verification cross-reference matrix.

Maps requirements to verification evidence.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:783](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L783)*

## Functions

### `create_ontic_requirements`

```python
def create_ontic_requirements() -> do178c.RequirementsDatabase
```

Create example requirements for The Physics OS system.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:880](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L880)*

### `create_sample_safety_assessment`

```python
def create_sample_safety_assessment() -> do178c.SafetyAssessment
```

Create example safety assessment for The Physics OS.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py:938](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\certification\do178c.py#L938)*
