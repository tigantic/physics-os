"""Plan DSL — typed, composable surgical plan representation.

A surgical plan is a directed acyclic graph (DAG) of typed operations.
Each operation:
  - Has named, typed, bounded parameters
  - Declares which mesh regions and BCs it affects
  - Is composable (sequence, parallel, branch)
  - Is deterministic (same plan + same twin = same BCs)

The DSL separates *what* the surgeon intends (domain language)
from *how* it's simulated (boundary conditions, loads, constraints).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union, cast

import numpy as np

from ..core.provenance import hash_dict
from ..core.types import ProcedureType, StructureType, Vec3


class ParamType(str, Enum):
    """Types for operator parameters."""
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    VEC3 = "vec3"
    ENUM = "enum"
    MESH_REGION = "mesh_region"


class OpCategory(str, Enum):
    """Categories of surgical operations."""
    RESECTION = "resection"           # Remove material
    OSTEOTOMY = "osteotomy"           # Cut bone
    GRAFT = "graft"                   # Add material
    SUTURE = "suture"                 # Mechanical connection
    REPOSITIONING = "repositioning"   # Move structure
    AUGMENTATION = "augmentation"     # Filler/implant
    RELEASE = "release"               # Divide tissue
    SCORING = "scoring"               # Weaken cartilage
    REDUCTION = "reduction"           # Reduce volume/projection


class PlanValidationError(Exception):
    """Raised when a surgical plan fails validation."""
    pass


# ── Parameter definition ──────────────────────────────────────────

@dataclass(frozen=True)
class OperatorParam:
    """Definition of a single operator parameter."""
    name: str
    param_type: ParamType
    unit: str = ""
    description: str = ""
    default: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    enum_values: Tuple[str, ...] = ()

    def validate(self, value: Any) -> Any:
        """Validate and coerce a parameter value."""
        if self.param_type == ParamType.FLOAT:
            val = float(value)
            if self.min_value is not None and val < self.min_value:
                raise PlanValidationError(
                    f"{self.name}: {val} < min {self.min_value}"
                )
            if self.max_value is not None and val > self.max_value:
                raise PlanValidationError(
                    f"{self.name}: {val} > max {self.max_value}"
                )
            return val
        elif self.param_type == ParamType.INT:
            val = int(value)
            if self.min_value is not None and val < int(self.min_value):
                raise PlanValidationError(
                    f"{self.name}: {val} < min {int(self.min_value)}"
                )
            if self.max_value is not None and val > int(self.max_value):
                raise PlanValidationError(
                    f"{self.name}: {val} > max {int(self.max_value)}"
                )
            return val
        elif self.param_type == ParamType.BOOL:
            return bool(value)
        elif self.param_type == ParamType.VEC3:
            if isinstance(value, Vec3):
                return value
            if isinstance(value, (list, tuple)) and len(value) == 3:
                return Vec3(float(value[0]), float(value[1]), float(value[2]))
            raise PlanValidationError(f"{self.name}: expected Vec3, got {type(value)}")
        elif self.param_type == ParamType.ENUM:
            str_val: str = str(value)
            if self.enum_values and str_val not in self.enum_values:
                raise PlanValidationError(
                    f"{self.name}: {str_val!r} not in {self.enum_values}"
                )
            return str_val
        elif self.param_type == ParamType.MESH_REGION:
            return str(value)
        return value


# ── Surgical operator (leaf node) ─────────────────────────────────

@dataclass
class SurgicalOp:
    """A single surgical operation (leaf node in the plan DAG).

    This is an *abstract intent* — it says what the surgeon wants
    to do, not how to simulate it.  The PlanCompiler translates
    this into boundary conditions, loads, and mesh modifications.
    """
    name: str
    category: OpCategory
    procedure: ProcedureType
    params: Dict[str, Any] = field(default_factory=dict)
    param_defs: Dict[str, OperatorParam] = field(default_factory=dict)
    affected_structures: List[StructureType] = field(default_factory=list)
    description: str = ""
    order: int = 0  # execution order within a sequence

    def validate(self) -> List[str]:
        """Validate all parameters against their definitions."""
        errors = []
        for pname, pdef in self.param_defs.items():
            if pname not in self.params:
                if pdef.default is not None:
                    self.params[pname] = pdef.default
                else:
                    errors.append(f"Missing required parameter: {pname}")
                    continue
            try:
                self.params[pname] = pdef.validate(self.params[pname])
            except PlanValidationError as e:
                errors.append(str(e))
        return errors

    def content_hash(self) -> str:
        """Deterministic hash of the operation and its parameters."""
        data = {
            "name": self.name,
            "category": self.category.value,
            "params": {
                k: (
                    [v.x, v.y, v.z] if isinstance(v, Vec3) else v
                )
                for k, v in sorted(self.params.items())
            },
        }
        return hash_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "type": "op",
            "name": self.name,
            "category": self.category.value,
            "procedure": self.procedure.value,
            "params": {
                k: ([v.x, v.y, v.z] if isinstance(v, Vec3) else v)
                for k, v in self.params.items()
            },
            "affected_structures": [s.value for s in self.affected_structures],
            "description": self.description,
            "order": self.order,
        }


# ── Composite nodes ───────────────────────────────────────────────

@dataclass
class SequenceNode:
    """Ordered sequence of operations (executed in order)."""
    name: str
    steps: List[Union[SurgicalOp, "SequenceNode", "BranchNode", "CompositeOp"]] = field(
        default_factory=list
    )
    description: str = ""

    def add(self, step: Union[SurgicalOp, "SequenceNode", "BranchNode", "CompositeOp"]) -> None:
        step_order = len(self.steps)
        if isinstance(step, SurgicalOp):
            step.order = step_order
        self.steps.append(step)

    def validate(self) -> List[str]:
        errors = []
        for step in self.steps:
            errors.extend(step.validate())
        return errors

    def content_hash(self) -> str:
        step_hashes = [s.content_hash() for s in self.steps]
        return hash_dict({"type": "sequence", "name": self.name, "steps": step_hashes})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "sequence",
            "name": self.name,
            "steps": [s.to_dict() for s in self.steps],
            "description": self.description,
        }


@dataclass
class BranchNode:
    """Conditional branch — simulate multiple variants."""
    name: str
    condition_param: str  # parameter name that determines the branch
    branches: Dict[str, Union[SurgicalOp, SequenceNode, "CompositeOp"]] = field(
        default_factory=dict
    )
    description: str = ""

    def validate(self) -> List[str]:
        errors = []
        if not self.branches:
            errors.append(f"Branch '{self.name}' has no branches")
        for label, branch in self.branches.items():
            errors.extend(branch.validate())
        return errors

    def content_hash(self) -> str:
        branch_hashes = {k: v.content_hash() for k, v in sorted(self.branches.items())}
        return hash_dict({"type": "branch", "name": self.name, "branches": branch_hashes})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "branch",
            "name": self.name,
            "condition_param": self.condition_param,
            "branches": {k: v.to_dict() for k, v in self.branches.items()},
            "description": self.description,
        }


@dataclass
class CompositeOp:
    """Named composite of multiple operations (reusable macro)."""
    name: str
    procedure: ProcedureType
    sequence: SequenceNode
    description: str = ""

    def validate(self) -> List[str]:
        return self.sequence.validate()

    def content_hash(self) -> str:
        return hash_dict({
            "type": "composite",
            "name": self.name,
            "sequence": self.sequence.content_hash(),
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "composite",
            "name": self.name,
            "procedure": self.procedure.value,
            "sequence": self.sequence.to_dict(),
            "description": self.description,
        }


# ── Surgical Plan ─────────────────────────────────────────────────

PlanNode = Union[SurgicalOp, SequenceNode, BranchNode, CompositeOp]


class SurgicalPlan:
    """Complete surgical plan — the top-level DAG.

    A plan is a named, versioned, hashable description of
    a surgical procedure.  It composes typed operations into
    an execution DAG that the PlanCompiler translates into
    boundary conditions for the physics engine.
    """

    def __init__(
        self,
        name: str,
        procedure: ProcedureType,
        *,
        description: str = "",
        version: str = "1.0",
    ) -> None:
        self._name = name
        self._procedure = procedure
        self._description = description
        self._version = version
        self._root = SequenceNode(name="root")

    @property
    def name(self) -> str:
        return self._name

    @property
    def procedure(self) -> ProcedureType:
        return self._procedure

    @property
    def root(self) -> SequenceNode:
        return self._root

    def add_step(self, step: PlanNode) -> None:
        """Add a step to the plan's root sequence."""
        self._root.add(step)

    def validate(self) -> List[str]:
        """Validate the entire plan. Returns list of validation errors."""
        errors = self._root.validate()

        # Check for structure conflicts
        affected = self._collect_affected_structures()
        # (More advanced conflict detection would go here)

        return errors

    def content_hash(self) -> str:
        """Deterministic content hash of the entire plan."""
        return hash_dict({
            "name": self._name,
            "procedure": self._procedure.value,
            "version": self._version,
            "root": self._root.content_hash(),
        })

    def to_dict(self) -> Dict[str, Any]:
        """Serialize plan to dict."""
        return {
            "name": self._name,
            "procedure": self._procedure.value,
            "description": self._description,
            "version": self._version,
            "plan_hash": self.content_hash(),
            "root": self._root.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SurgicalPlan:
        """Deserialize plan from dict."""
        plan = cls(
            name=data["name"],
            procedure=ProcedureType(data["procedure"]),
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
        )
        root_data = data["root"]
        plan._root = cast(SequenceNode, cls._deserialize_node(root_data))
        return plan

    @classmethod
    def _deserialize_node(cls, data: Dict[str, Any]) -> PlanNode:
        """Recursively deserialize a plan node."""
        node_type = data["type"]
        if node_type == "op":
            return SurgicalOp(
                name=data["name"],
                category=OpCategory(data["category"]),
                procedure=ProcedureType(data["procedure"]),
                params=data.get("params", {}),
                affected_structures=[
                    StructureType(s) for s in data.get("affected_structures", [])
                ],
                description=data.get("description", ""),
                order=data.get("order", 0),
            )
        elif node_type == "sequence":
            seq = SequenceNode(
                name=data["name"],
                description=data.get("description", ""),
            )
            for step_data in data.get("steps", []):
                seq.steps.append(cls._deserialize_node(step_data))
            return seq
        elif node_type == "branch":
            branch = BranchNode(
                name=data["name"],
                condition_param=data.get("condition_param", ""),
                description=data.get("description", ""),
            )
            for label, branch_data in data.get("branches", {}).items():
                branch.branches[label] = cast(
                    Union[SurgicalOp, SequenceNode, CompositeOp],
                    cls._deserialize_node(branch_data),
                )
            return branch
        elif node_type == "composite":
            seq_data = data.get("sequence", {"type": "sequence", "name": "empty", "steps": []})
            return CompositeOp(
                name=data["name"],
                procedure=ProcedureType(data["procedure"]),
                sequence=cast(SequenceNode, cls._deserialize_node(seq_data)),
                description=data.get("description", ""),
            )
        else:
            raise ValueError(f"Unknown node type: {node_type}")

    def _collect_affected_structures(self) -> Set[StructureType]:
        """Collect all structures affected by the plan."""
        structures: Set[StructureType] = set()
        self._walk(self._root, lambda op: structures.update(op.affected_structures))
        return structures

    def _walk(self, node: PlanNode, fn: Callable[[SurgicalOp], None]) -> None:
        """Walk the plan DAG, calling fn on each SurgicalOp."""
        if isinstance(node, SurgicalOp):
            fn(node)
        elif isinstance(node, SequenceNode):
            for step in node.steps:
                self._walk(step, fn)
        elif isinstance(node, BranchNode):
            for branch in node.branches.values():
                self._walk(branch, fn)
        elif isinstance(node, CompositeOp):
            self._walk(node.sequence, fn)

    def all_ops(self) -> List[SurgicalOp]:
        """Return all leaf SurgicalOp nodes in execution order."""
        ops: List[SurgicalOp] = []
        self._walk(self._root, ops.append)
        return ops

    def __repr__(self) -> str:
        n_ops = len(self.all_ops())
        return f"SurgicalPlan({self._name!r}, {self._procedure.value}, {n_ops} ops)"
