"""
Circom Circuit Parser and Under-Constrained Signal Hunter.

ORACLE's ZK-Circuit vulnerability detection module.

Key Vulnerability Types:
1. Under-constrained signals - Can be set to any value
2. Non-deterministic constraints - Multiple valid witness values  
3. Missing range checks - Allows overflow/wraparound
4. Nullifier collisions - Identity can be reused
5. Merkle proof forgery - Invalid membership proofs accepted

Approach:
1. Parse Circom AST (template, signal, constraint)
2. Build constraint graph (which signals constrain which)
3. Identify signals with insufficient constraints
4. Flag signals that can be manipulated without detection
"""

from __future__ import annotations

import re
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class SignalType(Enum):
    INPUT = "input"
    OUTPUT = "output"
    INTERMEDIATE = "intermediate"


class ConstraintType(Enum):
    QUADRATIC = "quadratic"     # a * b === c
    LINEAR = "linear"           # a === b + c
    EQUALITY = "equality"       # a === b
    RANGE = "range"             # LessThan, GreaterThan, etc.
    HASH = "hash"               # Poseidon, MiMC, etc.
    MERKLE = "merkle"           # Merkle proof verification


@dataclass
class Signal:
    """A signal (wire) in a Circom circuit."""
    name: str
    signal_type: SignalType
    dimensions: list[int] = field(default_factory=list)  # For arrays
    line_number: int = 0
    is_public: bool = False
    
    # Constraint analysis
    constrained_by: list[str] = field(default_factory=list)  # Constraints using this signal
    constrains: list[str] = field(default_factory=list)      # Signals this constrains
    constraint_count: int = 0


@dataclass
class Constraint:
    """A constraint in a Circom circuit."""
    expr: str                       # The constraint expression
    constraint_type: ConstraintType
    signals_involved: list[str] = field(default_factory=list)
    line_number: int = 0
    is_quadratic: bool = False


@dataclass
class Template:
    """A Circom template (circuit component)."""
    name: str
    signals: dict[str, Signal] = field(default_factory=dict)
    constraints: list[Constraint] = field(default_factory=list)
    sub_components: list[str] = field(default_factory=list)  # Instantiated templates
    line_number: int = 0
    source: str = ""


@dataclass
class Finding:
    """A vulnerability finding."""
    severity: str           # CRITICAL, HIGH, MEDIUM, LOW, INFO
    vuln_type: str          # under-constrained, non-deterministic, etc.
    signal_name: str
    template_name: str
    description: str
    impact: str
    line_number: int = 0
    recommendation: str = ""
    
    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "type": self.vuln_type,
            "signal": self.signal_name,
            "template": self.template_name,
            "description": self.description,
            "impact": self.impact,
            "line": self.line_number,
            "recommendation": self.recommendation,
        }


class CircomParser:
    """
    Parse Circom source code and build constraint graph.
    
    Uses regex-based parsing (Circom doesn't have good tree-sitter support).
    """
    
    def __init__(self):
        self.templates: dict[str, Template] = {}
        self.includes: list[str] = []
        
    def parse(self, source: str, file_path: Optional[str] = None) -> list[Template]:
        """
        Parse Circom source into Template structures.
        
        Args:
            source: Circom source code
            file_path: Optional file path for reference
            
        Returns:
            List of Template objects found in the source
        """
        templates = []
        
        # Extract includes
        include_pattern = r'include\s+"([^"]+)"'
        self.includes = re.findall(include_pattern, source)
        
        # Extract templates using brace counting for proper nesting
        template_start_pattern = r'template\s+(\w+)\s*\([^)]*\)\s*\{'
        
        for match in re.finditer(template_start_pattern, source):
            name = match.group(1)
            start_pos = match.end() - 1  # Position of opening brace
            
            # Count braces to find matching close
            brace_count = 1
            pos = start_pos + 1
            while pos < len(source) and brace_count > 0:
                if source[pos] == '{':
                    brace_count += 1
                elif source[pos] == '}':
                    brace_count -= 1
                pos += 1
            
            body = source[start_pos + 1:pos - 1]  # Content between braces
            
            line_number = source[:match.start()].count('\n') + 1
            
            template = Template(
                name=name,
                line_number=line_number,
                source=source[match.start():pos]
            )
            
            # Extract signals
            self._extract_signals(body, template)
            
            # Extract constraints
            self._extract_constraints(body, template)
            
            # Extract component instantiations
            self._extract_components(body, template)
            
            templates.append(template)
            self.templates[name] = template
        
        return templates
    
    def _extract_signals(self, body: str, template: Template):
        """Extract signal declarations from template body."""
        # Input signals: signal input name;
        input_pattern = r'signal\s+input\s+(\w+)(\[[^\]]+\])?\s*;'
        for match in re.finditer(input_pattern, body):
            name = match.group(1)
            dims = self._parse_dimensions(match.group(2)) if match.group(2) else []
            template.signals[name] = Signal(
                name=name,
                signal_type=SignalType.INPUT,
                dimensions=dims,
                line_number=template.line_number + body[:match.start()].count('\n'),
            )
        
        # Output signals: signal output name;
        output_pattern = r'signal\s+output\s+(\w+)(\[[^\]]+\])?\s*;'
        for match in re.finditer(output_pattern, body):
            name = match.group(1)
            dims = self._parse_dimensions(match.group(2)) if match.group(2) else []
            template.signals[name] = Signal(
                name=name,
                signal_type=SignalType.OUTPUT,
                dimensions=dims,
                is_public=True,
                line_number=template.line_number + body[:match.start()].count('\n'),
            )
        
        # Intermediate signals: signal name;
        intermediate_pattern = r'signal\s+(?!input|output)(\w+)(\[[^\]]+\])?\s*;'
        for match in re.finditer(intermediate_pattern, body):
            name = match.group(1)
            dims = self._parse_dimensions(match.group(2)) if match.group(2) else []
            template.signals[name] = Signal(
                name=name,
                signal_type=SignalType.INTERMEDIATE,
                dimensions=dims,
                line_number=template.line_number + body[:match.start()].count('\n'),
            )
    
    def _parse_dimensions(self, dim_str: str) -> list[int]:
        """Parse array dimensions from [N] or [N][M] etc."""
        if not dim_str:
            return []
        dims = re.findall(r'\[([^\]]+)\]', dim_str)
        result = []
        for d in dims:
            try:
                result.append(int(d))
            except ValueError:
                result.append(-1)  # Variable dimension
        return result
    
    def _extract_constraints(self, body: str, template: Template):
        """Extract constraint expressions from template body."""
        # Track all signal usages including array indexing
        # Pattern: signal[index] <== expr  OR  signal <== expr  OR expr ==> signal
        # Circom supports both <== and ==> (reversed direction)
        constraint_pattern = r'([^;]+)\s*(===|<==|==>)\s*([^;]+);'
        
        for match in re.finditer(constraint_pattern, body):
            lhs = match.group(1).strip()
            op = match.group(2)
            rhs = match.group(3).strip()
            
            # Skip pure component instantiations like: component c = Template()
            # But keep template function calls like: x <== Poseidon(2)([a, b])
            if 'component' in lhs:
                continue
            
            expr = f"{lhs} {op} {rhs}"
            
            # Identify constraint type
            if '*' in rhs or '*' in lhs:
                ctype = ConstraintType.QUADRATIC
            elif '+' in rhs or '-' in rhs or '+' in lhs or '-' in lhs:
                ctype = ConstraintType.LINEAR
            else:
                ctype = ConstraintType.EQUALITY
            
            # Find signals involved - now also look for array indexing
            signals = self._extract_signal_names(expr, template)
            
            constraint = Constraint(
                expr=expr,
                constraint_type=ctype,
                signals_involved=signals,
                line_number=template.line_number + body[:match.start()].count('\n'),
                is_quadratic=(ctype == ConstraintType.QUADRATIC),
            )
            
            template.constraints.append(constraint)
            
            # Update signal constraint counts
            for sig_name in signals:
                if sig_name in template.signals:
                    template.signals[sig_name].constraint_count += 1
    
    def _extract_signal_names(self, expr: str, template: Template) -> list[str]:
        """Extract signal names from an expression, including array-indexed signals."""
        # Find all identifiers (including those followed by array brackets)
        # Pattern matches: signal, signal[i], signal[i][j], etc.
        identifiers = re.findall(r'\b([a-zA-Z_]\w*)\s*(?:\[[^\]]*\])*', expr)
        
        # Filter to known signals
        signals = []
        for ident in identifiers:
            if ident in template.signals:
                signals.append(ident)
        
        return list(set(signals))
    
    def _extract_components(self, body: str, template: Template):
        """Extract component instantiations."""
        # component name = TemplateName(...);
        component_pattern = r'component\s+\w+\s*=\s*(\w+)\s*\('
        for match in re.finditer(component_pattern, body):
            template_name = match.group(1)
            if template_name not in template.sub_components:
                template.sub_components.append(template_name)


class UnderConstrainedHunter:
    """
    Hunt for under-constrained signals in Circom circuits.
    
    An under-constrained signal can take on multiple valid values
    without violating any constraints - this allows witness forgery.
    """
    
    def __init__(self, parser: CircomParser):
        self.parser = parser
        self.findings: list[Finding] = []
    
    def hunt(self, templates: list[Template]) -> list[Finding]:
        """
        Hunt for under-constrained signals across all templates.
        
        Returns:
            List of Finding objects
        """
        self.findings = []
        
        for template in templates:
            self._analyze_template(template)
        
        return self.findings
    
    def _analyze_template(self, template: Template):
        """Analyze a single template for under-constrained signals."""
        
        # Check 1: Signals with zero constraints
        for sig_name, signal in template.signals.items():
            if signal.constraint_count == 0:
                # Check if it's used in a component call (may be constrained there)
                if not self._is_used_in_component(sig_name, template):
                    self.findings.append(Finding(
                        severity="CRITICAL",
                        vuln_type="under-constrained",
                        signal_name=sig_name,
                        template_name=template.name,
                        description=f"Signal '{sig_name}' has ZERO constraints. It can be set to any value.",
                        impact="Attacker can forge proofs by setting this signal arbitrarily.",
                        line_number=signal.line_number,
                        recommendation=f"Add a constraint that binds '{sig_name}' to other signals or constants.",
                    ))
        
        # Check 2: Output signals only constrained by inputs (no intermediate binding)
        for sig_name, signal in template.signals.items():
            if signal.signal_type == SignalType.OUTPUT:
                if signal.constraint_count == 1:
                    # Single constraint - might be directly assignable
                    constraint = self._find_constraint_for_signal(sig_name, template)
                    if constraint and self._is_trivially_assignable(constraint, sig_name):
                        self.findings.append(Finding(
                            severity="HIGH",
                            vuln_type="weak-constraint",
                            signal_name=sig_name,
                            template_name=template.name,
                            description=f"Output signal '{sig_name}' has only one trivial constraint.",
                            impact="May allow partial witness manipulation.",
                            line_number=signal.line_number,
                            recommendation="Add additional constraints or range checks.",
                        ))
        
        # Check 3: Missing range checks on inputs that should be bounded
        for sig_name, signal in template.signals.items():
            if signal.signal_type == SignalType.INPUT:
                if not self._has_range_check(sig_name, template):
                    # Check if the signal name suggests it should be bounded
                    if any(kw in sig_name.lower() for kw in ['index', 'length', 'depth', 'bit', 'byte']):
                        self.findings.append(Finding(
                            severity="MEDIUM",
                            vuln_type="missing-range-check",
                            signal_name=sig_name,
                            template_name=template.name,
                            description=f"Input signal '{sig_name}' lacks range check but name suggests bounded value.",
                            impact="Could allow out-of-bounds or overflow attacks.",
                            line_number=signal.line_number,
                            recommendation="Add LessThan or range check constraint.",
                        ))
        
        # Check 4: Non-deterministic assignment patterns
        self._check_non_determinism(template)
        
        # Check 5: Nullifier patterns (Semaphore-specific)
        self._check_nullifier_soundness(template)
    
    def _is_used_in_component(self, sig_name: str, template: Template) -> bool:
        """Check if signal is passed to a sub-component or used in var assignment (Circom 1.x and 2.x)."""
        pattern = rf'\b{re.escape(sig_name)}\b'
        
        # Check 1: Used in constraint expressions (Circom 1.x style)
        for constraint in template.constraints:
            if re.search(pattern, constraint.expr):
                return True
        
        # Check 2: Used in Circom 2.x inline component calls
        # Patterns: Component()(signal), Component()([signal, ...]), etc.
        # Search the entire template source
        source = template.source
        
        # Pattern for Circom 2.x: var ... = ComponentName(...)(...)
        # where signal appears in the second parentheses
        component_call_pattern = rf'\w+\s*\([^)]*\)\s*\([^)]*\b{re.escape(sig_name)}\b[^)]*\)'
        if re.search(component_call_pattern, source):
            return True
        
        # Pattern for array usage in component call: [signal, ...]
        array_pattern = rf'\[\s*[^\]]*\b{re.escape(sig_name)}\b[^\]]*\]'
        # Check if array pattern is followed by or preceded by component call
        if re.search(array_pattern, source):
            # Simple heuristic: if signal appears in [...] in the source, likely used
            return True
        
        # Check 3: Used in var assignment that flows into constraints
        # Pattern: var x = ... signal ... ; where x is later used in constraints
        var_assignment_pattern = rf'var\s+\w+[^;]*\b{re.escape(sig_name)}\b[^;]*;'
        if re.search(var_assignment_pattern, source):
            return True
        
        # Check 4: Used in ternary expression or arithmetic that flows into signal
        # Pattern: signal <== ... ? ... signal ... : ...
        ternary_pattern = rf'<==\s*[^;]*\?[^;]*\b{re.escape(sig_name)}\b[^;]*;'
        if re.search(ternary_pattern, source):
            return True
        
        # Check 5: Assigned to a var (separate declaration/assignment)
        # Pattern: varName = signal;
        var_reassign_pattern = rf'\w+\s*=\s*\b{re.escape(sig_name)}\b\s*;'
        if re.search(var_reassign_pattern, source):
            return True
        
        return False
    
    def _find_constraint_for_signal(self, sig_name: str, template: Template) -> Optional[Constraint]:
        """Find the constraint that assigns to a signal."""
        for constraint in template.constraints:
            if sig_name in constraint.signals_involved:
                return constraint
        return None
    
    def _is_trivially_assignable(self, constraint: Constraint, sig_name: str) -> bool:
        """Check if constraint is a trivial assignment like x <== y (not from component)."""
        # Component outputs (like hasher.outs[0]) are computed, not trivially assignable
        if '.' in constraint.expr:
            # Contains component access - this is a computed value
            return False
        
        if constraint.constraint_type == ConstraintType.EQUALITY:
            return True
        if constraint.constraint_type == ConstraintType.LINEAR:
            # Check if it's just x <== y + constant
            expr = constraint.expr
            if '+' in expr or '-' in expr:
                parts = re.split(r'[+-]', expr)
                non_constant = [p.strip() for p in parts if not p.strip().isdigit()]
                if len(non_constant) <= 2:
                    return True
        return False
    
    def _has_range_check(self, sig_name: str, template: Template) -> bool:
        """Check if signal has a range check constraint."""
        for constraint in template.constraints:
            if sig_name in constraint.signals_involved:
                # Look for LessThan, GreaterThan, Bits2Num patterns
                if any(kw in constraint.expr for kw in ['LessThan', 'GreaterThan', 'Num2Bits', 'IsZero']):
                    return True
        return False
    
    def _check_non_determinism(self, template: Template):
        """Check for non-deterministic constraint patterns."""
        # Look for x^2 === y patterns without sign determination
        for constraint in template.constraints:
            if constraint.is_quadratic:
                expr = constraint.expr
                # Check for x * x === y pattern
                if re.search(r'(\w+)\s*\*\s*\1\s*===', expr):
                    self.findings.append(Finding(
                        severity="HIGH",
                        vuln_type="non-deterministic",
                        signal_name="(square operation)",
                        template_name=template.name,
                        description=f"Squaring constraint '{expr}' allows ±x as valid witnesses.",
                        impact="Two valid proofs possible for same public inputs.",
                        line_number=constraint.line_number,
                        recommendation="Add sign bit constraint or use absolute value pattern.",
                    ))
    
    def _check_nullifier_soundness(self, template: Template):
        """Check nullifier patterns for soundness issues."""
        # Look for nullifier OUTPUT signals (not inputs - inputs are typically correct)
        for sig_name, signal in template.signals.items():
            if 'nullifier' in sig_name.lower() and signal.signal_type == SignalType.OUTPUT:
                # Check if nullifier is properly bound to identity/secret
                has_identity_binding = False
                
                for constraint in template.constraints:
                    if sig_name in constraint.signals_involved:
                        # Check if constraint involves secret/identity/private inputs
                        if any(kw in constraint.expr.lower() for kw in ['secret', 'identity', 'nullifier', 'private']):
                            has_identity_binding = True
                        # Check if it's assigned from a hash component
                        if any(kw in constraint.expr for kw in ['Poseidon', 'Pedersen', 'MiMC', 'hasher']):
                            has_identity_binding = True
                
                if not has_identity_binding:
                    self.findings.append(Finding(
                        severity="CRITICAL",
                        vuln_type="nullifier-unsound",
                        signal_name=sig_name,
                        template_name=template.name,
                        description=f"Nullifier output '{sig_name}' not bound to identity secret.",
                        impact="Identity can generate multiple proofs with same nullifier.",
                        line_number=signal.line_number,
                        recommendation="Nullifier must be hash(secret, scope) or similar.",
                    ))


def hunt_circom(source_path: str, focus: str = "under-constrained") -> list[dict]:
    """
    Main entry point for Circom vulnerability hunting.
    
    Args:
        source_path: Path to .circom file or directory
        focus: Vulnerability focus (under-constrained, all)
        
    Returns:
        List of finding dictionaries
    """
    path = Path(source_path)
    
    if not path.exists():
        print(f"Error: Path does not exist: {source_path}")
        return []
    
    parser = CircomParser()
    all_templates = []
    
    if path.is_file():
        source = path.read_text()
        templates = parser.parse(source, str(path))
        all_templates.extend(templates)
        print(f"📄 Parsed: {path.name} ({len(templates)} template(s))")
    else:
        # Directory - find all .circom files
        circom_files = list(path.rglob("*.circom"))
        print(f"📁 Found {len(circom_files)} Circom files")
        
        for circom_file in circom_files:
            try:
                source = circom_file.read_text()
                templates = parser.parse(source, str(circom_file))
                all_templates.extend(templates)
                print(f"  📄 {circom_file.name}: {len(templates)} template(s)")
            except Exception as e:
                print(f"  ⚠️  Error parsing {circom_file.name}: {e}")
    
    if not all_templates:
        print("No templates found!")
        return []
    
    print(f"\n🔍 Analyzing {len(all_templates)} template(s)...")
    print()
    
    # Run hunters
    hunter = UnderConstrainedHunter(parser)
    findings = hunter.hunt(all_templates)
    
    # Convert Finding objects to dicts for CLI compatibility
    return [f.to_dict() for f in findings]


def format_findings(findings: list[Finding], as_json: bool = False) -> str:
    """Format findings for output."""
    if as_json:
        return json.dumps(findings, indent=2)
    
    if not findings:
        return "✅ No vulnerabilities found!"
    
    lines = []
    lines.append(f"🚨 Found {len(findings)} potential vulnerability/ies:\n")
    
    # Group by severity (findings are now dicts)
    by_severity = {"CRITICAL": [], "HIGH": [], "MEDIUM": [], "LOW": [], "INFO": []}
    for f in findings:
        sev = f["severity"] if isinstance(f, dict) else f.severity
        by_severity.get(sev, by_severity["INFO"]).append(f)
    
    for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
        items = by_severity[severity]
        if items:
            emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢", "INFO": "🔵"}[severity]
            lines.append(f"\n{emoji} {severity} ({len(items)}):")
            lines.append("-" * 50)
            
            for f in items:
                if isinstance(f, dict):
                    lines.append(f"\n  📍 {f['template']}::{f['signal']} (line {f['line']})")
                    lines.append(f"     Type: {f['type']}")
                    lines.append(f"     {f['description']}")
                    lines.append(f"     Impact: {f['impact']}")
                    if f.get('recommendation'):
                        lines.append(f"     Fix: {f['recommendation']}")
                else:
                    lines.append(f"\n  📍 {f.template_name}::{f.signal_name} (line {f.line_number})")
                    lines.append(f"     Type: {f.vuln_type}")
                    lines.append(f"     {f.description}")
                    lines.append(f"     Impact: {f.impact}")
                    if f.recommendation:
                        lines.append(f"     Fix: {f.recommendation}")
    
    return "\n".join(lines)
    
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python circom_parser.py <path_to_circom>")
        sys.exit(1)
    
    findings = hunt_circom(sys.argv[1])
    print(format_findings(findings))
