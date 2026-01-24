#!/usr/bin/env python3
"""
zkASM Elite Analyzer v1.0
=========================

Analyze Polygon zkEVM ROM (zkASM) for vulnerabilities in FREE value handling.

The attack surface for the $1M bounty is in zkASM/ROM, not PIL constraints.
This analyzer finds:
1. FREE inputs ($) that bypass validation
2. Arithmetic operations with unvalidated inputs
3. Missing range checks on prover-supplied values
4. EC operation edge cases
5. State-affecting operations with FREE values

Author: FLUIDELITE Team
Target: Polygon zkEVM ($1,000,000 bounty)
"""

import re
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
from enum import Enum, auto
from collections import defaultdict


class Severity(Enum):
    CRITICAL = auto()  # Immediate soundness break
    HIGH = auto()       # Potential exploit path
    MEDIUM = auto()     # Needs verification
    LOW = auto()        # Informational
    INFO = auto()       # Design observation


class VulnType(Enum):
    FREE_NO_VALIDATION = auto()     # $ used without range check
    ARITH_FREE_INPUT = auto()       # Arithmetic with unvalidated FREE
    EC_EDGE_CASE = auto()           # EC operation edge case
    DIVISION_UNCHECKED = auto()     # Division without zero check
    HASH_FREE_INPUT = auto()        # Hash with unvalidated FREE
    STATE_FREE_INPUT = auto()       # State modification via FREE
    JUMP_FREE_COND = auto()         # Jump based on unvalidated FREE
    ASSERT_BYPASS = auto()          # Assert that can be bypassed
    MEMORY_FREE_ADDR = auto()       # Memory access with FREE address
    OVERFLOW_POSSIBLE = auto()      # Arithmetic overflow possible


@dataclass
class Finding:
    vuln_type: VulnType
    severity: Severity
    file: str
    line: int
    code: str
    description: str
    context: List[str] = field(default_factory=list)
    validation_path: Optional[str] = None


@dataclass
class Instruction:
    line_num: int
    raw: str
    free_inputs: List[str]       # $, ${...} patterns
    operations: List[str]        # ARITH, MLOAD, etc.
    registers: Dict[str, str]    # A, B, C, D, E assignments
    jumps: List[str]             # JMPC, JMPN, CALL, etc.
    labels: List[str]            # Labels defined
    is_assert: bool = False
    has_range_check: bool = False


class ZkAsmEliteAnalyzer:
    """Elite analyzer for Polygon zkEVM ROM vulnerabilities."""
    
    # Instruction patterns
    FREE_SIMPLE = re.compile(r'\$\s*=>')
    FREE_EXPR = re.compile(r'\$\{([^}]+)\}')
    ARITH_OP = re.compile(r':ARITH(?:_\w+)?')
    HASH_OP = re.compile(r':HASH(?:K|P|S)\d*')
    MSTORE_OP = re.compile(r':MSTORE\(([^)]+)\)')
    MLOAD_OP = re.compile(r':MLOAD\(([^)]+)\)')
    SSTORE_OP = re.compile(r':SSTORE')
    SLOAD_OP = re.compile(r':SLOAD')
    JUMP_OPS = re.compile(r':(JMPC|JMPN|JMPZ|JMP|CALL|RETURN)')
    LABEL = re.compile(r'^(\w+):(?!\s*[A-Z])')
    ASSERT_OP = re.compile(r':ASSERT')
    EQ_CHECK = re.compile(r':EQ')
    LT_CHECK = re.compile(r':LT')
    REGISTER_ASSIGN = re.compile(r'([A-E])\s*=>')
    
    # Validation patterns - these indicate FREE is checked
    VALIDATION_PATTERNS = [
        re.compile(r':EQ,\s*JMPC'),     # Equality check with conditional jump
        re.compile(r':LT,\s*JMPC'),     # Less-than check with conditional jump
        re.compile(r':JMPZ'),           # Zero check jump
        re.compile(r':ASSERT'),         # Assert constraint
        re.compile(r'0n\s*=>\s*A.*:EQ'), # Compare to zero
    ]
    
    # High-risk operation patterns
    EC_OPERATIONS = re.compile(r':ARITH_EC(?:ADD|DBL|MUL)')
    DIV_OPERATIONS = re.compile(r'(?:div|DIV|/)')
    INV_OPERATIONS = re.compile(r'inv|INV')
    
    def __init__(self):
        self.findings: List[Finding] = []
        self.instructions: Dict[str, List[Instruction]] = {}
        self.labels: Dict[str, Tuple[str, int]] = {}  # label -> (file, line)
        self.free_usage_stats: Dict[str, int] = defaultdict(int)
        self.call_graph: Dict[str, Set[str]] = defaultdict(set)
        
    def analyze_directory(self, directory: Path) -> Dict:
        """Analyze all zkASM files in directory."""
        print(f"\n{'='*70}")
        print(f"    🔍 zkASM ELITE ANALYZER v1.0 - Polygon zkEVM ROM Analysis")
        print(f"{'='*70}\n")
        
        zkasm_files = list(directory.rglob("*.zkasm"))
        print(f"Found {len(zkasm_files)} zkASM files\n")
        
        # Parse all files
        total_instructions = 0
        total_free_inputs = 0
        
        for zkasm_file in zkasm_files:
            instructions = self._parse_file(zkasm_file)
            self.instructions[str(zkasm_file)] = instructions
            total_instructions += len(instructions)
            
            for instr in instructions:
                total_free_inputs += len(instr.free_inputs)
        
        print(f"Parsed {total_instructions} instructions with {total_free_inputs} FREE inputs\n")
        
        # Build label map and call graph
        self._build_call_graph()
        
        # Analyze for vulnerabilities
        self._analyze_free_validation()
        self._analyze_arithmetic()
        self._analyze_ec_operations()
        self._analyze_state_operations()
        self._analyze_jump_conditions()
        
        # Generate report
        return self._generate_report()
    
    def _parse_file(self, filepath: Path) -> List[Instruction]:
        """Parse zkASM file into structured instructions."""
        instructions = []
        
        try:
            content = filepath.read_text()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return []
        
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Skip empty lines and pure comments
            stripped = line.strip()
            if not stripped or stripped.startswith(';'):
                continue
            
            # Remove inline comments
            if ';' in stripped:
                stripped = stripped[:stripped.index(';')].strip()
            
            if not stripped:
                continue
            
            instr = self._parse_instruction(stripped, i)
            if instr:
                instructions.append(instr)
                
                # Track labels
                for label in instr.labels:
                    self.labels[label] = (str(filepath), i)
        
        return instructions
    
    def _parse_instruction(self, line: str, line_num: int) -> Optional[Instruction]:
        """Parse single zkASM instruction."""
        # Find FREE inputs
        free_simple = self.FREE_SIMPLE.findall(line)
        free_expr = self.FREE_EXPR.findall(line)
        free_inputs = free_simple + free_expr
        
        # Find operations
        operations = []
        if self.ARITH_OP.search(line):
            operations.append('ARITH')
        if self.HASH_OP.search(line):
            operations.append('HASH')
        if mstore := self.MSTORE_OP.search(line):
            operations.append(f'MSTORE({mstore.group(1)})')
        if mload := self.MLOAD_OP.search(line):
            operations.append(f'MLOAD({mload.group(1)})')
        if self.SSTORE_OP.search(line):
            operations.append('SSTORE')
        if self.SLOAD_OP.search(line):
            operations.append('SLOAD')
        
        # Find register assignments
        registers = {}
        for match in self.REGISTER_ASSIGN.finditer(line):
            reg = match.group(1)
            # Find what's assigned to register
            before = line[:match.start()].strip()
            if before:
                registers[reg] = before
        
        # Find jumps
        jumps = self.JUMP_OPS.findall(line)
        
        # Find labels
        labels = []
        label_match = self.LABEL.match(line)
        if label_match:
            labels.append(label_match.group(1))
        
        # Check for assertions
        is_assert = bool(self.ASSERT_OP.search(line))
        
        # Check for range checks
        has_range_check = bool(self.EQ_CHECK.search(line) or self.LT_CHECK.search(line))
        
        return Instruction(
            line_num=line_num,
            raw=line,
            free_inputs=free_inputs,
            operations=operations,
            registers=registers,
            jumps=jumps,
            labels=labels,
            is_assert=is_assert,
            has_range_check=has_range_check
        )
    
    def _build_call_graph(self):
        """Build function call graph from CALL instructions."""
        for filepath, instructions in self.instructions.items():
            current_label = None
            
            for instr in instructions:
                if instr.labels:
                    current_label = instr.labels[0]
                
                if current_label and 'CALL' in instr.jumps:
                    # Extract call target
                    call_match = re.search(r'CALL\((\w+)\)', instr.raw)
                    if call_match:
                        target = call_match.group(1)
                        self.call_graph[current_label].add(target)
    
    def _analyze_free_validation(self):
        """Find FREE inputs that lack validation."""
        for filepath, instructions in self.instructions.items():
            for i, instr in enumerate(instructions):
                if not instr.free_inputs:
                    continue
                
                # Track each FREE usage
                for free_input in instr.free_inputs:
                    self.free_usage_stats['total'] += 1
                    
                    # Check if this FREE has validation in context
                    has_validation = self._check_validation_context(
                        instructions, i, window=5
                    )
                    
                    if has_validation:
                        self.free_usage_stats['validated'] += 1
                    else:
                        self.free_usage_stats['unvalidated'] += 1
                        
                        # Check if it affects critical operations
                        if self._is_critical_usage(instr):
                            self.findings.append(Finding(
                                vuln_type=VulnType.FREE_NO_VALIDATION,
                                severity=Severity.HIGH,
                                file=filepath,
                                line=instr.line_num,
                                code=instr.raw,
                                description=f"FREE input without validation in critical operation",
                                context=self._get_context(instructions, i, 3)
                            ))
    
    def _check_validation_context(self, instructions: List[Instruction], 
                                   idx: int, window: int = 5) -> bool:
        """Check if FREE at idx has validation within window."""
        # Check before and after
        start = max(0, idx - window)
        end = min(len(instructions), idx + window + 1)
        
        for i in range(start, end):
            instr = instructions[i]
            
            # Check for validation patterns
            for pattern in self.VALIDATION_PATTERNS:
                if pattern.search(instr.raw):
                    return True
            
            # Check for MLOAD from validated source
            if ':MLOAD' in instr.raw and instr.has_range_check:
                return True
        
        return False
    
    def _is_critical_usage(self, instr: Instruction) -> bool:
        """Check if instruction affects critical state."""
        critical_ops = ['ARITH', 'SSTORE', 'HASH']
        return any(op in instr.operations for op in critical_ops) or \
               'ARITH' in instr.raw
    
    def _analyze_arithmetic(self):
        """Analyze arithmetic operations for vulnerabilities."""
        for filepath, instructions in self.instructions.items():
            for i, instr in enumerate(instructions):
                # Check for ARITH with FREE inputs
                if 'ARITH' in instr.operations and instr.free_inputs:
                    # Check for division
                    if self.DIV_OPERATIONS.search(instr.raw):
                        self.findings.append(Finding(
                            vuln_type=VulnType.DIVISION_UNCHECKED,
                            severity=Severity.HIGH,
                            file=filepath,
                            line=instr.line_num,
                            code=instr.raw,
                            description="Division with FREE input - potential div-by-zero",
                            context=self._get_context(instructions, i, 3)
                        ))
                    
                    # Check for inversion
                    elif self.INV_OPERATIONS.search(instr.raw):
                        self.findings.append(Finding(
                            vuln_type=VulnType.ARITH_FREE_INPUT,
                            severity=Severity.MEDIUM,
                            file=filepath,
                            line=instr.line_num,
                            code=instr.raw,
                            description="Field inversion with FREE input",
                            context=self._get_context(instructions, i, 3)
                        ))
    
    def _analyze_ec_operations(self):
        """Analyze EC operations for edge cases."""
        for filepath, instructions in self.instructions.items():
            for i, instr in enumerate(instructions):
                if self.EC_OPERATIONS.search(instr.raw):
                    # Check for point at infinity handling
                    context = self._get_context(instructions, i, 10)
                    has_infinity_check = any(
                        'infinity' in c.lower() or 'FPEC' in c
                        for c in context
                    )
                    
                    if not has_infinity_check and instr.free_inputs:
                        self.findings.append(Finding(
                            vuln_type=VulnType.EC_EDGE_CASE,
                            severity=Severity.MEDIUM,
                            file=filepath,
                            line=instr.line_num,
                            code=instr.raw,
                            description="EC operation with FREE input - check infinity handling",
                            context=context
                        ))
    
    def _analyze_state_operations(self):
        """Analyze state-affecting operations."""
        for filepath, instructions in self.instructions.items():
            for i, instr in enumerate(instructions):
                # SSTORE with FREE value
                if 'SSTORE' in instr.operations:
                    if instr.free_inputs:
                        self.findings.append(Finding(
                            vuln_type=VulnType.STATE_FREE_INPUT,
                            severity=Severity.CRITICAL,
                            file=filepath,
                            line=instr.line_num,
                            code=instr.raw,
                            description="SSTORE with FREE input - state corruption possible",
                            context=self._get_context(instructions, i, 5)
                        ))
                
                # HASH with FREE input (could be used for state root)
                if 'HASH' in instr.operations and instr.free_inputs:
                    self.findings.append(Finding(
                        vuln_type=VulnType.HASH_FREE_INPUT,
                        severity=Severity.MEDIUM,
                        file=filepath,
                        line=instr.line_num,
                        code=instr.raw,
                        description="HASH with FREE input",
                        context=self._get_context(instructions, i, 3)
                    ))
    
    def _analyze_jump_conditions(self):
        """Analyze conditional jumps based on FREE values."""
        for filepath, instructions in self.instructions.items():
            for i, instr in enumerate(instructions):
                if instr.jumps and instr.free_inputs:
                    # Check if jump condition uses unvalidated FREE
                    if not instr.has_range_check and 'JMPC' in instr.jumps:
                        self.findings.append(Finding(
                            vuln_type=VulnType.JUMP_FREE_COND,
                            severity=Severity.MEDIUM,
                            file=filepath,
                            line=instr.line_num,
                            code=instr.raw,
                            description="Conditional jump with FREE input",
                            context=self._get_context(instructions, i, 3)
                        ))
    
    def _get_context(self, instructions: List[Instruction], 
                     idx: int, window: int) -> List[str]:
        """Get surrounding instructions as context."""
        start = max(0, idx - window)
        end = min(len(instructions), idx + window + 1)
        return [instructions[i].raw for i in range(start, end)]
    
    def _generate_report(self) -> Dict:
        """Generate analysis report."""
        # Count by severity
        severity_counts = defaultdict(int)
        vuln_type_counts = defaultdict(int)
        
        for finding in self.findings:
            severity_counts[finding.severity.name] += 1
            vuln_type_counts[finding.vuln_type.name] += 1
        
        # Print summary
        print("\n" + "="*70)
        print("    📊 ANALYSIS SUMMARY")
        print("="*70)
        
        print(f"\n📝 FREE INPUT STATISTICS:")
        print(f"   Total FREE inputs: {self.free_usage_stats['total']}")
        print(f"   Validated: {self.free_usage_stats.get('validated', 0)}")
        print(f"   Unvalidated (flagged): {self.free_usage_stats.get('unvalidated', 0)}")
        
        print(f"\n🔴 FINDINGS BY SEVERITY:")
        for sev in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']:
            count = severity_counts.get(sev, 0)
            if count > 0:
                icon = {'CRITICAL': '🚨', 'HIGH': '🔴', 'MEDIUM': '🟠', 
                        'LOW': '🟡', 'INFO': 'ℹ️'}.get(sev, '')
                print(f"   {icon} {sev}: {count}")
        
        print(f"\n📋 FINDINGS BY TYPE:")
        for vtype, count in sorted(vuln_type_counts.items(), 
                                   key=lambda x: -x[1])[:10]:
            print(f"   {vtype}: {count}")
        
        # Print top critical findings
        critical_findings = [f for f in self.findings 
                           if f.severity in [Severity.CRITICAL, Severity.HIGH]]
        
        if critical_findings:
            print(f"\n" + "="*70)
            print(f"    🚨 TOP CRITICAL/HIGH FINDINGS")
            print("="*70)
            
            for finding in critical_findings[:20]:
                print(f"\n[{finding.severity.name}] {finding.vuln_type.name}")
                print(f"   File: {Path(finding.file).name}:{finding.line}")
                print(f"   Code: {finding.code[:80]}...")
                print(f"   Desc: {finding.description}")
        
        return {
            'total_findings': len(self.findings),
            'severity_counts': dict(severity_counts),
            'vuln_type_counts': dict(vuln_type_counts),
            'free_stats': dict(self.free_usage_stats),
            'critical_findings': [
                {
                    'type': f.vuln_type.name,
                    'severity': f.severity.name,
                    'file': f.file,
                    'line': f.line,
                    'code': f.code,
                    'description': f.description
                }
                for f in critical_findings[:50]
            ]
        }


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python zkasm_elite_analyzer.py <zkasm_directory>")
        print("Example: python zkasm_elite_analyzer.py zk_targets/zkevm-rom/main")
        sys.exit(1)
    
    directory = Path(sys.argv[1])
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        sys.exit(1)
    
    analyzer = ZkAsmEliteAnalyzer()
    results = analyzer.analyze_directory(directory)
    
    # Save results to JSON
    output_file = Path("zkasm_analysis_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")
    print(f"\n{'='*70}")
    print(f"    zkASM Elite Analyzer v1.0 Complete")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
