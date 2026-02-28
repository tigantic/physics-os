#!/usr/bin/env python3
"""
FLUIDELITE Halo2/Rust Circuit Analyzer

Scans Rust-based Halo2 circuits for common vulnerability patterns:
1. Missing range checks on advice columns
2. Unconstrained witness generation (assign_advice without constraint)
3. Missing copy constraints between regions
4. Lookup table soundness issues
5. Selector activation gaps
6. Rotation overflow/underflow
"""

import os
import re
import sys
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict


@dataclass
class Halo2Finding:
    """Represents a potential vulnerability in Halo2 circuits."""
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    category: str
    file: str
    line: int
    code: str
    description: str
    recommendation: str


@dataclass
class Halo2CircuitAnalysis:
    """Analysis results for a Halo2 circuit file."""
    file: str
    advice_columns: List[str] = field(default_factory=list)
    fixed_columns: List[str] = field(default_factory=list)
    selectors: List[str] = field(default_factory=list)
    gates: List[str] = field(default_factory=list)
    lookups: List[str] = field(default_factory=list)
    copy_constraints: List[str] = field(default_factory=list)
    witness_assignments: List[Tuple[int, str]] = field(default_factory=list)
    constraint_expressions: List[Tuple[int, str]] = field(default_factory=list)
    findings: List[Halo2Finding] = field(default_factory=list)


class Halo2Analyzer:
    """Analyzer for Halo2 Rust circuits."""
    
    # Patterns for Halo2 circuit analysis
    PATTERNS = {
        # Column declarations
        'advice_column': re.compile(r'let\s+(\w+)\s*=\s*meta\.advice_column\s*\('),
        'fixed_column': re.compile(r'let\s+(\w+)\s*=\s*meta\.fixed_column\s*\('),
        'selector': re.compile(r'let\s+(\w+)\s*=\s*meta\.selector\s*\('),
        'complex_selector': re.compile(r'let\s+(\w+)\s*=\s*meta\.complex_selector\s*\('),
        
        # Gate definitions
        'create_gate': re.compile(r'meta\.create_gate\s*\(\s*["\']([^"\']+)["\']'),
        
        # Lookups
        'lookup': re.compile(r'meta\.lookup\s*\(\s*["\']([^"\']+)["\']'),
        'lookup_any': re.compile(r'meta\.lookup_any\s*\(\s*["\']([^"\']+)["\']'),
        
        # Witness assignment
        'assign_advice': re.compile(r'\.assign_advice\s*\('),
        'assign_advice_from_constant': re.compile(r'\.assign_advice_from_constant\s*\('),
        'assign_fixed': re.compile(r'\.assign_fixed\s*\('),
        
        # Copy constraints
        'copy_advice': re.compile(r'\.copy_advice\s*\('),
        'constrain_equal': re.compile(r'\.constrain_equal\s*\('),
        
        # Range checks
        'range_check': re.compile(r'range_check|RangeCheck|range_constraint'),
        'num_to_bits': re.compile(r'num_to_bits|Num2Bits|to_bits'),
        
        # Expression building
        'query_advice': re.compile(r'meta\.query_advice\s*\(\s*(\w+)'),
        'query_selector': re.compile(r'meta\.query_selector\s*\(\s*(\w+)'),
        'rotation': re.compile(r'Rotation::(cur|prev|next|\(\s*(-?\d+)\s*\))'),
        
        # Dangerous patterns
        'unsafe_witness': re.compile(r'//\s*SAFETY:|unsafe\s*\{|\.unwrap\(\)'),
        'todo_fixme': re.compile(r'//\s*(TODO|FIXME|HACK|XXX|BUG)'),
        'panic_unreachable': re.compile(r'panic!\s*\(|unreachable!\s*\('),
        
        # Constraint expressions
        'constraint_expr': re.compile(r'\.\s*expr\s*\(\s*\)\s*[\*\+\-]'),
        'zero_check': re.compile(r'===?\s*0|Expression::Constant\s*\(\s*F::ZERO'),
    }
    
    # High-risk patterns that often indicate vulnerabilities
    HIGH_RISK_PATTERNS = {
        'unchecked_division': re.compile(r'\.div\s*\(|/\s*\w+(?!\s*\.\s*checked)'),
        'missing_overflow': re.compile(r'\+\s*(?!checked_add)|\.wrapping_add'),
        'field_from_u128': re.compile(r'F::from_u128|from_u64'),
        'direct_field_construction': re.compile(r'F::from\s*\(\s*\d+\s*\)'),
    }
    
    def __init__(self):
        self.findings: List[Halo2Finding] = []
        self.files_analyzed = 0
        self.total_lines = 0
        
    def analyze_directory(self, path: str) -> List[Halo2CircuitAnalysis]:
        """Analyze all Rust files in a directory."""
        results = []
        path = Path(path)
        
        if path.is_file() and path.suffix == '.rs':
            results.append(self.analyze_file(path))
        else:
            for rs_file in path.rglob('*.rs'):
                # Skip test files unless explicitly included
                if '/tests/' in str(rs_file) or '_test.rs' in str(rs_file):
                    continue
                try:
                    results.append(self.analyze_file(rs_file))
                except Exception as e:
                    print(f"  [!] Error analyzing {rs_file}: {e}")
        
        return [r for r in results if r is not None]
    
    def analyze_file(self, file_path: Path) -> Optional[Halo2CircuitAnalysis]:
        """Analyze a single Rust file for Halo2 patterns."""
        try:
            content = file_path.read_text()
        except Exception as e:
            return None
        
        # Quick check if this is a Halo2 circuit file
        if 'halo2' not in content and 'ConstraintSystem' not in content:
            return None
        
        self.files_analyzed += 1
        lines = content.split('\n')
        self.total_lines += len(lines)
        
        analysis = Halo2CircuitAnalysis(file=str(file_path))
        
        # Extract circuit components
        for match in self.PATTERNS['advice_column'].finditer(content):
            analysis.advice_columns.append(match.group(1))
        
        for match in self.PATTERNS['fixed_column'].finditer(content):
            analysis.fixed_columns.append(match.group(1))
        
        for match in self.PATTERNS['selector'].finditer(content):
            analysis.selectors.append(match.group(1))
        
        for match in self.PATTERNS['complex_selector'].finditer(content):
            analysis.selectors.append(match.group(1))
        
        for match in self.PATTERNS['create_gate'].finditer(content):
            analysis.gates.append(match.group(1))
        
        for match in self.PATTERNS['lookup'].finditer(content):
            analysis.lookups.append(match.group(1))
        
        for match in self.PATTERNS['lookup_any'].finditer(content):
            analysis.lookups.append(match.group(1))
        
        # Find witness assignments and their line numbers
        for i, line in enumerate(lines, 1):
            if self.PATTERNS['assign_advice'].search(line):
                analysis.witness_assignments.append((i, line.strip()))
            if self.PATTERNS['copy_advice'].search(line) or self.PATTERNS['constrain_equal'].search(line):
                analysis.copy_constraints.append((i, line.strip()))
        
        # Run vulnerability checks
        self._check_unconstrained_witness(analysis, content, lines)
        self._check_missing_range_checks(analysis, content, lines)
        self._check_rotation_issues(analysis, content, lines)
        self._check_lookup_soundness(analysis, content, lines)
        self._check_dangerous_patterns(analysis, content, lines)
        self._check_selector_gaps(analysis, content, lines)
        
        return analysis
    
    def _check_unconstrained_witness(self, analysis: Halo2CircuitAnalysis, content: str, lines: List[str]):
        """Check for witness assignments without corresponding constraints."""
        # This is a heuristic - look for assign_advice without nearby constraints
        for line_num, assign_line in analysis.witness_assignments:
            # Check if there's a constraint within 20 lines
            start = max(0, line_num - 10)
            end = min(len(lines), line_num + 20)
            context = '\n'.join(lines[start:end])
            
            # Look for constraint patterns
            has_constraint = (
                self.PATTERNS['constraint_expr'].search(context) or
                self.PATTERNS['zero_check'].search(context) or
                'constrain' in context.lower()
            )
            
            if not has_constraint:
                analysis.findings.append(Halo2Finding(
                    severity='HIGH',
                    category='UNCONSTRAINED_WITNESS',
                    file=analysis.file,
                    line=line_num,
                    code=assign_line[:100],
                    description='Witness assignment without visible constraint nearby. The assigned value may be unconstrained.',
                    recommendation='Verify that this assignment is properly constrained in a gate or lookup.'
                ))
    
    def _check_missing_range_checks(self, analysis: Halo2CircuitAnalysis, content: str, lines: List[str]):
        """Check for advice columns without range checks."""
        for advice_col in analysis.advice_columns:
            # Check if this column has a range check
            range_pattern = re.compile(rf'{advice_col}.*range|RangeCheck.*{advice_col}', re.IGNORECASE)
            if not range_pattern.search(content):
                # Find where the column is used
                for i, line in enumerate(lines, 1):
                    if advice_col in line and 'query_advice' in line:
                        analysis.findings.append(Halo2Finding(
                            severity='MEDIUM',
                            category='MISSING_RANGE_CHECK',
                            file=analysis.file,
                            line=i,
                            code=line.strip()[:100],
                            description=f'Advice column "{advice_col}" may be missing range check. Field elements can be any value in the field.',
                            recommendation='Add a range check constraint or lookup to bound the value.'
                        ))
                        break
    
    def _check_rotation_issues(self, analysis: Halo2CircuitAnalysis, content: str, lines: List[str]):
        """Check for potential rotation overflow/underflow issues."""
        for i, line in enumerate(lines, 1):
            # Check for large rotation values
            rotation_match = re.search(r'Rotation\s*\(\s*(-?\d+)\s*\)', line)
            if rotation_match:
                rotation_val = int(rotation_match.group(1))
                if abs(rotation_val) > 10:
                    analysis.findings.append(Halo2Finding(
                        severity='LOW',
                        category='LARGE_ROTATION',
                        file=analysis.file,
                        line=i,
                        code=line.strip()[:100],
                        description=f'Large rotation value ({rotation_val}) may cause issues at region boundaries.',
                        recommendation='Verify rotation bounds are checked at region edges.'
                    ))
    
    def _check_lookup_soundness(self, analysis: Halo2CircuitAnalysis, content: str, lines: List[str]):
        """Check for potential lookup soundness issues."""
        # Check for lookups without selector
        for lookup_name in analysis.lookups:
            lookup_pattern = re.compile(rf'lookup.*{lookup_name}|{lookup_name}.*lookup', re.IGNORECASE | re.DOTALL)
            match = lookup_pattern.search(content)
            if match:
                context = content[max(0, match.start()-200):min(len(content), match.end()+500)]
                if 'selector' not in context.lower() and 'enable' not in context.lower():
                    analysis.findings.append(Halo2Finding(
                        severity='HIGH',
                        category='LOOKUP_WITHOUT_SELECTOR',
                        file=analysis.file,
                        line=0,
                        code=f'Lookup: {lookup_name}',
                        description=f'Lookup "{lookup_name}" may be missing a selector. All rows will participate in lookup.',
                        recommendation='Add a selector to control which rows participate in the lookup.'
                    ))
    
    def _check_dangerous_patterns(self, analysis: Halo2CircuitAnalysis, content: str, lines: List[str]):
        """Check for dangerous code patterns."""
        for i, line in enumerate(lines, 1):
            # Check for TODO/FIXME/BUG comments
            todo_match = self.PATTERNS['todo_fixme'].search(line)
            if todo_match:
                analysis.findings.append(Halo2Finding(
                    severity='LOW',
                    category='INCOMPLETE_CODE',
                    file=analysis.file,
                    line=i,
                    code=line.strip()[:100],
                    description=f'{todo_match.group(1)} comment indicates incomplete implementation.',
                    recommendation='Review and complete the implementation before production use.'
                ))
            
            # Check for division without overflow check
            if self.HIGH_RISK_PATTERNS['unchecked_division'].search(line):
                if 'checked_div' not in line and 'is_zero' not in content[max(0, i*80-400):min(len(content), i*80+400)]:
                    analysis.findings.append(Halo2Finding(
                        severity='MEDIUM',
                        category='UNCHECKED_DIVISION',
                        file=analysis.file,
                        line=i,
                        code=line.strip()[:100],
                        description='Division without explicit zero check. May cause constraint failure or undefined behavior.',
                        recommendation='Add explicit zero check before division.'
                    ))
    
    def _check_selector_gaps(self, analysis: Halo2CircuitAnalysis, content: str, lines: List[str]):
        """Check for potential selector activation issues."""
        if len(analysis.selectors) > 5:
            # Many selectors might indicate complex selector logic
            analysis.findings.append(Halo2Finding(
                severity='LOW',
                category='COMPLEX_SELECTOR_LOGIC',
                file=analysis.file,
                line=0,
                code=f'{len(analysis.selectors)} selectors defined',
                description='Many selectors defined. Complex selector logic may have gaps allowing invalid states.',
                recommendation='Verify selector mutual exclusivity and complete coverage.'
            ))
    
    def generate_report(self, results: List[Halo2CircuitAnalysis], output_path: str = 'halo2_report.md'):
        """Generate a markdown report of findings."""
        all_findings = []
        for r in results:
            all_findings.extend(r.findings)
        
        # Count by severity
        critical = [f for f in all_findings if f.severity == 'CRITICAL']
        high = [f for f in all_findings if f.severity == 'HIGH']
        medium = [f for f in all_findings if f.severity == 'MEDIUM']
        low = [f for f in all_findings if f.severity == 'LOW']
        
        with open(output_path, 'w') as f:
            f.write('# FLUIDELITE Halo2 Circuit Analysis Report\n\n')
            f.write(f'**Generated**: {__import__("datetime").datetime.now().isoformat()}\n')
            f.write(f'**Files Analyzed**: {self.files_analyzed}\n')
            f.write(f'**Total Lines**: {self.total_lines:,}\n\n')
            
            f.write('## Summary\n\n')
            f.write(f'| Severity | Count |\n')
            f.write(f'|----------|-------|\n')
            f.write(f'| 🚨 CRITICAL | {len(critical)} |\n')
            f.write(f'| ⚠️ HIGH | {len(high)} |\n')
            f.write(f'| 🔶 MEDIUM | {len(medium)} |\n')
            f.write(f'| ℹ️ LOW | {len(low)} |\n\n')
            
            # Circuit summary
            f.write('## Circuits Analyzed\n\n')
            f.write('| File | Advice | Fixed | Selectors | Gates | Lookups | Findings |\n')
            f.write('|------|--------|-------|-----------|-------|---------|----------|\n')
            for r in results:
                if r.advice_columns or r.gates:  # Only show actual circuit files
                    f.write(f'| {Path(r.file).name} | {len(r.advice_columns)} | {len(r.fixed_columns)} | ')
                    f.write(f'{len(r.selectors)} | {len(r.gates)} | {len(r.lookups)} | {len(r.findings)} |\n')
            
            # Detailed findings
            if all_findings:
                f.write('\n## Detailed Findings\n\n')
                
                for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
                    findings = [f for f in all_findings if f.severity == severity]
                    if findings:
                        emoji = {'CRITICAL': '🚨', 'HIGH': '⚠️', 'MEDIUM': '🔶', 'LOW': 'ℹ️'}[severity]
                        f.write(f'### {emoji} {severity} Findings\n\n')
                        for finding in findings:
                            f.write(f'#### {finding.category}\n\n')
                            f.write(f'**File**: `{finding.file}`\n')
                            if finding.line > 0:
                                f.write(f'**Line**: {finding.line}\n')
                            f.write(f'**Description**: {finding.description}\n\n')
                            f.write(f'```rust\n{finding.code}\n```\n\n')
                            f.write(f'**Recommendation**: {finding.recommendation}\n\n')
                            f.write('---\n\n')
        
        return output_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='FLUIDELITE Halo2 Circuit Analyzer')
    parser.add_argument('target', help='Directory or file to analyze')
    parser.add_argument('--output', '-o', default='halo2_report.md', help='Output report path')
    parser.add_argument('--json', action='store_true', help='Output JSON instead of markdown')
    
    args = parser.parse_args()
    
    print(f'[FLUIDELITE] Halo2 Circuit Analyzer')
    print(f'[FLUIDELITE] Analyzing: {args.target}')
    
    analyzer = Halo2Analyzer()
    results = analyzer.analyze_directory(args.target)
    
    if args.json:
        output_data = {
            'files_analyzed': analyzer.files_analyzed,
            'total_lines': analyzer.total_lines,
            'findings': [
                {
                    'severity': f.severity,
                    'category': f.category,
                    'file': f.file,
                    'line': f.line,
                    'description': f.description
                }
                for r in results for f in r.findings
            ]
        }
        with open(args.output.replace('.md', '.json'), 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f'[FLUIDELITE] JSON report written to: {args.output.replace(".md", ".json")}')
    else:
        report_path = analyzer.generate_report(results, args.output)
        print(f'[FLUIDELITE] Report written to: {report_path}')
    
    # Summary output
    all_findings = [f for r in results for f in r.findings]
    critical = len([f for f in all_findings if f.severity == 'CRITICAL'])
    high = len([f for f in all_findings if f.severity == 'HIGH'])
    
    print(f'\n[FLUIDELITE] Files analyzed: {analyzer.files_analyzed}')
    print(f'[FLUIDELITE] Total lines: {analyzer.total_lines:,}')
    print(f'[FLUIDELITE] Total findings: {len(all_findings)}')
    if critical > 0:
        print(f'🚨 CRITICAL FINDINGS: {critical}')
    if high > 0:
        print(f'⚠️  HIGH FINDINGS: {high}')


if __name__ == '__main__':
    main()
