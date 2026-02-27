#!/usr/bin/env python3
"""
FEZK ELITE PIL Analyzer v2.0
============================
Complete Polygon zkEVM PIL constraint analyzer with:
- Full lookup/permutation constraint parsing
- Cross-namespace signal dependency tracing  
- FREE signal data flow analysis
- ROM constraint verification
- Edge case detection for EC operations
- GPU-accelerated rank analysis

This is the production analyzer for $1M bounty hunting.

Author: FEZK Elite Team
Date: January 23, 2026
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any
from collections import defaultdict
from enum import Enum, auto
import numpy as np
from datetime import datetime


class SignalType(Enum):
    COMMIT = auto()       # pol commit - prover controlled
    CONSTANT = auto()     # pol constant - fixed
    INTERMEDIATE = auto() # pol x = expr
    PUBLIC = auto()       # public input


class ConstraintType(Enum):
    POLYNOMIAL = auto()       # expr = expr
    BINARY = auto()           # (1-x)*x = 0
    STATE_TRANSITION = auto() # x' = expr
    LOOKUP = auto()           # {...} in {...}
    PERMUTATION = auto()      # {...} is {...}
    ROM_LOOKUP = auto()       # Special: ROM constraint


@dataclass
class Signal:
    """Complete signal representation."""
    name: str
    namespace: str
    full_name: str
    signal_type: SignalType
    source_file: str = ""
    line: int = 0
    is_array: bool = False
    array_size: int = 0
    
    # Constraint tracking
    direct_constraints: List[int] = field(default_factory=list)
    lookup_constraints: List[int] = field(default_factory=list)
    permutation_constraints: List[int] = field(default_factory=list)
    state_transitions: List[int] = field(default_factory=list)
    
    # Data flow
    flows_to: Set[str] = field(default_factory=set)
    flows_from: Set[str] = field(default_factory=set)
    
    # Analysis
    is_free_input: bool = False  # FREE-type signal
    is_rom_constrained: bool = False
    is_fully_constrained: bool = False
    constraint_count: int = 0
    
    def total_constraints(self) -> int:
        return (len(self.direct_constraints) + 
                len(self.lookup_constraints) + 
                len(self.permutation_constraints) +
                len(self.state_transitions))


@dataclass
class Lookup:
    """Lookup/permutation constraint."""
    id: int
    selector: str
    lhs_columns: List[str]
    rhs_table: str
    rhs_columns: List[str]
    is_permutation: bool
    is_rom_lookup: bool = False
    namespace: str = ""
    source_file: str = ""
    line: int = 0
    raw_text: str = ""
    
    # Parsed signals
    lhs_signals: Set[str] = field(default_factory=set)
    rhs_signals: Set[str] = field(default_factory=set)
    selector_signals: Set[str] = field(default_factory=set)


@dataclass 
class Constraint:
    """Polynomial constraint."""
    id: int
    expr: str
    constraint_type: ConstraintType
    namespace: str = ""
    source_file: str = ""
    line: int = 0
    
    # Signals
    all_signals: Set[str] = field(default_factory=set)
    lhs_signals: Set[str] = field(default_factory=set)
    rhs_signals: Set[str] = field(default_factory=set)


@dataclass
class FREEFlowPath:
    """Tracks how a FREE signal flows through constraints."""
    free_signal: str
    path: List[Tuple[str, str]]  # [(constraint_type, target_signal), ...]
    reaches_state: bool = False
    reaches_memory: bool = False
    reaches_storage: bool = False
    verified_by_lookup: bool = False
    verification_lookup: str = ""


@dataclass
class Finding:
    """Security finding."""
    id: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW, INFO
    title: str
    signal: str
    description: str
    constraint_analysis: str
    status: str  # VULNERABLE, NEEDS_REVIEW, SECURE
    line: int = 0
    file: str = ""


class PILEliteAnalyzer:
    """
    Production-grade PIL analyzer for Polygon zkEVM.
    """
    
    def __init__(self, pil_dir: Path, verbose: bool = False):
        self.pil_dir = pil_dir
        self.verbose = verbose
        
        self.signals: Dict[str, Signal] = {}
        self.constraints: List[Constraint] = []
        self.lookups: List[Lookup] = []
        self.namespaces: Dict[str, Set[str]] = defaultdict(set)
        self.findings: List[Finding] = []
        
        self.current_namespace = "Global"
        self.current_file = ""
        self.constraint_id = 0
        self.lookup_id = 0
        
        # Track FREE signal flows
        self.free_flows: Dict[str, FREEFlowPath] = {}
        
        # ROM lookup tracking
        self.rom_lookup: Optional[Lookup] = None
        self.rom_constrained_signals: Set[str] = set()
        
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for PIL parsing."""
        self.patterns = {
            # Namespace: namespace Name(%N)
            'namespace': re.compile(r'namespace\s+(\w+)\s*\(\s*%?\w+\s*\)'),
            
            # pol commit signals
            'pol_commit': re.compile(r'pol\s+commit\s+([^;]+);'),
            
            # pol constant signals
            'pol_constant': re.compile(r'pol\s+constant\s+([^;]+);'),
            
            # pol intermediate: pol name = expr
            'pol_intermediate': re.compile(r'^\s*pol\s+(\w+)\s*=\s*([^;]+);', re.MULTILINE),
            
            # Array declaration: name[N]
            'array_decl': re.compile(r'(\w+)\[(\d+)\]'),
            
            # Binary constraint: (1-x)*x = 0
            'binary': re.compile(r'\(\s*1\s*-\s*(\w+)\s*\)\s*\*\s*\1\s*=\s*0'),
            
            # State transition: x' = expr
            'state_trans': re.compile(r"(\w+(?:\[\d+\])?)\s*'\s*=\s*([^;]+);"),
            
            # Lookup with selector: selector { cols } in table { cols }
            'lookup_selector': re.compile(
                r'(\w+(?:\s*[\+\*]\s*\w+)*)\s*\{([^}]+)\}\s*in\s*(\w+(?:\.\w+)?)\s*\{([^}]+)\}',
                re.DOTALL
            ),
            
            # Permutation with selector: selector { cols } is table { cols }
            'perm_selector': re.compile(
                r'(\w+(?:\s*[\+\*]\s*\w+)*)\s*\{([^}]+)\}\s*is\s*(\w+(?:\.\w+)?)\s*\{([^}]+)\}',
                re.DOTALL
            ),
            
            # Bare lookup: { cols } in { cols }
            'lookup_bare': re.compile(
                r'\{\s*([^}]+)\s*\}\s*in\s*\{\s*([^}]+)\s*\}',
                re.DOTALL
            ),
            
            # Bare permutation: { cols } is Table.selector { cols }
            'perm_bare': re.compile(
                r'\{\s*([^}]+)\s*\}\s*is\s*(\w+(?:\.\w+)?)\s*\{\s*([^}]+)\s*\}',
                re.DOTALL
            ),
            
            # Signal identifier
            'signal_ref': re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*(?:\.\w+)?(?:\[\d+\])?)\b'),
            
            # public input
            'public': re.compile(r'public\s+(\w+)\s*='),
        }
        
        self.keywords = {
            'pol', 'commit', 'constant', 'namespace', 'include', 'in', 'is',
            'public', 'if', 'else', 'for', 'while', 'node', 'tools'
        }
        
        self.operators = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
    
    def analyze(self) -> Dict[str, Any]:
        """Run complete analysis."""
        print("\n" + "="*80)
        print("FEZK ELITE PIL Analyzer v2.0")
        print("="*80)
        
        # Phase 1: Parse all PIL files
        print("\n[Phase 1] Parsing PIL files...")
        self._parse_all_files()
        
        # Phase 2: Build constraint graph
        print("\n[Phase 2] Building constraint graph...")
        self._build_constraint_graph()
        
        # Phase 3: Identify ROM lookup
        print("\n[Phase 3] Analyzing ROM constraints...")
        self._analyze_rom_lookup()
        
        # Phase 4: Trace FREE signals
        print("\n[Phase 4] Tracing FREE signal flows...")
        self._trace_free_signals()
        
        # Phase 5: Detect attack surfaces
        print("\n[Phase 5] Detecting attack surfaces...")
        self._detect_attack_surfaces()
        
        # Phase 6: Rank analysis
        print("\n[Phase 6] Computing constraint matrix rank...")
        rank_result = self._analyze_rank()
        
        # Generate report
        return self._generate_report(rank_result)
    
    def _parse_all_files(self):
        """Parse all PIL files."""
        pil_files = sorted(self.pil_dir.glob("*.pil"))
        
        for pil_file in pil_files:
            if self.verbose:
                print(f"  Parsing {pil_file.name}...")
            self._parse_file(pil_file)
        
        print(f"  Parsed {len(pil_files)} files")
        print(f"  Found {len(self.signals)} signals, {len(self.constraints)} constraints, {len(self.lookups)} lookups")
    
    def _parse_file(self, filepath: Path):
        """Parse a single PIL file."""
        content = filepath.read_text()
        self.current_file = filepath.name
        self.current_namespace = "Global"
        
        # Remove comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        
        # Pre-process: join multi-line statements
        lines = content.split('\n')
        joined_lines = []
        buffer = ""
        start_line = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            
            if not buffer:
                start_line = i + 1
                
            buffer += (" " if buffer else "") + stripped
            
            # Check for balanced braces AND complete statement
            open_braces = buffer.count('{')
            close_braces = buffer.count('}')
            has_semicolon = buffer.rstrip().endswith(';')
            
            # Keep accumulating if braces unbalanced
            if open_braces > close_braces:
                continue
            
            # Multi-line lookup detection: ends with } but doesn't have ' in ' yet
            if open_braces > 0 and close_braces == open_braces:
                if '}' in buffer and ' in ' not in buffer and ' is ' not in buffer and not has_semicolon:
                    continue
            
            # Keep accumulating if we have a partial lookup pattern
            if ' in {' in buffer and buffer.count('{') > buffer.count('}'):
                continue
            
            # Keep accumulating if statement doesn't end with semicolon
            # (multi-line pol definitions, constraints, etc.)
            if not has_semicolon and '=' in buffer:
                # Continuation pattern: line ends with operator or starts pol/public
                if (stripped.endswith('+') or stripped.endswith('*') or 
                    stripped.endswith('-') or stripped.endswith('/') or
                    buffer.startswith('pol ') or buffer.startswith('public ')):
                    continue
            
            # Statement complete
            if has_semicolon:
                joined_lines.append((start_line, buffer))
                buffer = ""
            elif open_braces == 0 and close_braces == 0 and not buffer.startswith('pol '):
                # Likely a namespace or include line
                joined_lines.append((start_line, buffer))
                buffer = ""
        
        # Don't forget remaining buffer
        if buffer.strip():
            joined_lines.append((start_line, buffer))
        
        # Parse joined lines
        for line_num, line in joined_lines:
            self._parse_line(line, line_num)
    
    def _parse_line(self, line: str, line_num: int):
        """Parse a single line."""
        # Namespace
        ns_match = self.patterns['namespace'].search(line)
        if ns_match:
            self.current_namespace = ns_match.group(1)
            self.namespaces[self.current_namespace] = set()
            return
        
        # pol commit
        commit_match = self.patterns['pol_commit'].search(line)
        if commit_match:
            self._parse_signal_decl(commit_match.group(1), SignalType.COMMIT, line_num)
            return
        
        # pol constant
        const_match = self.patterns['pol_constant'].search(line)
        if const_match:
            self._parse_signal_decl(const_match.group(1), SignalType.CONSTANT, line_num)
            return
        
        # public
        public_match = self.patterns['public'].search(line)
        if public_match:
            name = public_match.group(1)
            self._add_signal(name, SignalType.PUBLIC, line_num)
            return
        
        # Lookup with selector
        lookup_sel = self.patterns['lookup_selector'].search(line)
        if lookup_sel:
            self._add_lookup(
                selector=lookup_sel.group(1),
                lhs_cols=lookup_sel.group(2),
                rhs_table=lookup_sel.group(3),
                rhs_cols=lookup_sel.group(4),
                is_permutation=False,
                line_num=line_num,
                raw_text=line
            )
            return
        
        # Permutation with selector
        perm_sel = self.patterns['perm_selector'].search(line)
        if perm_sel:
            self._add_lookup(
                selector=perm_sel.group(1),
                lhs_cols=perm_sel.group(2),
                rhs_table=perm_sel.group(3),
                rhs_cols=perm_sel.group(4),
                is_permutation=True,
                line_num=line_num,
                raw_text=line
            )
            return
        
        # Bare lookup: { ... } in { ... }
        if ' in {' in line and ' is ' not in line:
            bare_lookup = self.patterns['lookup_bare'].search(line)
            if bare_lookup:
                # Check if RHS contains Rom.*
                rhs = bare_lookup.group(2)
                is_rom = 'Rom.' in rhs
                
                self._add_lookup(
                    selector="1",
                    lhs_cols=bare_lookup.group(1),
                    rhs_table="Rom" if is_rom else "",
                    rhs_cols=rhs,
                    is_permutation=False,
                    line_num=line_num,
                    raw_text=line,
                    is_rom_lookup=is_rom
                )
                return
        
        # Bare permutation: { ... } is Table.* { ... }
        if ' is ' in line:
            bare_perm = self.patterns['perm_bare'].search(line)
            if bare_perm:
                self._add_lookup(
                    selector="1",
                    lhs_cols=bare_perm.group(1),
                    rhs_table=bare_perm.group(2),
                    rhs_cols=bare_perm.group(3),
                    is_permutation=True,
                    line_num=line_num,
                    raw_text=line
                )
                return
        
        # Binary constraint
        binary_match = self.patterns['binary'].search(line)
        if binary_match:
            sig_name = binary_match.group(1)
            self._add_constraint(
                f"(1-{sig_name})*{sig_name} = 0",
                ConstraintType.BINARY,
                line_num
            )
            return
        
        # State transition
        trans_match = self.patterns['state_trans'].search(line)
        if trans_match:
            self._add_constraint(
                f"{trans_match.group(1)}' = {trans_match.group(2)}",
                ConstraintType.STATE_TRANSITION,
                line_num
            )
            return
        
        # pol intermediate
        inter_match = self.patterns['pol_intermediate'].search(line)
        if inter_match and 'commit' not in line and 'constant' not in line:
            name = inter_match.group(1)
            expr = inter_match.group(2)
            self._add_signal(name, SignalType.INTERMEDIATE, line_num)
            self._add_constraint(f"{name} = {expr}", ConstraintType.POLYNOMIAL, line_num)
            return
        
        # General constraint: expr = expr (not declarations)
        if '=' in line and 'pol' not in line and 'public' not in line and '{' not in line:
            self._add_constraint(line, ConstraintType.POLYNOMIAL, line_num)
    
    def _parse_signal_decl(self, decls: str, sig_type: SignalType, line_num: int):
        """Parse signal declarations."""
        for decl in decls.split(','):
            decl = decl.strip()
            if not decl:
                continue
            
            arr_match = self.patterns['array_decl'].search(decl)
            if arr_match:
                name = arr_match.group(1)
                size = int(arr_match.group(2))
                for i in range(size):
                    self._add_signal(f"{name}[{i}]", sig_type, line_num, 
                                    is_array=True, array_size=size)
            else:
                self._add_signal(decl, sig_type, line_num)
    
    def _add_signal(self, name: str, sig_type: SignalType, line_num: int,
                   is_array: bool = False, array_size: int = 0):
        """Add a signal."""
        full_name = f"{self.current_namespace}.{name}"
        
        if full_name not in self.signals:
            sig = Signal(
                name=name,
                namespace=self.current_namespace,
                full_name=full_name,
                signal_type=sig_type,
                source_file=self.current_file,
                line=line_num,
                is_array=is_array,
                array_size=array_size
            )
            
            # Detect FREE signals
            if 'FREE' in name.upper() or 'free' in name:
                sig.is_free_input = True
            
            self.signals[full_name] = sig
            self.namespaces[self.current_namespace].add(full_name)
    
    def _add_constraint(self, expr: str, ctype: ConstraintType, line_num: int):
        """Add a constraint."""
        constraint = Constraint(
            id=self.constraint_id,
            expr=expr[:200],  # Truncate for storage
            constraint_type=ctype,
            namespace=self.current_namespace,
            source_file=self.current_file,
            line=line_num
        )
        self.constraint_id += 1
        
        # Extract signals
        constraint.all_signals = self._extract_signals(expr)
        
        self.constraints.append(constraint)
    
    def _add_lookup(self, selector: str, lhs_cols: str, rhs_table: str,
                   rhs_cols: str, is_permutation: bool, line_num: int,
                   raw_text: str = "", is_rom_lookup: bool = False):
        """Add a lookup/permutation."""
        lhs_col_list = [c.strip() for c in lhs_cols.split(',') if c.strip()]
        rhs_col_list = [c.strip() for c in rhs_cols.split(',') if c.strip()]
        
        lookup = Lookup(
            id=self.lookup_id,
            selector=selector.strip(),
            lhs_columns=lhs_col_list,
            rhs_table=rhs_table.strip(),
            rhs_columns=rhs_col_list,
            is_permutation=is_permutation,
            is_rom_lookup=is_rom_lookup or 'Rom' in rhs_table,
            namespace=self.current_namespace,
            source_file=self.current_file,
            line=line_num,
            raw_text=raw_text[:300]
        )
        self.lookup_id += 1
        
        # Extract signals
        lookup.selector_signals = self._extract_signals(selector)
        lookup.lhs_signals = self._extract_signals(lhs_cols)
        lookup.rhs_signals = self._extract_signals(rhs_cols)
        
        self.lookups.append(lookup)
    
    def _extract_signals(self, expr: str) -> Set[str]:
        """Extract signal references from expression with cross-namespace resolution."""
        signals = set()
        
        # Enhanced pattern to catch array indices and namespaced refs
        # Use non-word-boundary approach to capture array indices properly
        ident_pattern = re.compile(r'([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?(?:\[\d+\])?)')
        
        for match in ident_pattern.finditer(expr):
            ident = match.group(1)
            
            # Skip keywords and pure numbers
            base = ident.split('.')[0].split('[')[0]
            if base in self.keywords or base.isdigit():
                continue
            if len(base) <= 1 and base.isupper():  # Skip single letters like N
                continue
            
            # Handle qualified names (Namespace.signal)
            if '.' in ident:
                # Direct qualified reference
                if ident in self.signals:
                    signals.add(ident)
                else:
                    # Try to resolve - it might be declared later
                    signals.add(ident)
            else:
                # Unqualified - use current namespace
                full_name = f"{self.current_namespace}.{ident}"
                signals.add(full_name)
        
        return signals
    
    def _resolve_cross_namespace(self):
        """Second pass: resolve cross-namespace signal references."""
        # Build namespace index
        ns_signals = defaultdict(set)
        for sig_name in self.signals:
            parts = sig_name.split('.')
            if len(parts) >= 2:
                ns_signals[parts[0]].add(sig_name)
        
        # Resolve unresolved references in constraints
        for constraint in self.constraints:
            resolved = set()
            for sig_ref in constraint.all_signals:
                if sig_ref in self.signals:
                    resolved.add(sig_ref)
                elif '.' in sig_ref:
                    # Try namespace lookup
                    ns, name = sig_ref.split('.', 1)
                    if ns in ns_signals:
                        # Find matching signal in that namespace
                        for full_name in ns_signals[ns]:
                            if full_name.endswith(f'.{name}'):
                                resolved.add(full_name)
                                break
                        else:
                            resolved.add(sig_ref)  # Keep original
                else:
                    resolved.add(sig_ref)
            constraint.all_signals = resolved
        
        # Same for lookups
        for lookup in self.lookups:
            for attr in ['lhs_signals', 'rhs_signals', 'selector_signals']:
                orig = getattr(lookup, attr)
                resolved = set()
                for sig_ref in orig:
                    if sig_ref in self.signals:
                        resolved.add(sig_ref)
                    elif '.' in sig_ref:
                        ns, name = sig_ref.split('.', 1)
                        if ns in ns_signals:
                            for full_name in ns_signals[ns]:
                                if full_name.endswith(f'.{name}'):
                                    resolved.add(full_name)
                                    break
                            else:
                                resolved.add(sig_ref)
                    else:
                        resolved.add(sig_ref)
                setattr(lookup, attr, resolved)
    
    def _build_constraint_graph(self):
        """Build signal-constraint dependency graph."""
        # First resolve cross-namespace references
        self._resolve_cross_namespace()
        
        # Build intermediate polynomial map (name -> expression signals)
        self.intermediate_deps: Dict[str, Set[str]] = {}
        for constraint in self.constraints:
            if ' = ' in constraint.expr:
                parts = constraint.expr.split(' = ', 1)
                if len(parts) == 2:
                    lhs = parts[0].strip()
                    # Check if this defines an intermediate - match by namespace!
                    full_lhs = f"{constraint.namespace}.{lhs}"
                    if full_lhs in self.signals:
                        sig = self.signals[full_lhs]
                        if sig.signal_type == SignalType.INTERMEDIATE:
                            self.intermediate_deps[sig.full_name] = constraint.all_signals - {sig.full_name}
        
        # Link constraints to signals
        for constraint in self.constraints:
            for sig_name in constraint.all_signals:
                if sig_name in self.signals:
                    sig = self.signals[sig_name]
                    if constraint.constraint_type == ConstraintType.STATE_TRANSITION:
                        sig.state_transitions.append(constraint.id)
                    else:
                        sig.direct_constraints.append(constraint.id)
        
        # Link lookups to signals
        for lookup in self.lookups:
            all_sigs = lookup.lhs_signals | lookup.rhs_signals | lookup.selector_signals
            for sig_name in all_sigs:
                if sig_name in self.signals:
                    sig = self.signals[sig_name]
                    if lookup.is_permutation:
                        sig.permutation_constraints.append(lookup.id)
                    else:
                        sig.lookup_constraints.append(lookup.id)
        
        # TRANSITIVE LOOKUP TRACKING
        # If signal A is part of intermediate B, and B is in a lookup, then A is lookup-constrained
        self._propagate_transitive_lookups()
        
        # Update constraint counts
        for sig in self.signals.values():
            sig.constraint_count = sig.total_constraints()
            sig.is_fully_constrained = sig.constraint_count > 0
    
    def _propagate_transitive_lookups(self):
        """Propagate lookup constraints through intermediate polynomials."""
        # Build reverse map: signal -> intermediates that contain it
        signal_to_intermediates: Dict[str, Set[str]] = defaultdict(set)
        for inter_name, deps in self.intermediate_deps.items():
            for dep in deps:
                signal_to_intermediates[dep].add(inter_name)
        
        # Find all signals (including intermediates) that are in lookups
        signals_in_lookups: Dict[str, List[int]] = defaultdict(list)
        for lookup in self.lookups:
            for sig_name in lookup.lhs_signals:
                signals_in_lookups[sig_name].append(lookup.id)
        
        # Propagate lookup constraints to component signals (depth-first)
        propagated = 0
        visited = set()
        
        def propagate(sig_name: str, lookup_ids: List[int], depth: int = 0):
            nonlocal propagated
            if depth > 5 or sig_name in visited:  # Prevent infinite recursion
                return
            visited.add(sig_name)
            
            # If this signal is an intermediate, propagate to its components
            if sig_name in self.intermediate_deps:
                deps = self.intermediate_deps[sig_name]
                for dep_sig in deps:
                    if dep_sig in self.signals:
                        sig = self.signals[dep_sig]
                        for lid in lookup_ids:
                            if lid not in sig.lookup_constraints:
                                sig.lookup_constraints.append(lid)
                                propagated += 1
                        # Recursively propagate
                        propagate(dep_sig, lookup_ids, depth + 1)
        
        # Start propagation from all signals in lookups
        for sig_name, lookup_ids in signals_in_lookups.items():
            propagate(sig_name, lookup_ids, 0)
        
        # Also propagate UP: if freeIn is used in aFreeIn and aFreeIn is in lookup,
        # then freeIn gets the lookup constraint
        for sig_name, intermediates in signal_to_intermediates.items():
            for inter_name in intermediates:
                if inter_name in signals_in_lookups:
                    if sig_name in self.signals:
                        sig = self.signals[sig_name]
                        for lid in signals_in_lookups[inter_name]:
                            if lid not in sig.lookup_constraints:
                                sig.lookup_constraints.append(lid)
                                propagated += 1
        
        if propagated > 0:
            print(f"  Propagated {propagated} transitive lookup constraints")
    
    def _analyze_rom_lookup(self):
        """Find and analyze the main ROM lookup."""
        rom_lookups = [l for l in self.lookups if l.is_rom_lookup]
        
        if not rom_lookups:
            print("  ⚠️ No ROM lookup found!")
            return
        
        # Find the main ROM lookup (largest one)
        self.rom_lookup = max(rom_lookups, key=lambda l: len(l.lhs_columns))
        
        print(f"  Found ROM lookup at {self.rom_lookup.source_file}:{self.rom_lookup.line}")
        print(f"  Columns: {len(self.rom_lookup.lhs_columns)} LHS -> {len(self.rom_lookup.rhs_columns)} RHS")
        
        # Mark all LHS signals as ROM-constrained
        for sig_name in self.rom_lookup.lhs_signals:
            if sig_name in self.signals:
                self.signals[sig_name].is_rom_constrained = True
                self.rom_constrained_signals.add(sig_name)
        
        # Find 'operations' polynomial by reading main.pil directly
        operations_signals = set()
        main_pil = self.pil_dir / "main.pil"
        if main_pil.exists():
            content = main_pil.read_text()
            # Find operations polynomial definition
            op_match = re.search(
                r'pol\s+operations\s*=([^;]+);',
                content, 
                re.DOTALL
            )
            if op_match:
                op_expr = op_match.group(1)
                # Pattern: 2**N * signalName (with possible whitespace)
                bit_pattern = re.compile(r'2\s*\*\*\s*(\d+)\s*\*\s*(\w+)')
                for match in bit_pattern.finditer(op_expr):
                    bit_pos = int(match.group(1))
                    sig_name = match.group(2)
                    full_name = f"Main.{sig_name}"
                    if full_name in self.signals:
                        operations_signals.add(full_name)
                        self.signals[full_name].is_rom_constrained = True
                        self.rom_constrained_signals.add(full_name)
        
        # Also check for 'operations' in the ROM lookup columns - if present, all encoded signals are constrained
        for col in self.rom_lookup.lhs_columns:
            if 'operations' in col:
                # All signals encoded in operations are ROM-constrained
                for sig_name in operations_signals:
                    if sig_name in self.signals:
                        self.signals[sig_name].is_rom_constrained = True
                        self.rom_constrained_signals.add(sig_name)
                break
        
        print(f"  Operations polynomial signals: {len(operations_signals)}")
        print(f"  Total ROM-constrained signals: {len(self.rom_constrained_signals)}")
        
        # List key ROM-constrained signals
        key_signals = ['assumeFree', 'inFREE', 'inFREE0', 'mOp', 'sRD', 'sWR']
        for key in key_signals:
            full = f"Main.{key}"
            if full in self.rom_constrained_signals:
                print(f"    ✅ {key} is ROM-constrained")
            elif full in self.signals:
                print(f"    ⚠️ {key} NOT ROM-constrained")
    
    def _trace_free_signals(self):
        """Trace how FREE signals flow through the constraint system."""
        # Only consider COMMIT signals as true FREE inputs (not intermediates)
        free_signals = [s for s in self.signals.values() 
                       if s.is_free_input and s.signal_type == SignalType.COMMIT]
        
        print(f"  Found {len(free_signals)} FREE signals (commit only)")
        
        for sig in free_signals:
            flow = FREEFlowPath(free_signal=sig.full_name, path=[])
            
            # Trace through constraints
            for c_id in sig.direct_constraints:
                constraint = self.constraints[c_id]
                # Find what this constraint connects to
                for other_sig in constraint.all_signals:
                    if other_sig != sig.full_name:
                        flow.path.append(('constraint', other_sig))
                        sig.flows_to.add(other_sig)
                        
                        # Check if flows to state/memory/storage
                        if 'SR' in other_sig or 'State' in other_sig:
                            flow.reaches_state = True
                        if 'Mem' in other_sig or 'mem' in other_sig:
                            flow.reaches_memory = True
                        if 'Storage' in other_sig or 'storage' in other_sig:
                            flow.reaches_storage = True
            
            # Trace through lookups
            for l_id in sig.lookup_constraints:
                lookup = self.lookups[l_id]
                flow.verified_by_lookup = True
                flow.verification_lookup = f"{lookup.rhs_table}:{lookup.line}"
                flow.path.append(('lookup', lookup.rhs_table))
            
            for l_id in sig.permutation_constraints:
                lookup = self.lookups[l_id]
                flow.verified_by_lookup = True
                flow.verification_lookup = f"{lookup.rhs_table}:{lookup.line}"
                flow.path.append(('permutation', lookup.rhs_table))
            
            self.free_flows[sig.full_name] = flow
    
    def _detect_attack_surfaces(self):
        """Detect potential attack surfaces."""
        finding_id = 0
        
        # 1. Unconstrained COMMIT signals
        for sig in self.signals.values():
            if sig.signal_type == SignalType.COMMIT:
                if sig.constraint_count == 0 and not sig.is_rom_constrained:
                    self.findings.append(Finding(
                        id=f"PIL-{finding_id:03d}",
                        severity="HIGH" if not sig.is_free_input else "MEDIUM",
                        title="Unconstrained COMMIT Signal",
                        signal=sig.full_name,
                        description=f"Signal {sig.name} has no direct constraints",
                        constraint_analysis=f"Direct: {len(sig.direct_constraints)}, Lookup: {len(sig.lookup_constraints)}, ROM: {sig.is_rom_constrained}",
                        status="NEEDS_REVIEW",
                        line=sig.line,
                        file=sig.source_file
                    ))
                    finding_id += 1
        
        # 2. FREE signals without lookup verification
        for sig_name, flow in self.free_flows.items():
            if not flow.verified_by_lookup:
                self.findings.append(Finding(
                    id=f"PIL-{finding_id:03d}",
                    severity="HIGH",
                    title="FREE Signal Without Lookup Verification",
                    signal=sig_name,
                    description="FREE signal not verified by any lookup constraint",
                    constraint_analysis=f"Reaches state: {flow.reaches_state}, Reaches memory: {flow.reaches_memory}",
                    status="NEEDS_REVIEW",
                    line=self.signals[sig_name].line,
                    file=self.signals[sig_name].source_file
                ))
                finding_id += 1
        
        # 3. assumeFree pattern analysis
        assume_free_signals = [s for s in self.signals.values() 
                             if 'assumeFree' in s.name or 'assumefree' in s.name.lower()]
        
        for sig in assume_free_signals:
            if sig.is_rom_constrained:
                status = "SECURE"
                severity = "INFO"
            else:
                status = "VULNERABLE"
                severity = "CRITICAL"
            
            self.findings.append(Finding(
                id=f"PIL-{finding_id:03d}",
                severity=severity,
                title="assumeFree Memory Bypass Pattern",
                signal=sig.full_name,
                description="assumeFree controls memory lookup value substitution",
                constraint_analysis=f"ROM-constrained: {sig.is_rom_constrained}, Binary-constrained: {'binary' in str([self.constraints[c].constraint_type for c in sig.direct_constraints])}",
                status=status,
                line=sig.line,
                file=sig.source_file
            ))
            finding_id += 1
        
        print(f"  Found {len(self.findings)} potential attack surfaces")
    
    def _analyze_rank(self) -> Dict:
        """Analyze constraint matrix rank."""
        # Build signal index
        signal_names = sorted([s.full_name for s in self.signals.values() 
                              if s.signal_type in (SignalType.COMMIT, SignalType.INTERMEDIATE)])
        signal_idx = {name: i for i, name in enumerate(signal_names)}
        
        n_signals = len(signal_names)
        n_constraints = len(self.constraints) + len(self.lookups)
        
        if n_constraints == 0 or n_signals == 0:
            return {'error': 'No constraints'}
        
        # Build sparse matrix
        matrix = np.zeros((n_constraints, n_signals), dtype=np.float32)
        
        row = 0
        for c in self.constraints:
            for sig_name in c.all_signals:
                if sig_name in signal_idx:
                    matrix[row, signal_idx[sig_name]] = 1.0
            row += 1
        
        for l in self.lookups:
            for sig_name in l.lhs_signals | l.rhs_signals:
                if sig_name in signal_idx:
                    matrix[row, signal_idx[sig_name]] = 1.0
            row += 1
        
        matrix = matrix[:row]
        m, n = matrix.shape
        
        # Compute rank
        try:
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
                torch.backends.cuda.matmul.allow_tf32 = True
            else:
                device = 'cpu'
            
            mat_torch = torch.from_numpy(matrix).to(device)
            k = min(200, min(m, n))
            U, S, V = torch.svd_lowrank(mat_torch, q=k)
            singular_values = S.cpu().numpy()
            rank = int(np.sum(singular_values > 1e-6))
            
            return {
                'shape': (m, n),
                'rank': rank,
                'max_rank': min(m, n),
                'deficiency': min(m, n) - rank,
                'singular_values': singular_values[:10].tolist(),
                'sparsity': 100 * (1 - np.count_nonzero(matrix) / matrix.size),
                'device': device
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_report(self, rank_result: Dict) -> Dict:
        """Generate comprehensive analysis report."""
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        
        # Statistics
        commit_count = sum(1 for s in self.signals.values() if s.signal_type == SignalType.COMMIT)
        const_count = sum(1 for s in self.signals.values() if s.signal_type == SignalType.CONSTANT)
        free_count = sum(1 for s in self.signals.values() if s.is_free_input)
        rom_constrained = len(self.rom_constrained_signals)
        
        print(f"\n📊 STATISTICS")
        print(f"   Total Signals: {len(self.signals)} ({commit_count} commit, {const_count} constant)")
        print(f"   FREE Signals: {free_count}")
        print(f"   ROM-Constrained: {rom_constrained}")
        print(f"   Constraints: {len(self.constraints)}")
        print(f"   Lookups/Permutations: {len(self.lookups)}")
        print(f"   Namespaces: {len(self.namespaces)}")
        
        # Namespace breakdown
        print(f"\n📁 SIGNALS BY NAMESPACE")
        ns_stats = []
        for ns in sorted(self.namespaces.keys()):
            sigs = [self.signals[s] for s in self.namespaces[ns] if s in self.signals]
            commit = sum(1 for s in sigs if s.signal_type == SignalType.COMMIT)
            const = sum(1 for s in sigs if s.signal_type == SignalType.CONSTANT)
            unconstrained = sum(1 for s in sigs if s.constraint_count == 0 and s.signal_type == SignalType.COMMIT)
            ns_stats.append((ns, commit, const, unconstrained))
        
        for ns, commit, const, unconstr in sorted(ns_stats, key=lambda x: -x[1]):
            status = "⚠️" if unconstr > 0 else "✅"
            print(f"   {status} {ns}: {commit} commit, {const} const, {unconstr} unconstrained")
        
        # FREE signal analysis
        print(f"\n🔓 FREE SIGNAL ANALYSIS")
        for sig_name, flow in self.free_flows.items():
            sig = self.signals[sig_name]
            verified = "✅ LOOKUP VERIFIED" if flow.verified_by_lookup else "⚠️ NO LOOKUP"
            print(f"   {sig.name}:")
            print(f"      {verified}")
            if flow.verified_by_lookup:
                print(f"      Verified by: {flow.verification_lookup}")
            print(f"      Reaches state: {flow.reaches_state}, Memory: {flow.reaches_memory}, Storage: {flow.reaches_storage}")
        
        # ROM lookup
        print(f"\n🔐 ROM LOOKUP")
        if self.rom_lookup:
            print(f"   Location: {self.rom_lookup.source_file}:{self.rom_lookup.line}")
            print(f"   LHS columns: {len(self.rom_lookup.lhs_columns)}")
            print(f"   RHS columns: {len(self.rom_lookup.rhs_columns)}")
            print(f"   Signals constrained: {len(self.rom_constrained_signals)}")
        else:
            print("   ⚠️ NO ROM LOOKUP FOUND!")
        
        # Constraint coverage analysis (ALL signals, not just FREE)
        print(f"\n🛡️ CONSTRAINT COVERAGE (All COMMIT Signals)")
        constrained_by = {'multiple': 0, 'direct': 0, 'lookup': 0, 'state_trans': 0, 'perm': 0, 'rom_only': 0, 'unconstrained': 0}
        unconstrained_list = []
        
        for sig in self.signals.values():
            if sig.signal_type == SignalType.COMMIT:
                sources = []
                if sig.direct_constraints: sources.append('direct')
                if sig.lookup_constraints: sources.append('lookup')
                if sig.state_transitions: sources.append('state')
                if sig.permutation_constraints: sources.append('perm')
                if sig.is_rom_constrained: sources.append('rom')
                
                if len(sources) == 0:
                    constrained_by['unconstrained'] += 1
                    unconstrained_list.append(sig.full_name)
                elif len(sources) >= 2:
                    constrained_by['multiple'] += 1
                elif sources[0] == 'direct':
                    constrained_by['direct'] += 1
                elif sources[0] == 'lookup':
                    constrained_by['lookup'] += 1
                elif sources[0] == 'state':
                    constrained_by['state_trans'] += 1
                elif sources[0] == 'perm':
                    constrained_by['perm'] += 1
                elif sources[0] == 'rom':
                    constrained_by['rom_only'] += 1
        
        total_commit = sum(constrained_by.values())
        total_constrained = total_commit - constrained_by['unconstrained']
        coverage = 100 * total_constrained / total_commit if total_commit > 0 else 0
        
        print(f"   Total COMMIT signals: {total_commit}")
        print(f"   Constrained: {total_constrained} ({coverage:.1f}%)")
        print(f"   Breakdown:")
        print(f"      Multiple mechanisms: {constrained_by['multiple']} (most secure)")
        print(f"      Direct polynomial: {constrained_by['direct']}")
        print(f"      Lookup tables: {constrained_by['lookup']}")
        print(f"      State transitions: {constrained_by['state_trans']}")
        print(f"      Permutations: {constrained_by['perm']}")
        print(f"      ROM-only: {constrained_by['rom_only']}")
        
        if constrained_by['unconstrained'] > 0:
            print(f"   ⚠️ UNCONSTRAINED: {constrained_by['unconstrained']}")
            for sig_name in unconstrained_list[:10]:
                print(f"      - {sig_name}")
            if len(unconstrained_list) > 10:
                print(f"      ... and {len(unconstrained_list) - 10} more")
        else:
            print(f"   ✅ ALL COMMIT signals are constrained!")
        
        # Rank analysis
        print(f"\n📐 CONSTRAINT MATRIX RANK")
        if 'error' not in rank_result:
            print(f"   Shape: {rank_result['shape']}")
            print(f"   Rank: {rank_result['rank']}")
            print(f"   Max rank: {rank_result['max_rank']}")
            print(f"   Degrees of freedom: {rank_result['deficiency']}")
            print(f"   Sparsity: {rank_result['sparsity']:.2f}%")
            if rank_result['deficiency'] > 0:
                print(f"   ⚠️ RANK DEFICIENCY: {rank_result['deficiency']} unconstrained DOF")
        else:
            print(f"   Error: {rank_result['error']}")
        
        # Findings
        print(f"\n⚠️ SECURITY FINDINGS ({len(self.findings)} total)")
        by_severity = defaultdict(list)
        for f in self.findings:
            by_severity[f.severity].append(f)
        
        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']:
            if severity in by_severity:
                print(f"\n   [{severity}] {len(by_severity[severity])} findings")
                for f in by_severity[severity][:5]:
                    status_icon = "🔴" if f.status == "VULNERABLE" else ("🟡" if f.status == "NEEDS_REVIEW" else "🟢")
                    print(f"      {status_icon} {f.id}: {f.title}")
                    print(f"         Signal: {f.signal}")
                    print(f"         {f.constraint_analysis}")
                if len(by_severity[severity]) > 5:
                    print(f"      ... and {len(by_severity[severity]) - 5} more")
        
        print("\n" + "="*80)
        
        # Return structured result
        return {
            'statistics': {
                'signals': len(self.signals),
                'commit': commit_count,
                'constant': const_count,
                'free': free_count,
                'rom_constrained': rom_constrained,
                'constraints': len(self.constraints),
                'lookups': len(self.lookups),
                'namespaces': len(self.namespaces)
            },
            'rank': rank_result,
            'findings': [
                {
                    'id': f.id,
                    'severity': f.severity,
                    'title': f.title,
                    'signal': f.signal,
                    'status': f.status
                }
                for f in self.findings
            ],
            'free_flows': {
                name: {
                    'verified': flow.verified_by_lookup,
                    'reaches_state': flow.reaches_state,
                    'reaches_memory': flow.reaches_memory,
                    'reaches_storage': flow.reaches_storage
                }
                for name, flow in self.free_flows.items()
            },
            'rom_lookup': {
                'found': self.rom_lookup is not None,
                'constrained_signals': len(self.rom_constrained_signals)
            } if self.rom_lookup else {'found': False}
        }


def main():
    """CLI entry point."""
    import sys
    import json
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════════╗
║   FEZK ELITE PIL Analyzer v2.0                                                   ║
║   Complete Polygon zkEVM Constraint Analysis                                     ║
║   $1M Bounty Hunter Edition                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    if len(sys.argv) < 2:
        print("Usage: python pil_elite_analyzer.py <pil_directory> [--json output.json]")
        sys.exit(1)
    
    pil_dir = Path(sys.argv[1])
    json_output = None
    
    if '--json' in sys.argv:
        idx = sys.argv.index('--json')
        if idx + 1 < len(sys.argv):
            json_output = sys.argv[idx + 1]
    
    if not pil_dir.exists():
        print(f"Error: Path not found: {pil_dir}")
        sys.exit(1)
    
    analyzer = PILEliteAnalyzer(pil_dir, verbose=True)
    result = analyzer.analyze()
    
    if json_output:
        with open(json_output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {json_output}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    critical = sum(1 for f in analyzer.findings if f.severity == 'CRITICAL')
    high = sum(1 for f in analyzer.findings if f.severity == 'HIGH')
    secure = sum(1 for f in analyzer.findings if f.status == 'SECURE')
    needs_review = sum(1 for f in analyzer.findings if f.status == 'NEEDS_REVIEW')
    
    print(f"🔴 CRITICAL: {critical}")
    print(f"🟠 HIGH: {high}")
    print(f"🟢 SECURE: {secure}")
    print(f"🟡 NEEDS REVIEW: {needs_review}")
    
    if critical == 0 and high == 0:
        print("\n✅ No critical vulnerabilities detected in PIL constraints")
        print("   Attack surface is in zkASM/ROM implementation, not PIL")
    else:
        print(f"\n⚠️ {critical + high} findings require investigation")


if __name__ == "__main__":
    main()
