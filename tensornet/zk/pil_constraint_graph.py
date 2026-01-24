#!/usr/bin/env python3
"""
FEZK ELITE PIL Constraint Graph Analyzer v1.0
==============================================
Deep analysis of Polygon zkEVM PIL constraints with:
- Cross-namespace signal dependency tracking
- Lookup/permutation constraint graph building
- assumeFree attack surface analysis
- GPU-accelerated rank deficiency detection

This is the "main.pil Beast" analyzer.

Author: FEZK Elite Team
Date: January 23, 2026
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
import numpy as np


@dataclass
class Signal:
    """A PIL signal with full metadata."""
    name: str
    namespace: str
    full_name: str
    is_commit: bool = False      # Prover-controlled
    is_constant: bool = False    # Fixed in ROM/table
    is_intermediate: bool = False # pol x = expr
    source_file: str = ""
    line: int = 0
    
    # Constraint tracking
    constrained_by: Set[int] = field(default_factory=set)
    constrains: Set[str] = field(default_factory=set)  # Other signals this affects
    in_lookups: List[int] = field(default_factory=list)
    in_permutations: List[int] = field(default_factory=list)
    
    # Analysis flags
    attack_surface: bool = False
    attack_reason: str = ""


@dataclass
class Constraint:
    """A PIL constraint with dependency tracking."""
    id: int
    expr: str
    namespace: str
    source_file: str
    line: int
    
    # Type flags
    is_binary: bool = False        # (1-x)*x = 0
    is_state_transition: bool = False  # x' = expr
    is_polynomial: bool = False    # expr = 0
    
    # Signal dependencies
    lhs_signals: Set[str] = field(default_factory=set)
    rhs_signals: Set[str] = field(default_factory=set)
    all_signals: Set[str] = field(default_factory=set)


@dataclass
class Lookup:
    """A lookup or permutation constraint."""
    id: int
    selector: str
    lhs_columns: List[str]
    rhs_table: str
    rhs_columns: List[str]
    is_permutation: bool  # True for 'is', False for 'in'
    namespace: str
    source_file: str
    line: int
    
    # Analysis
    lhs_signals: Set[str] = field(default_factory=set)
    rhs_signals: Set[str] = field(default_factory=set)


class PILConstraintGraph:
    """
    Deep analysis of PIL constraint system.
    
    Builds a complete dependency graph including:
    - Direct polynomial constraints
    - Lookup table constraints
    - Permutation arguments
    - State transitions
    - Cross-namespace dependencies
    """
    
    def __init__(self, pil_dir: Path):
        self.pil_dir = pil_dir
        self.signals: Dict[str, Signal] = {}
        self.constraints: List[Constraint] = []
        self.lookups: List[Lookup] = []
        self.namespaces: Set[str] = set()
        
        self.current_namespace = "Global"
        self.current_file = ""
        self.constraint_id = 0
        self.lookup_id = 0
        
        # Attack surface tracking
        self.attack_surfaces: List[Dict] = []
        
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for PIL parsing."""
        self.patterns = {
            'namespace': re.compile(r'namespace\s+(\w+)\s*\('),
            'pol_commit': re.compile(r'pol\s+commit\s+([^;]+);'),
            'pol_constant': re.compile(r'pol\s+constant\s+([^;]+);'),
            'pol_def': re.compile(r'^\s*pol\s+(\w+)\s*=\s*([^;]+);', re.MULTILINE),
            'array_decl': re.compile(r'(\w+)\[(\d+)\]'),
            'binary_constraint': re.compile(r'\(\s*1\s*-\s*(\w+)\s*\)\s*\*\s*\1'),
            'state_transition': re.compile(r"(\w+(?:\[\d+\])?)\s*'\s*="),
            'signal_ref': re.compile(r'([A-Za-z_][A-Za-z0-9_]*(?:\.\w+)?(?:\[\d+\])?)'),
            'lookup_in': re.compile(
                r'(\{[^}]+\})\s*in\s*(?:(\w+(?:\.\w+)?)\s*)?\{([^}]+)\}',
                re.DOTALL
            ),
            'permutation_is': re.compile(
                r'(\{[^}]+\})\s*is\s*(\w+(?:\.\w+)?)\s*\{([^}]+)\}',
                re.DOTALL
            ),
            'selector_lookup': re.compile(
                r'(\w+(?:\s*\+\s*\w+)*(?:\s*\*\s*\w+)*)\s*\{([^}]+)\}\s*(in|is)\s*(\w+(?:\.\w+)?)\s*\{([^}]+)\}',
                re.DOTALL
            ),
        }
        
        self.keywords = {
            'pol', 'commit', 'constant', 'namespace', 'include', 'in', 'is',
            'public', 'constant', 'if', 'else', 'for'
        }
    
    def parse_all(self) -> None:
        """Parse all PIL files in directory."""
        pil_files = sorted(self.pil_dir.glob("*.pil"))
        
        for pil_file in pil_files:
            self._parse_file(pil_file)
        
        # Build cross-references
        self._build_cross_refs()
    
    def _parse_file(self, filepath: Path) -> None:
        """Parse a single PIL file."""
        content = filepath.read_text()
        self.current_file = filepath.name
        self.current_namespace = "Global"
        
        # Remove comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        lines = content.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            i += 1
            
            if not line or line.startswith('//'):
                continue
            
            # Multi-line constructs
            while ('{' in line and line.count('{') > line.count('}') and i < len(lines)):
                next_line = lines[i].strip()
                if not next_line.startswith('//'):
                    line += ' ' + next_line
                i += 1
            
            self._parse_line(line, i)
    
    def _parse_line(self, line: str, line_num: int) -> None:
        """Parse a single line of PIL."""
        # Strip inline comments
        if '//' in line:
            line = line.split('//')[0].strip()
        if not line:
            return
        
        # Namespace
        ns_match = self.patterns['namespace'].search(line)
        if ns_match:
            self.current_namespace = ns_match.group(1)
            self.namespaces.add(self.current_namespace)
            return
        
        # pol commit
        commit_match = self.patterns['pol_commit'].search(line)
        if commit_match:
            self._parse_signals(commit_match.group(1), is_commit=True, line_num=line_num)
            return
        
        # pol constant  
        const_match = self.patterns['pol_constant'].search(line)
        if const_match:
            self._parse_signals(const_match.group(1), is_constant=True, line_num=line_num)
            return
        
        # Selector lookup/permutation: selector { ... } in/is Table { ... }
        sel_lookup = self.patterns['selector_lookup'].search(line)
        if sel_lookup:
            selector = sel_lookup.group(1)
            lhs_cols = sel_lookup.group(2)
            lookup_type = sel_lookup.group(3)  # 'in' or 'is'
            rhs_table = sel_lookup.group(4)
            rhs_cols = sel_lookup.group(5)
            
            self._add_lookup(
                selector=selector,
                lhs_cols=lhs_cols,
                rhs_table=rhs_table,
                rhs_cols=rhs_cols,
                is_permutation=(lookup_type == 'is'),
                line_num=line_num
            )
            return
        
        # Lookups: { ... } in Table { ... } or { ... } in { ... }
        lookup_match = self.patterns['lookup_in'].search(line)
        if lookup_match and ' is ' not in line:
            # Detect if this is a ROM lookup based on rhs containing Rom.*
            rhs_cols = lookup_match.group(3) if len(lookup_match.groups()) >= 3 else ""
            rhs_table = lookup_match.group(2) if len(lookup_match.groups()) >= 2 and lookup_match.group(2) else ""
            
            # Check if ROM is mentioned in the RHS
            if 'Rom.' in rhs_cols or 'Rom.' in str(rhs_table):
                rhs_table = "Rom"
            
            self._add_lookup(
                selector="1",
                lhs_cols=lookup_match.group(1),
                rhs_table=rhs_table,
                rhs_cols=rhs_cols,
                is_permutation=False,
                line_num=line_num
            )
            return
        
        # Permutations: { ... } is Table { ... }
        perm_match = self.patterns['permutation_is'].search(line)
        if perm_match:
            self._add_lookup(
                selector="1",
                lhs_cols=perm_match.group(1),
                rhs_table=perm_match.group(2),
                rhs_cols=perm_match.group(3),
                is_permutation=True,
                line_num=line_num
            )
            return
        
        # Binary constraint: (1-x)*x = 0
        if self.patterns['binary_constraint'].search(line):
            self._add_constraint(line, is_binary=True, line_num=line_num)
            return
        
        # State transition: x' = expr
        if self.patterns['state_transition'].search(line):
            self._add_constraint(line, is_state_transition=True, line_num=line_num)
            return
        
        # Polynomial identity: expr = expr (but not declarations)
        if '=' in line and 'pol' not in line and 'public' not in line:
            if '{' not in line and '}' not in line:
                self._add_constraint(line, is_polynomial=True, line_num=line_num)
    
    def _parse_signals(self, decls: str, is_commit: bool = False, 
                      is_constant: bool = False, line_num: int = 0) -> None:
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
                    self._add_signal(f"{name}[{i}]", is_commit, is_constant, line_num)
            else:
                self._add_signal(decl, is_commit, is_constant, line_num)
    
    def _add_signal(self, name: str, is_commit: bool, is_constant: bool, line_num: int) -> None:
        """Add a signal to the registry."""
        full_name = f"{self.current_namespace}.{name}"
        
        if full_name not in self.signals:
            self.signals[full_name] = Signal(
                name=name,
                namespace=self.current_namespace,
                full_name=full_name,
                is_commit=is_commit,
                is_constant=is_constant,
                source_file=self.current_file,
                line=line_num
            )
    
    def _add_constraint(self, expr: str, is_binary: bool = False,
                       is_state_transition: bool = False, 
                       is_polynomial: bool = False,
                       line_num: int = 0) -> None:
        """Add a constraint."""
        constraint = Constraint(
            id=self.constraint_id,
            expr=expr,
            namespace=self.current_namespace,
            source_file=self.current_file,
            line=line_num,
            is_binary=is_binary,
            is_state_transition=is_state_transition,
            is_polynomial=is_polynomial
        )
        self.constraint_id += 1
        
        # Extract signals
        constraint.all_signals = self._extract_signals(expr)
        
        self.constraints.append(constraint)
    
    def _add_lookup(self, selector: str, lhs_cols: str, rhs_table: str,
                   rhs_cols: str, is_permutation: bool, line_num: int) -> None:
        """Add a lookup/permutation constraint."""
        # Clean up columns
        lhs_cols = lhs_cols.strip('{}').strip()
        rhs_cols = rhs_cols.strip('{}').strip()
        
        lookup = Lookup(
            id=self.lookup_id,
            selector=selector.strip(),
            lhs_columns=[c.strip() for c in lhs_cols.split(',') if c.strip()],
            rhs_table=rhs_table.strip() if rhs_table else "",
            rhs_columns=[c.strip() for c in rhs_cols.split(',') if c.strip()],
            is_permutation=is_permutation,
            namespace=self.current_namespace,
            source_file=self.current_file,
            line=line_num
        )
        self.lookup_id += 1
        
        # Extract signals
        lookup.lhs_signals = self._extract_signals(lhs_cols)
        lookup.rhs_signals = self._extract_signals(rhs_cols)
        
        self.lookups.append(lookup)
    
    def _extract_signals(self, expr: str) -> Set[str]:
        """Extract signal references from expression."""
        signals = set()
        
        for match in self.patterns['signal_ref'].finditer(expr):
            ident = match.group(1)
            
            # Skip keywords and pure numbers
            base_name = ident.split('.')[0].split('[')[0]
            if base_name in self.keywords or base_name.isdigit():
                continue
            
            # Handle qualified names (Namespace.signal)
            if '.' in ident:
                if ident in self.signals:
                    signals.add(ident)
                else:
                    # Try resolving
                    signals.add(ident)
            else:
                # Try current namespace
                full_name = f"{self.current_namespace}.{ident}"
                if full_name in self.signals:
                    signals.add(full_name)
                elif ident in self.signals:
                    signals.add(ident)
                else:
                    # Might be a constant or cross-namespace
                    signals.add(f"?.{ident}")
        
        return signals
    
    def _build_cross_refs(self) -> None:
        """Build cross-references between signals and constraints."""
        # Link constraints to signals
        for constraint in self.constraints:
            for sig_name in constraint.all_signals:
                if sig_name in self.signals:
                    self.signals[sig_name].constrained_by.add(constraint.id)
        
        # Link lookups to signals
        for lookup in self.lookups:
            for sig_name in lookup.lhs_signals | lookup.rhs_signals:
                if sig_name in self.signals:
                    if lookup.is_permutation:
                        self.signals[sig_name].in_permutations.append(lookup.id)
                    else:
                        self.signals[sig_name].in_lookups.append(lookup.id)
    
    def analyze_attack_surfaces(self) -> List[Dict]:
        """Find potential attack surfaces in the constraint system."""
        attack_surfaces = []
        
        # 1. Find unconstrained COMMIT signals
        for sig in self.signals.values():
            if sig.is_commit:
                total_constraints = (len(sig.constrained_by) + 
                                   len(sig.in_lookups) + 
                                   len(sig.in_permutations))
                
                if total_constraints == 0:
                    attack_surfaces.append({
                        'type': 'UNCONSTRAINED_COMMIT',
                        'signal': sig.full_name,
                        'severity': 'CRITICAL',
                        'file': sig.source_file,
                        'line': sig.line,
                        'reason': 'COMMIT signal with no constraints - prover can set any value'
                    })
                    sig.attack_surface = True
                    sig.attack_reason = 'Unconstrained'
        
        # 2. Analyze assumeFree pattern (Polygon-specific)
        assume_free_signals = [s for s in self.signals.values() 
                              if 'assumeFree' in s.name.lower() or 'assume' in s.name.lower()]
        
        for sig in assume_free_signals:
            if sig.is_commit:
                attack_surfaces.append({
                    'type': 'ASSUME_FREE_PATTERN',
                    'signal': sig.full_name,
                    'severity': 'HIGH',
                    'file': sig.source_file,
                    'line': sig.line,
                    'reason': 'assumeFree signal - controls memory access bypass',
                    'constraint_count': len(sig.constrained_by),
                    'lookup_count': len(sig.in_lookups)
                })
        
        # 3. Find FREE signals with weak constraints
        free_signals = [s for s in self.signals.values() 
                       if 'FREE' in s.name and s.is_commit]
        
        for sig in free_signals:
            # Check if constrained only by weak operations
            if len(sig.constrained_by) < 3 and len(sig.in_lookups) == 0:
                attack_surfaces.append({
                    'type': 'WEAK_FREE_CONSTRAINT',
                    'signal': sig.full_name,
                    'severity': 'MEDIUM',
                    'file': sig.source_file,
                    'line': sig.line,
                    'reason': f'FREE signal with only {len(sig.constrained_by)} direct constraints',
                    'constraints': list(sig.constrained_by)[:5]
                })
        
        # 4. Check lookup column mismatches
        for lookup in self.lookups:
            if len(lookup.lhs_columns) != len(lookup.rhs_columns):
                attack_surfaces.append({
                    'type': 'LOOKUP_COLUMN_MISMATCH',
                    'lookup_id': lookup.id,
                    'severity': 'CRITICAL',
                    'file': lookup.source_file,
                    'line': lookup.line,
                    'reason': f'LHS has {len(lookup.lhs_columns)} cols, RHS has {len(lookup.rhs_columns)} cols',
                    'lhs': lookup.lhs_columns[:3],
                    'rhs': lookup.rhs_columns[:3]
                })
        
        self.attack_surfaces = attack_surfaces
        return attack_surfaces
    
    def get_rom_lookup_analysis(self) -> Dict:
        """Analyze the main ROM lookup constraint."""
        rom_lookups = [l for l in self.lookups 
                      if 'Rom' in l.rhs_table or 'rom' in l.rhs_table.lower()]
        
        if not rom_lookups:
            return {'found': False}
        
        rom_lookup = rom_lookups[0]  # Main ROM lookup
        
        return {
            'found': True,
            'lookup_id': rom_lookup.id,
            'file': rom_lookup.source_file,
            'line': rom_lookup.line,
            'lhs_columns': len(rom_lookup.lhs_columns),
            'rhs_columns': len(rom_lookup.rhs_columns),
            'is_permutation': rom_lookup.is_permutation,
            'constrained_signals': list(rom_lookup.lhs_signals)[:20]
        }
    
    def build_constraint_matrix(self) -> Tuple[np.ndarray, List[str], List[int]]:
        """Build constraint coefficient matrix for rank analysis."""
        # Get ordered signal list
        signal_names = sorted(self.signals.keys())
        signal_idx = {name: i for i, name in enumerate(signal_names)}
        
        n_signals = len(signal_names)
        n_constraints = len(self.constraints) + len(self.lookups)
        
        if n_constraints == 0 or n_signals == 0:
            return np.array([]), [], []
        
        matrix = np.zeros((n_constraints, n_signals), dtype=np.float32)
        constraint_ids = []
        
        row = 0
        
        # Add polynomial constraints
        for c in self.constraints:
            for sig_name in c.all_signals:
                if sig_name in signal_idx:
                    matrix[row, signal_idx[sig_name]] = 1.0
            constraint_ids.append(f"C{c.id}")
            row += 1
        
        # Add lookup constraints (each lookup column adds effective constraints)
        for l in self.lookups:
            for sig_name in l.lhs_signals | l.rhs_signals:
                if sig_name in signal_idx:
                    matrix[row, signal_idx[sig_name]] = 1.0
            constraint_ids.append(f"L{l.id}")
            row += 1
        
        return matrix[:row], signal_names, constraint_ids
    
    def analyze_rank(self, use_gpu: bool = True) -> Dict:
        """Analyze constraint matrix rank."""
        matrix, signals, constraints = self.build_constraint_matrix()
        
        if matrix.size == 0:
            return {'error': 'No constraints'}
        
        m, n = matrix.shape
        k = min(150, min(m, n))
        
        try:
            import torch
            
            device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
            mat_torch = torch.from_numpy(matrix).to(device)
            
            # Use randomized SVD for efficiency
            U, S, V = torch.svd_lowrank(mat_torch, q=k)
            singular_values = S.cpu().numpy()
            
            # Numerical rank (threshold = 1e-6)
            rank = int(np.sum(singular_values > 1e-6))
            
            return {
                'shape': (m, n),
                'rank': rank,
                'max_rank': min(m, n),
                'deficiency': min(m, n) - rank,
                'singular_values': singular_values[:15].tolist(),
                'sparsity': 100 * (1 - np.count_nonzero(matrix) / matrix.size),
                'device': device
            }
            
        except Exception as e:
            # Numpy fallback
            try:
                U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
                rank = int(np.sum(S > 1e-6))
                return {
                    'shape': (m, n),
                    'rank': rank,
                    'max_rank': min(m, n),
                    'deficiency': min(m, n) - rank,
                    'singular_values': S[:15].tolist()
                }
            except Exception as e2:
                return {'error': str(e2)}
    
    def print_summary(self) -> None:
        """Print comprehensive analysis summary."""
        print("\n" + "="*80)
        print("FEZK ELITE - Polygon zkEVM Constraint Graph Analysis")
        print("="*80)
        
        # Counts
        commit_count = sum(1 for s in self.signals.values() if s.is_commit)
        const_count = sum(1 for s in self.signals.values() if s.is_constant)
        
        print(f"\n📊 STATISTICS")
        print(f"   Signals: {len(self.signals)} ({commit_count} commit, {const_count} constant)")
        print(f"   Constraints: {len(self.constraints)}")
        print(f"   Lookups/Permutations: {len(self.lookups)}")
        print(f"   Namespaces: {len(self.namespaces)}")
        
        # Namespaces breakdown
        print(f"\n📁 NAMESPACES")
        ns_counts = defaultdict(lambda: {'commit': 0, 'const': 0})
        for sig in self.signals.values():
            if sig.is_commit:
                ns_counts[sig.namespace]['commit'] += 1
            if sig.is_constant:
                ns_counts[sig.namespace]['const'] += 1
        
        for ns in sorted(ns_counts.keys(), key=lambda x: -ns_counts[x]['commit']):
            counts = ns_counts[ns]
            print(f"   {ns}: {counts['commit']} commit, {counts['const']} constant")
        
        # ROM Lookup
        print(f"\n🔐 ROM LOOKUP ANALYSIS")
        rom_analysis = self.get_rom_lookup_analysis()
        if rom_analysis['found']:
            print(f"   Found ROM lookup at {rom_analysis['file']}:{rom_analysis['line']}")
            print(f"   Columns: {rom_analysis['lhs_columns']} LHS, {rom_analysis['rhs_columns']} RHS")
            print(f"   Type: {'Permutation' if rom_analysis['is_permutation'] else 'Lookup'}")
            print(f"   Constrained signals: {len(rom_analysis['constrained_signals'])}")
        else:
            print("   ⚠️ NO ROM LOOKUP FOUND - CRITICAL ISSUE")
        
        # Attack surfaces
        print(f"\n⚠️ ATTACK SURFACE ANALYSIS")
        attacks = self.analyze_attack_surfaces()
        
        if not attacks:
            print("   ✅ No obvious attack surfaces detected")
        else:
            by_severity = defaultdict(list)
            for attack in attacks:
                by_severity[attack['severity']].append(attack)
            
            for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
                if severity in by_severity:
                    print(f"\n   [{severity}] {len(by_severity[severity])} findings:")
                    for attack in by_severity[severity][:5]:
                        print(f"      • {attack['type']}: {attack.get('signal', attack.get('lookup_id', '?'))}")
                        print(f"        {attack['reason']}")
        
        # Rank analysis
        print(f"\n📐 CONSTRAINT MATRIX RANK ANALYSIS")
        rank_result = self.analyze_rank()
        if 'error' not in rank_result:
            print(f"   Matrix shape: {rank_result['shape']}")
            print(f"   Numerical rank: {rank_result['rank']}")
            print(f"   Max possible rank: {rank_result['max_rank']}")
            print(f"   Degrees of freedom: {rank_result['deficiency']}")
            print(f"   Sparsity: {rank_result['sparsity']:.2f}%")
            if rank_result['deficiency'] > 0:
                print(f"   ⚠️ RANK DEFICIENCY DETECTED - {rank_result['deficiency']} underconstrained DOF")
        else:
            print(f"   Error: {rank_result['error']}")
        
        print("\n" + "="*80)


def main():
    """CLI entry point."""
    import sys
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════════╗
║   FEZK ELITE - Polygon zkEVM Constraint Graph Analyzer                           ║
║   Deep analysis of PIL constraint systems                                        ║
╚══════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    if len(sys.argv) < 2:
        print("Usage: python pil_constraint_graph.py <pil_directory>")
        sys.exit(1)
    
    pil_dir = Path(sys.argv[1])
    
    if not pil_dir.exists():
        print(f"Error: Path not found: {pil_dir}")
        sys.exit(1)
    
    analyzer = PILConstraintGraph(pil_dir)
    print("Parsing PIL files...")
    analyzer.parse_all()
    
    analyzer.print_summary()


if __name__ == "__main__":
    main()
